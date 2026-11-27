"""Binoculars (Hans et al. 2024) - zero-shot detection by ratio of
observer perplexity to cross-perplexity.

Algorithm (per document):

    PPL_o   = exp( -1/N * sum_i log p_observer(x_i | x_<i) )
    X-PPL   = exp( -1/N * sum_i H(p_observer(·|x_<i), p_performer(·|x_<i)) )
    B       = log(PPL_o) / log(X-PPL)
    score   = sigmoid( (threshold - B) / scale )

Low B means the text is "too predictable for the observer relative to how
the performer would have written it" -> machine-generated. The paper's
threshold depends on the model pair; our default (0.9015) is the commonly
quoted Falcon value and is **provisional** until a local probe calibrates
it (see ``CALIBRATION`` below).

Honest constraints:
- Weights = falcon-7b + falcon-7b-instruct, ~28.9GB on disk, ~28GB in
  fp16 VRAM. This engine is exclusive and will OOM on 8GB cards.
- A 4-bit path is NOT implemented: bitsandbytes is absent from the
  runtime and its cp314 wheels are unverified. Shipping an untested
  quantization branch would be worse than refusing to load.
- Documents longer than ``MAX_TOKENS`` are truncated at the tail; the
  returned segment covers the whole text, and the truncation is surfaced
  in ``detail``-level notes only (document-level engines report one span).
- The score is a document-level signal, not a verdict (same stance as
  the other method engines).

Load layout (per manifest): ``models/binoculars/falcon-7b/`` and
``models/binoculars/falcon-7b-instruct/``.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .base import EngineBase, Segment, WeightsMissing
from .registry import EngineSpec

VERSION_ROOT = Path(__file__).resolve().parent.parent.parent

# Paper-side defaults for the Falcon pair; provisional until probed locally.
DEFAULT_THRESHOLD = 0.9015
DEFAULT_SCALE = 0.05
CALIBRATION = {
    "threshold": DEFAULT_THRESHOLD,
    "scale": DEFAULT_SCALE,
    "note": "provisional Falcon-pair threshold; re-probe before trusting absolute values",
}

MIN_TOKENS = 16  # fewer tokens carry no usable ratio
MAX_TOKENS = 1024  # falcon context is 2048; leave head-room for the pair


def binoculars_ratio(
    logp_observer: Sequence[float], cross_entropy: Sequence[float]
) -> float:
    """B = log(PPL_o) / log(X-PPL). Pure math, unit-testable without weights."""
    n = len(logp_observer)
    if n == 0 or len(cross_entropy) != n:
        raise ValueError("logp_observer and cross_entropy must be non-empty and equal length")
    mean_logp = sum(float(x) for x in logp_observer) / n
    mean_ce = sum(float(x) for x in cross_entropy) / n
    if mean_ce <= 0.0:
        raise ValueError(f"cross-entropy must be positive, got {mean_ce!r}")
    ppl = math.exp(-mean_logp)
    x_ppl = math.exp(mean_ce)
    denom = math.log(x_ppl)
    if abs(denom) < 1e-12:
        raise ValueError("degenerate cross-perplexity (log X-PPL == 0)")
    return math.log(ppl) / denom


def ratio_to_score(ratio: float, threshold: float | None = None, scale: float | None = None) -> float:
    """Map B to [0, 1] with a logistic around the threshold (low B -> high AI)."""
    t = DEFAULT_THRESHOLD if threshold is None else threshold
    s = DEFAULT_SCALE if scale is None else scale
    if s <= 0:
        raise ValueError("scale must be positive")
    z = (t - ratio) / s
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class BinocularsEngine(EngineBase):
    """Two falcon-7b models (observer + performer) over one tokenizer."""

    def __init__(self, spec: EngineSpec) -> None:
        self.engine_id = spec.engine_id
        self.lang = spec.lang
        self.kind = spec.kind
        self.est_vram_mb = spec.est_vram_mb
        self.exclusive = spec.exclusive
        self._observer: Any = None
        self._performer: Any = None
        self._tokenizer: Any = None
        self._device = "cpu"

    # ---- paths / checks ------------------------------------------------

    def _pair_dirs(self) -> tuple[Path, Path]:
        root = VERSION_ROOT / "models" / self.engine_id
        return root / "falcon-7b", root / "falcon-7b-instruct"

    @staticmethod
    def _weights_present(d: Path) -> bool:
        has_cfg = (d / "config.json").is_file()
        has_w = (d / "model.safetensors").is_file() or any(d.glob("model-*.safetensors"))
        return has_cfg and has_w

    @property
    def loaded(self) -> bool:
        return self._observer is not None and self._performer is not None

    # ---- lifecycle -----------------------------------------------------

    def load(self) -> None:
        if self.loaded:
            return
        obs_dir, perf_dir = self._pair_dirs()
        missing = [str(d.name) for d in (obs_dir, perf_dir) if not self._weights_present(d)]
        if missing:
            raise WeightsMissing(
                f"{self.engine_id}: incomplete weight pair ({', '.join(missing)}) - "
                "28.9GB download required; 4-bit path not implemented"
            )

        import torch  # lazy: module import stays torch-free
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # CPU fp16 is not supported by the falcon kernels; fp32 is correct
        # but unusably slow - we still let it run so the failure mode is a
        # slow run rather than a fake success.
        dtype = torch.float16 if device == "cuda" else torch.float32
        if self.load_dtype == "float32":
            dtype = torch.float32
        elif self.load_dtype == "float16":
            if device != "cuda":
                raise WeightsMissing(
                    f"{self.engine_id}: float16 on CPU is unsupported; use auto or float32"
                )
            dtype = torch.float16

        try:
            # The pair shares one tokenizer (falcon-7b / -instruct are the
            # same tokenizer); loading it from the observer keeps a single
            # vocabulary object alive.
            self._tokenizer = AutoTokenizer.from_pretrained(obs_dir)
            # transformers 5.23 types from_pretrained as a _Wrapped whose
            # arg 1 must be a PreTrainedModel instance - a stub bug. Route
            # through Any so mypy stays useful for the rest of the module.
            causal_lm: Any = AutoModelForCausalLM
            self._observer = causal_lm.from_pretrained(
                obs_dir,
                dtype=dtype,
                variant=self.load_variant,
                low_cpu_mem_usage=True,
            ).to(device)
            self._performer = causal_lm.from_pretrained(
                perf_dir,
                dtype=dtype,
                variant=self.load_variant,
                low_cpu_mem_usage=True,
            ).to(device)
        except Exception as e:
            self._observer = self._performer = self._tokenizer = None
            # A half-loaded pair can leave ~14GB of unreferenced blocks in
            # the CUDA cache; release them before reporting the failure.
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            raise WeightsMissing(
                f"{self.engine_id}: weight load failed ({type(e).__name__}: {e})"
            ) from e
        self._observer.eval()
        self._performer.eval()
        self._device = device

    def unload(self) -> None:
        if not self.loaded:
            return
        self._observer = self._performer = self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ---- detection -----------------------------------------------------

    def detect(
        self,
        text: str,
        on_batch: Callable[[list[Segment]], None] | None = None,
    ) -> list[Segment]:
        if not self.loaded:
            raise WeightsMissing(f"{self.engine_id}: detect called before load()")
        import torch

        enc = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
        ids = enc["input_ids"].to(self._device)
        n_tokens = int(ids.shape[1])
        if n_tokens < MIN_TOKENS:
            return []  # too short for a meaningful ratio - no fake score

        with torch.inference_mode():
            obs_logits = self._observer(ids).logits.float()
            perf_logits = self._performer(ids).logits.float()

            # positions 0..N-2 predict tokens 1..N-1
            obs = obs_logits[:, :-1, :]
            perf = perf_logits[:, :-1, :]
            targets = ids[:, 1:].unsqueeze(-1)

            logp_obs = torch.log_softmax(obs, dim=-1).gather(2, targets).squeeze(-1)
            # cross-entropy H(p_obs, p_perf) = -sum_v p_obs * log p_perf
            p_obs = torch.softmax(obs, dim=-1)
            logp_perf = torch.log_softmax(perf, dim=-1)
            cross = -(p_obs * logp_perf).sum(dim=-1).squeeze(0)

            logp_list = logp_obs.squeeze(0).tolist()
            cross_list = cross.tolist()

        ratio = binoculars_ratio(logp_list, cross_list)
        score = round(ratio_to_score(ratio), 4)
        segment = Segment(0, len(text), score)
        if on_batch is not None:
            on_batch([segment])
        return [segment]

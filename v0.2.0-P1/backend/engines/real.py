"""Real transformer-classifier engine (the first non-stub implementation).

Contract (base.py): load / unload / detect -> [offset, length, score].
Label semantics are pinned per engine via two class knobs:

- ``SCORE_MODE = "softmax"``: 2-way classifier head; AI probability is
  ``softmax(logits)[:, AI_PROB_INDEX]`` (raid-mage documents index 1 =
  machine-generated).
- ``SCORE_MODE = "sigmoid"``: single-logit head; AI probability is
  ``sigmoid(logits[:, 0])`` (desklib README: 1 = AI-generated).

``MODEL_FACTORY`` lets a repo with a custom architecture (declared in
config.json) supply its own model builder - see custom_models.py. The
default path uses ``AutoModelForSequenceClassification`` and *rejects*
any checkpoint that does not fully cover the module tree (loading_info
gate): a partially-initialised model would silently emit garbage.

Design notes:
- torch / transformers are imported lazily inside ``load()`` so that the
  package (and every test) works on machines without torch installed.
- Documents are sliced into sentences; each sentence is scored, and ALL
  scored spans become Segments whose offset/length point at the
  *original* text (str code-point offsets) - D29 removed the threshold
  filter, display-layer bucketing is the client's job.
- Weight layout: ``models/<engine_id>/`` (config.json + model.safetensors +
  tokenizer files), resolved relative to the version root, never CWD.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .base import SENT_RE, EngineBase, Segment, WeightsMissing
from .registry import EngineSpec

VERSION_ROOT = Path(__file__).resolve().parent.parent.parent


def weights_dir(engine_id: str) -> Path:
    return VERSION_ROOT / "models" / engine_id


def _check_loading_info(engine_id: str, info: dict[str, list[str]]) -> None:
    """Reject anything less than a full, exact weight match.

    from_pretrained only *warns* on missing/unexpected/mismatched keys;
    those would leave random-initialised parameters scoring garbage.
    """
    problems = []
    for kind in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        keys = info.get(kind) or []
        if keys:
            problems.append(f"{kind}={len(keys)} (e.g. {sorted(keys)[:3]})")
    if problems:
        raise WeightsMissing(
            f"{engine_id}: checkpoint does not cover the module tree - {', '.join(problems)}"
        )


class RealClassifierEngine(EngineBase):
    """Sequence-classification detector over a local HF-style weight folder."""

    AI_PROB_INDEX = 1  # softmax mode: raid-mage logits[:, 1] = machine-generated
    MAX_TOKENS = 1024  # sentence-level inputs are short; generous cap
    BATCH = 16
    SCORE_MODE = "softmax"  # or "sigmoid" for single-logit heads
    MODEL_FACTORY: Callable[[Path, Any, str | None], Any] | None = None  # custom architectures
    #: model_type values whose position table is a hard ceiling. BERT/RoBERTa
    #: build ``embeddings + position_embeddings``, so one token past the table
    #: raises instead of truncating; DeBERTa-v2 (relative bias) and ModernBERT
    #: (rotary) have no such wall and are deliberately absent.
    POSITION_CAPPED_TYPES = frozenset({"bert", "roberta", "xlm-roberta", "gpt2", "gpt_neo"})

    def __init__(self, spec: EngineSpec, weights: Path | None = None) -> None:
        self.engine_id = spec.engine_id
        self.lang = spec.lang
        self.kind = spec.kind
        self.est_vram_mb = spec.est_vram_mb
        self.exclusive = spec.exclusive
        self._weights = weights if weights is not None else weights_dir(spec.engine_id)
        self._model: Any = None
        self._tokenizer: Any = None
        self._device = "cpu"

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _input_limit(self) -> int:
        """Longest input the loaded model actually accepts.

        MAX_TOKENS is a per-engine preference; for absolute-position families
        it must not exceed the model's table or the forward pass raises
        (round-22: xiaoyu is a 512-position BERT behind MAX_TOKENS=1024, so a
        single unpunctuated 600-char "sentence" crashed the whole detect).
        """
        cfg = getattr(self._model, "config", None)
        mtype = getattr(cfg, "model_type", "") or ""
        pos = getattr(cfg, "max_position_embeddings", None)
        if mtype in self.POSITION_CAPPED_TYPES and isinstance(pos, int) and 0 < pos < self.MAX_TOKENS:
            return pos
        return self.MAX_TOKENS

    def _resolve_dtype(self, torch: Any, device: str) -> Any:
        """D24⑤: explicit knob wins; "auto" keeps the platform default."""
        if self.load_dtype == "float16":
            return torch.float16
        if self.load_dtype == "float32":
            return torch.float32
        return torch.float16 if device == "cuda" else torch.float32

    def load(self) -> None:
        if self._model is not None:
            return
        if not (self._weights / "config.json").is_file():
            raise WeightsMissing(f"{self.engine_id}: no config.json in {self._weights.name}/")
        if self.load_variant == "fp16":
            has_st = (self._weights / "model.fp16.safetensors").is_file()
            has_bin = False
        else:
            has_st = (self._weights / "model.safetensors").is_file()
            has_bin = (self._weights / "pytorch_model.bin").is_file()
        if not has_st and not has_bin:
            raise WeightsMissing(f"{self.engine_id}: weights not downloaded (safetensors/bin)")

        import torch  # lazy: keep module import torch-free
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = self._resolve_dtype(torch, device)
        variant = self.load_variant
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self._weights)
            if self.MODEL_FACTORY is not None:
                model = self.MODEL_FACTORY(self._weights, dtype, variant)
            else:
                model, info = AutoModelForSequenceClassification.from_pretrained(
                    self._weights,
                    dtype=dtype,
                    variant=variant,
                    output_loading_info=True,
                )
                _check_loading_info(self.engine_id, info)
            model = model.to(device)
        except WeightsMissing:
            self._model = self._tokenizer = None
            raise
        except Exception as e:
            self._model = self._tokenizer = None
            raise WeightsMissing(
                f"{self.engine_id}: weight load failed ({type(e).__name__}: {e})"
            ) from e
        model.eval()
        self._model = model
        self._device = device

    def unload(self) -> None:
        if self._model is None:
            return
        self._model = self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def detect(
        self,
        text: str,
        on_batch: Callable[[list[Segment]], None] | None = None,
    ) -> list[Segment]:
        if self._model is None or self._tokenizer is None:
            raise WeightsMissing(f"{self.engine_id}: detect called before load()")
        import torch

        sents = [(m.start(), m.group()) for m in SENT_RE.finditer(text) if m.group().strip()]
        if not sents:
            return []

        limit = self._input_limit()
        segments: list[Segment] = []
        for i in range(0, len(sents), self.BATCH):
            chunk = sents[i : i + self.BATCH]
            enc = self._tokenizer(
                [s[1] for s in chunk],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=limit,
            ).to(self._device)
            with torch.inference_mode():
                logits = self._model(**enc).logits
            if self.SCORE_MODE == "sigmoid":
                probs = torch.sigmoid(logits.float().squeeze(-1))
            else:
                probs = torch.softmax(logits.float(), dim=-1)[:, self.AI_PROB_INDEX]
            # D29: every sentence comes back (no threshold filter) - the
            # client buckets scores for display; low scores are signal too.
            batch = [Segment(offset, len(sent), round(p, 4)) for (offset, sent), p in zip(chunk, probs.tolist())]
            segments.extend(batch)
            if on_batch is not None:
                on_batch(batch)
        return segments

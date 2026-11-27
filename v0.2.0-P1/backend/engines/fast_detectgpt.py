"""Fast-DetectGPT zero-shot method engine (Bao et al. 2024, adapted).

Method: conditional-probability curvature. For a sentence x, condition
on a short prefix, sample N continuations from the model, and compare
the log-probability of the *real* continuation against the samples:

    curvature(x) = (E[log p(X~)] - log p(x_cont)) / std[log p(X~)]

AI text lives deep inside the model's high-probability band: its real
tokens score above the model's own samples, giving positive curvature.

Span alignment (fixed 2026-11-20, round-6e audit): BOTH statistics are
means over the same continuation span - the samples gather generated
token positions only, and log p(x_cont) gathers only tokens after the
prefix. The previous version gathered the prompt prefix into every
sample mean (a constant across samples, >50% of positions for short
sentences), deflating the variance and contaminating the mean - a
plausible contributor to the weak/contradictory gpt2-family probes.

Determinism: every generation is seeded from a stable hash of the
sentence + sample index, so repeated detects on the same text produce
the same score (no cross-run drift in verification artifacts). Note
torch.manual_seed is process-global - transformers.generate() accepts
no generator, so this is the pragmatic price of reproducibility.

Status (round-6e, 2026-11-20): wired in default_engine - downloading
the production base makes the engine runnable, no more boot stub.
D19 history stands: gpt2-family probes were negative/contradictory
(small inverted, medium +0.9..+1.9 vs -3.3/-3.4 never reconciled);
the production base is EleutherAI/gpt-neo-2.7B at fp16 ORIGINAL
precision, exclusive (~5.4GB residency; official repo publishes fp32
ONLY - 10.67GB download, cast at load time; verified 2026-11-17).
Community fp16/int4 re-packages are barred by D18 (no SHA anchor,
untrusted re-packaging, int4 destroys logp fidelity). Weights land at
models/fast-detectgpt/ per models/manifest.json. CENTER/SCALE are
PROVISIONAL (probe_calibrate never ran) - re-probe MUST pass on
gpt-neo before any score from this engine is trusted.

Cost model: N_SAMPLES generate() calls + 1 recompute forward per
sentence, strictly sequential. On gpt2-small that was sub-second; on
gpt-neo-2.7B it is the real tail (engine is exclusive by design for
exactly this reason).
"""
from __future__ import annotations

import zlib
from collections.abc import Callable

from .base import SENT_RE, Segment, WeightsMissing
from .gpt2base import Gpt2BaseEngine


class FastDetectGptEngine(Gpt2BaseEngine):

    WEIGHTS_DIRNAME = "fast-detectgpt"

    THRESHOLD = 0.5  # final score = sigmoid((curvature - CENTER) / SCALE); calibrated by probe
    CENTER = 1.0  # provisional; probe_calibrate pins real values
    SCALE = 1.0
    N_SAMPLES = 25
    PREFIX_TOKENS = 16
    MIN_TOKENS = 20  # curvature statistics need enough tokens to be stable
    TOP_P = 0.9

    def _sample_logp(self, prompt_ids, new_tokens: int, seed: int) -> float:
        """Generate one continuation; return the mean log-prob of the
        GENERATED tokens only (prompt positions are excluded - the fix
        documented in the module docstring)."""
        import torch

        torch.manual_seed(seed)
        prompt_len = prompt_ids.shape[1]
        gen = self._model.generate(
            prompt_ids,
            do_sample=True,
            top_p=self.TOP_P,
            max_new_tokens=new_tokens,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        n_generated = gen.shape[1] - prompt_len
        if n_generated <= 0:  # degenerate: model emitted eos immediately
            return 0.0
        with torch.inference_mode():
            logits = self._model(gen).logits.float()
        # predictors for generated tokens sit at logits positions
        # [prompt_len-1 .. end-2] -> targets gen[:, prompt_len:]
        pred = logits[:, prompt_len - 1 : -1]
        actual = gen[:, prompt_len:]
        logp = torch.log_softmax(pred, dim=-1)
        return -logp.gather(-1, actual.unsqueeze(-1)).mean().item()

    def detect(
        self,
        text: str,
        on_batch: Callable[[list[Segment]], None] | None = None,
    ) -> list[Segment]:
        if self._model is None or self._tokenizer is None:
            raise WeightsMissing(f"{self.engine_id}: detect called before load()")
        import torch

        segments: list[Segment] = []
        for m in SENT_RE.finditer(text):
            sent = m.group()
            if not sent.strip():
                continue
            enc = self._tokenizer(
                sent, return_tensors="pt", truncation=True, max_length=self.MAX_TOKENS
            ).to(self._device)
            ids = enc["input_ids"]
            n = ids.shape[1]
            if n < self.MIN_TOKENS:
                continue

            prefix_len = min(self.PREFIX_TOKENS, n - 1)
            new_tokens = max(n - prefix_len, 8)

            with torch.inference_mode():
                logits = self._model(**enc).logits.float()
            # real continuation span only: predictions for tokens
            # [prefix_len..n-1] sit at logits[:, :-1] positions
            # [prefix_len-1 .. n-2]
            logp_all = torch.log_softmax(logits[:, :-1], dim=-1)
            logp_orig = (
                -logp_all[:, prefix_len - 1 :]
                .gather(-1, ids[:, prefix_len:].unsqueeze(-1))
                .mean()
                .item()
            )

            prefix = ids[:, :prefix_len]
            base_seed = zlib.crc32(sent.encode("utf-8"))
            samples = [
                self._sample_logp(prefix, new_tokens, base_seed * 100003 + i)
                for i in range(self.N_SAMPLES)
            ]
            mean_s = sum(samples) / len(samples)
            var_s = sum((s - mean_s) ** 2 for s in samples) / (len(samples) - 1)
            std_s = max(var_s**0.5, 1e-6)

            curvature = (mean_s - logp_orig) / std_s
            score = torch.sigmoid(
                torch.tensor((curvature - self.CENTER) / self.SCALE)
            ).item()
            if score >= self.THRESHOLD:
                segments.append(Segment(m.start(), len(sent), round(score, 4)))
        return segments

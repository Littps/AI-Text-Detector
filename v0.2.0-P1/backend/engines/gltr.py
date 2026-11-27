"""GLTR zero-shot method engine (Gehrmann et al. 2019, adapted).

Method: score each sentence by how well GPT-2 predicts it *given the
preceding text* - AI text is sampled FROM this distribution, so its
tokens sit in the top-k band far more often than human text.

Continuation scoring (one forward pass over the whole text):
- tokenize the full document (<= 1024 tokens) with offset mappings
- a single forward pass yields, for every token t >= 1, the NLL and the
  top-10 hit of the *actual* token given everything before it
- each sentence is scored over its own token span only, but every token
  conditions on the full prior context - sentence-initial tokens keep
  their context, which is exactly what naive per-sentence scoring loses

Naive per-sentence scoring is measured and rejected: sentence-initial
tokens have no context, so AI prose ppl jumps from ~44 (document) to
73-176 (single sentences) and AI/human scores collapse to 0.05-0.46 -
nothing crosses the threshold. Continuation mode restores separation:
AI sentences 0.68-0.96, work/casual human 0.08-0.25.

VRAM note: the transient tensors of one forward pass over T=1024 tokens
are chunked (CHUNK=256 rows of the vocab axis) so the fp32 log-softmax
peak stays ~52 MB instead of ~410 MB - otherwise the transient would be
double the engine's est_vram_mb and invisible to the LRU budget.

Truncation semantics: documents beyond MAX_TOKENS=1024 GPT-2 tokens are
silently truncated - sentences in the dropped tail get no score. GLTR
is a *signal* engine, so this is accepted, but the UI should surface it
(registry notes carries the same warning).

Fusion (calibrated on the 2026-11-17 probe set, scripts/probe_gltr.py):

    score = 0.65 * norm(top10_rate) + 0.35 * norm(ppl)
    norm(top10): 0.30 -> 0, 0.75 -> 1
    norm(ppl):   20 -> 1 (AI-like), 150 -> 0

Continuation-mode behaviour (fp16, GPT-2 small):
    AI sentences    score 0.68-0.96  (one short "Then, ..." slice 0.44)
    HU work/casual  score 0.08-0.25  (one casual slice 0.56, marginal)
    HU wiki-formal  score 1.00       <-- KNOWN FALSE POSITIVE

Formal human prose (encyclopedic/news) overlaps the AI band under this
zero-shot method - that is the documented limit of GLTR itself, not a
bug. The product should surface GLTR as a *signal*, not a verdict.
"""
from __future__ import annotations

import math
from collections.abc import Callable

from .base import SENT_RE, Segment, WeightsMissing
from .gpt2base import Gpt2BaseEngine


def _norm(value: float, lo: float, span: float) -> float:
    return max(0.0, min(1.0, (value - lo) / span))


class GltrEngine(Gpt2BaseEngine):
    """GLTR / perplexity signal over the shared GPT-2 base (EN only)."""

    MIN_TOKENS = 8  # shorter slices carry too little signal to rank
    CHUNK = 256  # vocab-axis rows per fp32 log-softmax block (~52 MB)
    TOP10_LO = 0.30
    TOP10_SPAN = 0.45
    PPL_LO = 20.0
    PPL_SPAN = 130.0
    W_TOP10 = 0.65
    W_PPL = 0.35
    MAX_TOKENS = 1024  # GPT-2 context; longer documents are truncated

    def detect(
        self,
        text: str,
        on_batch: Callable[[list[Segment]], None] | None = None,
    ) -> list[Segment]:
        if self._model is None or self._tokenizer is None:
            raise WeightsMissing(f"{self.engine_id}: detect called before load()")

        enc = self._tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.MAX_TOKENS,
        ).to(self._device)
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"][0].tolist()
        if ids.shape[1] < 2:
            return []

        import torch  # lazy: matches the module-level import policy

        # position i predicts token i+1. top10 in fp16 (order differences
        # vs fp32 only at <2^-11 ties, far below real logit gaps);
        # log-softmax must stay fp32 - summing 50257 fp16 log-probs
        # (~-5.4e5) overflows half precision.
        with torch.inference_mode():
            logits = self._model(input_ids=ids, attention_mask=enc["attention_mask"]).logits
            pred = logits[:, :-1]  # (1, T-1, V) fp16 view, no copy
            actual = ids[0, 1:]  # (T-1,)
            top10 = pred.topk(10, dim=-1).indices[0]  # (T-1, 10) int64
            hit = (top10 == actual.unsqueeze(-1)).any(dim=-1)  # (T-1,) bool
            del top10

            n_rows = pred.shape[1]
            nll = torch.empty(n_rows, device=pred.device, dtype=torch.float32)
            for b in range(0, n_rows, self.CHUNK):
                blk = pred[:, b : b + self.CHUNK].float()
                logp_blk = torch.log_softmax(blk, dim=-1)
                idx = actual[b : b + self.CHUNK].view(1, -1, 1)  # (1, K, 1)
                nll[b : b + self.CHUNK] = -logp_blk.gather(-1, idx).squeeze(-1).squeeze(0)
                del blk, logp_blk
            del logits, pred

        segments: list[Segment] = []
        for m in SENT_RE.finditer(text):
            sent = m.group()
            if not sent.strip():
                continue
            # token overlaps the sentence char span (BPE offsets start at
            # the leading space, so containment would drop sentence heads;
            # tokens spanning a sentence boundary are rare in EN prose and
            # cost at most one token of signal in each neighbour - asserted
            # segment-disjointness in scripts/probe_gltr.py)
            tok_idx = [
                t
                for t in range(1, len(offsets))
                if offsets[t][1] > offsets[t][0] and offsets[t][1] > m.start() and offsets[t][0] < m.end()
            ]
            if len(tok_idx) < self.MIN_TOKENS:
                continue  # too little signal to rank - no score, client renders uncolored
            rows = [t - 1 for t in tok_idx]  # feature position of token t
            ppl = math.exp(nll[rows].mean().item())
            h10 = hit[rows].float().mean().item()
            score = self.W_TOP10 * _norm(h10, self.TOP10_LO, self.TOP10_SPAN) + self.W_PPL * (
                1.0 - _norm(ppl, self.PPL_LO, self.PPL_SPAN)
            )
            segments.append(Segment(m.start(), len(sent), round(score, 4)))
        if on_batch is not None and segments:
            on_batch(segments)  # single forward pass: one shot at the end
        return segments

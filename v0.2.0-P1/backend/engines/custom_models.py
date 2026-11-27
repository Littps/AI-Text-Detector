"""Local replicas of upstream custom architectures.

Some repos declare a custom class in ``config.json`` (``architectures``
field) that ``AutoModelForSequenceClassification`` cannot build. Instead
of ``trust_remote_code=True`` (arbitrary remote execution - rejected) we
transcribe the upstream reference code locally, line by line, and load
the checkpoint into our own replica. The factory must fail loudly if the
state_dict does not match exactly - a partially initialised model would
silently emit garbage scores.

Each factory below pins:
- the module tree (so state_dict keys line up without remapping)
- the label semantics, quoted from the upstream inference snippet

Lazy imports: factories are called inside ``load()``, never at module
import time, so the package stays torch-free for tests.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .base import Segment, WeightsMissing
from .real import RealClassifierEngine, _check_loading_info

# engine_id -> factory(weights_dir, dtype, variant) -> PreTrainedModel
CUSTOM_FACTORIES: dict[str, Callable[[Path, Any, str | None], Any]] = {}


def desklib_factory(weights: Path, dtype: Any, variant: str | None = None) -> Any:
    """desklib/ai-text-detector-v1.01 - ``DesklibAIDetectionModel``.

    Reference: https://github.com/desklib/ai-text-detector (README, MIT).
    Transcribed verbatim from the upstream snippet:

    - backbone: ``AutoModel`` over microsoft/deberta-v3-large (self.model)
    - head: mean pooling over the attention mask -> Linear(hidden, 1)
    - label semantics (upstream ``predict_single_text``):
      ``sigmoid(logit) >= 0.5`` -> 1 = AI-generated, else human.

    State_dict layout (probed 2026-11-17): 392 keys - ``model.*`` (390,
    DebertaV2Model) + ``classifier.{weight (1,1024), bias (1,)}``.
    """
    import torch
    from torch import nn
    from transformers import AutoConfig, AutoModel, PreTrainedModel
    from transformers.modeling_outputs import SequenceClassifierOutput

    class DesklibAIDetectionModel(PreTrainedModel):
        config_class = AutoConfig

        def __init__(self, config: Any) -> None:
            super().__init__(config)
            self.model = AutoModel.from_config(config)
            self.classifier = nn.Linear(config.hidden_size, 1)
            # transformers 5.x protocol: post_init registers instance
            # attributes (all_tied_weights_keys, _no_split_modules, ...)
            # that from_pretrained requires. (4.x README used init_weights().)
            self.post_init()

        def forward(
            self,
            input_ids: Any = None,
            attention_mask: Any = None,
            **kwargs: Any,
        ) -> SequenceClassifierOutput:
            outputs = self.model(input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs[0]
            # Upstream mean pooling: mask-weighted sum / clamped mask sum.
            # Upstream casts the mask with .float() for fp32 checkpoints;
            # we follow the hidden state dtype instead so fp16 weights work.
            # clamp(min=1) matches upstream min=1e-9 for any non-empty mask
            # (the mask sums to the exact token count >= 1).
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            mask = mask.to(last_hidden_state.dtype)
            sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1.0)
            pooled_output = sum_embeddings / sum_mask
            logits = self.classifier(pooled_output)
            return SequenceClassifierOutput(logits=logits)

    config = AutoConfig.from_pretrained(weights)
    model, info = DesklibAIDetectionModel.from_pretrained(
        weights, config=config, dtype=dtype, variant=variant, output_loading_info=True
    )
    _check_loading_info("desklib-en", info)
    return model


class DesklibEngine(RealClassifierEngine):
    """desklib-en: DeBERTa-v3-large + mean pooling + single-logit head.

    SCORE_MODE=sigmoid because the head emits one logit; upstream README
    pins sigmoid(logit) >= 0.5 as AI-generated (1), else human (0).

    DOCUMENT-LEVEL on purpose: the upstream mean pooling aggregates a
    whole document and RAID training inputs are 100-500-word passages.
    Probing sentence slices (2026-11-17) shows the upstream human sample
    scores 0.42 as a full text but one of its sentences alone scores
    0.79 - short isolated sentences are out-of-distribution, so sentence
    slicing would manufacture false positives. detect() therefore scores
    the text as one document and returns a single full-span segment.

    Inference protocol follows upstream predict_single_text verbatim:
    padding='max_length', max_length=768, fp16-proven stable (fp32 vs
    fp16 logits differ < 0.01).
    """

    SCORE_MODE = "sigmoid"
    MODEL_FACTORY = staticmethod(desklib_factory)  # type: ignore[assignment]
    MAX_TOKENS = 768  # upstream max_len

    def detect(
        self,
        text: str,
        on_batch: Callable[[list[Segment]], None] | None = None,
    ) -> list[Segment]:
        if self._model is None or self._tokenizer is None:
            raise WeightsMissing(f"{self.engine_id}: detect called before load()")
        if not text.strip():
            return []
        import torch

        enc = self._tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self._input_limit(),
            return_tensors="pt",
        ).to(self._device)
        with torch.inference_mode():
            logit = self._model(**enc).logits.item()
        p = 1.0 / (1.0 + math.exp(-logit))
        # D29: the single document-level segment always comes back (the
        # doc_score formula degenerates to p on it - zero special-casing).
        segs = [Segment(0, len(text), round(p, 4))]
        if on_batch is not None:
            on_batch(segs)
        return segs


CUSTOM_FACTORIES["desklib-en"] = desklib_factory


class VanguardEngine(RealClassifierEngine):
    """vanguard-en: PAN-2026 ModernBERT, single-logit head.

    Standard ModernBertForSequenceClassification architecture (loads via
    the default Auto path - no custom factory), but the checkpoint ships
    num_labels=1: one logit, sigmoid semantics. Probed 2026-11-17:
    work-style human text ~0.000, AI text ~1.0 -> 1 = AI-generated.
    Sentence-level slicing works (human slices 0.000-0.005, AI slices
    0.25-1.0); short casual chat text is OOD and over-flags, like desklib.
    """

    SCORE_MODE = "sigmoid"


class XiaoyuEngine(RealClassifierEngine):
    """xiaoyu-zh: Chinese BERT, single-logit head.

    Standard BertForSequenceClassification architecture, num_labels=1.
    Probed 2026-11-17: AI zh +9.6 (sigmoid ~1.0), casual/food human zh
    -2.7/-5.3 (0.06/0.005) -> 1 = AI-generated; casual Chinese is
    IN distribution (unlike the RAID-trained EN models).
    """

    SCORE_MODE = "sigmoid"


class RouterEngine(RealClassifierEngine):
    """r1-lang-router: 20-language XLM-R (infrastructure, kind="router").

    Language routing never produces AI segments - detect() returns []
    so batches that include the router stay contract-clean. Call
    language() for the verdict (zh/en/... with top-1 confidence).
    Probed 2026-11-17: zh 0.99 / en 0.98 on casual + formal samples.
    """

    def detect(self, text: str, on_batch=None) -> list[Segment]:
        if self._model is None or self._tokenizer is None:
            raise WeightsMissing(f"{self.engine_id}: detect called before load()")
        return []  # routers do not flag AI text

    def language(self, text: str) -> tuple[str, float]:
        """Top-1 language label + confidence for one text chunk."""
        import torch

        enc = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self._input_limit()
        ).to(self._device)
        with torch.inference_mode():
            probs = torch.softmax(self._model(**enc).logits.float(), dim=-1)[0]
        idx = int(probs.argmax().item())
        return self._model.config.id2label[idx], round(probs[idx].item(), 4)


# engine_id -> engine class override; loader.default_engine consults this
# before falling back to the generic RealClassifierEngine.
CUSTOM_ENGINE_CLASSES: dict[str, type[RealClassifierEngine]] = {
    "desklib-en": DesklibEngine,
    "vanguard-en": VanguardEngine,
    "xiaoyu-zh": XiaoyuEngine,
    "r1-lang-router": RouterEngine,
}

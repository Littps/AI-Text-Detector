"""Engine contract: every detector implements load / unload / detect.

Design notes (PLAN P1):
- Results carry [offset, length, score] only - the raw text never leaves
  the backend.
- offsets are Python str indices (code points), not byte offsets.
- ``est_vram_mb`` feeds the LRU budget; ``exclusive`` engines (Binoculars
  class) demand an otherwise-empty GPU and may exceed the budget by policy.

P3-a (D29): segments are ALL scored spans (no threshold filter - the
caller buckets for display); ``detect`` accepts an optional ``on_batch``
callback so sentence-batch engines can stream partial results; the
loader derives ``doc_score`` (length-weighted mean) from the segments.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass


class DetectCancelled(Exception):
    """Cooperative stop signal raised from an ``on_batch`` callback.

    Round-19: the caller (server job registry) raises this between sentence
    batches when the window that asked for the run is gone. Engines let it
    propagate - a torch forward pass is not interruptible, but the next batch
    never starts. The loader maps it to status "cancelled" so it cannot be
    mistaken for a scoring error.
    """


# Sentence spans: starts at a non-space char, ends before/at sentence
# punctuation. Pure-space or pure-punctuation runs never match. Shared by
# every sentence-slicing engine (classifier + method kinds).
SENT_RE = re.compile(r"[^\s.!?。！？\n][^.!?。！？\n]*(?:[.!?。！？]+|$)")


@dataclass(frozen=True, slots=True)
class Segment:
    """One flagged span in the analysed text."""

    offset: int
    length: int
    score: float

    def as_dict(self) -> dict:
        return {"offset": self.offset, "length": self.length, "score": self.score}


@dataclass(frozen=True, slots=True)
class EngineResult:
    """Per-engine outcome inside a batch detect call."""

    engine_id: str
    status: str  # "ok" | "weights_missing" | "error" | "unknown_engine"
    segments: tuple[Segment, ...] = ()
    detail: str = ""
    doc_score: float | None = None  # length-weighted AI rate over the doc

    def as_dict(self) -> dict:
        return {
            "engine": self.engine_id,
            "status": self.status,
            "segments": [s.as_dict() for s in self.segments],
            "doc_score": self.doc_score,
            **({"detail": self.detail} if self.detail else {}),
        }


class WeightsMissing(RuntimeError):
    """Raised when an engine is registered but its weights are not on disk."""


class EngineBase(ABC):
    """Lifecycle contract implemented by every engine."""

    engine_id: str = ""
    lang: str = "multi"  # "zh" | "en" | "multi"
    kind: str = "classifier"  # "router" | "classifier" | "method"
    est_vram_mb: int = 0
    exclusive: bool = False
    # D24⑤/D25 load-time knobs, set by EngineLoader from the storage
    # state before acquire(): "auto" keeps the platform default (CUDA
    # fp16 / CPU fp32); "fp16" variant loads model.fp16.safetensors.
    load_dtype: str = "auto"  # "auto" | "float16" | "float32"
    load_variant: str | None = None  # None | "fp16"
    # Real footprint measured around load() (lru.py: torch.cuda.memory_allocated
    # delta). None means "not measured" - the declared est_vram_mb is used then.
    actual_vram_mb: int | None = None

    @abstractmethod
    def load(self) -> None:
        """Materialize weights onto the device. Idempotent."""

    @abstractmethod
    def unload(self) -> None:
        """Free weights and device cache. Idempotent."""

    @abstractmethod
    def detect(
        self,
        text: str,
        on_batch: Callable[[list[Segment]], None] | None = None,
    ) -> list[Segment]:
        """Score one document. Only called while the engine is loaded.

        Every scored span is returned (D29: no threshold filtering here -
        thresholding is a display-layer concern). Batch-capable engines
        call ``on_batch`` with each completed batch for streaming; single
        pass engines may ignore it.
        """

    @property
    @abstractmethod
    def loaded(self) -> bool:
        """True while weights are resident."""

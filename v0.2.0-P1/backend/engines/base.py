"""Engine contract: every detector implements load / unload / detect.

Design notes (PLAN P1):
- Results carry [offset, length, score] only - the raw text never leaves
  the backend.
- offsets are Python str indices (code points), not byte offsets.
- ``est_vram_mb`` feeds the LRU budget; ``exclusive`` engines (Binoculars
  class) demand an otherwise-empty GPU and may exceed the budget by policy.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


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

    def as_dict(self) -> dict:
        return {
            "engine": self.engine_id,
            "status": self.status,
            "segments": [s.as_dict() for s in self.segments],
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

    @abstractmethod
    def load(self) -> None:
        """Materialize weights onto the device. Idempotent."""

    @abstractmethod
    def unload(self) -> None:
        """Free weights and device cache. Idempotent."""

    @abstractmethod
    def detect(self, text: str) -> list[Segment]:
        """Score one document. Only called while the engine is loaded."""

    @property
    @abstractmethod
    def loaded(self) -> bool:
        """True while weights are resident."""

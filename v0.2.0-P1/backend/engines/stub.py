"""Placeholder engine: registered but not runnable yet.

Two honest failure modes, distinguished at raise time:
- weights absent -> "not downloaded yet" (the normal pre-download state
  every catalogue entry passes through);
- weights present but no implementation class -> "implementation not
  landed" (binoculars today: downloading its 28.9GB must NOT pretend the
  engine became usable).
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .base import EngineBase, Segment, WeightsMissing

_WEIGHTS_ROOT = Path(__file__).resolve().parent.parent.parent / "models"


def _missing_detail(engine_id: str) -> str:
    d = _WEIGHTS_ROOT / engine_id
    landed = d.is_dir() and (
        any(d.rglob("config.json"))
        or any(d.rglob("*.safetensors"))
        or any(d.rglob("*.bin"))
    )
    if landed:
        return (
            f"{engine_id}: engine implementation not landed (stub) - "
            "downloading weights alone does not enable this engine"
        )
    return f"{engine_id}: weights not downloaded yet (P2 downloader)"


class StubEngine(EngineBase):
    def __init__(self, engine_id: str, lang: str, kind: str, est_vram_mb: int, exclusive: bool = False) -> None:
        self.engine_id = engine_id
        self.lang = lang
        self.kind = kind
        self.est_vram_mb = est_vram_mb
        self.exclusive = exclusive
        self._loaded = False

    def load(self) -> None:
        raise WeightsMissing(_missing_detail(self.engine_id))

    def unload(self) -> None:
        self._loaded = False

    def detect(
        self,
        text: str,
        on_batch: Callable[[list[Segment]], None] | None = None,
    ) -> list[Segment]:
        raise WeightsMissing(_missing_detail(self.engine_id))

    @property
    def loaded(self) -> bool:
        return self._loaded

"""Placeholder engine: registered but weights not downloaded (P2 downloader).

Every catalogue entry starts as a StubEngine so the loader, the LRU
policy and the protocol can be exercised end-to-end before any real
weights exist. Real implementations replace stubs one by one.
"""
from __future__ import annotations

from .base import EngineBase, Segment, WeightsMissing


class StubEngine(EngineBase):
    def __init__(self, engine_id: str, lang: str, kind: str, est_vram_mb: int, exclusive: bool = False) -> None:
        self.engine_id = engine_id
        self.lang = lang
        self.kind = kind
        self.est_vram_mb = est_vram_mb
        self.exclusive = exclusive
        self._loaded = False

    def load(self) -> None:
        raise WeightsMissing(
            f"{self.engine_id}: weights not downloaded yet (P2 downloader)"
        )

    def unload(self) -> None:
        self._loaded = False

    def detect(self, text: str) -> list[Segment]:
        raise WeightsMissing(f"{self.engine_id}: not implemented (stub)")

    @property
    def loaded(self) -> bool:
        return self._loaded

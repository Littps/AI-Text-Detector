"""Test doubles for the engine layer."""
from collections.abc import Callable

from engines.base import EngineBase, Segment


class FakeEngine(EngineBase):
    """Controllable engine: records load/unload, returns fixed segments.

    ``batches`` enables streaming tests: detect() fires on_batch once per
    batch in order and returns their concatenation.
    """

    def __init__(
        self,
        engine_id: str,
        *,
        lang: str = "en",
        kind: str = "classifier",
        est_vram_mb: int = 100,
        exclusive: bool = False,
        segments: list[Segment] | None = None,
        batches: list[list[Segment]] | None = None,
    ) -> None:
        self.engine_id = engine_id
        self.lang = lang
        self.kind = kind
        self.est_vram_mb = est_vram_mb
        self.exclusive = exclusive
        self._segments = segments or []
        self._batches = batches
        self._loaded = False
        self.load_calls = 0
        self.unload_calls = 0

    def load(self) -> None:
        self._loaded = True
        self.load_calls += 1

    def unload(self) -> None:
        self._loaded = False
        self.unload_calls += 1

    def detect(
        self,
        text: str,
        on_batch: Callable[[list[Segment]], None] | None = None,
    ) -> list[Segment]:
        if not self._loaded:
            raise RuntimeError(f"{self.engine_id}: detect before load")
        if self._batches is not None:
            out: list[Segment] = []
            for b in self._batches:
                out.extend(b)
                if on_batch is not None:
                    on_batch(list(b))
            return out
        if on_batch is not None and self._segments:
            on_batch(list(self._segments))
        return list(self._segments)

    @property
    def loaded(self) -> bool:
        return self._loaded

"""GPU residency manager: lazy loading + LRU eviction under a VRAM budget.

Policy (PLAN P1, D14/D15):
- Normal engines share a 6GB budget; the least-recently-used engine is
  evicted first when a new one does not fit.
- Exclusive engines (Binoculars class) evict everything and may exceed
  the budget - they are allowed to run only against an otherwise-empty
  GPU, and the UI must warn about 8GB cards.
- A non-exclusive engine whose estimate exceeds the whole budget is a
  configuration error and is refused outright.
"""
from __future__ import annotations

from collections import OrderedDict

from .base import EngineBase


class VramPolicyError(RuntimeError):
    """Engine estimate cannot ever fit the configured budget."""


class LruVramManager:
    def __init__(self, budget_mb: int = 6144) -> None:
        if budget_mb <= 0:
            raise ValueError("budget_mb must be positive")
        self._budget = budget_mb
        self._resident: OrderedDict[str, EngineBase] = OrderedDict()

    @property
    def budget_mb(self) -> int:
        return self._budget

    def resident_ids(self) -> list[str]:
        """Resident engine ids, oldest first (eviction order)."""
        return list(self._resident)

    def used_mb(self) -> int:
        return sum(e.est_vram_mb for e in self._resident.values())

    def is_resident(self, engine_id: str) -> bool:
        return engine_id in self._resident

    def acquire(self, engine: EngineBase) -> None:
        """Ensure ``engine`` is loaded, evicting others as needed."""
        if engine.engine_id in self._resident:
            self._resident.move_to_end(engine.engine_id)
            return

        if engine.exclusive:
            self._evict_all(except_id=engine.engine_id)
        else:
            if engine.est_vram_mb > self._budget:
                raise VramPolicyError(
                    f"{engine.engine_id} needs {engine.est_vram_mb}MB "
                    f"> budget {self._budget}MB and is not exclusive"
                )
            while self.used_mb() + engine.est_vram_mb > self._budget:
                self._evict_oldest(keep=engine.engine_id)

        engine.load()
        self._resident[engine.engine_id] = engine
        self._resident.move_to_end(engine.engine_id)

    def release(self, engine_id: str) -> bool:
        engine = self._resident.pop(engine_id, None)
        if engine is None:
            return False
        engine.unload()
        return True

    def evict_all(self) -> None:
        self._evict_all()

    def _evict_oldest(self, keep: str | None = None) -> None:
        for eid in list(self._resident):
            if eid != keep:
                self.release(eid)
                return

    def _evict_all(self, except_id: str | None = None) -> None:
        for eid in list(self._resident):
            if eid != except_id:
                self.release(eid)

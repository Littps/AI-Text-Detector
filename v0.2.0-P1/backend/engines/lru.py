"""GPU residency manager: lazy loading + LRU eviction under a VRAM budget.

Policy (PLAN P1, D14/D15; 2026-11-21 retention rework):
- Normal engines share a budget; the least-recently-used engine is evicted
  first when a new one does not fit. Eviction prefers engines whose language
  does not appear in the current run (user rule: a zh engine should make room
  for a five-engine English run instead of evicting a sibling en engine).
- Exclusive engines (Binoculars class) evict everything and may exceed the
  budget - they are allowed to run only against an otherwise-empty GPU, and
  the UI must warn about 8GB cards.
- A non-exclusive engine whose estimate exceeds the whole budget is a
  configuration error and is refused outright.

Retention (how long a model stays in VRAM after a detection finishes):
- ``forever``  - keep everything; budget takes the whole card minus 512MB
- ``default``  - keep everything; budget is 80% of the card (desktop headroom)
- ``30m/10m/5m`` - idle engines are released once unused for that long
- ``none``     - the caller unloads everything right after a detect batch

VRAM reporting is measured, not declared: ``used_mb`` sums the real
``torch.cuda.memory_allocated`` delta captured around each ``load()``, and
falls back to ``est_vram_mb`` only when no GPU/torch measurement exists.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

from .base import EngineBase

# Retention tiers: None = never unload on idle, 0 = unload right after use.
RETENTION_TIERS: dict[str, float | None] = {
    "forever": None,
    "default": None,
    "30m": 1800.0,
    "10m": 600.0,
    "5m": 300.0,
    "none": 0.0,
}
DEFAULT_RETENTION = "default"

# Budget policy constants: a fixed 6GB line ignored the actual card, which is
# why the first screen showed "6144" on a 7.93GiB GPU and "used" was a sum of
# table estimates (2026-11-21 bug report).
BUDGET_HEADROOM_FRACTION = 0.8
BUDGET_RESERVE_MB = 512
FALLBACK_BUDGET_MB = 6144


def cuda_allocated_mb() -> int | None:
    """Real tensor footprint of this process, or None without a usable GPU.

    ``memory_allocated`` excludes the CUDA context (~300-500MB) and the
    caching allocator's spare blocks - exactly the weights-and-activations
    number the residency panel is supposed to show.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.memory_allocated() // (1 << 20))
    except Exception:  # noqa: BLE001 - measurement must never break residency
        return None


class VramPolicyError(RuntimeError):
    """Engine estimate cannot ever fit the configured budget."""


class LruVramManager:
    def __init__(
        self,
        budget_mb: int = FALLBACK_BUDGET_MB,
        retention: str = DEFAULT_RETENTION,
        total_vram_mb: int | None = None,
    ) -> None:
        if budget_mb <= 0:
            raise ValueError("budget_mb must be positive")
        self._budget = budget_mb
        self._resident: OrderedDict[str, EngineBase] = OrderedDict()
        self._last_used: dict[str, float] = {}
        self._retention = retention if retention in RETENTION_TIERS else DEFAULT_RETENTION
        self._total_vram_mb = total_vram_mb
        if total_vram_mb:
            self._budget = self._budget_for(total_vram_mb)

    # ---- policy --------------------------------------------------------

    def _budget_for(self, total_mb: int) -> int:
        if self._retention == "forever":
            return max(1024, total_mb - BUDGET_RESERVE_MB)
        return max(1024, int(total_mb * BUDGET_HEADROOM_FRACTION))

    @property
    def budget_mb(self) -> int:
        return self._budget

    @property
    def retention(self) -> str:
        return self._retention

    @property
    def ttl_s(self) -> float | None:
        return RETENTION_TIERS.get(self._retention, None)

    @property
    def total_vram_mb(self) -> int | None:
        return self._total_vram_mb

    def set_total_vram(self, total_mb: int | None) -> None:
        """Adopt the probed card size; recompute the budget, keep residents."""
        if not total_mb or total_mb <= 0:
            return
        self._total_vram_mb = total_mb
        self._budget = self._budget_for(total_mb)

    def set_retention(self, tier: str) -> dict[str, Any]:
        if tier not in RETENTION_TIERS:
            raise ValueError(f"unknown retention tier {tier!r}")
        self._retention = tier
        if self._total_vram_mb:
            self._budget = self._budget_for(self._total_vram_mb)
        return self.policy()

    def policy(self) -> dict[str, Any]:
        return {
            "retention": self._retention,
            "ttl_s": self.ttl_s,
            "budget_mb": self._budget,
            "total_vram_mb": self._total_vram_mb,
        }

    # ---- introspection -------------------------------------------------

    def resident_ids(self) -> list[str]:
        """Resident engine ids, oldest first (eviction order)."""
        return list(self._resident)

    def _footprint(self, engine: EngineBase) -> int:
        actual = getattr(engine, "actual_vram_mb", None)
        return int(actual) if actual is not None else engine.est_vram_mb

    def used_mb(self) -> int:
        return sum(self._footprint(e) for e in self._resident.values())

    def measured(self) -> bool:
        """True when at least one resident engine carries a measured number."""
        return any(getattr(e, "actual_vram_mb", None) is not None for e in self._resident.values())

    def resident_detail(self) -> list[dict[str, Any]]:
        now = time.monotonic()
        out: list[dict[str, Any]] = []
        for eid, engine in self._resident.items():
            actual = getattr(engine, "actual_vram_mb", None)
            last = self._last_used.get(eid, now)
            out.append(
                {
                    "id": eid,
                    "lang": engine.lang,
                    "est_mb": engine.est_vram_mb,
                    "actual_mb": actual,
                    "idle_s": max(0, int(now - last)),
                }
            )
        return out

    def is_resident(self, engine_id: str) -> bool:
        return engine_id in self._resident

    # ---- lifecycle -----------------------------------------------------

    def acquire(self, engine: EngineBase, keep_langs: set[str] | None = None) -> None:
        """Ensure ``engine`` is loaded, evicting others as needed."""
        now = time.monotonic()
        if engine.engine_id in self._resident:
            self._resident.move_to_end(engine.engine_id)
            self._last_used[engine.engine_id] = now
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
                if not self._evict_one(keep=engine.engine_id, keep_langs=keep_langs):
                    break

        before = cuda_allocated_mb()
        engine.load()
        after = cuda_allocated_mb()
        if before is not None and after is not None and after > before:
            # only a positive delta is a real footprint; a no-op load (stub /
            # test double) or a shrinking pool keeps the declared estimate
            engine.actual_vram_mb = after - before  # type: ignore[attr-defined]
        self._resident[engine.engine_id] = engine
        self._resident.move_to_end(engine.engine_id)
        self._last_used[engine.engine_id] = now

    def touch(self, engine_id: str) -> None:
        """Mark an engine as just used (idle timers restart after a detect)."""
        if engine_id in self._resident:
            self._last_used[engine_id] = time.monotonic()

    def release(self, engine_id: str) -> bool:
        engine = self._resident.pop(engine_id, None)
        if engine is None:
            return False
        self._last_used.pop(engine_id, None)
        engine.unload()
        if hasattr(engine, "actual_vram_mb"):
            engine.actual_vram_mb = None  # type: ignore[attr-defined]
        return True

    def sweep_idle(self, now: float | None = None) -> list[str]:
        """Release engines idle beyond the tier TTL; returns released ids.

        ``none`` (TTL 0) is handled by the caller right after a detect batch:
        sweeping it here would unload an engine between two engines of the
        same run.
        """
        ttl = self.ttl_s
        if ttl is None or ttl <= 0:
            return []
        clock = time.monotonic() if now is None else now
        released: list[str] = []
        for eid in list(self._resident):
            last = self._last_used.get(eid, clock)
            if clock - last >= ttl and self.release(eid):
                released.append(eid)
        return released

    def evict_all(self) -> None:
        self._evict_all()

    def _evict_one(self, keep: str | None = None, keep_langs: set[str] | None = None) -> bool:
        """Evict one engine: language-outsiders first, then plain LRU."""
        if keep_langs:
            for eid, engine in self._resident.items():
                if eid != keep and engine.lang not in keep_langs:
                    return self.release(eid)
        for eid in list(self._resident):
            if eid != keep:
                return self.release(eid)
        return False

    def _evict_oldest(self, keep: str | None = None) -> None:
        self._evict_one(keep=keep)

    def _evict_all(self, except_id: str | None = None) -> None:
        for eid in list(self._resident):
            if eid != except_id:
                self.release(eid)

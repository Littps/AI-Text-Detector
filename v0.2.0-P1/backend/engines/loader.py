"""EngineLoader: facade over the catalogue + LRU manager.

``detect(text, engine_ids)`` answers a whole batch in one call - engines
run sequentially on purpose: the LRU manager serializes GPU residency,
so parallel dispatch would only thrash the eviction policy.
"""
from __future__ import annotations

from collections.abc import Iterable

from .base import EngineBase, EngineResult, WeightsMissing
from .lru import LruVramManager, VramPolicyError
from .registry import ENGINE_SPECS, EngineSpec, validate
from .stub import StubEngine


class EngineLoader:
    def __init__(
        self,
        specs: tuple[EngineSpec, ...] = ENGINE_SPECS,
        engines: dict[str, EngineBase] | None = None,
        budget_mb: int = 6144,
    ) -> None:
        validate()
        self._specs = {s.engine_id: s for s in specs}
        if engines is None:
            engines = {
                s.engine_id: StubEngine(s.engine_id, s.lang, s.kind, s.est_vram_mb, s.exclusive)
                for s in specs
            }
        unknown = set(engines) - set(self._specs)
        if unknown:
            raise ValueError(f"engines not in catalogue: {sorted(unknown)}")
        self._engines = engines
        self._lru = LruVramManager(budget_mb=budget_mb)

    # ---- introspection -------------------------------------------------

    def status(self) -> list[dict]:
        out = []
        for sid, spec_obj in self._specs.items():
            engine = self._engines[sid]
            out.append(
                {
                    "id": sid,
                    "name": spec_obj.name,
                    "lang": spec_obj.lang,
                    "kind": spec_obj.kind,
                    "est_vram_mb": spec_obj.est_vram_mb,
                    "exclusive": spec_obj.exclusive,
                    "verified": spec_obj.verified,
                    "loaded": engine.loaded,
                }
            )
        return out

    def resident_ids(self) -> list[str]:
        return self._lru.resident_ids()

    @property
    def budget_mb(self) -> int:
        return self._lru.budget_mb

    def used_mb(self) -> int:
        return self._lru.used_mb()

    # ---- lifecycle -----------------------------------------------------

    def load(self, engine_ids: Iterable[str]) -> list[EngineResult]:
        results = []
        for sid in engine_ids:
            results.append(self._load_one(sid))
        return results

    def _load_one(self, engine_id: str) -> EngineResult:
        engine = self._engines.get(engine_id)
        if engine is None:
            return EngineResult(engine_id, "unknown_engine", detail="not in catalogue")
        try:
            self._lru.acquire(engine)
        except VramPolicyError as e:
            return EngineResult(engine_id, "error", detail=str(e))
        except WeightsMissing as e:
            return EngineResult(engine_id, "weights_missing", detail=str(e))
        return EngineResult(engine_id, "ok")

    def unload(self, engine_ids: Iterable[str]) -> list[EngineResult]:
        results = []
        for sid in engine_ids:
            ok = self._lru.release(sid)
            results.append(
                EngineResult(sid, "ok" if ok else "unknown_engine", detail="" if ok else "not resident")
            )
        return results

    # ---- detection -----------------------------------------------------

    def detect(self, text: str, engine_ids: Iterable[str]) -> list[EngineResult]:
        results: list[EngineResult] = []
        for sid in engine_ids:
            engine = self._engines.get(sid)
            if engine is None:
                results.append(EngineResult(sid, "unknown_engine", detail="not in catalogue"))
                continue
            try:
                self._lru.acquire(engine)
                segments = engine.detect(text)
            except WeightsMissing as e:
                results.append(EngineResult(sid, "weights_missing", detail=str(e)))
                continue
            except VramPolicyError as e:
                results.append(EngineResult(sid, "error", detail=str(e)))
                continue
            except Exception as e:  # noqa: BLE001 - one bad engine must not kill the batch
                results.append(EngineResult(sid, "error", detail=f"{type(e).__name__}: {e}"))
                continue
            results.append(EngineResult(sid, "ok", tuple(segments)))
        return results

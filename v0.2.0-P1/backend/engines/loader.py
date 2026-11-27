"""EngineLoader: facade over the catalogue + LRU manager.

``detect(text, engine_ids)`` answers a whole batch in one call - engines
run sequentially on purpose: the LRU manager serializes GPU residency,
so parallel dispatch would only thrash the eviction policy.

P3-a: an optional StoragePolicy (D25) links the disk state to the load
knobs - fp16 payloads on disk switch the engine to variant="fp16" /
forced-fp16 dtype before acquire() ever calls load().
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from manifest import Manifest
from storage import (
    DEFAULT_MODE,
    FP16_VARIANT,
    CastIndex,
    StoragePolicy,
    fp16_payloads_ok,
    read_cast_index,
)

from .base import DetectCancelled, EngineBase, EngineResult, Segment, WeightsMissing
from .lru import DEFAULT_RETENTION, LruVramManager, VramPolicyError
from .registry import ENGINE_SPECS, EngineSpec, validate
from .stub import StubEngine


def doc_score(segments: tuple[Segment, ...] | list[Segment]) -> float | None:
    """D29 whole-document AI rate: length-weighted segment mean.

    Σ(score×len)/Σ(len) over ALL scored spans; a document-level engine's
    single full-text segment degenerates to its raw score.
    """
    total = sum(s.length for s in segments)
    if not segments or total <= 0:
        return None
    return round(sum(s.score * s.length for s in segments) / total, 4)


def _has_weights(weights: Path) -> bool:
    """safetensors first; pytorch_model.bin is the legacy fallback layout."""
    return (weights / "model.safetensors").is_file() or (weights / "pytorch_model.bin").is_file()


def default_engine(spec: EngineSpec) -> EngineBase:
    """Real engine when weights exist locally, stub otherwise.

    Three routes:
    - kind="method": zero-shot engines over the shared GPT-2 base; they
      never live in models/<engine_id>, so the weight-folder check below
      does not apply. Unimplemented methods fall through to a stub.
    - config.json declares a custom architecture: dedicated subclass
      (custom_models.CUSTOM_ENGINE_CLASSES).
    - default: generic RealClassifierEngine when a weight folder exists.
    """
    if spec.kind == "method":
        if spec.engine_id == "gltr":
            from .gltr import GltrEngine

            return GltrEngine(spec)
        if spec.engine_id == "fast-detectgpt":
            from .fast_detectgpt import FastDetectGptEngine

            return FastDetectGptEngine(spec)
        if spec.engine_id == "binoculars":
            from .binoculars import BinocularsEngine

            return BinocularsEngine(spec)
        return StubEngine(spec.engine_id, spec.lang, spec.kind, spec.est_vram_mb, spec.exclusive)

    weights = Path(__file__).resolve().parent.parent.parent / "models" / spec.engine_id
    if (weights / "config.json").is_file() and _has_weights(weights):
        from .custom_models import CUSTOM_ENGINE_CLASSES
        from .real import RealClassifierEngine

        cls = CUSTOM_ENGINE_CLASSES.get(spec.engine_id, RealClassifierEngine)
        return cls(spec, weights)
    return StubEngine(spec.engine_id, spec.lang, spec.kind, spec.est_vram_mb, spec.exclusive)


class EngineLoader:
    def __init__(
        self,
        specs: tuple[EngineSpec, ...] = ENGINE_SPECS,
        engines: dict[str, EngineBase] | None = None,
        budget_mb: int = 6144,
        policy: StoragePolicy | None = None,
        manifest: Manifest | None = None,
        models_root: Path | None = None,
        retention: str = DEFAULT_RETENTION,
        total_vram_mb: int | None = None,
    ) -> None:
        validate()
        self._specs = {s.engine_id: s for s in specs}
        if engines is None:
            engines = {s.engine_id: default_engine(s) for s in specs}
        unknown = set(engines) - set(self._specs)
        if unknown:
            raise ValueError(f"engines not in catalogue: {sorted(unknown)}")
        self._engines = engines
        self._lru = LruVramManager(
            budget_mb=budget_mb, retention=retention, total_vram_mb=total_vram_mb
        )
        # D25 linkage: without a policy (tests, stubs) the load knobs stay
        # at their defaults and behaviour is byte-identical to P1.
        self._policy = policy
        self._manifest = manifest
        self._models_root = models_root

    # ---- introspection -------------------------------------------------

    def status(self) -> list[dict]:
        out = []
        resident = set(self._lru.resident_ids())
        for sid, spec_obj in self._specs.items():
            engine = self._engines[sid]
            out.append(
                {
                    "id": sid,
                    "name": spec_obj.name,
                    "lang": spec_obj.lang,
                    "kind": spec_obj.kind,
                    "est_vram_mb": spec_obj.est_vram_mb,
                    "actual_vram_mb": getattr(engine, "actual_vram_mb", None),
                    "exclusive": spec_obj.exclusive,
                    "verified": spec_obj.verified,
                    "loaded": engine.loaded,
                    "resident": sid in resident,
                }
            )
        return out

    def resident_ids(self) -> list[str]:
        return self._lru.resident_ids()

    def resident_detail(self) -> list[dict]:
        names = {s.engine_id: s.name for s in self._specs.values()}
        return [{**row, "name": names.get(row["id"], row["id"])} for row in self._lru.resident_detail()]

    @property
    def budget_mb(self) -> int:
        return self._lru.budget_mb

    @property
    def retention(self) -> str:
        return self._lru.retention

    def used_mb(self) -> int:
        return self._lru.used_mb()

    def measured(self) -> bool:
        return self._lru.measured()

    def set_total_vram(self, total_mb: int | None) -> None:
        self._lru.set_total_vram(total_mb)

    def set_retention(self, tier: str) -> dict:
        return self._lru.set_retention(tier)

    def policy(self) -> dict:
        return self._lru.policy()

    def langs_of(self, engine_ids: Iterable[str]) -> set[str]:
        """Languages of a run: drives language-aware eviction priority."""
        return {
            self._engines[sid].lang
            for sid in engine_ids
            if sid in self._engines
        }

    def sweep_idle(self) -> list[str]:
        return self._lru.sweep_idle()

    # ---- lifecycle -----------------------------------------------------

    def load(self, engine_ids: Iterable[str]) -> list[EngineResult]:
        results = []
        for sid in engine_ids:
            results.append(self._load_one(sid))
        return results

    def _configure_load(self, engine: EngineBase) -> None:
        """D24⑤/D25: pick load dtype + variant from the on-disk state.

        - fp16 payloads present (dual/fp16-only): forced fp16 dtype;
          variant="fp16" when a payload sits at the ``.fp16`` name (dual),
          None when it occupies the original name (fp16-only) -
          from_pretrained resolves exactly one candidate either way.
        - explicit "float32" with fp16-only payloads on disk is refused
          (the original-name file holds fp16 data; D25 warning copy).
        - keep-original / no policy: knobs untouched (P1 behaviour).
        """
        if self._policy is None or self._models_root is None:
            return
        eid = engine.engine_id
        mode = self._policy.mode(eid)
        if mode == DEFAULT_MODE:
            engine.load_variant = None
            return
        index: CastIndex | None = read_cast_index(self._models_root, eid)
        entry = self._manifest.get(eid) if self._manifest is not None else None
        if index is None or not fp16_payloads_ok(index, self._models_root, entry):
            engine.load_variant = None  # originals path; in-memory cast as before
            return
        if engine.load_dtype == "float32":
            raise ValueError(
                f"{eid}: fp16-only storage on disk - float32 load unavailable "
                "(switch storage mode back to keep-original first)"
            )
        engine.load_dtype = "float16"
        engine.load_variant = (
            FP16_VARIANT if any(c.dest != c.source_dest for c in index.casts) else None
        )

    def _load_one(self, engine_id: str) -> EngineResult:
        engine = self._engines.get(engine_id)
        if engine is None:
            return EngineResult(engine_id, "unknown_engine", detail="not in catalogue")
        try:
            self._configure_load(engine)
        except ValueError as e:
            return EngineResult(engine_id, "error", detail=str(e))
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

    def refresh_engine(self, engine_id: str) -> bool:
        """Rebuild one engine instance from the current on-disk state.

        A custom import that wires an official engine lands weights into
        models/<engine_id> AFTER boot; the instance created at boot was a
        StubEngine (weights absent back then). Re-running default_engine
        swaps it for the real implementation without a sidecar restart.
        A resident engine is released first so the LRU entry tracks the
        new instance. Callers must run on the event loop thread - detect
        workers hold the old reference, which stays valid until done.
        Returns True when the instance was rebuilt.
        """
        spec_obj = self._specs.get(engine_id)
        if spec_obj is None:
            return False
        engine = self._engines.get(engine_id)
        if engine is not None and engine.loaded:
            self._lru.release(engine_id)
        self._engines[engine_id] = default_engine(spec_obj)
        return True

    # ---- detection -----------------------------------------------------

    def detect(self, text: str, engine_ids: Iterable[str]) -> list[EngineResult]:
        ids = list(engine_ids)
        langs = self.langs_of(ids)
        return [self.detect_one(text, sid, keep_langs=langs) for sid in ids]

    def detect_one(
        self,
        text: str,
        engine_id: str,
        on_batch: Callable[[str, list[Segment]], None] | None = None,
        keep_langs: set[str] | None = None,
    ) -> EngineResult:
        """One engine over one document (D29 protocol extension).

        ``on_batch(engine_id, segments)`` fires per completed sentence
        batch (streaming); the returned result carries ``doc_score`` -
        the length-weighted mean AI rate, which degenerates to the raw
        score for document-level engines (desklib) with zero
        special-casing.
        """
        engine = self._engines.get(engine_id)
        if engine is None:
            return EngineResult(engine_id, "unknown_engine", detail="not in catalogue")

        # D25 round-25: the on-disk fp16 copies must steer the *real* load.
        # Before this, only ``_load_one`` (the explicit warm-up path) applied
        # the knobs; ``detect_one`` went straight to ``lru.acquire`` and loaded
        # the fp32 original - dual mode silently did nothing and fp16-only
        # only worked because the original name happened to hold fp16 data.
        try:
            self._configure_load(engine)
        except ValueError as e:
            return EngineResult(engine_id, "error", detail=str(e))

        def batch_cb(segments: list[Segment]) -> None:
            if on_batch is not None:
                on_batch(engine_id, segments)

        try:
            self._lru.acquire(engine, keep_langs=keep_langs)
            segments = engine.detect(text, batch_cb if on_batch is not None else None)
            self._lru.touch(engine_id)
        except DetectCancelled:
            # the caller asked to stop between batches: not a scoring failure,
            # so it must not be reported as "error" (round-19)
            return EngineResult(engine_id, "cancelled", detail="cancelled by client")
        except WeightsMissing as e:
            return EngineResult(engine_id, "weights_missing", detail=str(e))
        except VramPolicyError as e:
            return EngineResult(engine_id, "error", detail=str(e))
        except Exception as e:  # noqa: BLE001 - one bad engine must not kill the batch
            return EngineResult(engine_id, "error", detail=f"{type(e).__name__}: {e}")
        return EngineResult(engine_id, "ok", tuple(segments), doc_score=doc_score(segments))

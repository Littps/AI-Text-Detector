"""Model storage domain (D25): per-engine storage mode + cast-index.

Three modes per engine (default ``keep-original``):

- keep-original: only the fp32 originals from the manifest; load casts
  to fp16 in memory (status quo behaviour).
- dual: fp32 originals + one fp16 copy per safetensors weight; the copy
  lives at ``<stem>.fp16.safetensors`` and is loaded via
  ``from_pretrained(variant="fp16")`` (transformers 5.x resolves variant
  names without fallback - verified in modeling_utils._add_variant).
- fp16-only: fp16 payloads replace the originals (the fp32 file is
  overwritten only after the copy is fully verified) - ~50% disk. The
  manifest SHA chain is gone by design; the **cast-index** (fp16 sha256
  + source fp32 sha256) becomes the integrity anchor.

The cast-index is the D25 anti-trap: with fp16-only payloads on disk the
manifest-sized fp32 files are *expected absent*. Naive ``is_complete()``
would flag them missing and re-download 10.67 GB; every completeness
question must go through this module instead.

Crash safety of the fp16-only finalize (the only step that destroys
data): the index is written BEFORE the fp16 payload replaces the fp32
original, and ``CastFile.candidates()`` accepts either the finalized
name or the intermediate ``.fp16`` name - every crash window leaves a
state ``storage_status`` reads correctly and a rerun of ``cast`` repairs
idempotently (fp16 -> fp16 recast is lossless).

Pure stdlib; the torch-dependent cast routine imports lazily so the
module (and every test that does not cast) stays torch-free.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from downloader import (
    ChecksumError,
    Progress,
    missing_files,
    sha256_file,
    verify_file_integrity,
)
from manifest import EngineEntry, RemoteFile, safe_relative_path

MODES = ("keep-original", "dual", "fp16-only")
DEFAULT_MODE = "keep-original"
FP16_VARIANT = "fp16"

POLICY_VERSION = 1
INDEX_VERSION = 1


# ---- naming -------------------------------------------------------------


def fp16_dest(dest: str) -> str:
    """``<engine>/model.safetensors`` -> ``<engine>/model.fp16.safetensors``."""
    p = PurePosixPath(dest)
    return str(p.with_name(f"{p.stem}.{FP16_VARIANT}{p.suffix}"))


def weight_files(entry: EngineEntry) -> tuple[RemoteFile, ...]:
    """Cast candidates: safetensors weights only.

    pytorch_model.bin engines (simpleai-zh) are excluded by design: D25
    scopes the switch to large models and a pickle-format 409 MB
    detector gains nothing from a local cast.
    """
    return tuple(f for f in entry.files if f.dest.endswith(".safetensors"))


def _is_file_with_size(p: Path, size: int) -> bool:
    return p.is_file() and p.stat().st_size == size


# ---- cast-index ---------------------------------------------------------


@dataclass(frozen=True)
class CastFile:
    """One fp16 payload recorded in the index.

    dest is the *finalized* name: for dual it is the ``.fp16`` copy name;
    for fp16-only it equals source_dest (fp16 payload under the original
    name). ``candidates()`` yields every name the payload may legally sit
    at, so crash windows between finalize steps stay readable.
    """

    dest: str
    sha256: str
    bytes: int
    source_dest: str
    source_sha256: str

    def candidates(self) -> list[str]:
        if self.dest == self.source_dest:
            return [self.dest, fp16_dest(self.dest)]
        return [self.dest]


@dataclass(frozen=True)
class CastIndex:
    engine_id: str
    mode: str  # mode at cast time (audit); runtime follows StoragePolicy
    casts: tuple[CastFile, ...]
    casted_at: str


def cast_index_path(models_root: Path, engine_id: str) -> Path:
    # The engine id originates in the manifest.  Validate it here as well
    # because this path is also used by cleanup/uninstall.
    rel = safe_relative_path(f"{engine_id}/cast-index.json")
    return models_root / rel


def read_cast_index(models_root: Path, engine_id: str) -> CastIndex | None:
    """Parse the index; any malformed file reads as None (damaged -> recast)."""
    try:
        p = _safe_model_target(models_root, f"{engine_id}/cast-index.json")
        raw = json.loads(p.read_text(encoding="utf-8"))
        if raw.get("version") != INDEX_VERSION or raw.get("engine_id") != engine_id:
            return None
        casts_list = raw["casts"]
        if not isinstance(casts_list, list):
            return None
        casts: list[CastFile] = []
        for c in casts_list:
            if not isinstance(c, dict):
                return None
            dest = safe_relative_path(c["dest"])
            source_dest = safe_relative_path(c["source_dest"])
            if not source_dest.endswith(".safetensors"):
                return None
            if dest not in {source_dest, fp16_dest(source_dest)}:
                return None
            size = int(c["bytes"])
            if size < 0:
                return None
            sha = c.get("sha256")
            if not isinstance(sha, str) or not sha:
                return None
            casts.append(
                CastFile(
                    dest=dest,
                    sha256=sha,
                    bytes=size,
                    source_dest=source_dest,
                    source_sha256=str(c.get("source_sha256", "")),
                )
            )
        return CastIndex(
            engine_id=engine_id,
            mode=raw.get("mode", "dual"),
            casts=tuple(casts),
            casted_at=raw.get("casted_at", ""),
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def write_cast_index(models_root: Path, index: CastIndex) -> None:
    # Never let a malformed index become a future deletion primitive.
    for c in index.casts:
        safe_relative_path(c.dest)
        safe_relative_path(c.source_dest)
        if not c.source_dest.endswith(".safetensors"):
            raise ValueError(f"cast source is not safetensors: {c.source_dest!r}")
        if c.dest not in {c.source_dest, fp16_dest(c.source_dest)}:
            raise ValueError(f"cast target is not paired with source: {c.dest!r}")
    p = _safe_model_target(models_root, f"{index.engine_id}/cast-index.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(
        json.dumps(
            {
                "version": INDEX_VERSION,
                "engine_id": index.engine_id,
                "mode": index.mode,
                "casted_at": index.casted_at,
                "casts": [
                    {
                        "dest": c.dest,
                        "sha256": c.sha256,
                        "bytes": c.bytes,
                        "source_dest": c.source_dest,
                        "source_sha256": c.source_sha256,
                    }
                    for c in index.casts
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(tmp, p)


# ---- policy -------------------------------------------------------------


class StoragePolicy:
    """D25 per-engine storage mode, persisted at ``models/storage.json``."""

    def __init__(self, path: Path, modes: dict[str, str] | None = None) -> None:
        self._path = path
        self._modes: dict[str, str] = dict(modes or {})

    @classmethod
    def load(cls, path: Path) -> StoragePolicy:
        modes: dict[str, str] = {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("version") == POLICY_VERSION:
                for eid, mode in raw.get("engines", {}).items():
                    if mode in MODES:
                        modes[eid] = mode
        except (OSError, ValueError):
            pass  # absent/corrupt policy = all defaults
        return cls(path, modes)

    def mode(self, engine_id: str) -> str:
        return self._modes.get(engine_id, DEFAULT_MODE)

    def set_mode(self, engine_id: str, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown storage mode {mode!r}; expected one of {MODES}")
        if mode == DEFAULT_MODE:
            self._modes.pop(engine_id, None)  # default is implicit
        else:
            self._modes[engine_id] = mode
        self._save()

    def clear(self, engine_id: str) -> None:
        """Forget an uninstalled engine's mode."""
        if engine_id not in self._modes:
            return
        self._modes.pop(engine_id, None)
        self._save()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(
            json.dumps({"version": POLICY_VERSION, "engines": self._modes}, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self._path)


# ---- storage-aware completeness (the D25 anti-trap) ---------------------


@dataclass(frozen=True)
class StorageStatus:
    mode: str
    fp32_present: bool  # manifest-anchored originals all on disk
    fp16_present: bool  # recorded cast payloads available (size-checked)
    installed: bool  # mode-aware completeness (drives "needs download?")
    recast_needed: bool  # fp32 present but fp16 payloads missing/damaged
    stale_cast: bool  # fp16 payloads present but mode is keep-original
    dtype_options: tuple[str, ...]  # load-time dtypes this disk state allows

    def as_dict(self) -> dict:
        return {
            "storage_mode": self.mode,
            "fp32_present": self.fp32_present,
            "fp16_present": self.fp16_present,
            "installed": self.installed,
            "recast_needed": self.recast_needed,
            "stale_cast": self.stale_cast,
            "dtype_options": list(self.dtype_options),
        }


def fp16_payloads_ok(
    index: CastIndex,
    models_root: Path,
    entry: EngineEntry | None = None,
) -> bool:
    """All recorded payloads are present and match their recorded SHA.

    When ``entry`` is supplied, the index must describe the complete set of
    safetensors weights for that engine.  This closes the old partial-index
    hole where one valid cast made a two-weight engine look installed.
    """
    if not index.casts:
        return False
    if entry is not None:
        expected = {f.dest for f in weight_files(entry)}
        recorded = {c.source_dest for c in index.casts}
        if recorded != expected:
            return False
    for c in index.casts:
        if not any(
            verify_file_integrity(
                models_root,
                name,
                models_root / name,
                c.bytes,
                c.sha256,
            )
            for name in c.candidates()
        ):
            return False
    return True


def storage_status(entry: EngineEntry, models_root: Path, mode: str) -> StorageStatus:
    if mode not in MODES:
        raise ValueError(f"unknown storage mode {mode!r}")
    fp32_present = not missing_files(entry, models_root)
    index = read_cast_index(models_root, entry.engine_id)
    fp16_present = index is not None and fp16_payloads_ok(index, models_root, entry)

    # Round-25: dual -> fp16-only leaves a *dual-layout* index (payload at the
    # ``.fp16`` name) with the fp32 original still in place. The payloads are
    # reachable, but the mode's actual promise - "the original name holds fp16
    # data, disk halved" - has not happened, and without this flag the UI shows
    # no 待转换 and no 转换 button, so the finalize step could never run.
    dual_layout = index is not None and any(c.dest != c.source_dest for c in index.casts)
    finalize_needed = mode == "fp16-only" and fp32_present and fp16_present and dual_layout

    installed = {
        "keep-original": fp32_present,
        "dual": fp32_present and fp16_present,
        "fp16-only": fp16_present and not finalize_needed,
    }[mode]
    recast_needed = (
        mode in ("dual", "fp16-only") and fp32_present and not fp16_present
    ) or finalize_needed
    stale_cast = mode == "keep-original" and fp16_present
    # float32 load needs true fp32 originals: in fp16-only mode the
    # original-name file (if any) holds fp16 payloads, so fp32 load is
    # unavailable by design (D25 warning copy says exactly this).
    dtype_options = (
        ("float16", "float32") if (fp32_present and mode != "fp16-only") else ("float16",)
    )
    return StorageStatus(
        mode=mode,
        fp32_present=fp32_present,
        fp16_present=fp16_present,
        installed=installed,
        recast_needed=recast_needed,
        stale_cast=stale_cast,
        dtype_options=dtype_options,
    )


def is_complete_storage(entry: EngineEntry, models_root: Path, mode: str) -> bool:
    return storage_status(entry, models_root, mode).installed


def missing_files_storage(
    entry: EngineEntry, models_root: Path, mode: str
) -> list[RemoteFile]:
    """Download plan under a storage mode (the D25 anti-trap fix).

    Returns [] when the mode-intended state is already on disk. Missing
    fp16 payloads with fp32 originals present are a *recast* (run the
    cast job), never a re-download - that is exactly the 10.67 GB trap.
    """
    if mode == "keep-original":
        return missing_files(entry, models_root)
    st = storage_status(entry, models_root, mode)
    if st.installed:
        return []
    fp32_missing = missing_files(entry, models_root)
    if fp32_missing:
        return fp32_missing  # download originals first; cast runs after
    return []  # originals present, payloads missing -> recast path


# ---- cast execution -----------------------------------------------------


def cast_safetensors_fp16(
    src: Path,
    dst: Path,
    cancel: threading.Event,
    on_progress: Callable[[int, int], None] | None = None,
) -> None:
    """Stream-cast one safetensors file to fp16 (atomic via tmp+replace).

    Tensors are converted one at a time so peak RAM ~= fp16 payload size
    + one transient fp32 tensor (10.67 GB source -> ~5.4 GB working set
    instead of double). The caller verifies the result and lands it.
    """
    import torch  # lazy: keep module import torch-free
    from safetensors.torch import safe_open, save_file

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".incomplete")
    if cancel.is_set():
        raise KeyboardInterrupt("cancelled before cast")

    out: dict[str, torch.Tensor] = {}
    done_bytes = 0
    with safe_open(src, framework="pt") as f:
        metadata = f.metadata() or {}
        keys = list(f.keys())
        total = src.stat().st_size
        for i, k in enumerate(keys):
            if cancel.is_set():
                raise KeyboardInterrupt(f"cancelled at tensor {i}/{len(keys)}")
            t = f.get_tensor(k)
            done_bytes += t.numel() * t.element_size()
            out[k] = t.to(torch.float16)
            del t
            if on_progress is not None:
                on_progress(done_bytes, total)
    if cancel.is_set():
        raise KeyboardInterrupt("cancelled before save")
    merged = {**metadata, "cast": "fp16"} if metadata else {"cast": "fp16"}
    save_file(out, str(tmp), metadata=merged)
    os.replace(tmp, dst)


def cast_engine_files(
    entry: EngineEntry,
    models_root: Path,
    mode: str,
    cancel: threading.Event,
    progress: Callable[[Progress], None] | None = None,
) -> CastIndex:
    """Cast every safetensors weight to fp16 and record the index.

    dual: fp32 originals stay untouched; payloads land at ``.fp16`` names.
    fp16-only: payloads are verified + indexed first, then each fp32
    original is atomically overwritten by its fp16 payload (delete-free -
    os.replace keeps every crash window recoverable, see module doc).
    """
    if mode not in ("dual", "fp16-only"):
        raise ValueError(f"cast applies to dual/fp16-only, got {mode!r}")
    weights = weight_files(entry)
    if not weights:
        raise ValueError(f"{entry.engine_id}: no safetensors weights to cast")

    # idempotent reruns: after an fp16-only finalize the original-name file
    # holds fp16 payloads (manifest size no longer matches). Accept either
    # the fp32 original or the prior index's fp16 size as a valid source -
    # fp16 -> fp16 recast is lossless and repairs crash windows.
    prior = read_cast_index(models_root, entry.engine_id)
    prior_by_source = {c.source_dest: c for c in prior.casts} if prior else {}

    total = sum(f.bytes for f in weights)
    state = {"done": 0}
    casts: list[CastFile] = []
    for f in weights:
        src = models_root / f.dest
        prior_cast = prior_by_source.get(f.dest) if mode == "fp16-only" else None
        expected_bytes = f.bytes
        expected_sha = f.sha256
        if (
            prior_cast is not None
            and src.is_file()
            and src.stat().st_size == prior_cast.bytes
        ):
            # fp16-only reruns/repairs read the payload under the original
            # name, whose size is no longer the manifest's fp32 size.
            expected_bytes = prior_cast.bytes
            expected_sha = prior_cast.sha256
        if not src.is_file() or src.stat().st_size != expected_bytes:
            raise FileNotFoundError(f"{f.dest}: fp32 original missing/size-mismatch")
        if expected_sha and not verify_file_integrity(
            models_root, f.dest, src, expected_bytes, expected_sha
        ):
            raise ChecksumError(f"{f.dest}: source SHA256 mismatch")

        def cb(file_done: int, _total: int, _f: RemoteFile = f) -> None:
            if progress is not None:
                progress(
                    Progress(
                        entry.engine_id, _f.dest, state["done"] + file_done, total, 0.0
                    )
                )

        payload = models_root / fp16_dest(f.dest)
        cast_safetensors_fp16(src, payload, cancel, on_progress=cb)
        actual = sha256_file(payload)
        casts.append(
            CastFile(
                dest=fp16_dest(f.dest),
                sha256=actual,
                bytes=payload.stat().st_size,
                source_dest=f.dest,
                source_sha256=f.sha256 or "",
            )
        )
        state["done"] += f.bytes
        if progress is not None:
            progress(Progress(entry.engine_id, f.dest, state["done"], total, 0.0))

    index = CastIndex(
        engine_id=entry.engine_id,
        mode=mode,
        casts=tuple(casts),
        casted_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    # crash-safety: index BEFORE any fp16-only overwrite
    write_cast_index(models_root, index)

    if mode == "fp16-only":
        finalized = tuple(replace(c, dest=c.source_dest) for c in index.casts)
        for c in finalized:
            payload = models_root / fp16_dest(c.dest)
            if not _is_file_with_size(payload, c.bytes):
                raise FileNotFoundError(f"{c.dest}: fp16 payload vanished before finalize")
            os.replace(payload, models_root / c.dest)  # atomic fp16-over-fp32
        index = replace(index, casts=finalized)
        write_cast_index(models_root, index)
    return index


def cleanup_cast(models_root: Path, engine_id: str) -> list[str]:
    """Explicit cleanup: remove fp16 copies + index, never the original.

    The original-name file is left alone in every case: under
    keep-original it is the fp32 original, under fp16-only it is the
    only remaining payload - destroying it here would orphan the engine.
    (Switching fp16-only back to keep-original therefore requires a
    re-download; the UI copy says so before the user confirms.)
    """
    removed: list[str] = []
    index = read_cast_index(models_root, engine_id)
    if index is None:
        return removed
    for c in index.casts:
        for name in c.candidates():
            if name == c.source_dest:
                continue  # never touch the original-name file
            p = _safe_model_target(models_root, name)
            if p.is_file():
                p.unlink()
                removed.append(name)
    _safe_model_target(models_root, f"{engine_id}/cast-index.json").unlink(missing_ok=True)
    return removed


def _safe_model_target(models_root: Path, rel: str) -> Path:
    """Resolve a manifest-relative target without crossing a link/junction.

    Deleting an installed model is one of the few irreversible operations in
    the app. A model directory may be a user-created junction (the portable
    build historically used them), so a lexical ``models_root / rel`` check is
    not enough: removing through that parent could delete the real model cache
    or an arbitrary directory.
    """
    try:
        safe_relative_path(rel)
    except ValueError as e:
        raise ValueError(f"unsafe model path {rel!r}: {e}") from None
    pure = PurePosixPath(rel)
    root = models_root.resolve()
    target = (models_root / Path(*pure.parts)).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as e:
        raise ValueError(f"model path escapes root: {rel!r}") from e

    current = (models_root / Path(*pure.parts)).parent
    while current != models_root and current != current.parent:
        if current.exists() and (
            current.is_symlink()
            or bool(getattr(os.path, "isjunction", lambda _p: False)(current))
        ):
            raise ValueError(f"refusing to delete through link/junction: {current}")
        current = current.parent
    return models_root / Path(*pure.parts)


def uninstall_engine_files(
    entry: EngineEntry,
    models_root: Path,
    *,
    shared_dests: set[str] | None = None,
) -> dict:
    """Delete only files owned by one manifest engine.

    ``shared_dests`` protects files referenced by another catalogue entry.
    Junctioned parents are rejected rather than followed. Unknown files are
    deliberately left behind (and reported) instead of recursively deleting a
    directory the manifest does not own.
    """
    shared = set(shared_dests or ())
    rels: set[str] = set()
    for f in entry.files:
        rels.add(f.dest)
        rels.add(f"{f.dest}.incomplete")

    index = read_cast_index(models_root, entry.engine_id)
    if index is not None:
        for cast in index.casts:
            for name in cast.candidates():
                rels.add(name)
                rels.add(f"{name}.incomplete")
    rels.add(str(PurePosixPath(entry.engine_id) / "cast-index.json"))

    targets: list[tuple[str, Path, int]] = []
    for rel in sorted(rels):
        if rel in shared:
            continue
        target = _safe_model_target(models_root, rel)
        if target.is_dir():
            continue
        if not target.exists() and not target.is_symlink():
            continue
        try:
            size = target.lstat().st_size
        except OSError:
            size = 0
        targets.append((rel, target, size))

    # Move every file into a same-volume quarantine before deleting anything.
    # A locked file therefore fails before the first committed deletion, and
    # the rollback below restores files if a later move fails.
    quarantine_root = models_root / ".uninstall-quarantine"
    if quarantine_root.exists() and (
        quarantine_root.is_symlink()
        or bool(getattr(os.path, "isjunction", lambda _p: False)(quarantine_root))
    ):
        raise ValueError("refusing to use a linked uninstall quarantine")
    quarantine = quarantine_root / f"{entry.engine_id}-{uuid.uuid4().hex}"
    moved: list[tuple[str, Path, Path]] = []
    try:
        for rel, target, _size in targets:
            parked = quarantine / Path(*PurePosixPath(rel).parts)
            parked.parent.mkdir(parents=True, exist_ok=True)
            os.replace(target, parked)
            moved.append((rel, target, parked))
    except OSError:
        for _rel, target, parked in reversed(moved):
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(parked, target)
            except OSError:
                pass
        shutil.rmtree(quarantine, ignore_errors=True)
        raise

    removed = [rel for rel, _target, _parked in moved]
    bytes_freed = sum(size for _rel, _target, size in targets)
    parents = {target.parent for _rel, target, _size in targets}
    try:
        if quarantine.exists():
            shutil.rmtree(quarantine)
    except OSError:
        # The model files are already isolated; restoring them is safer than
        # leaving the engine in a state that looks installed but cannot load.
        for _rel, target, parked in reversed(moved):
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(parked, target)
            except OSError:
                pass
        raise

    root = models_root.resolve()
    for parent in sorted(parents, key=lambda p: len(p.parts), reverse=True):
        current = parent
        while current != root and current != current.parent:
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent

    leftovers: list[str] = []
    owned_dirs = {
        (models_root / Path(*PurePosixPath(rel).parts)).parent
        for rel in (f.dest for f in entry.files)
        if rel not in shared
    }
    for directory in sorted(owned_dirs, key=lambda p: len(p.parts), reverse=True):
        if not directory.exists() or not directory.is_dir():
            continue
        stack = [directory]
        while stack:
            current = stack.pop()
            for child in current.iterdir():
                if child.is_dir():
                    if child.is_symlink() or bool(
                        getattr(os.path, "isjunction", lambda _p: False)(child)
                    ):
                        leftovers.append(child.relative_to(models_root).as_posix())
                    else:
                        stack.append(child)
                elif child.is_file() or child.is_symlink():
                    leftovers.append(child.relative_to(models_root).as_posix())

    return {
        "engine_id": entry.engine_id,
        "removed": removed,
        "bytes_freed": bytes_freed,
        "leftovers": sorted(set(leftovers)),
        "shared_kept": sorted({f.dest for f in entry.files} & shared),
    }

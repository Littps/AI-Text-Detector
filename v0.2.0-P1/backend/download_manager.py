"""Async download/cast orchestration for the WS layer.

One job at a time; engines and their files transfer sequentially
(bandwidth-bound anyway, and progress stays legible). The transfer core
(downloader.FileDownloader) and the cast core (storage.cast_engine_files)
run in worker threads; progress callbacks hop back to the event loop via
call_soon_threadsafe and are forwarded to the requesting WS client.
Disconnect cancels the job - the .incomplete resume files make that
cheap, and the next start picks up.

P3-a additions (D25 model storage domain):
- completeness questions go through storage.* with the engine's storage
  mode - in fp16-only mode the manifest-sized fp32 files are expected
  absent, and "missing fp16 with fp32 present" means a *recast* job,
  never a re-download (the 10.67 GB anti-trap);
- a job may chain download -> cast: engines whose storage mode is
  dual/fp16-only are cast right after their originals land;
- download/start with nothing to download but a recast pending reports
  reason="recast_needed" (the client shows the cast affordance);
- storage/set_mode + storage/cast + storage/cleanup drive the D25
  three-mode switch through the same single-job discipline.
"""
from __future__ import annotations

import asyncio
import functools
import threading
from collections.abc import Callable
from pathlib import Path

from downloader import (
    DEFAULT_SOURCE,
    DownloadCancelled,
    DownloadError,
    FileDownloader,
    Progress,
    templates_for,
)
from manifest import EngineEntry, Manifest, RemoteFile
from storage import (
    StoragePolicy,
    cast_engine_files,
    cleanup_cast,
    missing_files_storage,
    storage_status,
    uninstall_engine_files,
)


class DownloadManager:
    def __init__(
        self,
        manifest: Manifest,
        models_root: Path,
        policy: StoragePolicy | None = None,
    ) -> None:
        self._manifest = manifest
        self._models_root = models_root
        self._policy = policy if policy is not None else StoragePolicy(models_root / "storage.json")
        self._fd = FileDownloader(models_root)
        self._task: asyncio.Task | None = None
        self._cancel = threading.Event()
        self._active_engine: str | None = None
        self._emit: Callable[[dict], None] | None = None  # loop thread only
        self._lifecycle_lock: asyncio.Lock | None = None
        # D36 source preference: reorders the host chain per job; the other
        # host stays as the fallback (the five rules are unchanged).
        self._source = DEFAULT_SOURCE

    def set_lifecycle_lock(self, lock: asyncio.Lock) -> None:
        """Share the detect/model lock with the async job wrapper."""
        self._lifecycle_lock = lock

    # ---- event sink ----------------------------------------------------

    def attach(self, emit: Callable[[dict], None]) -> None:
        """Bind the event sink (must be called on the event loop thread)."""
        self._emit = emit

    def detach(self) -> None:
        self._emit = None

    def _emit_nowait(self, payload: dict) -> None:
        if self._emit is not None:
            self._emit(payload)

    # ---- status / mode --------------------------------------------------

    def status(self) -> dict:
        running = self._task is not None and not self._task.done()
        engines = {}
        for eid in self._manifest.ids():
            entry = self._manifest.get(eid)
            st = storage_status(entry, self._models_root, self._policy.mode(eid))
            engines[eid] = {
                "repo": entry.repo_label,
                "installed": st.installed,
                "total_bytes": entry.total_bytes,
                "default_selected": entry.default_selected,
                "risk_level": entry.risk_level,
                "risk_text_key": entry.risk_text_key,
                "min_vram_mb": entry.min_vram_mb,
                "storage": st.as_dict(),
            }
        return {
            "engines": engines,
            "active": self._active_engine,
            "running": running,
            "source": self._source,
        }

    def set_mode(self, engine_id: str, mode: str) -> dict:
        """D25 three-mode switch (policy write only; files move via cast/cleanup)."""
        if self._task is not None and not self._task.done():
            raise RuntimeError("a job is already running")
        self._manifest.get(engine_id)  # KeyError on unknown id -> caller maps to err
        self._policy.set_mode(engine_id, mode)
        entry = self._manifest.get(engine_id)
        st = storage_status(entry, self._models_root, self._policy.mode(engine_id))
        return {"engine_id": engine_id, "storage": st.as_dict()}

    # ---- start / cancel -------------------------------------------------

    def start(self, engine_ids: list[str], source: str | None = None) -> dict:
        """Validate + spawn the transfer job. Raises KeyError on unknown id,
        ValueError on an unknown source preference."""
        if self._task is not None and not self._task.done():
            raise RuntimeError("a download job is already running")
        pref = DEFAULT_SOURCE if source is None else source
        templates = templates_for(pref)  # validate before any state moves
        plan: list[tuple[EngineEntry, list[RemoteFile]]] = []
        casts: list[tuple[EngineEntry, str]] = []
        skipped: list[dict] = []
        requested: list[str] = []
        for eid in engine_ids:
            entry = self._manifest.get(eid)
            mode = self._policy.mode(eid)
            st = storage_status(entry, self._models_root, mode)
            missing = missing_files_storage(entry, self._models_root, mode)
            if missing:
                plan.append((entry, missing))
                requested.append(eid)
                if mode in ("dual", "fp16-only"):
                    casts.append((entry, mode))  # chain: originals land -> cast
            elif st.installed:
                skipped.append({"engine_id": eid, "reason": "already_installed"})
            elif st.recast_needed:
                casts.append((entry, mode))  # fp32 present, payloads missing
                requested.append(eid)
            else:
                skipped.append({"engine_id": eid, "reason": "recast_needed"})
        self._source = pref
        if plan or casts:
            self._cancel.clear()
            self._task = asyncio.get_running_loop().create_task(
                self._run(plan, casts, pref, templates)
            )
        return {"started": requested, "skipped": skipped}

    def start_cast(self, engine_id: str) -> dict:
        """Explicit cast/recast action (storage/cast). Raises on bad state."""
        if self._task is not None and not self._task.done():
            raise RuntimeError("a job is already running")
        entry = self._manifest.get(engine_id)
        mode = self._policy.mode(engine_id)
        if mode not in ("dual", "fp16-only"):
            raise ValueError(
                f"{engine_id}: storage mode is {mode}; switch mode before casting"
            )
        st = storage_status(entry, self._models_root, mode)
        if not st.fp32_present and not st.fp16_present:
            raise ValueError(f"{engine_id}: fp32 originals missing; download first")
        self._cancel.clear()
        self._task = asyncio.get_running_loop().create_task(
            self._run([], [(entry, mode)], self._source, templates_for(self._source))
        )
        return {"cast_started": [engine_id], "mode": mode}

    def cleanup(self, engine_id: str) -> dict:
        """Explicit fp16-copy cleanup (storage/cleanup; keep-original repair)."""
        if self._task is not None and not self._task.done():
            raise RuntimeError("a job is already running")
        self._manifest.get(engine_id)
        removed = cleanup_cast(self._models_root, engine_id)
        return {"engine_id": engine_id, "removed": removed}

    def uninstall(self, engine_id: str) -> dict:
        """Remove one catalogue engine's files after the caller unloads it.

        Model directories are user data, so this is manifest-scoped: only
        recorded files, cast payloads, indexes and ``.incomplete`` downloads
        are removed. Files referenced by another catalogue entry are kept.
        """
        if self._task is not None and not self._task.done():
            raise RuntimeError("a job is already running")
        entry = self._manifest.get(engine_id)
        shared_dests = {
            f.dest
            for other in self._manifest.ids()
            if other != engine_id
            for f in self._manifest.get(other).files
        }
        result = uninstall_engine_files(
            entry,
            self._models_root,
            shared_dests=shared_dests,
        )
        self._policy.clear(engine_id)
        return result

    def cancel(self, engine_id: str | None = None) -> None:
        """Cooperative cancel. None cancels the whole job; an id cancels the
        job when that engine is part of it (jobs are the unit, not engines -
        a queued engine has no separately running task to stop)."""
        if engine_id is None or self._active_engine == engine_id:
            self._cancel.set()

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Cancel any running job and wait for the worker to notice."""
        if self._task is None or self._task.done():
            return
        self._cancel.set()
        try:
            await asyncio.wait_for(asyncio.shield(self._task), timeout)
        except (TimeoutError, asyncio.CancelledError):
            pass

    # ---- job runner ------------------------------------------------------

    async def _run(
        self,
        plan: list[tuple[EngineEntry, list[RemoteFile]]],
        casts: list[tuple[EngineEntry, str]],
        source: str = DEFAULT_SOURCE,
        templates: tuple[str, ...] | None = None,
    ) -> None:
        if self._lifecycle_lock is None:
            await self._run_locked(plan, casts, source, templates)
            return
        async with self._lifecycle_lock:
            await self._run_locked(plan, casts, source, templates)

    async def _run_locked(
        self,
        plan: list[tuple[EngineEntry, list[RemoteFile]]],
        casts: list[tuple[EngineEntry, str]],
        source: str = DEFAULT_SOURCE,
        templates: tuple[str, ...] | None = None,
    ) -> None:
        loop = asyncio.get_running_loop()
        try:
            for entry, files in plan:
                self._active_engine = entry.engine_id
                self._emit_nowait(
                    {
                        "kind": "download_engine_start",
                        "engine_id": entry.engine_id,
                        "files": len(files),
                        "source": source,
                    }
                )
                for f in files:

                    def progress_cb(p: Progress) -> None:
                        loop.call_soon_threadsafe(
                            self._emit_nowait,
                            {
                                "kind": "download_progress",
                                "engine_id": p.engine_id,
                                "dest": p.dest,
                                "done_bytes": p.done_bytes,
                                "total_bytes": p.total_bytes,
                            },
                        )

                    try:
                        await loop.run_in_executor(
                            None,
                            functools.partial(
                                self._fd.download,
                                entry.engine_id,
                                f,
                                self._cancel,
                                progress_cb,
                                templates=templates,
                            ),
                        )
                    except DownloadCancelled:
                        self._emit_nowait({"kind": "download_cancelled", "engine_id": entry.engine_id})
                        return
                    except DownloadError as e:
                        self._emit_nowait(
                            {
                                "kind": "download_error",
                                "engine_id": entry.engine_id,
                                "dest": f.dest,
                                "error": str(e),
                            }
                        )
                        return
                    except Exception as e:  # noqa: BLE001 - terminal event is mandatory
                        self._emit_nowait(
                            {
                                "kind": "download_error",
                                "engine_id": entry.engine_id,
                                "dest": f.dest,
                                "error": f"{type(e).__name__}: {e}",
                            }
                        )
                        return
                    self._emit_nowait(
                        {"kind": "download_file_done", "engine_id": entry.engine_id, "dest": f.dest}
                    )
                self._emit_nowait({"kind": "download_engine_done", "engine_id": entry.engine_id})

            for entry, mode in casts:
                self._active_engine = entry.engine_id
                self._emit_nowait(
                    {"kind": "cast_start", "engine_id": entry.engine_id, "mode": mode}
                )

                def cast_progress_cb(p: Progress) -> None:
                    loop.call_soon_threadsafe(
                        self._emit_nowait,
                        {
                            "kind": "cast_progress",
                            "engine_id": p.engine_id,
                            "dest": p.dest,
                            "done_bytes": p.done_bytes,
                            "total_bytes": p.total_bytes,
                        },
                    )

                try:
                    await loop.run_in_executor(
                        None,
                        functools.partial(
                            cast_engine_files,
                            entry,
                            self._models_root,
                            mode,
                            self._cancel,
                            cast_progress_cb,
                        ),
                    )
                except KeyboardInterrupt:
                    # cooperative cancel inside the cast loop; intermediates
                    # (fp16 .incomplete / .fp16 files) are kept - a rerun is
                    # idempotent (fp16->fp16 recast is lossless)
                    self._emit_nowait({"kind": "cast_cancelled", "engine_id": entry.engine_id})
                    return
                except Exception as e:  # noqa: BLE001 - report, keep job discipline
                    self._emit_nowait(
                        {
                            "kind": "cast_error",
                            "engine_id": entry.engine_id,
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )
                    return
                self._emit_nowait(
                    {"kind": "cast_done", "engine_id": entry.engine_id, "mode": mode}
                )
        finally:
            self._active_engine = None

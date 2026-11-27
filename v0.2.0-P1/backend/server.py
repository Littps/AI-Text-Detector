"""P1 sidecar: stdout-guided WS server with torch health + engine layer.

stdout = handshake JSON only ({"port":N,"status":"ready"}), never logs.
stderr = human logs. WS = data channel with token + Origin check.

P1 changes over P0:
- health probes torch lazily (works with or without torch installed)
- "engines" reports the catalogue + LRU residency
- "detect" answers a batch {text, engines[]} with [offset,length,score]
  segments only - the raw text never appears in any response

P2 additions:
- health carries the D22 hardware block (two-source GPU probe + tier)
- download/start|cancel|status drive the five-rule downloader; progress
  is pushed as "event" envelopes to the requesting connection

P3-a additions (D25 storage domain):
- download/status carries a per-engine storage block (mode-aware
  installed / recast_needed / stale_cast / dtype_options)
- storage/set_mode|cast|cleanup drive the three-mode model management;
  cast events ride the same "event" envelopes (cast_start/progress/
  done/error/cancelled)

D36 (source preference): download/start accepts
"source": "official" | "mirror" and download/status echoes the
preference in effect; the winner is reported on download_engine_start.

D40 (file import): file/import parses docx/pdf/txt/md into a session
table and answers metadata only; file/drop removes one; file/list
re-reads it. detect accepts "attachments": [id] next to "text" and
composes text-first, and the reply carries an "input" block
(char count + attachment names) so the results header can name the
sources. Parsing lives in filereader.py, storage in attachments.py.

Round-19 (reload resilience): every detect run is registered in a
DetectJob. "detect/status" answers what the last run is doing - a
reloaded page owns no request, so this is the only way to tell - and
"detect/cancel" asks for a cooperative stop between batches. A socket
that dies mid-run cancels the job from a close watcher, because the
message loop itself is parked in to_thread for the whole run.

Round-20 (graceful takeover): a dead socket no longer cancels the run on the
spot. A reload kills the old socket before the new page has even fetched its
bundle, so an on-the-spot cancel threw away every run a reload interrupted.
The close now schedules a reaper that waits ``DETECT_GRACE`` seconds: a page
that reconnects in that window inherits the run (``detect/status`` shows it
running, the orphan strip can cancel it) and its answer is parked in
``DetectJob.stash_result`` until ``detect/result`` reads it. Nobody takes over
-> the reaper cancels with reason "disconnect", same as before.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import platform
import random
import secrets
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import protocol
import runtimeenv
from attachments import (
    MAX_ITEMS as MAX_ATTACHMENTS,
)
from attachments import (
    MAX_TOTAL_CHARS,
    AttachmentStore,
    UnknownAttachment,
)
from custom_repo import CustomRepo
from detectjob import CANCELLED, DONE, ERROR, DetectCancelled, DetectJob
from download_manager import DownloadManager
from engines import EngineLoader
from hardware import collect
from logstore import LogStore, default_root
from manifest import Manifest
from storage import StoragePolicy
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

VERSION_ROOT = Path(__file__).resolve().parent.parent
TOKEN = os.environ.get("SIDE_TOKEN", "")


def _allowed_origins(extra: str = "") -> set[str]:
    """Production origins plus dev-phase extras (SIDE_EXTRA_ORIGINS).

    Tauri dev mode serves the UI from http://localhost:1420 which is not
    a production origin; the launcher injects it via the env var so the
    shipped default set stays untouched.
    """
    base = {"tauri://localhost", "http://tauri.localhost"}
    return base | {o.strip() for o in extra.split(",") if o.strip()}


ALLOWED_ORIGINS = _allowed_origins(os.environ.get("SIDE_EXTRA_ORIGINS", ""))


# round-30: stderr stays the live channel, the LogStore adds the history the
# "导出日志" dev option needs (today / last 3 / last 7 days survive restarts).
LOGS = LogStore(default_root(VERSION_ROOT))


def log(msg: str) -> None:
    print(f"[sidecar] {msg}", file=sys.stderr, flush=True)
    LOGS.write(msg)


def authorized(ws) -> bool:
    req = getattr(ws, "request", None)
    if req is None:
        return False
    origin = req.headers.get("Origin", "")
    if origin not in ALLOWED_ORIGINS:
        log(f"reject origin: {origin!r}")
        return False
    qs = parse_qs(urlparse(req.path).query)
    supplied = (qs.get("token") or [""])[0]
    if not TOKEN or not secrets.compare_digest(supplied, TOKEN):
        log("reject: bad/missing token")
        return False
    return True


# ---- torch probe (cached; works without torch installed) --------------

_torch_info: dict | None = None


def torch_info(force: bool = False) -> dict:
    global _torch_info
    if force or _torch_info is None:
        try:
            import torch

            info = {
                "torch_installed": True,
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                # wheel build variant: "12.9" for cu129, None for cpu-only
                "cuda_build": torch.version.cuda,
            }
            if info["cuda_available"]:
                props = torch.cuda.get_device_properties(0)
                info["device_name"] = torch.cuda.get_device_name(0)
                info["vram_total_mb"] = round(props.total_memory / (1024 * 1024))
        except Exception as e:  # noqa: BLE001 - probe must never crash the server
            info = {"torch_installed": False, "detail": f"{type(e).__name__}: {e}"}
        _torch_info = info
    return _torch_info


def dev_env_info() -> dict:
    """Dev-facing environment block (round-6 user request: debug info).

    Sensitive values are never echoed: SIDE_TOKEN reports presence only.
    """
    return {
        "force_tier": os.environ.get("SIDE_FORCE_TIER", "").strip().lower() or None,
        "token_set": bool(TOKEN),
        "extra_origins": os.environ.get("SIDE_EXTRA_ORIGINS", "") or None,
        "models_dir": str(MODELS_ROOT),
        "logs_dir": str(LOGS.root),
    }


# ---- engine layer ------------------------------------------------------

MODELS_ROOT = VERSION_ROOT / "models"
MANIFEST = Manifest.load(MODELS_ROOT / "manifest.json")
STORAGE_POLICY = StoragePolicy.load(MODELS_ROOT / "storage.json")
LOADER = EngineLoader(
    policy=STORAGE_POLICY,
    manifest=MANIFEST,
    models_root=MODELS_ROOT,
)

# ---- download layer (P2 / P3-a storage domain) ---------------------------

DOWNLOADS = DownloadManager(MANIFEST, MODELS_ROOT, policy=STORAGE_POLICY)

# Custom model library (round-6): local import + URL download. The emit
# callback resolves _emit_job_event lazily (defined later in this module) and
# hops to the event loop thread (downloads run in worker threads).
_MAIN_LOOP: list[asyncio.AbstractEventLoop] = []


def _custom_emit(payload: dict) -> None:
    loop = _MAIN_LOOP[0] if _MAIN_LOOP else None
    if loop is not None:
        loop.call_soon_threadsafe(_emit_job_event, payload)


CUSTOMS = CustomRepo(MODELS_ROOT, MANIFEST, _custom_emit)

# D40 composer attachments: parsed once at import, kept here so a detect frame
# carries ids instead of megabytes of text.
ATTACHMENTS = AttachmentStore()

# D22 hardware profile: probed once at boot, cached in memory.
HW = collect(MODELS_ROOT)

# Residency policy (2026-11-21): the budget follows the probed card instead of
# a hard-coded 6GB, and the retention tier decides how long a model may stay
# in VRAM after a detection. The UI re-applies its stored tier on connect.
LOADER.set_total_vram(HW.get("vram_total_mb"))
try:
    LOADER.set_retention(os.environ.get("SIDE_RETENTION", "").strip().lower() or "default")
except ValueError as e:
    log(f"ignored SIDE_RETENTION: {e}")


def vram_block() -> dict:
    """Measured residency numbers shared by engines / detect / events."""
    policy = LOADER.policy()
    return {
        "resident": LOADER.resident_ids(),
        "resident_detail": LOADER.resident_detail(),
        "vram_used_mb": LOADER.used_mb(),
        "vram_budget_mb": policy["budget_mb"],
        "vram_total_mb": HW.get("vram_total_mb"),
        "vram_measured": LOADER.measured(),
        "retention": policy["retention"],
        "retention_ttl_s": policy["ttl_s"],
    }


# One GPU, one owner: detect runs hold this lock, the idle sweeper takes it
# too, so a retention tick can never unload under a running inference.
DETECT_LOCK = asyncio.Lock()
DOWNLOADS.set_lifecycle_lock(DETECT_LOCK)

# Round-19: the run's public state (who owns it, is it still wanted). Lives
# outside the WS handler so a reload can ask "what were you doing?" instead of
# silently inheriting an orphaned GPU job.
DETECT = DetectJob()


def handle_custom_import(msg_id: str, payload: dict) -> str:
    source_dir = payload.get("source_dir")
    mode = payload.get("mode", "link")
    name = payload.get("name")
    if not isinstance(source_dir, str) or not source_dir.strip():
        return protocol.err(msg_id, "source_dir must be a non-empty string")
    if mode not in ("link", "copy"):
        return protocol.err(msg_id, "mode must be 'link' or 'copy'")
    if name is not None and (not isinstance(name, str) or not name.strip()):
        return protocol.err(msg_id, "name must be a non-empty string when present")
    try:
        result = CUSTOMS.import_local(source_dir.strip(), mode, name)
    except (OSError, ValueError) as e:
        return protocol.err(msg_id, str(e))
    # when the import wired an official engine, attach its fresh storage
    # verification so the UI can show whether it is truly usable
    if result.get("kind") == "engine":
        st = DOWNLOADS.status().get("engines", {}).get(result["engine"], {})
        result["verify"] = {
            k: st.get(k)
            for k in ("installed", "recast_needed", "stale_cast", "mode")
            if k in st
        }
    return protocol.ok(msg_id, result)


def handle_custom_download(msg_id: str, payload: dict) -> str:
    url = payload.get("url")
    mirror = payload.get("mirror")
    name = payload.get("name")
    if not isinstance(url, str):
        return protocol.err(msg_id, "url must be a string")
    if mirror not in ("mirror", "direct"):
        return protocol.err(msg_id, "mirror must be 'mirror' or 'direct'")
    if name is not None and (not isinstance(name, str) or not name.strip()):
        return protocol.err(msg_id, "name must be a non-empty string when present")
    try:
        job = CUSTOMS.start_url_download(url, mirror, name)
    except ValueError as e:
        return protocol.err(msg_id, str(e))
    return protocol.ok(msg_id, job)


def handle_detect(msg_id: str, payload: dict, emit=None) -> str:
    # D40: the composer can carry pasted text, imported attachments, or both.
    # Text goes first, attachments follow in the order the UI listed them; the
    # ids resolve against this session's table, so the payload stays small even
    # for a 300-page PDF.
    text = payload.get("text", "")
    engine_ids = payload.get("engines")
    att_ids = payload.get("attachments") or []
    if not isinstance(text, str):
        return protocol.err(msg_id, "text must be a string")
    if not isinstance(att_ids, list) or not all(isinstance(i, str) for i in att_ids):
        return protocol.err(msg_id, "attachments must be a list of ids")
    if (
        not isinstance(engine_ids, list)
        or not engine_ids
        or not all(isinstance(i, str) for i in engine_ids)
    ):
        return protocol.err(msg_id, "engines must be a non-empty list of ids")
    engine_ids = list(dict.fromkeys(engine_ids))
    if len(att_ids) > MAX_ATTACHMENTS or len(set(att_ids)) != len(att_ids):
        return protocol.err(msg_id, "attachments must be unique and within the item limit")
    try:
        composed, source_names = ATTACHMENTS.compose(text, att_ids)
    except UnknownAttachment as e:
        return protocol.err(msg_id, f"unknown attachment: {e.args[0]}")
    if not composed.strip():
        return protocol.err(msg_id, "text or attachments must be non-empty")
    if len(composed) > MAX_TOTAL_CHARS:
        return protocol.err(
            msg_id,
            f"composed input exceeds the {MAX_TOTAL_CHARS:,} character limit",
        )

    # D29 streaming: per-batch partials hop to the event loop as
    # detect_part events. No terminal frame - the reply itself carries
    # every engine's full segments + doc_score + status, so a terminal
    # event would only race the reply.
    #
    # Round-19: the run is registered before the first engine loads, so a
    # reload has something to query and a second request is refused instead
    # of queueing invisibly behind the GPU lock.
    job_id = DETECT.begin(engine_ids, chars=len(composed))
    if job_id is None:
        return protocol.err(msg_id, "a detect run is already in progress")

    def on_batch(engine_id: str, segments: list) -> None:
        # cooperative stop between sentence batches: raising here unwinds the
        # engine before its next forward pass (a running one cannot be cut)
        DETECT.check()
        if emit is not None:
            emit(
                {
                    "kind": "detect_part",
                    "engine_id": engine_id,
                    "segments": [s.as_dict() for s in segments],
                }
            )

    results = []
    # Language-aware eviction: within one run every engine's language is
    # "keep", so the LRU prefers unloading an unrelated-language leftover.
    langs = LOADER.langs_of(engine_ids)
    try:
        for sid in engine_ids:
            # ... and between engines: document-level engines report once at
            # the very end, so this is their only cancellation point
            DETECT.check()
            DETECT.engine_started(sid)
            r = LOADER.detect_one(
                composed, sid, on_batch=on_batch if emit is not None else None, keep_langs=langs
            )
            results.append(r.as_dict())
            if r.status == "cancelled":
                raise DetectCancelled(sid)
    except DetectCancelled:
        DETECT.finish(CANCELLED)
        log("detect cancelled before completion (client left or asked to stop)")
        return protocol.ok(msg_id, {"cancelled": True, "job": DETECT.snapshot()})
    except BaseException as e:
        DETECT.finish(ERROR, detail=f"{type(e).__name__}: {e}")
        raise
    final_state = DETECT.finish(DONE)
    if final_state == CANCELLED:
        log("detect cancelled at the finish boundary; discarding late results")
        return protocol.ok(msg_id, {"cancelled": True, "job": DETECT.snapshot()})
    if LOADER.retention == "none":
        # 不驻留: release right after the batch (sweeping at TTL 0 would
        # unload between two engines of this very run)
        LOADER.unload(LOADER.resident_ids())
    return protocol.ok(
        msg_id,
        {
            "job_id": job_id,
            "results": results,
            # what the detector actually saw: char count of the composed text
            # plus the attachment names, so the results header can say
            # "文字 320 字 + 报告.docx 12,480 字".
            #
            # "text" is included ONLY when attachments were composed: the UI
            # already owns the pasted text, but the file bodies live here, and
            # the per-sentence list / highlight view need the real sentences.
            # This is the one exception to "raw text never travels back".
            "input": {
                "chars": len(composed),
                "attachments": source_names,
                **({"text": composed} if att_ids else {}),
            },
            **vram_block(),
        },
    )


def handle_detect_status(msg_id: str, payload: dict) -> str:
    """Round-19: what is the last (or current) detect run doing?

    A reloaded page owns no request any more, so this is the only way to know
    whether the GPU is still busy and why. Text never appears here.
    """
    return protocol.ok(msg_id, {"job": DETECT.snapshot()})


def handle_detect_cancel(msg_id: str, payload: dict) -> str:
    """Ask the running job to stop at its next batch boundary."""
    requested = DETECT.cancel("client")
    if requested:
        log("detect cancel requested by client")
    return protocol.ok(msg_id, {"cancel_requested": requested, "job": DETECT.snapshot()})


def handle_detect_result(msg_id: str, payload: dict) -> str:
    """Hand back a reply parked because its socket died (round-20).

    One reader wins: the answer is consumed, so a second tab can never render
    a run it does not own. An error reply keeps its ``{"error": ...}`` shape
    and the client rejects it exactly like a direct failure.
    """
    expected_job_id = payload.get("job_id")
    if expected_job_id is not None and (
        not isinstance(expected_job_id, int) or isinstance(expected_job_id, bool)
    ):
        return protocol.err(msg_id, "job_id must be an integer when present")
    parked = DETECT.take_result(expected_job_id)
    if parked is None:
        return protocol.err(msg_id, "no detect result is waiting")
    reply, text = parked
    try:
        data = json.loads(reply).get("payload")
    except (json.JSONDecodeError, TypeError, AttributeError):
        return protocol.err(msg_id, "parked detect result is unreadable")
    if not isinstance(data, dict):
        return protocol.err(msg_id, "parked detect result is unreadable")
    if text:
        # a pure-text reply carries an input block with counts but no body;
        # fill the body in from the parked text so the highlight view still
        # has sentences to draw. Attachment replies already ship their own.
        block = data.get("input")
        block = block if isinstance(block, dict) else {}
        if not block.get("text"):
            data["input"] = {**block, "text": text}
    return protocol.ok(msg_id, data)


def handle_file_import(msg_id: str, payload: dict) -> str:
    """D40: parse documents into the session table.

    Two intake shapes, one parser:
      {"paths": [...]}  Tauri — the dialog hands back absolute paths
      {"blobs": [...]}  browser fallback — [{name, data(base64)}], because a
                        sandboxed webview cannot expose a real path
    """
    paths = payload.get("paths")
    blobs = payload.get("blobs")
    if paths is not None and blobs is not None:
        return protocol.err(msg_id, "send paths or blobs, not both")
    if paths is None and blobs is None:
        return protocol.err(msg_id, "paths or blobs required")
    try:
        if paths is not None:
            if (
                not isinstance(paths, list)
                or not paths
                or not all(isinstance(p, str) for p in paths)
            ):
                return protocol.err(msg_id, "paths must be a non-empty list of strings")
            result = ATTACHMENTS.import_paths(paths)
        else:
            if not isinstance(blobs, list) or not blobs:
                return protocol.err(msg_id, "blobs must be a non-empty list")
            if not all(isinstance(b, dict) for b in blobs):
                return protocol.err(msg_id, "every blob must be an object")
            result = ATTACHMENTS.import_blobs(blobs)
    except ValueError as e:
        return protocol.err(msg_id, str(e))
    return protocol.ok(msg_id, {**result, "all": ATTACHMENTS.meta()})


def handle_file_drop(msg_id: str, payload: dict) -> str:
    aid = payload.get("id")
    if not isinstance(aid, str) or not aid:
        return protocol.err(msg_id, "id must be a non-empty string")
    return protocol.ok(msg_id, {"dropped": ATTACHMENTS.drop(aid), "all": ATTACHMENTS.meta()})


def handle_file_list(msg_id: str, payload: dict) -> str:
    return protocol.ok(msg_id, {"items": ATTACHMENTS.meta()})


def handle_download_start(msg_id: str, payload: dict) -> str:
    engine_ids = payload.get("engines")
    source = payload.get("source")
    if (
        not isinstance(engine_ids, list)
        or not engine_ids
        or not all(isinstance(i, str) for i in engine_ids)
    ):
        return protocol.err(msg_id, "engines must be a non-empty list of ids")
    if source is not None and not isinstance(source, str):
        return protocol.err(msg_id, "source must be a string when present")
    try:
        result = DOWNLOADS.start(engine_ids, source)
    except KeyError as e:
        return protocol.err(msg_id, f"unknown engine: {e}")
    except (RuntimeError, ValueError) as e:
        return protocol.err(msg_id, str(e))
    # A prior run may have landed files while the sidecar was booting. Refresh
    # skipped/already-installed ids now; started ids are refreshed by the
    # terminal job event.
    for skipped in result.get("skipped", []):
        engine_id = skipped.get("engine_id")
        if isinstance(engine_id, str):
            try:
                LOADER.refresh_engine(engine_id)
            except Exception as e:  # noqa: BLE001 - status reply must survive
                log(f"engine refresh after downloaded skip failed: {e!r}")
    # forward job events to this connection (loop thread; send failures
    # after a disconnect are swallowed - the job itself is cancelled below)
    DOWNLOADS.attach(_emit_job_event)
    return protocol.ok(msg_id, result)


def _emit_job_event(payload: dict) -> None:
    # A first install starts with StubEngine instances. Once a manifest engine
    # has landed (and after a cast finalizes its load knobs), swap it in on
    # the loop thread so the next detect does not require a sidecar restart.
    if payload.get("kind") in {"download_engine_done", "cast_done"}:
        engine_id = payload.get("engine_id")
        if isinstance(engine_id, str):
            try:
                LOADER.refresh_engine(engine_id)
            except Exception as e:  # noqa: BLE001 - status event must survive
                log(f"engine refresh after {payload['kind']} failed: {e!r}")
    # keep a reference so the GC cannot collect a not-yet-run task
    t = asyncio.create_task(_safe_send(payload))
    _EVENT_TASKS.add(t)
    t.add_done_callback(_EVENT_TASKS.discard)


_EVENT_TASKS: set[asyncio.Task[None]] = set()


async def _safe_send(payload: dict) -> None:
    ws = ACTIVE_WS
    if ws is None:
        return
    # events use a fixed id so the client can correlate without a
    # request pending on the wire; a client gone mid-download is fine -
    # the handler's finally block cancels the job
    with contextlib.suppress(Exception):
        await ws.send(protocol.envelope("dl-event", "event", payload))


# the single connection that owns the current download job (P2 keeps one
# client at a time; the handshake/UI model is single-window)
ACTIVE_WS: Any | None = None  # set in handler()


def handle_download_cancel(msg_id: str, payload: dict) -> str:
    engine_id = payload.get("engine_id")
    if engine_id is not None and not isinstance(engine_id, str):
        return protocol.err(msg_id, "engine_id must be a string when present")
    DOWNLOADS.cancel(engine_id)
    return protocol.ok(msg_id, {"cancel_requested": True})


# ---- residency domain (2026-11-21 retention rework) ----------------------


def handle_resident_policy(msg_id: str, payload: dict) -> str:
    tier = payload.get("retention")
    if not isinstance(tier, str):
        return protocol.err(msg_id, "retention must be a string")
    try:
        LOADER.set_retention(tier)
    except ValueError as e:
        return protocol.err(msg_id, str(e))
    return protocol.ok(msg_id, {"policy": LOADER.policy(), **vram_block()})


async def handle_resident_unload(msg_id: str, payload: dict) -> str:
    engine_id = payload.get("engine_id")
    if engine_id is not None and (not isinstance(engine_id, str) or not engine_id.strip()):
        return protocol.err(msg_id, "engine_id must be a non-empty string when present")
    async with DETECT_LOCK:
        ids = [engine_id] if isinstance(engine_id, str) else LOADER.resident_ids()
        results = [r.as_dict() for r in await asyncio.to_thread(LOADER.unload, ids)]
    log(f"manual unload: {', '.join(ids) or '(nothing resident)'}")
    return protocol.ok(msg_id, {"unloaded": ids, "results": results, **vram_block()})


# ---- log export domain (round-30 dev option) ----------------------------


def _log_export_meta() -> dict:
    """Environment block for the export header. Never carries the token."""
    ti = torch_info()
    return {
        "app": os.environ.get("AIGC_APP_VERSION", "0.2.0-P1"),
        "python": platform.python_version(),
        "torch": ti.get("torch_version") or "not installed",
        "cuda": ti.get("cuda_build") or "-",
        "tier": HW.get("tier"),
        "platform": platform.platform(),
        "models_dir": str(MODELS_ROOT),
    }


def handle_logs_export(msg_id: str, payload: dict) -> str:
    days = payload.get("days")
    # bool is an int subclass: reject it explicitly so true/false cannot slip in
    if not isinstance(days, int) or isinstance(days, bool) or days not in (1, 3, 7):
        return protocol.err(msg_id, "days must be 1, 3 or 7")
    try:
        result = LOGS.export(days, meta=_log_export_meta())
    except ValueError as e:
        return protocol.err(msg_id, str(e))
    log(f"log export: {result['days']}d -> {result['files']} file(s), {result['bytes']} bytes")
    return protocol.ok(msg_id, result)


# ---- storage domain (P3-a, D25 three-mode model management) --------------


async def handle_storage_set_mode(msg_id: str, payload: dict) -> str:
    engine_id = payload.get("engine_id")
    mode = payload.get("mode")
    if not isinstance(engine_id, str) or not engine_id:
        return protocol.err(msg_id, "engine_id must be a non-empty string")
    if not isinstance(mode, str):
        return protocol.err(msg_id, "mode must be a string")
    try:
        async with DETECT_LOCK:
            result = await asyncio.to_thread(DOWNLOADS.set_mode, engine_id, mode)
    except KeyError as e:
        return protocol.err(msg_id, f"unknown engine: {e}")
    except (OSError, RuntimeError, ValueError) as e:
        return protocol.err(msg_id, str(e))
    return protocol.ok(msg_id, result)


def handle_storage_cast(msg_id: str, payload: dict) -> str:
    engine_id = payload.get("engine_id")
    if not isinstance(engine_id, str) or not engine_id:
        return protocol.err(msg_id, "engine_id must be a non-empty string")
    try:
        result = DOWNLOADS.start_cast(engine_id)
    except KeyError as e:
        return protocol.err(msg_id, f"unknown engine: {e}")
    except (ValueError, RuntimeError) as e:
        return protocol.err(msg_id, str(e))
    DOWNLOADS.attach(_emit_job_event)
    return protocol.ok(msg_id, result)


async def handle_storage_cleanup(msg_id: str, payload: dict) -> str:
    engine_id = payload.get("engine_id")
    if not isinstance(engine_id, str) or not engine_id:
        return protocol.err(msg_id, "engine_id must be a non-empty string")
    try:
        async with DETECT_LOCK:
            result = await asyncio.to_thread(DOWNLOADS.cleanup, engine_id)
    except KeyError as e:
        return protocol.err(msg_id, f"unknown engine: {e}")
    except (OSError, RuntimeError, ValueError) as e:
        return protocol.err(msg_id, str(e))
    return protocol.ok(msg_id, result)


async def handle_storage_uninstall(msg_id: str, payload: dict) -> str:
    """Remove a model only after its GPU allocation and all I/O jobs are idle."""
    engine_id = payload.get("engine_id")
    if not isinstance(engine_id, str) or not engine_id:
        return protocol.err(msg_id, "engine_id must be a non-empty string")
    if engine_id not in MANIFEST:
        return protocol.err(msg_id, f"unknown engine: {engine_id}")
    if DETECT.is_running() or any(not task.done() for task in _DETECT_TASKS):
        return protocol.err(msg_id, "cannot uninstall while a detect run is in progress")
    async with DETECT_LOCK:
        # Re-check after acquiring the shared lifecycle lock: a download task
        # may have been queued after the cheap preliminary status read.
        if DOWNLOADS.status()["running"]:
            return protocol.err(msg_id, "cannot uninstall while a download/cast job is running")
        try:
            if engine_id in LOADER.resident_ids():
                await asyncio.to_thread(LOADER.unload, [engine_id])
            result = await asyncio.to_thread(DOWNLOADS.uninstall, engine_id)
        except KeyError as e:
            return protocol.err(msg_id, f"unknown engine: {e}")
        except (OSError, RuntimeError, ValueError) as e:
            return protocol.err(msg_id, str(e))
    log(
        f"model uninstalled: {engine_id} "
        f"({result['bytes_freed']} bytes, {len(result['removed'])} files)"
    )
    return protocol.ok(msg_id, result)


def handle_runtime_status(msg_id: str, payload: dict) -> str:
    """Describe the live interpreter + native dependency chain."""
    force = payload.get("refresh") is True
    torch_state = torch_info(force=True) if force else torch_info()
    return protocol.ok(
        msg_id,
        # The diagnostics refresh button asks for a fresh torch probe; normal
        # status refreshes stay on the cached import to avoid a multi-second
        # import on every download/cast event.
        runtimeenv.collect(torch_info=torch_state, hardware=HW),
    )


def _detect_emitter_factory() -> Callable[[dict], None]:
    """Thread-safe detect_part emitter bound to the running event loop.

    Module-level on purpose: the WS handler runs inside ``async for`` and
    closures defined there trip B023. Partials funnel through the same
    event sink as download events (``_emit_job_event``).
    """
    loop = asyncio.get_running_loop()

    def emit(part: dict) -> None:
        loop.call_soon_threadsafe(_emit_job_event, part)

    return emit


# A detect runs as a task, never inline in the read loop (round-19).
# Inline it looked harmless - inference is off-loop in ``to_thread`` - but the
# handler was suspended for the whole run, so the next frame (``detect/cancel``,
# a ping, a health probe) only got read *after* the run it was meant to stop.
# The "run off-loop so the UI stays responsive" comment above was aspirational
# until this change.
_DETECT_TASKS: set[asyncio.Task[None]] = set()

# Round-20: how long a run outlives the socket that asked for it. A reload
# needs a few hundred ms to fetch the bundle and open the new socket; three
# seconds absorbs a cold reload without letting a closed window burn the GPU.
# The env escape hatch exists so tests do not have to sleep for real.
DETECT_GRACE = float(os.environ.get("AIGC_DETECT_GRACE", "3.0"))
_REAP_TASKS: set[asyncio.Task[None]] = set()


async def _reap_orphan(job_id: int) -> None:
    """Cancel a run nobody took over, after the grace window.

    Runs from the dead connection's ``finally``. The check is deliberately
    late: a reload opens its new socket *after* the old one closes, so acting
    on the close itself is what discarded every interrupted run before.
    """
    await asyncio.sleep(DETECT_GRACE)
    if ACTIVE_WS is not None:
        return  # a page took over; the job lives on and can be cancelled there
    if DETECT.job_id() != job_id:
        return  # a newer run owns the slot, leave it alone
    if DETECT.cancel("disconnect"):
        log("detect cancelled: no page took over within the grace window")


def _schedule_orphan_reap(job_id: int) -> None:
    """Keep a reference so the reaper cannot be collected before it runs."""
    t = asyncio.create_task(_reap_orphan(job_id))
    _REAP_TASKS.add(t)
    t.add_done_callback(_REAP_TASKS.discard)


def _parkable(reply: str) -> bool:
    """True when a reply is worth keeping for a page that has not connected.

    Only a real result qualifies: ``detect/status`` already says *why* a run
    ended (cancelled / error + detail), and parking those would make the next
    page render a "result" that is really a tombstone.
    """
    try:
        data = json.loads(reply).get("payload")
    except (json.JSONDecodeError, TypeError, AttributeError):
        return False
    return isinstance(data, dict) and bool(data.get("results"))


async def _run_detect(ws, msg_id: str, payload: dict) -> None:
    async with DETECT_LOCK:
        if not _socket_open(ws):
            # the reload landed before the worker picked the job up: do not
            # claim the GPU for an answer nobody can read
            log("detect skipped: client left before the run started")
            return
        try:
            reply = await asyncio.to_thread(
                handle_detect, msg_id, payload, _detect_emitter_factory()
            )
        except Exception as e:  # noqa: BLE001 - a worker crash must reach the client
            log(f"detect worker crashed: {e!r}")
            reply = protocol.err(msg_id, f"detect failed: {e}")
    # The socket may have died while the GPU was busy. Round-20: the run is
    # allowed to survive that, so its answer must too - park it and let the
    # page that takes over read it with ``detect/result``.
    delivered = False
    try:
        await ws.send(reply)
        delivered = True
    except Exception as e:  # noqa: BLE001 - a dead socket must not lose the answer
        log(f"detect reply not delivered: {e!r}")
    if not delivered:
        if _parkable(reply):
            # pure-text runs never echo their input, so a page that reloaded
            # has nothing to draw sentences from: park the pasted text with
            # the reply (D40 exception #2; wiped on read / next run / TTL)
            DETECT.stash_result(
                reply,
                payload.get("text") or None,
                DETECT.job_id(),
            )
            log("detect reply parked for the next connection")
        else:
            log("detect reply dropped: nothing worth handing over")


# ---- WS handler --------------------------------------------------------

def _socket_open(sock: Any) -> bool:
    """True while a socket can still receive a reply.

    Used by the detect worker: a reload can land between queueing the job and
    picking it up, and burning the GPU for a page that is already gone helps
    nobody. websockets >=14 exposes ``state``; older shapes fall back to
    ``close_code``.
    """
    state = getattr(sock, "state", None)
    name = getattr(state, "name", None)
    if isinstance(name, str):
        return name == "OPEN"
    return getattr(sock, "close_code", None) is None


async def _claim_slot(ws: Any) -> None:
    """Hand the single-client slot to the newest authorized socket.

    Round-28: this used to refuse a new socket while ``ACTIVE_WS`` was still
    OPEN. A leaked connection (HMR re-running the connect effect, a crashed
    page whose socket never sent FIN) then held the slot forever: every retry
    got 4002, the UI showed "closed", and hundreds of sockets piled up in
    TIME_WAIT. The shell enforces one window (D46), so a fresh authorized
    connection is a reload, not a second client. Closing the old socket lets
    its ``finally`` see ``ACTIVE_WS is not ws`` and leave an inherited detect
    job alone (round-20 takeover).
    """
    global ACTIVE_WS
    old = ACTIVE_WS
    if old is ws:
        return
    ACTIVE_WS = ws
    if old is not None:
        log(
            "client takeover: "
            f"{getattr(old, 'remote_address', None)} -> {getattr(ws, 'remote_address', None)}"
        )
        with contextlib.suppress(Exception):
            await old.close(4003, "replaced by newer connection")


async def handler(ws) -> None:
    global ACTIVE_WS
    if not authorized(ws):
        await ws.close(4001, "unauthorized")
        return
    await _claim_slot(ws)
    peer = ws.remote_address
    log(f"client connected: {peer}")
    # this connection's detect task; the read loop keeps running beside it so
    # detect/cancel and the health probes are answered mid-run
    detect_task: asyncio.Task[None] | None = None

    try:
        async for raw in ws:
            try:
                msg_id, mtype, payload = protocol.parse(raw)
            except (ValueError, TypeError) as e:
                await ws.send(protocol.err("m-err", str(e)))
                continue

            if mtype == "ping":
                await ws.send(
                    protocol.envelope(msg_id, "pong", {"echo": payload.get("ts")})
                )
            elif mtype == "health":
                await ws.send(
                    protocol.ok(
                        msg_id,
                        {
                            "status": "ready",
                            "python_version": platform.python_version(),
                            "platform": platform.platform(),
                            "hw": HW,
                            **torch_info(),
                            "env": dev_env_info(),
                        },
                    )
                )
            elif mtype == "engines":
                await ws.send(
                    protocol.ok(
                        msg_id,
                        {
                            "engines": LOADER.status(),
                            **vram_block(),
                        },
                    )
                )
            elif mtype == "detect":
                # Background task: the read loop must stay free to answer
                # detect/cancel (the stop button) and the health probes while
                # the GPU works. The lock keeps the idle sweeper from
                # unloading an engine in the middle of a run.
                if detect_task is not None and not detect_task.done():
                    await ws.send(protocol.err(msg_id, "a detect run is already in progress"))
                    continue
                detect_task = asyncio.create_task(_run_detect(ws, msg_id, payload))
                _DETECT_TASKS.add(detect_task)
                detect_task.add_done_callback(_DETECT_TASKS.discard)
            elif mtype == "detect/status":
                await ws.send(handle_detect_status(msg_id, payload))
            elif mtype == "detect/result":
                await ws.send(handle_detect_result(msg_id, payload))
            elif mtype == "detect/cancel":
                await ws.send(handle_detect_cancel(msg_id, payload))
            elif mtype == "file/import":
                # parsing a 300-page PDF is not free; keep the loop responsive
                reply = await asyncio.to_thread(handle_file_import, msg_id, payload)
                await ws.send(reply)
            elif mtype == "file/drop":
                await ws.send(handle_file_drop(msg_id, payload))
            elif mtype == "file/list":
                await ws.send(handle_file_list(msg_id, payload))
            elif mtype == "download/status":
                await ws.send(protocol.ok(msg_id, DOWNLOADS.status()))
            elif mtype == "download/start":
                await ws.send(handle_download_start(msg_id, payload))
            elif mtype == "download/cancel":
                await ws.send(handle_download_cancel(msg_id, payload))
            elif mtype == "resident/policy":
                await ws.send(handle_resident_policy(msg_id, payload))
            elif mtype == "resident/unload":
                await ws.send(await handle_resident_unload(msg_id, payload))
            elif mtype == "logs/export":
                await ws.send(handle_logs_export(msg_id, payload))
            elif mtype == "storage/set_mode":
                await ws.send(await handle_storage_set_mode(msg_id, payload))
            elif mtype == "storage/cast":
                await ws.send(handle_storage_cast(msg_id, payload))
            elif mtype == "storage/cleanup":
                await ws.send(await handle_storage_cleanup(msg_id, payload))
            elif mtype == "storage/uninstall":
                await ws.send(await handle_storage_uninstall(msg_id, payload))
            elif mtype == "runtime/status":
                reply = await asyncio.to_thread(handle_runtime_status, msg_id, payload)
                await ws.send(reply)
            elif mtype == "custom/list":
                await ws.send(protocol.ok(msg_id, CUSTOMS.status()))
            elif mtype == "custom/import":
                # link/copy of possibly multi-GB dirs runs in a worker thread
                async with DETECT_LOCK:
                    result = await asyncio.to_thread(handle_custom_import, msg_id, payload)
                    # hot-wire (round-6 user request): when the import matched an
                    # official manifest engine and storage verification passed,
                    # swap the boot-time StubEngine for the real implementation
                    # right here on the loop thread - no sidecar restart. detect
                    # workers keep the old instance reference until they finish,
                    # so the dict swap is race-free by reference semantics.
                    try:
                        pl = json.loads(result).get("payload") or {}
                        verify_ok = (pl.get("verify") or {}).get("installed")
                        if pl.get("kind") == "engine" and verify_ok and LOADER.refresh_engine(pl["engine"]):
                            log(f"engine hot-wired from custom import: {pl['engine']}")
                    except Exception as e:  # noqa: BLE001 - refresh must never kill the reply
                        log(f"engine refresh skipped: {e!r}")
                await ws.send(result)
            elif mtype == "custom/download":
                await ws.send(handle_custom_download(msg_id, payload))
            elif mtype == "custom/cancel":
                await ws.send(protocol.ok(msg_id, {"cancel_requested": CUSTOMS.cancel()}))
            else:
                await ws.send(protocol.err(msg_id, f"unknown type {mtype!r}"))
    except ConnectionClosed:
        # normal end of a page / takeover: the finally block does the cleanup
        pass
    except Exception as e:  # noqa: BLE001 - log and drop connection
        log(f"connection error: {e!r}")
    finally:
        # Round-20: a close no longer cancels on the spot - a reload kills
        # this socket *before* the new page finishes loading, and cancelling
        # here is exactly what threw the interrupted run away. Only the live
        # connection's death matters (a stale close must not touch a job the
        # replacement connection inherited), and the reaper waits out the
        # grace window before deciding nobody took over.
        if ACTIVE_WS is ws:
            DOWNLOADS.detach()
            DOWNLOADS.cancel(None)
            CUSTOMS.cancel()
            ACTIVE_WS = None
            # Not tied to `detect_task`: a page that took over an inherited
            # run owns no request of its own, yet its death still orphans it.
            if DETECT.is_running():
                _schedule_orphan_reap(DETECT.job_id())
        log(f"client disconnected: {peer}")


async def retention_sweeper(stop: asyncio.Event) -> None:
    """Idle-unload loop for the 30m/10m/5m retention tiers.

    Ticks every 30s, never concurrent with a detect run, and pushes a
    ``resident`` event so an open window updates the first-screen numbers
    without polling.
    """
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=30)
            return
        except TimeoutError:
            pass
        if LOADER.retention not in ("30m", "10m", "5m"):
            continue
        if not LOADER.resident_ids():
            continue
        async with DETECT_LOCK:
            released = await asyncio.to_thread(LOADER.sweep_idle)
        if released:
            log(f"retention sweep released: {', '.join(released)}")
            _emit_job_event({"kind": "resident", **vram_block()})


async def amain() -> None:
    last_err = None
    server = None
    _MAIN_LOOP.append(asyncio.get_running_loop())
    for _ in range(3):  # bind retry x3
        port = random.randint(20000, 60000)
        try:
            async with serve(handler, "127.0.0.1", port, max_size=32 * 1024 * 1024) as srv:
                server = srv
                print(json.dumps({"port": port, "status": "ready"}), flush=True)
                log(f"listening on 127.0.0.1:{port}")
                stop = asyncio.Event()
                sweeper = asyncio.create_task(retention_sweeper(stop))
                loop = asyncio.get_running_loop()
                import signal

                for sig in (signal.SIGINT, signal.SIGTERM):
                    # Windows adds no add_signal_handler support for SIGTERM;
                    # fall back silently - KeyboardInterrupt still works.
                    try:
                        loop.add_signal_handler(sig, stop.set)
                    except (NotImplementedError, RuntimeError):
                        pass
                await stop.wait()
                sweeper.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sweeper
                break
        except OSError as e:
            last_err = e
            log(f"bind {port} failed: {e!r}")
    else:
        log(f"giving up after 3 bind attempts: {last_err!r}")
        sys.exit(1)

    # Graceful shutdown: release every resident engine before exit so the
    # GPU is clean for whoever comes next (parent process restart, tests).
    log("shutting down: cancelling downloads, unloading engines")
    try:
        await DOWNLOADS.shutdown(timeout=5)
    except Exception as e:  # noqa: BLE001 - shutdown must never hang
        log(f"download shutdown failed: {e!r}")
    try:
        await asyncio.to_thread(LOADER.unload, LOADER.resident_ids())
    except Exception as e:  # noqa: BLE001 - shutdown must never hang
        log(f"unload during shutdown failed: {e!r}")
    if server is not None:
        server.close()
        await server.wait_closed()
    log("bye")
    LOGS.close()


if __name__ == "__main__":
    asyncio.run(amain())

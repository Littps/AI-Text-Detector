"""Downloader core - the five iron rules (PLAN P2, D18).

1. multi-source fallback: the chain leads with the preferred host and
   auto-switches on 403/5xx/connection failure. Official-first is the
   D18 default; D36 lets the user lead with hf-mirror from settings; the
   preference only reorders, it never drops a host;
2. Range resume: partial data lives in ``<file>.incomplete`` and the
   next attempt continues from its size; a server that ignores Range
   (plain 200) transparently restarts;
3. SHA256 verification before landing: streamed over the completed
   temp file, mismatch -> reject, never touch the final path;
4. disk space precheck before any request goes out;
5. atomic rename: os.replace(.incomplete -> final), crash-safe.

The core is synchronous stdlib (urllib) - the server runs it in a
worker thread; cancellation is a threading.Event checked between
chunks that deliberately keeps the .incomplete file for resume.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from manifest import EngineEntry, RemoteFile, safe_relative_path

CHUNK = 1 << 20  # 1 MiB
PROGRESS_INTERVAL = 0.25  # seconds between progress callbacks
UA = "aigc-detector-downloader/0.2"
INTEGRITY_CACHE_NAME = ".verified-files.json"

OFFICIAL_TEMPLATE = "https://huggingface.co/{repo}/resolve/main/{path}"
MIRROR_TEMPLATE = "https://hf-mirror.com/{repo}/resolve/main/{path}"

# Source preference (D36, settings -> 常规 -> 下载源). D18 made official-first
# the default; a user whose network cannot reach huggingface.co without a proxy
# can lead with hf-mirror instead. The preference only reorders the chain -
# both hosts stay in it, so rule 1 (multi-source fallback) is never weakened.
SOURCE_OFFICIAL = "official"
SOURCE_MIRROR = "mirror"
SOURCE_PREFERENCES: tuple[str, ...] = (SOURCE_OFFICIAL, SOURCE_MIRROR)
DEFAULT_SOURCE = SOURCE_OFFICIAL

SOURCE_TEMPLATES: dict[str, tuple[str, ...]] = {
    SOURCE_OFFICIAL: (OFFICIAL_TEMPLATE, MIRROR_TEMPLATE),
    SOURCE_MIRROR: (MIRROR_TEMPLATE, OFFICIAL_TEMPLATE),
}

# Pre-preference callers keep the D18 chain.
REMOTE_URL_TEMPLATES = SOURCE_TEMPLATES[DEFAULT_SOURCE]


def templates_for(source: str) -> tuple[str, ...]:
    """Ordered URL templates for a source preference (ValueError if unknown)."""
    try:
        return SOURCE_TEMPLATES[source]
    except KeyError:
        raise ValueError(
            f"unknown download source {source!r}; expected one of {SOURCE_PREFERENCES}"
        ) from None


class DownloadError(Exception):
    """Base class - all sources exhausted or transfer broke."""


class DownloadCancelled(DownloadError):
    """Cooperative cancel requested; .incomplete is kept for resume."""


class ChecksumError(DownloadError):
    """SHA256 mismatch - the temp file is garbage and was removed."""


class InsufficientDiskError(DownloadError):
    """Rule 4: free space below the file size before starting."""


_INTEGRITY_LOCK = threading.RLock()


def _integrity_cache_path(models_root: Path) -> Path:
    return models_root / INTEGRITY_CACHE_NAME


def _load_integrity_cache(models_root: Path) -> dict[str, dict]:
    try:
        raw = json.loads(_integrity_cache_path(models_root).read_text(encoding="utf-8"))
        files = raw.get("files")
        if raw.get("version") == 1 and isinstance(files, dict):
            return {
                str(rel): value
                for rel, value in files.items()
                if isinstance(rel, str) and isinstance(value, dict)
            }
    except (OSError, ValueError, TypeError):
        pass
    return {}


def _save_integrity_cache(models_root: Path, files: dict[str, dict]) -> None:
    path = _integrity_cache_path(models_root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            json.dumps({"version": 1, "files": files}, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    except OSError:
        # A read-only/corrupt cache must never make a valid download fail.
        pass


def _fingerprint(path: Path) -> dict[str, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return {
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "ctime_ns": st.st_ctime_ns,
        "ino": int(getattr(st, "st_ino", 0)),
    }


def verify_file_integrity(
    models_root: Path,
    rel: str,
    path: Path,
    expected_bytes: int,
    expected_sha256: str | None,
) -> bool:
    """Verify a model file, caching SHA results by a stat fingerprint.

    ``models/manifest.json`` supplies SHA for LFS files.  The cache prevents
    the UI's repeated status polls from hashing multi-gigabyte weights, while
    a changed file (including a same-size in-place edit) gets a new
    ``mtime_ns``/``ctime_ns`` fingerprint and is re-hashed.
    """
    try:
        safe_relative_path(rel)
        st = path.stat()
    except (OSError, ValueError):
        return False
    if not path.is_file() or st.st_size != expected_bytes:
        return False
    if not expected_sha256:
        return True

    digest = expected_sha256.lower()
    with _INTEGRITY_LOCK:
        files = _load_integrity_cache(models_root)
        cached = files.get(rel)
        current = _fingerprint(path)
        if (
            current is not None
            and cached
            and cached.get("sha256") == digest
            and all(cached.get(k) == v for k, v in current.items())
        ):
            return True

    try:
        actual = sha256_file(path)
    except OSError:
        actual = ""
    ok = actual.lower() == digest

    with _INTEGRITY_LOCK:
        # Re-read while holding the lock: another worker may have populated
        # the cache while this file was being hashed.
        files = _load_integrity_cache(models_root)
        if ok:
            current = _fingerprint(path)
            if current is not None:
                files[rel] = {"sha256": digest, **current}
        else:
            files.pop(rel, None)
        _save_integrity_cache(models_root, files)
    return ok


def remember_file_integrity(
    models_root: Path,
    rel: str,
    path: Path,
    expected_sha256: str | None,
) -> None:
    """Record a freshly verified file without reading it a second time."""
    if not expected_sha256:
        return
    try:
        safe_relative_path(rel)
        current = _fingerprint(path)
    except (OSError, ValueError):
        return
    if current is None:
        return
    with _INTEGRITY_LOCK:
        files = _load_integrity_cache(models_root)
        files[rel] = {"sha256": expected_sha256.lower(), **current}
        _save_integrity_cache(models_root, files)


@dataclass(frozen=True)
class Progress:
    engine_id: str
    dest: str
    done_bytes: int
    total_bytes: int
    t: float  # time.monotonic() sample


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def incomplete_path(final: Path) -> Path:
    return final.with_name(final.name + ".incomplete")


def missing_files(entry: EngineEntry, models_root: Path) -> list[RemoteFile]:
    """Files not complete on disk (absent, wrong size, or bad SHA256)."""
    out: list[RemoteFile] = []
    for f in entry.files:
        p = models_root / f.dest
        if not verify_file_integrity(
            models_root, f.dest, p, f.bytes, f.sha256
        ):
            out.append(f)
    return out


def is_complete(entry: EngineEntry, models_root: Path) -> bool:
    return not missing_files(entry, models_root)


class FileDownloader:
    """Transfers one RemoteFile to models_root/<dest> under the five rules."""

    def __init__(
        self,
        models_root: Path,
        templates: tuple[str, ...] = REMOTE_URL_TEMPLATES,
        chunk: int = CHUNK,
        proxy: str | None = None,
    ) -> None:
        self.models_root = models_root
        self.templates = templates
        self.chunk = chunk
        # None -> default opener chain (honors system/env proxy settings);
        # explicit string forces one proxy (dev machines, managed installs).
        # ProxyHandler({}) would DISABLE env proxies, so it is only used
        # when a proxy is explicitly given.
        if proxy:
            handlers: list[urllib.request.BaseHandler] = [
                urllib.request.ProxyHandler({"http": proxy, "https": proxy})
            ]
        else:
            handlers = []
        self._opener = urllib.request.build_opener(*handlers)

    def download(
        self,
        engine_id: str,
        f: RemoteFile,
        cancel: threading.Event,
        progress: Callable[[Progress], None] | None = None,
        *,
        templates: tuple[str, ...] | None = None,
    ) -> Path:
        """Transfer one file.

        ``templates`` overrides the instance chain for this call - the job
        picks the order once from the user's source preference (D36) and
        passes it down, so every file of a job leads with the same host
        while the other host remains the fallback.
        """
        final = self.models_root / f.dest
        final.parent.mkdir(parents=True, exist_ok=True)
        chain = templates if templates else self.templates

        tmp = incomplete_path(final)
        if not tmp.exists() and verify_file_integrity(
            self.models_root, f.dest, final, f.bytes, f.sha256
        ):
            return final  # already there (raced with another worker)

        # rule 4 - precheck before any network byte.  A resume only needs
        # the bytes still missing, not a second full-file allocation.
        resume = tmp.stat().st_size if tmp.exists() else 0
        if resume > f.bytes:
            resume = 0  # stale file will be discarded by the attempt loop
        needed = max(0, f.bytes - resume)
        free = shutil.disk_usage(final.parent).free
        if free < needed:
            raise InsufficientDiskError(
                f"{f.dest}: need {needed} bytes, only {free} free"
            )

        last_err: Exception | None = None
        for attempt in range(2):  # attempt 2 starts clean if 1 degraded
            if attempt == 1:
                tmp.unlink(missing_ok=True)
            resume = tmp.stat().st_size if tmp.exists() else 0
            if resume == f.bytes and f.bytes > 0:
                break  # fully fetched previously; go verify
            if resume > f.bytes:
                tmp.unlink(missing_ok=True)  # stale garbage from an older plan
                resume = 0
            for tmpl in chain:
                url = tmpl.format(repo=f.repo, path=f.path)
                try:
                    self._fetch(url, tmp, engine_id, f, resume, cancel, progress)
                    break
                except DownloadCancelled:
                    raise
                except (
                    urllib.error.HTTPError,
                    urllib.error.URLError,
                    http.client.HTTPException,  # IncompleteRead etc.
                    OSError,
                ) as e:
                    last_err = e  # rule 1 - next source
                    continue
            else:
                continue
            break
        else:
            raise DownloadError(f"{f.dest}: all sources failed; last={last_err!r}")

        # rule 3 - verify size + hash, then rule 5 - atomic rename
        size = tmp.stat().st_size
        if size != f.bytes:
            tmp.unlink(missing_ok=True)
            raise DownloadError(f"{f.dest}: downloaded {size}B, expected {f.bytes}B")
        if f.sha256 is not None:
            actual = sha256_file(tmp)
            if actual != f.sha256:
                tmp.unlink(missing_ok=True)
                raise ChecksumError(
                    f"{f.dest}: sha256 mismatch (got {actual[:16]}..., want {f.sha256[:16]}...)"
                )
        os.replace(tmp, final)
        remember_file_integrity(self.models_root, f.dest, final, f.sha256)
        return final

    def _fetch(
        self,
        url: str,
        tmp: Path,
        engine_id: str,
        f: RemoteFile,
        resume: int,
        cancel: threading.Event,
        progress: Callable[[Progress], None] | None,
    ) -> None:
        headers = {"User-Agent": UA}
        if resume:
            headers["Range"] = f"bytes={resume}-"
        req = urllib.request.Request(url, headers=headers)
        with self._opener.open(req, timeout=60) as resp:
            status = resp.status
            if status == 206:
                pass  # server honors Range - appending
            elif status == 200:
                resume = 0  # server ignored Range - restart cleanly
            else:
                raise DownloadError(f"unexpected status {status}")
            done = resume
            last_emit = 0.0
            with open(tmp, "ab" if resume else "wb") as out:
                while True:
                    if cancel.is_set():
                        raise DownloadCancelled(f"{engine_id}: cancelled")
                    chunk = resp.read(self.chunk)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    now = time.monotonic()
                    if progress and now - last_emit >= PROGRESS_INTERVAL:
                        last_emit = now
                        progress(Progress(engine_id, f.dest, done, f.bytes, now))
            if progress:
                progress(Progress(engine_id, f.dest, done, f.bytes, time.monotonic()))

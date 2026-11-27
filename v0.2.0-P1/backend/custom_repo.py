"""Custom model library (round-6 user request): local import + URL download.

Two entry points complement the manifest-driven downloader:

- ``import_local``: point at an already-downloaded HF repo directory. If the
  file set matches an official engine's manifest dest basenames the files are
  hard-linked (same drive, instant) or copied into ``models/<dest>`` and the
  engine becomes installed immediately; otherwise the directory is registered
  into ``models/custom/<name>/`` for later versions to wire up.
- ``start_url_download``: download an arbitrary URL into
  ``models/custom/<name>/`` on a lightweight single-file thread (Range
  resume + sha256 + atomic rename). A mirror hint map rewrites known
  vendor hosts (huggingface.co -> hf-mirror.com) when the user opts in.

Both paths record entries in ``models/custom/custom_models.json``. This
module never touches the manifest downloader's five-rules core (D18/D25) -
it runs on its own thread with its own cancel event.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import unquote, urlsplit

from downloader import (
    remember_file_integrity,
    sha256_file,
    verify_file_integrity,
)
from manifest import Manifest

# Progress emissions are throttled to this interval (matches downloader.py).
_PROGRESS_INTERVAL = 0.25
# Chunk size for streamed download (1 MiB).
_CHUNK = 1024 * 1024

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")

# Mirror hints: known vendor hosts and their public mirrors. Only hosts
# listed here are rewritten; everything else downloads direct (honest
# behavior - we do not pretend a mirror exists when it does not).
_MIRRORS: tuple[tuple[str, str], ...] = (
    ("huggingface.co", "hf-mirror.com"),
    ("cdn-lfs.huggingface.co", "hf-mirror.com"),
    ("cdn-lfs-us-1.huggingface.co", "hf-mirror.com"),
    ("cdn-lfs-eu-1.huggingface.co", "hf-mirror.com"),
)


def mirror_for(url: str) -> tuple[str | None, str]:
    """Return (mirrored_url, note) for a known vendor host, else (None, note)."""
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        return None, "URL 无法解析"
    for src, dst in _MIRRORS:
        if host == src or host.endswith("." + src):
            mirrored = url.replace(f"{host}", dst, 1)
            return mirrored, f"已知镜像 {dst}"
    if host.endswith("modelscope.cn"):
        return None, "ModelScope 为国内源，直连即快，无需镜像"
    return None, "该源无已知镜像站，将直连下载"


def _sanitize(name: str, fallback: str) -> str:
    """Strip anything that could escape the models/custom/ directory."""
    cleaned = _SAFE_NAME.sub("-", name).strip(".-")
    return cleaned[:64] or fallback


def _sha256_of(path: Path, cancel: threading.Event) -> str | None:
    """Streaming sha256; returns None if cancelled mid-hash."""
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            while True:
                if cancel.is_set():
                    return None
                block = fh.read(_CHUNK)
                if not block:
                    break
                h.update(block)
    except OSError:
        return None
    return h.hexdigest()


class CustomRepo:
    """Registry + import + URL-download for the custom model library."""

    def __init__(
        self,
        models_root: Path,
        manifest: Manifest,
        emit: Callable[[dict], None],
    ) -> None:
        self._root = models_root
        self._manifest = manifest
        self._emit = emit  # thread-safe callback into the WS event sink
        self._custom_root = models_root / "custom"
        self._registry_path = self._custom_root / "custom_models.json"
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._job: dict[str, Any] | None = None
        self._custom_root.mkdir(parents=True, exist_ok=True)

    # ---- registry ---------------------------------------------------------

    def _read_registry(self) -> list[dict[str, Any]]:
        try:
            raw = json.loads(self._registry_path.read_text(encoding="utf-8"))
            return raw["models"] if isinstance(raw.get("models"), list) else []
        except (OSError, ValueError, KeyError):
            return []

    def _write_registry(self, models: list[dict[str, Any]]) -> None:
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._registry_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"version": 1, "models": models}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self._registry_path)

    def _append(self, entry: dict[str, Any]) -> None:
        with self._lock:
            models = self._read_registry()
            models.append(entry)
            self._write_registry(models)

    def status(self) -> dict[str, Any]:
        return {"models": self._read_registry(), "job": self.job_status()}

    def job_status(self) -> dict[str, Any] | None:
        job = self._job
        if job is None:
            return None
        return {
            "phase": job["phase"],
            "name": job["name"],
            "url": job["url"],
            "file": job.get("file"),
            "done": job.get("done", 0),
            "total": job.get("total", 0),
            # bytes inherited from a .part left by a cancelled attempt; the UI
            # needs it to tell "resumed at 11 MB" apart from "started at 11 MB"
            "resumed": job.get("resumed", 0),
            "note": job.get("note"),
        }

    # ---- engine matching ---------------------------------------------------

    def match_engines(self, source_dir: Path) -> list[str]:
        """Engine ids whose manifest dest basenames all exist in source_dir.

        basenames (not full dests) are the match key because users copy the
        flat HF repo root; shared-base engines (gltr -> gpt2/) therefore need
        the full-set-equality tie-break in ``import_local``.
        """
        src_names = {p.name for p in source_dir.iterdir() if p.is_file()}
        hits: list[str] = []
        for eid in self._manifest.ids():
            entry = self._manifest.get(eid)
            needed = {PurePosixPath(f.dest).name for f in entry.files}
            if needed and needed <= src_names:
                hits.append(eid)
        return hits

    # ---- local import ------------------------------------------------------

    def import_local(
        self,
        source_dir: str,
        mode: str = "link",
        name: str | None = None,
    ) -> dict[str, Any]:
        """Import a local directory. Matching official engines are wired into
        models/<dest> (link preferred, copy on cross-device); non-matching
        dirs are registered under models/custom/<name>/."""
        src = Path(source_dir).expanduser()
        if not src.is_dir():
            raise ValueError(f"目录不存在或不可读: {source_dir}")
        resolved = src.resolve()
        if self._root.resolve() in resolved.parents or resolved == self._root.resolve():
            raise ValueError("不能把 models/ 内部的目录导入到它自己")
        files = [p for p in resolved.iterdir() if p.is_file()]
        if not files:
            raise ValueError("目录里没有文件")

        hits = self.match_engines(resolved)
        src_names = {p.name for p in files}
        if hits:
            # A single superset hit wires the engine (extra repo files like
            # README are ignored); several hits need a full-set tie-break,
            # and genuine ambiguity is refused honestly.
            exact = [
                e
                for e in hits
                if {PurePosixPath(f.dest).name for f in self._manifest.get(e).files}
                == src_names
            ]
            if len(hits) == 1 or len(exact) == 1:
                return self._import_to_engine(resolved, files, (exact or hits)[0], mode)
            raise ValueError(
                f"文件集同时匹配多个引擎 {hits}，无法消歧——请改用官方下载通道"
            )

        label = _sanitize(name or resolved.name, f"local-{time.strftime('%Y%m%d-%H%M%S')}")
        return self._import_to_custom(resolved, files, label, mode, origin_url=None)

    def _transfer(
        self,
        src: Path,
        dst: Path,
        mode: str,
        *,
        expected_bytes: int | None = None,
        expected_sha256: str | None = None,
    ) -> str:
        """link (instant, same drive) with copy fallback, or copy. Returns
        the actual mode used."""
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            rel = dst.relative_to(self._root).as_posix()
            if expected_bytes is not None and verify_file_integrity(
                self._root,
                rel,
                dst,
                expected_bytes,
                expected_sha256,
            ):
                return "exists"  # verified file already in place: keep it
            if expected_bytes is None and dst.stat().st_size == src.stat().st_size:
                return "exists"
            # A corrupt same-size target is intentionally overwritten from
            # the verified source below. Leaving it in place would make the
            # next status poll report a false install.
        if mode == "link":
            try:
                os.link(src, dst)
                if expected_sha256 and expected_bytes is not None:
                    remember_file_integrity(
                        self._root,
                        dst.relative_to(self._root).as_posix(),
                        dst,
                        expected_sha256,
                    )
                return "link"
            except OSError:  # cross-device / unsupported -> fall back to copy
                pass
        shutil.copy2(src, dst)
        if expected_sha256 and expected_bytes is not None:
            remember_file_integrity(
                self._root,
                dst.relative_to(self._root).as_posix(),
                dst,
                expected_sha256,
            )
        return "copy"

    def _import_to_engine(
        self,
        src: Path,
        files: list[Path],
        engine_id: str,
        mode: str,
    ) -> dict[str, Any]:
        entry = self._manifest.get(engine_id)
        by_name = {PurePosixPath(f.dest).name: f.dest for f in entry.files}
        transferred: list[dict[str, Any]] = []
        for p in files:
            dest = by_name.get(p.name)
            if dest is None:
                continue  # extra repo file (README etc.) - ignore
            f = next(f for f in entry.files if f.dest == dest)
            if p.stat().st_size != f.bytes:
                raise ValueError(f"{p.name}: 文件大小不符（共享目录可能已损坏）")
            if f.sha256 is not None and sha256_file(p) != f.sha256:
                raise ValueError(f"{p.name}: SHA256 校验失败，拒绝导入")
            used = self._transfer(
                p,
                self._root / dest,
                mode,
                expected_bytes=f.bytes,
                expected_sha256=f.sha256,
            )
            transferred.append({"name": p.name, "dest": dest, "via": used})
        self._append(
            {
                "id": uuid.uuid4().hex[:8],
                "name": engine_id,
                "source": "local",
                "origin_url": None,
                "mirrored": False,
                "note": f"匹配官方引擎 {engine_id}，已就位 {len(transferred)} 个文件",
                "dir": str(PurePosixPath(entry.files[0].dest).parent),
                "engine": engine_id,
                "files": [
                    {"name": t["name"], "bytes": (src / t["name"]).stat().st_size, "sha256": None}
                    for t in transferred
                ],
                "imported_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        return {
            "kind": "engine",
            "engine": engine_id,
            "transferred": transferred,
            "note": f"已按 manifest 就位到 {engine_id}（硬链接优先，跨盘自动转复制）；"
            "引擎是否可用以存储校验为准",
        }

    def _import_to_custom(
        self,
        src: Path,
        files: list[Path],
        label: str,
        mode: str,
        origin_url: str | None,
    ) -> dict[str, Any]:
        dst_dir = self._custom_root / label
        if dst_dir.exists() and any(dst_dir.iterdir()):
            raise ValueError(f"custom/{label} 已存在且非空，请换一个名称")
        dst_dir.mkdir(parents=True, exist_ok=True)
        self._cancel.clear()
        entries: list[dict[str, Any]] = []
        for p in files:
            used = self._transfer(p, dst_dir / p.name, mode)
            sha = _sha256_of(dst_dir / p.name, self._cancel)
            entries.append(
                {
                    "name": p.name,
                    "bytes": p.stat().st_size,
                    "sha256": sha,
                    "via": used,
                }
            )
        self._append(
            {
                "id": uuid.uuid4().hex[:8],
                "name": label,
                "source": "local" if origin_url is None else "url",
                "origin_url": origin_url,
                "mirrored": False,
                "note": "未匹配任何官方引擎的文件集，登记进自定义库（引擎接入为后续版本）",
                "dir": f"custom/{label}",
                "files": entries,
                "imported_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        return {
            "kind": "custom",
            "name": label,
            "transferred": [{"name": e["name"], "via": e["via"]} for e in entries],
            "note": "已登记进自定义模型库（未匹配官方引擎文件集）",
        }

    # ---- URL download ------------------------------------------------------

    def _only_partial(self, label: str, fname: str) -> bool:
        """True when custom/<label> holds nothing but <fname>.part, i.e. the
        remains of a cancelled attempt that a retry can resume in place."""
        try:
            return {p.name for p in (self._custom_root / label).iterdir()} == {
                f"{fname}.part"
            }
        except OSError:
            return False

    def start_url_download(
        self,
        url: str,
        mirror_choice: str,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Validate and launch a single-file download thread. Returns the
        initial job status (or raises ValueError with a user-facing note)."""
        if not isinstance(url, str) or not url.strip().lower().startswith(("http://", "https://")):
            raise ValueError("URL 必须以 http:// 或 https:// 开头")
        url = url.strip()
        if self._job is not None and self._job["phase"] in ("running", "hashing"):
            raise ValueError("已有下载任务进行中（先取消再开始新任务）")

        mirrored = False
        note: str | None = None
        if mirror_choice == "mirror":
            mirrored_url, hint = mirror_for(url)
            if mirrored_url is None:
                raise ValueError(f"无法使用镜像：{hint}")
            url = mirrored_url
            mirrored = True
            note = hint
        elif mirror_choice != "direct":
            raise ValueError("mirror 必须是 'mirror' 或 'direct'")

        path = unquote(urlsplit(url).path)
        raw_name = PurePosixPath(path).name or PureWindowsPath(path).name
        fname = _sanitize(raw_name, "")
        if not fname:
            fname = f"model-{time.strftime('%Y%m%d-%H%M%S')}.bin"
        label = _sanitize(name or fname.rsplit(".", 1)[0], f"url-{time.strftime('%Y%m%d-%H%M%S')}")
        # A cancelled attempt leaves custom/<label>/<file>.part behind, so the
        # naive "pick a free dir" loop always forked <label>-2 and the Range
        # resume below was unreachable. Reuse the dir when the only thing in it
        # is that partial file; anything else still gets a fresh label.
        base = label
        i = 2
        while (self._custom_root / label).exists() and not self._only_partial(label, fname):
            label = f"{base}-{i}"
            i += 1

        dest_dir = self._custom_root / label
        dest = dest_dir / fname
        # read the inherited byte count here, not in the worker thread: the
        # immediate response (and the UI's first render) must already say
        # "resumed at N MB" instead of 0 followed by a jump
        part = dest.with_suffix(dest.suffix + ".part")
        resumed = part.stat().st_size if part.exists() else 0
        with self._lock:
            self._cancel = threading.Event()
            self._job = {
                "phase": "running",
                "name": label,
                "url": url,
                "file": fname,
                "done": resumed,
                "total": 0,
                "resumed": resumed,
                "note": note,
                "cancel": self._cancel,
                "mirrored": mirrored,
                "dest": dest,
            }
        t = threading.Thread(target=self._run_download, daemon=True)
        t.start()
        return self.job_status()  # type: ignore[return-value]

    def cancel(self) -> bool:
        if self._job is None:
            return False
        self._cancel.set()
        return True

    def _emit_job(self, **fields: Any) -> None:
        st = self.job_status()
        if st is None:
            return
        try:
            self._emit({"kind": "custom-dl", **st, **fields})
        except Exception:  # noqa: BLE001 - event sink must never kill the job
            log_note = f"custom-dl emit failed: fields={fields}"
            print(log_note, file=sys.stderr, flush=True)

    def _run_download(self) -> None:
        job = self._job
        assert job is not None
        cancel: threading.Event = job["cancel"]
        dest: Path = job["dest"]
        url: str = job["url"]
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            part = dest.with_suffix(dest.suffix + ".part")
            done = part.stat().st_size if part.exists() else 0
            job["resumed"] = done
            self._emit_job(phase="start")
            headers = {"User-Agent": "aigc-detector-custom/0.1"}
            if done > 0:
                headers["Range"] = f"bytes={done}-"
            req = urllib.request.Request(url, headers=headers)
            last_emit = 0.0
            with urllib.request.urlopen(req, timeout=60) as resp:
                if resp.status == 206:
                    # "bytes <start>-<end>/<size>": the last field is the whole
                    # file, NOT the remaining bytes. Adding `done` to it made
                    # total too big and the completeness check below rejected
                    # every resume with "下载不完整".
                    rng = resp.headers.get("Content-Range", "")
                    total = int(rng.rsplit("/", 1)[1]) if "/" in rng else 0
                    if not total:  # no usable Content-Range: don't invent one
                        total = done + int(resp.headers.get("Content-Length") or 0)
                else:
                    done = 0  # server ignored Range; restart from scratch
                    job["resumed"] = 0
                    total = int(resp.headers.get("Content-Length") or 0)
                job["total"] = total
                self._emit_job()
                with part.open("ab" if done else "wb") as out:
                    while True:
                        if cancel.is_set():
                            job["phase"] = "cancelled"
                            self._emit_job()
                            return
                        block = resp.read(_CHUNK)
                        if not block:
                            break
                        out.write(block)
                        done += len(block)
                        job["done"] = done
                        now = time.monotonic()
                        if now - last_emit >= _PROGRESS_INTERVAL:
                            last_emit = now
                            self._emit_job()
            actual = part.stat().st_size
            if total and actual != total:
                raise ValueError(f"下载不完整：{actual}/{total} 字节")
            # sha256 + atomic rename (keep .part on failure for resume)
            job["phase"] = "hashing"
            self._emit_job()
            sha = _sha256_of(part, cancel)
            if sha is None:
                job["phase"] = "cancelled"
                self._emit_job()
                return
            os.replace(part, dest)
            self._append(
                {
                    "id": uuid.uuid4().hex[:8],
                    "name": job["name"],
                    "source": "url",
                    "origin_url": url,
                    "mirrored": bool(job.get("mirrored")),
                    "note": job.get("note"),
                    "dir": f"custom/{job['name']}",
                    "files": [{"name": dest.name, "bytes": actual, "sha256": sha}],
                    "imported_at": datetime.now(UTC).isoformat(timespec="seconds"),
                }
            )
            job["phase"] = "done"
            self._emit_job()
        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as e:
            if self._job is not None:
                self._job["phase"] = "error"
                self._job["note"] = str(e)
                self._emit_job()
        finally:
            # keep the terminal job record visible; clear cancel event
            self._cancel = threading.Event()

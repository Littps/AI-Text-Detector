"""Day-partitioned sidecar log files + range export (round-30 dev option).

Why a file at all: stderr already carries the live log (the Tauri shell
inherits it), but stderr dies with the terminal and cannot answer "what
happened yesterday". "导出日志" needs a history that survives a restart, so
``log()`` mirrors every line into ``logs/sidecar-YYYY-MM-DD.log``.

Privacy rule: this module only ever receives what ``server.log`` writes -
connection / engine / download / residency events. Document text never
reaches it, and the exported header says so explicitly.

Robustness rule: logging must never break the app. Every filesystem error is
swallowed here (missing dir, read-only install, locked file), because a
diagnostics feature that throws is worse than no diagnostics.
"""
from __future__ import annotations

import os
import re
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, TextIO

FILE_RE = re.compile(r"^sidecar-(\d{4}-\d{2}-\d{2})\.log$")

#: older files are pruned on start-up; 30 days is plenty for a desktop tool
KEEP_DAYS = 30
#: hard cap per exported file list, so one runaway log cannot stall the socket
MAX_EXPORT_BYTES = 24 * 1024 * 1024
#: the ranges the UI offers
VALID_DAYS = (1, 3, 7)


class LogStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.Lock()
        self._open_day: date | None = None
        self._fh: TextIO | None = None
        self._enabled = True
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Do not fall back to Path(): iterating/pruning the current
            # directory would turn a failed log setup into unrelated file
            # deletion. Keep the object usable but make all I/O a no-op.
            self._enabled = False
        self.prune()

    # ---- writing ---------------------------------------------------------
    def _file_for(self, day: date) -> Path:
        return self.root / f"sidecar-{day.isoformat()}.log"

    def _handle(self, day: date) -> TextIO | None:
        if not self._enabled:
            return None
        if self._open_day == day and self._fh is not None:
            return self._fh
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
        try:
            fh = self._file_for(day).open("a", encoding="utf-8", newline="\n")
        except OSError:
            self._open_day = None
            return None
        self._open_day = day
        self._fh = fh
        return fh

    def write(self, msg: str) -> None:
        stamp = datetime.now()
        line = f"{stamp.isoformat(timespec='seconds')} {msg}\n"
        with self._lock:
            fh = self._handle(stamp.date())
            if fh is None:
                return
            try:
                fh.write(line)
                fh.flush()
            except (OSError, ValueError):
                # disk full / handle closed behind our back: drop the line,
                # close so the next write reopens a fresh handle
                try:
                    fh.close()
                except OSError:
                    pass
                self._fh = None
                self._open_day = None

    # ---- maintenance -----------------------------------------------------
    def prune(self, today: date | None = None) -> list[str]:
        """Delete files older than KEEP_DAYS. Only our own naming is touched."""
        if not self._enabled:
            return []
        today = today or date.today()
        cutoff = today - timedelta(days=KEEP_DAYS - 1)
        removed: list[str] = []
        try:
            entries = list(self.root.iterdir())
        except OSError:
            return removed
        for p in entries:
            m = FILE_RE.match(p.name)
            if not m:
                continue
            try:
                day = date.fromisoformat(m.group(1))
            except ValueError:
                continue
            if day < cutoff:
                try:
                    p.unlink()
                    removed.append(p.name)
                except OSError:
                    pass
        return removed

    # ---- export ----------------------------------------------------------
    def _days_back(self, days: int, today: date) -> list[date]:
        return [today - timedelta(days=n) for n in range(days - 1, -1, -1)]

    def export(self, days: int, today: date | None = None, meta: dict[str, Any] | None = None) -> dict:
        """Concatenate every in-range file, oldest first, behind a header.

        Returns the text plus enough metadata for the UI to name the file and
        tell the user what they got (days / files / lines / bytes).
        """
        if days not in VALID_DAYS:
            raise ValueError(f"days must be one of {VALID_DAYS}")
        today = today or date.today()
        wanted = self._days_back(days, today)

        chunks: list[str] = []
        used: list[str] = []
        lines = 0
        total = 0
        truncated = False
        for day in wanted:
            path = self._file_for(day)
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not raw:
                continue
            used.append(path.name)
            total += len(raw)
            if total > MAX_EXPORT_BYTES:
                # keep the tail: the newest lines are the interesting ones
                keep = MAX_EXPORT_BYTES - (total - len(raw))
                raw = raw[-max(keep, 0) :]
                truncated = True
            chunks.append(f"===== {path.name} =====\n{raw.rstrip()}\n")
            lines += raw.count("\n") + (0 if raw.endswith("\n") else 1)

        head = self._header(days, wanted, used, lines, meta or {})
        if not chunks:
            body = "（该范围内没有任何日志——应用可能从未运行过，或日志目录已被清空）\n"
        else:
            body = "\n".join(chunks)
        marker = f"\n[已截断：导出上限 {MAX_EXPORT_BYTES // (1024 * 1024)} MB，保留最新内容]\n"
        if truncated:
            body = body[:MAX_EXPORT_BYTES]  # char-level guard before byte accounting
        text = head + body
        # Header/footer and UTF-8 Chinese text can push the logical byte count
        # over the cap even when the per-file character budget passed. Enforce
        # the real wire limit after assembling the final document.
        body_room = max(
            0,
            MAX_EXPORT_BYTES
            - len(head.encode("utf-8"))
            - len(marker.encode("utf-8")),
        )
        body_bytes = body.encode("utf-8")
        if len(body_bytes) > body_room:
            body = body_bytes[-body_room:].decode("utf-8", errors="ignore")
            truncated = True
        text = head + body
        if truncated:
            text += marker
        return {
            "text": text,
            "days": days,
            "from": wanted[0].isoformat(),
            "to": wanted[-1].isoformat(),
            "files": len(used),
            "lines": lines,
            "bytes": len(text.encode("utf-8")),
            "truncated": truncated,
        }

    def close(self) -> None:
        """Close the daily handle deterministically (Windows test cleanup)."""
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                except OSError:
                    pass
                self._fh = None
                self._open_day = None

    def _header(
        self,
        days: int,
        wanted: list[date],
        used: list[str],
        lines: int,
        meta: dict[str, Any],
    ) -> str:
        label = {1: "今天", 3: "近 3 天", 7: "近 7 天"}.get(days, f"近 {days} 天")
        out = [
            "# AI Text Detector · 日志导出",
            f"# 生成时间: {datetime.now().isoformat(timespec='seconds')}",
            f"# 范围: {label}（{wanted[0].isoformat()} ~ {wanted[-1].isoformat()}）",
            f"# 命中文件: {len(used)} 个" + (f"（{', '.join(used)}）" if used else ""),
            f"# 行数: {lines}",
            "# 隐私: 日志只记录连接、引擎加载、下载、驻留等事件，不包含任何待检测文本。",
        ]
        for key, title in (
            ("app", "应用"),
            ("python", "Python"),
            ("torch", "Torch"),
            ("cuda", "CUDA"),
            ("tier", "硬件档位"),
            ("platform", "系统"),
            ("models_dir", "模型目录"),
        ):
            val = meta.get(key)
            if val:
                out.append(f"# {title}: {val}")
        out.append("")
        out.append("")
        return "\n".join(out)


def default_root(version_root: Path) -> Path:
    """``AIGC_LOG_DIR`` wins (tests / portable runs), else ``<app>/logs``."""
    override = os.environ.get("AIGC_LOG_DIR", "").strip()
    return Path(override) if override else version_root / "logs"

"""Manifest access layer (models/manifest.json).

The manifest is the downloader's single truth source: per-engine file
list with canonical SHA256 (from HF LFS oids) and byte sizes, plus the
D20 risk fields (risk_level / risk_text_key / min_vram_mb /
default_selected). Pure stdlib, torch-free - safe to import anywhere.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath


@dataclass(frozen=True)
class RemoteFile:
    """One file of one engine."""

    repo: str  # e.g. "openai-community/gpt2"
    path: str  # repo-side path, e.g. "model.safetensors"
    dest: str  # on-disk path relative to models/ root (explicit layout:
    #            engines default to "<engine_id>/<path>", gltr's shared
    #            base lives at "gpt2/<path>", binoculars nests per-repo)
    sha256: str | None  # None only for non-LFS remote-only files (size-checked)
    bytes: int

    def url(self, host: str) -> str:
        """Download URL on a given host (official first, mirror fallback)."""
        return f"https://{host}/{self.repo}/resolve/main/{self.path}"


@dataclass(frozen=True)
class EngineEntry:
    engine_id: str
    files: tuple[RemoteFile, ...]
    default_selected: bool
    risk_level: str  # "none" | "medium" | "high"
    risk_text_key: str | None
    min_vram_mb: int

    @property
    def total_bytes(self) -> int:
        return sum(f.bytes for f in self.files)

    @property
    def repo_label(self) -> str:
        return "+".join(dict.fromkeys(f.repo for f in self.files))


_WINDOWS_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}
_CONTROL_CHARS = re.compile(r"[\x00-\x1f]")


def safe_relative_path(rel: str) -> str:
    """Validate a forward-slash path that must stay inside ``models/``.

    Windows accepts several spellings that a POSIX-only check misses:
    ``..\\escape``, drive-relative ``C:foo``, UNC paths, alternate data
    streams (``file:stream``), and trailing dots/spaces that the filesystem
    normalises.  Manifest destinations are intentionally restricted to the
    canonical portable form because they are used for file writes and
    irreversible cleanup.
    """
    if not isinstance(rel, str) or not rel or _CONTROL_CHARS.search(rel):
        raise ValueError("path must be a non-empty string without control characters")
    if "\\" in rel:
        raise ValueError("path must use forward slashes")
    if rel.startswith("/"):
        raise ValueError("path must be relative")

    parts = rel.split("/")
    if any(not part or part in (".", "..") for part in parts):
        raise ValueError("path must not contain empty, current, or parent segments")

    for part in parts:
        if ":" in part:
            raise ValueError("path must not contain a drive or alternate stream")
        if part.endswith((" ", ".")):
            raise ValueError("path must not end a segment with a dot or space")
        # Windows treats these names as devices even when an extension is
        # present (e.g. ``NUL.txt``).
        stem = part.split(".", 1)[0].strip().upper()
        if stem.lower() in _WINDOWS_RESERVED:
            raise ValueError("path contains a Windows reserved name")

    p = PurePosixPath(rel)
    win = PureWindowsPath(rel)
    if (
        p.is_absolute()
        or win.is_absolute()
        or win.drive
        or win.root
        or str(p) != rel
    ):
        raise ValueError("path must be a canonical relative path")
    return rel


def _safe_dest(engine_id: str, dest: str) -> str:
    """Reject destinations that would escape the models/ root.

    ``models_root / dest`` replaces the whole base when dest is absolute
    (POSIX root or a ``C:``-style drive prefix), and ``..`` segments
    traverse upwards - both would turn the downloader into an
    arbitrary-file-writer if the manifest were tampered with.
    """
    try:
        return safe_relative_path(dest)
    except ValueError as e:
        raise ValueError(f"manifest {engine_id}: unsafe dest {dest!r} ({e})") from None


class Manifest:
    def __init__(self, engines: dict[str, EngineEntry], generated_at: str) -> None:
        self._engines = engines
        self.generated_at = generated_at

    @classmethod
    def load(cls, path: Path) -> Manifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("version") != 1:
            raise ValueError(f"unsupported manifest version {raw.get('version')!r}")
        engines: dict[str, EngineEntry] = {}
        for eid, e in raw["engines"].items():
            files = tuple(
                RemoteFile(
                    repo=f["repo"],
                    path=f["path"],
                    dest=_safe_dest(eid, f["dest"]),
                    sha256=f.get("sha256"),
                    bytes=int(f["bytes"]),
                )
                for f in e["files"]
            )
            engines[eid] = EngineEntry(
                engine_id=eid,
                files=files,
                default_selected=bool(e["default_selected"]),
                risk_level=e.get("risk_level", "none"),
                risk_text_key=e.get("risk_text_key"),
                min_vram_mb=int(e.get("min_vram_mb", 0)),
            )
        return cls(engines, raw.get("generated_at", ""))

    def get(self, engine_id: str) -> EngineEntry:
        try:
            return self._engines[engine_id]
        except KeyError:
            raise KeyError(f"unknown engine in manifest: {engine_id!r}") from None

    def __contains__(self, engine_id: object) -> bool:
        return engine_id in self._engines

    def ids(self) -> list[str]:
        return list(self._engines)

    def default_ids(self) -> list[str]:
        return [eid for eid, e in self._engines.items() if e.default_selected]

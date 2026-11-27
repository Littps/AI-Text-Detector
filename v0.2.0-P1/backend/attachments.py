"""Session-scoped attachment table for the composer (D40).

Why the text lives here instead of in the browser: the UI only ever sends file
*paths* (Tauri) or raw bytes (browser fallback) once, and the detect request
then carries an ordered list of ids.  A 300-page PDF therefore never travels
back and forth over the websocket, and the card's char count comes from the
same parse the detector will use — so "12,480 字" cannot disagree with what is
actually scored.

Two ceilings protect the sidecar: at most 20 attachments, and at most 4M
characters of extracted text in total.  They are checked *before* the text is
retained, so a rejected import leaves the table untouched.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field

import filereader

MAX_ITEMS = 20
MAX_TOTAL_CHARS = 4_000_000


class UnknownAttachment(KeyError):
    """The UI referenced an id the sidecar no longer holds."""


@dataclass
class Attachment:
    id: str
    name: str
    ext: str
    chars: int
    bytes: int
    text: str
    notes: list[str] = field(default_factory=list)

    def meta(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "ext": self.ext,
            "chars": self.chars,
            "bytes": self.bytes,
            "notes": self.notes,
        }


class AttachmentStore:
    def __init__(self) -> None:
        self._items: dict[str, Attachment] = {}
        self._seq = 0

    # ---- intake --------------------------------------------------------
    def _next_id(self) -> str:
        self._seq += 1
        return f"att-{self._seq}"

    def _room(self) -> int:
        """Characters still allowed before hitting the total ceiling."""
        return MAX_TOTAL_CHARS - sum(a.chars for a in self._items.values())

    def import_paths(self, paths: list[str]) -> dict:
        if len(paths) + len(self._items) > MAX_ITEMS:
            raise ValueError(f"最多同时导入 {MAX_ITEMS} 个文件")
        items: list[dict] = []
        errors: list[dict] = []
        for raw in paths:
            try:
                got = filereader.read_path(raw)
                items.append(self._retain(got))
            except filereader.ReadError as e:
                errors.append({"name": _basename(raw), "error": str(e)})
        return {"items": items, "errors": errors}

    def import_blobs(self, blobs: list[dict]) -> dict:
        """Browser fallback: [{name, data}] where data is base64."""
        if len(blobs) + len(self._items) > MAX_ITEMS:
            raise ValueError(f"最多同时导入 {MAX_ITEMS} 个文件")
        items: list[dict] = []
        errors: list[dict] = []
        for blob in blobs:
            name = blob.get("name")
            data = blob.get("data")
            if not isinstance(name, str) or not isinstance(data, str):
                errors.append({"name": str(name or "?"), "error": "文件数据格式不对"})
                continue
            try:
                raw = base64.b64decode(data, validate=True)
            except (binascii.Error, ValueError):
                errors.append({"name": name, "error": "文件内容传输损坏，请重试"})
                continue
            try:
                got = filereader.read_bytes(name, raw)
                items.append(self._retain(got))
            except filereader.ReadError as e:
                errors.append({"name": name, "error": str(e)})
        return {"items": items, "errors": errors}

    def _retain(self, got: filereader.Extracted) -> dict:
        if got.chars > self._room():
            raise filereader.ReadError(
                f"附件总量超过上限（剩余 {self._room():,} 字，本文件 {got.chars:,} 字）"
            )
        att = Attachment(
            id=self._next_id(),
            name=got.name,
            ext=got.ext,
            chars=got.chars,
            bytes=got.bytes,
            text=got.text,
            notes=got.notes,
        )
        self._items[att.id] = att
        return att.meta()

    # ---- lifecycle -----------------------------------------------------
    def drop(self, aid: str) -> bool:
        return self._items.pop(aid, None) is not None

    def clear(self) -> int:
        n = len(self._items)
        self._items.clear()
        self._seq = 0
        return n

    def meta(self) -> list[dict]:
        return [a.meta() for a in self._items.values()]

    # ---- compose -------------------------------------------------------
    def compose(self, text: str, ids: list[str]) -> tuple[str, list[str]]:
        """Text first, then attachments in the order the UI sent them.

        The body is intentionally not ``strip()``-ed: Python span offsets
        index the exact string the detector receives, and the UI keeps the
        same source text for highlighting. Trimming here shifts every span
        after leading whitespace. Pure whitespace is still ignored when there
        are no attachments, so an empty composer stays rejectable.
        """
        parts: list[str] = []
        names: list[str] = []
        if text.strip():
            parts.append(text)
        for aid in ids:
            att = self._items.get(aid)
            if att is None:
                raise UnknownAttachment(aid)
            parts.append(att.text)
            names.append(att.name)
        return "\n\n".join(parts), names


def _basename(path: str) -> str:
    # paths may be Windows (C:\a\b.docx) or POSIX; no os.path here on purpose,
    # because the sidecar runs on Windows but the browser fallback sends names
    return path.replace("\\", "/").rsplit("/", 1)[-1] or path

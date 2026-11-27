"""Composer file import: turn a picked document into plain text (D40).

Three readers, no heavyweight dependencies:

    .txt / .md   plain read with encoding detection
    .docx        zipfile + ElementTree over word/document.xml
    .pdf         pypdf, page by page

Why docx is parsed by hand: python-docx pulls in lxml (a multi-megabyte binary
wheel) for what is, for text extraction, a short walk over w:p / w:t / w:tab /
w:br.  The sidecar ships as an embeddable Python, so dependency weight is paid
twice (wheel + installed tree).

A PDF without a text layer is a *user-facing* condition, not an internal error:
the message goes verbatim into the attachment card so the UI can tell the user
to copy the text instead.  Every reader returns text only; layout (multi-column,
tables) is flattened in reading order.

Limits are deliberate: a 20 MiB document cap keeps a stray 300 MiB video from
being slurped into memory, and a 2M character cap keeps a pathological PDF from
becoming a websocket frame nobody can send.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

SUPPORTED = (".docx", ".pdf", ".txt", ".md")

MAX_BYTES = 20 * 1024 * 1024  # per document
MAX_CHARS = 2_000_000  # after extraction
MIN_PDF_CHARS = 20  # below this a PDF is treated as a scan


class ReadError(Exception):
    """Carries a message meant for the user, not a traceback."""


@dataclass
class Extracted:
    text: str
    name: str
    ext: str
    chars: int
    bytes: int
    notes: list[str] = field(default_factory=list)


# ---- entry points -------------------------------------------------------
def read_path(path: str | Path) -> Extracted:
    p = Path(path)
    try:
        size = p.stat().st_size
    except OSError as e:
        raise ReadError(f"读不到这个文件：{e.strerror or e}") from e
    if not p.is_file():
        raise ReadError("这是一个文件夹，不是文件")
    if size > MAX_BYTES:
        raise ReadError(f"文件太大（{size / 1048576:.1f} MB，上限 20 MB）")
    try:
        data = p.read_bytes()
    except OSError as e:
        raise ReadError(f"读不到这个文件：{e.strerror or e}") from e
    return read_bytes(p.name, data)


def read_bytes(name: str, data: bytes) -> Extracted:
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED:
        raise ReadError(f"不支持 {ext or '这个'} 格式，目前支持 docx / pdf / txt / md")
    if len(data) > MAX_BYTES:
        raise ReadError(f"文件太大（{len(data) / 1048576:.1f} MB，上限 20 MB）")

    notes: list[str] = []
    if ext == ".docx":
        text = _read_docx(data)
    elif ext == ".pdf":
        text, notes = _read_pdf(data)
    else:
        text = _read_text(data)

    text = _normalise(text)
    if not text:
        raise ReadError("这个文件里没有可检测的文字")
    if len(text) > MAX_CHARS:
        raise ReadError(f"正文太长（{len(text):,} 字，上限 {MAX_CHARS:,} 字）")
    return Extracted(text=text, name=name, ext=ext.lstrip("."), chars=len(text), bytes=len(data), notes=notes)


# ---- readers ------------------------------------------------------------
def _read_text(data: bytes) -> str:
    return _decode(data)


def _read_docx(data: bytes) -> str:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise ReadError("不是有效的 docx（可能已损坏，或是 .doc 旧格式）") from e
    with zf:
        try:
            xml = zf.read("word/document.xml")
        except KeyError as e:
            raise ReadError("这个 docx 里找不到正文（word/document.xml）") from e
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        raise ReadError("docx 的正文结构损坏，解析失败") from e

    paras: list[str] = []
    for p in root.iter(f"{W}p"):
        buf: list[str] = []
        for node in p.iter():
            tag = node.tag
            if tag == f"{W}t":
                buf.append(node.text or "")
            elif tag == f"{W}tab":
                buf.append("\t")
            elif tag in (f"{W}br", f"{W}cr"):
                buf.append("\n")
        paras.append("".join(buf))
    # a docx made only of tables still yields paragraphs (inside w:tc), so a
    # single join keeps reading order; empty paragraphs become blank lines
    return "\n".join(paras)


def _read_pdf(data: bytes) -> tuple[str, list[str]]:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover - dependency is declared
        raise ReadError("sidecar 缺少 pypdf，无法读取 PDF") from e

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as e:  # pypdf raises a zoo of types on malformed files
        raise ReadError(f"PDF 打不开（{type(e).__name__}）") from e

    if reader.is_encrypted:
        try:
            if reader.decrypt("") == 0:  # type: ignore[attr-defined]
                raise ReadError("这个 PDF 有密码保护，先解除密码再导入")
        except ReadError:
            raise
        except Exception as e:
            raise ReadError("这个 PDF 有密码保护，先解除密码再导入") from e

    try:
        pages = reader.pages
    except Exception as e:
        raise ReadError(f"PDF 页结构损坏（{type(e).__name__}）") from e

    chunks: list[str] = []
    blank = 0
    for page in pages:
        try:
            t = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - one malformed page must not sink the document
            t = ""
        if not t.strip():
            blank += 1
        chunks.append(t)

    text = "\n\n".join(chunks).strip()
    if len(text) < MIN_PDF_CHARS:
        raise ReadError("这份 PDF 没有可提取的文字层（很可能是扫描件或纯图片），请复制文字后粘贴")

    notes: list[str] = []
    total = len(chunks)
    if blank:
        notes.append(f"{blank}/{total} 页没有文字（可能是图片页）")
    return text, notes


# ---- helpers ------------------------------------------------------------
def _decode(data: bytes) -> str:
    """Try the encodings a Chinese user's .txt actually arrives in."""
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig", errors="replace")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    for enc in ("utf-8", "gb18030"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _normalise(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u3000", " ")  # nbsp / ideographic space
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

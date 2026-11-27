"""D40 file import: the three readers behind the composer attachment cards."""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import filereader as fr

FIX = Path(__file__).resolve().parent / "fixtures"

DOCX_NS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def make_docx(body: str) -> bytes:
    """Build the smallest valid docx: one part, no styles, no rels."""
    xml = f'<?xml version="1.0" encoding="UTF-8"?><w:document {DOCX_NS}><w:body>{body}</w:body></w:document>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types/>')
        zf.writestr("word/document.xml", xml)
    return buf.getvalue()


# ---- txt / md -----------------------------------------------------------
def test_txt_utf8() -> None:
    r = fr.read_bytes("a.txt", "第一行\n第二行\n".encode())
    assert r.text == "第一行\n第二行"
    assert r.ext == "txt" and r.chars == 7


def test_txt_with_bom_is_stripped() -> None:
    r = fr.read_bytes("b.txt", "\ufeff标题\n".encode("utf-8"))
    assert r.text == "标题"  # the BOM must not survive into the detected text


def test_txt_gbk_is_detected() -> None:
    r = fr.read_bytes("c.txt", "这是 GBK 编码的文本".encode("gb18030"))
    assert "GBK" in r.text


def test_markdown_is_treated_as_text() -> None:
    r = fr.read_bytes("notes.md", b"# Title\n\nbody\n")
    assert r.text == "# Title\n\nbody"
    assert r.ext == "md"


def test_crlf_and_nbsp_are_normalised() -> None:
    r = fr.read_bytes("d.txt", "a\u00a0b\r\n\r\n\r\n\r\nc\r\n".encode())
    assert r.text == "a b\n\nc"


def test_empty_text_file_is_an_error() -> None:
    with pytest.raises(fr.ReadError, match="没有可检测的文字"):
        fr.read_bytes("empty.txt", b"   \n\n  ")


# ---- docx ---------------------------------------------------------------
def test_docx_paragraphs_tabs_and_breaks() -> None:
    body = (
        "<w:p><w:r><w:t>第一段</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>标签</w:t><w:tab/><w:t>值</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>换行</w:t><w:br/><w:t>之后</w:t></w:r></w:p>"
    )
    r = fr.read_bytes("x.docx", make_docx(body))
    assert r.text == "第一段\n标签\t值\n换行\n之后"


def test_docx_table_cells_stay_in_reading_order() -> None:
    body = (
        "<w:tbl><w:tr><w:tc><w:p><w:r><w:t>单元格 A</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>单元格 B</w:t></w:r></w:p></w:tc></w:tr></w:tbl>"
    )
    assert fr.read_bytes("t.docx", make_docx(body)).text == "单元格 A\n单元格 B"


def test_docx_real_world_fixture() -> None:
    r = fr.read_path(FIX / "sample.docx")
    assert "第一段" in r.text and "quick brown fox" in r.text
    assert r.ext == "docx" and r.chars > 100


def test_broken_zip_is_reported() -> None:
    with pytest.raises(fr.ReadError, match="不是有效的 docx"):
        fr.read_bytes("bad.docx", b"PK\x03\x04 truncated")


def test_zip_without_document_part_is_reported() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/other.xml", "<x/>")
    with pytest.raises(fr.ReadError, match="找不到正文"):
        fr.read_bytes("old.docx", buf.getvalue())


# ---- pdf ---------------------------------------------------------------
def test_pdf_real_world_fixture() -> None:
    r = fr.read_path(FIX / "sample.pdf")
    assert "第一段" in r.text and "detection" in r.text
    assert r.notes == []


def test_scanned_pdf_without_text_layer_is_user_facing_error() -> None:
    with pytest.raises(fr.ReadError, match="没有可提取的文字层"):
        fr.read_path(FIX / "blank.pdf")


def test_pdf_garbage_is_reported() -> None:
    with pytest.raises(fr.ReadError, match="PDF 打不开"):
        fr.read_bytes("x.pdf", b"%PDF-1.4 but not really")


# ---- guard rails --------------------------------------------------------
def test_unsupported_extension() -> None:
    with pytest.raises(fr.ReadError, match="目前支持 docx / pdf / txt / md"):
        fr.read_bytes("photo.png", b"\x89PNG")


def test_byte_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fr, "MAX_BYTES", 8)
    with pytest.raises(fr.ReadError, match="文件太大"):
        fr.read_bytes("big.txt", b"0123456789")


def test_char_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fr, "MAX_CHARS", 4)
    with pytest.raises(fr.ReadError, match="正文太长"):
        fr.read_bytes("long.txt", b"0123456789")


def test_missing_path_is_reported() -> None:
    with pytest.raises(fr.ReadError, match="读不到这个文件"):
        fr.read_path(FIX / "nope-does-not-exist.docx")


def test_directory_is_rejected(tmp_path: Path) -> None:
    d = tmp_path / "folder.txt"
    d.mkdir()
    with pytest.raises(fr.ReadError, match="文件夹"):
        fr.read_path(d)

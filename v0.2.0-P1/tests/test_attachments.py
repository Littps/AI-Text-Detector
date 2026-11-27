"""D40 attachment table: intake, ceilings, compose order, metadata-only output."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import attachments as at

FIX = Path(__file__).resolve().parent / "fixtures"


def store() -> at.AttachmentStore:
    return at.AttachmentStore()


def blob(name: str, data: bytes) -> dict:
    return {"name": name, "data": base64.b64encode(data).decode()}


def test_import_paths_reports_metadata_only() -> None:
    s = store()
    res = s.import_paths([str(FIX / "sample.txt")])
    assert res["errors"] == []
    item = res["items"][0]
    assert item["name"] == "sample.txt" and item["chars"] > 100
    # the whole point of the table: text never leaves the sidecar
    assert "text" not in item


def test_one_bad_file_does_not_sink_the_batch() -> None:
    s = store()
    res = s.import_paths([str(FIX / "sample.txt"), str(FIX / "blank.pdf")])
    assert len(res["items"]) == 1
    assert len(res["errors"]) == 1
    assert "文字层" in res["errors"][0]["error"]
    assert res["errors"][0]["name"] == "blank.pdf"


def test_import_blobs_base64() -> None:
    s = store()
    res = s.import_blobs([blob("note.md", "# 标题\n正文".encode())])
    assert res["items"][0]["ext"] == "md"
    assert s.compose("", [res["items"][0]["id"]])[0] == "# 标题\n正文"


def test_bad_base64_is_per_item_error() -> None:
    s = store()
    res = s.import_blobs([{"name": "x.txt", "data": "!!!not-base64!!!"}])
    assert res["items"] == [] and "损坏" in res["errors"][0]["error"]


def test_blob_without_name_is_error() -> None:
    s = store()
    res = s.import_blobs([{"data": base64.b64encode(b"x").decode()}])
    assert res["items"] == [] and "格式" in res["errors"][0]["error"]


def test_ids_are_unique_and_sequential() -> None:
    s = store()
    a = s.import_paths([str(FIX / "sample.txt")])["items"][0]
    b = s.import_paths([str(FIX / "sample.txt")])["items"][0]
    assert a["id"] == "att-1" and b["id"] == "att-2"


def test_drop_and_list() -> None:
    s = store()
    aid = s.import_paths([str(FIX / "sample.txt")])["items"][0]["id"]
    assert s.drop(aid) is True
    assert s.drop(aid) is False
    assert s.meta() == []


def test_compose_puts_text_first_then_attachments_in_order() -> None:
    s = store()
    one = s.import_paths([str(FIX / "sample.txt")])["items"][0]["id"]
    two = s.import_blobs([blob("extra.md", "额外的尾巴".encode())])["items"][0]["id"]
    text, names = s.compose("手输的第一段", [two, one])
    assert text.startswith("手输的第一段")
    assert text.index("额外的尾巴") < text.index("第一段：这是")
    assert names == ["extra.md", "sample.txt"]


def test_compose_skips_blank_text() -> None:
    s = store()
    aid = s.import_paths([str(FIX / "sample.txt")])["items"][0]["id"]
    text, names = s.compose("   \n  ", [aid])
    assert not text.startswith("\n\n") and names == ["sample.txt"]


def test_compose_preserves_leading_whitespace_for_span_offsets() -> None:
    s = store()
    exact = "\n\n  Hello world."
    assert s.compose(exact, [])[0] == exact


def test_unknown_id_raises() -> None:
    with pytest.raises(at.UnknownAttachment):
        store().compose("x", ["att-77"])


def test_item_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(at, "MAX_ITEMS", 1)
    s = store()
    s.import_paths([str(FIX / "sample.txt")])
    with pytest.raises(ValueError, match="最多同时导入"):
        s.import_paths([str(FIX / "sample.txt")])


def test_total_char_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(at, "MAX_TOTAL_CHARS", 50)
    s = store()
    res = s.import_paths([str(FIX / "sample.txt")])
    assert res["items"] == [] and "上限" in res["errors"][0]["error"]
    assert s.meta() == []  # a rejected import must not consume the budget


def test_clear_resets_ids() -> None:
    s = store()
    s.import_paths([str(FIX / "sample.txt")])
    assert s.clear() == 1
    again = s.import_paths([str(FIX / "sample.txt")])["items"][0]
    assert again["id"] == "att-1"


def test_windows_and_posix_basenames() -> None:
    assert at._basename(r"C:\Users\<user>\报告.docx") == "报告.docx"
    assert at._basename("/home/<user>/a.pdf") == "a.pdf"

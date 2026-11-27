"""round-30: LogStore day partitioning + range export, and the WS handler.

The handler tests use the module-level ``LOGS`` (conftest points it at a
scratch dir), so they also prove ``log()`` reaches the file at runtime.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import logstore
import pytest
from logstore import KEEP_DAYS, LogStore
from server import handle_logs_export, log


@pytest.fixture()
def store(tmp_path: Path) -> LogStore:
    return LogStore(tmp_path)


def _write_day(store: LogStore, day: date, *lines: str) -> Path:
    path = store.root / f"sidecar-{day.isoformat()}.log"
    path.write_text("".join(f"2026-01-01T00:00:00 {ln}\n" for ln in lines), encoding="utf-8")
    return path


def test_write_appends_to_todays_file(store: LogStore) -> None:
    store.write("client connected: ('127.0.0.1', 5000)")
    path = store.root / f"sidecar-{date.today().isoformat()}.log"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "client connected" in text
    # every line carries an ISO timestamp so a human can correlate events
    assert text.startswith(date.today().isoformat())


def test_write_never_raises_when_dir_is_unwritable(tmp_path: Path) -> None:
    store = LogStore(tmp_path)
    store.root = tmp_path / "does-not-exist" / "deeper"
    store.write("still fine")  # must be swallowed, not raised


def test_export_today_only_covers_today(store: LogStore) -> None:
    today = date(2026, 11, 26)
    _write_day(store, today, "today-a", "today-b")
    _write_day(store, today - timedelta(days=1), "yesterday")

    res = store.export(1, today=today)
    assert res["from"] == "2026-11-26" and res["to"] == "2026-11-26"
    assert "today-a" in res["text"] and "today-b" in res["text"]
    assert "yesterday" not in res["text"]
    assert res["files"] == 1
    assert res["lines"] == 2


def test_export_three_days_is_inclusive_and_ordered(store: LogStore) -> None:
    today = date(2026, 11, 26)
    for back, tag in ((0, "d0"), (1, "d1"), (2, "d2"), (3, "d3")):
        _write_day(store, today - timedelta(days=back), tag)

    res = store.export(3, today=today)
    assert (res["from"], res["to"]) == ("2026-11-24", "2026-11-26")
    assert res["files"] == 3
    body = res["text"]
    # oldest first: the reader scrolls down to the newest events
    assert body.index("d2") < body.index("d1") < body.index("d0")
    assert "d3" not in body


def test_export_seven_days_reports_range(store: LogStore) -> None:
    today = date(2026, 11, 26)
    _write_day(store, today, "x")
    res = store.export(7, today=today)
    assert (res["from"], res["to"]) == ("2026-11-20", "2026-11-26")
    assert res["days"] == 7


def test_export_header_states_privacy_and_env(store: LogStore) -> None:
    today = date(2026, 11, 26)
    _write_day(store, today, "detect: 3 engines")
    res = store.export(1, today=today, meta={"tier": "standard", "torch": "2.9.0+cu129"})

    assert res["text"].startswith("# AI Text Detector")
    assert "不包含任何待检测文本" in res["text"]
    assert "standard" in res["text"] and "2.9.0+cu129" in res["text"]
    # the export is a bug report, not a credential leak
    assert "SIDE_TOKEN" not in res["text"]


def test_export_empty_range_still_explains_itself(store: LogStore) -> None:
    res = store.export(7)
    assert res["files"] == 0
    assert "没有任何日志" in res["text"]


def test_export_rejects_unsupported_ranges(store: LogStore) -> None:
    for bad in (0, 2, 30, -1):
        with pytest.raises(ValueError):
            store.export(bad)


def test_prune_only_touches_our_own_files(store: LogStore) -> None:
    today = date(2026, 11, 26)
    old = _write_day(store, today - timedelta(days=KEEP_DAYS), "ancient")
    fresh = _write_day(store, today - timedelta(days=KEEP_DAYS - 1), "fresh")
    foreign = store.root / "notes.txt"
    foreign.write_text("keep me", encoding="utf-8")

    removed = store.prune(today=today)
    assert old.name in removed
    assert not old.exists()
    assert fresh.exists() and foreign.exists()


def test_handler_rejects_bad_days() -> None:
    err = json.loads(handle_logs_export("m1", {"days": True}))
    assert "error" in err["payload"]
    err = json.loads(handle_logs_export("m2", {"days": "3"}))
    assert "error" in err["payload"]


def test_handler_exports_after_a_real_log_line() -> None:
    marker = "round30-test-marker"
    log(marker)
    res = json.loads(handle_logs_export("m3", {"days": 1}))
    payload = res["payload"]
    assert marker in payload["text"]
    assert payload["days"] == 1 and payload["files"] >= 1
    assert payload["bytes"] > 0


def test_default_root_honours_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AIGC_LOG_DIR", str(tmp_path / "custom"))
    assert logstore.default_root(tmp_path) == tmp_path / "custom"
    monkeypatch.delenv("AIGC_LOG_DIR")
    assert logstore.default_root(tmp_path) == tmp_path / "logs"

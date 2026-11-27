"""Unit tests for backend/server.py pure helpers.

Most cases need no event loop; the round-20 grace/reaper block drives its two
coroutines with ``asyncio.run`` instead of pulling in pytest-asyncio.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from downloader import templates_for
from server import _allowed_origins, authorized


def test_production_origins_always_present() -> None:
    origins = _allowed_origins()
    assert "tauri://localhost" in origins
    assert "http://tauri.localhost" in origins
    assert len(origins) == 2


def test_dev_extra_origins_injected() -> None:
    origins = _allowed_origins("http://localhost:1420")
    assert "http://localhost:1420" in origins
    assert "tauri://localhost" in origins  # base untouched


def test_dev_extra_multiple_and_whitespace() -> None:
    origins = _allowed_origins("http://localhost:1420, http://localhost:5173 ,")
    assert origins == {
        "tauri://localhost",
        "http://tauri.localhost",
        "http://localhost:1420",
        "http://localhost:5173",
    }


def test_dev_extra_empty_segments_ignored() -> None:
    assert _allowed_origins(",,") == {
        "tauri://localhost",
        "http://tauri.localhost",
    }


# ---- authorized(): token + origin gate (L4 constant-time compare) ------

import server  # after the pure-helper imports above


class _FakeRequest:
    def __init__(self, origin: str, token: str) -> None:
        self.headers = {"Origin": origin}
        self.path = f"/?token={token}"


class _FakeWs:
    def __init__(self, origin: str, token: str) -> None:
        self.request = _FakeRequest(origin, token)


def test_authorized_ok(monkeypatch: object) -> None:
    monkeypatch.setattr(server, "TOKEN", "s3cret", raising=False)  # type: ignore[attr-defined]
    ws = _FakeWs("tauri://localhost", "s3cret")
    assert authorized(ws) is True  # type: ignore[arg-type]


def test_authorized_rejects_wrong_token(monkeypatch: object) -> None:
    monkeypatch.setattr(server, "TOKEN", "s3cret", raising=False)  # type: ignore[attr-defined]
    ws = _FakeWs("tauri://localhost", "wrong")
    assert authorized(ws) is False  # type: ignore[arg-type]


def test_authorized_rejects_bad_origin(monkeypatch: object) -> None:
    monkeypatch.setattr(server, "TOKEN", "s3cret", raising=False)  # type: ignore[attr-defined]
    ws = _FakeWs("http://evil.example", "s3cret")
    assert authorized(ws) is False  # type: ignore[arg-type]


def test_authorized_fails_closed_without_token(monkeypatch: object) -> None:
    monkeypatch.setattr(server, "TOKEN", "", raising=False)  # type: ignore[attr-defined]
    ws = _FakeWs("tauri://localhost", "")
    assert authorized(ws) is False  # type: ignore[arg-type]


# ---- handle_detect: D29 streaming partials (pure-function scope) --------

from engines.base import Segment
from engines.loader import EngineLoader
from engines.registry import EngineSpec
from fakes import FakeEngine


def _detect_loader():
    spec = EngineSpec(
        engine_id="fake-a", name="A", lang="en", kind="classifier", est_vram_mb=10, repo="-"
    )
    return EngineLoader(
        specs=(spec,),
        engines={
            "fake-a": FakeEngine(
                "fake-a",
                batches=[[Segment(0, 4, 0.9), Segment(4, 2, 0.1)], [Segment(6, 3, 0.5)]],
            )
        },
        budget_mb=100,
    )


def test_handle_detect_streams_partials(monkeypatch) -> None:
    monkeypatch.setattr(server, "LOADER", _detect_loader())
    parts: list[dict] = []
    reply = server.handle_detect(
        "m1", {"text": "abcdefghi", "engines": ["fake-a"]}, parts.append
    )
    assert len(parts) == 2  # two batches streamed
    assert parts[0]["engine_id"] == "fake-a" and parts[0]["kind"] == "detect_part"
    assert [s["offset"] for s in parts[0]["segments"]] == [0, 4]
    assert [s["offset"] for s in parts[1]["segments"]] == [6]
    payload = json.loads(reply)["payload"]
    r = payload["results"][0]
    assert len(r["segments"]) == 3  # reply still carries the full set
    assert r["doc_score"] == round((0.9 * 4 + 0.1 * 2 + 0.5 * 3) / 9, 4)
    # D40: a pure-text run keeps the old contract (raw text never travels back)
    assert payload["input"] == {"chars": 9, "attachments": []}


def test_handle_detect_without_emit_has_no_partials(monkeypatch) -> None:
    monkeypatch.setattr(server, "LOADER", _detect_loader())
    parts: list[dict] = []
    reply = server.handle_detect("m2", {"text": "abcdefghi", "engines": ["fake-a"]})
    assert parts == []
    payload = json.loads(reply)["payload"]
    assert payload["results"][0]["status"] == "ok"


def test_handle_detect_validates_payload(monkeypatch) -> None:
    monkeypatch.setattr(server, "LOADER", _detect_loader())
    assert "error" in json.loads(server.handle_detect("m3", {"text": "", "engines": ["fake-a"]}))["payload"]
    assert "error" in json.loads(server.handle_detect("m4", {"text": "abc", "engines": []}))["payload"]


def test_handle_detect_rejects_duplicate_attachment_ids(monkeypatch) -> None:
    _fresh_store(monkeypatch)
    store = server.ATTACHMENTS
    aid = store.import_blobs([_txt_blob("a.txt", "正文")])["items"][0]["id"]
    reply = json.loads(
        server.handle_detect(
            "m-dup",
            {"text": "", "engines": ["fake-a"], "attachments": [aid, aid]},
        )
    )
    assert "unique" in reply["payload"]["error"]


def test_handle_detect_rejects_composed_limit(monkeypatch) -> None:
    monkeypatch.setattr(server, "LOADER", _detect_loader())
    monkeypatch.setattr(server, "MAX_TOTAL_CHARS", 3)
    reply = json.loads(
        server.handle_detect(
            "m-limit",
            {"text": "abcd", "engines": ["fake-a"]},
        )
    )
    assert "character limit" in reply["payload"]["error"]


# ---- D36 download source forwarding ------------------------------------

class _StubDownloads:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str | None]] = []

    def start(self, ids: list[str], source: str | None = None) -> dict:
        templates_for(source or "official")  # the real manager's validation
        self.calls.append((list(ids), source))
        return {"started": ids, "skipped": []}

    def attach(self, emit: object) -> None:  # pragma: no cover - signature mirror
        pass


def test_download_start_forwards_source(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubDownloads()
    monkeypatch.setattr(server, "DOWNLOADS", stub)
    res = json.loads(server.handle_download_start("m1", {"engines": ["a"], "source": "mirror"}))
    assert res["payload"]["started"] == ["a"]
    assert stub.calls == [(["a"], "mirror")]


def test_download_start_without_source_keeps_default(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubDownloads()
    monkeypatch.setattr(server, "DOWNLOADS", stub)
    json.loads(server.handle_download_start("m2", {"engines": ["a"]}))
    assert stub.calls == [(["a"], None)]  # manager falls back to the D18 default


def test_download_start_rejects_unknown_source(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubDownloads()
    monkeypatch.setattr(server, "DOWNLOADS", stub)
    res = json.loads(server.handle_download_start("m3", {"engines": ["a"], "source": "auto"}))
    assert "unknown download source" in res["payload"]["error"]
    assert stub.calls == []


def test_download_start_rejects_non_string_source(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubDownloads()
    monkeypatch.setattr(server, "DOWNLOADS", stub)
    res = json.loads(server.handle_download_start("m4", {"engines": ["a"], "source": 3}))
    assert "source must be a string" in res["payload"]["error"]
    assert stub.calls == []


# ---- D40 file import + detect attachments -------------------------------

import base64 as _b64

import attachments as _att

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _fresh_store(monkeypatch: pytest.MonkeyPatch) -> _att.AttachmentStore:
    s = _att.AttachmentStore()
    monkeypatch.setattr(server, "ATTACHMENTS", s)
    return s


def _txt_blob(name: str, body: str) -> dict:
    return {"name": name, "data": _b64.b64encode(body.encode()).decode()}


def test_file_import_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    _fresh_store(monkeypatch)
    res = json.loads(server.handle_file_import("f1", {"paths": [str(FIXTURES / "sample.txt")]}))
    payload = res["payload"]
    assert len(payload["items"]) == 1 and payload["errors"] == []
    assert payload["items"][0]["chars"] > 100
    assert payload["all"] == payload["items"]  # nothing hidden from the UI


def test_file_import_blobs_browser_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _fresh_store(monkeypatch)
    res = json.loads(
        server.handle_file_import("f2", {"blobs": [_txt_blob("手输.md", "# 标题\n正文")]})
    )
    assert res["payload"]["items"][0]["name"] == "手输.md"


def test_file_import_rejects_both_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    _fresh_store(monkeypatch)
    res = json.loads(server.handle_file_import("f3", {"paths": ["a"], "blobs": []}))
    assert "not both" in res["payload"]["error"]


def test_file_import_requires_one_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    _fresh_store(monkeypatch)
    res = json.loads(server.handle_file_import("f4", {}))
    assert "required" in res["payload"]["error"]


def test_file_import_rejects_non_list_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    _fresh_store(monkeypatch)
    res = json.loads(server.handle_file_import("f5", {"paths": "C:/a.txt"}))
    assert "list of strings" in res["payload"]["error"]


def test_file_drop_and_list(monkeypatch: pytest.MonkeyPatch) -> None:
    _fresh_store(monkeypatch)
    imp = json.loads(server.handle_file_import("f6", {"blobs": [_txt_blob("a.txt", "内容")]}))
    aid = imp["payload"]["items"][0]["id"]
    listed = json.loads(server.handle_file_list("f7", {}))
    assert [i["id"] for i in listed["payload"]["items"]] == [aid]
    dropped = json.loads(server.handle_file_drop("f8", {"id": aid}))
    assert dropped["payload"]["dropped"] is True and dropped["payload"]["all"] == []
    assert json.loads(server.handle_file_drop("f9", {"id": aid}))["payload"]["dropped"] is False


def test_file_drop_rejects_bad_id(monkeypatch: pytest.MonkeyPatch) -> None:
    _fresh_store(monkeypatch)
    res = json.loads(server.handle_file_drop("f10", {"id": 3}))
    assert "non-empty string" in res["payload"]["error"]


def test_detect_attachment_only(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _fresh_store(monkeypatch)
    monkeypatch.setattr(server, "LOADER", _detect_loader())
    aid = store.import_blobs([_txt_blob("报告.txt", "abcdefghi")])["items"][0]["id"]
    reply = json.loads(server.handle_detect("d1", {"text": "", "engines": ["fake-a"], "attachments": [aid]}))
    payload = reply["payload"]
    assert payload["results"][0]["status"] == "ok"
    # with attachments the composed body comes back once (the UI has no copy)
    assert payload["input"] == {"chars": 9, "attachments": ["报告.txt"], "text": "abcdefghi"}


def test_detect_text_plus_attachments_is_one_document(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _fresh_store(monkeypatch)
    monkeypatch.setattr(server, "LOADER", _detect_loader())
    aid = store.import_blobs([_txt_blob("b.txt", "abcdefghi")])["items"][0]["id"]
    reply = json.loads(server.handle_detect("d2", {"text": "abc", "engines": ["fake-a"], "attachments": [aid]}))
    # "abc" + "\n\n" + "abcdefghi"
    assert reply["payload"]["input"]["chars"] == 3 + 2 + 9


def test_detect_unknown_attachment(monkeypatch: pytest.MonkeyPatch) -> None:
    _fresh_store(monkeypatch)
    monkeypatch.setattr(server, "LOADER", _detect_loader())
    reply = json.loads(server.handle_detect("d3", {"text": "abc", "engines": ["fake-a"], "attachments": ["att-99"]}))
    assert "unknown attachment" in reply["payload"]["error"]


def test_detect_rejects_bad_attachment_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    _fresh_store(monkeypatch)
    monkeypatch.setattr(server, "LOADER", _detect_loader())
    bad_list = json.loads(server.handle_detect("d4", {"text": "abc", "engines": ["fake-a"], "attachments": "att-1"}))
    assert "list of ids" in bad_list["payload"]["error"]
    bad_item = json.loads(server.handle_detect("d5", {"text": "abc", "engines": ["fake-a"], "attachments": [3]}))
    assert "list of ids" in bad_item["payload"]["error"]
    bad_text = json.loads(server.handle_detect("d6", {"text": 7, "engines": ["fake-a"]}))
    assert "text must be a string" in bad_text["payload"]["error"]


def test_detect_empty_everything_still_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _fresh_store(monkeypatch)
    monkeypatch.setattr(server, "LOADER", _detect_loader())
    reply = json.loads(server.handle_detect("d7", {"text": "  ", "engines": ["fake-a"], "attachments": []}))
    assert "must be non-empty" in reply["payload"]["error"]


# ---- round-19: detect lifecycle registry --------------------------------

from detectjob import DetectCancelled, DetectJob


def _fresh_job(monkeypatch: pytest.MonkeyPatch) -> DetectJob:
    job = DetectJob()
    monkeypatch.setattr(server, "DETECT", job)
    return job


def test_detect_refuses_a_second_run_while_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    """One GPU, one run: queueing behind the lock would look like a hang."""
    _fresh_job(monkeypatch)
    monkeypatch.setattr(server, "LOADER", _detect_loader())
    assert server.DETECT.begin(["fake-a"]) is not None
    reply = json.loads(server.handle_detect("m1", {"text": "abc", "engines": ["fake-a"]}))
    assert "already in progress" in reply["payload"]["error"]
    server.DETECT.finish("cancelled")


def test_detect_cancel_stops_between_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    job = _fresh_job(monkeypatch)
    monkeypatch.setattr(server, "LOADER", _detect_loader())
    seen: list[dict] = []

    def emit(part: dict) -> None:
        seen.append(part)
        job.cancel("test")  # ask to stop after the first batch

    reply = json.loads(
        server.handle_detect("m2", {"text": "abcdefghi", "engines": ["fake-a"]}, emit)
    )
    payload = reply["payload"]
    assert payload["cancelled"] is True
    assert payload["job"]["state"] == "cancelled"
    assert payload["job"]["reason"] == "test"
    assert len(seen) == 1  # second batch never ran
    # the slot is free again: a cancelled run must not wedge the app
    assert job.begin(["fake-a"]) is not None


def test_detect_status_reports_foreign_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reloaded page owns no request - status is its only window into the job."""
    job = _fresh_job(monkeypatch)
    job.begin(["fake-a", "fake-b"], chars=4096)
    payload = json.loads(server.handle_detect_status("m1", {}))["payload"]["job"]
    assert payload["state"] == "running"
    assert payload["engines"] == ["fake-a", "fake-b"]
    assert payload["chars"] == 4096
    assert "text" not in payload  # never echoes document content
    job.finish("done")


def test_detect_cancel_handler_requests_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    job = _fresh_job(monkeypatch)
    idle = json.loads(server.handle_detect_cancel("m0", {}))["payload"]
    assert idle["cancel_requested"] is False
    job.begin(["fake-a"])
    res = json.loads(server.handle_detect_cancel("m1", {}))["payload"]
    assert res["cancel_requested"] is True
    assert res["job"]["cancel_requested"] is True
    with pytest.raises(DetectCancelled):
        job.check()


def test_socket_open_treats_a_closed_peer_as_free() -> None:
    """A reload opens the new socket before the old one is reaped."""

    class _State:
        def __init__(self, name: str) -> None:
            self.name = name

    class _Sock:
        def __init__(self, name: str | None, close_code: int | None = None) -> None:
            if name is not None:
                self.state = _State(name)
            self.close_code = close_code

    assert server._socket_open(_Sock("OPEN")) is True
    assert server._socket_open(_Sock("CLOSED", 1000)) is False
    assert server._socket_open(_Sock(None)) is True  # legacy shape, still open
    assert server._socket_open(_Sock(None, 1006)) is False


# ---- round-28: a fresh authorized socket takes the single-client slot ----
#
# The old rule answered 4002 while the previous socket was still OPEN. One
# leaked socket (HMR / a page that vanished without FIN) then made every
# retry fail forever; the UI sat on "closed". Newest connection wins now.


class _FakeSocket:
    def __init__(self, port: int) -> None:
        self.remote_address = ("127.0.0.1", port)
        self.closed: tuple[int | None, str | None] | None = None

    async def close(self, code: int | None = None, reason: str | None = None) -> None:
        self.closed = (code, reason)


def test_claim_slot_replaces_an_open_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    old, new = _FakeSocket(50001), _FakeSocket(50002)
    monkeypatch.setattr(server, "ACTIVE_WS", old)
    asyncio.run(server._claim_slot(new))
    assert server.ACTIVE_WS is new
    # 4003, not 4002: the old page is told it was replaced, not "unauthorized"
    assert old.closed == (4003, "replaced by newer connection")


def test_claim_slot_is_idempotent_for_the_same_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = _FakeSocket(50003)
    monkeypatch.setattr(server, "ACTIVE_WS", ws)
    asyncio.run(server._claim_slot(ws))
    assert server.ACTIVE_WS is ws
    assert ws.closed is None


def test_claim_slot_fills_an_empty_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = _FakeSocket(50004)
    monkeypatch.setattr(server, "ACTIVE_WS", None)
    asyncio.run(server._claim_slot(ws))
    assert server.ACTIVE_WS is ws
    assert ws.closed is None


# ---- round-20: grace window + parked reply ------------------------------
#
# These are the only async tests in the file: the reaper's whole job is a
# delayed decision, and the reply parking lives in the detect task. Each one
# swaps DETECT_LOCK for a fresh lock so asyncio.run() can bind its own loop.


def test_detect_result_returns_the_parked_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    job = _fresh_job(monkeypatch)
    monkeypatch.setattr(server, "LOADER", _detect_loader())
    reply = server.handle_detect("m-old", {"text": "abcdefghi", "engines": ["fake-a"]})
    job.stash_result(reply, "abcdefghi")
    payload = json.loads(server.handle_detect_result("m-new", {}))["payload"]
    assert payload["results"][0]["status"] == "ok"
    assert payload["results"][0]["doc_score"] == 0.5889
    assert payload["input"]["text"] == "abcdefghi"
    assert payload["input"]["chars"] == 9
    # consumed: a second tab cannot render the same run
    again = json.loads(server.handle_detect_result("m-new2", {}))["payload"]
    assert "no detect result is waiting" in again["error"]


def test_detect_result_empty_slot_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _fresh_job(monkeypatch)
    payload = json.loads(server.handle_detect_result("m1", {}))["payload"]
    assert "no detect result is waiting" in payload["error"]


def test_detect_result_keeps_error_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    job = _fresh_job(monkeypatch)
    job.begin(["fake-a"])
    job.finish("error", detail="boom")
    job.stash_result(server.protocol.err("m-old", "detect failed: boom"))
    payload = json.loads(server.handle_detect_result("m1", {}))["payload"]
    assert "detect failed: boom" in payload["error"]


def test_detect_result_rejects_unreadable(monkeypatch: pytest.MonkeyPatch) -> None:
    job = _fresh_job(monkeypatch)
    job.begin(["fake-a"])
    job.finish("done")
    job.stash_result("not json at all")
    payload = json.loads(server.handle_detect_result("m1", {}))["payload"]
    assert "unreadable" in payload["error"]


def test_detect_result_without_input_block_rebuilds_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pure-text reply has no input.text; the parked text must fill it in."""
    job = _fresh_job(monkeypatch)
    job.begin(["fake-a"])
    job.finish("done")
    envelope = server.protocol.ok("m-old", {"results": [], "input": {"chars": 5, "attachments": []}})
    job.stash_result(envelope, "五个字啊")  # 4 chars, count is the caller's problem
    payload = json.loads(server.handle_detect_result("m1", {}))["payload"]
    assert payload["input"]["text"] == "五个字啊"


def test_run_detect_parks_the_reply_when_the_socket_is_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fresh_job(monkeypatch)
    monkeypatch.setattr(server, "LOADER", _detect_loader())
    monkeypatch.setattr(server, "DETECT_LOCK", asyncio.Lock())

    class _DeadWs:
        class state:  # mimics websockets' enum shape
            name = "OPEN"

        async def send(self, raw: str) -> None:
            raise OSError("socket is gone")

    asyncio.run(
        server._run_detect(
            _DeadWs(), "m1", {"text": "abcdefghi", "engines": ["fake-a"]}
        )
    )
    assert server.DETECT.has_result() is True
    assert server.DETECT.snapshot()["state"] == "done"
    payload = json.loads(server.handle_detect_result("m2", {}))["payload"]
    assert payload["results"][0]["status"] == "ok"
    assert payload["input"]["text"] == "abcdefghi"


def test_reap_cancels_when_nobody_takes_over(monkeypatch: pytest.MonkeyPatch) -> None:
    job = _fresh_job(monkeypatch)
    monkeypatch.setattr(server, "ACTIVE_WS", None)
    monkeypatch.setattr(server, "DETECT_GRACE", 0.0)
    jid = job.begin(["fake-a"])
    asyncio.run(server._reap_orphan(jid))
    # cancel is cooperative: the flag is set now, the worker turns it into a
    # CANCELLED state at its next batch boundary (a forward pass cannot be cut)
    snap = job.snapshot()
    assert snap["cancel_requested"] is True
    assert snap["reason"] == "disconnect"
    with pytest.raises(DetectCancelled):
        job.check()


def test_reap_keeps_the_run_when_a_page_took_over(monkeypatch: pytest.MonkeyPatch) -> None:
    job = _fresh_job(monkeypatch)
    monkeypatch.setattr(server, "ACTIVE_WS", object())
    monkeypatch.setattr(server, "DETECT_GRACE", 0.0)
    jid = job.begin(["fake-a"])
    asyncio.run(server._reap_orphan(jid))
    assert job.is_running() is True


def test_reap_leaves_a_newer_job_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    job = _fresh_job(monkeypatch)
    monkeypatch.setattr(server, "ACTIVE_WS", None)
    monkeypatch.setattr(server, "DETECT_GRACE", 0.0)
    old = job.begin(["fake-a"])
    job.finish("done")
    new = job.begin(["fake-b"])
    assert new != old
    asyncio.run(server._reap_orphan(old))
    assert job.is_running() is True  # the reaper must not kill the new run


def test_parkable_only_accepts_real_results() -> None:
    """A tombstone is not a result: status already explains why a run ended."""
    ok = server.protocol.ok("m", {"results": [{"engine": "x"}], "input": {}})
    assert server._parkable(ok) is True
    cancelled = server.protocol.ok("m", {"cancelled": True, "job": {"state": "cancelled"}})
    assert server._parkable(cancelled) is False
    assert server._parkable(server.protocol.err("m", "detect failed")) is False
    assert server._parkable("not json") is False
    assert server._parkable(server.protocol.ok("m", {"results": []})) is False


def test_runtime_status_reply_separates_ws_and_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server, "torch_info", lambda: {"torch_installed": False})
    reply = json.loads(server.handle_runtime_status("m-runtime", {}))
    payload = reply["payload"]
    assert payload["local_service_ready"] is True
    assert payload["inference_ready"] is False
    assert any(item["id"] == "websockets" for item in payload["components"])
    assert any(item["name"] == "msvcp140.dll" for item in payload["dlls"])


def test_uninstall_is_rejected_while_detect_is_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(server.DETECT, "is_running", lambda: True)
    reply = asyncio.run(
        server.handle_storage_uninstall(
            "m-uninstall",
            {"engine_id": server.MANIFEST.ids()[0]},
        )
    )
    data = json.loads(reply)
    assert "detect run is in progress" in data["payload"]["error"]


def test_uninstall_is_rejected_while_detect_task_is_queued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _PendingTask:
        def done(self) -> bool:
            return False

    monkeypatch.setattr(server.DETECT, "is_running", lambda: False)
    monkeypatch.setattr(server, "_DETECT_TASKS", {_PendingTask()})
    reply = asyncio.run(
        server.handle_storage_uninstall(
            "m-uninstall",
            {"engine_id": server.MANIFEST.ids()[0]},
        )
    )
    data = json.loads(reply)
    assert "detect run is in progress" in data["payload"]["error"]

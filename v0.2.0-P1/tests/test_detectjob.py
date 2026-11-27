"""Unit tests for the round-19 single-flight detect registry."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from detectjob import CANCELLED, DONE, ERROR, IDLE, RUNNING, DetectCancelled, DetectJob


def test_fresh_job_is_idle() -> None:
    job = DetectJob()
    snap = job.snapshot()
    assert snap["state"] == IDLE
    assert snap["job_id"] == 0
    assert snap["engines"] == []
    assert snap["cancel_requested"] is False
    assert snap["elapsed_s"] == 0.0


def test_begin_claims_the_slot_once() -> None:
    job = DetectJob()
    first = job.begin(["a", "b"], chars=12)
    assert first == 1
    assert job.snapshot()["state"] == RUNNING
    # a second request while the GPU is busy must be refused, not queued
    assert job.begin(["c"]) is None
    job.finish(DONE)
    assert job.begin(["c"]) == 2  # slot frees up again


def test_snapshot_carries_counts_not_text() -> None:
    job = DetectJob()
    job.begin(["fake-a"], chars=4096)
    job.engine_started("fake-a")
    snap = job.snapshot()
    assert snap["chars"] == 4096
    assert snap["current"] == "fake-a"
    assert snap["engines"] == ["fake-a"]
    # the registry keeps no document body, ever
    assert not any(isinstance(v, str) and len(v) > 64 for v in snap.values())


def test_cancel_marks_reason_and_check_raises() -> None:
    job = DetectJob()
    job.begin(["a"])
    assert job.snapshot()["cancel_requested"] is False
    job.check()  # no-op while running normally
    assert job.cancel("disconnect") is True
    assert job.snapshot()["reason"] == "disconnect"
    with pytest.raises(DetectCancelled, match="disconnect"):
        job.check()


def test_cancel_of_idle_job_is_refused() -> None:
    job = DetectJob()
    assert job.cancel("disconnect") is False
    assert job.snapshot()["state"] == IDLE


def test_finish_freezes_state_and_elapsed() -> None:
    job = DetectJob()
    job.begin(["a"])
    job.finish(DONE)
    snap = job.snapshot()
    assert snap["state"] == DONE
    assert snap["finished_at"] is not None
    assert snap["elapsed_s"] >= 0.0


def test_cancelled_run_never_reports_done() -> None:
    """A run whose socket died is not "done" even if the last batch returned."""
    job = DetectJob()
    job.begin(["a"])
    job.cancel("disconnect")
    job.finish(DONE)
    snap = job.snapshot()
    assert snap["state"] == CANCELLED
    assert snap["reason"] == "disconnect"


def test_finish_is_idempotent() -> None:
    job = DetectJob()
    job.begin(["a"])
    job.finish(DONE)
    job.finish(ERROR, detail="too late")
    assert job.snapshot()["state"] == DONE  # first finish wins
    assert job.snapshot()["detail"] is None


def test_next_begin_clears_a_stale_cancel_flag() -> None:
    job = DetectJob()
    job.begin(["a"])
    job.cancel("client")
    job.finish(CANCELLED)
    job.begin(["b"])
    job.check()  # must not raise: the new run was never cancelled
    assert job.snapshot()["reason"] is None


# ---- round-20: cross-connection result slot ------------------------------


def test_result_slot_roundtrip_and_single_reader() -> None:
    job = DetectJob()
    job.begin(["a"])
    job.finish(DONE)
    assert job.has_result() is False
    job.stash_result('{"v":1}', "正文")
    assert job.has_result() is True
    assert job.snapshot()["has_result"] is True
    assert job.take_result() == ('{"v":1}', "正文")
    # one reader wins: a second tab must not render the same run again
    assert job.take_result() is None
    assert job.snapshot()["has_result"] is False


def test_result_slot_is_cleared_by_the_next_run() -> None:
    job = DetectJob()
    job.begin(["a"])
    job.finish(DONE)
    job.stash_result("old", "旧文")
    job.begin(["b"])
    assert job.has_result() is False
    assert job.take_result() is None


def test_result_slot_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    import detectjob

    job = DetectJob()
    job.begin(["a"])
    job.finish(DONE)
    job.stash_result("reply", "正文")
    monkeypatch.setattr(detectjob, "RESULT_TTL", -1.0)  # already past the window
    assert job.has_result() is False
    assert job.take_result() is None
    assert job.snapshot()["has_result"] is False


def test_result_slot_survives_a_finished_job() -> None:
    """The parked answer belongs to a job that is already done/cancelled."""
    job = DetectJob()
    job.begin(["a"])
    job.cancel("disconnect")
    job.finish(DONE)  # widened into CANCELLED by the flag
    job.stash_result('{"cancelled":true}', None)
    assert job.snapshot()["state"] == CANCELLED
    assert job.has_result() is True
    assert job.take_result() == ('{"cancelled":true}', None)


def test_readers_report_id_and_running() -> None:
    job = DetectJob()
    assert job.job_id() == 0
    assert job.is_running() is False
    jid = job.begin(["a"])
    assert job.job_id() == jid
    assert job.is_running() is True
    job.finish(DONE)
    assert job.is_running() is False


def test_result_slot_can_be_bound_to_a_job_id() -> None:
    job = DetectJob()
    first = job.begin(["a"])
    job.finish(DONE)
    job.stash_result("first-reply", "first-text", first)
    assert job.take_result(first + 1) is None  # wrong reader must not consume it
    assert job.take_result(first) == ("first-reply", "first-text")

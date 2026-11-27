"""Single-flight detect job registry (round-19).

``detect`` used to live entirely inside the WS handler: the reply was the only
copy of the result. A webview reload killed the socket while the torch worker
kept burning the GPU with nobody left to read the answer, and the fresh page
had no way to even learn that a run was in flight. This registry closes both
holes:

* cooperative cancel - workers call ``check()`` between sentence batches and
  between engines; the connection's close watcher sets the flag as soon as
  the socket dies (the message loop itself may be parked inside ``to_thread``
  for the whole run and cannot notice);
* ``snapshot()`` - a reconnected page asks ``detect/status`` and can tell the
  user what happened (running / cancelled by reload / last error).

The registry never stores document text while a run is live: engine ids, a
char count and timings only. "Raw text never travels back" still holds - with
two documented exceptions:

* the attachment body handed back by detect (D40);
* the pasted text parked in the cross-connection result slot below - it only
  exists when the reply could not be delivered at all, and it is wiped the
  moment a page reads it, the next run begins, or ``RESULT_TTL`` elapses.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Iterable

from engines.base import DetectCancelled

IDLE = "idle"
RUNNING = "running"
DONE = "done"
ERROR = "error"
CANCELLED = "cancelled"

# How long an undelivered reply waits for a page that reloaded mid-run. Long
# enough for a cold vite reload, short enough that a closed app does not leave
# a document sitting in memory for the rest of the session.
RESULT_TTL = 600.0


class DetectJob:
    """State of the one detect run a single-window app can have in flight."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._job_id = 0
        self._state = IDLE
        self._engines: tuple[str, ...] = ()
        self._current: str | None = None
        self._chars = 0
        self._started_at: float | None = None
        self._started_mono: float | None = None
        self._finished_at: float | None = None
        self._elapsed: float | None = None
        self._reason: str | None = None
        self._detail: str | None = None
        # cross-connection result slot (graceful-takeover, round-20): a reply
        # whose socket died before it could be sent, plus the pasted text that
        # reply was computed from. Both are cleared on read / new run / TTL.
        self._reply: str | None = None
        self._reply_text: str | None = None
        self._reply_at: float | None = None
        self._reply_job_id: int | None = None

    # -- worker side -----------------------------------------------------

    def begin(self, engine_ids: Iterable[str], chars: int = 0) -> int | None:
        """Claim the slot; returns the job id, or None when one is running."""
        with self._lock:
            if self._state == RUNNING:
                return None
            self._job_id += 1
            self._state = RUNNING
            self._engines = tuple(engine_ids)
            self._current = None
            self._chars = int(chars)
            self._started_at = time.time()
            self._started_mono = time.monotonic()
            self._finished_at = None
            self._elapsed = None
            self._reason = None
            self._detail = None
            # a new run invalidates any result the last one parked
            self._reply = None
            self._reply_text = None
            self._reply_at = None
            self._reply_job_id = None
            # inside the lock: a cancel racing a begin must not leak into the
            # fresh run (that would make every run abort on its first check)
            self._cancel.clear()
            return self._job_id

    def engine_started(self, engine_id: str) -> None:
        with self._lock:
            if self._state == RUNNING:
                self._current = engine_id

    def check(self) -> None:
        """Abort the caller when the job was cancelled."""
        if not self._cancel.is_set():
            return
        with self._lock:
            reason = self._reason or "cancelled"
        raise DetectCancelled(reason)

    def cancel(self, reason: str) -> bool:
        """Request cancellation; True when a running job took the signal."""
        with self._lock:
            if self._state != RUNNING:
                return False
            if self._reason is None:
                self._reason = reason
            self._cancel.set()
            return True

    def finish(self, state: str, detail: str | None = None) -> str:
        """Close the current job (idempotent - the first finish wins)."""
        with self._lock:
            if self._state != RUNNING:
                return self._state
            # A run cancelled mid-flight is not "done" even when the last
            # batch happened to return: nobody is waiting for that answer.
            if state == DONE and self._cancel.is_set():
                state = CANCELLED
            self._state = state
            self._detail = detail
            self._finished_at = time.time()
            if self._started_mono is not None:
                self._elapsed = round(time.monotonic() - self._started_mono, 3)
            self._current = None
            return self._state

    # -- result hand-off (graceful takeover) -------------------------------

    def stash_result(
        self,
        reply: str,
        text: str | None = None,
        job_id: int | None = None,
    ) -> None:
        """Park a reply whose socket died before it could be written.

        A reload can kill the requesting socket while the GPU keeps working;
        with a grace window the run survives, so its answer must too. The next
        page reads it through ``take_result()`` - one reader, then gone.
        """
        with self._lock:
            self._reply = reply
            self._reply_text = text
            self._reply_at = time.time()
            self._reply_job_id = self._job_id if job_id is None else job_id

    def has_result(self) -> bool:
        with self._lock:
            self._expire_reply_locked()
            return self._reply is not None

    def take_result(
        self,
        expected_job_id: int | None = None,
    ) -> tuple[str, str | None] | None:
        """Consume the parked reply; None when absent or past ``RESULT_TTL``."""
        with self._lock:
            self._expire_reply_locked()
            if self._reply is None:
                return None
            if expected_job_id is not None and self._reply_job_id != expected_job_id:
                return None
            parked = (self._reply, self._reply_text)
            self._reply = None
            self._reply_text = None
            self._reply_at = None
            self._reply_job_id = None
            return parked

    def _expire_reply_locked(self) -> None:
        """Drop a stale answer so a closed window cannot hold text forever."""
        if self._reply is None or self._reply_at is None:
            return
        if time.time() - self._reply_at > RESULT_TTL:
            self._reply = None
            self._reply_text = None
            self._reply_at = None
            self._reply_job_id = None

    # -- small readers -----------------------------------------------------

    def job_id(self) -> int:
        with self._lock:
            return self._job_id

    def is_running(self) -> bool:
        with self._lock:
            return self._state == RUNNING

    # -- client side -----------------------------------------------------

    def snapshot(self) -> dict:
        """Wire-safe view: no text, no engine objects, plain JSON types."""
        with self._lock:
            self._expire_reply_locked()
            if self._state == RUNNING and self._started_mono is not None:
                elapsed = round(time.monotonic() - self._started_mono, 3)
            else:
                elapsed = self._elapsed or 0.0
            return {
                "state": self._state,
                "job_id": self._job_id,
                "engines": list(self._engines),
                "current": self._current,
                "chars": self._chars,
                "started_at": self._started_at,
                "finished_at": self._finished_at,
                "elapsed_s": elapsed,
                "cancel_requested": self._cancel.is_set(),
                "reason": self._reason,
                "detail": self._detail,
                # true while a finished reply waits for a page that reloaded
                # mid-run; the UI fetches it with ``detect/result``
                "has_result": self._reply is not None,
                "result_job_id": self._reply_job_id,
            }

"""tests for backend/downloader.py - five iron rules against a mock HTTP server.

Covers PLAN P2 acceptance verbatim: resume-after-kill, 403 auto-switch,
SHA256-mismatch rejection - plus cancel/resume, restart-when-no-Range,
short-stream and disk precheck.
"""
from __future__ import annotations

import hashlib
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

import downloader
import pytest
from downloader import (
    ChecksumError,
    DownloadCancelled,
    DownloadError,
    FileDownloader,
    InsufficientDiskError,
    incomplete_path,
    missing_files,
    verify_file_integrity,
)
from manifest import RemoteFile

DATA = bytes(range(256)) * 8192  # 2 MiB deterministic payload
SHA = hashlib.sha256(DATA).hexdigest()
CHUNK = 256 * 1024


class MockServer:
    def __init__(self) -> None:
        self.mode = "normal"  # normal|no_range|fail_prefix|corrupt|shortstream
        self.requests: list[tuple[str, int]] = []  # (range-header-or-"403", served)

    def reset(self) -> None:
        self.requests = []


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        mock: MockServer = self.server.mock  # type: ignore[attr-defined]
        if mock.mode == "fail_prefix" and self.path.startswith("/fail/"):
            mock.requests.append(("403", 0))
            self.send_error(403)
            return
        data = DATA
        corrupt = mock.mode == "corrupt"
        rng = self.headers.get("Range")
        if rng and mock.mode not in ("no_range", "shortstream"):
            start = int(rng.split("=")[1].split("-")[0])
            body = data[start:]
            code = 206
        else:
            start = 0
            body = data
            code = 200
        if corrupt:
            body = bytes(255 - (b % 256) for b in body)
        clen = len(body)
        send_body = body
        if mock.mode == "shortstream":
            clen = len(data)  # claim full, deliver half, hang up
            send_body = body[: len(data) // 2]
        self.send_response(code)
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{len(data)-1}/{len(data)}")
        self.send_header("Content-Length", str(clen))
        self.end_headers()
        self.wfile.write(send_body)
        mock.requests.append((rng or None, len(send_body)))

    def log_message(self, *args: object) -> None:
        pass  # keep test output clean


@pytest.fixture()
def srv():
    mock = MockServer()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.mock = mock  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def tmpl(httpd: ThreadingHTTPServer, prefix: str = "") -> str:
    port = httpd.server_address[1]
    return f"http://127.0.0.1:{port}{prefix}/{{repo}}/resolve/main/{{path}}"


def make_file(dest: str = "e1/model.bin") -> RemoteFile:
    return RemoteFile(repo="test/repo", path="model.bin", dest=dest, sha256=SHA, bytes=len(DATA))


def run(httpd: ThreadingHTTPServer, tmp_path: Path, f: RemoteFile, **kw):
    fd = FileDownloader(tmp_path, templates=(tmpl(httpd, kw.pop("prefix", "")),), chunk=CHUNK, **kw)
    return fd


def test_resume_from_incomplete(srv, tmp_path) -> None:
    """Kill-mid-download recovery: only the missing half moves (rule 2)."""
    f = make_file()
    final = tmp_path / f.dest
    tmp = incomplete_path(final)
    tmp.parent.mkdir(parents=True)
    half = len(DATA) // 2
    tmp.write_bytes(DATA[:half])

    samples: list = []
    out = run(srv, tmp_path, f).download("e1", f, threading.Event(), samples.append)

    assert out == final and final.read_bytes() == DATA and not tmp.exists()
    assert srv.mock.requests[0][0] == f"bytes={half}-"  # explicit Range out
    assert srv.mock.requests[0][1] == len(DATA) - half  # only the tail moved
    done = [p.done_bytes for p in samples]
    assert done == sorted(set(done)) and done[-1] == len(DATA)  # monotonic


def test_restart_when_server_ignores_range(srv, tmp_path) -> None:
    """Plain-200 server: transparent full restart, no corrupt append."""
    f = make_file()
    tmp = incomplete_path(tmp_path / f.dest)
    tmp.parent.mkdir(parents=True)
    tmp.write_bytes(DATA[: len(DATA) // 3])
    srv.mock.mode = "no_range"

    run(srv, tmp_path, f).download("e1", f, threading.Event())
    assert (tmp_path / f.dest).read_bytes() == DATA


def test_fallback_on_403(srv, tmp_path) -> None:
    """PLAN acceptance: simulated 403 -> auto-switch to next source."""
    f = make_file()
    srv.mock.mode = "fail_prefix"
    fd = FileDownloader(
        tmp_path,
        templates=(tmpl(srv, "/fail"), tmpl(srv)),  # first host 403s
        chunk=CHUNK,
    )
    out = fd.download("e1", f, threading.Event())
    assert out.read_bytes() == DATA
    assert ("403", 0) in srv.mock.requests  # the rejected attempt is logged


def test_fallback_on_connection_refused(srv, tmp_path) -> None:
    f = make_file()
    dead = FileDownloader(
        tmp_path,
        templates=("http://127.0.0.1:1/{repo}/resolve/main/{path}", tmpl(srv)),
        chunk=CHUNK,
    )
    assert dead.download("e1", f, threading.Event()).read_bytes() == DATA


def test_templates_for_orders_preferred_host_first() -> None:
    """D36: the preference only reorders - both hosts stay in the chain."""
    assert downloader.templates_for("official") == (
        downloader.OFFICIAL_TEMPLATE,
        downloader.MIRROR_TEMPLATE,
    )
    assert downloader.templates_for("mirror") == (
        downloader.MIRROR_TEMPLATE,
        downloader.OFFICIAL_TEMPLATE,
    )
    assert downloader.REMOTE_URL_TEMPLATES == downloader.templates_for("official")
    with pytest.raises(ValueError, match="unknown download source"):
        downloader.templates_for("auto")


def test_download_templates_kwarg_leads_with_first_host(srv, tmp_path) -> None:
    """Per-job chain override wins over the instance chain (D36 plumbing)."""
    f = make_file()
    srv.mock.mode = "fail_prefix"
    fd = FileDownloader(tmp_path, templates=(tmpl(srv),), chunk=CHUNK)  # instance chain
    out = fd.download(
        "e1", f, threading.Event(), templates=(tmpl(srv, "/fail"), tmpl(srv))
    )
    assert out.read_bytes() == DATA
    assert ("403", 0) in srv.mock.requests  # the leading host was tried first


def test_sha256_mismatch_rejects(srv, tmp_path) -> None:
    """PLAN acceptance: corrupt payload -> refuse to land (rule 3)."""
    f = make_file()
    srv.mock.mode = "corrupt"
    with pytest.raises(ChecksumError):
        run(srv, tmp_path, f).download("e1", f, threading.Event())
    final = tmp_path / f.dest
    assert not final.exists() and not incomplete_path(final).exists()


def test_short_stream_rejected(srv, tmp_path) -> None:
    """Server claims full length, hangs up early -> error, no landing."""
    f = make_file()
    srv.mock.mode = "shortstream"
    with pytest.raises(DownloadError):
        run(srv, tmp_path, f).download("e1", f, threading.Event())
    final = tmp_path / f.dest
    assert not final.exists() and not incomplete_path(final).exists()


def test_cancel_keeps_incomplete_then_resumes(srv, tmp_path) -> None:
    """Cancel between chunks keeps .incomplete; next run resumes (rules 2+5)."""
    f = make_file()
    tmp = incomplete_path(tmp_path / f.dest)
    tmp.parent.mkdir(parents=True)
    half = len(DATA) // 2
    tmp.write_bytes(DATA[:half])

    cancel = threading.Event()
    cancel.set()  # cancel before first chunk
    with pytest.raises(DownloadCancelled):
        run(srv, tmp_path, f).download("e1", f, cancel)
    assert tmp.stat().st_size == half  # kept for resume

    srv.mock.reset()
    out = run(srv, tmp_path, f).download("e1", f, threading.Event())
    assert out.read_bytes() == DATA and not tmp.exists()
    assert srv.mock.requests[0][0] == f"bytes={half}-"


def test_disk_precheck_blocks_request(srv, tmp_path, monkeypatch) -> None:
    """Rule 4: free space below file size -> refuse before any request."""
    f = make_file()
    monkeypatch.setattr(downloader.shutil, "disk_usage", lambda p: SimpleNamespace(free=1))
    with pytest.raises(InsufficientDiskError):
        run(srv, tmp_path, f).download("e1", f, threading.Event())
    assert srv.mock.requests == []  # zero network activity


def test_disk_precheck_only_requests_missing_tail(srv, tmp_path, monkeypatch) -> None:
    """A resumable partial should not require space for the whole file again."""
    f = make_file()
    tmp = incomplete_path(tmp_path / f.dest)
    tmp.parent.mkdir(parents=True)
    half = len(DATA) // 2
    tmp.write_bytes(DATA[:half])
    monkeypatch.setattr(
        downloader.shutil,
        "disk_usage",
        lambda _p: SimpleNamespace(free=len(DATA) - half),
    )
    out = run(srv, tmp_path, f).download("e1", f, threading.Event())
    assert out.read_bytes() == DATA


def test_same_size_corrupt_final_is_not_trusted(srv, tmp_path) -> None:
    """A size-only check must not turn a damaged existing file into success."""
    f = make_file()
    final = tmp_path / f.dest
    final.parent.mkdir(parents=True)
    final.write_bytes(b"X" * len(DATA))
    assert [x.dest for x in missing_files(
        type("Entry", (), {"files": (f,)})(), tmp_path
    )] == [f.dest]

    srv.mock.reset()
    out = run(srv, tmp_path, f).download("e1", f, threading.Event())
    assert out == final and out.read_bytes() == DATA
    assert srv.mock.requests  # the damaged final was replaced, not skipped
    assert missing_files(type("Entry", (), {"files": (f,)})(), tmp_path) == []


def test_integrity_cache_invalidates_same_size_content_change(tmp_path: Path) -> None:
    p = tmp_path / "e1/model.bin"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"A" * 32)
    expected = hashlib.sha256(b"A" * 32).hexdigest()
    assert verify_file_integrity(tmp_path, "e1/model.bin", p, 32, expected)
    assert verify_file_integrity(tmp_path, "e1/model.bin", p, 32, expected)

    p.write_bytes(b"B" * 32)  # same size, different content
    assert not verify_file_integrity(tmp_path, "e1/model.bin", p, 32, expected)


def test_all_sources_fail_message(srv, tmp_path) -> None:
    f = make_file()
    dead = FileDownloader(
        tmp_path,
        templates=("http://127.0.0.1:1/{repo}/resolve/main/{path}",),
        chunk=CHUNK,
    )
    with pytest.raises(DownloadError, match="all sources failed"):
        dead.download("e1", f, threading.Event())


class TestOpenerProxy:
    """L1 fix: proxy=None must keep the default (env-honoring) opener chain."""

    def test_no_proxy_uses_default_chain(self, tmp_path: Path) -> None:
        fd = FileDownloader(tmp_path)
        assert [type(h) for h in fd._opener.handlers] == [
            type(h) for h in urllib.request.build_opener().handlers
        ]

    def test_explicit_proxy_overrides(self, tmp_path: Path) -> None:
        fd = FileDownloader(tmp_path, proxy="http://127.0.0.1:9")
        ph = [h for h in fd._opener.handlers if isinstance(h, urllib.request.ProxyHandler)]
        assert ph and ph[-1].proxies.get("https") == "http://127.0.0.1:9"

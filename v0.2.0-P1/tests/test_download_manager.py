"""D36 source-preference plumbing in download_manager.py (no network).

``start()`` validates the preference and records it for ``download/status``;
``_run()`` hands the one ordered chain to every file of the job and reports
the preference on ``download_engine_start`` so the UI can show which host
actually led the transfer.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import pytest
from download_manager import DownloadManager
from downloader import templates_for
from manifest import EngineEntry, Manifest, RemoteFile


def make_manifest(
    models_root: Path, *, installed: bool = False
) -> tuple[Manifest, EngineEntry, RemoteFile]:
    f = RemoteFile(repo="test/repo", path="model.bin", dest="e1/model.bin", sha256=None, bytes=4)
    entry = EngineEntry(
        engine_id="e1",
        files=(f,),
        default_selected=True,
        risk_level="none",
        risk_text_key=None,
        min_vram_mb=0,
    )
    if installed:
        p = models_root / f.dest
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"abcd")
    return Manifest({"e1": entry}, "test"), entry, f


def test_status_reports_default_source(tmp_path: Path) -> None:
    manifest, _, _ = make_manifest(tmp_path)
    assert DownloadManager(manifest, tmp_path).status()["source"] == "official"


def test_start_records_preference(tmp_path: Path) -> None:
    """Nothing to download -> no task, but the preference still shows up."""
    manifest, _, _ = make_manifest(tmp_path, installed=True)
    mgr = DownloadManager(manifest, tmp_path)
    out = mgr.start(["e1"], "mirror")
    assert out["skipped"] == [{"engine_id": "e1", "reason": "already_installed"}]
    assert mgr.status()["source"] == "mirror"


def test_start_rejects_unknown_source(tmp_path: Path) -> None:
    manifest, _, _ = make_manifest(tmp_path, installed=True)
    mgr = DownloadManager(manifest, tmp_path)
    with pytest.raises(ValueError, match="unknown download source"):
        mgr.start(["e1"], "auto")
    assert mgr.status()["source"] == "official"  # rejected before state moved


class _RecordingDownloader:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, tuple[str, ...] | None]] = []

    def download(self, engine_id, f, cancel, progress=None, *, templates=None):
        self.calls.append((engine_id, f.dest, templates))
        return Path("unused")


def test_run_hands_ordered_chain_to_every_file(tmp_path: Path) -> None:
    manifest, entry, f = make_manifest(tmp_path)
    mgr = DownloadManager(manifest, tmp_path)
    rec = _RecordingDownloader()
    mgr._fd = rec  # type: ignore[assignment]
    events: list[dict] = []
    mgr.attach(events.append)

    chain = templates_for("mirror")
    asyncio.run(mgr._run([(entry, [f])], [], "mirror", chain))

    assert rec.calls == [("e1", "e1/model.bin", chain)]
    assert [e["kind"] for e in events] == [
        "download_engine_start",
        "download_file_done",
        "download_engine_done",
    ]
    assert events[0]["source"] == "mirror"


def test_unexpected_download_exception_has_a_terminal_error_event(tmp_path: Path) -> None:
    manifest, entry, f = make_manifest(tmp_path)
    mgr = DownloadManager(manifest, tmp_path)

    class _BrokenDownloader:
        def download(self, *_args, **_kwargs):
            raise OSError("disk write denied")

    mgr._fd = _BrokenDownloader()  # type: ignore[assignment]
    events: list[dict] = []
    mgr.attach(events.append)
    asyncio.run(mgr._run([(entry, [f])], []))
    assert events[-1]["kind"] == "download_error"
    assert "OSError" in events[-1]["error"]

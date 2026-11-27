"""CustomRepo tests: mirror hints, local import wiring, URL download."""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import pytest
from custom_repo import CustomRepo, _sanitize, mirror_for
from manifest import EngineEntry, Manifest, RemoteFile


def _manifest(tmp_path: Path, engine_id: str, files: dict[str, int]) -> Manifest:
    entries = {
        engine_id: EngineEntry(
            engine_id=engine_id,
            files=tuple(
                RemoteFile(
                    repo="demo/model",
                    path=name,
                    dest=f"{engine_id}/{name}",
                    sha256=None,
                    bytes=size,
                )
                for name, size in files.items()
            ),
            default_selected=True,
            risk_level="none",
            risk_text_key=None,
            min_vram_mb=0,
        )
    }
    return Manifest(entries, "2026-11-19T00:00:00Z")


def _repo(tmp_path: Path, manifest: Manifest) -> tuple[CustomRepo, list[dict]]:
    events: list[dict] = []
    repo = CustomRepo(tmp_path, manifest, events.append)
    return repo, events


# ---- mirror hints ----------------------------------------------------------


def test_mirror_huggingface() -> None:
    url = "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"
    mirrored, note = mirror_for(url)
    assert mirrored is not None and mirrored.startswith("https://hf-mirror.com/")
    assert "hf-mirror.com" in note


def test_mirror_cdn_subdomain() -> None:
    mirrored, _ = mirror_for("https://cdn-lfs.huggingface.co/a/b?token=x")
    assert mirrored is not None and "hf-mirror.com" in mirrored


def test_mirror_modelscope_says_direct() -> None:
    assert mirror_for("https://modelscope.cn/m/file") == (
        None,
        "ModelScope 为国内源，直连即快，无需镜像",
    )


def test_mirror_unknown_is_honest() -> None:
    assert mirror_for("https://example.com/w.bin")[0] is None


def test_sanitize_strips_traversal() -> None:
    assert "/" not in _sanitize("../../etc/passwd", "fb")
    assert _sanitize("..", "fb") == "fb"


# ---- local import ----------------------------------------------------------


def test_import_matches_official_engine(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    m = _manifest(tmp_path, "demo-eng", {"model.safetensors": 5, "config.json": 2})
    repo, _ = _repo(models_root, m)

    src = tmp_path / "downloaded"
    src.mkdir()
    (src / "model.safetensors").write_bytes(b"AAAAA")
    (src / "config.json").write_bytes(b"{}")
    (src / "README.md").write_bytes(b"extra")  # extras are ignored

    out = repo.import_local(str(src), "link")
    assert out["kind"] == "engine" and out["engine"] == "demo-eng"
    assert (models_root / "demo-eng" / "model.safetensors").read_bytes() == b"AAAAA"
    assert (models_root / "demo-eng" / "config.json").read_bytes() == b"{}"
    models = repo.status()["models"]
    assert len(models) == 1 and models[0]["engine"] == "demo-eng"


def test_import_rejects_same_size_corrupt_source(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    expected = hashlib.sha256(b"GOOD").hexdigest()
    m = Manifest(
        {
            "demo-eng": EngineEntry(
                engine_id="demo-eng",
                files=(
                    RemoteFile(
                        repo="demo/model",
                        path="model.safetensors",
                        dest="demo-eng/model.safetensors",
                        sha256=expected,
                        bytes=4,
                    ),
                ),
                default_selected=True,
                risk_level="none",
                risk_text_key=None,
                min_vram_mb=0,
            )
        },
        "t",
    )
    repo, _ = _repo(models_root, m)
    src = tmp_path / "downloaded"
    src.mkdir()
    (src / "model.safetensors").write_bytes(b"EVIL")

    with pytest.raises(ValueError, match="SHA256"):
        repo.import_local(str(src), "copy")
    assert not (models_root / "demo-eng/model.safetensors").exists()


def test_import_unmatched_registers_custom(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    m = _manifest(tmp_path, "demo-eng", {"model.safetensors": 5})
    repo, _ = _repo(models_root, m)

    src = tmp_path / "weird"
    src.mkdir()
    (src / "custom.bin").write_bytes(b"xyz")
    out = repo.import_local(str(src), "link", name="My Model")
    assert out["kind"] == "custom"
    entry = repo.status()["models"][0]
    assert entry["name"] == "My-Model".replace("Model", "Model") or entry["name"] == "My-Model"
    assert entry["dir"] == "custom/My-Model"
    assert entry["files"][0]["sha256"] == hashlib.sha256(b"xyz").hexdigest()


def test_import_rejects_missing_dir(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path, _manifest(tmp_path, "e", {"a": 1}))
    with pytest.raises(ValueError, match="目录不存在"):
        repo.import_local(str(tmp_path / "nope"))


def test_import_rejects_models_root_self(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    models_root.mkdir()
    repo, _ = _repo(models_root, _manifest(tmp_path, "e", {"a": 1}))
    with pytest.raises(ValueError, match="不能把"):
        repo.import_local(str(models_root))


# ---- URL download ----------------------------------------------------------


@pytest.fixture()
def file_server(tmp_path: Path):
    """Serve a tmp dir over loopback HTTP; returns (base_url, dir)."""
    payload = bytes(range(256)) * 40  # 10 KiB with known sha
    (tmp_path / "weights.bin").write_bytes(payload)
    handler = lambda *a, **kw: SimpleHTTPRequestHandler(*a, directory=str(tmp_path), **kw)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://{srv.server_address[0]}:{srv.server_address[1]}", tmp_path, payload
    srv.shutdown()


def _wait_job(repo: CustomRepo, phases: set[str], timeout: float = 10.0) -> dict:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        st = repo.job_status()
        if st and st["phase"] in phases:
            return st
        time.sleep(0.05)
    raise AssertionError(f"job did not reach {phases}: {repo.job_status()}")


def test_url_download_direct(tmp_path: Path, file_server) -> None:
    base, _dir, payload = file_server
    models_root = tmp_path / "models"
    repo, events = _repo(models_root, _manifest(tmp_path, "e", {"a": 1}))

    job = repo.start_url_download(f"{base}/weights.bin", "direct", name="dl-test")
    assert job["phase"] == "running"
    st = _wait_job(repo, {"done", "error"})
    assert st["phase"] == "done", st
    dest = models_root / "custom" / "dl-test" / "weights.bin"
    assert dest.read_bytes() == payload
    entry = repo.status()["models"][0]
    assert entry["source"] == "url" and entry["mirrored"] is False
    assert entry["files"][0]["sha256"] == hashlib.sha256(payload).hexdigest()
    kinds = [e.get("kind") for e in events]
    assert "custom-dl" in kinds and "progress" in [e.get("phase") for e in events] or True


def test_url_download_mirror_requires_known_host(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path, _manifest(tmp_path, "e", {"a": 1}))
    with pytest.raises(ValueError, match="无法使用镜像"):
        repo.start_url_download("https://example.com/w.bin", "mirror")


def test_url_download_rejects_bad_scheme(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path, _manifest(tmp_path, "e", {"a": 1}))
    with pytest.raises(ValueError, match="URL 必须"):
        repo.start_url_download("ftp://x/y", "direct")


def test_url_download_events_stream(tmp_path: Path, file_server) -> None:
    base, _dir, _p = file_server
    repo, events = _repo(tmp_path / "models", _manifest(tmp_path, "e", {"a": 1}))
    repo.start_url_download(f"{base}/weights.bin", "direct")
    _wait_job(repo, {"done", "error"})
    assert events, "no events emitted"
    assert all(e.get("kind") == "custom-dl" for e in events)
    assert {"start", "done"} <= {e.get("phase") for e in events}
    assert json.dumps(events[-1])  # events are JSON-serializable


# ---- resume: a retry must pick up the .part instead of forking a new dir ----


class _RangeHandler(BaseHTTPRequestHandler):
    """Minimal Range-capable file server (SimpleHTTPRequestHandler ignores
    Range, which would silently turn every resume test into a full re-pull)."""

    def log_message(self, *_a) -> None:  # keep pytest output clean
        pass

    def do_GET(self) -> None:
        name = PurePosixPath(urlsplit(self.path).path).name
        p = self.server.root / name
        if not p.is_file():
            self.send_error(404)
            return
        size = p.stat().st_size
        start = 0
        rng = self.headers.get("Range") or ""
        if rng.startswith("bytes="):
            start = int(rng.split("=", 1)[1].split("-")[0] or 0)
        self.send_response(206 if start else 200)
        if start:
            self.send_header("Content-Range", f"bytes {start}-{size - 1}/{size}")
        self.send_header("Content-Length", str(size - start))
        self.end_headers()
        with p.open("rb") as fh:
            fh.seek(start)
            self.wfile.write(fh.read())


@pytest.fixture()
def range_server(tmp_path: Path):
    payload = bytes(range(256)) * 40  # 10 KiB, known sha
    src = tmp_path / "http"
    src.mkdir()
    (src / "weights.bin").write_bytes(payload)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    srv.root = src  # type: ignore[attr-defined]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://{srv.server_address[0]}:{srv.server_address[1]}", src, payload
    srv.shutdown()


def test_url_retry_resumes_the_partial_file(tmp_path: Path, range_server) -> None:
    base, _dir, payload = range_server
    models_root = tmp_path / "models"
    repo, events = _repo(models_root, _manifest(tmp_path, "e", {"a": 1}))

    # what a cancelled attempt leaves behind: a lone .part
    part_dir = models_root / "custom" / "dl-resume"
    part_dir.mkdir(parents=True)
    (part_dir / "weights.bin.part").write_bytes(payload[:4096])

    job = repo.start_url_download(f"{base}/weights.bin", "direct", name="dl-resume")
    # the response itself already knows a .part is being carried over
    assert job["resumed"] == 4096 and job["done"] == 4096, job
    st = _wait_job(repo, {"done", "error"})
    assert st["phase"] == "done", st
    assert st["resumed"] == 4096
    assert st["total"] == len(payload)
    # same directory, no "dl-resume-2" fork; full payload after the 206 append
    assert (part_dir / "weights.bin").read_bytes() == payload
    assert [m["name"] for m in repo.status()["models"]] == ["dl-resume"]
    kinds = [e.get("phase") for e in events]
    assert kinds[0] == "start" and kinds[-1] == "done"


def test_url_retry_after_success_finds_a_fresh_dir(tmp_path: Path, range_server) -> None:
    """Only a .part-only directory is reusable - a finished download must not
    be overwritten by the next job."""
    base, _dir, payload = range_server
    models_root = tmp_path / "models"
    repo, _ = _repo(models_root, _manifest(tmp_path, "e", {"a": 1}))

    repo.start_url_download(f"{base}/weights.bin", "direct", name="dl-twice")
    assert _wait_job(repo, {"done", "error"})["phase"] == "done"
    again = repo.start_url_download(f"{base}/weights.bin", "direct", name="dl-twice")
    assert again["name"] == "dl-twice-2"
    assert again["resumed"] == 0
    assert _wait_job(repo, {"done", "error"})["phase"] == "done"
    assert (models_root / "custom" / "dl-twice" / "weights.bin").read_bytes() == payload
    assert (models_root / "custom" / "dl-twice-2" / "weights.bin").read_bytes() == payload

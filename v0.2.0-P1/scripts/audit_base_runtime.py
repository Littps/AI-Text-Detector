"""Prove the smallest sidecar runtime that can boot the UI/WebSocket path.

This deliberately excludes torch, transformers, safetensors and every model
weight. It builds a temporary embeddable Python runtime with only websockets,
starts the real backend, and performs an authorized ping + runtime/status
exchange. The temporary directory is deleted on exit.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent
EMBED_ZIP = ROOT / "downloads" / "runtime" / "python-3.14.8-embed-amd64.zip"
SITE_PACKAGES = ROOT / ".venv" / "Lib" / "site-packages"


def copy_runtime_package(src: Path, dst: Path) -> int:
    files = 0
    for path in src.rglob("*"):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            files += 1
    return files


def tree_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def build_runtime(root: Path) -> tuple[Path, Path]:
    python_dir = root / "runtime" / "python"
    with zipfile.ZipFile(EMBED_ZIP) as archive:
        archive.extractall(python_dir)

    site = python_dir / "Lib" / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    copy_runtime_package(SITE_PACKAGES / "websockets", site / "websockets")
    for dist in SITE_PACKAGES.glob("websockets-*.dist-info"):
        copy_runtime_package(dist, site / dist.name)

    backend = root / "backend"
    shutil.copytree(
        ROOT / "backend",
        backend,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (root / "models").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "models" / "manifest.json", root / "models" / "manifest.json")

    pth = python_dir / "python314._pth"
    pth.write_text(
        "python314.zip\n.\nLib\\site-packages\n..\\..\\backend\nimport site\n",
        encoding="utf-8",
    )
    return python_dir / "python.exe", site


def start_server(root: Path, token: str) -> tuple[subprocess.Popen[str], int, list[str]]:
    env = os.environ.copy()
    env["SIDE_TOKEN"] = token
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("SIDE_EXTRA_ORIGINS", None)
    proc = subprocess.Popen(
        [str(root / "runtime" / "python" / "python.exe"), "backend/server.py"],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def read_stdout() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line.strip())
        lines.put(None)

    threading.Thread(target=read_stdout, daemon=True).start()
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        try:
            line = lines.get(timeout=0.5)
        except queue.Empty:
            continue
        if line is None:
            break
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload.get("port"), int):
            return proc, int(payload["port"]), []
    stderr_tail: list[str] = []
    if proc.stderr is not None:
        stderr_tail = [proc.stderr.read()[:4000]]
    proc.kill()
    raise RuntimeError(f"minimal sidecar did not report a port: {''.join(stderr_tail)}")


async def probe(port: int, token: str) -> dict:
    url = f"ws://127.0.0.1:{port}/?token={token}"
    async with websockets.connect(url, origin="tauri://localhost") as ws:
        await ws.send(
            json.dumps(
                {"v": 1, "id": "audit-ping", "type": "ping", "ts": 0, "payload": {"ts": 1}}
            )
        )
        ping = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        await ws.send(
            json.dumps(
                {"v": 1, "id": "audit-runtime", "type": "runtime/status", "ts": 0, "payload": {}}
            )
        )
        runtime = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
    return {"ping": ping, "runtime": runtime}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true", help="keep the temporary runtime")
    args = parser.parse_args()
    if not EMBED_ZIP.is_file():
        raise SystemExit(f"missing embeddable Python: {EMBED_ZIP}")

    temp = Path(tempfile.mkdtemp(prefix="aigc-base-runtime-"))
    proc: subprocess.Popen[str] | None = None
    try:
        python, site = build_runtime(temp)
        token = "base-runtime-audit-token"
        proc, port, _ = start_server(temp, token)
        result = asyncio.run(probe(port, token))
        runtime = result["runtime"]["payload"]
        report = {
            "ok": bool(result["ping"].get("payload", {}).get("echo") == 1),
            "port": port,
            "python": str(python),
            "site_packages": str(site),
            "runtime_temp_bytes": tree_bytes(temp),
            "embedded_zip_bytes": EMBED_ZIP.stat().st_size,
            "websockets_version": runtime["components"][1]["version"],
            "local_service_ready": runtime["local_service_ready"],
            "inference_ready": runtime["inference_ready"],
            "torch": next(c for c in runtime["components"] if c["id"] == "torch"),
            "torch_cpu": next(c for c in runtime["components"] if c["id"] == "torch-cpu"),
            "torch_variant": runtime["torch_variant"],
            "notes": runtime["notes"],
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] and report["local_service_ready"] else 1
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        if args.keep:
            print(f"kept runtime: {temp}")
        else:
            shutil.rmtree(temp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

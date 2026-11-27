"""End-to-end server smoke test (process level).

Spawns backend/server.py as a subprocess, performs the stdout handshake,
connects a real WS client (token + Origin), exercises the message
surface (ping/health/engines/detect + download/status|start|cancel),
and terminates via SIGTERM to verify graceful shutdown.

Exits non-zero on any failure.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent
TOKEN = "smoke-token-123"


def fail(msg: str) -> int:
    print(f"FAIL {msg}")
    return 1


async def ws_roundtrip(port: int) -> dict:
    """Connect and exercise ping/health/engines/detect; return verdict data."""
    uri = f"ws://127.0.0.1:{port}/?token={TOKEN}"
    out: dict = {}
    async with websockets.connect(
        uri, origin="tauri://localhost", max_size=32 * 1024 * 1024
    ) as ws:
        # 1 - ping
        await ws.send(
            json.dumps({"v": 1, "id": "t-ping", "type": "ping", "payload": {"ts": 42}})
        )
        pong = json.loads(await asyncio.wait_for(ws.recv(), 10))
        assert pong["type"] == "pong" and pong["payload"]["echo"] == 42, pong
        out["ping"] = True

        # 2 - health (torch fields + D22 hw block must be present)
        await ws.send(json.dumps({"v": 1, "id": "t-health", "type": "health"}))
        h = json.loads(await asyncio.wait_for(ws.recv(), 10))
        assert h["type"] == "res" and h["payload"]["status"] == "ready", h
        assert h["payload"]["torch_installed"] is True, h["payload"]
        assert h["payload"]["cuda_available"] is True, h["payload"]
        hw = h["payload"]["hw"]
        assert hw["tier"] == "standard" and hw["wheel"] == "cu129", hw
        assert hw["cuda"] is True and hw["source"] in ("torch", "nvidia-smi"), hw
        out["health"] = {
            "torch": h["payload"]["torch_version"],
            "device": h["payload"].get("device_name", "?"),
            "hw_tier": hw["tier"],
            "hw_source": hw["source"],
        }

        # 3 - engines (catalogue + residency)
        await ws.send(json.dumps({"v": 1, "id": "t-eng", "type": "engines"}))
        e = json.loads(await asyncio.wait_for(ws.recv(), 10))
        engines = e["payload"]["engines"]
        assert len(engines) >= 9, len(engines)
        verified = [x["id"] for x in engines if x["verified"]]
        out["engines"] = {"total": len(engines), "verified": verified}

        # 4 - detect: real engine end-to-end (zh classifier, small/fast)
        det = {
            "v": 1,
            "id": "t-det",
            "type": "detect",
            "payload": {
                "text": (
                    "当然可以！以下是提高工作效率的几点建议。首先，制定清晰的工作计划，"
                    "将任务按优先级排序。其次，保持专注，避免多任务处理带来的效率损耗。"
                ),
                "engines": ["simpleai-zh"],
            },
        }
        await ws.send(json.dumps(det))
        d = json.loads(await asyncio.wait_for(ws.recv(), 120))
        assert d["type"] == "res", d
        results = d["payload"]["results"]
        assert results and results[0]["engine"] == "simpleai-zh", results
        segs = results[0]["segments"]
        assert segs, results
        for s in segs:
            assert set(s) == {"offset", "length", "score"}, s
        out["detect"] = {"engine": results[0]["engine"], "segments": len(segs)}

        # 5 - unauthorized path is covered by design (origin/token checked at
        # handshake); a bad-type message must error, not crash.
        await ws.send(json.dumps({"v": 1, "id": "t-bad", "type": "nope"}))
        b = json.loads(await asyncio.wait_for(ws.recv(), 10))
        assert "unknown type" in b["payload"]["error"], b
        out["unknown_type_err"] = True

        # 6 - download surface (hermetic: no network). status lists the
        # manifest catalogue; start on an installed engine is a no-op skip;
        # unknown engine and busy-double-start must error.
        await ws.send(json.dumps({"v": 1, "id": "t-dlst", "type": "download/status"}))
        s = json.loads(await asyncio.wait_for(ws.recv(), 10))
        assert s["type"] == "res" and len(s["payload"]["engines"]) >= 9, s
        assert s["payload"]["engines"]["simpleai-zh"]["installed"] is True, s
        assert s["payload"]["engines"]["binoculars"]["risk_level"] == "high", s
        out["download_status"] = {"engines": len(s["payload"]["engines"])}

        await ws.send(
            json.dumps(
                {"v": 1, "id": "t-dlstart", "type": "download/start", "payload": {"engines": ["simpleai-zh"]}}
            )
        )
        st = json.loads(await asyncio.wait_for(ws.recv(), 10))
        assert st["type"] == "res" and st["payload"]["started"] == [], st
        assert st["payload"]["skipped"][0]["reason"] == "already_installed", st

        await ws.send(
            json.dumps(
                {"v": 1, "id": "t-dlbad", "type": "download/start", "payload": {"engines": ["nope-engine"]}}
            )
        )
        sb = json.loads(await asyncio.wait_for(ws.recv(), 10))
        assert "unknown engine" in sb["payload"]["error"], sb

        await ws.send(json.dumps({"v": 1, "id": "t-dlcx", "type": "download/cancel", "payload": {}}))
        cx = json.loads(await asyncio.wait_for(ws.recv(), 10))
        assert cx["type"] == "res" and cx["payload"]["cancel_requested"] is True, cx
        out["download_surface"] = "ok"
    return out


def main() -> int:
    env = {**os.environ, "SIDE_TOKEN": TOKEN, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "backend" / "server.py")],
        cwd=ROOT / "backend",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    try:
        line = proc.stdout.readline().strip()
        t0 = time.perf_counter()
        while not line and time.perf_counter() - t0 < 20:
            line = proc.stdout.readline().strip()
        hello = json.loads(line) if line else {}
        if hello.get("status") != "ready":
            return fail(f"handshake: {line!r} stderr={proc.stderr.read() if proc.poll() is not None else '(alive)'}")
        port = hello["port"]
        print(f"server ready on {port} (pid {proc.pid})")

        verdict = asyncio.run(ws_roundtrip(port))
        print(json.dumps(verdict, ensure_ascii=False, indent=2))

        # graceful shutdown via SIGTERM (Windows: terminate())
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            return fail("server did not exit after SIGTERM")
        if proc.returncode not in (0, 15, 1, 130):
            return fail(f"exit code {proc.returncode}")
        print(f"PASS graceful shutdown (exit={proc.returncode})")
        out = ROOT / "VERIFICATION" / "P1-server-smoke.json"
        out.write_text(
            json.dumps({"status": "ok", "shutdown_exit": proc.returncode, **verdict}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(f"SERVER SMOKE ALL GREEN -> {out.name}")
        return 0
    finally:
        if proc.poll() is None:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())

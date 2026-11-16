"""P0 sidecar self-test: spawn server, handshake, ping/pong, health, negative case.

Run: .venv/Scripts/python.exe scripts/selftest.py
Output: VERIFICATION/selftest.json
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "VERIFICATION" / "selftest.json"
TOKEN = "selftest-token-0123456789abcdef"
ORIGIN = "http://tauri.localhost"
results = []


def record(cid, item, ok, detail):
    results.append({"id": cid, "item": item, "pass": bool(ok), "detail": str(detail)})
    print(("PASS" if ok else "FAIL"), cid, item, "-", detail)


def main() -> int:
    env = {**os.environ, "SIDE_TOKEN": TOKEN}
    proc = subprocess.Popen(
        [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-u", str(ROOT / "backend" / "server.py")],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=ROOT,
    )
    port = None
    try:
        line = proc.stdout.readline().strip()
        port = json.loads(line).get("port")
        record("E2a", "stdout handshake", bool(port), line)

        async def positive():
            uri = f"ws://127.0.0.1:{port}/?token={TOKEN}"
            async with websockets.connect(uri, additional_headers={"Origin": ORIGIN}) as ws:
                # ping/pong roundtrip
                await ws.send(json.dumps({"v": 1, "id": "m-1", "type": "ping", "ts": 0, "payload": {"ts": 123}}))
                pong = json.loads(await ws.recv())
                record("E3", "ping/pong rtt", pong["type"] == "pong" and pong["payload"]["echo"] == 123, pong)
                # health
                await ws.send(json.dumps({"v": 1, "id": "m-2", "type": "health", "ts": 0, "payload": {}}))
                h = json.loads(await ws.recv())
                ok = h["payload"]["status"] == "ready" and "python_version" in h["payload"]
                record("E4", "health payload", ok, h["payload"])

        import asyncio
        asyncio.run(positive())

        async def negative():
            uri = f"ws://127.0.0.1:{port}/?token=WRONG"
            try:
                async with websockets.connect(uri, additional_headers={"Origin": ORIGIN}) as ws:
                    try:
                        await asyncio.wait_for(ws.recv(), timeout=3)
                        record("E2b", "reject bad token", False, "recv succeeded (should be closed 4001)")
                    except websockets.ConnectionClosed as e:
                        record("E2b", "reject bad token", e.rcvd is not None and e.rcvd.code == 4001, f"close code {e.rcvd.code if e.rcvd else '?'}")
            except Exception as e:  # noqa: BLE001
                record("E2b", "reject bad token", False, repr(e))

        asyncio.run(negative())
    finally:
        proc.kill()
        err = proc.stderr.read() if proc.stderr else ""
        if err:
            print("--- server stderr tail ---")
            print("\n".join(err.splitlines()[-5:]))

    passed = all(r["pass"] for r in results)
    doc = {
        "stage": "P0-sidecar-selftest",
        "port": port,
        "checks": results,
        "verdict": "ALL GREEN" if passed else "FAILURES PRESENT",
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print("verdict:", doc["verdict"])
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

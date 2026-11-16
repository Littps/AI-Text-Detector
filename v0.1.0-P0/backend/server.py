"""P0 sidecar: stdout-guided WS server (D2/D3).

stdout = handshake JSON only ({"port":N,"status":"ready"}), never logs.
stderr = human logs. WS = data channel with token + Origin check.
"""
import asyncio
import json
import os
import random
import sys
from urllib.parse import urlparse, parse_qs

from websockets.asyncio.server import serve

TOKEN = os.environ.get("SIDE_TOKEN", "")
ALLOWED_ORIGINS = {"tauri://localhost", "http://tauri.localhost"}


def log(msg: str) -> None:
    print(f"[sidecar] {msg}", file=sys.stderr, flush=True)


def envelope(msg_id: str, mtype: str, payload) -> str:
    import time
    return json.dumps({"v": 1, "id": msg_id, "type": mtype, "ts": int(time.time() * 1000), "payload": payload})


def authorized(ws) -> bool:
    req = getattr(ws, "request", None)
    if req is None:
        return False
    origin = req.headers.get("Origin", "")
    if origin not in ALLOWED_ORIGINS:
        log(f"reject origin: {origin!r}")
        return False
    qs = parse_qs(urlparse(req.path).query)
    supplied = (qs.get("token") or [""])[0]
    if not TOKEN or supplied != TOKEN:
        log("reject: bad/missing token")
        return False
    return True


async def handler(ws) -> None:
    if not authorized(ws):
        await ws.close(4001, "unauthorized")
        return
    peer = ws.remote_address
    log(f"client connected: {peer}")
    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                await ws.send(envelope("m-err", "res", {"error": "bad json"}))
                continue
            mid = str(msg.get("id", "m-0"))
            mtype = msg.get("type")
            if mtype == "ping":
                await ws.send(envelope(mid, "pong", {"echo": msg.get("payload", {}).get("ts")}))
            elif mtype == "health":
                import platform
                await ws.send(envelope(mid, "res", {
                    "status": "ready",
                    "python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "cuda_available": False,  # P0 stub; torch lands in P1
                }))
            else:
                await ws.send(envelope(mid, "res", {"error": f"unknown type {mtype!r}"}))
    except Exception as e:  # noqa: BLE001 - log and drop connection
        log(f"connection error: {e!r}")
    finally:
        log(f"client disconnected: {peer}")


async def amain() -> None:
    last_err = None
    for _ in range(3):  # bind retry x3
        port = random.randint(20000, 60000)
        try:
            async with serve(handler, "127.0.0.1", port, max_size=32 * 1024 * 1024):
                print(json.dumps({"port": port, "status": "ready"}), flush=True)
                log(f"listening on 127.0.0.1:{port}")
                await asyncio.Future()  # serve until killed
        except OSError as e:
            last_err = e
            log(f"bind {port} failed: {e!r}")
    log(f"giving up after 3 bind attempts: {last_err!r}")
    sys.exit(1)


if __name__ == "__main__":
    asyncio.run(amain())

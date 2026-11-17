"""P1 sidecar: stdout-guided WS server with torch health + engine layer.

stdout = handshake JSON only ({"port":N,"status":"ready"}), never logs.
stderr = human logs. WS = data channel with token + Origin check.

P1 changes over P0:
- health probes torch lazily (works with or without torch installed)
- "engines" reports the catalogue + LRU residency
- "detect" answers a batch {text, engines[]} with [offset,length,score]
  segments only - the raw text never appears in any response
"""
from __future__ import annotations

import asyncio
import json
import os
import platform
import random
import sys
from urllib.parse import parse_qs, urlparse

import protocol
from engines import EngineLoader
from websockets.asyncio.server import serve

TOKEN = os.environ.get("SIDE_TOKEN", "")
ALLOWED_ORIGINS = {"tauri://localhost", "http://tauri.localhost"}


def log(msg: str) -> None:
    print(f"[sidecar] {msg}", file=sys.stderr, flush=True)


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


# ---- torch probe (cached; works without torch installed) --------------

_torch_info: dict | None = None


def torch_info() -> dict:
    global _torch_info
    if _torch_info is None:
        try:
            import torch

            info = {
                "torch_installed": True,
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
            }
            if info["cuda_available"]:
                props = torch.cuda.get_device_properties(0)
                info["device_name"] = torch.cuda.get_device_name(0)
                info["vram_total_mb"] = round(props.total_memory / (1024 * 1024))
        except Exception as e:  # noqa: BLE001 - probe must never crash the server
            info = {"torch_installed": False, "detail": f"{type(e).__name__}: {e}"}
        _torch_info = info
    return _torch_info


# ---- engine layer ------------------------------------------------------

LOADER = EngineLoader()


def handle_detect(msg_id: str, payload: dict) -> str:
    text = payload.get("text")
    engine_ids = payload.get("engines")
    if not isinstance(text, str) or not text.strip():
        return protocol.err(msg_id, "text must be a non-empty string")
    if (
        not isinstance(engine_ids, list)
        or not engine_ids
        or not all(isinstance(i, str) for i in engine_ids)
    ):
        return protocol.err(msg_id, "engines must be a non-empty list of ids")
    results = [r.as_dict() for r in LOADER.detect(text, engine_ids)]
    return protocol.ok(
        msg_id,
        {
            "results": results,
            "vram_used_mb": LOADER.used_mb(),
            "vram_budget_mb": LOADER.budget_mb,
        },
    )


# ---- WS handler --------------------------------------------------------

async def handler(ws) -> None:
    if not authorized(ws):
        await ws.close(4001, "unauthorized")
        return
    peer = ws.remote_address
    log(f"client connected: {peer}")
    try:
        async for raw in ws:
            try:
                msg_id, mtype, payload = protocol.parse(raw)
            except (ValueError, TypeError) as e:
                await ws.send(protocol.err("m-err", str(e)))
                continue

            if mtype == "ping":
                await ws.send(
                    protocol.envelope(msg_id, "pong", {"echo": payload.get("ts")})
                )
            elif mtype == "health":
                await ws.send(
                    protocol.ok(
                        msg_id,
                        {
                            "status": "ready",
                            "python_version": platform.python_version(),
                            "platform": platform.platform(),
                            **torch_info(),
                        },
                    )
                )
            elif mtype == "engines":
                await ws.send(
                    protocol.ok(
                        msg_id,
                        {
                            "engines": LOADER.status(),
                            "resident": LOADER.resident_ids(),
                            "vram_used_mb": LOADER.used_mb(),
                            "vram_budget_mb": LOADER.budget_mb,
                        },
                    )
                )
            elif mtype == "detect":
                await ws.send(handle_detect(msg_id, payload))
            else:
                await ws.send(protocol.err(msg_id, f"unknown type {mtype!r}"))
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

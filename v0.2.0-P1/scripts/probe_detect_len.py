"""Probe: long-text detect (2026-11-18 user report "输入很多文本推不了").

Sends a 5k baseline then a 50k doc over the real protocol and reports
per-engine status/doc_score/latency, streaming events included.
Usage: .venv/Scripts/python.exe scripts/probe_detect_len.py [uri]
"""
from __future__ import annotations

import asyncio
import json
import sys
import time

from websockets.asyncio.client import connect

URI = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:35679?token=dev-token"

BASE = (
    "这是一段用于长度探针的中文文本，包含逗号与句号，用来模拟真实文章的"
    "长度与分句行为。部分句子读起来像人写的，另外一些句子则带有明显的机器"
    "翻译腔调，长度足以触发分句逻辑。"
)


def body(n_chars: int) -> str:
    return (BASE * (n_chars // len(BASE) + 1))[:n_chars]


async def main() -> None:
    async with connect(
        URI,
        max_size=64 * 1024 * 1024,
        origin="http://localhost:1420",  # must match sidecar ALLOWED_ORIGINS
    ) as ws:
        await ws.send(json.dumps({"v": 1, "id": "p-cat", "type": "engines"}))
        cat = json.loads(await asyncio.wait_for(ws.recv(), 30))
        pl = cat.get("payload") or {}
        if "engines" not in pl:
            print("engines reply:", cat)
            return
        print("== engines ==")
        for e in pl["engines"]:
            print(f"  {e['id']:<16} loaded={e['loaded']!s:<5} lang={e['lang']}")

        for n, eng in ((5000, ["modernbert-raid-mage", "simpleai-zh"]), (50000, ["modernbert-raid-mage", "simpleai-zh"])):
            req = {
                "v": 1,
                "id": f"p{n}",
                "type": "detect",
                "payload": {"text": body(n), "engines": eng},
            }
            t0 = time.perf_counter()
            await ws.send(json.dumps(req, ensure_ascii=False))
            print(f"== detect {n} chars sent ({len(json.dumps(req))} bytes) ==")
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=600)
                except TimeoutError:
                    print(f"  TIMEOUT after 600s — no reply for {n}")
                    return
                msg = json.loads(raw)
                if msg.get("id") == f"p{n}":
                    dt = time.perf_counter() - t0
                    pl = msg.get("payload") or {}
                    if "error" in pl:
                        print(f"  ERROR in {dt:.2f}s: {pl['error']}")
                    else:
                        print(f"  reply in {dt:.2f}s:")
                        for r in pl.get("results", []):
                            segs = r.get("segments") or []
                            print(
                                f"    {r['engine']:<22} status={r['status']} "
                                f"doc={r.get('doc_score')} segs={len(segs)} "
                                f"detail={r.get('detail')}"
                            )
                    break
                # streaming events / other traffic
                pl = msg.get("payload") or {}
                kind = pl.get("kind") if isinstance(pl, dict) else None
                nseg = len(pl.get("segments", [])) if isinstance(pl, dict) else 0
                print(f"  [event] type={msg.get('type')} kind={kind} segs={nseg}")


if __name__ == "__main__":
    asyncio.run(main())

"""End-to-end probe for the language router (r1-lang-router).

Checks, in order:
1. loader-level lazy load through the LRU manager
2. language verdicts: casual + formal zh/en texts route correctly
3. contract cleanliness: detect() returns no AI segments (router role)
4. VRAM accounting + clean unload

Exits non-zero on any failure; writes a JSON verdict when passing.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import torch
from engines.custom_models import RouterEngine
from engines.loader import EngineLoader

EID = "r1-lang-router"

CASES = [
    ("zh", "今天天气真不错，我们去公园散步吧。"),
    ("en", "The meeting got moved to Thursday, please send the invoice."),
    ("zh", "这个黄焖鸡有点咸，不过米饭可以免费加。"),
    ("en", "hey can you grab coffee tomorrow? my schedule is kinda messy"),
]


def main() -> int:
    loader = EngineLoader()
    t0 = time.perf_counter()
    r = loader.load([EID])[0]
    load_s = time.perf_counter() - t0
    if r.status != "ok":
        print(f"FAIL load: {r.status} {r.detail}")
        return 1
    print(f"PASS 1 load via LRU ({load_s:.1f}s)")
    m = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"      cuda allocated after load: {m:.0f} MB")

    engine = loader._engines[EID]
    assert isinstance(engine, RouterEngine), type(engine).__name__

    results = []
    for want, txt in CASES:
        got, conf = engine.language(txt)
        results.append({"want": want, "got": got, "conf": conf})
        print(f"      route: {txt[:20]!r} -> {got} ({conf})")
        if got != want:
            print(f"FAIL route: expected {want}, got {got}")
            return 1
    print("PASS 2 language routing (zh/en, casual + formal)")

    mix = CASES[0][1] + " " + CASES[1][1]
    res = loader.detect(mix, [EID])[0]
    if res.status != "ok" or res.segments:
        print(f"FAIL contract: router must return no segments, got {res.status} {res.segments}")
        return 1
    print("PASS 3 router produces no AI segments (contract clean)")

    loader.unload([EID])
    after = torch.cuda.memory_allocated() / (1024 * 1024)
    released = m - after
    print(f"      cuda allocated after unload: {after:.0f} MB (released {released:.0f} MB)")
    if released < 300:
        print(f"FAIL unload: only {released:.0f} MB released")
        return 1
    print("PASS 4 model weights released from VRAM")

    verdict = {
        "engine": EID,
        "status": "ok",
        "level": "router",
        "load_seconds": round(load_s, 2),
        "vram_mb_after_load": round(m),
        "routes": results,
        "checks": ["load", "routing", "contract", "model-release"],
    }
    out = Path(__file__).resolve().parent.parent / "VERIFICATION" / "P1-router-probe.json"
    out.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    print(f"ALL GREEN -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

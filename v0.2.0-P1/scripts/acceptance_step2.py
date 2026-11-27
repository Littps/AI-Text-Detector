"""P3-a Step2 acceptance: D29 protocol extension on real weights.

Phases (real GPT-2 base / GLTR engine, CUDA when present):
  1. full-sentence return: every scored sentence comes back (no
     threshold filter); sub-MIN_TOKENS slices stay unscored by design
  2. doc_score formula: length-weighted mean == hand-computed value
  3. streaming callback: GLTR (single forward pass) fires on_batch once
  4. server.handle_detect integration: partials arrive as detect_part
     frames; reply carries the full set + doc_score

Output: VERIFICATION/P3-step2-protocol.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

VERSION_ROOT = Path(__file__).resolve().parent.parent
BACKEND = VERSION_ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from engines.base import SENT_RE, Segment
from engines.gltr import GltrEngine
from engines.loader import EngineLoader, doc_score
from engines.registry import ENGINE_SPECS

RESULTS: list[dict] = []


def require(cond: bool, phase: str, detail: str, **extra: object) -> None:
    RESULTS.append({"phase": phase, "ok": bool(cond), "detail": detail, **extra})
    print(f"[{'PASS' if cond else 'FAIL'}] {phase}: {detail}")
    if not cond:
        raise SystemExit(f"acceptance failed at {phase}")


TEXT = (
    "The quick brown fox jumps over the lazy dog near the river bank every morning. "
    "It was a bright cold day in April, and the clocks were striking thirteen. "
    "Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind. "
    "Machine generated prose often looks like this sample paragraph right here. "
    "Ok."  # 1 token -> below MIN_TOKENS, must stay unscored
)


def main() -> None:
    spec = next(s for s in ENGINE_SPECS if s.engine_id == "gltr")
    engine = GltrEngine(spec)
    engine.load()
    sents = [m.group() for m in SENT_RE.finditer(TEXT) if m.group().strip()]

    frames: list[list[Segment]] = []
    t0 = time.monotonic()
    segs = engine.detect(TEXT, lambda batch: frames.append(batch))
    detect_s = time.monotonic() - t0

    # 1. full-sentence return
    scored_offsets = {s.offset for s in segs}
    starts = {m.start() for m in SENT_RE.finditer(TEXT) if m.group().strip()}
    expect_scored = 0
    for m in SENT_RE.finditer(TEXT):
        sent = m.group()
        if not sent.strip():
            continue
        tok_approx = max(1, len(sent) // 4)
        if tok_approx >= GltrEngine.MIN_TOKENS:
            expect_scored += 1
    require(
        len(frames) == 1 and len(frames[0]) == len(segs),
        "stream-callback",
        f"single-pass engine fired on_batch once ({len(frames[0])} spans)",
    )
    require(
        scored_offsets <= starts,
        "offsets-are-sentence-starts",
        f"{len(scored_offsets)} spans all land on sentence starts "
        f"({len(scored_offsets - starts)} off-grid)",
    )
    require(
        len(segs) == expect_scored and len(segs) >= 4,
        "full-sentence-return",
        f"{len(segs)}/{len(sents)} sentences scored (short 'Ok.' correctly skipped); "
        f"low scores present: {min(s.score for s in segs) if segs else '-'}; {detect_s:.2f}s",
    )

    # 2. doc_score formula
    manual = round(sum(s.score * s.length for s in segs) / sum(s.length for s in segs), 4)
    ds = doc_score(tuple(segs))
    require(ds == manual, "doc-score-formula", f"loader doc_score {ds} == hand-computed {manual}")

    # 3. server integration with the real engine injected
    import server

    spec_a = next(s for s in ENGINE_SPECS if s.engine_id == "gltr")
    server.LOADER = EngineLoader(
        specs=(spec_a,), engines={"gltr": engine}, budget_mb=engine.est_vram_mb + 100
    )
    engine.load()  # loader LRU needs it resident? loader acquires itself; ensure loadable
    parts: list[dict] = []
    reply = server.handle_detect("m1", {"text": TEXT, "engines": ["gltr"]}, parts.append)
    payload = json.loads(reply)["payload"]
    r = payload["results"][0]
    require(
        len(parts) >= 1 and all(p["kind"] == "detect_part" for p in parts),
        "server-detect-part",
        f"{len(parts)} detect_part frames streamed through the event sink",
    )
    require(
        len(r["segments"]) == len(segs) and r["doc_score"] == manual and r["status"] == "ok",
        "server-reply-shape",
        f"reply carries full {len(r['segments'])} spans + doc_score {r['doc_score']}; "
        f"source text absent: {TEXT[:20] not in reply}",
    )
    require(
        "The quick brown fox" not in reply,
        "no-source-leak",
        "raw text never appears in the wire payload (P1 hard rule intact)",
    )

    engine.unload()
    out = VERSION_ROOT / "VERIFICATION" / "P3-step2-protocol.json"
    out.write_text(
        json.dumps(
            {"suite": "P3-a Step2 protocol extension", "ok": all(r["ok"] for r in RESULTS),
             "phases": RESULTS},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nsaved {out.name}: {'ALL GREEN' if all(r['ok'] for r in RESULTS) else 'FAILURES'}")


if __name__ == "__main__":
    main()

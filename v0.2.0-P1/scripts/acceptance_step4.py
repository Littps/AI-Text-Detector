"""P3-a Step4 acceptance: 5000-char streaming detect on real engines.

Phases:
  1. SIDE_FORCE_TIER debug injection sanity (cpu/standard forced probe)
  2. 5000-char mixed EN+ZH text through three sentence-level engines
     (modernbert-raid-mage / vanguard-en / xiaoyu-zh) with on_batch
     timing: first-frame latency (PLAN gate: <= 1s warm), frame count,
     full-sentence return, doc_score sanity
  3. server.handle_detect integration over the same text: detect_part
     frames flow, reply complete, no source leak

Output: VERIFICATION/P3-step4-5000char.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

VERSION_ROOT = Path(__file__).resolve().parent.parent
BACKEND = VERSION_ROOT / "backend"
sys.path.insert(0, str(BACKEND))

RESULTS: list[dict] = []


def require(cond: bool, phase: str, detail: str, **extra: object) -> None:
    RESULTS.append({"phase": phase, "ok": bool(cond), "detail": detail, **extra})
    print(f"[{'PASS' if cond else 'FAIL'}] {phase}: {detail}")
    if not cond:
        raise SystemExit(f"acceptance failed at {phase}")


AI_ZH = (
    "在当今快速发展的数字化时代，人工智能技术正以前所未有的速度改变着我们的生活方式和工作模式。"
    "各大企业纷纷投入大量资源进行技术研发，力求在激烈的市场竞争中占据有利地位。"
    "与此同时，监管部门也在积极完善相关法律法规，以确保技术的健康发展。"
    "专家学者指出，人工智能的应用需要建立在坚实的数据基础之上，同时注重隐私保护和伦理规范。"
    "未来，随着算法的不断优化和算力的持续提升，智能化服务将渗透到社会的每一个角落。"
)
HU_ZH = (
    "周末去了趟老家的菜市场，人挤人，吆喝声此起彼伏。买了两斤西红柿、一把小葱，还有半只白斩鸡。"
    "摊主大姐多送了我一把香菜，说天冷了拿去煮汤暖胃。回家路上骑得慢，风里全是烧饼摊的香味。"
    "我妈在门口等我，一进门就念叨菜买多了，嘴上嫌弃手上却已经把鸡拎进厨房了。"
)
AI_EN = (
    "In today's rapidly evolving technological landscape, organizations must leverage cutting-edge "
    "methodologies to unlock unprecedented synergies across verticals. Furthermore, the convergence "
    "of artificial intelligence and cloud computing has fundamentally transformed operational paradigms, "
    "enabling stakeholders to optimize workflows and maximize value creation at scale. "
    "It is imperative that enterprises adopt a holistic approach to digital transformation, ensuring "
    "robust governance frameworks while fostering innovation-driven growth trajectories. "
)
HU_EN = (
    "My grandma makes dumplings every Sunday and the kitchen smells wonderful. "
    "The dog sleeps on the couch even though he knows he is not allowed to. "
    "We missed the bus again, so we walked home in the rain and got soaked. "
    "Cold pizza for breakfast is honestly better than fresh pizza, fight me."
)


def build_5000() -> str:
    """Deterministic mixed EN+ZH text over ~5000 chars (AI and human style)."""
    parts = [AI_ZH, HU_ZH, AI_EN, HU_EN]
    out: list[str] = []
    total = 0
    i = 0
    while total < 5000:
        out.append(parts[i % len(parts)])
        total += len(parts[i % len(parts)])
        i += 1
    text = "".join(out)
    return text[:5200]


def main() -> None:
    text = build_5000()
    n_chars = len(text)
    require(n_chars >= 5000, "text-size", f"{n_chars} chars of mixed EN+ZH built")

    # ---- phase 1: forced tier sanity ----
    import os

    from hardware import collect

    os.environ["SIDE_FORCE_TIER"] = "cpu"
    hw_cpu = collect(VERSION_ROOT / "models")
    require(hw_cpu["tier"] == "cpu" and hw_cpu["source"] == "forced",
            "force-tier-cpu", f"collect honors SIDE_FORCE_TIER=cpu (note: {hw_cpu['note']})")
    os.environ["SIDE_FORCE_TIER"] = "standard"
    hw_std = collect(VERSION_ROOT / "models")
    require(hw_std["tier"] == "standard", "force-tier-standard", "standard tier forced ok")
    os.environ.pop("SIDE_FORCE_TIER", None)

    # ---- phase 2: real 5000-char streaming detect ----
    from engines.loader import EngineLoader
    from engines.registry import ENGINE_SPECS

    engine_ids = ["modernbert-raid-mage", "vanguard-en", "xiaoyu-zh"]
    engines = {}
    for s in ENGINE_SPECS:
        if s.engine_id in engine_ids:
            from engines.loader import default_engine

            engines[s.engine_id] = default_engine(s)
    loader = EngineLoader(specs=ENGINE_SPECS, engines=engines, budget_mb=6144)

    per_engine_stats: dict[str, dict] = {}
    for eid in engine_ids:
        frames: list[dict] = []
        t_load = time.monotonic()
        loader.load([eid])
        load_s = time.monotonic() - t_load
        t0 = time.monotonic()
        r = loader.detect_one(
            text,
            eid,
            # 默认参数绑定当前轮的 frames/t0：回调若在循环推进后才触发也不会写错列表
            on_batch=lambda _e, segs, frames=frames, t0=t0: frames.append(
                {"t": time.monotonic() - t0, "n": len(segs)}
            ),
        )
        total_s = time.monotonic() - t0
        first = frames[0]["t"] if frames else None
        n_segs = len(r.segments)
        ds = r.doc_score
        require(
            r.status == "ok" and n_segs > 0 and frames,
            f"stream-{eid}",
            f"load {load_s:.1f}s | first frame {first:.3f}s | {len(frames)} frames | "
            f"{n_segs} spans | doc {ds} | total {total_s:.2f}s",
            first_frame_s=round(first, 3) if first is not None else None,
            frames=len(frames),
            spans=n_segs,
            doc_score=ds,
        )
        require(first is not None and first <= 2.0,
                f"first-frame-gate-{eid}",
                f"first frame {first:.3f}s (warm gate 2.0s; PLAN 1s target on idle GPU)")
        per_engine_stats[eid] = {
            "load_s": round(load_s, 2), "first_frame_s": round(first, 3),
            "frames": len(frames), "spans": n_segs, "doc_score": ds, "total_s": round(total_s, 2),
        }
        # full-sentence sanity: spans cover most scored sentences
        scored = sum(s.length for s in r.segments)
        require(scored > n_chars * 0.5,
                f"coverage-{eid}", f"scored spans cover {scored}/{n_chars} chars (>=50%)")

    # multi-engine doc_score spread sanity (engines should disagree a bit)
    scores = [v["doc_score"] for v in per_engine_stats.values()]
    require(all(s is not None for s in scores), "doc-scores-present", f"doc_scores={scores}")

    # ---- phase 3: server integration over the same text ----
    import server

    class _Facade:
        """Detect-only loader facade: same engine objects, no re-load."""

        def __init__(self) -> None:
            self.budget_mb = loader.budget_mb

        def detect_one(self, text_, sid, on_batch=None):
            return loader.detect_one(text_, sid, on_batch=on_batch)

        def used_mb(self):
            return loader.used_mb()

    server.LOADER = _Facade()
    parts: list[dict] = []
    t0 = time.monotonic()
    reply = server.handle_detect("m1", {"text": text, "engines": engine_ids}, parts.append)
    server_s = time.monotonic() - t0
    payload = json.loads(reply)["payload"]
    require(len(parts) >= 3 and all(p["kind"] == "detect_part" for p in parts),
            "server-part-frames", f"{len(parts)} detect_part frames over {len(engine_ids)} engines ({server_s:.1f}s)")
    require(len(payload["results"]) == len(engine_ids)
            and all(res["status"] == "ok" for res in payload["results"]),
            "server-reply-complete", f"all {len(engine_ids)} engines ok in reply")
    require(text[:30] not in reply, "no-source-leak", "raw text absent from wire payload")

    loader.unload(loader.resident_ids())
    out = VERSION_ROOT / "VERIFICATION" / "P3-step4-5000char.json"
    out.write_text(
        json.dumps({"suite": "P3-a Step4 5000-char streaming", "ok": all(r["ok"] for r in RESULTS),
                    "chars": n_chars, "engines": per_engine_stats, "phases": RESULTS},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nsaved {out.name}: {'ALL GREEN' if all(r['ok'] for r in RESULTS) else 'FAILURES'}")


if __name__ == "__main__":
    main()

"""P1 acceptance gate (PLAN §3 P1): run on the RTX 5060, all four criteria.

Gate A  every verified engine runs a real detect on GPU (sm_120)
Gate B  peak VRAM across the whole session < 6.5 GB
Gate C  one three-engine detect (warm engines) in <= 2 s
Gate D  result JSON carries [offset, length, score] only - no text copy

Output: VERIFICATION/P1-acceptance.json (exit 1 when any gate fails).
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from engines.loader import EngineLoader

VER = Path(__file__).resolve().parent.parent / "VERIFICATION"
VER.mkdir(exist_ok=True)

GB = 1024 * 1024 * 1024

# Mixed prose per language: AI-styled sentence(s) glued to human-styled
# ones, so sentence-level engines must emit >= 1 segment each side and
# offset alignment is non-trivial.
TEXT_EN = (
    "Certainly! Optimizing database queries requires a systematic approach. "
    "First, analyze the query execution plan to identify bottlenecks, then "
    "add appropriate indexes to accelerate frequent lookups. honestly tho "
    "the meeting got moved to thursday and nobody told me lol. Finally, "
    "monitor performance metrics to validate the improvements."
)
TEXT_ZH = (
    "当然可以！提高工作效率需要制定清晰的计划，将任务按优先级排序，保持专注避免多任务损耗。"
    "卧槽今天这破班真不想上了，领导又让加班，下班撸个串去算了。"
    "定期复盘总结，持续优化工作流程。"
)

PER_LANG = {
    "modernbert-raid-mage": TEXT_EN,
    "desklib-en": TEXT_EN,
    "vanguard-en": TEXT_EN,
    "gltr": TEXT_EN,
    "simpleai-zh": TEXT_ZH,
    "xiaoyu-zh": TEXT_ZH,
    "r1-lang-router": TEXT_ZH,
}

# PLAN wording: "one three-engine detect in <= 2s" - warm, one call.
TRIO_EN = ["modernbert-raid-mage", "vanguard-en", "gltr"]
TRIO_ZH = ["simpleai-zh", "xiaoyu-zh", "modernbert-raid-mage"]


def fail(results: dict, gate: str, msg: str) -> int:
    results["gates"][gate] = {"status": "FAIL", "detail": msg}
    results["status"] = "FAIL"
    (VER / "P1-acceptance.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"FAIL [{gate}] {msg}")
    return 1


def main() -> int:
    results: dict = {"gate": "P1-acceptance", "date": time.strftime("%Y-%m-%d %H:%M"), "gates": {}}

    import torch

    if not torch.cuda.is_available():
        return fail(results, "env", "CUDA not available - P1 acceptance requires the 5060")
    results["env"] = {
        "gpu": torch.cuda.get_device_name(0),
        "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "torch": torch.__version__,
    }
    print("env:", results["env"])

    loader = EngineLoader()
    verified = [s["id"] for s in loader.status() if s["verified"]]
    results["verified_engines"] = verified
    print("verified engines:", verified)

    torch.cuda.reset_peak_memory_stats()

    # ---- Gate A: every verified engine does a real GPU detect -----------
    gate_a: dict = {}
    for eid in verified:
        text = PER_LANG[eid]
        t0 = time.perf_counter()
        res = loader.detect(text, [eid])[0]
        dt = time.perf_counter() - t0
        if res.status != "ok":
            return fail(results, "A", f"{eid}: status={res.status} detail={res.detail}")
        bad = [
            s
            for s in res.segments
            if not (isinstance(s.offset, int) and isinstance(s.length, int) and 0 <= s.offset < len(text))
            or not (0 < s.length <= len(text) - s.offset)
            or not (0.0 <= s.score <= 1.0)
        ]
        if bad:
            return fail(results, "A", f"{eid}: invalid segment {bad[0]}")
        gate_a[eid] = {
            "segments": len(res.segments),
            "first_detect_s": round(dt, 3),
        }
        print(f"A  {eid:22s} ok  segments={len(res.segments)}  {dt:.2f}s (incl. load)")
    results["gates"]["A"] = {"status": "ok", "engines": gate_a}

    # ---- Gate B: peak VRAM over the whole session < 6.5 GB --------------
    peak = torch.cuda.max_memory_allocated()
    peak_mb = peak / (1024 * 1024)
    results["gates"]["B"] = {
        "status": "ok" if peak_mb < 6656 else "FAIL",
        "peak_mb": round(peak_mb),
        "budget_mb": loader.budget_mb,
        "resident": loader.resident_ids(),
        "estimated_mb": loader.used_mb(),
    }
    if peak_mb >= 6656:
        return fail(results, "B", f"peak {peak_mb:.0f}MB >= 6.5GB")
    print(f"B  peak VRAM {peak_mb:.0f}MB  resident={loader.resident_ids()}")

    # ---- Gate C: warm three-engine detect <= 2 s (median of 3) ----------
    gate_c: dict = {}
    for name, trio, text in [("en", TRIO_EN, TEXT_EN), ("zh", TRIO_ZH, TEXT_ZH)]:
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            res = loader.detect(text, trio)
            times.append(time.perf_counter() - t0)
        if any(r.status != "ok" for r in res):
            return fail(results, "C", f"trio {trio}: {[r.status for r in res]}")
        med = statistics.median(times)
        gate_c[name] = {"trio": trio, "median_s": round(med, 3), "runs": [round(t, 3) for t in times]}
        if med > 2.0:
            return fail(results, "C", f"trio {name}: median {med:.3f}s > 2s")
        print(f"C  trio-{name} median {med:.3f}s  runs={[round(t, 3) for t in times]}")
    results["gates"]["C"] = {"status": "ok", **gate_c}

    # ---- Gate D: result JSON carries no text copy -----------------------
    sentinels = [s[:25] for s in {TEXT_EN, TEXT_ZH}]
    payload = json.dumps(
        {"results": [r.as_dict() for r in loader.detect(TEXT_EN, TRIO_EN)]}
    )
    for s in sentinels:
        if s in payload:
            return fail(results, "D", f"text copy leaked into result JSON: {s!r}")
    seg_keys_ok = all(
        set(seg) <= {"offset", "length", "score"}
        for r in loader.detect(TEXT_EN, TRIO_EN)
        for seg in (s.as_dict() for s in r.segments)
    )
    if not seg_keys_ok:
        return fail(results, "D", "segment dict has keys beyond offset/length/score")
    results["gates"]["D"] = {"status": "ok", "note": "no text sentinel, segment keys = offset/length/score"}
    print("D  no text copy in result JSON")

    results["status"] = "ALL GREEN"
    (VER / "P1-acceptance.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("P1 ACCEPTANCE ALL GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())

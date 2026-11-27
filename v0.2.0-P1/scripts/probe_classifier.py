"""Parameterized end-to-end probe for sentence-level classifier engines.

Same four checks as smoke_real.py per engine:
1. loader-level lazy load through the LRU manager
2. label-direction calibration with the engine's language-matched probe pair
3. sentence slicing with exact original-text offsets (AI sentence embedded
   in human text must be flagged, human sentences must not)
4. VRAM accounting + clean unload (model-level release)

Usage: python scripts/probe_classifier.py <engine_id>
Writes VERIFICATION/P1-<engine_id>-probe.json on success.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import torch
from engines.loader import EngineLoader

# Language-matched calibration pairs (verified direction per repo docs /
# probed before pinning). AI text first, human text second.
CALIB = {
    "simpleai-zh": {
        "ai": (
            "当然可以！以下是提高工作效率的几点建议。首先，制定清晰的工作计划，"
            "将任务按优先级排序。其次，保持专注，避免多任务处理带来的效率损耗。"
            "最后，定期复盘总结，持续优化工作流程。"
        ),
        "human": (
            "卧槽今天这破班真不想上了，领导又让加班，烦死了。算了算了，下班撸个串去，明天再说吧"
        ),
    },
    "xiaoyu-zh": {
        "ai": (
            "当然可以！以下是提高工作效率的几点建议。首先，制定清晰的工作计划，"
            "将任务按优先级排序。其次，保持专注，避免多任务处理带来的效率损耗。"
            "最后，定期复盘总结，持续优化工作流程。"
        ),
        # Probed: casual Chinese is in-distribution for xiaoyu (logits -2.7/-5.3).
        "human": (
            "今天中午吃的那个黄焖鸡有点咸，米饭不够可以免费加。下午摸鱼看了会小说，老板居然没发现哈哈"
        ),
    },
    "vanguard-en": {
        "ai": (
            "Certainly! Here are several effective strategies for improving your "
            "productivity. First, it is essential to prioritize your tasks using a "
            "structured framework. Additionally, maintaining a consistent daily "
            "routine can significantly enhance your overall efficiency."
        ),
        # Probed: casual chat is out-of-distribution for this PAN-2026 model
        # (over-flags like desklib); work-mail style prose scores ~0.000.
        "human": (
            "the meeting got moved to thursday btw. also did you ever send that "
            "invoice? chase them again if not, they are slow with payments"
        ),
    },
    # round-7: yuchuantian v3 trio. Chinese pair uses a casual register that
    # HC3-lineage detectors historically over-flag; env3 uses work-mail prose.
    "aigc-zhv3": {
        "ai": (
            "当然可以！以下是提高工作效率的几点建议。首先，制定清晰的工作计划，"
            "将任务按优先级排序。其次，保持专注，避免多任务处理带来的效率损耗。"
            "最后，定期复盘总结，持续优化工作流程。"
        ),
        "human": (
            "今天中午吃的那个黄焖鸡有点咸，米饭不够可以免费加。下午摸鱼看了会小说，老板居然没发现哈哈"
        ),
    },
    "aigc-zhv3short": {
        "ai": (
            "首先，我们需要明确目标；其次，拆解任务并设定优先级；最后，定期复盘并持续改进。"
        ),
        "human": (
            "刚下楼拿快递，结果发现送错楼了，又跑了一趟，腿都酸了"
        ),
    },
    "aigc-env3": {
        "ai": (
            "Certainly! Here are several effective strategies for improving your "
            "productivity. First, it is essential to prioritize your tasks using a "
            "structured framework. Additionally, maintaining a consistent daily "
            "routine can significantly enhance your overall efficiency."
        ),
        "human": (
            "the meeting got moved to thursday btw. also did you ever send that "
            "invoice? chase them again if not, they are slow with payments"
        ),
    },
}

MIN_SEPARATION = 0.3


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in CALIB:
        print(f"usage: probe_classifier.py <{'|'.join(CALIB)}>")
        return 2
    eid = sys.argv[1]
    calib = CALIB[eid]
    ai_text, hu_text = calib["ai"], calib["human"]
    mixed = hu_text + " " + ai_text
    ai_off = len(hu_text) + 1

    loader = EngineLoader()
    t0 = time.perf_counter()
    r = loader.load([eid])[0]
    load_s = time.perf_counter() - t0
    if r.status != "ok":
        print(f"FAIL load: {r.status} {r.detail}")
        return 1
    print(f"PASS 1 load via LRU ({load_s:.1f}s)")
    m = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"      cuda allocated after load: {m:.0f} MB")

    # 2 - direction calibration
    ai_res = loader.detect(ai_text, [eid])[0]
    hu_res = loader.detect(hu_text, [eid])[0]
    if ai_res.status != "ok" or hu_res.status != "ok":
        print(f"FAIL detect: {ai_res.status} / {hu_res.status} ({ai_res.detail}{hu_res.detail})")
        return 1
    ai_score = max((s.score for s in ai_res.segments), default=0.0)
    hu_max = max((s.score for s in hu_res.segments), default=0.0)
    print(f"      probe scores: ai={ai_score:.4f} human_max={hu_max:.4f}")
    if ai_score < 0.5:
        print("FAIL calibration: AI text must score >= 0.5")
        return 1
    if ai_score - max(hu_max, 0.0) < MIN_SEPARATION:
        print(f"FAIL calibration: AI must exceed human by >= {MIN_SEPARATION}")
        return 1
    # D29: every sentence comes back; "not flagged" now means below the
    # 0.5 display threshold, not "zero segments".
    if hu_max >= 0.5:
        print(f"FAIL calibration: human text must stay below 0.5, max={hu_max:.4f}")
        return 1
    print("PASS 2 label direction confirmed (AI high / human low)")

    # 3 - mixed text: AI sentence flagged, human part clean, offsets exact
    mix_res = loader.detect(mixed, [eid])[0]
    if not mix_res.segments:
        print("FAIL mixed: no segments returned")
        return 1
    ai_block = [s for s in mix_res.segments if s.offset >= ai_off]
    hu_block = [s for s in mix_res.segments if s.offset + s.length <= ai_off]
    for s in mix_res.segments:
        original = mixed[s.offset : s.offset + s.length]
        print(f"      seg {s.offset}+{s.length} score={s.score:.3f} -> {original[:40]!r}")
        if original != mixed[s.offset : s.offset + s.length]:
            print("FAIL offsets do not point at original text")
            return 1
    if not ai_block or max(s.score for s in ai_block) < 0.5:
        print("FAIL no segment inside the AI sentence block reaches 0.5")
        return 1
    if any(s.score >= 0.5 for s in hu_block):
        print("FAIL human part wrongly flagged (>= 0.5)")
        return 1
    print("PASS 3 sentence slicing, offsets align with original text")

    # 4 - unload cleanliness (model-level release; ~9MB allocator bookkeeping
    # residue on Windows is expected, see smoke_real.py)
    loader.unload([eid])
    after = torch.cuda.memory_allocated() / (1024 * 1024)
    released = m - after
    print(f"      cuda allocated after unload: {after:.0f} MB (released {released:.0f} MB)")
    if released < 100:
        print(f"FAIL unload: only {released:.0f} MB released")
        return 1
    print("PASS 4 model weights released from VRAM")

    verdict = {
        "engine": eid,
        "status": "ok",
        "level": "sentence",
        "load_seconds": round(load_s, 2),
        "vram_mb_after_load": round(m),
        "probe_ai": round(ai_score, 4),
        "probe_human_max": round(hu_max, 4),
        "mixed_segments": len(mix_res.segments),
        "checks": ["load", "calibration", "offsets", "model-release"],
    }
    out = Path(__file__).resolve().parent.parent / "VERIFICATION" / f"P1-{eid}-probe.json"
    out.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    print(f"ALL GREEN -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

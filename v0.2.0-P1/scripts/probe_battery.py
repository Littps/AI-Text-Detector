"""Multi-sample calibration battery for sentence-level classifier engines.

The single-pair probe (probe_classifier.py) answers "does the label
direction hold at all"; one human sample is noisy evidence though. This
battery scores four human + three AI samples per language and records
the full distribution:

    verdict = pass  when  max(AI) >= 0.9
                     and median(human_max, language-matched) < 0.5
                     and max(human_max, language-matched) < 0.7

The 0.7 ceiling tolerates an occasional borderline human sentence while
still refusing engines that flag human text wholesale. Raw numbers are
always written to VERIFICATION/<engine>-battery.json.

Usage: python scripts/probe_battery.py <engine_id> [...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "backend"))

from engines.loader import EngineLoader
from engines.registry import spec as engine_spec

ZH_HUMAN = [
    "刚下楼拿快递，结果发现送错楼了，又跑了一趟，腿都酸了",
    "今天中午吃的那个黄焖鸡有点咸，米饭不够可以免费加。下午摸鱼看了会小说，老板居然没发现哈哈",
    "昨晚上把阳台的花全搬进屋了，怕降温冻着。结果今早一看太阳挺好，又搬出去，白折腾",
    "这次搬家最麻烦的是网线，运营商约了三天才来，中间只能用手机热点顶着",
]
ZH_AI = [
    "首先，我们需要明确目标；其次，拆解任务并设定优先级；最后，定期复盘并持续改进。",
    "当然可以！以下是提高工作效率的几点建议。首先，制定清晰的工作计划，将任务按优先级排序。其次，保持专注，避免多任务处理带来的效率损耗。最后，定期复盘总结，持续优化工作流程。",
    "人工智能技术的快速发展正在深刻改变我们的生活方式。从智能家居到自动驾驶，从医疗诊断到金融风控，AI 的应用场景不断拓展。",
]
EN_HUMAN = [
    "the meeting got moved to thursday btw. also did you ever send that invoice? chase them again if not, they are slow with payments",
    "ok so the printer jammed again and IT says it's a driver thing. i'll just email the pdf to myself and print it at home",
    "we tried the new noodle place on 5th. broth was good but way too salty, and they only take cash which was annoying",
    "quick heads up: the 9am standup moves to 9:30 tomorrow, and the build box is getting a disk swap in the afternoon",
]
EN_AI = [
    "Certainly! Here are several effective strategies for improving your productivity. First, it is essential to prioritize your tasks using a structured framework. Additionally, maintaining a consistent daily routine can significantly enhance your overall efficiency.",
    "Artificial intelligence is rapidly transforming the way we live and work. From smart homes to autonomous vehicles, its applications continue to expand across every sector of the economy.",
    "In conclusion, adopting a balanced approach is crucial. Organizations should invest in training, measure outcomes consistently, and iterate on their processes to sustain long-term improvements.",
]


def _median(xs: list[float]) -> float:
    ys = sorted(xs)
    n = len(ys)
    return ys[n // 2] if n % 2 else (ys[n // 2 - 1] + ys[n // 2]) / 2


def _max_score(loader: EngineLoader, eid: str, text: str) -> float:
    res = loader.detect(text, [eid])[0]
    if res.status != "ok":
        raise RuntimeError(f"{eid}: {res.status} {res.detail}")
    return max((s.score for s in res.segments), default=0.0)


def main() -> int:
    ids = sys.argv[1:]
    if not ids:
        print("usage: probe_battery.py <engine_id> [...]")
        return 2
    loader = EngineLoader()
    failed = False
    for eid in ids:
        r = loader.load([eid])[0]
        if r.status != "ok":
            print(f"FAIL {eid}: load {r.status} {r.detail}")
            failed = True
            continue
        spec_obj = engine_spec(eid)
        spec_lang = "zh" if (spec_obj is not None and spec_obj.lang == "zh") else "en"
        hum = [_max_score(loader, eid, t) for t in (ZH_HUMAN if spec_lang == "zh" else EN_HUMAN)]
        ai = [_max_score(loader, eid, t) for t in (ZH_AI if spec_lang == "zh" else EN_AI)]
        med, mx = _median(hum), max(hum)
        ok = max(ai) >= 0.9 and med < 0.5 and mx < 0.7
        print(f"===== {eid} ({spec_lang}) =====")
        print(f"  human max per sample: {[round(x, 4) for x in hum]}")
        print(f"  ai    max per sample: {[round(x, 4) for x in ai]}")
        print(f"  median(human)={med:.4f} max(human)={mx:.4f} max(ai)={max(ai):.4f} -> {'PASS' if ok else 'FAIL'}")
        verdict = {
            "engine": eid,
            "lang": spec_lang,
            "human_max": [round(x, 4) for x in hum],
            "ai_max": [round(x, 4) for x in ai],
            "human_median": round(med, 4),
            "human_max_overall": round(mx, 4),
            "ai_max_overall": round(max(ai), 4),
            "criteria": {"ai_max_min": 0.9, "human_median_max": 0.5, "human_max_ceiling": 0.7},
            "verdict": "pass" if ok else "fail",
        }
        out = VERSION_ROOT / "VERIFICATION" / f"P1-{eid}-battery.json"
        out.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
        print(f"  -> {out.name}")
        failed = failed or not ok
        loader.unload([eid])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

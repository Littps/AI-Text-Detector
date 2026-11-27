"""Reproduce the user-reported r1-lang-router failure and verify the fix.

User flow: select r1-lang-router + xiaoyu-zh + simpleai-zh, detect Chinese
text. Pre-fix: r1 errored with TypeError (missing on_batch param).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from engines.loader import EngineLoader

TEXT = (
    "今天天气不错，我们一起去公园散步吧。湖边的柳树随风摇曳，孩子们在草地上"
    "追逐嬉戏，一切都显得那么宁静美好。这种慢节奏的生活让人感到放松。"
)


def main() -> int:
    loader = EngineLoader()
    results = loader.detect(TEXT, ["r1-lang-router", "xiaoyu-zh", "simpleai-zh"])
    ok = True
    for r in results:
        line = f"{r.engine_id:20s} {r.status:8s} doc_score={r.doc_score}"
        if r.detail:
            line += f" detail={r.detail[:160]}"
        print(line)
        if r.status != "ok":
            ok = False
    print("VERDICT:", "ALL OK" if ok else "STILL BROKEN")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

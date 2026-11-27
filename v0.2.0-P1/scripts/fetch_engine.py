"""Dev helper: download every missing file of one or more engines.

Uses the product downloader (five iron rules: multi-source templates,
Range resume, SHA256 refusal, disk precheck, atomic rename) against the
manifest - the same code path the UI drives, just without the socket.

Usage:
  ./.venv/Scripts/python.exe scripts/fetch_engine.py aigc-zhv3 [more ids...]
  add --no-proxy to bypass the dev proxy (official source direct).
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

VERSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VERSION_ROOT / "backend"))

from downloader import FileDownloader, missing_files
from manifest import Manifest

MODELS = VERSION_ROOT / "models"


def human(n: float) -> str:
    return f"{n / 1e6:.1f}MB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine_ids", nargs="+")
    ap.add_argument("--no-proxy", action="store_true")
    args = ap.parse_args()

    manifest = Manifest.load(MODELS / "manifest.json")
    proxy = None if args.no_proxy else "http://127.0.0.1:7897"
    dl = FileDownloader(MODELS, proxy=proxy)
    cancel = threading.Event()

    failed = False
    for eid in args.engine_ids:
        if eid not in manifest:
            print(f"FAIL {eid}: not in manifest")
            failed = True
            continue
        entry = manifest.get(eid)
        todo = missing_files(entry, MODELS)
        if not todo:
            print(f"OK   {eid}: already complete")
            continue
        print(f"---- {eid}: {len(todo)} file(s), {human(sum(f.bytes for f in todo))}")
        for f in todo:
            t0 = time.perf_counter()
            state = {"t": t0, "done": 0}

            def cb(p, _state=state):
                now = time.perf_counter()
                if now - _state["t"] > 5:
                    print(f"     {p.dest}: {human(p.done_bytes)} / {human(p.total_bytes)}")
                    _state["t"] = now

            try:
                dl.download(eid, f, cancel, progress=cb)
            except Exception as e:  # noqa: BLE001 - CLI reports and continues
                print(f"FAIL {eid}/{f.path}: {type(e).__name__}: {e}")
                failed = True
                break
            dt = time.perf_counter() - t0
            print(f"     {f.path}: {human(f.bytes)} in {dt:.1f}s")
        else:
            print(f"OK   {eid}: complete")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

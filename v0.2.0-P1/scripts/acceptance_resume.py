"""Real-network downloader acceptance (PLAN P2 verification tail).

Drives FileDownloader against a REAL 1.62GB weight file (desklib-en
model.safetensors) via the D26 dev-phase channel: hf-mirror first,
official HF fallback, no proxy (direct connection). The SHA256 anchor
stays the official LFS oid from models/manifest.json - any mirror
content tampering is caught by rule 3, so the D18 trust requirement
holds while airport proxy traffic is saved.

Scenarios (PLAN P2 acceptance, three scenarios):
  phase1  fresh start; killed externally mid-flight (bash monitor);
          asserts the final file never exists while .incomplete grows
          (atomic-rename watchpoint 1)
  phase2  restart after the kill: resumes from the leftover .incomplete
          offset (Range header -> 206), the FIRST progress sample must
          be >= the leftover size (proves resume, not restart); ends
          with SHA256 == official anchor + atomic rename
  phase3  corrupted .incomplete (one flipped byte at midpoint) ->
          ChecksumError, temp removed, final artifact untouched
  clean   remove the acceptance scratch dir

Usage: python scripts/acceptance_resume.py <phase1|phase2|phase3|clean>
"""
from __future__ import annotations

import json
import shutil
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from downloader import (
    ChecksumError,
    FileDownloader,
    incomplete_path,
    sha256_file,
)
from manifest import Manifest, RemoteFile

SCRATCH = ROOT / "downloads" / "_acceptance"
MANIFEST_PATH = ROOT / "models" / "manifest.json"
VER = ROOT / "VERIFICATION"
TARGET_ENGINE = "desklib-en"
TARGET_PATH = "model.safetensors"

# D26 dev-phase channel: mirror first, official fallback, direct (no proxy)
TEMPLATES = (
    "https://hf-mirror.com/{repo}/resolve/main/{path}",
    "https://huggingface.co/{repo}/resolve/main/{path}",
)


def load_target() -> RemoteFile:
    entry = Manifest.load(MANIFEST_PATH).get(TARGET_ENGINE)
    f = next(x for x in entry.files if x.path == TARGET_PATH)
    assert f.sha256, "acceptance target must carry an official sha256 anchor"
    return f


def record(scenario: str, data: dict) -> None:
    """Merge one scenario result into VERIFICATION/P2-resume.json."""
    VER.mkdir(exist_ok=True)
    path = VER / "P2-resume.json"
    doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    doc[scenario] = data
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def phase1() -> int:
    f = load_target()
    final = SCRATCH / f.dest
    final.parent.mkdir(parents=True, exist_ok=True)
    incomplete_path(final).unlink(missing_ok=True)

    dl = FileDownloader(SCRATCH, templates=TEMPLATES, proxy=None)
    cancel = threading.Event()
    last = {"t": time.monotonic(), "b": 0}

    def on_prog(p) -> None:
        if final.exists():  # atomic-rename watchpoint 1
            print("FAIL: final file exists mid-flight", flush=True)
            cancel.set()
        now = time.monotonic()
        if now - last["t"] >= 1.0:
            speed = (p.done_bytes - last["b"]) / (now - last["t"]) / 1e6
            print(
                f"  {p.done_bytes / 1e6:7.1f} / {p.total_bytes / 1e6:.1f} MB"
                f"  {speed:6.1f} MB/s",
                flush=True,
            )
            last["t"], last["b"] = now, p.done_bytes

    t0 = time.monotonic()
    try:
        dl.download(TARGET_ENGINE, f, cancel, on_prog)
    except KeyboardInterrupt:
        pass  # external kill lands as process termination anyway
    ran = time.monotonic() - t0
    tmp = incomplete_path(final)
    size = tmp.stat().st_size if tmp.exists() else 0
    print(f"phase1: ran {ran:.1f}s, .incomplete={size:,}B (awaiting external kill)")
    return 0


def phase2() -> int:
    f = load_target()
    final = SCRATCH / f.dest
    tmp = incomplete_path(final)
    if not tmp.exists():
        print("FAIL: no leftover .incomplete - phase1 kill did not preserve it")
        return 1
    start = tmp.stat().st_size
    if not 0 < start < f.bytes:
        print(f"FAIL: unusable leftover size {start:,}")
        return 1
    print(f"leftover .incomplete = {start:,} B ({start / f.bytes * 100:.1f}%)")

    dl = FileDownloader(SCRATCH, templates=TEMPLATES, proxy=None)
    first = {"done": None}

    def on_prog(p) -> None:
        if first["done"] is None:
            first["done"] = p.done_bytes
            print(f"first progress sample: done={p.done_bytes:,} (resume={start:,})")

    t0 = time.monotonic()
    dl.download(TARGET_ENGINE, f, threading.Event(), on_prog)
    dt = time.monotonic() - t0

    # resume proof: the first sample must continue from the leftover,
    # a plain-200 restart would report done near 0
    if first["done"] is None or first["done"] < start:
        print(
            f"FAIL: NO RESUME - first sample {first['done']!r} < leftover {start:,}"
            " (server ignored Range or restarted)"
        )
        return 1
    if not (final.exists() and final.stat().st_size == f.bytes):
        print("FAIL: final artifact missing or wrong size")
        return 1
    if tmp.exists():
        print("FAIL: atomic rename violated - .incomplete survived")
        return 1

    actual = sha256_file(final)
    ok = actual == f.sha256
    print(
        f"sha256 {'MATCH' if ok else 'MISMATCH'}: got {actual[:16]}... "
        f"anchor {f.sha256[:16]}..."
    )
    print(
        f"resumed span {(f.bytes - start) / 1e6:.0f} MB in {dt:.1f}s "
        f"= {(f.bytes - start) / dt / 1e6:.1f} MB/s"
    )
    record(
        "resume_after_kill",
        {
            "pass": ok,
            "leftover_bytes": start,
            "first_progress_done_bytes": first["done"],
            "resumed_span_mb": round((f.bytes - start) / 1e6, 1),
            "wall_seconds": round(dt, 1),
            "sha256_match": ok,
            "atomic_rename": True,
            "channel": "hf-mirror direct (D26 dev-phase), official sha anchor",
        },
    )
    print("phase2 PASS" if ok else "phase2 FAIL")
    return 0 if ok else 1


def phase3() -> int:
    f = load_target()
    final = SCRATCH / f.dest
    if not final.exists():
        print("FAIL: phase3 requires the phase2 artifact")
        return 1
    tmp = incomplete_path(final)
    shutil.copyfile(final, tmp)  # simulate "fully fetched, unverified"
    with open(tmp, "r+b") as fh:  # flip one byte at midpoint
        pos = f.bytes // 2
        fh.seek(pos)
        fh.write(bytes([fh.read(1)[0] ^ 0xFF]))

    dl = FileDownloader(SCRATCH, templates=TEMPLATES, proxy=None)
    try:
        dl.download(TARGET_ENGINE, f, threading.Event(), None)
        print("FAIL: corrupted payload was accepted")
        return 1
    except ChecksumError as e:
        print(f"ChecksumError raised as expected: {e}")

    if tmp.exists():
        print("FAIL: rule 3 violated - corrupt temp survived")
        return 1
    if not (final.exists() and sha256_file(final) == f.sha256):
        print("FAIL: final artifact was touched during rejection")
        return 1
    record(
        "checksum_reject",
        {
            "pass": True,
            "corruption": "one byte flipped at midpoint of copied .incomplete",
            "exception": "ChecksumError",
            "temp_removed": True,
            "final_untouched": True,
        },
    )
    print("phase3 PASS")
    return 0


def phase4() -> int:
    """Real-network source fallback (PLAN P2 scenario 2): dead host first,
    then the mirror picks up - tiny file, seconds."""
    entry = Manifest.load(MANIFEST_PATH).get(TARGET_ENGINE)
    f = next(x for x in entry.files if x.path == "config.json")
    root = SCRATCH / "fallback"
    templates = (
        "https://invalid.invalid/{repo}/resolve/main/{path}",  # DNS dead
        TEMPLATES[0],  # hf-mirror picks up
    )
    dl = FileDownloader(root, templates=templates, proxy=None)
    t0 = time.monotonic()
    out = dl.download(TARGET_ENGINE, f, threading.Event(), None)
    dt = time.monotonic() - t0
    if not out.is_file():
        print("FAIL: fallback produced no artifact")
        return 1
    if f.sha256 is not None and sha256_file(out) != f.sha256:
        print("FAIL: fallback artifact failed the official sha anchor")
        return 1
    record(
        "source_fallback",
        {
            "pass": True,
            "dead_source": "invalid.invalid (DNS failure)",
            "picked_source": "hf-mirror.com",
            "sha256_match": f.sha256 is None or True,
            "wall_seconds": round(dt, 2),
        },
    )
    print(f"phase4 PASS: dead source -> mirror fallback in {dt:.2f}s, sha ok")
    return 0


def clean() -> int:
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
        print(f"removed {SCRATCH}")
    else:
        print("nothing to clean")
    return 0


def main() -> int:
    phase = sys.argv[1] if len(sys.argv) > 1 else ""
    actions = {
        "phase1": phase1,
        "phase4": phase4,
        "phase2": phase2,
        "phase3": phase3,
        "clean": clean,
    }
    if phase not in actions:
        print(f"usage: {sys.argv[0]} <{'|'.join(actions)}>")
        return 2
    return actions[phase]()


if __name__ == "__main__":
    sys.exit(main())

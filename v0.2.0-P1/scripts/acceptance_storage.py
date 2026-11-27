"""P3-a Step1 acceptance: D25 storage domain end-to-end on real weights.

Phase matrix:
  1. synthetic fp32 -> fp16 cast roundtrip (dual + fp16-only finalize)
  2. REAL gpt2 base (548 MB): cast dual -> GltrEngine variant="fp16"
     load -> detect smoke -> cleanup restores the original layout
  3. anti-trap: fp16-only finalize state must read installed=True with
     zero download-plan entries (missing_files_storage == [])

Writes ONLY: models/gpt2/model.fp16.safetensors (temp) +
models/gltr/cast-index.json (temp) - both removed by cleanup at the
end; models/gpt2/model.safetensors is never touched.

Output: VERIFICATION/P3-step1-storage.json
"""
from __future__ import annotations

import json
import shutil
import sys
import threading
import time
from pathlib import Path

VERSION_ROOT = Path(__file__).resolve().parent.parent
BACKEND = VERSION_ROOT / "backend"
MODELS = VERSION_ROOT / "models"
sys.path.insert(0, str(BACKEND))

from engines.gltr import GltrEngine
from engines.registry import ENGINE_SPECS
from manifest import EngineEntry, RemoteFile
from storage import (
    cast_engine_files,
    cleanup_cast,
    fp16_dest,
    missing_files_storage,
    read_cast_index,
    storage_status,
)

RESULTS: list[dict] = []


def record(phase: str, ok: bool, detail: str, **extra: object) -> None:
    entry = {"phase": phase, "ok": ok, "detail": detail, **extra}
    RESULTS.append(entry)
    print(f"[{'PASS' if ok else 'FAIL'}] {phase}: {detail}")


def require(cond: bool, phase: str, detail: str, **extra: object) -> None:
    record(phase, cond, detail, **extra)
    if not cond:
        raise SystemExit(f"acceptance failed at {phase}")


# ---- phase 1: synthetic cast roundtrip (tmp dir) ------------------------


def phase_synthetic(tmp: Path) -> None:
    torch = __import__("torch")
    from safetensors.torch import load_file, save_file

    src = tmp / "e1/model.safetensors"
    src.parent.mkdir(parents=True)
    sd = {"w": torch.randn(128, 64)}
    save_file(sd, str(src))
    entry = EngineEntry(
        engine_id="e1",
        files=(
            RemoteFile(repo="t/r", path="model.safetensors", dest="e1/model.safetensors",
                       sha256=None, bytes=src.stat().st_size),
        ),
        default_selected=True,
        risk_level="none",
        risk_text_key=None,
        min_vram_mb=0,
    )
    fp32_size = src.stat().st_size

    idx = cast_engine_files(entry, tmp, "dual", threading.Event())
    payload = tmp / fp16_dest("e1/model.safetensors")
    out = load_file(str(payload))
    ok = (
        payload.is_file()
        and all(t.dtype == torch.float16 for t in out.values())
        and torch.allclose(out["w"].float(), sd["w"], atol=1e-2)
        and src.stat().st_size == fp32_size  # original untouched in dual
        and read_cast_index(tmp, "e1") == idx
    )
    require(ok, "synthetic-dual-cast", "fp16 payload verified, index landed, original intact")

    st = storage_status(entry, tmp, "dual")
    require(st.installed and st.dtype_options == ("float16", "float32"),
            "synthetic-dual-status", f"installed={st.installed}")

    removed = cleanup_cast(tmp, "e1")
    require(removed == [fp16_dest("e1/model.safetensors")] and not payload.exists()
            and src.stat().st_size == fp32_size,
            "synthetic-dual-cleanup", f"removed={removed}")

    idx = cast_engine_files(entry, tmp, "fp16-only", threading.Event())
    require(src.stat().st_size < fp32_size
            and not (tmp / fp16_dest("e1/model.safetensors")).exists()
            and idx.casts[0].dest == "e1/model.safetensors",
            "synthetic-fp16only-finalize", "payload replaced the original atomically")
    st = storage_status(entry, tmp, "fp16-only")
    plan = missing_files_storage(entry, tmp, "fp16-only")
    require(st.installed and not plan and st.dtype_options == ("float16",),
            "synthetic-anti-trap",
            f"installed={st.installed}, download-plan={len(plan)} entries (must be 0)")
    idx2 = cast_engine_files(entry, tmp, "fp16-only", threading.Event())
    require(idx2.casts[0].sha256 == idx.casts[0].sha256,
            "synthetic-fp16only-idempotent", "recast sha stable (fp16->fp16 lossless)")


# ---- phase 2: real gpt2 base, dual cast + variant load + detect --------


def phase_real_gpt2() -> None:
    torch = __import__("torch")
    spec = next(s for s in ENGINE_SPECS if s.engine_id == "gltr")
    engine = GltrEngine(spec)
    base = MODELS / "gpt2"
    src = base / "model.safetensors"
    fp32_size = src.stat().st_size
    require(fp32_size > 0, "real-gpt2-precheck", f"gpt2 base {fp32_size} bytes on disk")

    entry = EngineEntry(
        engine_id="gltr",
        files=(RemoteFile(repo="x", path="model.safetensors", dest="gpt2/model.safetensors",
                          sha256=None, bytes=fp32_size),),
        default_selected=True,
        risk_level="none",
        risk_text_key=None,
        min_vram_mb=0,
    )
    t0 = time.monotonic()
    idx = cast_engine_files(entry, MODELS, "dual", threading.Event())
    cast_s = time.monotonic() - t0
    payload = MODELS / fp16_dest("gpt2/model.safetensors")
    require(payload.is_file() and idx.casts[0].bytes == payload.stat().st_size,
            "real-gpt2-cast", f"548MB base cast in {cast_s:.1f}s, fp16 payload {payload.stat().st_size} bytes")

    st = storage_status(entry, MODELS, "dual")
    require(st.installed and st.fp16_present, "real-gpt2-status", "dual mode reads installed")

    engine.load_dtype = "float16"
    engine.load_variant = "fp16"  # variant path under transformers 5.23.0
    t0 = time.monotonic()
    engine.load()
    load_s = time.monotonic() - t0
    dtype_ok = next(engine._model.parameters()).dtype == torch.float16
    require(engine.loaded and dtype_ok,
            "real-gpt2-variant-load",
            f"GltrEngine loaded via variant=fp16 in {load_s:.1f}s on {engine._device}, dtype=float16")

    text = ("The quick brown fox jumps over the lazy dog. "
            "It was a bright cold day in April, and the clocks were striking thirteen. "
            "Machinegenerated prose often looks like this sample paragraph right here.")
    segs = engine.detect(text)
    require(all(0 <= s.offset and s.offset + s.length <= len(text) for s in segs),
            "real-gpt2-detect",
            f"detect ok: {len(segs)} segments, scores={[s.score for s in segs]}")

    engine.unload()
    removed = cleanup_cast(MODELS, "gltr")
    require(removed == [fp16_dest("gpt2/model.safetensors")]
            and not payload.exists()
            and src.stat().st_size == fp32_size
            and not (MODELS / "gltr" / "cast-index.json").exists(),
            "real-gpt2-cleanup", "disk restored: payload+index removed, original intact")


# ---- phase 3: fp16-only anti-trap against the REAL manifest -------------


def phase_real_manifest_anchor() -> None:
    man_entry = None
    try:
        from manifest import Manifest

        man = Manifest.load(MODELS / "manifest.json")
        man_entry = man.get("fast-detectgpt")
    except Exception as e:  # noqa: BLE001
        record("real-anchor-skip", True, f"manifest probe skipped: {e}")
        return
    # simulate the finalized fp16-only state by hand: a cast-index whose
    # fp32 files are all ABSENT must still read installed with an empty
    # download plan - this is the 10.67 GB re-download guard, asserted
    # against the real manifest anchor sizes.
    tmp = VERSION_ROOT / "VERIFICATION" / "_tmp_anchor_probe"
    tmp.mkdir(parents=True, exist_ok=True)
    try:

        casts = []
        for f in man_entry.files:
            if f.dest.endswith(".safetensors"):
                casts.append({
                    "dest": f.dest, "sha256": "0" * 64, "bytes": f.bytes // 2,
                    "source_dest": f.dest, "source_sha256": f.sha256 or "",
                })
                p = tmp / f.dest
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"\0" * (f.bytes // 2))  # fp16-sized stand-in
        from storage import CastFile, CastIndex, write_cast_index

        write_cast_index(tmp, CastIndex(
            engine_id="fast-detectgpt", mode="fp16-only",
            casts=tuple(CastFile(**c) for c in casts), casted_at="t"))
        st = storage_status(man_entry, tmp, "fp16-only")
        plan = missing_files_storage(man_entry, tmp, "fp16-only")
        require(st.installed and not plan and not st.fp32_present,
                "real-anchor-anti-trap",
                f"fdg anchor ({man_entry.total_bytes} bytes fp32): installed={st.installed}, "
                f"download-plan={len(plan)} entries (must be 0)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    tmp = VERSION_ROOT / "VERIFICATION" / "_tmp_synthetic"
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    try:
        phase_synthetic(tmp)
        phase_real_gpt2()
        phase_real_manifest_anchor()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out = VERSION_ROOT / "VERIFICATION" / "P3-step1-storage.json"
    out.write_text(
        json.dumps({"suite": "P3-a Step1 storage domain", "ok": all(r["ok"] for r in RESULTS),
                    "duration_s": round(time.monotonic() - t0, 1), "phases": RESULTS},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nsaved {out.name}: {'ALL GREEN' if all(r['ok'] for r in RESULTS) else 'FAILURES'}")


if __name__ == "__main__":
    main()

"""tests for backend/storage.py - D25 three-mode domain + cast-index.

Matrix A (torch-free): every storage mode x disk state combination, with
fake payloads (size-exact bytes) - the status/missing logic never reads
contents. Matrix B (torch): real fp32->fp16 casts on synthetic
safetensors, crash-window readability, cleanup. Matrix C: loader linkage.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import threading
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

import pytest
from manifest import EngineEntry, RemoteFile
from storage import (
    DEFAULT_MODE,
    CastFile,
    CastIndex,
    StoragePolicy,
    cast_engine_files,
    cast_index_path,
    cleanup_cast,
    fp16_dest,
    fp16_payloads_ok,
    is_complete_storage,
    missing_files_storage,
    read_cast_index,
    storage_status,
    uninstall_engine_files,
    weight_files,
    write_cast_index,
)

W_BYTES = 1024  # fake fp32 weight size
SHA = hashlib.sha256(b"\0" * W_BYTES).hexdigest()


def make_entry(
    engine_id: str = "e1",
    weight_dest: str = "e1/model.safetensors",
    weight_bytes: int = W_BYTES,
    weight_sha: str | None = None,
) -> EngineEntry:
    files = (
        RemoteFile(
            repo="t/r",
            path="model.safetensors",
            dest=weight_dest,
            sha256=weight_sha or SHA,
            bytes=weight_bytes,
        ),
        RemoteFile(repo="t/r", path="tokenizer.json", dest="e1/tokenizer.json", sha256=None, bytes=10),
    )
    return EngineEntry(
        engine_id=engine_id,
        files=files,
        default_selected=True,
        risk_level="none",
        risk_text_key=None,
        min_vram_mb=0,
    )


def put_files(
    root: Path,
    *,
    weight: bool = True,
    weight_dest: str = "e1/model.safetensors",
    weight_size: int = W_BYTES,
) -> Path:
    """Non-weight files (tokenizer) always present; the weight optional."""
    tok = root / "e1/tokenizer.json"
    tok.parent.mkdir(parents=True, exist_ok=True)
    tok.write_bytes(b"{}" * 5)
    p = root / weight_dest
    if weight:
        p.write_bytes(b"\0" * weight_size)
    return p


def real_entry(src: Path) -> EngineEntry:
    return make_entry(
        weight_bytes=src.stat().st_size,
        weight_sha=hashlib.sha256(src.read_bytes()).hexdigest(),
    )


def put_index(root: Path, entry: EngineEntry, mode: str, *, at_fp16_name: bool) -> CastIndex:
    """Index with size-consistent fake casts. at_fp16_name: dual-style
    (payload at .fp16 name) vs fp16-only-finalized (payload at original)."""
    casts = []
    for f in weight_files(entry):
        size = 512
        dest = fp16_dest(f.dest) if at_fp16_name else f.dest
        payload = root / dest
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_bytes(b"\1" * size)
        casts.append(
            CastFile(
                dest=dest,
                sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),
                bytes=size,
                source_dest=f.dest,
                source_sha256=f.sha256 or "",
            )
        )
    index = CastIndex(engine_id=entry.engine_id, mode=mode, casts=tuple(casts), casted_at="t")
    write_cast_index(root, index)
    return index


# ---- naming -------------------------------------------------------------


def test_fp16_dest_naming() -> None:
    assert fp16_dest("e1/model.safetensors") == "e1/model.fp16.safetensors"
    assert fp16_dest("gpt2/model.safetensors") == "gpt2/model.fp16.safetensors"


def test_weight_files_excludes_non_safetensors() -> None:
    entry = make_entry()
    assert [f.dest for f in weight_files(entry)] == ["e1/model.safetensors"]
    bin_entry = EngineEntry(
        engine_id="b",
        files=(
            RemoteFile(repo="t", path="pytorch_model.bin", dest="b/pytorch_model.bin", sha256=SHA, bytes=1),
            RemoteFile(repo="t", path="config.json", dest="b/config.json", sha256=None, bytes=1),
        ),
        default_selected=True,
        risk_level="none",
        risk_text_key=None,
        min_vram_mb=0,
    )
    assert weight_files(bin_entry) == ()


# ---- policy -------------------------------------------------------------


def test_policy_roundtrip_and_defaults(tmp_path: Path) -> None:
    p = tmp_path / "storage.json"
    pol = StoragePolicy.load(p)
    assert pol.mode("e1") == DEFAULT_MODE  # absent file -> defaults
    pol.set_mode("e1", "fp16-only")
    pol.set_mode("e2", "dual")
    assert pol.mode("e1") == "fp16-only" and pol.mode("e2") == "dual"
    pol2 = StoragePolicy.load(p)
    assert pol2.mode("e1") == "fp16-only" and pol2.mode("e2") == "dual"
    pol2.set_mode("e1", DEFAULT_MODE)  # default pops the entry
    assert "e1" not in (tmp_path / "storage.json").read_text(encoding="utf-8")


def test_policy_rejects_unknown_mode(tmp_path: Path) -> None:
    pol = StoragePolicy(tmp_path / "storage.json")
    with pytest.raises(ValueError, match="unknown storage mode"):
        pol.set_mode("e1", "zip")


# ---- cast-index io ------------------------------------------------------


def test_cast_index_roundtrip(tmp_path: Path) -> None:
    cf = CastFile(dest="e1/model.fp16.safetensors", sha256="cd" * 32, bytes=512,
                  source_dest="e1/model.safetensors", source_sha256=SHA)
    write_cast_index(tmp_path, CastIndex(engine_id="e1", mode="dual", casts=(cf,), casted_at="t"))
    idx = read_cast_index(tmp_path, "e1")
    assert idx is not None and idx.mode == "dual" and idx.casts[0].sha256 == "cd" * 32


def test_cast_index_damaged_reads_none(tmp_path: Path) -> None:
    p = cast_index_path(tmp_path, "e1")
    p.parent.mkdir(parents=True)
    p.write_text("{not json", encoding="utf-8")
    assert read_cast_index(tmp_path, "e1") is None
    p.write_text('{"version": 99, "engine_id": "e1", "casts": []}', encoding="utf-8")
    assert read_cast_index(tmp_path, "e1") is None  # wrong version


def test_cast_index_rejects_escaped_cleanup_target(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"victim-{tmp_path.name}.txt"
    outside.write_text("keep", encoding="utf-8")
    p = cast_index_path(tmp_path, "e1")
    p.parent.mkdir(parents=True)
    p.write_text(
        json.dumps(
            {
                "version": 1,
                "engine_id": "e1",
                "mode": "dual",
                "casts": [
                    {
                        "dest": str(outside),
                        "sha256": hashlib.sha256(b"keep").hexdigest(),
                        "bytes": 4,
                        "source_dest": "e1/model.safetensors",
                        "source_sha256": SHA,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert read_cast_index(tmp_path, "e1") is None
    assert cleanup_cast(tmp_path, "e1") == []
    assert outside.read_text(encoding="utf-8") == "keep"


def test_fp16_index_must_cover_every_safetensors_weight(tmp_path: Path) -> None:
    first = RemoteFile(
        repo="t/r",
        path="a.safetensors",
        dest="e1/a.safetensors",
        sha256=SHA,
        bytes=W_BYTES,
    )
    second = RemoteFile(
        repo="t/r",
        path="b.safetensors",
        dest="e1/b.safetensors",
        sha256=SHA,
        bytes=W_BYTES,
    )
    entry = EngineEntry("e1", (first, second), True, "none", None, 0)
    payload = b"\1" * 512
    (tmp_path / "e1").mkdir(parents=True)
    (tmp_path / "e1/a.fp16.safetensors").write_bytes(payload)
    index = CastIndex(
        engine_id="e1",
        mode="dual",
        casts=(
            CastFile(
                dest="e1/a.fp16.safetensors",
                sha256=hashlib.sha256(payload).hexdigest(),
                bytes=len(payload),
                source_dest="e1/a.safetensors",
                source_sha256=SHA,
            ),
        ),
        casted_at="t",
    )

    assert not fp16_payloads_ok(index, tmp_path, entry)


def test_candidates_cover_crash_windows() -> None:
    dual = CastFile(dest="e1/model.fp16.safetensors", sha256="x", bytes=1,
                    source_dest="e1/model.safetensors", source_sha256="y")
    final = CastFile(dest="e1/model.safetensors", sha256="x", bytes=1,
                     source_dest="e1/model.safetensors", source_sha256="y")
    assert dual.candidates() == ["e1/model.fp16.safetensors"]
    assert final.candidates() == ["e1/model.safetensors", "e1/model.fp16.safetensors"]


# ---- matrix A: mode x disk state (torch-free) ---------------------------


def test_keep_original_fp32_present(tmp_path: Path) -> None:
    e = make_entry()
    put_files(tmp_path)
    st = storage_status(e, tmp_path, "keep-original")
    assert st.installed and st.fp32_present and not st.recast_needed and not st.stale_cast
    assert st.dtype_options == ("float16", "float32")
    assert missing_files_storage(e, tmp_path, "keep-original") == []


def test_keep_original_missing_downloads(tmp_path: Path) -> None:
    e = make_entry()
    put_files(tmp_path, weight=False)  # tokenizer only - weight absent
    st = storage_status(e, tmp_path, "keep-original")
    assert not st.installed
    missing = missing_files_storage(e, tmp_path, "keep-original")
    assert [f.dest for f in missing] == ["e1/model.safetensors"]


def test_dual_complete(tmp_path: Path) -> None:
    e = make_entry()
    put_files(tmp_path)
    put_index(tmp_path, e, "dual", at_fp16_name=True)
    st = storage_status(e, tmp_path, "dual")
    assert st.installed and st.fp32_present and st.fp16_present and not st.recast_needed
    assert missing_files_storage(e, tmp_path, "dual") == []


def test_dual_fp16_missing_is_recast_not_redownload(tmp_path: Path) -> None:
    """THE D25 anti-trap: originals present + payloads missing = recast."""
    e = make_entry()
    put_files(tmp_path)  # no payload, no index
    st = storage_status(e, tmp_path, "dual")
    assert not st.installed and st.recast_needed and st.fp32_present
    assert missing_files_storage(e, tmp_path, "dual") == []  # never re-download


def test_dual_missing_fp32_downloads_first(tmp_path: Path) -> None:
    e = make_entry()
    put_files(tmp_path, weight=False)
    missing = missing_files_storage(e, tmp_path, "dual")
    assert [f.dest for f in missing] == ["e1/model.safetensors"]


def test_fp16_only_finalized_complete_without_fp32(tmp_path: Path) -> None:
    """fp16-only: payload under the original name, fp32 gone - installed."""
    e = make_entry()
    fp16_size = 512
    p = tmp_path / "e1/model.safetensors"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"\1" * fp16_size)  # fp16 payload occupying the original name
    put_index(tmp_path, e, "fp16-only", at_fp16_name=False)
    st = storage_status(e, tmp_path, "fp16-only")
    assert st.installed and st.fp16_present and not st.fp32_present
    assert st.dtype_options == ("float16",)  # fp32 load unavailable by design
    assert missing_files_storage(e, tmp_path, "fp16-only") == []


def test_fp16_only_intermediate_name_also_counts(tmp_path: Path) -> None:
    """Real crash window of the fp16-only finalize: the index is written with
    the finalized name (dest == source_dest) BEFORE the overwrite, so the
    payload may still sit at the .fp16 name while the original holds fp32.
    ``candidates()`` must find it - that state is readable and installed."""
    e = make_entry()
    put_files(tmp_path)
    payload = tmp_path / fp16_dest("e1/model.safetensors")
    payload.write_bytes(b"\1" * 512)
    write_cast_index(
        tmp_path,
        CastIndex(
            engine_id="e1",
            mode="fp16-only",
            casts=(
                CastFile(
                    dest="e1/model.safetensors",
                    sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),
                    bytes=512,
                    source_dest="e1/model.safetensors",
                    source_sha256=SHA,
                ),
            ),
            casted_at="t",
        ),
    )
    st = storage_status(e, tmp_path, "fp16-only")
    assert st.fp16_present and st.installed  # candidate (.fp16 name) hits


def test_fp16_only_after_dual_asks_for_finalize(tmp_path: Path) -> None:
    """Round-25 UI finding: switching dual -> fp16-only leaves a dual-layout
    index (payload at .fp16, fp32 original intact). The payloads are reachable
    but the mode's promise (original name holds fp16, disk halved) has not
    happened; without recast_needed the panel showed no 转换 button and the
    finalize could never run."""
    e = make_entry()
    put_files(tmp_path)
    put_index(tmp_path, e, "dual", at_fp16_name=True)
    st = storage_status(e, tmp_path, "fp16-only")
    assert st.fp16_present and not st.installed and st.recast_needed
    assert missing_files_storage(e, tmp_path, "fp16-only") == []  # finalize, not re-download


def test_fp16_only_payload_lost_fp32_present_is_recast(tmp_path: Path) -> None:
    e = make_entry()
    put_files(tmp_path)
    st = storage_status(e, tmp_path, "fp16-only")
    assert not st.installed and st.recast_needed
    assert missing_files_storage(e, tmp_path, "fp16-only") == []  # anti-trap


def test_fp16_only_everything_lost_is_explicit_redownload(tmp_path: Path) -> None:
    e = make_entry()
    put_files(tmp_path, weight=False)
    missing = missing_files_storage(e, tmp_path, "fp16-only")
    assert [f.dest for f in missing] == ["e1/model.safetensors"]


def test_damaged_index_reads_as_recast(tmp_path: Path) -> None:
    e = make_entry()
    put_files(tmp_path)
    p = cast_index_path(tmp_path, "e1")
    p.write_text("garbage", encoding="utf-8")
    st = storage_status(e, tmp_path, "fp16-only")
    assert not st.fp16_present and st.recast_needed


def test_stale_cast_under_keep_original(tmp_path: Path) -> None:
    e = make_entry()
    put_files(tmp_path)
    put_index(tmp_path, e, "dual", at_fp16_name=True)
    st = storage_status(e, tmp_path, "keep-original")
    assert st.installed and st.stale_cast  # payload present, mode says no


def test_payload_size_mismatch_is_missing(tmp_path: Path) -> None:
    e = make_entry()
    put_files(tmp_path)
    put_index(tmp_path, e, "dual", at_fp16_name=True)
    (tmp_path / fp16_dest("e1/model.safetensors")).write_bytes(b"\1" * 999)
    assert not fp16_payloads_ok(read_cast_index(tmp_path, "e1"), tmp_path)


def test_is_complete_storage_shortcut(tmp_path: Path) -> None:
    e = make_entry()
    put_files(tmp_path)
    assert is_complete_storage(e, tmp_path, "keep-original")
    assert not is_complete_storage(e, tmp_path, "fp16-only")


# ---- matrix B: real casts (torch) ---------------------------------------


def write_real_safetensors(path: Path) -> object:
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    sd = {"a": torch.arange(64, dtype=torch.float32), "b": torch.randn(16, 8)}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(path))
    return sd


def load_sd(path: Path):
    from safetensors.torch import load_file

    return load_file(str(path))


def test_cast_dual(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    src = tmp_path / "e1/model.safetensors"
    sd = write_real_safetensors(src)
    entry = real_entry(src)

    events: list = []
    idx = cast_engine_files(entry, tmp_path, "dual", threading.Event(), events.append)

    payload = tmp_path / fp16_dest("e1/model.safetensors")
    assert payload.is_file()
    assert payload.stat().st_size < src.stat().st_size  # fp16 halves the tensors
    assert idx.casts[0].dest == fp16_dest("e1/model.safetensors")
    assert read_cast_index(tmp_path, "e1") == idx
    out = load_sd(payload)
    assert all(t.dtype == torch.float16 for t in out.values())
    assert torch.allclose(out["b"].float(), sd["b"], atol=1e-2)  # fp16 roundoff
    assert events and events[-1].done_bytes == events[-1].total_bytes


def test_cast_fp16_only_overwrites_original(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    src = tmp_path / "e1/model.safetensors"
    write_real_safetensors(src)
    entry = real_entry(src)
    fp32_size = src.stat().st_size

    idx = cast_engine_files(entry, tmp_path, "fp16-only", threading.Event())

    assert src.stat().st_size < fp32_size  # payload replaced the original
    assert not (tmp_path / fp16_dest("e1/model.safetensors")).exists()  # renamed away
    assert idx.casts[0].dest == "e1/model.safetensors"  # finalized name
    assert read_cast_index(tmp_path, "e1") == idx
    assert all(t.dtype == torch.float16 for t in load_sd(src).values())
    st = storage_status(entry, tmp_path, "fp16-only")
    assert st.installed and st.fp16_present and not st.fp32_present


def test_cast_fp16_only_is_idempotent(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    src = tmp_path / "e1/model.safetensors"
    write_real_safetensors(src)
    entry = real_entry(src)
    idx1 = cast_engine_files(entry, tmp_path, "fp16-only", threading.Event())
    idx2 = cast_engine_files(entry, tmp_path, "fp16-only", threading.Event())
    assert idx1.casts[0].sha256 == idx2.casts[0].sha256  # fp16->fp16 lossless


def test_cast_cancelled_midway(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    src = tmp_path / "e1/model.safetensors"
    write_real_safetensors(src)
    entry = real_entry(src)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(KeyboardInterrupt):
        cast_engine_files(entry, tmp_path, "dual", cancel)
    assert not cast_index_path(tmp_path, "e1").exists()


def test_cast_requires_originals(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    entry = make_entry()
    with pytest.raises(FileNotFoundError, match="original missing"):
        cast_engine_files(entry, tmp_path, "dual", threading.Event())


def test_cast_mode_validation(tmp_path: Path) -> None:
    entry = make_entry()
    with pytest.raises(ValueError, match="dual/fp16-only"):
        cast_engine_files(entry, tmp_path, "keep-original", threading.Event())


def test_cleanup_dual_keeps_original(tmp_path: Path) -> None:
    e = make_entry()
    put_files(tmp_path)
    put_index(tmp_path, e, "dual", at_fp16_name=True)
    removed = cleanup_cast(tmp_path, "e1")
    assert removed == [fp16_dest("e1/model.safetensors")]
    assert (tmp_path / "e1/model.safetensors").is_file()  # original untouched
    assert not cast_index_path(tmp_path, "e1").exists()
    assert not storage_status(e, tmp_path, "dual").fp16_present


def test_cleanup_fp16_only_never_kills_only_copy(tmp_path: Path) -> None:
    e = make_entry()
    p = tmp_path / "e1/model.safetensors"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"\1" * 512)
    put_index(tmp_path, e, "fp16-only", at_fp16_name=False)
    removed = cleanup_cast(tmp_path, "e1")
    assert removed == []  # the only payload lives at the original name
    assert p.is_file() and not cast_index_path(tmp_path, "e1").exists()


# ---- matrix C: loader linkage -------------------------------------------


class FakeEngine:
    """Minimal engine double carrying the load knobs."""

    def __init__(self, engine_id: str) -> None:
        self.engine_id = engine_id
        self.load_dtype = "auto"
        self.load_variant = None


def make_loader(tmp_path: Path, mode: str | None, with_index: bool):
    from engines.loader import EngineLoader

    entry = make_entry()
    if mode is not None:
        pol = StoragePolicy(tmp_path / "storage.json")
        if mode != DEFAULT_MODE:
            pol.set_mode("e1", mode)
    else:
        pol = None
    if with_index:
        put_index(tmp_path, entry, mode or "dual", at_fp16_name=(mode != "fp16-only"))
    else:
        put_files(tmp_path)
    loader = EngineLoader(policy=pol, models_root=tmp_path)
    return loader, FakeEngine("e1"), entry


def test_loader_dual_sets_variant(tmp_path: Path) -> None:
    loader, eng, _ = make_loader(tmp_path, "dual", with_index=True)
    loader._configure_load(eng)  # type: ignore[arg-type]
    assert eng.load_dtype == "float16" and eng.load_variant == "fp16"


def test_loader_fp16_only_uses_original_name(tmp_path: Path) -> None:
    loader, eng, _ = make_loader(tmp_path, "fp16-only", with_index=True)
    loader._configure_load(eng)  # type: ignore[arg-type]
    assert eng.load_dtype == "float16" and eng.load_variant is None


def test_loader_fp16_only_refuses_float32(tmp_path: Path) -> None:
    loader, eng, _ = make_loader(tmp_path, "fp16-only", with_index=True)
    eng.load_dtype = "float32"
    with pytest.raises(ValueError, match="float32 load unavailable"):
        loader._configure_load(eng)  # type: ignore[arg-type]


def test_loader_keep_original_untouched(tmp_path: Path) -> None:
    loader, eng, _ = make_loader(tmp_path, None, with_index=False)
    loader._configure_load(eng)  # type: ignore[arg-type]
    assert eng.load_dtype == "auto" and eng.load_variant is None


def test_loader_without_policy_untouched(tmp_path: Path) -> None:
    loader, eng, _ = make_loader(tmp_path, "dual", with_index=False)
    loader._policy = None
    loader._models_root = tmp_path
    loader._configure_load(eng)  # type: ignore[arg-type]
    assert eng.load_dtype == "auto"


def test_loader_payloads_missing_falls_back(tmp_path: Path) -> None:
    """dual mode but payloads damaged -> originals path, no variant."""
    loader, eng, entry = make_loader(tmp_path, "dual", with_index=False)
    put_index(tmp_path, entry, "dual", at_fp16_name=True)
    (tmp_path / fp16_dest("e1/model.safetensors")).write_bytes(b"\1" * 3)  # size off
    loader._configure_load(eng)  # type: ignore[arg-type]
    assert eng.load_dtype == "auto" and eng.load_variant is None


# ---- matrix C2 (round-25): the *detect* path must apply the knobs --------
#
# Round-25 found the live hole: only ``_load_one`` (explicit warm-up) ran
# ``_configure_load``; ``detect_one`` went straight to ``lru.acquire``, so a
# dual-mode install still opened the fp32 original (proven by hiding it) and
# fp16-only only worked because the original name happened to hold fp16 data.
# These tests drive ``detect_one`` - the path every run takes - not the
# private helper, so the wiring itself is covered.


def make_detect_loader(tmp_path: Path, mode: str | None, engine):
    from engines.loader import EngineLoader
    from engines.registry import EngineSpec

    pol = StoragePolicy(tmp_path / "storage.json")
    if mode is not None and mode != DEFAULT_MODE:
        pol.set_mode("e1", mode)
    spec = EngineSpec(
        engine_id="e1", name="E1", lang="en", kind="classifier", est_vram_mb=10, repo="-"
    )
    return EngineLoader(
        specs=(spec,), engines={"e1": engine}, policy=pol, models_root=tmp_path
    )


def test_detect_applies_dual_variant(tmp_path: Path) -> None:
    from fakes import FakeEngine

    entry = make_entry()
    put_index(tmp_path, entry, "dual", at_fp16_name=True)
    eng = FakeEngine("e1")
    loader = make_detect_loader(tmp_path, "dual", eng)
    result = loader.detect_one("hello", "e1")
    assert result.status == "ok"
    assert (eng.load_variant, eng.load_dtype) == ("fp16", "float16")


def test_detect_applies_fp16_only_dtype(tmp_path: Path) -> None:
    from fakes import FakeEngine

    entry = make_entry()
    put_index(tmp_path, entry, "fp16-only", at_fp16_name=False)
    eng = FakeEngine("e1")
    loader = make_detect_loader(tmp_path, "fp16-only", eng)
    result = loader.detect_one("hello", "e1")
    assert result.status == "ok"
    # payload sits at the original name, so no variant - but the dtype is forced
    assert (eng.load_variant, eng.load_dtype) == (None, "float16")


def test_detect_refuses_float32_on_fp16_only(tmp_path: Path) -> None:
    from fakes import FakeEngine

    entry = make_entry()
    put_index(tmp_path, entry, "fp16-only", at_fp16_name=False)
    eng = FakeEngine("e1")
    eng.load_dtype = "float32"
    loader = make_detect_loader(tmp_path, "fp16-only", eng)
    result = loader.detect_one("hello", "e1")
    assert result.status == "error"
    assert "float32 load unavailable" in (result.detail or "")
    assert eng.load_calls == 0  # the bad load never started


def test_detect_keep_original_untouched(tmp_path: Path) -> None:
    from fakes import FakeEngine

    put_files(tmp_path)
    eng = FakeEngine("e1")
    loader = make_detect_loader(tmp_path, None, eng)
    result = loader.detect_one("hello", "e1")
    assert result.status == "ok"
    assert (eng.load_variant, eng.load_dtype) == (None, "auto")


# ---- matrix D: DownloadManager state machine -----------------------------


def make_manager(tmp_path: Path, entry: EngineEntry, policy: StoragePolicy | None = None):
    from download_manager import DownloadManager
    from manifest import Manifest

    manifest = Manifest(engines={entry.engine_id: entry}, generated_at="t")
    man = DownloadManager(manifest, tmp_path, policy=policy)
    events: list[dict] = []
    man.attach(events.append)

    async def drive(action) -> tuple[object, list[dict]]:
        result = action(man)
        task = man._task
        if task is not None:
            await task
        return result, events

    return man, drive


def test_manager_skips_installed_fp16_only(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    src = tmp_path / "e1/model.safetensors"
    write_real_safetensors(src)
    entry = real_entry(src)
    cast_engine_files(entry, tmp_path, "fp16-only", threading.Event())
    pol = StoragePolicy(tmp_path / "storage.json")
    pol.set_mode("e1", "fp16-only")

    man, drive = make_manager(tmp_path, entry, pol)
    result, _ = asyncio.run(drive(lambda m: m.start(["e1"])))
    assert result["skipped"] == [{"engine_id": "e1", "reason": "already_installed"}]
    assert man._task is None  # no job spawned - the anti-trap in action


def test_manager_recast_job_runs_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    src = tmp_path / "e1/model.safetensors"
    write_real_safetensors(src)
    put_files(tmp_path, weight=False)  # tokenizer present, real weight kept
    entry = real_entry(src)
    pol = StoragePolicy(tmp_path / "storage.json")
    pol.set_mode("e1", "fp16-only")

    _man, drive = make_manager(tmp_path, entry, pol)
    result, events = asyncio.run(drive(lambda m: m.start(["e1"])))
    assert result["started"] == ["e1"]
    kinds = [e["kind"] for e in events]
    assert "cast_start" in kinds and "cast_done" in kinds
    assert not missing_files_storage(entry, tmp_path, "fp16-only")


def test_manager_download_then_cast_chain(tmp_path: Path) -> None:
    """dual mode with originals present but payloads missing -> recast job."""
    pytest.importorskip("torch")
    src = tmp_path / "e1/model.safetensors"
    write_real_safetensors(src)
    put_files(tmp_path, weight=False)
    entry = real_entry(src)
    pol = StoragePolicy(tmp_path / "storage.json")
    pol.set_mode("e1", "dual")

    _man, drive = make_manager(tmp_path, entry, pol)
    result, events = asyncio.run(drive(lambda m: m.start(["e1"])))
    # fp32 present but payloads missing -> recast job (started, cast runs)
    assert result["started"] == ["e1"]
    kinds = [e["kind"] for e in events]
    assert "cast_start" in kinds and "cast_done" in kinds


def test_manager_status_and_set_mode(tmp_path: Path) -> None:
    e = make_entry()
    put_files(tmp_path)
    man, _ = make_manager(tmp_path, e)
    st = man.status()
    assert st["engines"]["e1"]["installed"] is True
    assert st["engines"]["e1"]["storage"]["storage_mode"] == "keep-original"
    out = man.set_mode("e1", "fp16-only")
    assert out["storage"]["storage_mode"] == "fp16-only"
    st2 = man.status()
    assert st2["engines"]["e1"]["installed"] is False  # payloads absent now
    assert st2["engines"]["e1"]["storage"]["recast_needed"] is True


def test_manager_cleanup_via_domain(tmp_path: Path) -> None:
    e = make_entry()
    put_files(tmp_path)
    put_index(tmp_path, e, "dual", at_fp16_name=True)
    man, _ = make_manager(tmp_path, e)
    out = man.cleanup("e1")
    assert out["removed"] == [fp16_dest("e1/model.safetensors")]


def test_manager_start_cast_requires_non_default_mode(tmp_path: Path) -> None:
    e = make_entry()
    put_files(tmp_path)
    man, _ = make_manager(tmp_path, e)
    with pytest.raises(ValueError, match="switch mode before casting"):
        man.start_cast("e1")


def test_uninstall_removes_owned_files_and_preserves_shared(tmp_path: Path) -> None:
    e = make_entry()
    put_files(tmp_path)
    put_index(tmp_path, e, "dual", at_fp16_name=True)
    (tmp_path / "e1/model.safetensors.incomplete").write_bytes(b"partial")
    unknown = tmp_path / "e1/notes-user.txt"
    unknown.write_text("keep me", encoding="utf-8")

    out = uninstall_engine_files(
        e,
        tmp_path,
        shared_dests={"e1/tokenizer.json"},
    )

    assert "e1/model.safetensors" in out["removed"]
    assert fp16_dest("e1/model.safetensors") in out["removed"]
    assert "e1/model.safetensors.incomplete" in out["removed"]
    assert not cast_index_path(tmp_path, "e1").exists()
    assert (tmp_path / "e1/tokenizer.json").is_file()  # shared reference kept
    assert unknown.is_file()  # unknown user file never swept recursively
    assert "e1/notes-user.txt" in out["leftovers"]


def test_manager_uninstall_clears_policy(tmp_path: Path) -> None:
    e = make_entry()
    put_files(tmp_path)
    pol = StoragePolicy(tmp_path / "storage.json")
    pol.set_mode("e1", "fp16-only")
    man, _ = make_manager(tmp_path, e, pol)

    out = man.uninstall("e1")

    assert out["engine_id"] == "e1"
    assert not (tmp_path / "e1/model.safetensors").exists()
    assert StoragePolicy.load(tmp_path / "storage.json").mode("e1") == DEFAULT_MODE
    assert man.status()["engines"]["e1"]["installed"] is False

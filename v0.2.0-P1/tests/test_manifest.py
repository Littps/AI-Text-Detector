"""tests for models/manifest.json - manifest <-> registry <-> disk.

The manifest is the downloader's truth source; these gates keep it from
drifting away from the engine registry or the actual disk layout.
"""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))

import pytest
from engines.registry import ENGINE_SPECS
from manifest import Manifest, _safe_dest

MANIFEST_PATH = ROOT / "models" / "manifest.json"

# D19/D20: seven engine defaults; round-7 adds the v3 trio (zhv3 /
# zhv3short / env3) to the default download set; the two heavy method
# engines stay optional.
DEFAULT_TEN = {
    "modernbert-raid-mage",
    "desklib-en",
    "simpleai-zh",
    "xiaoyu-zh",
    "vanguard-en",
    "r1-lang-router",
    "gltr",
    "aigc-zhv3",
    "aigc-zhv3short",
    "aigc-env3",
}


@pytest.fixture(scope="module")
def manifest() -> Manifest:
    return Manifest.load(MANIFEST_PATH)


class TestConsistency:
    def test_engine_ids_match_registry(self, manifest: Manifest) -> None:
        assert set(manifest.ids()) == {s.engine_id for s in ENGINE_SPECS}

    def test_default_set_is_the_verified_ten(self, manifest: Manifest) -> None:
        assert set(manifest.default_ids()) == DEFAULT_TEN

    def test_risk_fields(self, manifest: Manifest) -> None:
        fdg = manifest.get("fast-detectgpt")
        assert fdg.default_selected is False
        assert fdg.risk_level == "medium" and fdg.risk_text_key == "risk.fdg.exclusive"
        assert fdg.min_vram_mb == 8000
        bino = manifest.get("binoculars")
        assert bino.default_selected is False
        assert bino.risk_level == "high" and bino.risk_text_key == "risk.binoculars.8g"
        # round-7: fp16 pair is ~28GB - the old 12GB figure assumed an
        # unimplemented 4-bit path.
        assert bino.min_vram_mb == 28000

    def test_default_engines_carry_no_risk(self, manifest: Manifest) -> None:
        for eid in manifest.default_ids():
            e = manifest.get(eid)
            assert e.risk_level == "none" and e.risk_text_key is None, eid

    def test_gltr_maps_to_shared_gpt2_dir(self, manifest: Manifest) -> None:
        e = manifest.get("gltr")
        assert e.files and all(f.dest.startswith("gpt2/") for f in e.files)
        assert any(f.path == "model.safetensors" for f in e.files)

    def test_binoculars_nests_two_repos(self, manifest: Manifest) -> None:
        e = manifest.get("binoculars")
        repos = {f.repo for f in e.files}
        assert repos == {"tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct"}
        # dest layout: binoculars/<repo-name>/<file>
        assert {f.dest.split("/")[1] for f in e.files} == {"falcon-7b", "falcon-7b-instruct"}

    def test_every_file_builds_a_sane_url(self, manifest: Manifest) -> None:
        for eid in manifest.ids():
            for f in manifest.get(eid).files:
                url = f.url("huggingface.co")
                assert url.startswith(f"https://huggingface.co/{f.repo}/resolve/main/{f.path}"), url

    def test_landed_files_match_manifest_sizes(self, manifest: Manifest) -> None:
        """Fast integrity gate: any file already on disk must match the
        manifest byte size (sha was pinned against the HF LFS oid at
        generation time - gen_manifest aborts on mismatch)."""
        models_root = ROOT / "models"
        checked = 0
        for eid in manifest.ids():
            for f in manifest.get(eid).files:
                p = models_root / f.dest
                if p.is_file():
                    assert p.stat().st_size == f.bytes, f"{eid}/{f.dest}"
                    checked += 1
        assert checked >= 30  # the seven landed engines carry ~41 files


class TestLoaderParity:
    def test_landed_engines_load_real(self, manifest: Manifest) -> None:
        """Every manifest engine whose files are complete must route to a
        real engine in the loader (config.json + weights on disk)."""
        from engines.loader import default_engine
        from engines.registry import spec as reg_spec

        models_root = ROOT / "models"
        for eid in manifest.ids():
            entry = manifest.get(eid)
            landed = all((models_root / f.dest).is_file() for f in entry.files)
            if not landed:
                continue
            eng = default_engine(reg_spec(eid))
            assert type(eng).__name__ != "StubEngine", f"{eid} landed but routes to stub"


class TestSafeDest:
    """L2 fix: dest must never escape the models/ root."""

    def test_normal_dests_pass(self) -> None:
        assert _safe_dest("e1", "e1/model.safetensors") == "e1/model.safetensors"
        assert _safe_dest("gltr", "gpt2/config.json") == "gpt2/config.json"
        assert _safe_dest("b", "b/falcon-7b/config.json") == "b/falcon-7b/config.json"

    @pytest.mark.parametrize(
        "dest", ["/abs/path.bin", "C:/evil/x.bin", "../escape.bin", "e1/../../up.bin", "a/../b", ""]
    )
    def test_unsafe_dests_rejected(self, dest: str) -> None:
        with pytest.raises(ValueError, match="unsafe dest"):
            _safe_dest("e1", dest)

    @pytest.mark.parametrize(
        "dest",
        [
            r"..\escape.bin",
            r"C:foo",
            r"C:..\evil",
            r"\\server\share\x",
            "e1/model.safetensors:stream",
            "e1/NUL.txt",
            "e1/trailing.",
            "e1/trailing ",
            "e1//double",
            "e1/\x01control",
        ],
    )
    def test_windows_escape_spellings_rejected(self, dest: str) -> None:
        with pytest.raises(ValueError, match="unsafe dest"):
            _safe_dest("e1", dest)

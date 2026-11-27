"""tests for backend/hardware.py - pure functions, no CUDA required."""
from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

import pytest
from hardware import (
    GpuInfo,
    clean_cpu_name,
    collect,
    parse_nvidia_smi_output,
    probe_torch,
    tier_for,
    wheel_for,
)


class TestCpuIdentity:
    """The device panel shows a real CPU model, not "AMD64"."""

    def test_registry_padding_is_collapsed(self) -> None:
        # Windows pads ProcessorNameString to a fixed width and often doubles
        # inner spaces; a plain strip() would leave both artefacts visible.
        raw = "  AMD Ryzen 7 5800H  with Radeon Graphics        "
        assert clean_cpu_name(raw) == "AMD Ryzen 7 5800H with Radeon Graphics"

    def test_normal_name_is_untouched(self) -> None:
        assert clean_cpu_name("Intel(R) Core(TM) i5-13490F") == "Intel(R) Core(TM) i5-13490F"

    def test_empty_falls_back(self) -> None:
        assert clean_cpu_name("   ") == ""


class TestTierMatrix:
    """PLAN P2 tier matrix, boundary-exact."""

    @pytest.mark.parametrize(
        ("has_cuda", "cap", "vram", "want"),
        [
            # no cuda -> cpu regardless of everything else
            (False, None, 0, "cpu"),
            (False, (12, 0), 24000, "cpu"),
            # capability unknown -> conservative cpu (D20: wrong guess costs
            # one warning, never a feature)
            (True, None, 8151, "cpu"),
            # pre-Turing cannot run cu129
            (True, (6, 1), 11000, "cpu"),
            (True, (7, 4), 11000, "cpu"),
            # Turing floor AND vram >= 6GB (PLAN: standard needs 6GB)
            (True, (7, 5), 6144, "standard"),
            # vram below the 6GB floor falls to cpu even with modern cap
            # (M1 fix: 4GB cards must not be defaulted into OOM)
            (True, (7, 5), 6000, "cpu"),
            (True, (8, 6), 4096, "cpu"),
            (True, (8, 6), 6143, "cpu"),
            # RTX 5060: sm_120, 8151MB -> standard
            (True, (12, 0), 8151, "standard"),
            # plus boundary: >= 12288
            (True, (8, 9), 12288, "plus"),
            (True, (8, 9), 12287, "standard"),
            (True, (9, 0), 24000, "plus"),
        ],
    )
    def test_matrix(self, has_cuda: bool, cap, vram: int, want: str) -> None:
        assert tier_for(has_cuda, cap, vram) == want

    def test_wheel(self) -> None:
        assert wheel_for("cpu") == "cpu"
        assert wheel_for("standard") == "cu129"
        assert wheel_for("plus") == "cu129"


class TestParseNvidiaSmi:
    RTX5060 = "NVIDIA GeForce RTX 5060, 8151, 12.0\n"

    def test_real_shape(self) -> None:
        gpu = parse_nvidia_smi_output(self.RTX5060)
        assert gpu is not None
        assert gpu.name == "NVIDIA GeForce RTX 5060"
        assert gpu.vram_total_mb == 8151
        assert gpu.capability == (12, 0)

    def test_driver_version_shape(self) -> None:
        gpu = parse_nvidia_smi_output(
            "NVIDIA GeForce RTX 5060, 8151, 12.0, 590.72\n"
        )
        assert gpu is not None
        assert gpu.driver_version == "590.72"

    def test_multi_gpu_takes_first(self) -> None:
        out = "GPU A, 8151, 12.0\nGPU B, 24000, 8.9\n"
        gpu = parse_nvidia_smi_output(out)
        assert gpu is not None
        assert gpu.name == "GPU A"

    def test_gpu_name_with_comma(self) -> None:
        # OEM names may contain a comma: maxsplit=2 keeps name intact (L3 fix)
        out = "NVIDIA GeForce RTX 3060, Ti, 8192, 8.6\n"
        gpu = parse_nvidia_smi_output(out)
        assert gpu is not None
        assert gpu.name == "NVIDIA GeForce RTX 3060, Ti"
        assert gpu.vram_total_mb == 8192
        assert gpu.capability == (8, 6)

    def test_old_driver_no_capability(self) -> None:
        gpu = parse_nvidia_smi_output("GTX 1070, 8192, [N/A]\n")
        assert gpu is not None
        assert gpu.capability is None
        assert gpu.vram_total_mb == 8192

    def test_garbage(self) -> None:
        assert parse_nvidia_smi_output("") is None
        assert parse_nvidia_smi_output("only one field\n") is None
        assert parse_nvidia_smi_output("a, not_a_number, 12.0\n") is None


class TestCollect:
    """Two-source cross-check: torch wins on conflict, smi fills the gap."""

    def _hw(self, smi, torch_gpu, **kw) -> dict:
        defaults = {
            "ram_mb": lambda: 32000,
            "disk_free": lambda p: 100000,
        }
        defaults.update(kw)
        return collect(
            Path("."),
            gpu_smi=lambda: smi,
            gpu_torch=lambda: torch_gpu,
            ram_mb=defaults["ram_mb"],
            disk_free=defaults["disk_free"],
        )

    def test_both_agree(self) -> None:
        gpu = GpuInfo("RTX 5060", 8151, (12, 0))
        hw = self._hw(gpu, gpu)
        assert hw["source"] == "torch"
        assert hw["tier"] == "standard"
        assert hw["wheel"] == "cu129"
        assert hw["cuda"] is True
        assert hw["capability"] == [12, 0]
        assert hw["ram_total_mb"] == 32000
        assert hw["disk_free_mb"] == 100000
        assert "note" not in hw

    def test_conflict_torch_wins_with_note(self) -> None:
        smi = GpuInfo("Mystery Card", 8151, (7, 5))
        torch_gpu = GpuInfo("RTX 5060", 8151, (12, 0))
        hw = self._hw(smi, torch_gpu)
        assert hw["tier"] == "standard"
        assert hw["capability"] == [12, 0]  # torch's, not smi's
        assert "disagree" in hw["note"]

    def test_first_boot_torch_absent(self) -> None:
        smi = GpuInfo("RTX 5060", 8151, (12, 0), "590.72")
        hw = self._hw(smi, None)
        assert hw["source"] == "nvidia-smi"
        assert hw["tier"] == "standard"
        assert hw["driver_version"] == "590.72"

    def test_nothing_found_cpu_tier(self) -> None:
        hw = self._hw(None, None)
        assert hw["source"] == "none"
        assert hw["cuda"] is False
        assert hw["tier"] == "cpu"
        assert hw["wheel"] == "cpu"
        assert hw["vram_total_mb"] == 0

    def test_smi_without_capability_is_cpu(self) -> None:
        # pre-515 driver: vram known but capability missing -> conservative
        smi = GpuInfo("GTX 1070", 8192, None)
        hw = self._hw(smi, None)
        assert hw["cuda"] is False
        assert hw["tier"] == "cpu"

    def test_cpu_identity_lands_in_the_hw_block(self, tmp_path: Path) -> None:
        hw = collect(
            tmp_path,
            gpu_smi=lambda: None,
            gpu_torch=lambda: None,
            ram_mb=lambda: 32000,
            disk_free=lambda p: 1,
            cpu_name_fn=lambda: "Ryzen Test 7500F",
        )
        assert hw["cpu_name"] == "Ryzen Test 7500F"
        assert hw["cpu_threads"] >= 1
        assert hw["ram_total_mb"] == 32000


def test_torch_native_import_error_does_not_crash_boot(monkeypatch) -> None:
    import builtins

    original = builtins.__import__

    def fail_torch(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise OSError("simulated missing msvcp140.dll")
        return original(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_torch)
    assert probe_torch() is None


# ---- P3-a Step4: SIDE_FORCE_TIER debug injection --------------------------


def test_force_tier_cpu(tmp_path, monkeypatch):
    import hardware

    monkeypatch.setenv("SIDE_FORCE_TIER", "cpu")
    hw = hardware.collect(tmp_path, gpu_smi=lambda: hardware.GpuInfo("RTX 5060", (12, 0), 8192), gpu_torch=lambda: None)
    assert hw["tier"] == "cpu" and hw["cuda"] is False and hw["source"] == "forced"
    assert "SIDE_FORCE_TIER" in hw["note"]


def test_force_tier_standard_and_plus(tmp_path, monkeypatch):
    import hardware

    monkeypatch.setenv("SIDE_FORCE_TIER", "standard")
    hw = hardware.collect(tmp_path, gpu_smi=lambda: None, gpu_torch=lambda: None)
    assert hw["tier"] == "standard" and hw["vram_total_mb"] == 8192 and hw["cuda"] is True

    monkeypatch.setenv("SIDE_FORCE_TIER", "plus")
    hw = hardware.collect(tmp_path, gpu_smi=lambda: None, gpu_torch=lambda: None)
    assert hw["tier"] == "plus" and hw["vram_total_mb"] == 16384


def test_force_tier_invalid_ignored(tmp_path, monkeypatch):
    import hardware

    monkeypatch.setenv("SIDE_FORCE_TIER", "bogus")
    hw = hardware.collect(tmp_path, gpu_smi=lambda: None, gpu_torch=lambda: None)
    assert hw["source"] != "forced"  # real probe path taken (cpu here)

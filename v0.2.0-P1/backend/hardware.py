"""Hardware probe (D22, PLAN P2 sub-design): two-source GPU detection.

Primary source is ``nvidia-smi`` (works before torch is installed, and
follows the driver so new cards report their real capability). torch is
the verification source and wins on conflict - it is the actual
executor. ``tier_for`` is a pure, torch-free function so the tier matrix
is unit-testable without CUDA.

D20 principle: the tier NEVER locks options. It only selects defaults
(wheel, default-checked engines) and warning copy. A mis-detected new
card costs the user one extra warning, never a missing feature.
"""
from __future__ import annotations

import contextlib
import ctypes
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# capability >= (7, 5) = Turing or newer = the cu129 wheel floor.
MIN_CAPABILITY: tuple[int, int] = (7, 5)
# PLAN P2 tier matrix: standard tier requires vram >= 6GB (the 7-engine
# default set peaks at 3110MB and needs headroom); 4GB cards fall to the
# CPU tier (lightweight defaults) instead of being pushed into OOM.
STANDARD_MIN_VRAM_MB = 6144
PLUS_VRAM_MB = 12288  # >= 12GB -> "plus" tier (fdg/binoculars unlock)

WHEEL_CUDA = "cu129"
WHEEL_CPU = "cpu"


@dataclass(frozen=True)
class GpuInfo:
    name: str
    vram_total_mb: int
    capability: tuple[int, int] | None  # None = source could not report
    driver_version: str | None = None


def parse_nvidia_smi_output(stdout: str) -> GpuInfo | None:
    """Parse ``--format=csv,noheader,nounits`` output (first GPU wins).

    Multi-GPU hosts: P2 targets laptop/desktop single-GPU; the first line
    is used and the rest ignored. An old driver (<515) omits compute_cap
    -> capability None -> the collector falls back conservatively.
    """
    lines = [ln.strip() for ln in stdout.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    parts = [p.strip() for p in lines[0].split(",")]
    if len(parts) < 3:
        return None
    # Current probes append driver_version. Keep accepting the historical
    # three-field shape so older fixtures and saved diagnostics stay valid.
    has_driver = False
    if len(parts) >= 4:
        last_dotted = parts[-1].split(".")
        has_driver = (
            len(last_dotted) == 2
            and last_dotted[0].isdigit()
            and int(last_dotted[0]) > 99
        )
    if has_driver:
        name, mem, cap, driver = ", ".join(parts[:-3]), parts[-3], parts[-2], parts[-1]
    else:
        name, mem, cap, driver = ", ".join(parts[:-2]), parts[-2], parts[-1], None
    try:
        vram = int(float(mem))
    except ValueError:
        return None
    capability: tuple[int, int] | None = None
    try:
        major, minor = cap.split(".")
        capability = (int(major), int(minor))
    except ValueError:
        pass  # e.g. "[N/A]" from pre-515 drivers - keep vram, lose capability
    if driver in ("", "[N/A]", "N/A"):
        driver = None
    return GpuInfo(
        name=name or "?",
        vram_total_mb=vram,
        capability=capability,
        driver_version=driver,
    )


def probe_nvidia_smi() -> GpuInfo | None:
    """Primary source. None when nvidia-smi is absent, times out, or errors."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return parse_nvidia_smi_output(out.stdout)


def probe_torch() -> GpuInfo | None:
    """Verification source. None when torch missing or no CUDA device."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(0)
        return GpuInfo(
            name=torch.cuda.get_device_name(0),
            vram_total_mb=round(props.total_memory / (1024 * 1024)),
            capability=(int(props.major), int(props.minor)),
        )
    except Exception:  # noqa: BLE001 - missing DLLs/ABI errors must not crash boot
        return None


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def ram_total_mb() -> int:
    """Physical RAM. Windows via GlobalMemoryStatusEx, POSIX fallback."""
    with contextlib.suppress(Exception):
        st = _MemoryStatusEx()
        st.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return int(st.ullTotalPhys // (1024 * 1024))
    try:
        import os

        sysconf = getattr(os, "sysconf", None)  # POSIX-only; absent on Windows
        if sysconf is not None:
            page = sysconf("SC_PAGE_SIZE")
            pages = sysconf("SC_PHYS_PAGES")
            return page * pages // (1024 * 1024)
    except (ValueError, OSError, TypeError):
        return 0
    return 0


def disk_free_mb(path: Path) -> int:
    try:
        return shutil.disk_usage(path).free // (1024 * 1024)
    except OSError:
        return 0


# Windows keeps the CPU model only in the registry: platform.processor()
# returns the architecture ("AMD64"), which is useless in a device panel.
_CPU_REG_PATH = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"


def clean_cpu_name(raw: str) -> str:
    """Collapse the padding Windows leaves in ProcessorNameString.

    The registry value is space-padded to a fixed width and often has doubled
    inner spaces ("AMD Ryzen 7 5800H  with Radeon Graphics        "), so a plain
    strip is not enough - split/join normalises every run of whitespace.
    """
    return " ".join(raw.split())


def cpu_name() -> str:
    """Human-readable CPU model. Never raises; degrades to '未知 CPU'."""
    if sys.platform == "win32":
        with contextlib.suppress(OSError, ImportError, ValueError):
            import winreg  # Windows-only module, imported lazily

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _CPU_REG_PATH) as key:
                name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                cleaned = clean_cpu_name(str(name))
                if cleaned:
                    return cleaned
    cpuinfo = Path("/proc/cpuinfo")
    with contextlib.suppress(OSError):
        if cpuinfo.exists():
            for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith("model name"):
                    cleaned = clean_cpu_name(line.split(":", 1)[-1])
                    if cleaned:
                        return cleaned
    return clean_cpu_name(platform.processor()) or "未知 CPU"


def cpu_threads() -> int:
    """Logical cores, used only as a secondary hint in the device panel."""
    return int(os.cpu_count() or 0)


def tier_for(has_cuda: bool, capability: tuple[int, int] | None, vram_mb: int) -> str:
    """Pure tier function (PLAN P2 matrix).

    cpu      : no CUDA, unknown capability, pre-Turing (cu129 wheel
               cannot run), or vram < 6GB -> CPU wheel, lightweight
               default set.
    standard : vram >= 6GB - the 7-engine default set (3110MB proven).
    plus     : vram >= 12GB - fdg/binoculars unlocked (still not default).
    """
    if not has_cuda or capability is None or capability < MIN_CAPABILITY:
        return "cpu"
    if vram_mb >= PLUS_VRAM_MB:
        return "plus"
    if vram_mb >= STANDARD_MIN_VRAM_MB:
        return "standard"
    return "cpu"


def wheel_for(tier: str) -> str:
    return WHEEL_CPU if tier == "cpu" else WHEEL_CUDA


def collect(
    models_dir: Path,
    *,
    gpu_smi: Callable[[], GpuInfo | None] = probe_nvidia_smi,
    gpu_torch: Callable[[], GpuInfo | None] = probe_torch,
    ram_mb: Callable[[], int] = ram_total_mb,
    disk_free: Callable[[Path], int] = disk_free_mb,
    cpu_name_fn: Callable[[], str] = cpu_name,
) -> dict:
    """Build the health ``hw`` block with two-source cross-checking.

    torch wins on conflict (it executes the inference); nvidia-smi fills
    in when torch is absent (first boot, pre-install). Both absent ->
    most conservative CPU tier. Sources that disagree are reported in
    ``note`` for diagnostics - never silently mixed.
    """
    # Diagnostics are executable on a damaged machine by definition. A
    # third-party/corrupt torch install can raise OSError while importing
    # MSVCP140 or a CUDA DLL, not only ImportError; no probe callback may
    # prevent the sidecar from reaching its stdout handshake.
    try:
        smi = gpu_smi()
    except Exception:  # noqa: BLE001 - probe must never crash boot
        smi = None
    try:
        torch_gpu = gpu_torch()
    except Exception:  # noqa: BLE001 - probe must never crash boot
        torch_gpu = None

    # P3-a Step4 acceptance hook: force a tier without needing real
    # low-end hardware. Accepted values: cpu | standard | plus. Anything
    # else is ignored (silently - this is a dev/acceptance switch only,
    # never documented for end users).
    forced = os.environ.get("SIDE_FORCE_TIER", "").strip().lower()
    if forced in ("cpu", "standard", "plus"):
        tier = forced
        forced_note = f"tier forced to {tier} via SIDE_FORCE_TIER (acceptance mode)"
        hw: dict = {
            "cuda": tier != "cpu",
            "capability": [7, 5] if tier != "cpu" else None,
            "gpu_name": f"forced:{tier}",
            "vram_total_mb": {"cpu": 0, "standard": 8192, "plus": 16384}[tier],
            "ram_total_mb": ram_mb(),
            "cpu_name": cpu_name_fn(),
            "cpu_threads": cpu_threads(),
            "disk_free_mb": disk_free(models_dir),
            "tier": tier,
            "wheel": wheel_for(tier),
            "source": "forced",
            "note": forced_note,
            "driver_version": None,
        }
        return hw

    source = "none"
    gpu: GpuInfo | None = None
    note: str | None = None
    if torch_gpu is not None and smi is not None:
        source = "torch"
        gpu = torch_gpu
        cap_diff = torch_gpu.capability != smi.capability
        vram_diff = abs(torch_gpu.vram_total_mb - smi.vram_total_mb) > 256
        if cap_diff or vram_diff:
            note = (
                f"sources disagree (torch={torch_gpu.capability}/{torch_gpu.vram_total_mb}MB, "
                f"nvidia-smi={smi.capability}/{smi.vram_total_mb}MB); torch wins"
            )
    elif torch_gpu is not None:
        source = "torch"
        gpu = torch_gpu
    elif smi is not None:
        source = "nvidia-smi"
        gpu = smi

    has_cuda = gpu is not None and gpu.capability is not None
    tier = tier_for(has_cuda, gpu.capability if gpu else None, gpu.vram_total_mb if gpu else 0)
    hw = {
        "cuda": has_cuda,
        "capability": list(gpu.capability) if gpu and gpu.capability else None,
        "gpu_name": gpu.name if gpu else None,
        "vram_total_mb": gpu.vram_total_mb if gpu else 0,
        "ram_total_mb": ram_mb(),
        "cpu_name": cpu_name_fn(),
        "cpu_threads": cpu_threads(),
        "disk_free_mb": disk_free(models_dir),
        "tier": tier,
        "wheel": wheel_for(tier),
        "source": source,
        "driver_version": smi.driver_version if smi is not None else None,
    }
    if note:
        hw["note"] = note
    return hw

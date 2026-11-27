"""Runtime component audit for the desktop shell and Python sidecar.

The page is intentionally useful in partially installed states:

- Python + websockets are enough for the frontend to connect over WS.
- torch / transformers / safetensors are required only for model inference.
- the Visual C++ runtime is shared by Python, torch native DLLs and WebView2.

Everything here is metadata/stdlib based. Importing torch from this module
would cost several seconds and would make a diagnostics page unavailable when
torch is the component being diagnosed, so the already-cached ``torch_info()``
from ``server.py`` is injected instead.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import shutil
import sys
import sysconfig
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

VC_RUNTIME_DLLS = (
    "msvcp140.dll",
    "msvcp140_1.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
)
INFERENCE_PACKAGES = (
    "torch",
    "transformers",
    "tokenizers",
    "safetensors",
    "numpy",
    "accelerate",
)


def _package(name: str) -> tuple[str | None, Path | None]:
    try:
        version = importlib.metadata.version(name)
    except (importlib.metadata.PackageNotFoundError, OSError, ValueError):
        version = None
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, OSError):
        spec = None
    location = None
    if spec is not None:
        origin = spec.origin
        if origin and origin not in {"built-in", "frozen"}:
            location = Path(origin).parent
        elif spec.submodule_search_locations:
            location = Path(next(iter(spec.submodule_search_locations)))
    return version, location


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path.absolute())
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _find_dll(name: str, roots: list[Path]) -> Path | None:
    for root in roots:
        candidate = root / name
        if candidate.is_file():
            return candidate
    found = shutil.which(name)
    return Path(found) if found else None


def _component(
    *,
    component_id: str,
    label: str,
    role: str,
    state: str,
    version: str | None,
    detail: str,
    location: str | None,
    managed_by: str,
    required: bool,
) -> dict[str, Any]:
    return {
        "id": component_id,
        "label": label,
        "role": role,
        "state": state,
        "version": version,
        "detail": detail,
        "location": location,
        "managed_by": managed_by,
        "required": required,
    }


def collect(
    *,
    torch_info: dict[str, Any] | None = None,
    hardware: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the runtime status payload consumed by ``runtime/status``.

    ``torch_info`` and ``hardware`` come from the server's cached probes. The
    function remains callable on its own for tests and first-run diagnostics.
    """
    app_root = Path(__file__).resolve().parent.parent
    python_exe = Path(sys.executable).resolve()
    site_packages = Path(sysconfig.get_paths().get("purelib") or "").resolve()
    torch_version, torch_location = _package("torch")
    transformers_version, transformers_location = _package("transformers")
    tokenizers_version, tokenizers_location = _package("tokenizers")
    numpy_version, numpy_location = _package("numpy")
    accelerate_version, accelerate_location = _package("accelerate")
    websockets_version, websockets_location = _package("websockets")
    safetensors_version, safetensors_location = _package("safetensors")
    pypdf_version, pypdf_location = _package("pypdf")

    torch_info = torch_info or {}
    hardware = hardware or {}
    cuda_available = bool(torch_info.get("cuda_available"))
    cuda_build = torch_info.get("cuda_build")
    recommended_wheel = str(hardware.get("wheel") or "cpu")

    if app_root / "runtime" / "python" in python_exe.parents:
        mode = "bundled"
    elif ".venv" in python_exe.parts:
        mode = "development"
    else:
        mode = "external"

    system32 = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32"
    dll_roots = _dedupe_paths(
        [
            python_exe.parent,
            site_packages / "torch" / "lib",
            app_root,
            *([app_root, app_root / "runtime" / "python"] if mode == "bundled" else []),
            system32,
        ]
    )
    dlls: list[dict[str, Any]] = []
    for name in VC_RUNTIME_DLLS:
        found = _find_dll(name, dll_roots)
        dlls.append(
            {
                "name": name,
                "found": found is not None,
                "path": str(found) if found else None,
                "bytes": found.stat().st_size if found else 0,
            }
        )
    vc_ready = all(dll["found"] for dll in dlls)

    torch_loaded = bool(torch_info.get("torch_installed", torch_version is not None))
    torch_state = "ready" if torch_version and torch_loaded else "missing"
    torch_detail = f"CUDA 构建 {cuda_build or 'CPU'}" if torch_loaded else (
        "PyTorch 包存在但导入失败"
        if torch_version
        else "未安装，模型推理不可用"
    )
    if torch_loaded and recommended_wheel == "cu129" and not cuda_available:
        torch_state = "warning"
        torch_detail = "已安装 PyTorch，但当前 CUDA 不可用；将回落 CPU 推理"
    if torch_loaded and recommended_wheel == "cpu" and cuda_build:
        torch_state = "warning"
        torch_detail = "当前硬件推荐 CPU 轮子，但安装的是 CUDA 构建"

    # CUDA and CPU wheels are alternatives, not two packages that must both
    # be present. Keep a separate card so a machine without a usable NVIDIA
    # driver still has an explicit, discoverable CPU fallback.
    if torch_loaded and not cuda_build:
        torch_cpu_state = "ready"
        torch_cpu_detail = "CPU 版 PyTorch 已安装；无 CUDA 时可独立运行"
        torch_cpu_version = torch_version
    elif torch_loaded:
        torch_cpu_state = "warning"
        torch_cpu_detail = (
            f"当前安装 CUDA {cuda_build} 构建；CPU 版是可选的无驱动/省显存替代"
        )
        torch_cpu_version = None
    else:
        torch_cpu_state = "missing" if recommended_wheel == "cpu" else "warning"
        torch_cpu_detail = (
            "当前硬件推荐 CPU 版 PyTorch；安装后才能启用本地推理"
            if recommended_wheel == "cpu"
            else "当前未完成 PyTorch 导入；CPU 版可作为无 CUDA 备选"
        )
        torch_cpu_version = None
    torch_cpu_required = recommended_wheel == "cpu" and not torch_loaded

    components = [
        _component(
            component_id="python",
            label="Python 运行时",
            role="本地服务",
            state="ready",
            version=platform.python_version(),
            detail=f"{platform.python_implementation()} · {mode}",
            location=str(python_exe),
            managed_by="应用托管" if mode == "bundled" else "开发环境",
            required=True,
        ),
        _component(
            component_id="websockets",
            label="WebSocket 服务",
            role="前后端通信",
            state="ready" if websockets_version else "missing",
            version=websockets_version,
            detail=(
                "本地消息通道可用"
                if websockets_version
                else "缺少 websockets，前端无法连接后端"
            ),
            location=str(websockets_location) if websockets_location else None,
            managed_by="应用托管" if mode == "bundled" else "开发环境",
            required=True,
        ),
        _component(
            component_id="torch",
            label="PyTorch",
            role="推理引擎",
            state=torch_state,
            version=torch_version,
            detail=torch_detail,
            location=str(torch_location) if torch_location else None,
            managed_by="应用托管" if mode == "bundled" else "开发环境",
            required=False,
        ),
        _component(
            component_id="torch-cpu",
            label="PyTorch CPU 版",
            role="推理引擎",
            state=torch_cpu_state,
            version=torch_cpu_version,
            detail=torch_cpu_detail,
            location=str(torch_location) if torch_cpu_version and torch_location else None,
            managed_by="应用托管" if mode == "bundled" else "开发环境",
            required=torch_cpu_required,
        ),
        _component(
            component_id="transformers",
            label="Transformers",
            role="模型加载",
            state="ready" if transformers_version else "missing",
            version=transformers_version,
            detail=(
                "模型加载接口可用"
                if transformers_version
                else "缺少模型加载接口，检测引擎不可用"
            ),
            location=str(transformers_location) if transformers_location else None,
            managed_by="应用托管" if mode == "bundled" else "开发环境",
            required=False,
        ),
        _component(
            component_id="safetensors",
            label="Safetensors",
            role="权重读取",
            state="ready" if safetensors_version else "missing",
            version=safetensors_version,
            detail=(
                "安全权重格式可读取"
                if safetensors_version
                else "缺少 safetensors，模型权重无法加载"
            ),
            location=str(safetensors_location) if safetensors_location else None,
            managed_by="应用托管" if mode == "bundled" else "开发环境",
            required=False,
        ),
        _component(
            component_id="tokenizers",
            label="Tokenizers",
            role="文本切分",
            state="ready" if tokenizers_version else "missing",
            version=tokenizers_version,
            detail=(
                "模型分词器可用"
                if tokenizers_version
                else "缺少 tokenizers，模型无法把文本转换成张量"
            ),
            location=str(tokenizers_location) if tokenizers_location else None,
            managed_by="应用托管" if mode == "bundled" else "开发环境",
            required=False,
        ),
        _component(
            component_id="numpy",
            label="NumPy",
            role="数值数组",
            state="ready" if numpy_version else "missing",
            version=numpy_version,
            detail=(
                "数值数组运行时可用"
                if numpy_version
                else "缺少 numpy，模型加载链路不完整"
            ),
            location=str(numpy_location) if numpy_location else None,
            managed_by="应用托管" if mode == "bundled" else "开发环境",
            required=False,
        ),
        _component(
            component_id="accelerate",
            label="Accelerate",
            role="设备调度",
            state="ready" if accelerate_version else "missing",
            version=accelerate_version,
            detail=(
                "设备加载策略可用"
                if accelerate_version
                else "缺少 accelerate，部分模型加载策略不可用"
            ),
            location=str(accelerate_location) if accelerate_location else None,
            managed_by="应用托管" if mode == "bundled" else "开发环境",
            required=False,
        ),
        _component(
            component_id="pypdf",
            label="PDF 解析",
            role="文档导入",
            state="ready" if pypdf_version else "warning",
            version=pypdf_version,
            detail=(
                "PDF 文本提取可用"
                if pypdf_version
                else "PDF 导入不可用；纯文本、TXT、MD、DOCX 不受影响"
            ),
            location=str(pypdf_location) if pypdf_location else None,
            managed_by="应用托管" if mode == "bundled" else "开发环境",
            required=False,
        ),
        _component(
            component_id="vc-runtime",
            label="VC++ 运行库",
            role="系统依赖",
            state="ready" if vc_ready else "warning",
            version=None,
            detail=(
                "MSVCP140 / VCRUNTIME140 已就绪"
                if vc_ready
                else "部分 DLL 未找到，原生扩展可能加载失败"
            ),
            location=next(
                (str(dll["path"]) for dll in dlls if dll["found"]),
                None,
            ),
            managed_by="系统 / 应用",
            required=True,
        ),
    ]

    local_service_ready = bool(websockets_version) and python_exe.is_file()
    package_versions = {
        "torch": torch_version,
        "transformers": transformers_version,
        "tokenizers": tokenizers_version,
        "safetensors": safetensors_version,
        "numpy": numpy_version,
        "accelerate": accelerate_version,
    }
    inference_ready = local_service_ready and torch_loaded and all(
        bool(package_versions[name]) for name in INFERENCE_PACKAGES
    )
    notes: list[str] = []
    if mode == "development":
        notes.append("当前运行于开发 venv；发布包应使用应用托管的 runtime。")
    if not vc_ready:
        notes.append("建议随安装包携带 VC++ 运行库，避免目标机缺 DLL。")
    if inference_ready and not cuda_available and recommended_wheel == "cu129":
        notes.append("硬件适合 CUDA 轮子，但当前 torch 未检测到可用 CUDA。")

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": mode,
        "python_version": platform.python_version(),
        "python_executable": str(python_exe),
        "app_root": str(app_root),
        "runtime_root": (
            str(app_root / "runtime") if mode == "bundled" else str(python_exe.parent.parent)
        ),
        "site_packages": str(site_packages),
        "model_root": str(app_root / "models"),
        "recommended_wheel": recommended_wheel,
        "torch_variant": (
            "cpu" if torch_loaded and not cuda_build else "cuda" if torch_loaded else "unavailable"
        ),
        "hardware": {
            "cuda": bool(hardware.get("cuda")),
            "gpu_name": hardware.get("gpu_name"),
            "driver_version": hardware.get("driver_version"),
            "capability": hardware.get("capability"),
            "tier": hardware.get("tier"),
            "wheel": recommended_wheel,
            "source": hardware.get("source"),
        },
        "local_service_ready": local_service_ready,
        "inference_ready": inference_ready,
        "components": components,
        "dlls": dlls,
        "notes": notes,
    }

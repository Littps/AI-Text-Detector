"""Runtime diagnostics stay useful when inference dependencies are absent."""
from __future__ import annotations

from pathlib import Path

import runtimeenv


def test_status_separates_local_service_from_inference() -> None:
    status = runtimeenv.collect(
        torch_info={"torch_installed": False},
        hardware={"wheel": "cu129"},
    )
    components = {item["id"]: item for item in status["components"]}
    assert components["python"]["state"] == "ready"
    assert components["websockets"]["state"] == "ready"
    assert components["torch"]["state"] == "missing"
    assert status["local_service_ready"] is True
    assert status["inference_ready"] is False


def test_status_reports_cuda_mismatch_as_warning() -> None:
    status = runtimeenv.collect(
        torch_info={"torch_installed": True, "cuda_available": False, "cuda_build": "12.9"},
        hardware={"wheel": "cu129"},
    )
    torch = next(item for item in status["components"] if item["id"] == "torch")
    assert torch["state"] == "warning"
    assert "CUDA" in torch["detail"]


def test_cpu_torch_is_a_separate_ready_component() -> None:
    status = runtimeenv.collect(
        torch_info={"torch_installed": True, "cuda_available": False, "cuda_build": None},
        hardware={"wheel": "cpu"},
    )
    components = {item["id"]: item for item in status["components"]}
    assert status["torch_variant"] == "cpu"
    assert components["torch-cpu"]["state"] == "ready"
    assert "CPU 版" in components["torch-cpu"]["detail"]


def test_cuda_build_is_optional_for_cpu_fallback() -> None:
    status = runtimeenv.collect(
        torch_info={
            "torch_installed": True,
            "cuda_available": True,
            "cuda_build": "12.9",
        },
        hardware={"wheel": "cu129"},
    )
    components = {item["id"]: item for item in status["components"]}
    assert status["torch_variant"] == "cuda"
    assert components["torch-cpu"]["state"] == "warning"
    assert components["torch-cpu"]["required"] is False


def test_cpu_fallback_is_missing_when_hardware_recommends_cpu() -> None:
    status = runtimeenv.collect(
        torch_info={"torch_installed": False},
        hardware={"wheel": "cpu"},
    )
    components = {item["id"]: item for item in status["components"]}
    assert status["torch_variant"] == "unavailable"
    assert components["torch-cpu"]["state"] == "missing"
    assert components["torch-cpu"]["required"] is True


def test_vc_runtime_search_can_find_temp_dll(tmp_path: Path) -> None:
    fake = tmp_path / "msvcp140.dll"
    fake.write_bytes(b"dll")
    assert runtimeenv._find_dll("msvcp140.dll", [tmp_path]) == fake
    assert runtimeenv._find_dll("definitely-missing-runtime.dll", [tmp_path]) is None

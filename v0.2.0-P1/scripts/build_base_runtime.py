"""Build the small runtime bundled with the desktop shell.

The base runtime intentionally contains only CPython embeddable and
``websockets``. PyTorch, Transformers, model weights and the rest of the
inference stack remain optional downloads managed by the app.

The Tauri build runs this script before bundling. The output directory is
ignored by git because it is a reproducible build artifact, not source.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMBED = ROOT / "downloads" / "runtime" / "python-3.14.8-embed-amd64.zip"
DEFAULT_SITE = ROOT / ".venv" / "Lib" / "site-packages"
DEFAULT_OUTPUT = ROOT / "runtime-base"
DEFAULT_BACKEND_SOURCE = ROOT / "backend"
DEFAULT_BACKEND_OUTPUT = ROOT / "backend-bundle"


def _copy_package(source: Path, target: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"missing package directory: {source}")
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def _copy_websockets(site_packages: Path, destination: Path) -> None:
    package = site_packages / "websockets"
    dist_infos = sorted(site_packages.glob("websockets-*.dist-info"))
    if not dist_infos:
        raise FileNotFoundError(f"websockets dist-info not found under {site_packages}")
    _copy_package(package, destination / "websockets")
    _copy_package(dist_infos[-1], destination / dist_infos[-1].name)


def build(
    output: Path,
    embed_zip: Path,
    site_packages: Path,
    backend_source: Path,
    backend_output: Path,
) -> Path:
    if not embed_zip.is_file():
        raise FileNotFoundError(f"embeddable Python archive not found: {embed_zip}")
    if not backend_source.is_dir():
        raise FileNotFoundError(f"backend source directory not found: {backend_source}")
    if output.resolve() == ROOT.resolve():
        raise ValueError("refusing to replace the project root")
    if output.exists():
        shutil.rmtree(output)
    if backend_output.resolve() == ROOT.resolve():
        raise ValueError("refusing to replace the project root with backend bundle")
    if backend_output.exists():
        shutil.rmtree(backend_output)

    python_dir = output / "python"
    python_dir.mkdir(parents=True)
    with zipfile.ZipFile(embed_zip) as archive:
        archive.extractall(python_dir)

    site = python_dir / "Lib" / "site-packages"
    site.mkdir(parents=True)
    _copy_websockets(site_packages, site)

    # The pth is evaluated relative to runtime/python. The Tauri resources keep
    # backend/ beside runtime/, so this mirrors the source-checkout layout.
    (python_dir / "python314._pth").write_text(
        "python314.zip\n.\nLib\\site-packages\n..\\..\\backend\nimport site\n",
        encoding="utf-8",
    )
    shutil.copytree(
        backend_source,
        backend_output,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    return output


def validate(output: Path) -> dict[str, str]:
    python = output / "python" / "python.exe"
    if not python.is_file():
        raise FileNotFoundError(f"base runtime interpreter not built: {python}")
    proc = subprocess.run(
        [
            str(python),
            "-c",
            "import websockets; print(websockets.__version__)",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "base runtime failed to import websockets:\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    return {"websockets": proc.stdout.strip()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--embed-zip", type=Path, default=DEFAULT_EMBED)
    parser.add_argument("--site-packages", type=Path, default=DEFAULT_SITE)
    parser.add_argument("--backend-source", type=Path, default=DEFAULT_BACKEND_SOURCE)
    parser.add_argument("--backend-output", type=Path, default=DEFAULT_BACKEND_OUTPUT)
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    output = build(
        args.output,
        args.embed_zip,
        args.site_packages,
        args.backend_source,
        args.backend_output,
    )
    result = validate(output) if not args.no_validate else {}
    print(f"base runtime: {output}")
    if result:
        print(f"websockets: {result['websockets']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

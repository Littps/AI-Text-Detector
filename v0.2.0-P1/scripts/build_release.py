"""Build the Windows release with reproducible path remapping.

Rust embeds source paths in panic-location metadata. Without remapping, a
release binary built on a developer workstation can contain the builder's
home directory and Cargo registry path. This wrapper keeps those paths out of
the shipped executable while forwarding all arguments to ``cargo tauri``.

Usage:
    python scripts/build_release.py --bundles nsis
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _path_variants(path: Path) -> tuple[str, ...]:
    return tuple(dict.fromkeys((str(path), path.as_posix())))


def build_rustflags(existing: str | None = None) -> str:
    """Return rustflags that remove builder-specific absolute paths."""
    workspace = ROOT
    home = Path.home()
    cargo_home = Path(os.environ.get("CARGO_HOME") or (home / ".cargo"))
    mappings: list[tuple[Path, str]] = [
        (home, r"C:\build\home"),
        (cargo_home / "registry", r"C:\build\cargo-registry"),
        (workspace, r"C:\build\project"),
    ]
    parts = [existing.strip()] if existing and existing.strip() else []
    for source, replacement in mappings:
        for variant in _path_variants(source):
            parts.append(f"--remap-path-prefix={variant}={replacement}")
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    env = os.environ.copy()
    env["RUSTFLAGS"] = build_rustflags(env.get("RUSTFLAGS"))
    cmd = ["cargo", "tauri", "build", *forwarded]
    print(f"rustflags: {env['RUSTFLAGS']}")
    print("build:", subprocess.list2cmdline(cmd))
    return subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

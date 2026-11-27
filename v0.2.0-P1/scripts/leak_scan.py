"""Pre-commit leak scan: personal paths, hostnames, IPs, secrets, emails.

Run before committing / releasing::

    python scripts/leak_scan.py [--root DIR] [--all]

Scope is exactly the set of files git would publish - tracked files plus
untracked-but-not-ignored ones (``git ls-files -c -o --exclude-standard``).
Ignored trees (models/, .venv/, node_modules/, internal docs) are out of
scope by construction.

Exit code 1 when anything is found, so it can gate a commit hook.

Why Python instead of grep: the bundled Git-for-Windows ``xargs`` aborts with
``argument list too long`` style assertions on this machine, which silently
turns a shell pipeline into a no-op - a leak scan that cannot fail is worse
than no scan.
"""
from __future__ import annotations

import argparse
import getpass
import os
import re
import subprocess
import sys
from pathlib import Path

SKIP_SUFFIX = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".icns", ".webp",
    ".woff", ".woff2", ".ttf", ".otf", ".onnx",
    ".safetensors", ".bin", ".pt", ".pth", ".zip", ".7z", ".gz",
    ".pdf",
}

# Local-only addresses that are fine to publish.
IP_ALLOW = {"127.0.0.1", "0.0.0.0", "255.255.255.255"}

INTERNAL_DOCS = [
    "docs/PLAN.md",
    "docs/DECISIONS.md",
    "docs/ENV-AUDIT.md",
    "docs/P0-KICKOFF.md",
    "docs/AUDIT-P2.md",
    "docs/WORKSPACE.md",
]

SECRET_PATTERNS = [
    ("hf-token", re.compile(r"\bhf_[A-Za-z0-9]{20,}")),
    ("openai-key", re.compile(r"\bsk-[A-Za-z0-9]{16,}")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("aws-key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
]

PATH_PATTERNS = [
    ("windows-user-path", re.compile(r"[A-Za-z]:\\+Users\\+[^\\\s\"'<>|]+")),
    ("posix-user-path", re.compile("/c/" + "Users/[^\\s\"'<>|]+")),
    ("home-path", re.compile(r"/home/[A-Za-z0-9._-]+")),
]

HOST_PATTERNS = [
    ("windows-hostname", re.compile(r"\b(?:DESKTOP|LAPTOP)-[A-Z0-9]{7}\b")),
]

IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
EMAIL_ALLOW_DOMAINS = (
    "example.com", "users.noreply.github.com", "github.com",
    "vitejs.dev", "tauri.app", "playwright.dev",
)
# "128x128@2x.png" style asset names look like emails; "foo@bar.types" too.
EMAIL_BAD_TLD = {
    "png", "jpg", "jpeg", "gif", "svg", "webp", "ico", "icns", "json",
    "js", "ts", "tsx", "py", "md", "lock", "txt", "bin", "onnx", "types",
}
# This scanner necessarily contains the patterns it hunts for.
SELF_NAME = "leak_scan.py"


def valid_ipv4(text: str) -> bool:
    parts = text.split(".")
    if len(parts) != 4:
        return False
    for p in parts:
        if not p.isdigit() or (len(p) > 1 and p[0] == "0") or int(p) > 255:
            return False
    return True


def repo_files(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-c", "-o", "--exclude-standard"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def identity_tokens() -> list[str]:
    """Local identity strings that must never reach a public repo."""
    names = {os.environ.get("USERNAME", ""), os.environ.get("USER", "")}
    try:
        username = getpass.getuser()
    except (KeyError, OSError):
        username = ""  # restricted env: no login name available
    if username:
        names.add(username)
    names.discard("")
    names.discard("root")
    return sorted(n for n in names if len(n) >= 3)


def scan(root: Path, files: list[str]) -> list[tuple[str, str, int, str]]:
    """Return (rule, file, line_no, snippet) tuples."""
    hits: list[tuple[str, str, int, str]] = []
    users = identity_tokens()
    for rel in files:
        if Path(rel).suffix.lower() in SKIP_SUFFIX:
            continue
        path = root / rel
        try:
            if path.stat().st_size > 4 * 1024 * 1024:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            snippet = line.strip()[:200]
            if not rel.endswith(SELF_NAME):
                for rule, pat in PATH_PATTERNS:
                    if pat.search(line):
                        hits.append((rule, rel, i, snippet))
            for rule, pat in HOST_PATTERNS:
                if pat.search(line):
                    hits.append((rule, rel, i, snippet))
            for ip in IP_RE.findall(line):
                if ip not in IP_ALLOW and valid_ipv4(ip):
                    hits.append(("non-loopback-ip:" + ip, rel, i, snippet))
            for rule, pat in SECRET_PATTERNS:
                if pat.search(line):
                    hits.append((rule, rel, i, snippet))
            for mail in EMAIL_RE.findall(line):
                tld = mail.rsplit(".", 1)[-1].lower()
                if any(d in mail for d in EMAIL_ALLOW_DOMAINS):
                    continue
                if tld in EMAIL_BAD_TLD:
                    continue
                hits.append(("email:" + mail, rel, i, snippet))
            for name in users:
                if re.search(rf"\b{re.escape(name)}\b", line, re.IGNORECASE):
                    hits.append(("local-identity:" + name, rel, i, snippet))
    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".", help="repo root (default: cwd)")
    ap.add_argument("--all", action="store_true", help="also list scanned files")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    files = repo_files(root)
    print(f"repo: {root}")
    print(f"files in publish scope: {len(files)}")
    print(f"local identity tokens watched: {identity_tokens()}")
    if args.all:
        for f in files:
            print("  ", f)

    # 1) internal docs must never be tracked
    tracked = set(subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True,
    ).stdout.splitlines())
    doc_leaks = [d for d in INTERNAL_DOCS if d in tracked]
    if doc_leaks:
        print("\n[!!] internal docs are tracked:")
        for d in doc_leaks:
            print("   -", d)

    hits = scan(root, files)
    if hits:
        print(f"\n[!!] {len(hits)} suspicious line(s):")
        for rule, rel, line_no, snippet in hits:
            print(f"   {rule}  {rel}:{line_no}  {snippet}")
    else:
        print("\n[ok] no path / host / ip / secret / email hits")

    if doc_leaks or hits:
        return 1
    print("[ok] publish scope looks clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())

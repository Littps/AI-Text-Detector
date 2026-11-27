"""Generate models/manifest.json - the P2 downloader catalogue.

Truth anchoring (D18 + five iron rules):
- every selected file's canonical SHA256 comes from the HF LFS oid when
  the file is LFS-tracked; locally present files are streamed-hashed and
  MUST match the remote oid/size or generation aborts (manifest never
  lies about weights);
- non-LFS small files (config/tokenizer) get size verification and a
  local sha256 when present (remote-only ones carry sha256=None and are
  size-checked by the downloader);
- remote-only engines (fdg, binoculars) record exact HF sizes so the
  download UI can show honest numbers before any byte moves.

Run: ./.venv/Scripts/python.exe scripts/gen_manifest.py [--no-proxy]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

VERSION_ROOT = Path(__file__).resolve().parents[1]
MODELS = VERSION_ROOT / "models"

# Tokenizer/config companions worth shipping (explicit allowlist, not a
# blanket grab - repos carry tflite/h5/msgpack/rust duplicates).
CONFIG_PATTERNS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "spm.model",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "added_tokens.json",
)
WEIGHT_EXCLUDE = (".bin", ".h5", ".msgpack", ".ot", ".tflite")

# D19/D20: 7 verified engines default; fdg/binoculars optional non-default.
# dest_prefix = on-disk directory under models/ (defaults to engine_id;
# gltr's shared GPT-2 base lives at models/gpt2/, hence the override).
ENGINE_TABLE = [
    {"engine_id": "modernbert-raid-mage", "repos": [("GeorgeDrayson/modernbert-ai-detection-raid-mage", "")]},
    {"engine_id": "desklib-en", "repos": [("desklib/ai-text-detector-v1.01", "")]},
    {"engine_id": "simpleai-zh", "repos": [("Hello-SimpleAI/chatgpt-detector-roberta-chinese", "")]},
    {"engine_id": "xiaoyu-zh", "repos": [("ronghua/xiaoyu-chinese-ai-detect", "")]},
    {"engine_id": "vanguard-en", "repos": [("ShantanuT01/vanguard-ai-text-detector", "")]},
    {"engine_id": "r1-lang-router", "repos": [("papluca/xlm-roberta-base-language-detection", "")]},
    {"engine_id": "gltr", "repos": [("openai-community/gpt2", "")], "dest_prefix": "gpt2"},
    # round-7: yuchuantian v3 trio (ICLR'24 Spotlight lineage, Apache-2.0).
    # zh = chinese-roberta-wwm-ext, en = roberta-base; .bin-only repos.
    {"engine_id": "aigc-zhv3", "repos": [("yuchuantian/AIGC_detector_zhv3", "")]},
    {"engine_id": "aigc-zhv3short", "repos": [("yuchuantian/AIGC_detector_zhv3short", "")]},
    {"engine_id": "aigc-env3", "repos": [("yuchuantian/AIGC_detector_env3", "")]},
    {
        "engine_id": "fast-detectgpt",
        "repos": [("EleutherAI/gpt-neo-2.7B", "")],
        "default": False,
        "risk_level": "medium",
        "risk_text_key": "risk.fdg.exclusive",
        "min_vram_mb": 8000,
    },
    {
        "engine_id": "binoculars",
        "repos": [("tiiuae/falcon-7b", "falcon-7b"), ("tiiuae/falcon-7b-instruct", "falcon-7b-instruct")],
        "default": False,
        "risk_level": "high",
        "risk_text_key": "risk.binoculars.8g",
        "min_vram_mb": 28000,
    },
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def hf_tree(repo: str, proxy: str | None) -> dict[str, dict]:
    """path -> {size, sha256(LFS only)} for one repo, via HF API."""
    url = f"https://huggingface.co/api/models/{repo}/tree/main"
    req = urllib.request.Request(url, headers={"User-Agent": "aigc-detector-manifest/1.0"})
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy} if proxy else {})
    )
    with opener.open(req, timeout=30) as resp:
        data = json.load(resp)
    out: dict[str, dict] = {}
    for f in data:
        if f.get("type") != "file":
            continue
        lfs = f.get("lfs")
        out[f["path"]] = {
            "size": int(lfs["size"]) if lfs else int(f.get("size", 0)),
            "sha256": lfs["oid"] if lfs else None,
        }
    return out


def select_files(tree: dict[str, dict]) -> list[str]:
    """Pick weights (safetensors-first) + config/tokenizer companions."""
    paths = list(tree)
    weights: list[str] = []
    if "model.safetensors" in tree:
        weights = ["model.safetensors"]
    else:
        sharded = sorted(p for p in paths if p.startswith("model-") and p.endswith(".safetensors"))
        if sharded:
            weights = sharded + ["model.safetensors.index.json"]
        elif "pytorch_model.bin" in tree:
            weights = ["pytorch_model.bin"]
    companions = [p for p in paths if p in CONFIG_PATTERNS or p in ("configuration_falcon.py", "modeling_falcon.py")]
    return sorted(set(weights + companions))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-proxy", action="store_true", help="bypass the dev proxy")
    args = ap.parse_args()
    proxy = None if args.no_proxy else "http://127.0.0.1:7897"

    engines: dict[str, dict] = {}
    for entry in ENGINE_TABLE:
        eid = entry["engine_id"]
        files: list[dict] = []
        warnings: list[str] = []
        for repo, prefix in entry["repos"]:
            tree = hf_tree(repo, proxy)
            dest_prefix = entry.get("dest_prefix", eid)
            rel_dir = Path(dest_prefix) / prefix if prefix else Path(dest_prefix)
            local_dir = MODELS / rel_dir
            for path in select_files(tree):
                remote = tree[path]
                # dest is relative to models/ root (explicit disk layout)
                dest = str(rel_dir / path).replace("\\", "/")
                local = local_dir / path
                sha: str | None = remote["sha256"]
                size = remote["size"]
                if local.is_file():
                    local_size = local.stat().st_size
                    if local_size != size:
                        print(
                            f"ABORT {eid}: {dest} local {local_size}B != remote {size}B",
                            file=sys.stderr,
                        )
                        return 1
                    local_sha = sha256_file(local)
                    if sha is not None and local_sha != sha:
                        print(
                            f"ABORT {eid}: {dest} local sha {local_sha[:16]} != LFS oid {sha[:16]}",
                            file=sys.stderr,
                        )
                        return 1
                    sha = local_sha  # non-LFS: pin the local hash
                else:
                    warnings.append(f"remote-only: {dest} ({size / 1e9:.2f}GB)")
                files.append({"repo": repo, "path": path, "dest": dest, "sha256": sha, "bytes": size})
            # local extras the manifest does not carry - never silent
            selected = {p for p in select_files(tree)}
            for local_file in local_dir.rglob("*") if local_dir.is_dir() else []:
                rel = str(local_file.relative_to(local_dir)).replace("\\", "/")
                if local_file.is_file() and rel not in selected:
                    warnings.append(f"local extra (not selected): {rel}")
        total = sum(f["bytes"] for f in files)
        engines[eid] = {
            "files": files,
            "default_selected": entry.get("default", True),
            "risk_level": entry.get("risk_level", "none"),
            "risk_text_key": entry.get("risk_text_key"),
            "min_vram_mb": entry.get("min_vram_mb", 0),
            "total_bytes": total,
        }
        flag = "OK " if not warnings else "WARN"
        print(f"{flag} {eid:24s} {len(files):2d} files  {total / 1e9:7.2f}GB")
        for w in warnings:
            print(f"     - {w}")

    manifest = {
        "version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "generator": "scripts/gen_manifest.py",
        "engines": engines,
    }
    out = MODELS / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

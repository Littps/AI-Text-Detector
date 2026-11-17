"""Engine catalogue - the default recommendation set (9 items, D14).

This is the *catalogue layer*: it describes what CAN be loaded, not what
IS loaded. ``verified=False`` means the repo id has not been probed yet
(queued for the P2 downloader work).
"""
from __future__ import annotations

from dataclasses import dataclass

# VRAM estimates are fp16 unless noted; Binoculars is bnb-4bit (D14).
VALID_LANGS = {"zh", "en", "multi"}
VALID_KINDS = {"router", "classifier", "method"}


@dataclass(frozen=True, slots=True)
class EngineSpec:
    engine_id: str
    name: str
    lang: str
    kind: str
    est_vram_mb: int
    repo: str
    exclusive: bool = False
    verified: bool = False  # repo id probed successfully?
    notes: str = ""


ENGINE_SPECS: tuple[EngineSpec, ...] = (
    EngineSpec(
        engine_id="r1-lang-router",
        name="Language router (papluca xlm-r)",
        lang="multi",
        kind="router",
        est_vram_mb=560,
        repo="papluca/xlm-roberta-base-language-detection",
        notes="Infrastructure: route requests to zh/en engines by detected language.",
    ),
    EngineSpec(
        engine_id="simpleai-zh",
        name="SimpleAI Chinese detector",
        lang="zh",
        kind="classifier",
        est_vram_mb=400,
        repo="Hello-SimpleAI/chatgpt-detector-roberta-chinese",
        notes="HC3-Chinese corpus; repo id to verify in P2.",
    ),
    EngineSpec(
        engine_id="xiaoyu-zh",
        name="Xiaoyu Chinese detector",
        lang="zh",
        kind="classifier",
        est_vram_mb=220,
        repo="ronghua/xiaoyu-chinese-ai-detect",
        notes="Doc+sentence dual output; int8 101MB / int4 60MB variants exist.",
    ),
    EngineSpec(
        engine_id="modernbert-raid-mage",
        name="ModernBERT raid-mage (EN)",
        lang="en",
        kind="classifier",
        est_vram_mb=300,
        repo="GeorgeDrayson/modernbert-ai-detection-raid-mage",
        notes="arXiv 2502.15654 official weights.",
    ),
    EngineSpec(
        engine_id="desklib-en",
        name="Desklib DeBERTa-v3-large (EN)",
        lang="en",
        kind="classifier",
        est_vram_mb=870,
        repo="desklib/ai-text-detector-v1.01",
        notes="RAID leaderboard top pick; precise mode.",
    ),
    EngineSpec(
        engine_id="vanguard-en",
        name="PAN vanguard ModernBERT (EN)",
        lang="en",
        kind="classifier",
        est_vram_mb=755,
        repo="ShantanuT01/vanguard-ai-text-detector",
    ),
    EngineSpec(
        engine_id="gltr",
        name="GLTR / perplexity (EN)",
        lang="en",
        kind="method",
        est_vram_mb=250,
        repo="gpt2",
        notes="Token-rank statistics on a small GPT-2 base; zero-shot.",
    ),
    EngineSpec(
        engine_id="fast-detectgpt",
        name="Fast-DetectGPT (EN)",
        lang="en",
        kind="method",
        est_vram_mb=250,
        repo="gpt2",
        notes="Conditional-probability curvature; base LM = GPT-2 family.",
    ),
    EngineSpec(
        engine_id="binoculars",
        name="Binoculars (EN, heavy)",
        lang="en",
        kind="method",
        est_vram_mb=7500,
        repo="falcon-7b + falcon-7b-instruct (bnb 4bit)",
        exclusive=True,
        notes="fp16 pair would be ~28GB; 4bit only, runs alone. May OOM on 8GB.",
    ),
)

_BY_ID = {s.engine_id: s for s in ENGINE_SPECS}


def spec(engine_id: str) -> EngineSpec | None:
    return _BY_ID.get(engine_id)


def all_ids() -> list[str]:
    return [s.engine_id for s in ENGINE_SPECS]


def validate() -> None:
    """Catalogue sanity checks; called once at import from loader."""
    ids = [s.engine_id for s in ENGINE_SPECS]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate engine_id in registry")
    for s in ENGINE_SPECS:
        if s.lang not in VALID_LANGS:
            raise ValueError(f"{s.engine_id}: bad lang {s.lang!r}")
        if s.kind not in VALID_KINDS:
            raise ValueError(f"{s.engine_id}: bad kind {s.kind!r}")
        if s.est_vram_mb <= 0:
            raise ValueError(f"{s.engine_id}: est_vram_mb must be positive")

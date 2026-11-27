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
        verified=True,
        notes=(
            "Infrastructure: route requests to zh/en engines by detected "
            "language. Probe ok 2026-11-17 (zh 0.99 / en 0.98, casual + "
            "formal); RouterEngine.detect returns no segments by contract."
        ),
    ),
    EngineSpec(
        engine_id="simpleai-zh",
        name="SimpleAI Chinese detector",
        lang="zh",
        kind="classifier",
        est_vram_mb=400,
        repo="Hello-SimpleAI/chatgpt-detector-roberta-chinese",
        verified=True,
        notes=(
            "HC3-Chinese corpus; standard BertForSequenceClassification, "
            "id2label {0: Human, 1: ChatGPT} = AI_PROB_INDEX 1. Probe ok "
            "2026-11-17 (fp16 202 MB; AI 0.94 / casual human 0.0001)."
        ),
    ),
    EngineSpec(
        engine_id="xiaoyu-zh",
        name="Xiaoyu Chinese detector",
        lang="zh",
        kind="classifier",
        est_vram_mb=220,
        repo="ronghua/xiaoyu-chinese-ai-detect",
        verified=True,
        notes=(
            "Single-logit head (sigmoid 1 = AI); casual Chinese is in "
            "distribution. Probe ok 2026-11-17 (fp16 202 MB; AI 0.9998 / "
            "human 0.005); sha256sums.txt shipped with weights. DOMAIN CAVEAT "
            "2026-11-25 (D44): formal/written Chinese is systematically "
            "over-flagged - human WORKSPACE.md sentences scored 6/20 >= 0.5, "
            "max 0.9998, and raising context made it monotonically worse. "
            "Second-opinion engine only; removed from balanced/zh."
        ),
    ),
    EngineSpec(
        engine_id="modernbert-raid-mage",
        name="ModernBERT raid-mage (EN)",
        lang="en",
        kind="classifier",
        est_vram_mb=300,
        repo="GeorgeDrayson/modernbert-ai-detection-raid-mage",
        verified=True,
        notes="arXiv 2502.15654 official weights; probe ok 2026-11-17 (ungated, loads fp16).",
    ),
    EngineSpec(
        engine_id="desklib-en",
        name="Desklib DeBERTa-v3-large (EN)",
        lang="en",
        kind="classifier",
        est_vram_mb=870,
        repo="desklib/ai-text-detector-v1.01",
        verified=True,
        notes=(
            "RAID leaderboard top pick; DOCUMENT-level detector (custom architecture "
            "replicated locally, no trust_remote_code). Probe ok 2026-11-17: official "
            "README pair 0.9974 AI / <0.5 human, fp16 stable (828 MB), sha256 verified. "
            "Sentence slices are OOD by design - short casual text will over-flag; "
            "pair with a sentence-level engine for localization."
        ),
    ),
    EngineSpec(
        engine_id="vanguard-en",
        name="PAN vanguard ModernBERT (EN)",
        lang="en",
        kind="classifier",
        est_vram_mb=755,
        repo="ShantanuT01/vanguard-ai-text-detector",
        verified=True,
        notes=(
            "Single-logit ModernBERT (sigmoid 1 = AI). Probe ok 2026-11-17 "
            "(fp16 755 MB; work-style human 0.000 / AI 1.0). Casual chat is "
            "OOD and over-flags - pair with zh engines' robustness notes."
        ),
    ),
    EngineSpec(
        engine_id="aigc-zhv3",
        name="AIGC Detector zh v3 (CN)",
        lang="zh",
        kind="classifier",
        est_vram_mb=220,
        repo="yuchuantian/AIGC_detector_zhv3",
        verified=True,
        notes=(
            "ICLR'24 Spotlight lineage (Tian et al., PU loss); v3 targets "
            "DeepSeek-V3 / GPT-4 era generators. chinese-roberta-wwm-ext, "
            "BertForSequenceClassification, pytorch_model.bin 390MB. "
            "Probe 2026-11-21: strict pair AI 0.9962 / human 0.0775; battery "
            "human median 0.110 max 0.392. HC3-lineage caveat: formal/wiki "
            "prose can over-flag."
        ),
    ),
    EngineSpec(
        engine_id="aigc-zhv3short",
        name="AIGC Detector zh v3 short (CN)",
        lang="zh",
        kind="classifier",
        est_vram_mb=220,
        repo="yuchuantian/AIGC_detector_zhv3short",
        verified=True,
        notes=(
            "Short-text variant of zhv3 (same backbone/size), intended for "
            "sentence-level inputs. Battery 2026-11-21: AI 0.9998, human "
            "median 0.292 / max 0.597 - one very short casual sentence "
            "over-flags (0.60), so treat single-sentence verdicts as signal."
        ),
    ),
    EngineSpec(
        engine_id="aigc-env3",
        name="AIGC Detector en v3 (EN)",
        lang="en",
        kind="classifier",
        est_vram_mb=250,
        repo="yuchuantian/AIGC_detector_env3",
        verified=True,
        notes=(
            "English sibling of zhv3: roberta-base 125M, 475MB bin. "
            "Apache-2.0; adds new-generator coverage next to the RAID-trained "
            "en tier. Battery 2026-11-21: AI 0.9996, human median 0.321 / max "
            "0.591 (casual work-mail occasionally over-flags)."
        ),
    ),
    EngineSpec(
        engine_id="gltr",
        name="GLTR / perplexity (EN)",
        lang="en",
        kind="method",
        est_vram_mb=250,
        repo="gpt2",
        verified=True,
        notes=(
            "Zero-shot top-k + perplexity fusion, CONTINUATION scoring on "
            "GPT-2 small (fp16 243 MB, sha256 verified). Probe ok 2026-11-17: "
            "AI 0.68-0.96 / work-style human 0.08-0.25; encyclopedic prose "
            "over-flags (zero-shot limit, signal not verdict). Chunked fp32 "
            "log-softmax keeps detect transients ~52MB. Documents over 1024 "
            "GPT-2 tokens are truncated - tail sentences unscored, UI should "
            "surface this."
        ),
    ),
    EngineSpec(
        engine_id="fast-detectgpt",
        name="Fast-DetectGPT (EN)",
        lang="en",
        kind="method",
        est_vram_mb=5400,
        repo="EleutherAI/gpt-neo-2.7B",
        exclusive=True,
        notes=(
            "Real implementation, WIRED since round-6e (2026-11-20): runnability "
            "now follows the plain install-state rule - no boot stub. Sampling "
            "fixed to score generated tokens only (prompt contamination was "
            "deflating the variance). D19 history: gpt2-family probes negative/"
            "contradictory; production base gpt-neo-2.7B fp32-only 10.67GB "
            "download, fp16 ~5.4GB residency, exclusive. CENTER/SCALE "
            "provisional - re-probe MUST pass before scores are trusted."
        ),
    ),
    EngineSpec(
        engine_id="binoculars",
        name="Binoculars (EN, heavy)",
        lang="en",
        kind="method",
        est_vram_mb=28000,
        repo="tiiuae/falcon-7b + tiiuae/falcon-7b-instruct",
        exclusive=True,
        notes=(
            "Real implementation landed round-7: document-level ratio "
            "B = log(PPL_observer)/log(X-PPL), logistic mapping around a "
            "provisional 0.9015 threshold (uncalibrated - no local weights). "
            "fp16 pair ~28GB VRAM, exclusive; 8GB cards WILL OOM. No 4-bit "
            "path: bitsandbytes absent and its cp314 wheels are unverified. "
            "Download is 28.87GB; truncates at 1024 tokens."
        ),
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

"""EnsemJudge - ensemble **framework** (rules-only subset).

Reference: Wang et al., "EnsemJudge: Enhancing Reliability in Chinese
LLM-Generated Text Detection through Diverse Model Ensembles", NLPCC 2025
Shared Task 1 (top-1 system). The published system has 18 voters:

- 2 fine-tuned Chinese encoders (roberta-wwm-ext-large / bert-base)
- GLM-4-9B LoRA + 3 x Qwen2.5-7B LoRA voters
- 3 Fast-DetectGPT voters and 1 Binoculars voter over those LLM backbones
- 5 rule voters + 2 joint-feature MLP fusers

Honest scope statement:
- **Not a faithful reproduction.** The 7B/9B neural voters are absent:
  their bases alone are ~33GB and the official runtime asks for 2xV100.
  The upstream code is licensed for competition evaluation only and the
  weights repo declares no licence, so we do not copy its tables/code.
- What this module provides: the *framework* (length-based strategy
  assignment, per-strategy voter weights, weighted voting, logistic
  score) plus five rule voters implemented independently from the paper
  description. Scores are heuristic signals, not EnsemJudge outputs.
- Not registered in the engine catalogue: without the neural voters a
  catalogue entry would imply a capability we cannot deliver.

The user plans to supply the missing theoretical logic later; the
extension point is ``Voter`` + ``PROFILES`` - adding a neural voter only
means appending one entry with its per-strategy weights.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .base import EngineBase, Segment

# ---------------------------------------------------------------------------
# strategy assignment (length clusters from the paper's description)
# ---------------------------------------------------------------------------

EXTREME_SHORT_MAX = 30  # characters
SHORT_MAX = 80


def length_profile(text: str) -> str:
    n = len(text.strip())
    if n <= EXTREME_SHORT_MAX:
        return "extreme_short"
    if n <= SHORT_MAX:
        return "short"
    return "medium"


@dataclass(frozen=True)
class Verdict:
    profile: str
    score: float
    weighted_sum: float
    total_weight: int
    votes: dict[str, int]  # voter name -> +1 AI / -1 human / 0 abstain


@dataclass(frozen=True)
class Voter:
    name: str
    fn: Callable[[str], int]
    weights: dict[str, int] = field(default_factory=dict)

    def weight(self, profile: str) -> int:
        return self.weights.get(profile, 0)

    def vote(self, text: str) -> int:
        v = self.fn(text)
        return v if v in (-1, 0, 1) else 0


# ---------------------------------------------------------------------------
# rule voters (independent implementations, documented as approximations)
# ---------------------------------------------------------------------------

# Small curated boilerplate list - deliberately short and transparent.
_BOILERPLATE_ZH = (
    "首先", "其次", "最后", "综上所述", "总而言之", "值得注意的是",
    "需要注意的是", "总的来说", "不仅可以", "而且还能", "在当今社会",
    "随着科技的发展", "希望对你有帮助", "以下是",
)
_BOILERPLATE_EN = (
    "first and foremost", "in conclusion", "it is important to note",
    "furthermore", "moreover", "additionally", "in today's world",
    "here are several", "i hope this helps", "overall,",
)

_ZW = ("\u200b", "\u200c", "\u200d", "\ufeff")
_REPEAT_PUNCT = re.compile(r"([，。！？；：、,.!?;:])\1+")
_SENT_SPLIT = re.compile(r"[。！？.!?]+")
_CLAUSE_SPLIT = re.compile(r"[，,；;、]")


def vote_special_token(text: str) -> int:
    """Zero-width chars / double blank lines / mixed-width spaces -> human.

    AI output is usually clean; these artifacts come from copy-paste and
    hand editing. A single hit is weak, so require two signals.
    """
    hits = sum(1 for z in _ZW if z in text)
    hits += len(re.findall(r"\n\s*\n", text))
    hits += len(re.findall(r"[\u3000]", text))
    return -1 if hits >= 2 else 0


def vote_consecutive_punctuation(text: str) -> int:
    """Repeated punctuation runs (!!!, 。。。。) -> human signal."""
    return -1 if _REPEAT_PUNCT.search(text) else 0


def vote_sentence_segment(text: str) -> int:
    """Uniform clause counts across sentences -> AI signal.

    AI prose tends to keep a steady 3-4 clause rhythm; human writing is
    burstier. Needs at least 3 sentences to say anything.
    """
    sents = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    if len(sents) < 3:
        return 0
    counts = [len(_CLAUSE_SPLIT.findall(s)) + 1 for s in sents]
    mean = sum(counts) / len(counts)
    var = sum((c - mean) ** 2 for c in counts) / len(counts)
    cv = math.sqrt(var) / mean if mean else 0.0
    if cv < 0.25 and 2.5 <= mean <= 5.0:
        return 1
    if cv > 0.75:
        return -1
    return 0


def vote_common_phrase(text: str) -> int:
    low = text.lower()
    hits = sum(1 for p in _BOILERPLATE_ZH if p in text)
    hits += sum(1 for p in _BOILERPLATE_EN if p in low)
    per_100 = hits * 100 / max(len(text), 1)
    if hits >= 2 and per_100 >= 0.5:
        return 1
    return 0


_HUMAN_TOKENS_ZH = ("卧槽", "哈哈", "烦死", "摸鱼", "撸串", "咋", "啥", "呗", "算了")
_MACHINE_TOKENS_ZH = ("此外", "因此", "然而", "综上", "首先", "其次", "总之", "值得注意")
_HUMAN_TOKENS_EN = ("btw", "ok so", "lol", "gonna", "kinda", "tbh", "idk")
_MACHINE_TOKENS_EN = ("furthermore", "moreover", "therefore", "additionally", "consequently")


def vote_common_token(text: str) -> int:
    low = text.lower()
    hum = sum(text.count(t) for t in _HUMAN_TOKENS_ZH) + sum(low.count(t) for t in _HUMAN_TOKENS_EN)
    mach = sum(text.count(t) for t in _MACHINE_TOKENS_ZH) + sum(
        low.count(t) for t in _MACHINE_TOKENS_EN
    )
    if hum == mach:
        return 0
    return 1 if mach > hum else -1


# ---------------------------------------------------------------------------
# framework
# ---------------------------------------------------------------------------

# Weights are ours, not the upstream competition table (licence + missing
# voters). Rule voters only -> small, interpretable weights.
DEFAULT_VOTERS: tuple[Voter, ...] = (
    Voter("special_token", vote_special_token, {"extreme_short": 5, "short": 5, "medium": 5}),
    Voter(
        "consecutive_punctuation",
        vote_consecutive_punctuation,
        {"extreme_short": 10, "short": 10, "medium": 10},
    ),
    Voter("sentence_segment", vote_sentence_segment, {"extreme_short": 0, "short": 10, "medium": 10}),
    Voter("common_phrase", vote_common_phrase, {"extreme_short": 10, "short": 15, "medium": 15}),
    Voter("common_token", vote_common_token, {"extreme_short": 15, "short": 15, "medium": 10}),
)


def judge(text: str, voters: tuple[Voter, ...] = DEFAULT_VOTERS, temperature: float = 20.0) -> Verdict:
    """Weighted vote -> logistic score in [0, 1] (0.5 = no signal)."""
    profile = length_profile(text)
    votes: dict[str, int] = {}
    total = 0
    acc = 0
    for v in voters:
        w = v.weight(profile)
        if w == 0:
            continue
        vote = v.vote(text)
        votes[v.name] = vote
        total += w
        acc += w * vote
    if total == 0:
        return Verdict(profile, 0.5, 0.0, 0, votes)
    scale = temperature / total
    z = acc * scale
    score = 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
    return Verdict(profile, round(score, 4), acc, total, votes)


class EnsemJudgeEngine(EngineBase):
    """Framework-only engine: no weights, document-level single segment.

    Kept out of the catalogue on purpose (see module docstring). If it is
    later registered, ``load`` is a no-op and ``detect`` returns one
    whole-text segment, exactly like the other document-level engines.
    """

    def __init__(self) -> None:
        self.engine_id = "ensemjudge"
        self.lang = "zh"
        self.kind = "method"
        self.est_vram_mb = 1
        self.exclusive = False
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def detect(self, text: str, on_batch: Callable[[list[Segment]], None] | None = None) -> list[Segment]:
        if not self._loaded:
            self.load()
        verdict = judge(text)
        segs = [Segment(0, len(text), verdict.score)]
        if on_batch is not None:
            on_batch(segs)
        return segs

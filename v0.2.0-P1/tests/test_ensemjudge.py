"""EnsemJudge framework tests (rules-only subset, no 7B/9B weights)."""
from __future__ import annotations

import pytest
from engines.ensemjudge import (
    DEFAULT_VOTERS,
    EXTREME_SHORT_MAX,
    SHORT_MAX,
    EnsemJudgeEngine,
    Voter,
    judge,
    length_profile,
    vote_common_phrase,
    vote_common_token,
    vote_consecutive_punctuation,
    vote_sentence_segment,
    vote_special_token,
)


def test_length_profiles_follow_the_documented_cutoffs():
    assert length_profile("短" * EXTREME_SHORT_MAX) == "extreme_short"
    assert length_profile("字" * (EXTREME_SHORT_MAX + 1)) == "short"
    assert length_profile("文" * SHORT_MAX) == "short"
    assert length_profile("篇" * (SHORT_MAX + 1)) == "medium"


def test_special_token_flags_copy_paste_artifacts():
    clean = "这是一段正常文本，没有奇怪字符。"
    assert vote_special_token(clean) == 0
    assert vote_special_token("第一行\n\n第二行\u200b\u200b") == -1


def test_consecutive_punctuation_is_a_human_signal():
    assert vote_consecutive_punctuation("正常的一句话。") == 0
    assert vote_consecutive_punctuation("真的吗？？？") == -1


def test_sentence_segment_needs_three_sentences():
    assert vote_sentence_segment("短句，一句话。") == 0
    uniform = "第一点，我们要明确目标，拆解任务，设定优先级。第二点，我们要保持专注，减少干扰，持续执行。第三点，我们要定期复盘，记录问题，优化流程。"
    assert vote_sentence_segment(uniform) == 1


def test_boilerplate_and_token_voters_move_in_the_expected_direction():
    ai = "首先，制定计划；其次，执行计划；最后，复盘总结。"
    human = "卧槽今天真烦，算了算了，下班撸串去。"
    assert vote_common_phrase(ai) == 1
    assert vote_common_token(ai) == 1
    assert vote_common_token(human) == -1


def test_judge_returns_neutral_without_any_vote():
    # A one-character text trips no voter -> 0.5, total weight still counted.
    v = judge("。")
    assert v.votes == {} or all(x == 0 for x in v.votes.values())
    assert v.score == pytest.approx(0.5)


def test_judge_pushes_ai_text_high_and_human_text_low():
    ai = "首先，我们需要明确目标；其次，拆解任务并设定优先级；最后，定期复盘并持续改进。"
    human = "卧槽今天这破班真不想上了，领导又让加班，烦死了。算了算了，下班撸个串去，明天再说吧"
    assert judge(ai).score > 0.8
    assert judge(human).score < 0.2


def test_weighted_vote_matches_manual_math():
    only = (Voter("yes", lambda t: 1, {"medium": 10}),)
    v = judge("文" * (SHORT_MAX + 1), voters=only, temperature=20.0)  # medium profile
    assert v.weighted_sum == 10 and v.total_weight == 10
    assert v.score == pytest.approx(0.9999999984, rel=1e-6)


def test_default_voters_are_all_zero_weight_for_extreme_short_segment_rules():
    seg = next(v for v in DEFAULT_VOTERS if v.name == "sentence_segment")
    assert seg.weight("extreme_short") == 0


def test_engine_contract_is_document_level():
    engine = EnsemJudgeEngine()
    assert not engine.loaded
    out = engine.detect("首先，制定计划；其次，执行计划；最后，复盘总结。")
    assert engine.loaded
    assert len(out) == 1 and out[0].offset == 0 and out[0].length > 0
    engine.unload()
    assert not engine.loaded

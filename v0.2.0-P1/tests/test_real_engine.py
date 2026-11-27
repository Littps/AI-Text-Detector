"""Round-22: the input-length ceiling must follow the model, not the wish.

``MAX_TOKENS = 1024`` is a preference. BERT/RoBERTa models build
``embeddings + position_embeddings``, so a single token past the table raises
``RuntimeError`` - it does not truncate. xiaoyu is a 512-position BERT, so one
unpunctuated 600-character "sentence" used to abort a whole detect run.

The clamp is deliberately type-gated: DeBERTa-v2 (relative bias) and
ModernBERT (rotary) have no position wall and must keep their larger limits.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from engines.real import RealClassifierEngine
from engines.registry import spec as engine_spec


def _limit(model_type: str, pos: int | None, max_tokens: int = 1024) -> int:
    engine = RealClassifierEngine(engine_spec("xiaoyu-zh"))
    engine.MAX_TOKENS = max_tokens
    engine._model = SimpleNamespace(
        config=SimpleNamespace(model_type=model_type, max_position_embeddings=pos)
    )
    return engine._input_limit()


def test_bert_position_table_caps_the_input() -> None:
    """xiaoyu / simpleai / zhv3: 512 positions behind a 1024 preference."""
    assert _limit("bert", 512) == 512


def test_roberta_offset_table_caps_the_input() -> None:
    """aigc-env3 and r1-lang-router ship 514 (512 + 2 padding offset)."""
    assert _limit("roberta", 514) == 514
    assert _limit("xlm-roberta", 514) == 514


def test_modernbert_keeps_the_larger_limit() -> None:
    """8192 rotary positions: MAX_TOKENS stays the binding constraint."""
    assert _limit("modernbert", 8192) == 1024


def test_deberta_v2_is_not_capped() -> None:
    """desklib uses relative bias; 768 was verified to run on a 512 config."""
    assert _limit("deberta-v2", 512, max_tokens=768) == 768


def test_missing_or_odd_config_falls_back_to_the_preference() -> None:
    assert _limit("bert", None) == 1024
    assert _limit("bert", 0) == 1024
    engine = RealClassifierEngine(engine_spec("xiaoyu-zh"))
    engine._model = SimpleNamespace()  # no config at all
    assert engine._input_limit() == 1024


def test_a_roomy_table_does_not_raise_the_preference() -> None:
    """The clamp only ever lowers: a 4096-position BERT still uses 1024."""
    assert _limit("bert", 4096) == 1024

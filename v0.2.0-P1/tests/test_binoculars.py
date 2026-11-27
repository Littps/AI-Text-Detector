"""Binoculars math + wiring tests (no 28GB weights required).

The pure ratio function and the logistic mapping are checked against
hand-computed values; the detect plumbing is exercised with fake models
that return deterministic logits (uniform distributions -> PPL == X-PPL
-> B == 1.0 exactly).
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from engines import binoculars
from engines.binoculars import BinocularsEngine, binoculars_ratio, ratio_to_score
from engines.loader import default_engine
from engines.registry import spec


def test_ratio_is_one_when_observer_and_performer_agree():
    logp = [-math.log(4.0)] * 6
    ce = [math.log(4.0)] * 6
    assert binoculars_ratio(logp, ce) == pytest.approx(1.0)


def test_ratio_rises_when_observer_is_more_confident_than_cross_entropy():
    # observer logp -0.1 (ppl ~1.1), cross-entropy 2.0 (x-ppl ~7.4) -> B < 1
    logp = [-0.1] * 8
    ce = [2.0] * 8
    b = binoculars_ratio(logp, ce)
    assert b == pytest.approx(math.log(math.exp(0.1)) / math.log(math.exp(2.0)), rel=1e-9)
    assert b < 1.0


def test_ratio_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        binoculars_ratio([], [])
    with pytest.raises(ValueError):
        binoculars_ratio([-1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        binoculars_ratio([-1.0], [0.0])


def test_score_is_monotone_and_centered_on_threshold():
    assert ratio_to_score(0.5) > ratio_to_score(0.9)
    assert ratio_to_score(0.9015) == pytest.approx(0.5)
    assert 0.0 < ratio_to_score(0.0) <= 1.0
    assert 0.0 <= ratio_to_score(10.0) < 0.01


def test_score_rejects_bad_scale():
    with pytest.raises(ValueError):
        ratio_to_score(1.0, scale=0.0)


def test_loader_wires_the_real_class():
    assert isinstance(default_engine(spec("binoculars")), BinocularsEngine)


def test_load_without_weights_reports_the_pair_gap(tmp_path, monkeypatch):
    monkeypatch.setattr(binoculars, "VERSION_ROOT", tmp_path)
    engine = BinocularsEngine(spec("binoculars"))
    with pytest.raises(binoculars.WeightsMissing) as ei:
        engine.load()
    assert "incomplete weight pair" in str(ei.value)
    assert "4-bit path not implemented" in str(ei.value)


class _FakeTok:
    def __call__(self, text, **_kw):
        import torch

        # 20 tokens (above MIN_TOKENS), all inside the fake vocab of 5
        ids = torch.tensor([[1, 2, 3, 4] * 5], dtype=torch.long)
        return {"input_ids": ids}


class _FakeLM:
    """Uniform logits over a tiny vocab -> identical distributions."""

    def __init__(self, vocab: int = 5) -> None:
        self.vocab = vocab

    def __call__(self, ids):
        import torch

        logits = torch.zeros((1, ids.shape[1], self.vocab), dtype=torch.float32)
        return SimpleNamespace(logits=logits)

    def eval(self):
        return self


def _loaded_fake_engine() -> BinocularsEngine:
    engine = BinocularsEngine(spec("binoculars"))
    engine._tokenizer = _FakeTok()
    engine._observer = _FakeLM()
    engine._performer = _FakeLM()
    engine._device = "cpu"
    return engine


def test_detect_matches_manual_math_for_identical_models():
    engine = _loaded_fake_engine()
    out = engine.detect("x" * 20)
    assert len(out) == 1
    seg = out[0]
    # uniform obs == uniform perf -> B = 1.0 -> sigmoid((0.9015-1)/0.05)
    assert seg.offset == 0 and seg.length == 20
    assert seg.score == pytest.approx(round(ratio_to_score(1.0), 4))


def test_detect_streams_one_document_segment():
    engine = _loaded_fake_engine()
    seen = []
    engine.detect("x" * 20, on_batch=seen.append)
    assert len(seen) == 1 and len(seen[0]) == 1


def test_detect_returns_nothing_for_tiny_inputs(monkeypatch):
    engine = _loaded_fake_engine()

    class _ShortTok:
        def __call__(self, text, **_kw):
            import torch

            return {"input_ids": torch.arange(1, 4, dtype=torch.long).unsqueeze(0)}

    engine._tokenizer = _ShortTok()
    assert engine.detect("ab") == []

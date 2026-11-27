"""EngineLoader facade: batch detect, injection, honest failure statuses."""
import json

import pytest
from engines.base import Segment
from engines.loader import EngineLoader
from engines.registry import EngineSpec
from fakes import FakeEngine

# Mini-catalogue for injection tests (engines must exist in the catalogue).
TEST_SPECS = tuple(
    EngineSpec(engine_id=eid, name=n, lang=lang, kind=kind, est_vram_mb=mb, repo="-")
    for eid, n, lang, kind, mb in [
        ("fake-a", "Fake A", "en", "classifier", 100),
        ("fake-b", "Fake B", "zh", "classifier", 150),
        ("fake-heavy", "Fake Heavy", "en", "method", 7500),
    ]
)


def make_loader(budget=1000):
    engines = {
        "fake-a": FakeEngine("fake-a", segments=[Segment(0, 4, 0.9)]),
        "fake-b": FakeEngine("fake-b", lang="zh", est_vram_mb=150, segments=[Segment(5, 3, 0.5)]),
        "fake-heavy": FakeEngine("fake-heavy", kind="method", est_vram_mb=7500, exclusive=True),
    }
    return EngineLoader(specs=TEST_SPECS, engines=engines, budget_mb=budget)


def test_default_loader_fast_detectgpt_is_real_but_missing_weights(monkeypatch):
    """Round-6e: fdg is wired to its real class - failure is now the plain
    install-state rule (no config.json in its weights dir), not a boot stub.
    WEIGHTS_DIRNAME is redirected so the test stays valid even after the
    production base is downloaded on a dev machine."""
    from engines.fast_detectgpt import FastDetectGptEngine

    monkeypatch.setattr(FastDetectGptEngine, "WEIGHTS_DIRNAME", "fdg-test-missing")
    loader = EngineLoader()
    results = loader.detect("some text", ["fast-detectgpt"])
    assert len(results) == 1
    r = results[0]
    assert r.status == "weights_missing"
    assert "no config.json" in r.detail


def test_default_engine_wiring_round7():
    """Every method-type engine with a module routes to the real class
    (round-7: binoculars joined fdg and gltr)."""
    from engines.binoculars import BinocularsEngine
    from engines.fast_detectgpt import FastDetectGptEngine
    from engines.gltr import GltrEngine
    from engines.loader import default_engine
    from engines.registry import spec

    assert isinstance(default_engine(spec("fast-detectgpt")), FastDetectGptEngine)
    assert isinstance(default_engine(spec("binoculars")), BinocularsEngine)
    assert isinstance(default_engine(spec("gltr")), GltrEngine)


def test_stub_message_distinguishes_missing_weights_vs_missing_impl(tmp_path, monkeypatch):
    """Weights present but no implementation -> say THAT (the 28.9GB
    binoculars trap); weights absent -> the normal not-downloaded message."""
    import engines.stub as stub_mod

    monkeypatch.setattr(stub_mod, "_WEIGHTS_ROOT", tmp_path)
    stub = stub_mod.StubEngine("binoculars", "en", "method", 7500, exclusive=True)

    # nothing on disk: normal pre-download message
    with pytest.raises(stub_mod.WeightsMissing, match="not downloaded yet"):
        stub.load()

    # weights landed (nested like binoculars' per-repo dests) but no impl class
    landed = tmp_path / "binoculars" / "tiiuae-falcon-7b"
    landed.mkdir(parents=True)
    (landed / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(stub_mod.WeightsMissing, match="implementation not landed"):
        stub.load()
    with pytest.raises(stub_mod.WeightsMissing, match="implementation not landed"):
        stub.detect("some text")


def test_status_lists_full_catalogue():
    loader = EngineLoader()
    status = loader.status()
    assert len(status) == 12
    assert all(e["loaded"] is False for e in status)
    by_id = {e["id"]: e for e in status}
    assert by_id["binoculars"]["exclusive"] is True


def test_unknown_engine_id_isolated():
    loader = make_loader()
    results = loader.detect("t", ["fake-a", "nope"])
    assert results[0].status == "ok"
    assert results[1].status == "unknown_engine"


def test_batch_detect_returns_segments():
    loader = make_loader()
    results = loader.detect("abcd efgh", ["fake-a", "fake-b"])
    assert [r.status for r in results] == ["ok", "ok"]
    assert results[0].segments[0] == Segment(0, 4, 0.9)
    assert results[1].segments[0] == Segment(5, 3, 0.5)


def test_response_never_contains_source_text():
    """The raw text must never leave the backend (PLAN P1 hard rule)."""
    loader = make_loader()
    secret = "UNIQUE-SENTENCE-42"
    results = loader.detect(secret, ["fake-a", "fake-b"])
    wire = json.dumps([r.as_dict() for r in results])
    assert secret not in wire
    assert "abcd" not in json.dumps(make_loader().detect("abcd", ["fake-a"])[0].as_dict())


def test_engine_eviction_via_loader_lru():
    loader = make_loader(budget=200)  # a(100)+b(150) don't fit together
    loader.load(["fake-a"])
    loader.load(["fake-b"])
    assert loader.resident_ids() == ["fake-b"]
    assert loader.used_mb() == 150
    status = {e["id"]: e for e in loader.status()}
    assert status["fake-a"]["loaded"] is False
    assert status["fake-b"]["loaded"] is True


def test_exclusive_heavy_through_loader():
    loader = make_loader(budget=6144)
    loader.load(["fake-a"])
    loader.load(["fake-heavy"])  # evicts everything
    assert loader.resident_ids() == ["fake-heavy"]


def test_unload_via_loader():
    loader = make_loader()
    loader.load(["fake-a"])
    results = loader.unload(["fake-a", "ghost"])
    assert [r.status for r in results] == ["ok", "unknown_engine"]
    assert loader.resident_ids() == []


def test_engines_not_in_catalogue_rejected():
    with pytest.raises(ValueError):
        EngineLoader(specs=TEST_SPECS, engines={"ghost": FakeEngine("ghost")})


def test_default_stub_engine_cannot_load():
    loader = EngineLoader()
    # fast-detectgpt serves as a stub until its P2 base decision (its gpt2
    # family bases showed no curvature discrimination); gltr became real
    # once models/gpt2 landed 2026-11-17.
    results = loader.load(["fast-detectgpt"])
    assert results[0].status == "weights_missing"
    assert loader.resident_ids() == []


# ---- D29: doc_score + streaming ------------------------------------------


def test_doc_score_length_weighted():
    from engines.loader import doc_score

    segs = (Segment(0, 10, 0.8), Segment(10, 30, 0.2))  # 10:30 split
    assert doc_score(segs) == round((0.8 * 10 + 0.2 * 30) / 40, 4)  # 0.35
    assert doc_score((Segment(0, 7, 0.42),)) == 0.42  # doc-level degenerates
    assert doc_score(()) is None


def test_detect_one_carries_doc_score():
    loader = make_loader()  # fake-a: single segment (0,4,0.9) -> doc 0.9
    r = loader.detect_one("abcd", "fake-a")
    assert r.doc_score == 0.9
    assert r.as_dict()["doc_score"] == 0.9


def test_detect_one_streams_batches():
    from engines.base import Segment as S

    loader = EngineLoader(
        specs=TEST_SPECS,
        engines={
            "fake-a": FakeEngine(
                "fake-a",
                batches=[
                    [S(0, 4, 0.9), S(4, 2, 0.1)],
                    [S(6, 3, 0.5)],
                ],
            )
        },
        budget_mb=1000,
    )
    frames: list[tuple[str, list]] = []
    r = loader.detect_one("abcdefghi", "fake-a", on_batch=lambda eid, segs: frames.append((eid, segs)))
    assert r.status == "ok"
    assert len(r.segments) == 3
    assert [len(s) for _, s in frames] == [2, 1]
    assert frames[0][0] == "fake-a"
    # doc_score over all spans: (0.9*4 + 0.1*2 + 0.5*3) / 9
    assert r.doc_score == round((0.9 * 4 + 0.1 * 2 + 0.5 * 3) / 9, 4)


def test_detect_one_without_callback_still_works():
    loader = make_loader()
    r = loader.detect_one("abcd", "fake-a", on_batch=None)
    assert r.status == "ok" and len(r.segments) == 1


# ---- D29 regression: every engine's detect must accept (text, on_batch) ----
# r1-lang-router missed the on_batch parameter when streaming landed and
# blew up with "TypeError: takes 2 positional arguments but 3 were given".


def test_all_engine_detect_signatures_accept_on_batch():
    import inspect

    loader = EngineLoader()
    for sid, engine in loader._engines.items():
        params = [
            p
            for p in inspect.signature(engine.detect).parameters.values()
            if p.name != "self"
        ]
        assert len(params) >= 2, (
            f"{sid}: detect lost the on_batch parameter (params={params})"
        )

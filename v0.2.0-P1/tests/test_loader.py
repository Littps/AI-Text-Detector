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


def test_default_loader_uses_stub_engines():
    loader = EngineLoader()
    results = loader.detect("some text", ["gltr"])
    assert len(results) == 1
    r = results[0]
    assert r.status == "weights_missing"
    assert "P2 downloader" in r.detail


def test_status_lists_full_catalogue():
    loader = EngineLoader()
    status = loader.status()
    assert len(status) == 9
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
    results = loader.load(["simpleai-zh"])
    assert results[0].status == "weights_missing"
    assert loader.resident_ids() == []

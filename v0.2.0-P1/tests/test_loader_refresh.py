"""EngineLoader.refresh_engine: hot-swap after a custom import lands weights.

Boot creates a StubEngine whenever models/<engine_id> is absent. When a
custom import later wires an official engine's file set into that folder,
the server calls refresh_engine so detection picks the real implementation
without a sidecar restart (round-6 user request: import -> usable now).
"""

from unittest.mock import MagicMock

import pytest
from engines import loader as loader_mod
from engines.loader import EngineLoader
from engines.registry import EngineSpec


def _spec(engine_id: str = "simpleai-zh") -> EngineSpec:
    return EngineSpec(
        engine_id=engine_id,
        name=f"{engine_id} display",
        lang="zh",
        kind="classifier",
        est_vram_mb=400,
        repo="example/repo",
        verified=True,
    )


def test_refresh_swaps_stub_for_real_instance(monkeypatch, tmp_path) -> None:
    """before weights land -> stub instance; after -> whatever default_engine builds."""
    stub = MagicMock()
    stub.loaded = False
    real = MagicMock()
    real.loaded = False
    calls: list[str] = []

    def fake_default(spec: EngineSpec) -> MagicMock:
        calls.append(spec.engine_id)
        return real if (tmp_path / "weights-landed").exists() else stub

    monkeypatch.setattr(loader_mod, "default_engine", fake_default)
    loader = EngineLoader(specs=(_spec(),), engines={"simpleai-zh": stub})
    assert loader._engines["simpleai-zh"] is stub

    (tmp_path / "weights-landed").mkdir()
    assert loader.refresh_engine("simpleai-zh") is True
    assert loader._engines["simpleai-zh"] is real
    assert calls == ["simpleai-zh"]


def test_refresh_releases_resident_before_swap(monkeypatch) -> None:
    """a resident engine must be released first so the LRU tracks the new instance."""
    resident = MagicMock()
    resident.loaded = True
    spec = _spec()
    monkeypatch.setattr(loader_mod, "default_engine", lambda s: MagicMock(loaded=False))
    loader = EngineLoader(specs=(spec,), engines={"simpleai-zh": resident})

    released: list[str] = []
    monkeypatch.setattr(loader._lru, "release", lambda eid: released.append(eid) or True)
    loader.refresh_engine("simpleai-zh")
    assert released == ["simpleai-zh"]
    assert loader._engines["simpleai-zh"] is not resident


def test_refresh_unknown_engine_returns_false() -> None:
    loader = EngineLoader(specs=(), engines={})
    assert loader.refresh_engine("not-in-catalogue") is False


def test_refresh_keeps_non_resident_engine_quiet(monkeypatch) -> None:
    """a non-resident engine swaps without touching the LRU."""
    idle = MagicMock()
    idle.loaded = False
    spec = _spec("xiaoyu-zh")
    monkeypatch.setattr(loader_mod, "default_engine", lambda s: MagicMock(loaded=False))
    loader = EngineLoader(specs=(spec,), engines={"xiaoyu-zh": idle})

    released: list[str] = []
    monkeypatch.setattr(loader._lru, "release", lambda eid: released.append(eid) or True)
    assert loader.refresh_engine("xiaoyu-zh") is True
    assert released == []  # never resident -> no release call
    assert loader._engines["xiaoyu-zh"] is not idle


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])

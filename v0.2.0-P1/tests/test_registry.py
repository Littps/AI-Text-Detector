"""Catalogue sanity: the default recommendation set must stay coherent."""
import pytest
from engines.registry import ENGINE_SPECS, all_ids, spec, validate


def test_catalogue_size_tracks_the_recommended_set():
    # 9 since D14 -> 12 after the round-7 v3 trio (zhv3 / zhv3short / env3).
    assert len(ENGINE_SPECS) == 12


def test_unique_ids():
    ids = all_ids()
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("field,valid", [("lang", {"zh", "en", "multi"}), ("kind", {"router", "classifier", "method"})])
def test_enum_fields_valid(field, valid):
    for s in ENGINE_SPECS:
        assert getattr(s, field) in valid, f"{s.engine_id}: bad {field}"


def test_positive_estimates():
    assert all(s.est_vram_mb > 0 for s in ENGINE_SPECS)


def test_binoculars_is_exclusive_and_heavy():
    s = spec("binoculars")
    assert s is not None
    assert s.exclusive is True
    assert s.est_vram_mb >= 7000  # bnb 4bit on purpose (D14)


def test_language_coverage():
    langs = {s.lang for s in ENGINE_SPECS}
    assert "zh" in langs and "en" in langs


def test_validate_passes_on_default_catalogue():
    validate()  # must not raise


def test_spec_lookup():
    assert spec("no-such-engine") is None
    assert spec("gltr") is not None

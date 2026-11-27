"""LRU residency policy under a VRAM budget."""
import time

import pytest
from engines.lru import LruVramManager, VramPolicyError
from fakes import FakeEngine


def make(a_id="a", a_mb=100, b_id="b", b_mb=150, budget=300):
    return (
        LruVramManager(budget_mb=budget),
        FakeEngine(a_id, est_vram_mb=a_mb),
        FakeEngine(b_id, est_vram_mb=b_mb),
    )


def test_both_fit_within_budget():
    lru, a, b = make()
    lru.acquire(a)
    lru.acquire(b)
    assert lru.resident_ids() == ["a", "b"]
    assert lru.used_mb() == 250


def test_evicts_oldest_when_over_budget():
    lru, a, b = make()
    c = FakeEngine("c", est_vram_mb=100)
    lru.acquire(a)
    lru.acquire(b)
    lru.acquire(c)  # 250+100 > 300 -> evict a -> 150+100 fits
    assert lru.resident_ids() == ["b", "c"]
    assert a.loaded is False
    assert a.unload_calls == 1


def test_touch_refreshes_recency():
    lru, a, b = make()
    lru.acquire(a)
    lru.acquire(b)
    lru.acquire(a)  # touch: a becomes most recent
    c = FakeEngine("c", est_vram_mb=100)
    lru.acquire(c)  # evicts b, keeps a
    assert lru.resident_ids() == ["a", "c"]


def test_acquire_is_idempotent():
    lru, a, _ = make()
    lru.acquire(a)
    lru.acquire(a)
    assert a.load_calls == 1
    assert lru.resident_ids() == ["a"]


def test_exclusive_evicts_everything_else():
    lru, a, b = make()
    lru.acquire(a)
    lru.acquire(b)
    heavy = FakeEngine("heavy", est_vram_mb=7500, exclusive=True)
    lru.acquire(heavy)
    assert lru.resident_ids() == ["heavy"]
    assert a.loaded is False and b.loaded is False


def test_exclusive_may_exceed_budget():
    lru = LruVramManager(budget_mb=6144)
    heavy = FakeEngine("heavy", est_vram_mb=7500, exclusive=True)
    lru.acquire(heavy)  # policy: exclusive bypasses the budget
    assert lru.is_resident("heavy")


def test_refuse_non_exclusive_over_budget():
    lru = LruVramManager(budget_mb=1000)
    big = FakeEngine("big", est_vram_mb=2000)
    with pytest.raises(VramPolicyError):
        lru.acquire(big)
    assert lru.resident_ids() == []
    assert big.loaded is False


def test_release():
    lru, a, _ = make()
    lru.acquire(a)
    assert lru.release("a") is True
    assert a.unload_calls == 1
    assert lru.resident_ids() == []
    assert lru.release("a") is False  # already gone
    assert lru.release("never-loaded") is False


def test_evict_all():
    lru, a, b = make()
    lru.acquire(a)
    lru.acquire(b)
    lru.evict_all()
    assert lru.resident_ids() == []
    assert a.unload_calls == 1 and b.unload_calls == 1


def test_invalid_budget():
    with pytest.raises(ValueError):
        LruVramManager(budget_mb=0)


# ---- 2026-11-21 retention rework -----------------------------------------

def test_budget_follows_the_actual_card():
    """A fixed 6GB line was the bug: the budget now scales with the card."""
    lru = LruVramManager(budget_mb=6144, total_vram_mb=8192)
    assert lru.budget_mb == 6553  # 80% headroom policy
    lru.set_total_vram(16384)
    assert lru.budget_mb == 13107


def test_forever_takes_the_card_minus_reserve():
    lru = LruVramManager(total_vram_mb=8192, retention="forever")
    assert lru.budget_mb == 7680
    assert lru.ttl_s is None


def test_unknown_retention_is_rejected():
    lru = LruVramManager()
    with pytest.raises(ValueError):
        lru.set_retention("sometimes")


def test_retention_tiers_carry_their_ttl():
    lru = LruVramManager()
    assert lru.ttl_s is None  # default = keep
    for tier, ttl in (("30m", 1800.0), ("10m", 600.0), ("5m", 300.0), ("none", 0.0)):
        assert lru.set_retention(tier)["ttl_s"] == ttl


def test_language_outsider_evicted_before_recency():
    """User rule: a zh leftover makes room for an English run.

    en is OLDER than zh here, so plain LRU would drop en; the keep_langs hint
    must drop the zh engine instead.
    """
    lru = LruVramManager(budget_mb=300)
    en = FakeEngine("en", lang="en", est_vram_mb=150)
    zh = FakeEngine("zh", lang="zh", est_vram_mb=100)
    lru.acquire(en)
    lru.acquire(zh)  # 250/300
    en2 = FakeEngine("en2", lang="en", est_vram_mb=100)
    lru.acquire(en2, keep_langs={"en"})
    assert lru.resident_ids() == ["en", "en2"]
    assert zh.loaded is False


def test_sweep_idle_releases_only_after_ttl():
    lru = LruVramManager(retention="5m")
    a = FakeEngine("a", est_vram_mb=100)
    base = time.monotonic()
    lru.acquire(a)
    assert lru.sweep_idle(now=base + 299) == []
    assert lru.sweep_idle(now=base + 400) == ["a"]
    assert a.unload_calls == 1


def test_none_tier_is_never_swept_inline():
    """TTL 0 would unload between two engines of the same run - the detect
    handler releases the batch instead, so sweep must be a no-op here."""
    lru = LruVramManager(retention="none")
    lru.acquire(FakeEngine("a", est_vram_mb=100))
    assert lru.sweep_idle(now=time.monotonic() + 10_000) == []


def test_measured_footprint_wins_over_the_estimate():
    lru = LruVramManager(budget_mb=1000)
    a = FakeEngine("a", est_vram_mb=100)
    b = FakeEngine("b", est_vram_mb=150)
    a.actual_vram_mb = 40
    lru.acquire(a)
    lru.acquire(b)
    assert lru.used_mb() == 190
    assert lru.measured() is True
    detail = lru.resident_detail()
    assert detail[0]["actual_mb"] == 40 and detail[1]["actual_mb"] is None


def test_release_clears_the_measured_footprint():
    lru = LruVramManager()
    a = FakeEngine("a", est_vram_mb=100)
    a.actual_vram_mb = 40
    lru.acquire(a)
    lru.release("a")
    assert a.actual_vram_mb is None

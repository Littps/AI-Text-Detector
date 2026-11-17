"""LRU residency policy under a VRAM budget."""
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

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

from app.reliability.circuit_breaker import BreakerState, CircuitBreaker, CircuitOpenError


def test_starts_closed():
    cb = CircuitBreaker()
    assert cb.state == BreakerState.CLOSED


def test_successful_calls_stay_closed():
    cb = CircuitBreaker(failure_threshold=3)
    for _ in range(10):
        result = cb.call(lambda: "ok")
    assert result == "ok"
    assert cb.state == BreakerState.CLOSED


def test_opens_after_threshold_consecutive_failures():
    cb = CircuitBreaker(failure_threshold=3)

    def failing():
        raise RuntimeError("api down")

    for _ in range(3):
        with pytest.raises(RuntimeError):
            cb.call(failing)

    assert cb.state == BreakerState.OPEN


def test_stays_closed_below_threshold():
    cb = CircuitBreaker(failure_threshold=3)

    def failing():
        raise RuntimeError("api down")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            cb.call(failing)

    assert cb.state == BreakerState.CLOSED


def test_open_breaker_fails_fast_without_calling_fn():
    """The core guarantee: once open, the underlying function is never
    even attempted - no wasted API calls to a service that's failing."""
    cb = CircuitBreaker(failure_threshold=1)
    call_count = {"n": 0}

    def failing():
        call_count["n"] += 1
        raise RuntimeError("api down")

    with pytest.raises(RuntimeError):
        cb.call(failing)
    assert call_count["n"] == 1

    # breaker is now open - this call should fail fast, WITHOUT
    # incrementing call_count again
    with pytest.raises(CircuitOpenError):
        cb.call(failing)
    assert call_count["n"] == 1


def test_a_single_success_resets_consecutive_failure_count():
    cb = CircuitBreaker(failure_threshold=3)

    def failing():
        raise RuntimeError("fail")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            cb.call(failing)
    assert cb.state == BreakerState.CLOSED

    cb.call(lambda: "ok")  # success resets the counter

    for _ in range(2):
        with pytest.raises(RuntimeError):
            cb.call(failing)
    # only 2 consecutive failures since the reset - still below threshold of 3
    assert cb.state == BreakerState.CLOSED


def test_transitions_to_half_open_after_cooldown():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=0.05)

    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
    assert cb.state == BreakerState.OPEN

    time.sleep(0.1)
    assert cb.state == BreakerState.HALF_OPEN


def test_half_open_success_closes_breaker():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=0.05)

    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
    time.sleep(0.1)
    assert cb.state == BreakerState.HALF_OPEN

    result = cb.call(lambda: "recovered")

    assert result == "recovered"
    assert cb.state == BreakerState.CLOSED


def test_half_open_failure_reopens_breaker():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=0.05)

    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
    time.sleep(0.1)
    assert cb.state == BreakerState.HALF_OPEN

    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("still down")))

    assert cb.state == BreakerState.OPEN


def test_reset_forces_closed():
    cb = CircuitBreaker(failure_threshold=1)
    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
    assert cb.state == BreakerState.OPEN

    cb.reset()

    assert cb.state == BreakerState.CLOSED


def test_call_passes_through_args_and_kwargs():
    cb = CircuitBreaker()
    result = cb.call(lambda a, b, c=None: (a, b, c), 1, 2, c=3)
    assert result == (1, 2, 3)

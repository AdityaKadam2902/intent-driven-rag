"""
Circuit breaker for LLM API calls - Section 9 of the architecture docs.

CRITICAL DESIGN CONSTRAINT: this breaker's state lives entirely
in-process (plain Python instance attributes), never in Redis or any
shared cache. This is deliberate, not an oversight - if the breaker's
memory lived in the same place as the cache, a cache outage would also
break the safety mechanism meant to protect against API outages. Two
systems failing together instead of one being a backup for the other
is exactly the failure mode this design avoids. See the architecture
doc's "closing gap #7" callout for the full reasoning.

Three states:
  CLOSED     - normal operation, calls go through
  OPEN       - failures crossed the threshold, calls fail fast without
               even attempting the API, until the cooldown expires
  HALF_OPEN  - cooldown expired, ONE call is allowed through as a probe;
               success closes the breaker again, failure re-opens it
"""
import time
from dataclasses import dataclass, field
from enum import Enum


class BreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is attempted while the breaker is OPEN -
    callers should catch this and return a graceful fallback, never
    let it surface as a raw error to the user."""
    pass


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    recovery_timeout_seconds: float = 30.0

    _state: BreakerState = field(default=BreakerState.CLOSED, init=False)
    _consecutive_failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)

    @property
    def state(self) -> BreakerState:
        # If we're OPEN and the cooldown has elapsed, transition to
        # HALF_OPEN lazily - checked here so every caller sees a
        # consistent, up-to-date state without a background thread.
        if self._state == BreakerState.OPEN and self._opened_at is not None:
            if time.monotonic() - self._opened_at >= self.recovery_timeout_seconds:
                self._state = BreakerState.HALF_OPEN
        return self._state

    def call(self, fn, *args, **kwargs):
        """Runs fn(*args, **kwargs) through the breaker. Raises
        CircuitOpenError immediately (without calling fn at all) if
        the breaker is OPEN. In HALF_OPEN, allows exactly one call
        through as a probe."""
        current_state = self.state

        if current_state == BreakerState.OPEN:
            raise CircuitOpenError(
                f"Circuit is open - {self._consecutive_failures} consecutive "
                f"failures, cooling down for "
                f"{self.recovery_timeout_seconds}s before retrying."
            )

        try:
            result = fn(*args, **kwargs)
        except Exception:
            self._record_failure()
            raise
        else:
            self._record_success()
            return result

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._state = BreakerState.OPEN
            self._opened_at = time.monotonic()

    def _record_success(self):
        self._consecutive_failures = 0
        self._state = BreakerState.CLOSED
        self._opened_at = None

    def reset(self):
        """Manually force back to CLOSED - mainly for tests, but also
        a reasonable operational escape hatch if needed."""
        self._state = BreakerState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = None

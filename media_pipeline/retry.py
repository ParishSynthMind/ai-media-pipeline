"""Retry provider calls on transient errors with exponential backoff and jitter."""
from __future__ import annotations

import logging
import random
import time
from typing import Callable, List, Optional, TypeVar

from .providers.base import ProviderError

log = logging.getLogger(__name__)
T = TypeVar("T")


def backoff_delays(retries: int, base: float = 1.0, cap: float = 30.0,
                   rng: Optional[random.Random] = None) -> List[float]:
    rng = rng or random.Random()
    return [min(cap, base * (2 ** i)) + rng.uniform(0, base / 2) for i in range(retries)]


def call_with_retries(fn: Callable[[], T], retries: int = 3, base_delay: float = 1.0,
                      sleep: Callable[[float], None] = time.sleep,
                      before_retry: Optional[Callable[[int, ProviderError], None]] = None) -> T:
    """Call fn(); on a retryable ProviderError wait and try again, up to `retries` more times.

    `before_retry(n, error)` runs before each retry and may raise to stop early
    (the pipeline uses it to re-check the cost cap before every paid call).
    """
    delays = backoff_delays(retries, base_delay)
    for n in range(retries + 1):
        try:
            return fn()
        except ProviderError as e:
            if not e.retryable or n == retries:
                raise
            log.warning("provider error (%s); retry %d/%d in %.1fs", e, n + 1, retries, delays[n])
            if before_retry:
                before_retry(n + 1, e)
            sleep(delays[n])
    raise AssertionError("unreachable")

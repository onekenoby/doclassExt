"""
Tiny token-bucket rate-gate: smooths bursts so we stay inside the
Gemini per-second quota.  One gate is shared by all threads.
"""

from __future__ import annotations
import threading
import time


class RateGate:
    def __init__(self, rate_per_sec: float = 1.0):
        """
        rate_per_sec – average number of requests allowed per second.
        Free tier ≈ 2 req/s; raise if your paid quota is higher.
        """
        self.rate = rate_per_sec
        self.tokens = rate_per_sec         # start full
        self.last = time.time()
        self.lock = threading.Lock()

    def wait(self) -> None:
        "Block until one token is available, then consume it."
        with self.lock:
            now = time.time()
            elapsed = now - self.last
            self.last = now
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            if self.tokens < 1:
                sleep_for = (1 - self.tokens) / self.rate
                time.sleep(sleep_for)
                self.tokens = 0
            else:
                self.tokens -= 1

"""Hard run budgets.

An autonomous agent that can call an LLM and hit the web in a loop *will* run
away if you let it. Every tool call passes through :class:`Budget`, which raises
before the call that would breach a limit rather than after.
"""

from __future__ import annotations

import threading
import time

from .config import Settings
from .errors import BudgetExceeded
from .models import RunMetrics


class Budget:
    def __init__(
        self,
        settings: Settings,
        *,
        initial: RunMetrics | None = None,
        clock=time.monotonic,
    ) -> None:
        self.settings = settings
        self._clock = clock
        self._started = clock()
        # Research nodes for different sub-questions run concurrently, so every
        # read-modify-write here is guarded.
        self._lock = threading.Lock()
        self.llm_calls = (initial.llm_calls if initial else 0)
        self.searches = (initial.searches if initial else 0)
        self.tokens_in = (initial.tokens_in if initial else 0)
        self.tokens_out = (initial.tokens_out if initial else 0)
        self.cost_usd = (initial.cost_usd if initial else 0.0)
        # Headroom granted so a report can still be written after research is
        # cut short — a partial report beats no report.
        self._extra_llm_calls = 0
        self._extra_cost_usd = 0.0
        self._extra_seconds = 0.0

    # -- read ------------------------------------------------------------
    @property
    def elapsed_s(self) -> float:
        return self._clock() - self._started

    def remaining(self) -> dict[str, float]:
        return {
            "llm_calls": max(0, self.settings.max_llm_calls - self.llm_calls),
            "searches": max(0, self.settings.max_searches - self.searches),
            "cost_usd": round(max(0.0, self.settings.max_cost_usd - self.cost_usd), 6),
            "wall_clock_s": round(max(0.0, self.settings.max_wall_clock_s - self.elapsed_s), 2),
        }

    def snapshot(self) -> RunMetrics:
        return RunMetrics(
            llm_calls=self.llm_calls,
            searches=self.searches,
            tokens_in=self.tokens_in,
            tokens_out=self.tokens_out,
            cost_usd=round(self.cost_usd, 6),
            duration_ms=round(self.elapsed_s * 1000, 2),
        )

    # -- enforcement -----------------------------------------------------
    def reserve_for_finalization(
        self, llm_calls: int = 3, cost_usd: float = 0.50, seconds: float = 120.0
    ) -> None:
        """Grant one-off headroom so synthesis can complete after research is cut short.

        Without this, exhausting the research budget would also make it
        impossible to write up what was found — the least useful possible
        outcome of hitting a limit.
        """
        with self._lock:
            self._extra_llm_calls += llm_calls
            self._extra_cost_usd += cost_usd
            self._extra_seconds += seconds

    def check(self, *, llm: bool = False, search: bool = False) -> None:
        """Raise if the next call would exceed a limit.

        Called *before* the call it guards, so a breaching call never happens.
        """
        max_llm_calls = self.settings.max_llm_calls + self._extra_llm_calls
        max_cost_usd = self.settings.max_cost_usd + self._extra_cost_usd
        max_wall_clock = self.settings.max_wall_clock_s + self._extra_seconds

        with self._lock:
            if llm and self.llm_calls >= max_llm_calls:
                raise BudgetExceeded(f"LLM call budget reached ({max_llm_calls})")
            if search and self.searches >= self.settings.max_searches:
                raise BudgetExceeded(f"Search budget reached ({self.settings.max_searches})")
            if self.cost_usd >= max_cost_usd:
                raise BudgetExceeded(
                    f"Cost budget reached (${self.cost_usd:.4f} >= ${max_cost_usd:.2f})"
                )
        if self.elapsed_s >= max_wall_clock:
            raise BudgetExceeded(
                f"Wall-clock budget reached ({self.elapsed_s:.0f}s >= {max_wall_clock:.0f}s)"
            )

    # -- recording -------------------------------------------------------
    def record_llm(self, *, tokens_in: int, tokens_out: int, cost_usd: float) -> None:
        with self._lock:
            self.llm_calls += 1
            self.tokens_in += tokens_in
            self.tokens_out += tokens_out
            self.cost_usd += cost_usd

    def record_search(self, count: int = 1) -> None:
        with self._lock:
            self.searches += count

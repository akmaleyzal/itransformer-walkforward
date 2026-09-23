"""Per-origin window accounting, checked against a table fixed before any run.

The committed table holds what the input data must yield per origin. This module
measures the same quantities from the artifact, and the budget test asserts exact
equality per origin: a pooled comparison would hide positional drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from itransformer_btc.config import (
    ORIGINS,
    PRED_LEN,
    SEQ_LEN,
    STARTS_LOST_PER_BREAK,
    TEST_BLOCKS,
    WINDOW_SPAN,
    Origin,
)
from itransformer_btc.segments import (
    HOUR_MS,
    BreakSummary,
    break_summary,
    build_segments,
)
from itransformer_btc.windows import count_windows

#: ``(break_runs, excluded_positions, windows_kept)`` per origin's 21-month training
#: sub-block, measured from the input data before any model ran. Pinned rather
#: than recomputed, so the test catches drift instead of agreeing with itself.
COMMITTED_TRAIN_BUDGET: Final[dict[str, tuple[int, int, int]]] = {
    "2020-01": (11, 87, 13_934),
    "2020-06": (13, 63, 13_701),
    "2020-11": (12, 48, 13_741),
    "2021-04": (13, 41, 13_716),
    "2021-09": (14, 30, 13_560),
    "2022-02": (14, 32, 13_558),
    "2022-07": (9, 20, 14_165),
    "2022-12": (8, 19, 14_285),
    "2023-05": (2, 6, 15_021),
    "2023-10": (1, 2, 15_072),
    "2024-03": (1, 2, 15_120),
    "2024-08": (1, 2, 15_096),
    "2025-01": (1, 2, 15_096),
    "2025-06": (0, 0, 15_217),
    "2025-11": (0, 0, 15_217),
}


@dataclass(frozen=True, slots=True)
class OriginBudget:
    """Measured window accounting for one origin's training sub-block."""

    origin: Origin
    summary: BreakSummary
    windows_measured: int
    windows_closed_form: int
    test_block_starts: tuple[int, ...]

    @property
    def label(self) -> str:
        return self.origin.label

    @property
    def loss_pct(self) -> float:
        """Window starts lost to breaks, as a percentage of a gap-free span."""
        ceiling = self.summary.calendar_hours - STARTS_LOST_PER_BREAK
        return 100.0 * (1.0 - self.windows_measured / ceiling) if ceiling else 0.0

    @property
    def closed_form_agrees(self) -> bool:
        """Whether the closed-form count matches the segment-wise count.

        A disagreement means a segment is shorter than one window.
        """
        return self.windows_measured == self.windows_closed_form


def surviving_block_starts(frame: pl.DataFrame, origin: Origin, b: int) -> int:
    """Window starts surviving inside test block ``b``, out of 720.

    Every hour of a test block is an admissible forecast origin, because a lookback
    that reaches back across the block boundary is information a forecaster has.
    An hour is lost only when a break falls inside the 120 bars its window spans.
    """
    lo, hi = origin.block(b)
    lo_ms = int(lo.timestamp() * 1000)
    hi_ms = int(hi.timestamp() * 1000)
    span_ms = (WINDOW_SPAN - 1) * HOUR_MS

    usable = set(
        frame.filter(pl.col("usable")).get_column("ts_ms").to_list()
    )
    survivors = 0
    for start in range(lo_ms, hi_ms, HOUR_MS):
        # The window is contiguous exactly when every hour it spans is usable.
        if all((start + (k - SEQ_LEN) * HOUR_MS) in usable for k in range(WINDOW_SPAN)):
            survivors += 1
    return survivors


def origin_budget(frame: pl.DataFrame, origin: Origin) -> OriginBudget:
    """Measure one origin's 21-month training sub-block and its six test blocks."""
    summary = break_summary(frame, origin.train_start, origin.train_sub_end)
    segments = build_segments(frame, origin.train_start, origin.train_sub_end)

    ceiling = summary.calendar_hours - STARTS_LOST_PER_BREAK
    blocks = [surviving_block_starts(frame, origin, b) for b in range(1, TEST_BLOCKS + 1)]

    return OriginBudget(
        origin=origin,
        summary=summary,
        windows_measured=count_windows(segments, SEQ_LEN, PRED_LEN),
        windows_closed_form=ceiling - summary.window_starts_lost,
        test_block_starts=tuple(blocks),
    )


def budget_table(frame: pl.DataFrame) -> list[OriginBudget]:
    """Measure every origin in the grid."""
    return [origin_budget(frame, origin) for origin in ORIGINS]


def format_markdown(budgets: list[OriginBudget]) -> str:
    """Render the measured table in the layout of ``docs/ORIGIN_WINDOW_BUDGET.md``."""
    head = (
        "| # | Origin | Training sub-block | Breaks | Excluded | Windows kept "
        "| Loss | Test-block starts B1…B6 |\n"
        "|---:|---|---|---:|---:|---:|---:|---|\n"
    )
    rows = [
        f"| {b.origin.index:>2} | {b.origin.origin:%Y-%m-%d} "
        f"| {b.origin.train_start:%Y-%m-%d} → {b.origin.train_sub_end:%Y-%m-%d} "
        f"| {b.summary.break_runs} | {b.summary.excluded_positions} "
        f"| {b.windows_measured:,} | {b.loss_pct:.1f}% "
        f"| {' / '.join(str(n) for n in b.test_block_starts)} |"
        for b in budgets
    ]
    return head + "\n".join(rows) + "\n"

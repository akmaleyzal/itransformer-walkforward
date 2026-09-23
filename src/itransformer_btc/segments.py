"""The segment law: what breaks the series, and where.

A segment is a maximal run of contiguous, usable hourly bars. The series breaks
at every missing bar, because no price formed and imputation is undefined, and
at every zero-volume or ``high == low`` bar, which carries no trade information
either. Returns and windows are computed inside segments only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import polars as pl

from itransformer_btc.config import DATA_END, DATA_START, STARTS_LOST_PER_BREAK

HOUR_MS: Final = 3_600_000

DEFAULT_PARQUET: Final = Path("data/raw/BTCUSDT_1h.parquet")


@dataclass(frozen=True, slots=True)
class Segment:
    """A maximal run of contiguous usable bars, as half-open row indices.

    Attributes:
        start_row: First row index into the usable frame, inclusive.
        end_row: One past the last row index, exclusive.
        start_ts: Epoch ms of the first bar.
        end_ts: Epoch ms of the last bar, inclusive.
    """

    start_row: int
    end_row: int
    start_ts: int
    end_ts: int

    @property
    def n_bars(self) -> int:
        return self.end_row - self.start_row

    def window_starts(self, span: int) -> int:
        """How many ``span``-bar windows start inside this segment; never negative."""
        return max(0, self.n_bars - span + 1)


def load_bars(path: Path | str = DEFAULT_PARQUET) -> pl.DataFrame:
    """Load the immutable input parquet, sorted, with an epoch-ms ``ts_ms`` column.

    Raises:
        FileNotFoundError: If the file is absent.
        ValueError: If the frame is empty, has duplicate timestamps, or reaches
            outside the half-open data window.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The Stage 1 artifacts live in data/raw/; "
            f"regenerate with "
            f"`python spot_klines_btc.py --rebuild-only --outdir ./data/raw`."
        )

    frame = (
        pl.read_parquet(path)
        .with_columns(pl.col("open_time").dt.epoch("ms").alias("ts_ms"))
        .sort("ts_ms")
    )

    if frame.height == 0:
        raise ValueError(f"{path} is empty")

    n_unique = frame.select(pl.col("ts_ms").n_unique()).item()
    if n_unique != frame.height:
        raise ValueError(
            f"{path} carries {frame.height - n_unique} duplicate timestamps; "
            f"Stage 1 de-duplicates, so this is not Stage 1 output"
        )

    lo, hi = frame.select(
        pl.col("ts_ms").min().alias("lo"), pl.col("ts_ms").max().alias("hi")
    ).row(0)
    window_lo = int(DATA_START.timestamp() * 1000)
    window_hi = int(DATA_END.timestamp() * 1000)
    if lo < window_lo or hi >= window_hi:
        raise ValueError(
            f"{path} reaches outside the half-open data window "
            f"[{DATA_START.isoformat()}, {DATA_END.isoformat()}): "
            f"first={lo} last={hi}. Re-emit with --rebuild-only, which "
            f"applies clip_to_window()."
        )
    return frame


def usable_mask(frame: pl.DataFrame) -> pl.DataFrame:
    """Flag each bar usable or not.

    A bar is unusable when it has zero volume or ``high == low``. ``zero_trades`` is
    recorded for the data audit but does not by itself exclude a bar.

    Returns:
        The input plus boolean ``zero_volume``, ``flat_bar``, ``zero_trades`` and
        ``usable``.
    """
    return frame.with_columns(
        (pl.col("volume") <= 0).alias("zero_volume"),
        (pl.col("high") <= pl.col("low")).alias("flat_bar"),
        (pl.col("trades") <= 0).alias("zero_trades"),
    ).with_columns(
        (~pl.col("zero_volume") & ~pl.col("flat_bar")).alias("usable")
    )


def build_segments(
    frame: pl.DataFrame,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[Segment]:
    """Split the usable bars of ``[start, end)`` into contiguous segments.

    A new segment starts wherever the previous usable bar is not exactly one hour
    earlier, which covers missing and excluded bars alike.
    """
    if "usable" not in frame.columns:
        frame = usable_mask(frame)

    span = frame.filter(pl.col("usable"))
    if start is not None:
        span = span.filter(pl.col("ts_ms") >= int(start.timestamp() * 1000))
    if end is not None:
        span = span.filter(pl.col("ts_ms") < int(end.timestamp() * 1000))
    if span.height == 0:
        return []

    ts = span.get_column("ts_ms").to_list()
    segments: list[Segment] = []
    seg_start = 0
    for i in range(1, len(ts)):
        if ts[i] - ts[i - 1] != HOUR_MS:
            segments.append(Segment(seg_start, i, ts[seg_start], ts[i - 1]))
            seg_start = i
    segments.append(Segment(seg_start, len(ts), ts[seg_start], ts[-1]))
    return segments


@dataclass(frozen=True, slots=True)
class BreakSummary:
    """Measured break profile of a span; every field is counted."""

    calendar_hours: int
    bars_present: int
    bars_usable: int
    missing_bars: int
    zero_volume_bars: int
    flat_bars: int
    zero_trade_bars: int
    excluded_positions: int
    break_runs: int

    @property
    def segments(self) -> int:
        """Segments the span splits into."""
        return max(1, self.break_runs + 1)

    @property
    def window_starts_lost(self) -> int:
        """``119 x break_runs + excluded_positions`` window starts lost to breaks."""
        return STARTS_LOST_PER_BREAK * self.break_runs + self.excluded_positions


def break_summary(
    frame: pl.DataFrame,
    start: datetime,
    end: datetime,
) -> BreakSummary:
    """Measure every break-inducing condition in ``[start, end)``.

    A break run is a maximal stretch of excluded calendar hours, missing or
    unusable. Each run costs 119 window starts, so adjacent exclusions are counted
    once.
    """
    if "usable" not in frame.columns:
        frame = usable_mask(frame)

    lo = int(start.timestamp() * 1000)
    hi = int(end.timestamp() * 1000)
    span = frame.filter((pl.col("ts_ms") >= lo) & (pl.col("ts_ms") < hi))

    calendar_hours = (hi - lo) // HOUR_MS
    usable = int(span.select(pl.col("usable").sum()).item())
    counts = span.select(
        pl.col("zero_volume").sum().alias("zv"),
        pl.col("flat_bar").sum().alias("fb"),
        pl.col("zero_trades").sum().alias("zt"),
    ).row(0)

    # Walk the calendar, not the rows: a missing bar has no row to inspect, and
    # a run mixing missing with unusable positions must count once.
    usable_ts = set(span.filter(pl.col("usable")).get_column("ts_ms").to_list())
    break_runs = 0
    in_run = False
    for t in range(lo, hi, HOUR_MS):
        if t in usable_ts:
            in_run = False
        else:
            if not in_run:
                break_runs += 1
            in_run = True

    return BreakSummary(
        calendar_hours=calendar_hours,
        bars_present=span.height,
        bars_usable=usable,
        missing_bars=calendar_hours - span.height,
        zero_volume_bars=int(counts[0]),
        flat_bars=int(counts[1]),
        zero_trade_bars=int(counts[2]),
        excluded_positions=calendar_hours - usable,
        break_runs=break_runs,
    )

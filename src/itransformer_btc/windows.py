"""Window enumeration, validated by timestamp rather than by position.

A window ``[s, s+L+H)`` is valid only if ``t[s+L+H-1] - t[s] == (L+H-1)`` hours.
Positional sliding after a row drop would close a gap silently, so every emitted
window is checked. Enumerating only windows that lie wholly inside a span also
applies the purge: the last target ends at the span boundary.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from itransformer_btc.config import PRED_LEN, SEQ_LEN
from itransformer_btc.segments import HOUR_MS, Segment, build_segments, usable_mask


def enumerate_windows(
    frame: pl.DataFrame,
    start: datetime | None = None,
    end: datetime | None = None,
    seq_len: int = SEQ_LEN,
    pred_len: int = PRED_LEN,
) -> list[int]:
    """Every valid window start inside ``[start, end)``, as epoch ms.

    Windows are built inside segments and never across a break.

    Args:
        frame: Bars carrying ``ts_ms``; ``usable`` is derived if absent.
        start: Inclusive lower bound of the span.
        end: Exclusive upper bound of the span.
        seq_len: Lookback ``L``.
        pred_len: Horizon ``H``.

    Returns:
        Window-start timestamps in ascending order.

    Raises:
        ValueError: If an emitted window fails the timestamp identity, which means
            the segment builder is wrong.
    """
    span = seq_len + pred_len
    segments = build_segments(frame, start, end)
    if not segments:
        return []

    rows = (frame if "usable" in frame.columns else usable_mask(frame)).filter(
        pl.col("usable")
    )
    if start is not None:
        rows = rows.filter(pl.col("ts_ms") >= int(start.timestamp() * 1000))
    if end is not None:
        rows = rows.filter(pl.col("ts_ms") < int(end.timestamp() * 1000))
    ts = rows.get_column("ts_ms").to_list()

    starts: list[int] = []
    for seg in segments:
        for s in range(seg.start_row, seg.end_row - span + 1):
            last = s + span - 1
            if ts[last] - ts[s] != (span - 1) * HOUR_MS:
                raise ValueError(
                    f"window at ts={ts[s]} spans a break: "
                    f"t[{last}] - t[{s}] = {ts[last] - ts[s]} ms, expected "
                    f"{(span - 1) * HOUR_MS} ms. The segment builder is wrong; "
                    f"do not relax this check."
                )
            starts.append(ts[s])
    return starts


def count_windows(
    segments: list[Segment],
    seq_len: int = SEQ_LEN,
    pred_len: int = PRED_LEN,
) -> int:
    """Total window starts across segments, ``max(0, n - span + 1)`` each.

    The closed form ``(bars - 119) - (119 x breaks + missing)`` agrees only while
    every segment is longer than one window; this count is exact either way.
    """
    span = seq_len + pred_len
    return sum(seg.window_starts(span) for seg in segments)

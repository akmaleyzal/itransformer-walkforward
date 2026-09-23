"""Effective dimensionality (K_eff), measured before any model trains.

K_eff is RQ1's regressor. Every value that enters inference is computed per
origin on that origin's 21-month training sub-block, so the regressor never
reads the test period. Four variants are reported side by side, because the
model consumes a K x 96 block rather than one bar: the contemporaneous PR, the
PR after per-window normalisation (what ``use_norm`` feeds the embedding), the
stable rank of each window block, and the PR of the lookback correlation.

Upstream:
    Written here on numpy, after L. Laloux, P. Cizeau, J.-P. Bouchaud and
    M. Potters, Phys. Rev. Lett. 83(7), 1999, and V. Plerou et al., Phys. Rev. E
    65(6), 066126, 2002. PR is taken on correlation matrices, never covariance,
    because the variates do not share units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl

from itransformer_btc.config import (
    DATA_START,
    FIRST_ORIGIN,
    K_LADDER,
    ORIGINS,
    SEQ_LEN,
    WINDOW_SPAN,
    OriginLike,
)
from itransformer_btc.features import ladder_columns
from itransformer_btc.segments import HOUR_MS
from itransformer_btc.splits import window_starts

#: The gate's floor, fixed before measuring anything.
GATE_PR_FLOOR: float = 5.0

#: Windows sampled for the lookback measures, by a fixed stride.
LOOKBACK_SAMPLE: int = 2_000


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """``PR = (sum lambda)^2 / sum lambda^2``, in ``[1, K]``.

    Negative eigenvalues from rounding are clipped to zero, not dropped.

    Raises:
        ValueError: If the spectrum sums to zero.
    """
    lam = np.clip(np.asarray(eigenvalues, dtype=np.float64), 0.0, None)
    total = lam.sum()
    if total <= 0:
        raise ValueError("degenerate spectrum: eigenvalues sum to zero")
    return float(total * total / np.square(lam).sum())


def contemporaneous_pr(values: np.ndarray) -> float:
    """PR of the ``K x K`` correlation matrix of ``(n, K)`` bar-level values."""
    corr = np.atleast_2d(
        np.corrcoef(np.asarray(values, dtype=np.float64), rowvar=False)
    )
    return participation_ratio(np.linalg.eigvalsh(corr))


def window_normalised_pr(windows: np.ndarray) -> float:
    """PR after standardising each channel within its window, as ``use_norm`` does.

    A gap between this and the raw PR at K=12 would mean the normalisation, not the
    data, drives the apparent redundancy of the volatility family.
    """
    x = np.asarray(windows, dtype=np.float64)
    mean = x.mean(axis=1, keepdims=True)
    std = np.sqrt(x.var(axis=1, keepdims=True) + 1e-12)
    return contemporaneous_pr(((x - mean) / std).reshape(-1, x.shape[2]))


def stable_rank(matrix: np.ndarray) -> float:
    """``||M||_F^2 / ||M||_2^2``, in ``[1, min(rows, cols)]``."""
    singular = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    if singular[0] <= 0:
        raise ValueError("degenerate window block: largest singular value is 0")
    return float(np.square(singular).sum() / (singular[0] ** 2))


def lookback_stable_rank(windows: np.ndarray, sample: int = LOOKBACK_SAMPLE) -> float:
    """Mean stable rank of each window's ``K x L`` block, standardised within the window.

    Windows are sampled by a fixed stride, so the value is deterministic. After
    standardisation the stable rank equals ``K / lambda_1`` of the within-window
    correlation matrix, so it is on the same ``[1, K]`` scale as the PR.
    """
    x = np.asarray(windows, dtype=np.float64)
    if len(x) == 0:
        raise ValueError("no windows to measure")
    stride = max(1, len(x) // sample)
    blocks = np.transpose(x[::stride][:sample], (0, 2, 1))   # (m, K, L)
    blocks = blocks - blocks.mean(axis=2, keepdims=True)
    blocks = blocks / np.sqrt(np.square(blocks).mean(axis=2, keepdims=True) + 1e-12)
    return float(np.mean([stable_rank(b) for b in blocks]))


def lookback_correlation_pr(windows: np.ndarray) -> float:
    """PR of the ``K*L x K*L`` correlation spectrum of the flattened windows.

    The only variant that sees cross-lag structure. Its ceiling is ``K*L``, so rungs
    are compared through ``KeffRow.pr_lookback_ratio``.
    """
    x = np.asarray(windows, dtype=np.float64)
    flat = x.reshape(len(x), -1)
    flat = flat - flat.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.square(flat).mean(axis=0, keepdims=True) + 1e-24)
    flat = flat / scale
    gram = (flat.T @ flat) / max(1, len(flat) - 1)
    return participation_ratio(np.linalg.eigvalsh(gram))


@dataclass(frozen=True, slots=True)
class KeffRow:
    """One (origin, rung) cell of Table 2b."""

    origin: str
    origin_index: int
    k: int
    n_rows: int
    n_windows: int
    pr_raw: float
    pr_window_norm: float
    stable_rank_lookback: float
    pr_lookback_corr: float

    @property
    def pr_lookback_ratio(self) -> float:
        """``pr_lookback_corr / (K * L)``: the cross-lag PR as a share of its ceiling."""
        return self.pr_lookback_corr / (self.k * SEQ_LEN)

    @property
    def divergence(self) -> float:
        """``stable_rank_lookback - pr_raw``: what the lookback adds to the bar-level view."""
        return self.stable_rank_lookback - self.pr_raw


def _training_windows(
    features: pl.DataFrame, origin: OriginLike, k: int, seq_len: int = SEQ_LEN
) -> tuple[np.ndarray, np.ndarray]:
    """Rows and windows of one origin's 21-month training sub-block, the span the scaler sees."""
    columns = ladder_columns(k)
    ts = features.get_column("ts_ms").to_numpy()
    values = features.select(columns).to_numpy()

    lo = int(origin.train_start.timestamp() * 1000)
    hi = int(origin.train_sub_end.timestamp() * 1000)
    rows = values[(ts >= lo) & (ts < hi)]

    starts = window_starts(
        ts, origin.train_start, origin.train_sub_end, "contained", WINDOW_SPAN
    )
    if len(starts) == 0:
        raise ValueError(f"origin {origin.label}: no training window to measure")
    idx = starts[:, None] + np.arange(seq_len)[None, :]
    return rows, values[idx]


def keff_row(features: pl.DataFrame, origin: OriginLike, k: int) -> KeffRow:
    """Every K_eff variant for one (origin, rung) cell."""
    rows, windows = _training_windows(features, origin, k)
    return KeffRow(
        origin=origin.label,
        origin_index=origin.index,
        k=k,
        n_rows=len(rows),
        n_windows=len(windows),
        pr_raw=contemporaneous_pr(rows),
        pr_window_norm=window_normalised_pr(windows),
        stable_rank_lookback=lookback_stable_rank(windows),
        pr_lookback_corr=lookback_correlation_pr(windows),
    )


def keff_table(
    features: pl.DataFrame,
    origins: list[OriginLike] | None = None,
    rungs: tuple[int, ...] = K_LADDER,
) -> pl.DataFrame:
    """Table 2b: every rung at every origin, on training spans only.

    K=1 stays in although its PR is 1 by definition, so the RQ1 panel is balanced.
    """
    grid = list(origins if origins is not None else ORIGINS)
    return pl.DataFrame(
        [
            {
                "origin": row.origin,
                "origin_index": row.origin_index,
                "k": row.k,
                "n_rows": row.n_rows,
                "n_windows": row.n_windows,
                "pr_raw": row.pr_raw,
                "pr_window_norm": row.pr_window_norm,
                "stable_rank_lookback": row.stable_rank_lookback,
                "pr_lookback_corr": row.pr_lookback_corr,
                "pr_lookback_ratio": row.pr_lookback_ratio,
                "divergence": row.divergence,
            }
            for origin in grid
            for k in rungs
            for row in (keff_row(features, origin, k),)
        ]
    )


def corr_k_keff(table: pl.DataFrame, column: str = "pr_raw") -> float:
    """``corr(K, K_eff)`` across rungs; near 1 means K and K_eff are hard to tell apart."""
    means = table.group_by("k").agg(pl.col(column).mean().alias("keff")).sort("k")
    k = means.get_column("k").to_numpy().astype(np.float64)
    keff = means.get_column("keff").to_numpy()
    return float(np.corrcoef(k, keff)[0, 1])


def gate_pr(features: pl.DataFrame, k: int = 8) -> float:
    """The gate value: PR at K=8 on ``[2018-01, 2020-01)``, a span before every origin."""
    columns = ladder_columns(k)
    ts = features.get_column("ts_ms").to_numpy()
    lo = int(DATA_START.timestamp() * 1000)
    hi = int(FIRST_ORIGIN.timestamp() * 1000)
    rows = features.select(columns).to_numpy()[(ts >= lo) & (ts < hi)]
    if len(rows) == 0:
        raise ValueError("the pre-first-origin span holds no usable bar")
    return contemporaneous_pr(rows)


def gate_verdict(measured: float, floor: float = GATE_PR_FLOOR) -> str:
    """The gate's action. Below the floor the result is disclosed, not re-cut.

    F1-F5 admit only one consistent ladder, so there is no alternative cut to take.
    """
    if measured >= floor:
        return (
            f"PASS: measured PR at K=8 is {measured:.3f} >= {floor:.1f}. "
            f"Proceed; report the value in Table 2b."
        )
    return (
        f"DISCLOSE: measured PR at K=8 is {measured:.3f} < {floor:.1f}. "
        f"Proceed unchanged and disclose the value; the ladder is not re-cut, "
        f"because F1-F5 leave no second consistent cut."
    )


#: Rolling window: long enough for an 8 x 8 correlation, short enough to see regimes.
ROLLING_WINDOW_DAYS: Final = 90

#: One day between consecutive rolling windows.
ROLLING_STEP_DAYS: Final = 1


def _rolling_spans(
    ts: np.ndarray, window_days: int, step_days: int
) -> list[tuple[int, int, int]]:
    """``(window_end_ms, lo, hi)`` per window, sliced by time rather than by position.

    Position slicing would stretch a window over gaps, and gaps cluster early in the
    sample.
    """
    window_ms = window_days * 24 * HOUR_MS
    step_ms = step_days * 24 * HOUR_MS
    spans: list[tuple[int, int, int]] = []
    for end in range(int(ts[0]) + window_ms, int(ts[-1]) + 1, step_ms):
        lo = int(np.searchsorted(ts, end - window_ms, side="left"))
        hi = int(np.searchsorted(ts, end, side="left"))
        spans.append((end, lo, hi))
    return spans


def rolling_pr(
    features: pl.DataFrame,
    k: int = 8,
    window_days: int = ROLLING_WINDOW_DAYS,
    step_days: int = ROLLING_STEP_DAYS,
) -> pl.DataFrame:
    """Rolling PR over the full sample (Figure 2b), descriptive only.

    It reads the test period, so it never enters a regression or the gate.

    Returns:
        ``window_end_ms, n_rows, pr``, one row per window.
    """
    columns = ladder_columns(k)
    ts = features.get_column("ts_ms").to_numpy()
    values = features.select(columns).to_numpy()

    rows = []
    for end, lo, hi in _rolling_spans(ts, window_days, step_days):
        block = values[lo:hi]
        if len(block) <= len(columns):
            continue        # a correlation needs more rows than columns
        rows.append(
            {"window_end_ms": end, "n_rows": len(block), "pr": contemporaneous_pr(block)}
        )
    return pl.DataFrame(rows)


def rolling_ols_r2(
    features: pl.DataFrame,
    k: int = 8,
    window_days: int = ROLLING_WINDOW_DAYS,
    step_days: int = ROLLING_STEP_DAYS,
) -> pl.DataFrame:
    """In-window R^2 of ``r_{t+1}`` on the K features at t (Figure 2b), descriptive only.

    It asks whether the feature-return relation moves over time, not whether it
    forecasts. Pairs that straddle a gap are dropped.

    Returns:
        ``window_end_ms, n_pairs, r2``, one row per window.
    """
    columns = ladder_columns(k)
    ts = features.get_column("ts_ms").to_numpy()
    values = features.select(columns).to_numpy()
    target = features.get_column("r").to_numpy()

    contiguous = np.zeros(len(ts), dtype=bool)
    contiguous[:-1] = np.diff(ts) == HOUR_MS

    rows = []
    for end, lo, hi in _rolling_spans(ts, window_days, step_days):
        # Clamped so the last window cannot ask for a successor that does not
        # exist; the mask and both slices then have one length by construction
        # rather than by a length check that fires after an IndexError would.
        n = min(hi, len(ts) - 1) - lo
        if n <= 0:
            continue
        keep = contiguous[lo : lo + n]
        x = values[lo : lo + n][keep]
        y = target[lo + 1 : lo + 1 + n][keep]
        if len(x) <= len(columns) + 1:
            continue
        design = np.column_stack([np.ones(len(x)), x])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ coefficients
        total = float(((y - y.mean()) ** 2).sum())
        rows.append({
            "window_end_ms": end,
            "n_pairs": len(y),
            "r2": float(1.0 - (residual ** 2).sum() / total) if total > 0 else 0.0,
        })
    return pl.DataFrame(rows)

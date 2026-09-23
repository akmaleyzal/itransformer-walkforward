"""Market-efficiency diagnostics for the data section (Table 2).

Reported for the whole sample and for each origin's 21-month training sub-block,
so the claim that efficiency varies over time is shown rather than assumed. The
test blocks are never read. ``arch`` and ``statsmodels`` take numpy arrays, so
pandas never enters.

Upstream:
    Variance ratio: ``arch.unitroot.VarianceRatio`` (https://github.com/bashtage/arch,
    NCSA), after A. W. Lo and A. C. MacKinlay, Rev. Financial Stud. 1(1), 1988.
    ADF: ``statsmodels.tsa.stattools.adfuller`` (BSD-3-Clause), after D. A. Dickey
    and W. A. Fuller, J. Amer. Statist. Assoc. 74(366), 1979.
    ``hurst_rs``: written here, after H. E. Hurst, Trans. Amer. Soc. Civil Eng.
    116(1), 1951.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl

from itransformer_btc.config import ORIGINS, OriginLike

#: Lags for the Lo-MacKinlay variance ratio. Powers of two spanning two hours to
#: two thirds of a day, which brackets the horizons this study forecasts.
VR_LAGS: Final[tuple[int, ...]] = (2, 4, 8, 16)

#: Smallest R/S block. Below ~16 points the rescaled range is dominated by its
#: own small-sample bias and the log-log slope bends upward on white noise.
HURST_MIN_N: Final = 16

#: Blocks needed for the log-log regression to mean anything. Two points define a
#: line exactly and would report a slope with no residual to doubt it.
HURST_MIN_BLOCK_SIZES: Final = 3

VR_TREND: Final = "c"


@dataclass(frozen=True, slots=True)
class VarianceRatioRow:
    """One Lo-MacKinlay variance ratio; ``vr`` near 1 is consistent with a random walk."""

    lag: int
    vr: float
    statistic: float
    p_value: float


@dataclass(frozen=True, slots=True)
class ADFRow:
    """Augmented Dickey-Fuller on log-returns."""

    statistic: float
    p_value: float
    used_lag: int
    n_obs: int


def hurst_rs(x: np.ndarray, min_n: int = HURST_MIN_N, max_n: int | None = None) -> float:
    """Hurst exponent by rescaled range: the OLS slope of log(R/S) on log(n).

    Block sizes are dyadic; at each size the mean R/S over non-overlapping blocks
    enters the regression.

    Args:
        x: The series, normally log-returns.
        min_n: Smallest block.
        max_n: Largest block, by default ``len(x) // 4``.

    Raises:
        ValueError: If fewer than three usable block sizes fit.
    """
    x = np.asarray(x, dtype=np.float64)
    n_total = len(x)
    if max_n is None:
        max_n = n_total // 4
    sizes = [n for n in (min_n * 2**i for i in range(64)) if n <= max_n]
    if len(sizes) < HURST_MIN_BLOCK_SIZES:
        raise ValueError(
            f"R/S needs at least {HURST_MIN_BLOCK_SIZES} block sizes between "
            f"{min_n} and {max_n}; got {len(sizes)} at n={n_total}"
        )

    logs_n: list[float] = []
    logs_rs: list[float] = []
    for n in sizes:
        blocks = x[: (n_total // n) * n].reshape(-1, n)
        deviate = np.cumsum(blocks - blocks.mean(axis=1, keepdims=True), axis=1)
        spread = deviate.max(axis=1) - deviate.min(axis=1)
        sd = blocks.std(axis=1, ddof=1)
        # A constant block has no scale to rescale by. Dropping it is not
        # imputation -- nothing is invented, the block simply carries no R/S.
        keep = sd > 0
        if not keep.any():
            continue
        logs_n.append(float(np.log(n)))
        logs_rs.append(float(np.log(float((spread[keep] / sd[keep]).mean()))))

    if len(logs_n) < HURST_MIN_BLOCK_SIZES:
        raise ValueError(
            f"only {len(logs_n)} block sizes carried a non-zero standard deviation"
        )
    slope, _ = np.polyfit(np.asarray(logs_n), np.asarray(logs_rs), 1)
    return float(slope)


def variance_ratios(
    r: np.ndarray, lags: tuple[int, ...] = VR_LAGS
) -> list[VarianceRatioRow]:
    """Lo-MacKinlay variance ratio at each lag, from a return series.

    ``VarianceRatio`` expects a level series and differences it itself, so the
    returns are cumulated first; passing returns would report ``VR = 1/lag``. The
    returns are computed per segment, so the cumulation crosses no gap.
    """
    from arch.unitroot import VarianceRatio

    level = np.cumsum(np.asarray(r, dtype=np.float64))
    rows: list[VarianceRatioRow] = []
    for lag in lags:
        ratio = VarianceRatio(level, lags=lag, trend=VR_TREND, overlap=True)
        rows.append(
            VarianceRatioRow(
                lag=lag,
                vr=float(ratio.vr),
                statistic=float(ratio.stat),
                p_value=float(ratio.pvalue),
            )
        )
    return rows


def adf(r: np.ndarray) -> ADFRow:
    """Augmented Dickey-Fuller with AIC lag selection, on log-returns."""
    from statsmodels.tsa.stattools import adfuller

    stat, p_value, used_lag, n_obs, *_ = adfuller(
        np.asarray(r, dtype=np.float64), autolag="AIC"
    )
    return ADFRow(
        statistic=float(stat),
        p_value=float(p_value),
        used_lag=int(used_lag),
        n_obs=int(n_obs),
    )


def _row(span: str, r: np.ndarray) -> dict[str, float | str | int]:
    """One Table 2 row: ADF, Hurst and the variance ratio at every lag."""
    unit_root = adf(r)
    row: dict[str, float | str | int] = {
        "span": span,
        "n": int(len(r)),
        "adf_stat": unit_root.statistic,
        "adf_p": unit_root.p_value,
        "hurst": hurst_rs(r),
    }
    for ratio in variance_ratios(r):
        row[f"vr_{ratio.lag}"] = ratio.vr
        row[f"vr_p_{ratio.lag}"] = ratio.p_value
    return row


def _training_returns(features: pl.DataFrame, origin: OriginLike) -> np.ndarray:
    """Log-returns of one origin's 21-month training sub-block."""
    lo = int(origin.train_start.timestamp() * 1000)
    hi = int(origin.train_sub_end.timestamp() * 1000)
    return (
        features.filter((pl.col("ts_ms") >= lo) & (pl.col("ts_ms") < hi))
        .get_column("r")
        .to_numpy()
    )


#: Shortest sub-block the R/S regression can describe: enough rows for
#: :data:`HURST_MIN_BLOCK_SIZES` dyadic sizes, each averaged over four blocks.
MIN_SPAN_ROWS: Final = HURST_MIN_N * 2 ** (HURST_MIN_BLOCK_SIZES - 1) * 4


def efficiency_table(
    features: pl.DataFrame, origins: list[OriginLike] | None = None
) -> pl.DataFrame:
    """Table 2: one row for the whole sample, then one per origin's training sub-block.

    Returns:
        ``span, n, adf_stat, adf_p, hurst`` plus ``vr_{lag}`` / ``vr_p_{lag}`` per lag.
        A sub-block shorter than ``MIN_SPAN_ROWS`` is skipped.
    """
    grid = list(origins if origins is not None else ORIGINS)
    rows = [_row("full", features.get_column("r").to_numpy())]
    for origin in grid:
        returns = _training_returns(features, origin)
        if len(returns) < MIN_SPAN_ROWS:
            continue
        rows.append(_row(origin.label, returns))
    return pl.DataFrame(rows)

"""The twelve variates and the K ladder cut over them.

Every variate is a per-bar function of the current bar (``r`` also reads the
previous close), so no feature uses a rolling window and no test bar can reach a
training feature. The ladder is cumulative: rung K is the first K columns and the
target ``r`` is channel 0 at every rung. Nothing outside families F1-F5 enters.

Upstream:
    Written here in polars from the published closed forms. F2 volatility:
    M. Parkinson, J. Business 53(1), 1980; M. B. Garman and M. J. Klass,
    J. Business 53(1), 1980; L. C. G. Rogers and S. E. Satchell, Ann. Appl.
    Probab. 1(4), 1991. Rogers-Satchell vanishes on shadowless bars, so it is
    taken as ``log(RS + 1e-9)``.
"""

from __future__ import annotations

import math
from typing import Final

import polars as pl

from itransformer_btc.segments import HOUR_MS, usable_mask

VARIATE_ORDER: Final[tuple[str, ...]] = (
    # F1 price trajectory — K=1 is `r` alone
    "r",
    "upper_shadow",
    "lower_shadow",
    # F3 intensity, first member — completes K=4
    "log_quote_volume",
    # K=8: intensity, order flow, intrabar location
    "log_trade_count",
    "taker_buy_ratio",
    "signed_flow",
    "vwap_location",
    # K=12: the F2 volatility estimators + the dependent intensity member
    "log_parkinson",
    "log_garman_klass",
    "log_rogers_satchell",
    "log_mean_trade_size",
)

TARGET: Final = "r"
TARGET_INDEX: Final = 0

#: Parkinson's normaliser, ``1 / (4 ln 2)``.
_PARKINSON_C: Final = 1.0 / (4.0 * math.log(2.0))
#: Garman–Klass's second-term coefficient, ``2 ln 2 - 1`` ≈ 0.386. Strictly
#: below 0.5, which is what keeps the estimator positive: ``|ln(C/O)| <=
#: ln(H/L)`` because C and O both lie in ``[L, H]``, so GK >= 0.114 (ln H/L)^2.
_GK_C: Final = 2.0 * math.log(2.0) - 1.0

_RS_STABILISER: Final = 1e-9


def ladder_columns(k: int) -> list[str]:
    """The variate names at rung ``k`` (1, 4, 8 or 12)."""
    if k not in (1, 4, 8, 12):
        raise ValueError(f"K must be a documented rung 1/4/8/12, got {k}")
    return list(VARIATE_ORDER[:k])


def build_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Compute all twelve variates per segment, dropping each segment's first bar.

    ``r`` is computed within each segment, so no return spans a gap.

    Returns:
        ``ts_ms``, ``usable`` and the twelve variates in ladder order as Float64.

    Raises:
        ValueError: If any variate is null or non-finite, which means the segment
            law did not run.
    """
    if "usable" not in frame.columns:
        frame = usable_mask(frame)

    rows = frame.filter(pl.col("usable")).sort("ts_ms")

    # Segment identity from the timestamp alone. Excluded bars are already gone,
    # so they show up here as jumps, exactly as downtime does.
    rows = rows.with_columns(
        (pl.col("ts_ms").diff().fill_null(HOUR_MS) != HOUR_MS).cum_sum().alias("_seg")
    )

    log_h_l = (pl.col("high") / pl.col("low")).log()
    log_c_o = (pl.col("close") / pl.col("open")).log()
    vwap = pl.col("quote_volume") / pl.col("volume")

    out = rows.with_columns(
        # -- F1 price trajectory, 3 dof -------------------------------------
        (pl.col("close").log() - pl.col("close").log().shift(1).over("_seg")).alias("r"),
        (pl.col("high") / pl.max_horizontal("open", "close")).log().alias("upper_shadow"),
        (pl.min_horizontal("open", "close") / pl.col("low")).log().alias("lower_shadow"),

        # -- F3 intensity, 2 dof — the third is the difference of the first two
        pl.col("quote_volume").log().alias("log_quote_volume"),
        pl.col("trades").log().alias("log_trade_count"),
        (pl.col("quote_volume") / pl.col("trades")).log().alias("log_mean_trade_size"),

        # Base-denominated: the buyer-initiated share of traded volume.
        (pl.col("taker_buy_base") / pl.col("volume")).alias("taker_buy_ratio"),

        # Total, because H == L bars break the series.
        ((vwap - pl.col("close")) / (pl.col("high") - pl.col("low"))).alias("vwap_location"),

        # Per-bar, never smoothed; only Rogers-Satchell needs the stabiliser.
        (_PARKINSON_C * log_h_l.pow(2)).log().alias("log_parkinson"),
        (0.5 * log_h_l.pow(2) - _GK_C * log_c_o.pow(2)).log().alias("log_garman_klass"),
        (
            (pl.col("high") / pl.col("close")).log() * (pl.col("high") / pl.col("open")).log()
            + (pl.col("low") / pl.col("close")).log() * (pl.col("low") / pl.col("open")).log()
            + _RS_STABILISER
        ).log().alias("log_rogers_satchell"),
    ).with_columns(
        # The product of two K=8 members; the dependence is disclosed.
        (
            (2.0 * pl.col("taker_buy_ratio") - 1.0) * pl.col("log_quote_volume")
        ).alias("signed_flow"),
    )

    # The first bar of each segment has no predecessor inside its segment.
    out = out.filter(pl.col("r").is_not_null())

    out = out.select(["ts_ms", "usable", *VARIATE_ORDER]).with_columns(
        [pl.col(c).cast(pl.Float64) for c in VARIATE_ORDER]
    )

    offenders = {
        name: n
        for name in VARIATE_ORDER
        if (
            n := int(
                out.select(
                    (~pl.col(name).is_finite() | pl.col(name).is_null()).sum()
                ).item()
            )
        )
    }
    if offenders:
        raise ValueError(
            f"non-finite variate values: {offenders}. Every variate is total "
            f"once zero-volume and H == L bars are excluded by the segment law, "
            f"so this means the exclusion did not run."
        )
    return out

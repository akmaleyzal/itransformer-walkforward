"""Market-efficiency tests, measured rather than assumed.

The load-bearing detail these tests pin is the one the API hides:
``arch.unitroot.VarianceRatio`` consumes a **level** series and differences it
itself. Fed the log-returns directly it returns VR = 1/lag --- 0.49, 0.25, 0.12,
0.06 at lags 2, 4, 8, 16 on white noise --- the signature of over-differencing,
and a number that would have been reported as decisive evidence against a random
walk when it is evidence of nothing but a misuse.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from itransformer_btc.efficiency import (
    adf,
    efficiency_table,
    hurst_rs,
    variance_ratios,
)

HOUR_MS = 3_600_000
DATA_START_MS = 1_514_764_800_000  # 2018-01-01T00:00Z


def test_hurst_of_white_noise_is_near_half():
    """H ~ 0.5 is the no-long-memory reading expected for log-returns."""
    rng = np.random.default_rng(42)
    assert 0.42 < hurst_rs(rng.standard_normal(20_000)) < 0.58


def test_hurst_of_a_random_walk_is_near_one():
    """The control that proves the estimator responds to memory at all.

    Without it, a function returning the constant 0.5 passes the test above and
    Table 2's Hurst column would be decoration.
    """
    rng = np.random.default_rng(43)
    assert hurst_rs(np.cumsum(rng.standard_normal(20_000))) > 0.80


def test_variance_ratio_of_white_noise_returns_is_near_one():
    """VR ~ 1 is the random-walk reading. The input is the RETURN series and the
    implementation is responsible for handing ``arch`` the level it expects."""
    rng = np.random.default_rng(44)
    for row in variance_ratios(rng.standard_normal(20_000), lags=(2, 4, 8, 16)):
        assert 0.9 < row.vr < 1.1, f"lag {row.lag}: VR={row.vr}"
        assert row.p_value > 0.01, f"lag {row.lag}: p={row.p_value}"


def test_variance_ratio_detects_real_positive_autocorrelation():
    """The other control. An AR(1) in returns is a genuine departure from a
    random walk and VR must exceed 1 --- otherwise the test above is satisfied by
    an implementation that always returns 1."""
    rng = np.random.default_rng(45)
    noise = rng.standard_normal(20_000)
    r = np.empty_like(noise)
    r[0] = noise[0]
    for i in range(1, len(r)):
        r[i] = 0.3 * r[i - 1] + noise[i]
    rows = {row.lag: row for row in variance_ratios(r, lags=(2, 4))}
    assert rows[2].vr > 1.1
    assert rows[2].p_value < 0.01


def test_adf_rejects_a_unit_root_on_white_noise():
    rng = np.random.default_rng(46)
    assert adf(rng.standard_normal(5_000)).p_value < 0.01


def test_adf_does_not_reject_a_unit_root_on_a_random_walk():
    rng = np.random.default_rng(47)
    assert adf(np.cumsum(rng.standard_normal(5_000))).p_value > 0.05


def _synthetic_features(n_hours: int, seed: int = 7) -> pl.DataFrame:
    """A feature frame carrying only what this module reads: ``ts_ms`` and ``r``."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "ts_ms": DATA_START_MS + np.arange(n_hours, dtype=np.int64) * HOUR_MS,
            "r": rng.standard_normal(n_hours) * 0.01,
        }
    )


def test_efficiency_table_puts_the_full_sample_first_then_origins():
    table = efficiency_table(_synthetic_features(30_000))
    spans = table.get_column("span").to_list()
    assert spans[0] == "full"
    assert len(spans) >= 3, spans
    assert spans[1:] == sorted(spans[1:]), "origin rows stay in walk-forward order"
    assert set(table.columns) >= {
        "span", "n", "adf_stat", "adf_p", "hurst",
        "vr_2", "vr_p_2", "vr_4", "vr_p_4", "vr_8", "vr_p_8", "vr_16", "vr_p_16",
    }


def test_efficiency_table_origin_rows_are_training_only():
    """Every origin row is cut from ``[train_start, train_sub_end)`` --- the same
    21-month sub-block the scaler is fitted on. These rows are descriptive and
    gate nothing, but reading a test block to describe
    the data would still be indefensible when avoiding it costs one filter."""
    frame = _synthetic_features(30_000)
    table = efficiency_table(frame)
    origin_rows = table.filter(pl.col("span") != "full")
    assert origin_rows.height >= 2
    assert (origin_rows.get_column("n").to_numpy() < frame.height).all()

"""K_eff, the estimators, directional accuracy and the executor's queue.

Reads ``data/raw/BTCUSDT_1h.parquet`` for the K_eff checks; everything else runs
on closed forms or synthetic inputs whose answer is known in advance, so the
file stays CPU-only and fast.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch

from itransformer_btc import keff, metrics, runner
from itransformer_btc.config import ORIGINS
from itransformer_btc.features import build_features
from itransformer_btc.segments import load_bars, usable_mask
from itransformer_btc.train import set_seed


@pytest.fixture(scope="session")
def feats() -> pl.DataFrame:
    return build_features(usable_mask(load_bars()))


# -- K_eff -------------------------------------------------------------------


def test_keff_reads_training_spans_only(feats: pl.DataFrame) -> None:
    """RQ1's regressor may not see a bar its outcome is measured on."""
    origin = ORIGINS[0]
    rows, windows = keff._training_windows(feats, origin, 8)
    ts = feats.get_column("ts_ms").to_numpy()
    lo = int(origin.train_start.timestamp() * 1000)
    hi = int(origin.train_sub_end.timestamp() * 1000)
    assert len(rows) == int(((ts >= lo) & (ts < hi)).sum())
    assert len(windows) > 0
    assert hi <= int(origin.test_start.timestamp() * 1000)


def test_participation_ratio_hits_both_bounds() -> None:
    assert keff.participation_ratio(np.array([5.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert keff.participation_ratio(np.ones(8)) == pytest.approx(8.0)
    with pytest.raises(ValueError):
        keff.participation_ratio(np.zeros(4))


def test_lookback_stable_rank_is_scale_free() -> None:
    """Rescaling one channel must not move it; otherwise volume's units dominate."""
    rng = np.random.default_rng(0)
    windows = rng.normal(size=(64, 96, 4))
    plain = keff.lookback_stable_rank(windows)
    rescaled = windows.copy()
    rescaled[:, :, 1] *= 1_000.0
    assert keff.lookback_stable_rank(rescaled) == pytest.approx(plain, rel=1e-9)
    assert 1.0 <= plain <= 4.0


def test_keff_row_is_bounded_and_reports_its_divergence(feats: pl.DataFrame) -> None:
    row = keff.keff_row(feats, ORIGINS[0], 8)
    assert 1.0 <= row.pr_raw <= 8.0
    assert 1.0 <= row.pr_window_norm <= 8.0
    assert 1.0 <= row.stable_rank_lookback <= 8.0
    assert 0.0 < row.pr_lookback_ratio <= 1.0
    assert row.divergence == pytest.approx(row.stable_rank_lookback - row.pr_raw)


def test_gate_action_is_disclosure_not_a_recut() -> None:
    assert "PASS" in keff.gate_verdict(6.0)
    verdict = keff.gate_verdict(4.0)
    assert "DISCLOSE" in verdict and "not re-cut" in verdict


# -- forecast-loss tests ------------------------------------------------------


def test_hln_factor_guard_fires_where_it_must() -> None:
    """At h=24 the HLN factor is exactly 0 at T=24; a silent negative would be complex."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="HLN factor"):
        metrics.dm_test(rng.normal(size=24), rng.normal(size=24), h=24)
    assert metrics.dm_test(rng.normal(size=720), rng.normal(size=720), h=24).T == 720


def test_clark_west_lifts_the_statistic_on_a_nested_pair() -> None:
    """A useful large model looks worse under plain DM; the CW term recovers it."""
    rng = np.random.default_rng(7)
    y = rng.normal(size=720)
    small = np.zeros(720)
    large = 0.05 * y + rng.normal(scale=0.25, size=720)
    cw = metrics.clark_west_test(y, small, large, h=24)
    dm = metrics.dm_test((y - small) ** 2, (y - large) ** 2, h=24)
    assert cw.one_sided and not dm.one_sided
    assert cw.statistic > dm.statistic


# -- RQ2 ---------------------------------------------------------------------


def _amp_panel(thin: dict[str, list[int]] | None = None) -> pl.DataFrame:
    """A balanced 15x6 panel shaped like :func:`metrics.amplification`.

    ``thin`` names origins whose listed blocks keep only 300 of 720 forecast
    origins, so a coverage restriction bites there and nowhere else.
    """
    thin = thin or {}
    rows = []
    for i, origin in enumerate(o.label for o in ORIGINS):
        for block in range(1, 7):
            n = 300 if block in thin.get(origin, []) else 720
            rows.append({"origin_index": i + 1, "origin": origin, "block": block,
                         "mse_small": 1.0, "n_small": n, "mse_large": 1.0, "n_large": n,
                         "A": 0.001 * block + 0.0001 * i})
    return pl.DataFrame(rows)


def test_bootstrap_p_never_returns_zero() -> None:
    """A finite bootstrap cannot support p = 0; the floor is ``1/(B+1)``."""
    blocks = list(range(1, 7))
    panel = pl.DataFrame({"origin": [f"o{i:02d}" for i in range(15) for _ in blocks],
                          "block": blocks * 15,
                          "A": [1.0 - 0.5 * b for _ in range(15) for b in blocks]})
    result = metrics.panel_beta1(panel, B=999, seed=1)
    assert result.beta1 == pytest.approx(-0.5)
    assert result.p_rademacher >= 1.0 / 1000.0
    assert result.headline_p == max(result.p_rademacher, result.p_webb)
    assert result.n_clusters == 15 and result.n_observations == 90


def test_beta1_is_the_mean_of_within_origin_slopes() -> None:
    """Inference is a one-sample test on G slopes, not on 90 observations."""
    rng = np.random.default_rng(3)
    blocks = list(range(1, 7))
    panel = pl.DataFrame({"origin": [f"o{i:02d}" for i in range(15) for _ in blocks],
                          "block": blocks * 15,
                          "A": [float(rng.normal()) for _ in range(15) for _ in blocks]})
    result = metrics.panel_beta1(panel, B=99, seed=1)
    assert result.beta1 == pytest.approx(float(result.within_slopes.mean()))
    assert len(result.within_slopes) == 15


def test_unbalanced_or_empty_panels_are_refused() -> None:
    panel = pl.DataFrame({"origin": ["a", "a", "a", "b", "b"], "block": [1, 2, 3, 1, 2],
                          "A": [0.1, 0.2, 0.3, 0.1, 0.2]})
    with pytest.raises(ValueError, match="unbalanced"):
        metrics.panel_beta1(panel, B=99)
    with pytest.raises(ValueError, match="empty panel"):
        metrics.panel_beta1(panel.clear(), B=99)


def test_beta1_with_coverage_returns_none_when_the_restriction_unbalances() -> None:
    full, restricted = metrics.beta1_with_coverage(_amp_panel(thin={"2020-01": [3], "2021-09": [6]}), B=999)
    assert full.n_observations == 90 and full.n_clusters == 15
    assert restricted is None


def test_beta1_with_coverage_estimates_when_whole_origins_drop_out() -> None:
    thin = {"2020-01": [1, 2, 3, 4, 5, 6], "2020-06": [1, 2, 3, 4, 5, 6]}
    full, restricted = metrics.beta1_with_coverage(_amp_panel(thin=thin), B=999)
    assert full.n_observations == 90
    assert restricted is not None
    assert restricted.n_clusters == 13 and restricted.n_observations == 78


def test_beta1_with_coverage_is_a_no_op_when_every_block_is_complete() -> None:
    full, restricted = metrics.beta1_with_coverage(_amp_panel(), B=999)
    assert restricted is not None
    assert restricted.n_observations == full.n_observations
    assert restricted.beta1 == pytest.approx(full.beta1)


# -- RQ1 ---------------------------------------------------------------------


def test_tost_needs_the_margin_to_conclude_equivalence() -> None:
    """A non-significant delta is a failure to reject, not equivalence."""
    tight = np.full(15, 1e-6) + np.linspace(-1e-7, 1e-7, 15)
    assert metrics.tost_equivalence(tight, margin=1e-3).equivalent
    assert not metrics.tost_equivalence(tight, margin=1e-9).equivalent


def test_j_test_identifies_the_true_regressor() -> None:
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 4)
    k = np.tile(np.array([1.0, 4.0, 8.0, 12.0]), 20)
    k_eff = np.tile(np.array([1.0, 3.3, 4.3, 4.0]), 20)
    y = -0.1 * k_eff + rng.normal(scale=0.05, size=80) + np.repeat(rng.normal(size=20), 4)
    _, p_k_needs_keff = metrics.j_test(y, k, k_eff, groups)
    _, p_keff_needs_k = metrics.j_test(y, k_eff, k, groups)
    assert p_k_needs_keff < 0.01
    assert p_keff_needs_k > 0.05


def test_raw_scale_table_reconciles_the_two_scales() -> None:
    frame = pl.DataFrame({"mse": [0.25, 4.0], "sigma_g": [0.01, 0.02]})
    assert metrics.raw_scale_table(frame)["rmse_raw"].to_list() == pytest.approx([0.005, 0.04])


# -- directional accuracy ------------------------------------------------------


def test_non_overlapping_phase_is_midnight_utc() -> None:
    keep = metrics.non_overlapping_mask(np.arange(0, 48) * metrics.HOUR_MS)
    assert list(np.flatnonzero(keep)) == [0, 24]


def _forecasts(pred_sign: float, *, zero_first: bool = False) -> pl.DataFrame:
    """Two days of hourly issuances; the forecast is ``pred_sign`` times the truth."""
    ts = np.arange(48, dtype=np.int64) * metrics.HOUR_MS
    truth = np.random.default_rng(5).standard_normal((48, 24))
    truth[truth == 0] = 0.5
    pred = pred_sign * truth
    if zero_first:
        pred[:, 0] = 0.0
    return pl.DataFrame({"timestamp": np.repeat(ts, 24), "step": np.tile(np.arange(1, 25), 48),
                         "y_true": truth.reshape(-1), "y_pred": pred.reshape(-1)})


def test_directional_accuracy_scores_signs_on_raw_returns() -> None:
    right = metrics.directional_accuracy(_forecasts(1.0), sigma_g=1.0, mu_g=0.0)
    assert (right.da_1h, right.da_24h, right.da_cum) == (1.0, 1.0, 1.0)
    assert right.n_1h == 48 and right.n_daily == 2
    wrong = metrics.directional_accuracy(_forecasts(-1.0), sigma_g=1.0, mu_g=0.0)
    assert (wrong.da_1h, wrong.da_24h, wrong.da_cum) == (0.0, 0.0, 0.0)
    zero = metrics.directional_accuracy(_forecasts(1.0, zero_first=True), sigma_g=1.0, mu_g=0.0)
    assert zero.da_1h == 0.0, "a zero forecast counts as wrong"
    with pytest.raises(ValueError):
        metrics.directional_accuracy(_forecasts(1.0), sigma_g=0.0, mu_g=0.0)


def test_directional_summary_compares_with_the_majority_rate_per_origin() -> None:
    rows = [{"model": "itr", "k": 8, "origin_index": o, "seed": s,
             **{f"da_{v}": 0.6 for v in metrics.DA_VARIANTS},
             **{f"base_{v}": 0.5 + 0.01 * o for v in metrics.DA_VARIANTS},
             **{f"p_{v}": 0.01 for v in metrics.DA_VARIANTS}}
            for o in range(1, 16) for s in (42, 43)]
    summary = metrics.directional_accuracy_summary(pl.DataFrame(rows)).row(0, named=True)
    assert summary["n_origins"] == 15 and summary["runs"] == 30
    assert summary["dda_1h"] == pytest.approx(0.02)
    assert summary["wins_1h"] == 9, "origins 1-9 have a majority rate below 0.6"
    assert summary["pt_cum"] == 30


# -- the executor --------------------------------------------------------------


def test_parallel_executor_runs_every_cell_exactly_once(tmp_path, monkeypatch) -> None:
    """Two workers on one queue: no cell run twice, none dropped."""
    seen: list[str] = []
    seen_lock = threading.Lock()

    def fake_run(cell, cache, device, out_root, roots, guard):
        with seen_lock:
            seen.append(cell.run_id)
        time.sleep(0.001)
        return SimpleNamespace(train=[0]), SimpleNamespace(epochs_run=0, best_val_mse=0.0)

    monkeypatch.setattr(runner, "_run_cell", fake_run)
    monkeypatch.setattr(runner, "_check_alignment", lambda *a, **k: None)
    monkeypatch.setattr(runner, "pending", lambda cells, roots: list(cells))
    cells = runner.manifest()[:40]
    summary = runner.execute_parallel(cells, pl.DataFrame(),
                                      devices=[torch.device("cpu"), torch.device("cpu")],
                                      out_root=tmp_path, roots=[tmp_path], log=lambda _msg: None)
    assert summary.completed == len(cells)
    assert sorted(seen) == sorted(c.run_id for c in cells)
    assert len(seen) == len(set(seen)), "a cell was handed to both workers"


def test_set_seed_gives_the_same_cpu_draws_with_or_without_a_device() -> None:
    set_seed(42)
    first = torch.randn(4)
    set_seed(42, torch.device("cpu"))
    assert torch.equal(first, torch.randn(4))
    set_seed(43, torch.device("cpu"))
    assert not torch.equal(first, torch.randn(4))


def test_resume_refuses_a_run_from_a_different_code_vintage(tmp_path) -> None:
    """A code change keeps the run_id, so resume must compare the digest too."""
    root = tmp_path / "artifacts"
    (root / "preds").mkdir(parents=True)
    (root / "meta").mkdir(parents=True)
    run_id = "itr_o01_K08_H024_s42"
    (root / "preds" / f"{run_id}.parquet").write_bytes(b"PAR1")
    (root / "meta" / f"{run_id}.json").write_text(
        json.dumps({"status": "complete", "code_sha256": "stale" * 8}))

    assert runner.completed_run_ids([root]) == {run_id}
    assert runner.completed_run_ids([root], code_digest="stale" * 8) == {run_id}
    assert runner.completed_run_ids([root], code_digest="current") == set()
    cell = next(c for c in runner.manifest() if c.run_id == run_id)
    assert runner.pending([cell], [root]) == [cell]

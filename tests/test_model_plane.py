"""Feature, split and model tests. Reads ``data/raw/BTCUSDT_1h.parquet``; writes nothing."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
import torch
from torch import nn

from itransformer_btc.baselines import RIDGE_ALPHAS, RidgeConfig
from itransformer_btc.config import ORIGINS, PRED_LEN, SEQ_LEN
from itransformer_btc.features import (
    TARGET,
    TARGET_INDEX,
    VARIATE_ORDER,
    build_features,
    ladder_columns,
)
from itransformer_btc.model import ITransformerConfig, VanillaConfig
from itransformer_btc.segments import build_segments, load_bars, usable_mask
from itransformer_btc.splits import (
    OriginTensors,
    Scaler,
    SplitTensors,
    build_origin_tensors,
)
from itransformer_btc.train import (
    RunSpec,
    scale_invariance_check,
    set_seed,
    write_artifacts,
)


@pytest.fixture(scope="session")
def raw() -> pl.DataFrame:
    return usable_mask(load_bars())


@pytest.fixture(scope="session")
def feats(raw: pl.DataFrame) -> pl.DataFrame:
    return build_features(raw)


# -- the twelve variates -------------------------------------------------------


def test_twelve_variates_in_ladder_order(feats: pl.DataFrame) -> None:
    """Rung K is the first K columns, so ``r`` is channel 0 at every rung."""
    assert len(VARIATE_ORDER) == 12
    assert VARIATE_ORDER[TARGET_INDEX] == TARGET
    assert list(feats.columns[2:]) == list(VARIATE_ORDER)
    for k in (1, 4, 8, 12):
        assert ladder_columns(k) == list(VARIATE_ORDER[:k])
    with pytest.raises(ValueError):
        ladder_columns(16)


def test_every_variate_is_finite(feats: pl.DataFrame) -> None:
    """The segment law is what makes each variate a total function."""
    for name in VARIATE_ORDER:
        col = feats.get_column(name)
        assert col.is_finite().all(), name
        assert col.null_count() == 0, name


def test_f3_has_two_degrees_of_freedom(feats: pl.DataFrame) -> None:
    """The third intensity member is the difference of the first two.

    Exact to the last bit, not merely correlated: ``log(q/n) = log q - log n``.
    That is why F3 contributes 2 dof to K_eff and not 3, and why
    ``log_mean_trade_size`` sits in the deliberately redundant K=12 rung.
    """
    residual = feats.get_column("log_mean_trade_size") - (
        feats.get_column("log_quote_volume") - feats.get_column("log_trade_count")
    )
    assert float(residual.abs().max()) < 1e-9


def test_rogers_satchell_is_not_strictly_positive(raw: pl.DataFrame) -> None:
    """Not all three F2 estimators are strictly positive.

    RS vanishes on a shadowless bar — H equal to one of O/C and L equal to the
    other. Such a bar has H > L and passes the segment law; it is a marubozu,
    not a degenerate bar. Parkinson and Garman-Klass really are strictly
    positive, which is why the stabiliser applies to RS alone.
    """
    bars = raw.filter(pl.col("usable"))
    rs = (
        (pl.col("high") / pl.col("close")).log() * (pl.col("high") / pl.col("open")).log()
        + (pl.col("low") / pl.col("close")).log() * (pl.col("low") / pl.col("open")).log()
    )
    zeros = bars.select(rs.alias("rs")).filter(pl.col("rs") <= 0)
    assert zeros.height == 33, f"expected 33 shadowless bars, found {zeros.height}"


def test_the_stabiliser_lands_inside_the_measured_support(feats: pl.DataFrame) -> None:
    """kappa = 1e-9 chosen so log kappa is not an out-of-support spike.

    A hard floor far below support (1e-12 gives -27.6, about -11 sigma) would
    distort the instance normalisation of every window containing one and would
    smuggle a categorical marubozu flag into a continuous variate.
    """
    col = feats.get_column("log_rogers_satchell")
    floor = float(np.log(1e-9))
    assert float(col.min()) == pytest.approx(floor, abs=1e-6)
    assert float(col.quantile(0.001)) > floor, "the floor should sit in the low tail"
    assert int((col <= floor + 1e-9).sum()) <= 40


def test_features_drop_exactly_one_bar_per_segment(
    raw: pl.DataFrame, feats: pl.DataFrame
) -> None:
    """``r`` is per segment, so each segment's first bar has no predecessor.

    Computing ``r`` on a concatenated series instead would inject cross-gap
    returns into mu_g and sigma_g before any window is excluded — the 33-hour
    2018-02-08 outage booked as a one-hour return.
    """
    n_segments = len(build_segments(raw))
    assert int(raw.get_column("usable").sum()) - feats.height == n_segments


# -- splits -------------------------------------------------------------------


def test_purge_holds_and_scaler_sees_training_only(feats: pl.DataFrame) -> None:
    """FATAL. Targets may not cross a boundary; inputs may."""
    for origin in ORIGINS[:3]:
        t = build_origin_tensors(feats, origin, 8)
        assert t.train.ts.max() < int(origin.val_start.timestamp() * 1000)
        assert len(t.val) > 0
        assert t.scaler.columns == tuple(ladder_columns(8))


def test_test_blocks_use_origin_semantics(feats: pl.DataFrame) -> None:
    """720 forecast origins per clean block, never 601."""
    counts = [
        len(split)
        for origin in ORIGINS
        for split in build_origin_tensors(feats, origin, 1).test_blocks
    ]
    assert max(counts) == 720
    assert sum(1 for n in counts if n == 720) >= 70


def test_naive_rw_is_the_drift_free_baseline(feats: pl.DataFrame) -> None:
    """``y_z = -mu_g/sigma_g``, and the tilt changes sign.

    ``y_z = 0`` would silently mean ``r_hat = mu_g``, a constant-drift model
    wearing the EMH baseline's name. The magnitude is small, but it flips sign across origins, so it is not a constant
    a reader could subtract.
    """
    ratios = [
        build_origin_tensors(feats, origin, 1).scaler.target_mu_over_sigma
        for origin in ORIGINS
    ]
    assert max(abs(r) for r in ratios) < 0.02, "if this grows, revisit the drift baseline"
    assert min(ratios) < 0 < max(ratios), "the tilt must change sign across origins"


# -- the three models ----------------------------------------------------------

K_RUNGS = (1, 4, 8, 12)


def _tensors(k: int, n_train: int = 64, n_val: int = 32, seed: int = 0) -> OriginTensors:
    rng = np.random.default_rng(seed)

    def split(n: int, start: int) -> SplitTensors:
        x = rng.standard_normal((n, SEQ_LEN, k)).astype(np.float32)
        y = (x[:, -PRED_LEN:, TARGET_INDEX] * 0.3 + 0.1 * rng.standard_normal((n, PRED_LEN))).astype(np.float32)
        return SplitTensors(x=x, y=y, ts=np.arange(start, start + n, dtype=np.int64) * 3_600_000)

    scaler = Scaler(np.zeros(k), np.ones(k), tuple(ladder_columns(k)))
    return OriginTensors(origin=ORIGINS[0], k=k, scaler=scaler, train=split(n_train, 0),
                         val=split(n_val, 10_000), test_blocks=(split(16, 20_000),),
                         block_labels=(1,))


@pytest.mark.parametrize("k", K_RUNGS)
def test_every_model_maps_a_window_to_the_target_horizon(k: int) -> None:
    x = torch.randn(3, SEQ_LEN, k)
    for cfg in (ITransformerConfig(), VanillaConfig(k=k), RidgeConfig(k=k)):
        assert tuple(cfg.build().forecast_target(x).shape) == (3, PRED_LEN)


def test_parameter_counts_follow_the_architecture() -> None:
    itr = {k: ITransformerConfig().build().n_parameters() for k in K_RUNGS}
    vtr = {k: VanillaConfig(k=k).build().n_parameters() for k in K_RUNGS}
    assert len(set(itr.values())) == 1, "K changes the token count, not a weight shape"
    assert vtr[1] < vtr[4] < vtr[8] < vtr[12], "the embeddings and head are K channels wide"


def test_use_norm_scale_invariance() -> None:
    set_seed(42)
    model = ITransformerConfig().build().eval()
    base, scaled = scale_invariance_check(model, torch.randn(64, SEQ_LEN, 8), torch.randn(64, PRED_LEN))
    assert abs(base - scaled) / base < 1e-3


@pytest.mark.parametrize("cfg", [ITransformerConfig(dropout=0.0), VanillaConfig(k=8, dropout=0.0)],
                         ids=["itr", "vtr"])
def test_single_batch_overfits_with_dropout_off(cfg) -> None:
    set_seed(42)
    model = cfg.build().train()
    xs, ys = torch.randn(8, SEQ_LEN, 8), torch.randn(8, PRED_LEN)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model.forecast_target(xs), ys)
        loss.backward()
        opt.step()
    assert loss.item() < 1e-3


def test_only_the_target_channel_carries_gradient() -> None:
    set_seed(0)
    model = VanillaConfig(k=4).build()
    nn.functional.mse_loss(model.forecast_target(torch.randn(4, SEQ_LEN, 4)), torch.randn(4, PRED_LEN)).backward()
    head = model.model.decoder.projection.weight.grad
    assert head[TARGET_INDEX].abs().sum() > 0
    assert torch.count_nonzero(head[TARGET_INDEX + 1:]) == 0


def test_vanilla_decoder_reads_only_the_lookback() -> None:
    model = VanillaConfig(k=2).build()
    x = torch.arange(SEQ_LEN, dtype=torch.float32).repeat(1, 2, 1).reshape(1, 2, SEQ_LEN).transpose(1, 2)
    decoder = model.decoder_input(x)
    assert decoder.shape == (1, 48 + PRED_LEN, 2)
    assert torch.equal(decoder[:, :48], x[:, -48:])
    assert torch.count_nonzero(decoder[:, 48:]) == 0


def test_ridge_selects_alpha_on_validation_and_matches_sklearn() -> None:
    from sklearn.linear_model import Ridge

    tensors = _tensors(4)
    model, cfg, outcome = RidgeConfig(k=4).fit(tensors, RunSpec("rdg", 1, 4, PRED_LEN, 42), device=torch.device("cpu"))
    assert cfg.alpha in RIDGE_ALPHAS and outcome.epochs_run == 0
    flat = lambda s: s.x.reshape(len(s), -1).astype(np.float64)
    scores = {a: np.mean((Ridge(alpha=a).fit(flat(tensors.train), tensors.train.y).predict(flat(tensors.val)) - tensors.val.y) ** 2)
              for a in RIDGE_ALPHAS}
    assert cfg.alpha == min(scores, key=scores.get)
    reference = Ridge(alpha=cfg.alpha).fit(flat(tensors.train), tensors.train.y).predict(flat(tensors.val))
    np.testing.assert_allclose(model.forecast_target(torch.from_numpy(tensors.val.x)).detach().numpy(), reference, atol=1e-4)


def test_ridge_seeds_give_identical_predictions() -> None:
    tensors = _tensors(8)
    preds = [
        RidgeConfig(k=8).fit(tensors, RunSpec("rdg", 1, 8, PRED_LEN, seed), device=torch.device("cpu"))[0]
        .forecast_target(torch.from_numpy(tensors.test_blocks[0].x)).detach().numpy().tobytes()
        for seed in (42, 43, 44, 45, 46)
    ]
    assert len(set(preds)) == 1


def test_write_artifacts_records_the_run(tmp_path) -> None:
    tensors = _tensors(4)
    model, cfg, outcome = RidgeConfig(k=4).fit(tensors, RunSpec("rdg", 1, 4, PRED_LEN, 42), device=torch.device("cpu"))
    preds, meta = write_artifacts(model, tensors, RunSpec("rdg", 1, 4, PRED_LEN, 42), cfg, outcome,
                                  torch.device("cpu"), root=tmp_path, requested_config=RidgeConfig(k=4))
    record = json.loads(meta.read_text(encoding="utf-8"))
    assert record["config"]["alpha"] == cfg.alpha and record["requested_config"]["alpha"] is None
    assert record["loss_target"] == "target" and record["status"] == "complete"
    assert record["environment"]["upstream"].endswith("c2426e68ca13f74aaec08045c5c724d8ad328124")
    assert pl.read_parquet(preds).height == 16 * PRED_LEN

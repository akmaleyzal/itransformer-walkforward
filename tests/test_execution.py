"""Resume, the two-device executor, the pilot, and what training may read."""

import copy
import hashlib
import importlib.util
import json
import math
import sys
import threading
import time
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch

from itransformer_btc import comparisons, metrics, runner, train
from itransformer_btc.baselines import RidgeConfig
from itransformer_btc.config import ORIGINS
from itransformer_btc.model import ITransformerConfig, VanillaConfig
from itransformer_btc.splits import OriginTensors, Scaler, SplitTensors


def test_a_stale_run_reaches_the_executor(tmp_path, monkeypatch):
    cell = runner.RunCell("main", 1, 8, 24, 42)
    (tmp_path / "preds").mkdir()
    (tmp_path / "meta").mkdir()
    (tmp_path / "preds" / f"{cell.run_id}.parquet").write_bytes(b"PAR1")
    (tmp_path / "meta" / f"{cell.run_id}.json").write_text(json.dumps({
        "status": "complete", "code_sha256": "old", "input_sha256": "input"}))
    called = []

    def fit(self, tensors, spec, *, device=None):
        called.append(spec.run_id)
        return object(), self, SimpleNamespace(epochs_run=0, best_val_mse=0.)

    monkeypatch.setattr(ITransformerConfig, "fit", fit)
    monkeypatch.setattr(runner._TensorCache, "get", lambda *a: SimpleNamespace(train=[0]))
    monkeypatch.setattr(runner, "write_artifacts", lambda *a, **k: None)
    assert runner.pending([cell], [tmp_path]) == [cell]
    result = runner.execute_parallel([cell], pl.DataFrame(), roots=[tmp_path], out_root=tmp_path,
                                     devices=[torch.device("cpu"), torch.device("cpu")],
                                     log=lambda *a: None)
    assert called == [cell.run_id]
    assert (result.completed, result.skipped, result.failed) == (1, 0, 0)


def test_resume_requires_input_config_and_code(tmp_path, monkeypatch):
    cell = runner.RunCell("main", 1, 8, 24, 42)
    cfg = cell.model_config()
    (tmp_path / "preds").mkdir()
    (tmp_path / "meta").mkdir()
    (tmp_path / "weights").mkdir()
    pred_path = tmp_path / "preds" / f"{cell.run_id}.parquet"
    pl.DataFrame({name: [1] for name in ("block", "step", "timestamp", "forecast_origin",
                                        "input_start", "target_timestamp", "y_true", "y_pred")}
                 ).write_parquet(pred_path)
    weight_path = tmp_path / "weights" / f"{cell.run_id}.pt"
    torch.save(cfg.build().state_dict(), weight_path)
    monkeypatch.setattr(train, "_input_sha256", lambda *a: ("input", "file-digest"))
    metadata = {"run_id": cell.run_id, "status": "complete", "code_sha256": train.code_sha256(),
                "input_sha256": "input", "requested_config": asdict(cfg),
                "schedule": asdict(cfg.schedule()), "prediction_schema_version": 2,
                "weights_sha256": hashlib.sha256(weight_path.read_bytes()).hexdigest(),
                "predictions_sha256": hashlib.sha256(pred_path.read_bytes()).hexdigest()}
    path = tmp_path / "meta" / f"{cell.run_id}.json"
    path.write_text(json.dumps(metadata))
    assert train.is_complete(cell.run_id, tmp_path, strict=True, cfg=cfg)
    assert not train.is_complete(cell.run_id, tmp_path, strict=True, cfg=replace(cfg, d_ff=512))
    for field in ("input_sha256", "code_sha256", "requested_config"):
        bad = dict(metadata)
        bad.pop(field)
        path.write_text(json.dumps(bad))
        assert not train.is_complete(cell.run_id, tmp_path, strict=True, cfg=cfg)


def test_a_dependency_change_invalidates_the_caller_output():
    path = Path(__file__).resolve().parents[1] / "tools/build_notebook.py"
    spec = importlib.util.spec_from_file_location("execution_nb_generator", path)
    generator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = generator
    spec.loader.exec_module(generator)
    previous = {"cells": [
        {"cell_type": "code", "source": ["def f():\n    return 1\n"], "outputs": [],
         "metadata": {"itbtc": {"role": "module", "module": "probe.py", "section": "f"}}},
        {"cell_type": "code", "source": ["print(f())\n"], "execution_count": 2,
         "outputs": [{"output_type": "stream", "name": "stdout", "text": ["1\n"]}],
         "metadata": {"itbtc": {"role": "step", "step": "probe"}}},
    ]}
    current = copy.deepcopy(previous)
    current["cells"][0]["source"] = ["def f():\n    return 2\n"]
    assert generator.carry_outputs(current, previous)[0] == 0
    assert current["cells"][1]["outputs"] == []


def test_cluster_robust_j_test_matches_the_full_dummy_regression():
    import statsmodels.api as sm
    from scipy.stats import t
    from statsmodels.stats.sandwich_covariance import cov_cluster
    rng = np.random.default_rng(43)
    groups = np.repeat(np.arange(30), 4)
    clusters = groups // 6
    a = np.tile([1., 4., 8., 12.], 30)
    b = a + rng.normal(size=len(a))
    y = .2 * b + rng.normal(size=len(a)) + np.repeat(rng.normal(size=30), 4)
    demean = lambda v: v - v.reshape(-1, 4).mean(axis=1).repeat(4)
    yd, ad, bd = map(demean, (y, a, b))
    fitted = bd * ((bd @ yd) / (bd @ bd))
    design = np.column_stack([ad, fitted, np.eye(30)[groups]])
    model = sm.OLS(y, design).fit()
    statistic = model.params[1] / math.sqrt(cov_cluster(model, clusters)[1, 1])
    actual = metrics.j_test(y, a, b, groups, clusters=clusters)
    np.testing.assert_allclose(actual, [statistic, 2 * t.sf(abs(statistic), 4)], rtol=1e-9)


def test_direction_is_read_after_undoing_the_scaler():
    """In scaler space both series have opposite signs; in raw returns both are positive."""
    stamps = np.repeat(np.arange(3) * 24 * metrics.HOUR_MS, 24)
    frame = pl.DataFrame({"timestamp": stamps, "step": np.tile(np.arange(1, 25), 3),
                          "y_true": np.full(72, .2), "y_pred": np.full(72, -.1)})
    raw = metrics.directional_accuracy(frame, sigma_g=1., mu_g=1.)
    assert raw.da_1h == raw.da_24h == raw.da_cum == 1.
    assert metrics.directional_accuracy(frame, sigma_g=1., mu_g=0.).da_cum == 0.


def test_inputs_end_before_the_first_target():
    from itransformer_btc.splits import _gather, window_starts
    hours = np.arange(1000, dtype=np.int64)
    ts = hours * metrics.HOUR_MS
    start = datetime.fromtimestamp(300 * 3600, tz=timezone.utc)
    idx = window_starts(ts, start, start + timedelta(hours=30), "origin", 96 + 24, seq_len=96)
    split = _gather(hours[:, None], idx, ts, 96, 24)
    np.testing.assert_array_equal(split.ts, ts[300:330])
    np.testing.assert_array_equal(split.y[:, 0], hours[300:330])
    assert np.all(split.x[:, -1, 0] < split.y[:, 0])


def test_the_panel_measures_seed_loss_not_ensemble_loss(monkeypatch):
    monkeypatch.setattr(comparisons, "_run_ids", lambda *a: ["s1", "s2"])
    monkeypatch.setattr(comparisons, "load_meta", lambda *a: {
        "code_sha256": "code", "input_sha256": "input", "naive_rw_z": 0.})

    def prediction(run_id, roots):
        return pl.DataFrame({"block": [1, 1], "timestamp": [0, 0],
                             "target_timestamp": [0, metrics.HOUR_MS], "step": [1, 2],
                             "y_true": [1., 1.], "y_pred": [-1., -1.] if run_id == "s1" else [3., 3.]})

    monkeypatch.setattr(comparisons, "load_predictions", prediction)
    panel = comparisons.build_panel([("itr", 8)], [], pred_len=2, origin_indices=(1,))
    assert np.mean((panel.y_true[1] - panel.y_pred[("itr", 8), 1]) ** 2) == 0.
    assert np.mean(panel.seed_losses[("itr", 8), 1]) == 4.
    assert comparisons.per_origin_loss(panel, ("itr", 8)).item() == 4.


def test_the_pilot_fits_every_model_on_validation_and_caches_it(monkeypatch, tmp_path):
    val = SplitTensors(np.zeros((3, 96, 1), dtype=np.float32), np.zeros((3, 24), dtype=np.float32),
                       np.arange(3))
    tensors = SimpleNamespace(val=val, training_selection={"selected": 10}, naive_rw_z=0.0)
    monkeypatch.setattr(runner._TensorCache, "get", lambda *a: tensors)
    monkeypatch.setattr(runner, "_input_sha256", lambda *a: ("input", "fixture"))
    calls = []

    def fit(self, tensors, spec, *, device=None):
        calls.append(spec.run_id)
        return object(), self, SimpleNamespace(best_val_mse=1.0, epochs_run=2, wall_time_s=0.5)

    for cls in (ITransformerConfig, VanillaConfig, RidgeConfig):
        monkeypatch.setattr(cls, "fit", fit)
    monkeypatch.setattr(runner, "write_artifacts", lambda *a, **k: pytest.fail("the pilot writes no predictions"))
    kwargs = dict(devices=[torch.device("cpu")], out_root=tmp_path, log=lambda *a: None)
    result = runner.pilot(pl.DataFrame(), **kwargs)
    again = runner.pilot(pl.DataFrame(), **kwargs)
    assert len(result.rows) == 12 and len(calls) == 12, "the second pilot reads the cache"
    assert {row["model"] for row in result.rows} == {"itr", "rdg", "vtr"}
    assert result.mean_wall_s() == again.mean_wall_s() == {"itr": 0.5, "rdg": 0.5, "vtr": 0.5}


def test_a_mid_epoch_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    rng = np.random.default_rng(12)
    x = rng.normal(size=(8, 4, 1)).astype(np.float32)
    y = rng.normal(size=(8, 2)).astype(np.float32)
    split = SplitTensors(x, y, np.arange(8, dtype=np.int64))
    tensors = OriginTensors(ORIGINS[0], 1, Scaler(np.zeros(1), np.ones(1), ("r",)),
                            split, split, (split,), (1,))
    cfg = ITransformerConfig(seq_len=4, pred_len=2, d_model=8, d_ff=16, n_heads=2, e_layers=1, dropout=.2)
    spec = train.RunSpec("itr", 1, 1, 2, 42)
    settings = dict(device=torch.device("cpu"), max_epochs=3, patience=3, batch_size=4)
    monkeypatch.setattr(train, "_input_sha256", lambda *a: ("synthetic-input", "fixture"))
    expected, expected_outcome = train.train_one(tensors, spec, cfg, **settings)
    clock = [0.]
    monkeypatch.setattr(train, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    original_step = torch.optim.Adam.step
    calls = [0]

    def interrupt_step(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        calls[0] += 1
        if calls[0] == 3:  # first minibatch of epoch two
            clock[0] = 100.
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", interrupt_step)
    first = tmp_path / "session1"
    with train.TrainingSession(first, [], 50.):
        with pytest.raises(train.SessionBudgetExhausted, match="epoch 1"):
            train.train_one(tensors, spec, cfg, **settings)
    checkpoint = first / "checkpoints" / f"{spec.run_id}.pt"
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["epoch"] == 1
    monkeypatch.setattr(torch.optim.Adam, "step", original_step)
    clock[0] = 0.
    with train.TrainingSession(tmp_path / "session2", [first], 50.):
        actual, outcome = train.train_one(tensors, spec, cfg, **settings)
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], value, rtol=0, atol=0)
    assert outcome.best_val_mse == expected_outcome.best_val_mse
    # A checkpoint of another configuration is never loaded, even when unusable.
    payload["identity"]["config"]["d_ff"] = 999
    payload["model"] = {"invalid": torch.ones(1)}
    torch.save(payload, checkpoint)
    with train.TrainingSession(tmp_path / "session3", [first], 50.):
        fresh, _ = train.train_one(tensors, spec, cfg, **settings)
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(fresh.state_dict()[name], value, rtol=0, atol=0)


def test_the_training_sample_is_fixed_and_the_scaler_sees_training_only():
    from itransformer_btc.features import ladder_columns
    from itransformer_btc.splits import build_origin_tensors
    date = lambda hour: datetime.fromtimestamp(hour * 3600, tz=timezone.utc)
    origin = SimpleNamespace(label="synthetic", train_start=date(0), train_sub_end=date(1000),
                             val_start=date(1000), val_end=date(1300),
                             blocks=lambda: [(1, date(1300), date(1400))])
    values = np.random.default_rng(18).normal(size=(1500, 8))
    frame = pl.DataFrame({"ts_ms": np.arange(1500, dtype=np.int64) * metrics.HOUR_MS,
                          **{name: values[:, i] for i, name in enumerate(ladder_columns(8))}})
    base = build_origin_tensors(frame, origin, 8, seq_len=12, pred_len=4, train_window_limit=100)
    assert len(base.train) == 100
    future = frame.with_columns([pl.when(pl.col("ts_ms") >= 1000 * metrics.HOUR_MS)
                                 .then(pl.col(c) * 100).otherwise(pl.col(c)).alias(c)
                                 for c in ladder_columns(8)])
    changed = build_origin_tensors(future, origin, 8, seq_len=12, pred_len=4, train_window_limit=100)
    np.testing.assert_array_equal(changed.scaler.mean, base.scaler.mean)
    np.testing.assert_array_equal(changed.scaler.std, base.scaler.std)
    np.testing.assert_array_equal(changed.train.x, base.train.x)
    assert changed.training_selection == base.training_selection
    with pytest.raises(ValueError, match="required"):
        build_origin_tensors(frame, origin, 8, train_window_limit=2000)


def test_two_workers_pause_at_the_deadline_and_surface_a_misalignment(tmp_path, monkeypatch):
    """A pause leaves the run pending; a comparator on other windows stops the grid."""
    monkeypatch.setattr(runner._TensorCache, "get", lambda *a: SimpleNamespace(train=[0]))

    def pause(*a, **kw):
        raise train.SessionBudgetExhausted("deadline fixture")

    monkeypatch.setattr(ITransformerConfig, "fit", pause)
    devices = [torch.device("cpu"), torch.device("cpu")]
    result = runner.execute_parallel([runner.RunCell("main", 1, 8, 24, 42)], pl.DataFrame(),
                                     roots=[tmp_path], out_root=tmp_path, devices=devices,
                                     log=lambda *a: None)
    assert (result.completed, result.failed, result.remaining) == (0, 0, 1)
    monkeypatch.setattr(RidgeConfig, "fit", lambda *a, **kw:
                        (None, None, SimpleNamespace(epochs_run=0, best_val_mse=1.)))
    monkeypatch.setattr(runner, "write_artifacts", lambda *a, **kw: None)

    def mismatch(*a, **kw):
        raise AssertionError("actual target calendars differ")

    monkeypatch.setattr(runner, "assert_baseline_alignment", mismatch)
    with pytest.raises(RuntimeError, match="comparator was scored"):
        runner.execute_parallel([runner.RunCell("ridge", 1, 8, 24, 42)], pl.DataFrame(),
                                roots=[tmp_path], out_root=tmp_path, devices=devices,
                                log=lambda *a: None)


def test_visible_devices_reports_every_cuda_device(monkeypatch):
    """Off a GPU this branch is dead code, so the CUDA report is faked."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert [str(d) for d in runner.visible_devices()] == ["cuda:0", "cuda:1"]
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert [str(d) for d in runner.visible_devices()] == ["cuda:0"]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert [str(d) for d in runner.visible_devices()] == ["cpu"]


def test_the_parallel_grid_routes_work_to_both_devices(tmp_path, monkeypatch):
    """Two distinct devices, so the device each cell was fitted on is observable.

    The fake fit sleeps; without a per-run cost the first thread would drain the
    queue before the second starts.
    """
    seen: list[tuple[str, str]] = []
    lock = threading.Lock()

    def fit(self, tensors, spec, *, device=None):
        time.sleep(0.004)
        with lock:
            seen.append((str(device), spec.run_id))
        return object(), self, SimpleNamespace(epochs_run=1, best_val_mse=0.0)

    written: list[str] = []
    monkeypatch.setattr(ITransformerConfig, "fit", fit)
    monkeypatch.setattr(runner._TensorCache, "get", lambda *a: SimpleNamespace(train=[0]))
    monkeypatch.setattr(runner, "is_complete", lambda *a, **k: False)
    monkeypatch.setattr(runner, "write_artifacts",
                        lambda model, tensors, spec, *a, **k: written.append(spec.run_id))
    spawned: list[str] = []
    real_thread = threading.Thread

    class Recording(real_thread):
        def __init__(self, *a, **kw):
            if str(kw.get("name", "")).startswith("grid-"):
                spawned.append(kw["name"])
            super().__init__(*a, **kw)

    monkeypatch.setattr(runner.threading, "Thread", Recording)
    cells = runner.manifest()[:40]
    summary = runner.execute_parallel(cells, pl.DataFrame(),
                                      devices=[torch.device("cuda", 0), torch.device("cuda", 1)],
                                      out_root=tmp_path, roots=[tmp_path], log=lambda *a: None)
    assert sorted(spawned) == ["grid-cuda:0", "grid-cuda:1"], "one worker per device"
    assert {d for d, _ in seen} == {"cuda:0", "cuda:1"}, "a device sat idle"
    assert summary.completed == len(cells)
    assert sorted(r for _, r in seen) == sorted(c.run_id for c in cells)
    assert sorted(written) == sorted(c.run_id for c in cells), "a cell was dropped or doubled"


def test_consolidation_keeps_the_newer_checkpoint_and_copies_a_valid_bundle(tmp_path, monkeypatch):
    cell = runner.RunCell("main", 1, 8, 24, 42)
    prior, out = tmp_path / "previous", tmp_path / "current"
    for folder, name, payload in (("preds", cell.run_id + ".parquet", b"predictions"),
                                  ("weights", cell.run_id + ".pt", b"weights"),
                                  ("meta", cell.run_id + ".json", b"metadata"),
                                  ("checkpoints", "pending.pt", b"old epoch"),
                                  ("validation", "probe.json", b"cached fit")):
        (prior / folder).mkdir(exist_ok=True, parents=True)
        (prior / folder / name).write_bytes(payload)
    (out / "checkpoints").mkdir(parents=True)
    (out / "checkpoints/pending.pt").write_bytes(b"new epoch")
    calls = []

    def accepted(run_id, root, **kwargs):
        calls.append(kwargs)
        return root == prior

    monkeypatch.setattr(runner, "is_complete", accepted)
    assert runner.consolidate_resume_outputs([cell], [out, prior], out_root=out) == 1
    assert all(call["strict"] and "cfg" in call and "columns" in call for call in calls)
    assert (out / "preds" / (cell.run_id + ".parquet")).read_bytes() == b"predictions"
    assert (out / "meta" / (cell.run_id + ".json")).read_bytes() == b"metadata"
    assert (out / "checkpoints/pending.pt").read_bytes() == b"new epoch"
    assert (out / "validation/probe.json").read_bytes() == b"cached fit"

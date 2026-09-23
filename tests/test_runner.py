"""The manifest, resume discovery, the design freeze, and one run of each model end to end."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from itransformer_btc import model as model_module
from itransformer_btc.config import K_LADDER, ORIGINS, SEEDS
from itransformer_btc.features import VARIATE_ORDER
from itransformer_btc.runner import (
    ALL_ARMS,
    RunCell,
    design_digest,
    discover_roots,
    execute_parallel,
    manifest,
    pending,
    require_frozen_design,
    resume_check,
    session_plan,
)
from itransformer_btc.train import TrainSchedule, is_complete


def test_manifest_is_900_unique_runs_in_model_order() -> None:
    cells = manifest()
    assert len(cells) == 900 == len({c.run_id for c in cells})
    assert [c.model_tag for c in cells[::300]] == ["itr", "rdg", "vtr"]
    for tag in ("itr", "rdg", "vtr"):
        mine = [c for c in cells if c.model_tag == tag]
        assert len(mine) == len(ORIGINS) * len(K_LADDER) * len(SEEDS) == 300


def test_the_three_models_of_a_cell_share_one_tensor_build() -> None:
    keys = {RunCell(arm, 3, 8, 24, 42).tensor_key for arm in ALL_ARMS}
    assert len(keys) == 1


def test_discovery_finds_outputs_at_any_depth(tmp_path: Path) -> None:
    deep = tmp_path / "inputs" / "datasets" / "owner" / "slug" / "artifacts"
    (deep / "meta").mkdir(parents=True)
    (deep / "preds").mkdir()
    roots = discover_roots(tmp_path / "work", inputs=tmp_path / "inputs")
    assert roots[0] == tmp_path / "work" and deep in roots


def test_the_grid_refuses_an_unfrozen_or_changed_design() -> None:
    digest = design_digest()
    with pytest.raises(RuntimeError, match="not frozen"):
        require_frozen_design(None)
    with pytest.raises(RuntimeError, match="differs"):
        require_frozen_design("0" * 64)
    assert require_frozen_design(digest) == digest


def test_session_plan_divides_the_work_across_devices() -> None:
    cells = manifest()
    plan = session_plan({"itr": 36.0, "rdg": 6.0, "vtr": 72.0}, cells, devices=2,
                        session_left_h=10.0, usable_session_h=10.75, weekly_left_h=30.0)
    assert "wall-clock on 2 device(s): 4.75 h" in plan


def _features() -> pl.DataFrame:
    start = int(datetime(2018, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end = int(datetime(2020, 7, 1, tzinfo=timezone.utc).timestamp() * 1000)
    ts = np.arange(start, end, 3_600_000, dtype=np.int64)
    rng = np.random.default_rng(7)
    return pl.DataFrame({"ts_ms": ts, "usable": np.ones(len(ts), bool),
                         **{name: rng.standard_normal(len(ts)) for name in VARIATE_ORDER}})


def test_one_run_of_each_model_completes_and_resumes(tmp_path: Path, monkeypatch) -> None:
    short = TrainSchedule(max_epochs=1, patience=1)
    for cls in (model_module.ITransformerConfig, model_module.VanillaConfig):
        monkeypatch.setattr(cls, "schedule", lambda self, s=short: s)
    cells = [RunCell(arm, 1, 1, 24, 42) for arm in ALL_ARMS]
    out = tmp_path / "artifacts"
    summary = execute_parallel(cells, _features(), devices=[torch.device("cpu")], out_root=out,
                               roots=[out], log=lambda msg: None)
    assert summary.completed == 3 and summary.failed == 0
    for cell in cells:
        assert is_complete(cell.run_id, out, strict=True, cfg=cell.model_config(), columns=cell.columns())
        meta = json.loads((out / "meta" / f"{cell.run_id}.json").read_text(encoding="utf-8"))
        assert meta["training_selection"]["selected"] == 11_500
    selections = {json.loads((out / "meta" / f"{c.run_id}.json").read_text(encoding="utf-8"))
                  ["training_selection"]["forecast_times_sha256"] for c in cells}
    assert len(selections) == 1, "the three models must train on the same sample"
    assert pending(cells, [out]) == []
    attached, available = resume_check(cells, [tmp_path / "fresh", out], tmp_path / "fresh")
    assert attached == available == 3

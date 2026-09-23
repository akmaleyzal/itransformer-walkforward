"""A small synthetic grid in the exact on-disk format ``write_artifacts`` produces.

Every model sees the same targets; Ridge is better than the iTransformer and the
vanilla Transformer is worse, so the sign of every C1 and C2 contrast is known.
Ridge's predictions do not depend on the seed, as for the real closed-form fit.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from itransformer_btc.config import ORIGINS, PRED_LEN, SEQ_LEN, TRAIN_WINDOW_LIMIT
from itransformer_btc.runner import RunCell, manifest

HOUR_MS = 3_600_000
SIGMA_G, MU_G = 0.01, 1e-4
#: (signal share, noise scale) per model: lower noise means lower MSE.
SKILL = {"itr": (0.10, 0.6), "rdg": (0.20, 0.3), "vtr": (0.05, 0.9)}


def _issuances(origin_index: int, block: int, per_block: int) -> np.ndarray:
    """Weekly 00:00 UTC forecast origins; some blocks lose one, as a gap would."""
    start = ORIGINS[origin_index - 1].block(block)[0]
    n = per_block - int((origin_index + block) % 3 == 0)
    return int(start.timestamp() * 1000) + np.arange(n, dtype=np.int64) * 7 * 24 * HOUR_MS


def write_grid(root: Path, cells: Iterable[RunCell] | None = None, *, per_block: int = 4) -> Path:
    """Write ``preds/`` and ``meta/`` for ``cells`` (default: the full manifest)."""
    (root / "preds").mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    for cell in manifest() if cells is None else cells:
        tag = cell.model_tag
        share, scale = SKILL[tag]
        share *= 1.0 + cell.k / 24.0
        frames = []
        for b in range(1, 7):
            ts = _issuances(cell.origin_index, b, per_block)
            truth = np.random.default_rng(cell.origin_index * 100 + b).standard_normal((len(ts), PRED_LEN))
            noise_seed = (cell.origin_index, cell.k, b, 0 if tag == "rdg" else cell.seed, tag == "vtr")
            noise = np.random.default_rng(noise_seed).standard_normal(truth.shape)
            n = len(ts)
            frames.append(pl.DataFrame({
                "block": np.full(n * PRED_LEN, b, dtype=np.int8),
                "step": np.tile(np.arange(1, PRED_LEN + 1, dtype=np.int16), n),
                "timestamp": np.repeat(ts, PRED_LEN),
                "forecast_origin": np.repeat(ts, PRED_LEN),
                "input_start": np.repeat(ts - SEQ_LEN * HOUR_MS, PRED_LEN),
                "target_timestamp": (ts[:, None] + np.arange(PRED_LEN) * HOUR_MS).reshape(-1),
                "y_true": truth.reshape(-1).astype(np.float32),
                "y_pred": (share * truth + scale * noise).reshape(-1).astype(np.float32),
            }))
        pl.concat(frames).write_parquet(root / "preds" / f"{cell.run_id}.parquet")
        meta = {
            "run_id": cell.run_id, "status": "complete", "timestamp_semantics": "forecast_origin",
            "code_sha256": "c" * 64, "input_sha256": "d" * 64,
            "origin": ORIGINS[cell.origin_index - 1].label, "block_labels": [1, 2, 3, 4, 5, 6],
            "spec": {"pred_len": PRED_LEN}, "config": {"seq_len": SEQ_LEN},
            "schedule": None if tag == "rdg" else {"max_epochs": 30},
            "sigma_g": SIGMA_G, "mu_g": MU_G, "naive_rw_z": -MU_G / SIGMA_G,
            "n_train": TRAIN_WINDOW_LIMIT, "epochs_run": 0 if tag == "rdg" else 5 + cell.seed % 3,
            "n_parameters": 1000 * cell.k, "n_allocated_parameters": 1000 * cell.k,
        }
        (root / "meta" / f"{cell.run_id}.json").write_text(json.dumps(meta), encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def full_grid(tmp_path_factory) -> Path:
    """All 900 runs of the manifest."""
    return write_grid(tmp_path_factory.mktemp("grid") / "artifacts")

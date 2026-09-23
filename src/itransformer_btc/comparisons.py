"""Table 6: every pair of models, the statistic for each, and multiplicity control.

Pairs are compared on identical actual targets. Losses are averaged over seeds
first, then over blocks with equal weight, then compared per origin; the 15
origins are the clusters. Romano-Wolf controls the family-wise error over all
pairs and within each declared family, and the Model Confidence Set is reported
at 90% and 75%. All of it is diagnostic, because origins share training data.

Upstream:
    Written here after J. P. Romano and M. Wolf, Econometrica 73(4), 2005, and
    P. R. Hansen, A. Lunde and J. M. Nason, Econometrica 79(2), 2011.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

from itransformer_btc.config import ORIGINS, PRED_LEN, TEST_BLOCKS
from itransformer_btc.metrics import (
    hln_test,
    load_meta,
    load_predictions,
    parse_run_id,
)

ModelKey = tuple[str, int]

#: Naive-RW's sentinel key.
NAIVE: Final[ModelKey] = ("naive", 0)


DEFAULT_B: Final = 9_999

MCS_LEVELS: Final[tuple[float, ...]] = (0.10, 0.25)

FAMILY_ORDER: Final[tuple[str, ...]] = (
    "vs-naive", "ladder", "cross-model", "other",
)


def pair_family(left: ModelKey, right: ModelKey) -> str:
    """The claim a pair speaks to, decided from the keys alone.

    ``vs-naive``: any pair with Naive-RW. ``ladder``: the iTransformer at two
    rungs (RQ1). ``cross-model``: the iTransformer against Ridge or the vanilla
    Transformer at the same K (C1, C2). ``other``: every remaining pair.
    """
    if NAIVE in (left, right):
        return "vs-naive"
    if left[0] == right[0] == "itr":
        return "ladder"
    if left[1] == right[1] and "itr" in (left[0], right[0]):
        return "cross-model"
    return "other"


def label(key: ModelKey) -> str:
    """``itr-K8``, or ``Naive-RW`` for the sentinel."""
    return "Naive-RW" if key == NAIVE else f"{key[0]}-K{key[1]}"


# -- the aligned prediction panel --------------------------------------------


@dataclass(frozen=True, slots=True)
class PredictionPanel:
    """Aligned forecast points with mean-seed losses as the estimand.

    y_pred stores the ensemble for inspection only. Production loss comparisons
    read seed_losses, then equally average block RelMSE and origin values."""

    keys: tuple[ModelKey, ...]
    origin_indices: tuple[int, ...]
    origins: tuple[str, ...]
    #: origin index -> ``(n_rows,)`` block labels, sorted with the arrays below.
    block: dict[int, np.ndarray]
    #: origin index -> ``(n_rows,)`` realised target, shared by every model.
    y_true: dict[int, np.ndarray]
    #: ``(key, origin index)`` -> ``(n_rows,)`` seed-averaged forecast.
    y_pred: dict[tuple[ModelKey, int], np.ndarray]
    #: Forecast steps per window, so a per-origin reduction can recover ``T``.
    pred_len: int
    seed_losses: dict[tuple[ModelKey, int], np.ndarray] | None = None


def _run_ids(
    key: ModelKey, origin_index: int, roots: list[Path], pred_len: int
) -> list[str]:
    """Every seed of one cell that is actually on disk, in seed order."""
    model, k = key
    stem = f"{model}_o{origin_index:02d}_K{k:02d}_H{pred_len:03d}_s"
    found: set[str] = set()
    for root in roots:
        for path in (root / "preds").glob(f"{stem}*.parquet"):
            found.add(path.stem)
    return sorted(found, key=lambda run_id: int(parse_run_id(run_id)["seed"]))


def available_keys(
    keys: list[ModelKey],
    roots: list[Path],
    pred_len: int = PRED_LEN,
    origin_indices: tuple[int, ...] | None = None,
) -> tuple[list[ModelKey], list[ModelKey]]:
    """Split ``keys`` into those with a run at every origin, and the rest."""
    indices = origin_indices or tuple(o.index for o in ORIGINS)
    present: list[ModelKey] = []
    absent: list[ModelKey] = []
    for key in keys:
        if key == NAIVE or any(
            _run_ids(key, index, roots, pred_len) for index in indices
        ):
            present.append(key)
        else:
            absent.append(key)
    return present, absent


def build_panel(
    keys: list[ModelKey], roots: list[Path], pred_len: int = PRED_LEN,
    origin_indices: tuple[int, ...] | None = None,
    *, windows: pl.DataFrame | None = None,
) -> PredictionPanel:
    """Stack every model's predictions on identical actual targets, with seed-averaged losses.

    Raises:
        FileNotFoundError: If a model has no run at an origin.
        ValueError: If runs disagree on targets or lack provenance.
    """
    if not any(key != NAIVE for key in keys):
        raise ValueError("a comparison panel needs a persisted forecast")
    indices = origin_indices or tuple(o.index for o in ORIGINS)
    block, y_true, y_pred, seed_losses = {}, {}, {}, {}
    vintages = set()
    for index in indices:
        signature = None
        naive_z = None
        for key in keys:
            if key == NAIVE:
                continue
            runs = _run_ids(key, index, roots, pred_len)
            if not runs:
                raise FileNotFoundError(f"{key} has no run at origin {index} (H={pred_len})")
            stacked = []
            for run_id in runs:
                meta = load_meta(run_id, roots)
                vintage = (meta.get("input_sha256"), meta.get("code_sha256"))
                if any(not v or v == "unknown" for v in vintage):
                    raise ValueError(f"{run_id}: missing analysis provenance")
                vintages.add(vintage)
                frame = load_predictions(run_id, roots)
                if windows is not None:
                    keep = windows.filter((pl.col("origin_index") == index) & (pl.col("pred_len") == pred_len))
                    frame = frame.join(keep.select("block", "timestamp"), on=["block", "timestamp"], how="semi").sort(["block", "timestamp", "step"])
                sig = frame.select("block", "timestamp", "step", "target_timestamp")
                actual = frame["y_true"].to_numpy().astype(np.float64)
                if signature is None:
                    signature = sig
                    block[index] = frame["block"].to_numpy()
                    y_true[index] = actual
                    naive_z = float(meta["naive_rw_z"])
                elif not sig.equals(signature):
                    raise ValueError(f"{run_id}: evaluated window sets differ at origin {index}")
                elif not np.allclose(actual, y_true[index], rtol=1e-6, atol=1e-8):
                    raise ValueError(f"{run_id}: target values or scaler differ")
                stacked.append(frame["y_pred"].to_numpy().astype(np.float64))
            y_pred[key, index] = np.mean(stacked, axis=0)
            seed_losses[key, index] = np.mean(
                np.square(y_true[index][None, :] - np.stack(stacked)), axis=0
            )
        y_pred[NAIVE, index] = np.full(len(y_true[index]), naive_z)
        seed_losses[NAIVE, index] = np.square(y_true[index] - naive_z)
    if len(vintages) != 1:
        raise ValueError("comparison panel mixes code or input vintages")
    return PredictionPanel(
        keys=tuple(keys), origin_indices=tuple(indices),
        origins=tuple(ORIGINS[i-1].label for i in indices), block=block,
        y_true=y_true, y_pred=y_pred, pred_len=pred_len, seed_losses=seed_losses,
    )


def _per_window(values: np.ndarray, pred_len: int) -> np.ndarray:
    """Mean over the forecast steps of each window."""
    return values.reshape(-1, pred_len).mean(axis=1)


def differential(panel: PredictionPanel, left: ModelKey, right: ModelKey,
                 origin_index: int) -> np.ndarray:
    """Per-forecast loss differential, after averaging each model's loss across seeds."""
    def loss(key: ModelKey) -> np.ndarray:
        if panel.seed_losses is not None:
            return panel.seed_losses[key, origin_index]
        return np.square(panel.y_true[origin_index] - panel.y_pred[key, origin_index])
    return _per_window(loss(left) - loss(right), panel.pred_len)


def per_origin_differential(panel: PredictionPanel, left: ModelKey,
                            right: ModelKey) -> np.ndarray:
    """Per-origin loss differential with equal block weights."""
    return per_origin_loss(panel, left) - per_origin_loss(panel, right)


def per_origin_loss(panel: PredictionPanel, key: ModelKey) -> np.ndarray:
    """Per-origin RelMSE with equal block weights, from seed-averaged step losses."""
    means = []
    for index in panel.origin_indices:
        loss = (panel.seed_losses[key, index] if panel.seed_losses is not None else
                np.square(panel.y_true[index] - panel.y_pred[key, index]))
        values = []
        for b in np.unique(panel.block[index]):
            mask = panel.block[index] == b
            denominator = (panel.seed_losses[NAIVE, index][mask].mean()
                           if panel.seed_losses is not None else 1.0)
            if denominator <= 0:
                raise ValueError("Naive-RW block MSE must be positive")
            values.append(loss[mask].mean() / denominator)
        means.append(np.mean(values))
    return np.asarray(means)


# -- clustered inference over origins ----------------------------------------


def _studentised(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column means and their cluster standard errors, ``G = matrix.shape[0]``."""
    g = matrix.shape[0]
    return matrix.mean(axis=0), matrix.std(axis=0, ddof=1) / math.sqrt(g)


def cluster_bootstrap_t(
    per_origin: np.ndarray, B: int = DEFAULT_B, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Observed and bootstrap studentised statistics, resampling **origins**.

    Args:
        per_origin: ``(G, P)`` -- one mean differential per origin, per pair.
        B: Bootstrap draws.
        seed: Generator seed.

    Returns:
        ``(t_obs, t_boot)`` of shapes ``(P,)`` and ``(B, P)``. The bootstrap
        statistics are centred on the observed mean, so they are draws from the
        null. A resample that happens to pick one origin ``G`` times has no
        dispersion; it contributes 0 rather than an infinity.
    """
    g = per_origin.shape[0]
    theta, se = _studentised(per_origin)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_obs = np.where(se > 0, theta / se, 0.0)

    rng = np.random.default_rng(seed)
    draws = per_origin[rng.integers(0, g, size=(B, g))]
    theta_b = draws.mean(axis=1)
    se_b = draws.std(axis=1, ddof=1) / math.sqrt(g)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_boot = np.where(se_b > 0, (theta_b - theta) / se_b, 0.0)
    return t_obs, t_boot


def romano_wolf(
    per_origin: np.ndarray, B: int = DEFAULT_B, seed: int = 42
) -> np.ndarray:
    """Stepdown FWER-controlled p-values across every pair (Romano & Wolf 2005).

    Two-sided: the family asks whether two models differ in predictive ability
    at all, in either direction.

    Args:
        per_origin: ``(G, P)`` mean differential per origin, per pair.
        B: Bootstrap draws.
        seed: Generator seed.

    Returns:
        ``(P,)`` adjusted p-values, monotone in ``|t|``.
    """
    t_obs, t_boot = cluster_bootstrap_t(per_origin, B=B, seed=seed)
    order = list(np.argsort(-np.abs(t_obs)))
    adjusted = np.empty(per_origin.shape[1])
    remaining = list(order)
    running = 0.0
    for position in order:
        block_max = np.abs(t_boot[:, remaining]).max(axis=1)
        raw = (1 + int((block_max >= abs(t_obs[position])).sum())) / (1 + B)
        running = max(running, raw)  # stepdown monotonicity
        adjusted[position] = min(running, 1.0)
        remaining.remove(position)
    return adjusted


def model_confidence_set(
    losses: np.ndarray, alpha: float, B: int = DEFAULT_B, seed: int = 42
) -> list[int]:
    """Model Confidence Set at level ``alpha`` by the ``T_max`` statistic (Hansen, Lunde and Nason, 2011)."""
    g, m = losses.shape
    alive = list(range(m))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, g, size=(B, g))

    while len(alive) > 1:
        sub = losses[:, alive]
        deviation = sub - sub.mean(axis=1, keepdims=True)
        theta, se = _studentised(deviation)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(se > 0, theta / se, 0.0)
        t_max = float(t.max())

        draws = deviation[idx]
        theta_b = draws.mean(axis=1)
        se_b = draws.std(axis=1, ddof=1) / math.sqrt(g)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_b = np.where(se_b > 0, (theta_b - theta) / se_b, 0.0)
        p = (1 + int((t_b.max(axis=1) >= t_max).sum())) / (1 + B)

        if p >= alpha:
            break
        alive.pop(int(np.argmax(t)))  # eliminate the worst, then re-test
    return alive


# -- Table 6 -----------------------------------------------------------------


def _cell_diagnostics(
    panel: PredictionPanel, left: ModelKey, right: ModelKey
) -> dict[str, float | int | bool]:
    """Median HLN statistic over the (origin, block) cells, and how many reject."""
    stats: list[float] = []
    rejects = 0
    t_min = -1
    fallback = False
    name = f"{label(left)} vs {label(right)}"
    for index in panel.origin_indices:
        d = differential(panel, left, right, index)
        blocks = _per_window(
            panel.block[index].astype(np.float64), panel.pred_len
        ).round()
        for b in range(1, TEST_BLOCKS + 1):
            cell = d[blocks == float(b)]
            if len(cell) < 2:
                continue
            result = hln_test(cell, panel.pred_len, name=name)
            stats.append(result.statistic)
            rejects += int(result.p_value < 0.05)
            t_min = result.T if t_min < 0 else min(t_min, result.T)
            fallback = fallback or result.fallback_fired
    return {
        "s_star_median": float(np.median(stats)) if stats else float("nan"),
        "n_cells": len(stats),
        "n_cells_reject": rejects,
        "T_min": t_min,
        "fallback_fired": fallback,
    }


def pair_matrix(
    panel: PredictionPanel, B: int = DEFAULT_B, seed: int = 42
) -> pl.DataFrame:
    """Table 6: every unordered pair with its statistic, raw and Romano-Wolf p-values, and MCS flags."""
    keys = list(panel.keys)
    # No fitted pair is treated as nested, so every pair gets the same two-sided
    # unadjusted statistic; Clark-West is kept for comparisons with Naive-RW.
    pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1 :]]

    per_origin = np.column_stack(
        [per_origin_differential(panel, a, b) for a, b in pairs]
    )
    t_obs, t_boot = cluster_bootstrap_t(per_origin, B=B, seed=seed)
    p_adjusted = romano_wolf(per_origin, B=B, seed=seed)

    families = [pair_family(a, b) for a, b in pairs]
    p_family = np.ones(len(pairs))
    for name in FAMILY_ORDER:
        members = [i for i, f in enumerate(families) if f == name]
        if not members:
            continue
        p_family[members] = romano_wolf(per_origin[:, members], B=B, seed=seed)

    losses = np.column_stack([per_origin_loss(panel, k) for k in keys])
    members = {
        alpha: {keys[i] for i in model_confidence_set(losses, alpha, B=B, seed=seed)}
        for alpha in MCS_LEVELS
    }

    rows = []
    for position, (left, right) in enumerate(pairs):
        t = float(t_obs[position])
        count = int((np.abs(t_boot[:, position]) >= abs(t)).sum())
        rows.append(
            {
                "left": label(left),
                "right": label(right),
                "statistic_name": "unadjusted forecast-loss diagnostic",
                "inference_status": "exploratory; cross-origin dependence unresolved",
                "estimand": "mean seed loss, then equal block means",
                "t_cluster": t,
                "p_raw": (1 + count) / (1 + B),
                "p_romano_wolf": float(p_adjusted[position]),
                "family": families[position],
                "p_romano_wolf_family": float(p_family[position]),
                **_cell_diagnostics(panel, left, right),
                "h": panel.pred_len,
                "G": int(per_origin.shape[0]),
                "left_in_mcs_90": left in members[0.10],
                "right_in_mcs_90": right in members[0.10],
                "left_in_mcs_75": left in members[0.25],
                "right_in_mcs_75": right in members[0.25],
            }
        )
    return pl.DataFrame(rows)


def mcs_table(
    panel: PredictionPanel, B: int = DEFAULT_B, seed: int = 42
) -> pl.DataFrame:
    """MCS membership per model, with its mean loss and rank."""
    keys = list(panel.keys)
    losses = np.column_stack([per_origin_loss(panel, k) for k in keys])
    members = {
        alpha: {keys[i] for i in model_confidence_set(losses, alpha, B=B, seed=seed)}
        for alpha in MCS_LEVELS
    }
    mean_loss = losses.mean(axis=0)
    rank = {int(position): r + 1 for r, position in enumerate(np.argsort(mean_loss))}
    return pl.DataFrame(
        [
            {
                "model": label(key),
                "mean_loss": float(mean_loss[i]),
                "se_across_origins": float(
                    losses[:, i].std(ddof=1) / math.sqrt(losses.shape[0])
                ),
                "rank": rank[i],
                "in_mcs_90": key in members[0.10],
                "in_mcs_75": key in members[0.25],
            }
            for i, key in enumerate(keys)
        ]
    ).sort("rank")

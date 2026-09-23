"""Metrics, tests and the RQ estimators, computed from saved predictions only.

Everything here reads ``preds/{run_id}.parquet`` and ``meta/{run_id}.json``; no
model or tensor is touched, so every number can be regenerated from the files.

- Naive-RW is ``y_z = -mu_g / sigma_g`` in scaler space (a zero raw return).
- Ratios are formed from seed-averaged MSEs, never averaged across seeds.
- Blocks are assigned by forecast origin, the first target hour.
- Diebold-Mariano uses the Harvey-Leybourne-Newbold correction and a
  rectangular long-run variance at lag ``h - 1``; Clark-West is used only
  against Naive-RW.
- All inference is diagnostic: origins overlap in their training data.

Upstream:
    Written here on numpy after F. X. Diebold and R. S. Mariano, J. Bus. Econ.
    Statist. 13(3), 1995; D. Harvey, S. Leybourne and P. Newbold, Int. J.
    Forecast. 13(2), 1997; T. E. Clark and K. D. West, J. Econometrics 138(1),
    2007; M. H. Pesaran and A. Timmermann, J. Bus. Econ. Statist. 10(4), 1992;
    A. C. Cameron, J. B. Gelbach and D. L. Miller, Rev. Econ. Statist. 90(3),
    2008; J. G. MacKinnon, M. O. Nielsen and M. D. Webb, J. Econometrics 232(2),
    2023.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from itransformer_btc.config import BLOCK_HOURS, PRED_LEN

RUN_ID_PATTERN = re.compile(
    r"^(?P<model>[a-z0-9]+)_o(?P<origin>\d{2})_K(?P<k>\d{2})"
    r"_H(?P<h>\d{3})_s(?P<seed>\d+)$"
)

HOUR_MS = 3_600_000


# -- artifact I/O ------------------------------------------------------------


def parse_run_id(run_id: str) -> dict[str, int | str]:
    """Split a ``run_id`` into model, origin index, K, horizon and seed."""
    match = RUN_ID_PATTERN.match(run_id)
    if match is None:
        raise ValueError(f"{run_id!r} is not a {{model}}_o{{origin}}_K{{K}}_H{{H}}_s{{seed}} run_id")
    g = match.groupdict()
    return {
        "model": g["model"],
        "origin_index": int(g["origin"]),
        "k": int(g["k"]),
        "pred_len": int(g["h"]),
        "seed": int(g["seed"]),
    }


def _locate(run_id: str, roots: list[Path], kind: str, suffix: str) -> Path:
    for root in roots:
        candidate = Path(root) / kind / f"{run_id}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{kind}/{run_id}{suffix} in none of {[str(r) for r in roots]}"
    )


def load_predictions(run_id: str, roots: list[Path]) -> pl.DataFrame:
    """Read one run's predictions, with UTC target times and blocks by forecast origin."""
    path = _locate(run_id, roots, "preds", ".parquet")
    meta_path = path.parent.parent / "meta" / f"{run_id}.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frame = pl.read_parquet(path)
    length = int(meta["config"]["seq_len"])
    horizon = int(meta["spec"]["pred_len"])
    if length < 1 or horizon < 1:
        raise ValueError(f"{run_id}: invalid lookback or horizon")
    if meta.get("timestamp_semantics") != "forecast_origin":
        raise ValueError(f"{run_id}: unknown timestamp semantics")
    if not {"input_start", "forecast_origin", "target_timestamp"} <= set(frame.columns):
        raise ValueError(f"{run_id}: incomplete timestamp schema")
    if frame.filter(
        (pl.col("timestamp") != pl.col("forecast_origin")) |
        (pl.col("forecast_origin") - pl.col("input_start") != length * HOUR_MS) |
        (pl.col("target_timestamp") != pl.col("timestamp") +
         (pl.col("step").cast(pl.Int64) - 1) * HOUR_MS)
    ).height:
        raise ValueError(f"{run_id}: inconsistent target timestamps")
    required = ["block", "timestamp", "step", "forecast_origin", "input_start", "target_timestamp", "y_true", "y_pred"]
    if frame.is_empty() or any(frame[n].null_count() for n in required):
        raise ValueError(f"{run_id}: empty predictions or null prediction keys")
    counts = frame.group_by("timestamp").agg(
        pl.len().alias("n"), pl.col("step").n_unique().alias("unique"),
        pl.col("step").min().alias("first"), pl.col("step").max().alias("last"),
    )
    if counts.filter((pl.col("n") != horizon) | (pl.col("unique") != horizon) |
                     (pl.col("first") != 1) | (pl.col("last") != horizon)).height:
        raise ValueError(f"{run_id}: incomplete or duplicated forecast horizon")
    if frame.select(pl.any_horizontal(pl.col("y_true", "y_pred").is_null() |
                                     ~pl.col("y_true", "y_pred").is_finite()).any()).item():
        raise ValueError(f"{run_id}: non-finite predictions")
    return frame.sort(["block", "timestamp", "step"])


def load_meta(run_id: str, roots: list[Path]) -> dict:
    """Read metadata paired with the selected prediction file, never another root."""
    path = _locate(run_id, roots, "preds", ".parquet").parent.parent / "meta" / f"{run_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


# -- point metrics -----------------------------------------------------------


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(y_true) - np.asarray(y_pred))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rel_mse(model: float, naive: float) -> float:
    """``MSE_model / MSE_naive`` on the same windows, which controls for period difficulty."""
    return model / naive


def r2_oos(model: float, naive: float) -> float:
    """``1 - RelMSE``: positive means the model beats Naive-RW."""
    return 1.0 - rel_mse(model, naive)


def raw_rmse(mse_z: float, sigma_g: float) -> float:
    """RMSE back in raw log-return units: ``sqrt(mse_z) * sigma_g``."""
    return math.sqrt(mse_z) * sigma_g


def non_overlapping_mask(timestamps: np.ndarray) -> np.ndarray:
    """Issuances whose forecast period opens at 00:00 UTC: one per day, none overlapping."""
    return (np.asarray(timestamps) // HOUR_MS) % 24 == 0


def pesaran_timmermann(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    """Pesaran-Timmermann (1992) test of directional predictability.

    Returns:
        ``(statistic, one-sided p)`` against ``N(0,1)``. Without a null
        hypothesis, directional accuracy is a descriptive number; this supplies
        the null.

    Zero targets are excluded rather than assigned a direction: a zero
    log-return has no sign to predict, and assigning one would inflate the hit
    rate by whatever the model happened to output there.
    """
    a = np.sign(np.asarray(actual, dtype=np.float64))
    f = np.sign(np.asarray(predicted, dtype=np.float64))
    keep = a != 0
    a, f = a[keep], f[keep]
    n = len(a)
    if n < 2:
        return float("nan"), float("nan")

    hit = float(np.mean(a == f))
    py = float(np.mean(a > 0))
    px = float(np.mean(f > 0))
    p_star = py * px + (1 - py) * (1 - px)

    var_hit = p_star * (1 - p_star) / n
    var_star = (
        (2 * py - 1) ** 2 * px * (1 - px)
        + (2 * px - 1) ** 2 * py * (1 - py)
        + 4 * py * px * (1 - py) * (1 - px) / n
    ) / n
    denom = var_hit - var_star
    if denom <= 0:
        return float("nan"), float("nan")

    stat = (hit - p_star) / math.sqrt(denom)
    return stat, 0.5 * math.erfc(stat / math.sqrt(2.0))


def _hit_rate(actual: np.ndarray, predicted: np.ndarray) -> float:
    keep = np.sign(actual) != 0
    if not keep.any():
        return float("nan")
    return float(np.mean(np.sign(actual[keep]) == np.sign(predicted[keep])))


# -- per-block tables --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DirectionalAccuracy:
    """Hit rates of the three directional variants on raw returns.

    ``up_*`` is the share of positive actual moves, from which the per-origin
    majority baseline ``max(up, 1 - up)`` is formed. ``p_*`` are Pesaran-Timmermann
    p-values, reported as a diagnostic only.
    """

    da_1h: float
    da_24h: float
    da_cum: float
    up_1h: float
    up_24h: float
    up_cum: float
    p_1h: float
    p_24h: float
    p_cum: float
    n_1h: int
    n_daily: int


def _up_share(actual: np.ndarray) -> float:
    moved = np.sign(actual) != 0
    return float(np.mean(actual[moved] > 0)) if moved.any() else float("nan")


def directional_accuracy(frame: pl.DataFrame, *, sigma_g: float, mu_g: float) -> DirectionalAccuracy:
    """DA-1h, DA-24h and DA-cum of one run, on raw returns ``r = z sigma_g + mu_g``.

    DA-1h scores the first step at every issuance hour. DA-24h scores the last
    step and DA-cum the sign of the summed 24-hour return, both at one issuance a
    day (00:00 UTC), so no two scored windows overlap. A zero actual return is
    excluded and a zero forecast counts as wrong.
    """
    if not np.isfinite(sigma_g) or sigma_g <= 0 or not np.isfinite(mu_g):
        raise ValueError("directional accuracy requires a finite mean and positive scale")
    frame = frame.with_columns(
        (pl.col("y_true") * sigma_g + mu_g).alias("y_true"),
        (pl.col("y_pred") * sigma_g + mu_g).alias("y_pred"),
    )
    last_step = int(frame.get_column("step").max())
    step1 = frame.filter(pl.col("step") == 1)
    a1, f1 = step1.get_column("y_true").to_numpy(), step1.get_column("y_pred").to_numpy()
    last = frame.filter(pl.col("step") == last_step)
    keep_h = non_overlapping_mask(last.get_column("timestamp").to_numpy())
    a_h = last.get_column("y_true").to_numpy()[keep_h]
    f_h = last.get_column("y_pred").to_numpy()[keep_h]
    cum = frame.group_by("timestamp").agg(pl.col("y_true").sum(), pl.col("y_pred").sum()).sort("timestamp")
    keep_c = non_overlapping_mask(cum.get_column("timestamp").to_numpy())
    a_c = cum.get_column("y_true").to_numpy()[keep_c]
    f_c = cum.get_column("y_pred").to_numpy()[keep_c]
    return DirectionalAccuracy(
        da_1h=_hit_rate(a1, f1), da_24h=_hit_rate(a_h, f_h), da_cum=_hit_rate(a_c, f_c),
        up_1h=_up_share(a1), up_24h=_up_share(a_h), up_cum=_up_share(a_c),
        p_1h=pesaran_timmermann(a1, f1)[1], p_24h=pesaran_timmermann(a_h, f_h)[1],
        p_cum=pesaran_timmermann(a_c, f_c)[1], n_1h=len(a1), n_daily=int(keep_c.sum()),
    )


DA_VARIANTS: tuple[str, ...] = ("1h", "24h", "cum")


def directional_accuracy_table(run_ids: list[str], roots: list[Path], *,
                               windows: pl.DataFrame | None = None) -> pl.DataFrame:
    """One row per run: the three hit rates, the majority baselines and Pesaran-Timmermann p.

    With ``windows`` (from :func:`evaluation_windows`) every run is scored on the
    common calendar, so the baseline is the same for every model and K at an origin.
    """
    rows = []
    for run_id in run_ids:
        parts = parse_run_id(run_id)
        meta = load_meta(run_id, roots)
        frame = load_predictions(run_id, roots)
        if windows is not None:
            keep = windows.filter((pl.col("origin_index") == parts["origin_index"]) &
                                  (pl.col("pred_len") == parts["pred_len"]))
            frame = frame.join(keep.select("block", "timestamp"), on=["block", "timestamp"], how="semi")
        da = directional_accuracy(frame, sigma_g=float(meta["sigma_g"]), mu_g=float(meta["mu_g"]))
        row = {"run_id": run_id, "model": str(parts["model"]), "origin": str(meta["origin"]),
               "origin_index": int(parts["origin_index"]), "k": int(parts["k"]),
               "seed": int(parts["seed"]), "n_1h": da.n_1h, "n_daily": da.n_daily}
        for v in DA_VARIANTS:
            up = getattr(da, f"up_{v}")
            row[f"da_{v}"] = getattr(da, f"da_{v}")
            row[f"base_{v}"] = max(up, 1.0 - up)
            row[f"p_{v}"] = getattr(da, f"p_{v}")
        rows.append(row)
    return pl.DataFrame(rows)


def directional_accuracy_summary(table: pl.DataFrame) -> pl.DataFrame:
    """Per (model, K): mean and SE across origins of seed-averaged ``DA - baseline``.

    Also the number of origins where DA beats the baseline and, as a diagnostic,
    the number of runs with Pesaran-Timmermann p < 0.05.
    """
    per_origin = table.group_by("model", "k", "origin_index").agg(
        *[(pl.col(f"da_{v}") - pl.col(f"base_{v}")).mean().alias(f"dda_{v}") for v in DA_VARIANTS],
        *[(pl.col(f"p_{v}") < 0.05).sum().alias(f"pt_{v}") for v in DA_VARIANTS],
        pl.len().alias("runs"),
    )
    return per_origin.group_by("model", "k").agg(
        pl.col("origin_index").n_unique().alias("n_origins"),
        pl.col("runs").sum(),
        *[pl.col(f"dda_{v}").mean().alias(f"dda_{v}") for v in DA_VARIANTS],
        *[(pl.col(f"dda_{v}").std(ddof=1) / pl.col(f"dda_{v}").count().sqrt()).alias(f"dda_{v}_se")
          for v in DA_VARIANTS],
        *[(pl.col(f"dda_{v}") > 0).sum().alias(f"wins_{v}") for v in DA_VARIANTS],
        *[pl.col(f"pt_{v}").sum().alias(f"pt_{v}") for v in DA_VARIANTS],
    ).sort("model", "k")


def assert_same_windows(left: pl.DataFrame, right: pl.DataFrame, what: str) -> None:
    """Raise unless two runs share every forecast origin, step and actual target time."""
    columns = ["block", "timestamp", "step"]
    if "target_timestamp" in left.columns or "target_timestamp" in right.columns:
        if not all("target_timestamp" in f.columns for f in (left, right)):
            raise ValueError(f"{what}: target timestamp contract missing on one side")
        columns.append("target_timestamp")
    a, b = [f.select(columns).sort(columns) for f in (left, right)]
    if a.is_duplicated().any() or b.is_duplicated().any() or not a.equals(b):
        raise ValueError(f"{what}: evaluated window sets differ ({a.height} vs {b.height} points)")


def block_metrics(frame: pl.DataFrame, naive_z: float) -> pl.DataFrame:
    """Per-block MSE, MAE, RelMSE and ``R2_oos`` for one run."""
    return (
        frame.with_columns(
            (pl.col("y_true") - pl.col("y_pred")).pow(2).alias("_se"),
            (pl.col("y_true") - pl.col("y_pred")).abs().alias("_ae"),
            (pl.col("y_true") - naive_z).pow(2).alias("_se_naive"),
        )
        .group_by("block")
        .agg(
            pl.col("timestamp").n_unique().alias("n_windows"),
            pl.col("_se").count().alias("n_points"),
            pl.col("_se").mean().alias("mse"),
            pl.col("_ae").mean().alias("mae"),
            pl.col("_se_naive").mean().alias("mse_naive"),
        )
        .with_columns(
            (pl.col("mse") / pl.col("mse_naive")).alias("rel_mse"),
            (1.0 - pl.col("mse") / pl.col("mse_naive")).alias("r2_oos"),
        )
        .sort("block")
    )


def evaluation_windows(run_ids: list[str], roots: list[Path]) -> pl.DataFrame:
    """Common forecast times per origin, horizon and block across supplied runs.

    The intersection does not recover forecasts absent from an arm or outcomes
    absent from the data.
    """
    common = {}
    vintages = set()
    for run_id in sorted(run_ids):
        parts = parse_run_id(run_id)
        meta = load_meta(run_id, roots)
        vintage = (meta.get("input_sha256"), meta.get("code_sha256"))
        if any(not v or v == "unknown" for v in vintage):
            raise ValueError(f"{run_id}: missing analysis provenance")
        vintages.add(vintage)
        frame = load_predictions(run_id, roots)
        labels = meta.get("block_labels", [1, 2, 3, 4, 5, 6])
        for b in labels:
            key = (parts["origin_index"], parts["pred_len"], int(b))
            stamps = set(frame.filter(pl.col("block") == b)["timestamp"].unique().to_list())
            common[key] = common[key] & stamps if key in common else stamps
    if len(vintages) != 1:
        raise ValueError("analysis mixes code or input vintages")
    if any(not stamps for stamps in common.values()):
        raise ValueError("a required origin/horizon/block has no common forecast times")
    return pl.DataFrame([
        {"origin_index": i, "pred_len": h, "block": b, "timestamp": t}
        for (i, h, b), stamps in sorted(common.items()) for t in sorted(stamps)
    ])


def gather_grid(run_ids: list[str], roots: list[Path], *,
                windows: pl.DataFrame | None = None) -> pl.DataFrame:
    """Mean step errors on identical actual targets; average seeds afterwards.

    Common times and per-block hashes make downstream sample equality checkable.
    Forecast files and their original metadata are never modified.
    """
    import hashlib
    if windows is None:
        windows = evaluation_windows(run_ids, roots)
    rows = []
    raw_targets = {}
    for run_id in sorted(run_ids):
        parts = parse_run_id(run_id)
        meta = load_meta(run_id, roots)
        keep = windows.filter((pl.col("origin_index") == parts["origin_index"]) &
                              (pl.col("pred_len") == parts["pred_len"]))
        frame = load_predictions(run_id, roots).join(
            keep.select("block", "timestamp"), on=["block", "timestamp"], how="semi"
        ).sort(["block", "timestamp", "step"])
        hashes = []
        for (b,), block_frame in frame.group_by("block", maintain_order=True):
            key = (parts["origin_index"], parts["pred_len"], b)
            values = block_frame["y_true"].to_numpy().astype(np.float64) * float(meta["sigma_g"]) + float(meta["mu_g"])
            if key in raw_targets and not np.allclose(values, raw_targets[key], rtol=1e-5, atol=1e-8):
                raise ValueError(f"{run_id}: actual raw targets disagree")
            raw_targets[key] = values
            keys = block_frame.select("timestamp", "step", "target_timestamp").to_numpy().astype("<i8")
            hashes.append({"block": int(b), "evaluation_keys_sha256": hashlib.sha256(keys.tobytes()).hexdigest()})
        rows.append(block_metrics(frame, float(meta["naive_rw_z"])).join(
            pl.DataFrame(hashes), on="block"
        ).with_columns(
            pl.lit(run_id).alias("run_id"), pl.lit(str(parts["model"])).alias("model"),
            pl.lit(int(parts["origin_index"])).cast(pl.Int32).alias("origin_index"),
            pl.lit(str(meta["origin"])).alias("origin"),
            pl.lit(int(parts["k"])).cast(pl.Int32).alias("k"),
            pl.lit(int(parts["pred_len"])).cast(pl.Int32).alias("pred_len"),
            pl.lit(int(parts["seed"])).cast(pl.Int32).alias("seed"),
            pl.lit(float(meta["sigma_g"])).alias("sigma_g"),
        ))
    return pl.concat(rows)


def seed_average(grid: pl.DataFrame) -> pl.DataFrame:
    """Average MSE across seeds before any ratio is formed."""
    identity = ["model", "origin_index", "origin", "k", "pred_len", "block"]
    extra = []
    if "evaluation_keys_sha256" in grid.columns:
        if grid.group_by(identity).agg(pl.col("evaluation_keys_sha256").n_unique().alias("n")).filter(pl.col("n") != 1).height:
            raise ValueError("seeds were evaluated on different target calendars")
        extra = [pl.col("evaluation_keys_sha256").first()]
    return (
        grid.group_by(identity)
        .agg(
            pl.col("mse").mean().alias("mse"),
            pl.col("mae").mean().alias("mae"),
            pl.col("mse_naive").mean().alias("mse_naive"),
            pl.col("mse").std().alias("mse_seed_std"),
            pl.col("n_windows").first().alias("n_windows"),
            pl.col("sigma_g").first().alias("sigma_g"),
            pl.col("mse").count().alias("n_seeds"),
            *extra,
        )
        .with_columns(
            (pl.col("mse") / pl.col("mse_naive")).alias("rel_mse"),
            (1.0 - pl.col("mse") / pl.col("mse_naive")).alias("r2_oos"),
        )
        .sort(["model", "origin_index", "k", "pred_len", "block"])
    )


# -- RQ2: the K=1 against K=8 gap -------------------------------------------


def amplification(
    seed_avg: pl.DataFrame,
    k_small: int = 1,
    k_large: int = 8,
    model: str = "itr",
    pred_len: int = 24,
) -> pl.DataFrame:
    """``A(i,b) = (MSE_K1 - MSE_K8) / MSE_K1`` per origin and block: RQ2's outcome."""
    base = seed_avg.filter(
        (pl.col("model") == model) & (pl.col("pred_len") == pred_len)
    )
    small = (
        base.filter(pl.col("k") == k_small)
        .select(["origin_index", "origin", "block", "mse", "n_windows"])
        .rename({"mse": "mse_small", "n_windows": "n_small"})
    )
    large = (
        base.filter(pl.col("k") == k_large)
        .select(["origin_index", "block", "mse", "n_windows"])
        .rename({"mse": "mse_large", "n_windows": "n_large"})
    )

    joined = small.join(large, on=["origin_index", "block"], how="inner")
    if joined.height != small.height:
        raise ValueError(
            f"K={k_small} has {small.height} cells but only {joined.height} "
            f"matched K={k_large}; the panel must be balanced before beta1"
        )
    mismatched = joined.filter(pl.col("n_small") != pl.col("n_large"))
    if mismatched.height:
        raise ValueError(
            f"{mismatched.height} cells evaluate K={k_small} and "
            f"K={k_large} on different window counts; A would be a ratio across "
            f"two samples"
        )
    return joined.with_columns(
        ((pl.col("mse_small") - pl.col("mse_large")) / pl.col("mse_small")).alias("A")
    ).sort(["origin_index", "block"])


# -- forecast-loss tests: Diebold-Mariano, HLN, Clark-West --------------------


def _normal_quantile(p: float) -> float:
    """Acklam's inverse normal CDF — avoids a scipy import for one number."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q, r = p - 0.5, (p - 0.5) ** 2
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _rectangular_lrv(d: np.ndarray, h: int) -> float:
    """``(gamma_0 + 2 sum_{k=1}^{h-1} gamma_k) / T``, the variance of the mean differential."""
    t = len(d)
    dm = d - d.mean()
    total = float(dm @ dm) / t
    for k in range(1, min(h, t)):
        total += 2.0 * float(dm[k:] @ dm[:-k]) / t
    return total / t


def _bartlett_lrv(d: np.ndarray, h: int) -> float:
    """Bartlett-weighted long-run variance, used only if the rectangular one is not positive."""
    t = len(d)
    dm = d - d.mean()
    total = float(dm @ dm) / t
    for k in range(1, min(h, t)):
        total += 2.0 * (1.0 - k / h) * float(dm[k:] @ dm[:-k]) / t
    return total / t


@dataclass(frozen=True, slots=True)
class TestResult:
    """One forecast-comparison test, carrying everything needed to redo it."""

    name: str
    statistic: float
    p_value: float
    T: int
    h: int
    one_sided: bool
    fallback_fired: bool

    def __str__(self) -> str:
        tail = "  [Bartlett fallback fired]" if self.fallback_fired else ""
        side = "one-sided" if self.one_sided else "two-sided"
        return (
            f"{self.name}: S*={self.statistic:+.4f}  p={self.p_value:.4g} "
            f"({side})  T={self.T}  h={self.h}{tail}"
        )


def _upper_tail(stat: float, df: int) -> float:
    """``P(T_df > stat)`` — Student-t, falling back to the normal without scipy."""
    try:
        from scipy import stats as _stats

        return float(_stats.t.sf(stat, df=df))
    except ImportError:  # pragma: no cover - scipy ships with the Kaggle image
        return 0.5 * math.erfc(stat / math.sqrt(2.0))


def _hln_and_p(d: np.ndarray, h: int, name: str, one_sided: bool) -> TestResult:
    """Harvey-Leybourne-Newbold correction, referred to ``t(T-1)``.

    ``S* = S sqrt[(T + 1 - 2h + h(h-1)/T) / T]``, compared against Student-t
    with ``T-1`` degrees of freedom — **not** the standard normal. The factor is
    asserted positive before use: at ``h = 24`` it is exactly 0 at ``T = 24`` and
    0.047 at ``T = 30``, precisely the T a non-overlapping 30-day block would
    produce, so a silent negative would yield a complex statistic reported as a
    real one.
    """
    d = np.asarray(d, dtype=np.float64)
    t = len(d)
    if t < 2:
        raise ValueError(f"{name}: T={t} is too small for a loss differential")
    factor = (t + 1 - 2 * h + h * (h - 1) / t) / t
    if factor <= 0:
        raise ValueError(
            f"{name}: the HLN factor is {factor:.4f} <= 0 at T={t}, h={h}. "
            f"The test is not reported where the factor fails; state T instead."
        )

    variance = _rectangular_lrv(d, h)
    fallback = False
    if variance <= 0:
        variance = _bartlett_lrv(d, h)
        fallback = True
        if variance <= 0:
            raise ValueError(f"{name}: no positive long-run variance at T={t}")

    stat = float(d.mean() / math.sqrt(variance) * math.sqrt(factor))
    upper = _upper_tail(abs(stat), t - 1)
    if one_sided:
        p = upper if stat >= 0 else 1.0 - upper
    else:
        p = 2.0 * upper
    return TestResult(name, stat, float(min(p, 1.0)), t, h, one_sided, fallback)


def hln_test(
    d: np.ndarray, h: int, name: str = "HLN", one_sided: bool = False
) -> TestResult:
    """Harvey-Leybourne-Newbold test on a loss differential ``d`` at horizon ``h``.

    The statistic uses the rectangular long-run variance, falling back to Bartlett
    (and saying so) if it is not positive, and is compared with ``t(T-1)``.
    """
    return _hln_and_p(np.asarray(d, dtype=np.float64), h, name, one_sided)


def dm_test(
    loss_a: np.ndarray, loss_b: np.ndarray, h: int, name: str = "DM"
) -> TestResult:
    """Diebold-Mariano on the loss differential ``loss_a - loss_b``, two-sided.

    Against Naive-RW, which every model nests, use :func:`clark_west_test`.
    """
    return _hln_and_p(
        np.asarray(loss_a) - np.asarray(loss_b), h, name, one_sided=False
    )


def clark_west_test(
    y: np.ndarray,
    pred_small: np.ndarray,
    pred_large: np.ndarray,
    h: int,
    name: str = "Clark-West",
) -> TestResult:
    """Clark-West (2007) for a nested pair; here only a model against Naive-RW.

    ``f_t = (y - y_small)^2 - (y - y_large)^2 + (y_small - y_large)^2``

    The third term removes the larger model's estimation noise, which under the
    null would make it look worse. One-sided: the alternative is that the larger
    model helps. A diagnostic, and it can be positive beside a negative R2_oos.
    """
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(pred_small, dtype=np.float64)
    lg = np.asarray(pred_large, dtype=np.float64)
    f = np.square(y - s) - np.square(y - lg) + np.square(s - lg)
    return _hln_and_p(f, h, name, one_sided=True)


def per_origin_loss(frame: pl.DataFrame) -> pl.DataFrame:
    """Mean squared error per forecast origin: the series the DM test consumes."""
    return (
        frame.with_columns((pl.col("y_true") - pl.col("y_pred")).pow(2).alias("_se"))
        .group_by(["block", "timestamp"])
        .agg(pl.col("_se").mean().alias("loss"))
        .sort(["block", "timestamp"])
    )


# -- RQ2's core regression: beta1 with origin FE and a wild cluster bootstrap -


@dataclass(frozen=True, slots=True)
class Beta1Result:
    """``A(i,b) = alpha_i + beta1 b + eps`` with origin-clustered inference."""

    beta1: float
    t_statistic: float
    cluster_se: float
    p_rademacher: float
    p_webb: float
    n_clusters: int
    n_observations: int
    within_slopes: np.ndarray
    B: int

    @property
    def headline_p(self) -> float:
        """The more conservative of the Rademacher and Webb bootstrap p-values."""
        return max(self.p_rademacher, self.p_webb)

    def __str__(self) -> str:
        return (
            f"beta1 = {self.beta1:+.6f}   t = {self.t_statistic:+.3f}   "
            f"G = {self.n_clusters}   N = {self.n_observations}\n"
            f"WCR one-sided p (H1: beta1 < 0): Rademacher {self.p_rademacher:.4f}, "
            f"Webb {self.p_webb:.4f}  ->  headline {self.headline_p:.4f}\n"
            f"Effective independence is bounded near 4 by the training-window "
            f"overlap, well below G = {self.n_clusters}."
        )


def _weights(kind: str, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    if kind == "rademacher":
        return rng.choice(np.array([-1.0, 1.0]), size=shape)
    if kind == "webb":
        # Webb's 6-point distribution. At G = 15 Rademacher already admits
        # 2^15 = 32,768 distinct draws, a minimum two-sided p of about 6e-5, so
        # the original small-G justification for preferring Webb no longer
        # binds — both are reported and the more conservative is the headline.
        atoms = np.array([
            -math.sqrt(1.5), -1.0, -math.sqrt(0.5),
            math.sqrt(0.5), 1.0, math.sqrt(1.5),
        ])
        return rng.choice(atoms, size=shape)
    raise ValueError(f"unknown weight scheme {kind!r}")


def _balanced_matrix(panel: pl.DataFrame, value: str) -> tuple[np.ndarray, np.ndarray]:
    """``(G x B)`` outcome matrix and the block axis, or a loud failure.

    Built by hand rather than with ``pivot`` so the code does not depend on which
    polars major version the Kaggle image happens to ship.
    """
    origins = sorted(set(panel.get_column("origin").to_list()))
    blocks = sorted(set(int(b) for b in panel.get_column("block").to_list()))
    if not origins:
        raise ValueError("empty panel: no origin left to estimate beta1 on")
    index ={(o, b): i for i, (o, b) in enumerate([(o, b) for o in origins for b in blocks])}

    out = np.full(len(index), np.nan)
    for origin, block, val in zip(
        panel.get_column("origin").to_list(),
        panel.get_column("block").to_list(),
        panel.get_column(value).to_list(),
    ):
        out[index[(str(origin), int(block))]] = float(val)

    matrix = out.reshape(len(origins), len(blocks))
    if np.isnan(matrix).any():
        raise ValueError(
            "unbalanced panel: beta1's reduction to the mean of within-slopes "
            "holds only when every origin carries every block"
        )
    return matrix, np.array(blocks, dtype=np.float64)


def panel_beta1(
    panel: pl.DataFrame,
    value: str = "A",
    B: int = 99_999,
    seed: int = 42,
) -> Beta1Result:
    """Fit ``A(i,b) = alpha_i + beta1 b + eps`` and test ``H1: beta1 < 0``.

    Origin fixed effects; a restricted wild cluster bootstrap of the cluster-robust t
    with Rademacher and Webb weights; ``p = (1 + count) / (1 + B)``; reference
    ``t(G - 1)``.
    """
    a, x = _balanced_matrix(panel, value)
    g, n_blocks = a.shape
    xd = x - x.mean()
    sxx = float(xd @ xd)

    within = a - a.mean(axis=1, keepdims=True)
    beta = float((within * xd).sum() / (g * sxx))
    resid = within - beta * xd
    score = resid @ xd
    variance = float((score @ score) / (g * sxx) ** 2)
    se = math.sqrt(variance) if variance > 0 else float("nan")
    t_obs = beta / se if se == se and se > 0 else float("nan")

    # Restricted residuals: with beta1 = 0 imposed the fitted value is the origin
    # mean, so u_tilde is exactly the within-origin demeaned outcome. Because
    # each row of u_tilde already sums to zero, the bootstrap origin means are
    # unchanged and the whole replication collapses to s = u_tilde @ xd.
    s = within @ xd

    def _p(kind: str) -> float:
        rng = np.random.default_rng(seed)
        weights = _weights(kind, (B, g), rng)
        beta_star = (weights @ s) / (g * sxx)
        score_star = weights * s[None, :] - beta_star[:, None] * sxx
        var_star = np.square(score_star).sum(axis=1) / (g * sxx) ** 2
        ok = var_star > 0
        t_star = beta_star[ok] / np.sqrt(var_star[ok])
        # (1 + count) / (1 + B), not count / B (Davison & Hinkley 1997): the
        # observed statistic is one of its own reference distribution, and the
        # naive form returns a literal p = 0, which is not a probability any
        # finite bootstrap can support. At B = 99,999 the floor it reports is
        # 1e-5, and at G = 15 Rademacher's own granularity bounds it at ~3e-5
        # anyway — so the floor is honest rather than conservative padding.
        below = int(np.sum(t_star <= t_obs))  # H1: beta1 < 0, left tail
        return (1.0 + below) / (1.0 + int(ok.sum()))

    return Beta1Result(
        beta1=beta,
        t_statistic=t_obs,
        cluster_se=se,
        p_rademacher=_p("rademacher"),
        p_webb=_p("webb"),
        n_clusters=g,
        n_observations=g * n_blocks,
        within_slopes=(within * xd).sum(axis=1) / sxx,
        B=B,
    )


@dataclass(frozen=True, slots=True)
class EquivalenceResult:
    """TOST verdict on a rung expected to be flat."""

    mean_delta: float
    margin: float
    p_lower: float
    p_upper: float
    n: int

    @property
    def equivalent(self) -> bool:
        return max(self.p_lower, self.p_upper) < 0.05

    def __str__(self) -> str:
        verdict = "EQUIVALENT (flat)" if self.equivalent else "NOT shown equivalent"
        return (
            f"TOST: mean delta = {self.mean_delta:+.6f}, margin = +/-{self.margin:.6f}, "
            f"p = ({self.p_lower:.4f}, {self.p_upper:.4f}), G = {self.n}  ->  {verdict}"
        )


def tost_equivalence(
    deltas: np.ndarray, margin: float, alpha: float = 0.05
) -> EquivalenceResult:
    """Two one-sided tests of ``|mean(deltas)| < margin``, RQ1's equivalence check."""
    d = np.asarray(deltas, dtype=np.float64)
    n = len(d)
    if n < 2:
        raise ValueError("TOST needs at least two clusters")
    se = float(np.std(d, ddof=1) / math.sqrt(n))
    if se <= 0:
        raise ValueError("zero dispersion across clusters; TOST is undefined")
    mean = float(d.mean())
    return EquivalenceResult(
        mean_delta=mean,
        margin=abs(margin),
        p_lower=_upper_tail((mean + abs(margin)) / se, n - 1),   # H0: mu <= -margin
        p_upper=_upper_tail(-(mean - abs(margin)) / se, n - 1),  # H0: mu >= +margin
        n=n,
    )


def j_test(
    y: np.ndarray, x_a: np.ndarray, x_b: np.ndarray, groups: np.ndarray,
    *, clusters: np.ndarray | None = None,
) -> tuple[float, float]:
    """Davidson-MacKinnon J test of K against K_eff, with CR1 covariance and ``t(G - 1)``."""
    y, x_a, x_b = [np.asarray(v, dtype=np.float64) for v in (y, x_a, x_b)]
    groups = np.asarray(groups)
    clusters = groups if clusters is None else np.asarray(clusters)
    if any(v.ndim != 1 or len(v) != len(y) for v in (y, x_a, x_b, groups, clusters)):
        raise ValueError("J-test arrays must be one-dimensional with equal lengths")
    if not all(np.all(np.isfinite(v)) for v in (y, x_a, x_b)):
        raise ValueError("J-test inputs must be finite")
    def _demean(v: np.ndarray) -> np.ndarray:
        out = v.copy()
        for group in np.unique(groups):
            mask = groups == group
            out[mask] -= v[mask].mean()
        return out
    yd, ad, bd = [_demean(v) for v in (y, x_a, x_b)]
    n, g = len(y), len(np.unique(clusters))
    dof = n - 2 - len(np.unique(groups))
    if g < 2 or dof <= 0 or not bd @ bd > 0:
        return float("nan"), float("nan")
    fitted_b = bd * float((bd @ yd) / (bd @ bd))
    design = np.column_stack([ad, fitted_b])
    if np.linalg.matrix_rank(design) < 2:
        return float("nan"), float("nan")
    coef = np.linalg.lstsq(design, yd, rcond=None)[0]
    resid = yd - design @ coef
    bread = np.linalg.pinv(design.T @ design)
    scores = np.array([design[clusters == c].T @ resid[clusters == c]
                       for c in np.unique(clusters)])
    cov = (g / (g - 1)) * ((n - 1) / dof) * bread @ (scores.T @ scores) @ bread
    se = math.sqrt(max(float(cov[1, 1]), 0.0))
    if se <= 0:
        return float("nan"), float("nan")
    statistic = float(coef[1] / se)
    return statistic, 2.0 * _upper_tail(abs(statistic), g - 1)


def minimum_detectable_beta1(
    within_slopes: np.ndarray, alpha: float = 0.05, power: float = 0.80
) -> float:
    """Plug-in sensitivity of beta1 under independent-origin assumptions.

    The caller supplies slopes. In this study these are observed TEST slopes,
    so the computed value is post-analysis sensitivity, not prospective power."""
    g = len(within_slopes)
    if g < 2:
        return float("nan")
    se = float(np.std(within_slopes, ddof=1) / math.sqrt(g))
    return -(_normal_quantile(1 - alpha) + _normal_quantile(power)) * se


def raw_scale_table(seed_avg: pl.DataFrame) -> pl.DataFrame:
    """Add RMSE in raw log-return units to a seed-averaged table."""
    return seed_avg.with_columns(
        (pl.col("mse").sqrt() * pl.col("sigma_g")).alias("rmse_raw")
    )


def per_origin_relmse(seed_avg: pl.DataFrame, model: str, k: int | None = None) -> pl.DataFrame:
    """Equal-weight block RelMSE, matching the headline and comparison panel."""
    part = seed_avg.filter((pl.col("model") == model) & (pl.col("pred_len") == PRED_LEN))
    if k is not None:
        part = part.filter(pl.col("k") == k)
    return part.group_by("origin").agg(
        pl.col("rel_mse").mean(), pl.col("n_windows").sum()
    ).sort("origin")


def paired_contrast(
    seed_avg: pl.DataFrame,
    left: tuple[str, int | None],
    right: tuple[str, int | None],
) -> dict:
    """Exploratory paired contrast of mean seed loss, equal blocks and origins.

    Positive means left is worse. Calendar hashes must agree when supplied.
    The t(G-1) interval/p-value assumes independent origins, which the overlapping
    study does not establish. Feature content and PR both change in matched-K."""
    if "evaluation_keys_sha256" in seed_avg.columns:
        parts = []
        for tag, k in (left, right):
            part = seed_avg.filter((pl.col("model") == tag) & (pl.col("pred_len") == PRED_LEN))
            if k is not None:
                part = part.filter(pl.col("k") == k)
            parts.append(part.select("origin_index", "block", "evaluation_keys_sha256"))
        check = parts[0].join(parts[1], on=["origin_index", "block"], suffix="_right")
        if check.filter(pl.col("evaluation_keys_sha256") != pl.col("evaluation_keys_sha256_right")).height:
            raise ValueError("paired contrast uses different actual targets")
    a = per_origin_relmse(seed_avg, left[0], left[1])
    b = per_origin_relmse(seed_avg, right[0], right[1])
    joined = a.join(b, on="origin", how="inner", suffix="_right").sort("origin")
    diff = (
        joined.get_column("rel_mse").to_numpy()
        - joined.get_column("rel_mse_right").to_numpy()
    )
    g = int(diff.size)
    if g == 0:
        raise ValueError(f"paired contrast {left} vs {right}: the two arms share no origin")
    label = lambda arm: arm[0] if arm[1] is None else f"{arm[0]}-K{arm[1]}"
    if g < 2:
        return {
            "left": label(left), "right": label(right), "n_origins": g,
            "mean_diff": None, "se": None, "t": None, "p_two_sided": None,
            "ci_low": None, "ci_high": None, "left_better": None,
        }

    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / math.sqrt(g))
    if se > 0:
        t_stat = mean / se
        p = 2.0 * _upper_tail(abs(t_stat), g - 1)
        half = _t_critical(g - 1) * se
    else:
        t_stat, p, half = (0.0, 1.0, 0.0) if mean == 0.0 else (math.inf, 0.0, 0.0)

    return {
        "left": label(left),
        "right": label(right),
        "mean_diff": mean,
        "se": se,
        "t": t_stat,
        "p_two_sided": float(p),
        "ci_low": mean - half,
        "ci_high": mean + half,
        "n_origins": g,
        "left_better": int((diff < 0).sum()),
        "inference_status": "exploratory; independent-origin t approximation only",
    }


def _t_critical(df: int) -> float:
    """Two-sided 5% Student-t critical value, normal fallback without scipy."""
    try:
        from scipy import stats as _stats

        return float(_stats.t.ppf(0.975, df=df))
    except ImportError:  # pragma: no cover - scipy ships with the Kaggle image
        return 1.959963984540054


def beta1_with_coverage(
    panel: pl.DataFrame,
    min_coverage: float = 0.9,
    B: int = 99_999,
    seed: int = 42,
) -> tuple[Beta1Result, Beta1Result | None]:
    """``beta1`` on the full panel, and on blocks with at least ``min_coverage`` windows kept."""
    full = panel_beta1(panel, B=B, seed=seed)
    restricted = panel.filter((pl.col("n_large") / float(BLOCK_HOURS)) >= min_coverage)
    try:
        return full, panel_beta1(restricted, B=B, seed=seed)
    except ValueError:
        # Unbalanced after the restriction. Loosening the estimator to produce a
        # number here would answer a different question than the one asked.
        return full, None


def panel_beta1_covariate(
    panel: pl.DataFrame,
    value: str = "A",
    covariate: str = "coverage",
    B: int = 99_999,
    seed: int = 42,
) -> Beta1Result:
    """``A(i,b) = alpha_i + beta1 b + beta2 c(i,b) + eps``, clustered on origin."""
    a, blocks = _balanced_matrix(panel, value)
    c, _ = _balanced_matrix(panel, covariate)
    g, n_blocks = a.shape

    within = a - a.mean(axis=1, keepdims=True)
    cw = c - c.mean(axis=1, keepdims=True)
    xw = np.tile(blocks - blocks.mean(), (g, 1))

    scc = float((cw * cw).sum())
    if scc <= 0.0:
        raise ValueError(
            "coverage has no within-origin variation, so it cannot be a "
            "covariate here; panel_beta1 is the estimator that applies"
        )

    # Frisch-Waugh: residualise the regressor of interest and the outcome on the
    # control, both already swept of origin means. beta1 and the residuals of the
    # two-regressor fit are then exactly those of the simple fit on the residuals.
    xr = xw - (float((xw * cw).sum()) / scc) * cw
    ar = within - (float((within * cw).sum()) / scc) * cw
    sxx = float((xr * xr).sum())
    if sxx <= 0.0:
        raise ValueError("the block index is collinear with coverage within origin")

    beta = float((ar * xr).sum() / sxx)
    resid = ar - beta * xr
    score = (resid * xr).sum(axis=1)
    variance = float((score * score).sum()) / sxx**2
    se = math.sqrt(variance) if variance > 0 else float("nan")
    t_obs = beta / se if se == se and se > 0 else float("nan")

    # Restricted residuals: imposing beta1 = 0 leaves alpha_i and beta2, and ar is
    # already swept of both, so u_tilde is ar itself. Per-cluster inner products
    # against the fixed regressors are all a bootstrap draw needs.
    ux = (ar * xr).sum(axis=1)
    uc = (ar * cw).sum(axis=1)
    xx = (xr * xr).sum(axis=1)
    cx = (cw * xr).sum(axis=1)

    def _p(kind: str) -> float:
        rng = np.random.default_rng(seed)
        weights = _weights(kind, (B, g), rng)
        beta_star = (weights @ ux) / sxx
        delta_star = (weights @ uc) / scc
        score_star = (
            weights * ux[None, :]
            - delta_star[:, None] * cx[None, :]
            - beta_star[:, None] * xx[None, :]
        )
        var_star = np.square(score_star).sum(axis=1) / sxx**2
        ok = var_star > 0
        t_star = beta_star[ok] / np.sqrt(var_star[ok])
        below = int(np.sum(t_star <= t_obs))  # H1: beta1 < 0, left tail
        return (1.0 + below) / (1.0 + int(ok.sum()))

    return Beta1Result(
        beta1=beta,
        t_statistic=t_obs,
        cluster_se=se,
        p_rademacher=_p("rademacher"),
        p_webb=_p("webb"),
        n_clusters=g,
        n_observations=g * n_blocks,
        within_slopes=(ar * xr).sum(axis=1) / (xr * xr).sum(axis=1),
        B=B,
    )

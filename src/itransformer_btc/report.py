"""Tables, figures and ``paper_numbers.json`` for the three-model study.

Everything is computed from saved runs (``preds/`` and ``meta/``), the input bars
and the feature frame; no table or figure is edited by hand. Dispersion is the
standard error across origins, with seed variation reported beside it. All
inference is diagnostic, because consecutive origins share training data.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

from itransformer_btc.budget import budget_table
from itransformer_btc.comparisons import (
    NAIVE,
    ModelKey,
    available_keys,
    build_panel,
    label,
    mcs_table,
    pair_matrix,
)
from itransformer_btc.config import (
    BARS_ACTUAL,
    BARS_EXPECTED,
    BLOCK_HOURS,
    DATA_END,
    DATA_START,
    GAP_BLOCKS,
    K_LADDER,
    MISSING_BARS,
    ORIGIN_SPACING_MONTHS,
    ORIGINS,
    PRED_LEN,
    TRAIN_MONTHS,
    TRAIN_WINDOW_LIMIT,
)
from itransformer_btc.efficiency import efficiency_table
from itransformer_btc.keff import (
    GATE_PR_FLOOR,
    corr_k_keff,
    gate_pr,
    keff_table,
    rolling_ols_r2,
    rolling_pr,
)
from itransformer_btc.metrics import (
    amplification,
    beta1_with_coverage,
    directional_accuracy_summary,
    directional_accuracy_table,
    evaluation_windows,
    gather_grid,
    j_test,
    load_meta,
    minimum_detectable_beta1,
    paired_contrast,
    panel_beta1,
    panel_beta1_covariate,
    parse_run_id,
    raw_scale_table,
    seed_average,
    tost_equivalence,
)
from itransformer_btc.runner import completed_run_ids, manifest
from itransformer_btc.segments import break_summary
from itransformer_btc.train import code_sha256

MODEL_TAGS: Final[tuple[str, ...]] = ("itr", "rdg", "vtr")
MODEL_NAMES: Final[dict[str, str]] = {"itr": "iTransformer", "rdg": "Ridge", "vtr": "Transformer"}
COMPARISON_KEYS: Final[tuple[ModelKey, ...]] = (
    *((tag, k) for tag in MODEL_TAGS for k in K_LADDER), NAIVE,
)
#: C1 (Ridge) and C2 (vanilla Transformer) against the iTransformer at every rung.
PAIRED_CONTRASTS: Final = tuple(
    (("itr", k), (other, k), claim)
    for other, claim in (("rdg", "C1"), ("vtr", "C2")) for k in K_LADDER
)
#: Share of training data consecutive origins have in common.
ORIGIN_OVERLAP: Final = (TRAIN_MONTHS - ORIGIN_SPACING_MONTHS) / TRAIN_MONTHS
INFERENCE_STATUS: Final = "diagnostic: consecutive origins share training data, so origins are not independent"

SPLIT_COLOUR: Final = {"train": "#1f4e79", "val": "#5b8db8", "purge": "#f4a259",
                       "test": "#c1121f", "test_alt": "#e07a1f", "oos": "#6c757d"}
MODEL_COLOUR: Final = {"itr": "#1f4e79", "rdg": "#2e7d32", "vtr": "#c1121f"}
RUNG_STYLE: Final = {1: ":", 4: "-.", 8: "-", 12: "--"}
DEFAULT_MAX_EPOCHS: Final = 30
_MISSING: Final = "---"


# -- formatting -------------------------------------------------------------------


def fmt(value: float | int | None, digits: int = 4) -> str:
    """A number for a table cell; ``---`` for missing or non-finite values."""
    if value is None:
        return _MISSING
    number = float(value)
    if not math.isfinite(number):
        return _MISSING
    if digits == 0:
        return f"{int(round(number)):,}"
    return f"{number:.{digits}f}"


def tex_escape(text: str) -> str:
    for old, new in (("_", "\\_"), ("%", "\\%"), ("&", "\\&"), ("#", "\\#")):
        text = text.replace(old, new)
    return text


def tabular(caption: str, tag: str, header: list[str], rows: list[list[str]], align: str,
            note: str = "") -> str:
    """A booktabs table with an optional note, marked as generated."""
    lines = ["% GENERATED --- do not hand-edit.", "% Regenerate: python tools/build_report.py",
             "\\begin{table}[!t]", "\\centering", "\\caption{" + caption + "}",
             "\\label{" + tag + "}", "\\begin{tabular}{" + align + "}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(row) + " \\\\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines.append("\\vspace{2pt}\\par\\footnotesize " + note)
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def se_across(values: np.ndarray) -> float:
    """Standard error across the given values (origins)."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float("nan")
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def _star(p: float | None, threshold: float = 0.05) -> str:
    if p is None or not math.isfinite(float(p)):
        return ""
    return "$^{*}$" if float(p) < threshold else ""


# -- sections of paper_numbers.json -------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReportInputs:
    numbers: dict
    seed_avg: pl.DataFrame
    amplification: pl.DataFrame
    rolling_pr: pl.DataFrame
    rolling_r2: pl.DataFrame
    da_summary: pl.DataFrame


def _dataset_section(bars: pl.DataFrame) -> dict:
    """Table 1 and Table 5: the data, its breaks, and the window budget per origin."""
    summary = break_summary(bars, DATA_START, DATA_END)
    return {
        "window": [DATA_START.isoformat(), DATA_END.isoformat()],
        "bars_expected": BARS_EXPECTED, "bars_actual": BARS_ACTUAL,
        "missing_bars": MISSING_BARS, "gap_blocks": GAP_BLOCKS,
        "measured": {
            "calendar_hours": summary.calendar_hours, "bars_present": summary.bars_present,
            "bars_usable": summary.bars_usable, "missing_bars": summary.missing_bars,
            "zero_volume_bars": summary.zero_volume_bars, "flat_bars": summary.flat_bars,
            "zero_trade_bars": summary.zero_trade_bars,
            "excluded_positions": summary.excluded_positions,
            "break_runs": summary.break_runs, "segments": summary.segments,
        },
        "per_origin": [
            {"origin": b.label, "train_windows": b.windows_measured,
             "closed_form": b.windows_closed_form, "closed_form_agrees": b.closed_form_agrees,
             "loss_pct": b.loss_pct, "test_block_starts": list(b.test_block_starts),
             "worst_block_starts": int(min(b.test_block_starts))}
            for b in budget_table(bars)
        ],
    }


def _keff_section(features: pl.DataFrame, table: pl.DataFrame) -> dict:
    """Table 2b: K_eff per rung across origins, ``corr(K, K_eff)`` and the gate."""
    per_rung = table.group_by("k").agg(
        pl.col("pr_raw").mean().alias("PR_raw"), pl.col("pr_raw").std().alias("PR_raw_sd"),
        pl.col("pr_window_norm").mean().alias("PR_windownorm"),
        pl.col("stable_rank_lookback").mean().alias("stable_rank"),
        pl.col("pr_lookback_ratio").mean().alias("crosslag_share"),
    ).sort("k")
    measured = gate_pr(features, k=8)
    return {"per_rung": per_rung.to_dicts(), "corr_k_keff": corr_k_keff(table),
            "gate_pr_k8": measured, "gate_floor": GATE_PR_FLOOR,
            "gate_passed": bool(measured >= GATE_PR_FLOOR), "table": table.to_dicts()}


def _architecture_section(run_ids: list[str], roots: list[Path]) -> dict:
    """Table 3: parameters, epochs and runs at the epoch cap, per model and rung."""
    rows: dict[tuple[str, int], dict] = {}
    for run_id in run_ids:
        parts = parse_run_id(run_id)
        meta = load_meta(run_id, roots)
        key = (str(parts["model"]), int(parts["k"]))
        row = rows.setdefault(key, {
            "model": key[0], "k": key[1], "n_parameters": meta.get("n_parameters"),
            "n_allocated_parameters": meta.get("n_allocated_parameters"),
            "epochs": [], "n_runs": 0, "config": meta.get("config", {}),
            "schedule": meta.get("schedule"),
        })
        row["epochs"].append(int(meta.get("epochs_run", 0)))
        row["n_runs"] += 1
    out = []
    for row in rows.values():
        epochs = np.asarray(row.pop("epochs"), dtype=np.float64)
        cap = int((row.get("schedule") or {}).get("max_epochs", DEFAULT_MAX_EPOCHS))
        row.update(epochs_mean=float(epochs.mean()), epochs_max=int(epochs.max()),
                   max_epochs=cap if row.get("schedule") else 0,
                   epochs_at_cap=int((epochs >= cap).sum()) if row.get("schedule") else 0)
        out.append(row)
    out.sort(key=lambda row: (MODEL_TAGS.index(row["model"]), row["k"]))
    return {"cells": out}


def _research_questions(seed_avg: pl.DataFrame, keff_tbl: pl.DataFrame, *, B: int, seed: int):
    """RQ1 on the iTransformer ladder, and RQ2 on the K=1 against K=8 gap by block."""
    main = seed_avg.filter((pl.col("model") == "itr") & (pl.col("pred_len") == PRED_LEN))
    origin = main.group_by("origin", "k").agg(pl.col("rel_mse").mean(), pl.col("r2_oos").mean())
    rung = origin.group_by("k").agg(
        pl.col("rel_mse").mean().alias("RelMSE"),
        (pl.col("rel_mse").std() / pl.len().sqrt()).alias("SE_across_origins"),
        pl.col("r2_oos").mean().alias("R2_oos"), pl.len().alias("n_origins"),
    ).sort("k")
    wide = {k: origin.filter(pl.col("k") == k).sort("origin")["rel_mse"].to_numpy() for k in K_LADDER}
    d48, d812 = wide[4] - wide[8], wide[8] - wide[12]
    margin = 0.25 * abs(float(d48.mean()))
    race = main.join(keff_tbl.select("origin", "k", "pr_raw"), on=["origin", "k"])
    groups = race["origin_index"].to_numpy() * 100 + race["block"].to_numpy()
    clusters = race["origin_index"].to_numpy()
    y, k, pr = (race[n].to_numpy().astype(float) for n in ("rel_mse", "k", "pr_raw"))
    t_ab, p_ab = j_test(y, k, pr, groups, clusters=clusters)
    t_ba, p_ba = j_test(y, pr, k, groups, clusters=clusters)

    amp = amplification(seed_avg)
    beta = panel_beta1(amp, B=B, seed=seed)
    stride = []
    for offset in range(5):  # every fifth origin shares no training data
        labels = [o.label for o in ORIGINS[offset::5]]
        part = amp.filter(pl.col("origin").is_in(labels))
        if part.height == len(labels) * 6 and len(labels) >= 2:
            sub = panel_beta1(part, B=B, seed=seed)
            stride.append({"origins": labels, "G": sub.n_clusters, "beta1": sub.beta1,
                           "p_diagnostic": sub.headline_p})
    try:
        fit = panel_beta1_covariate(
            amp.with_columns((pl.col("n_large") / float(BLOCK_HOURS)).alias("coverage")), B=B, seed=seed)
        covariate = {"beta1": fit.beta1, "t": fit.t_statistic, "headline_p": fit.headline_p}
    except ValueError as error:
        covariate = {"status": "not estimable", "reason": str(error)}
    _, covered = beta1_with_coverage(amp, B=B, seed=seed)
    return {
        "rq1": {"rung_effects": rung.to_dicts(), "delta_4_to_8": float(d48.mean()),
                "delta_8_to_12": float(d812.mean()), "tost_margin": margin,
                "tost": str(tost_equivalence(d812, margin)),
                "j_test_k_augmented_by_keff": {"t": t_ab, "p": p_ab},
                "j_test_keff_augmented_by_k": {"t": t_ba, "p": p_ba}},
        "rq2": {"beta1": beta.beta1, "t": beta.t_statistic, "cluster_se": beta.cluster_se,
                "p_rademacher": beta.p_rademacher, "p_webb": beta.p_webb,
                "headline_p": beta.headline_p, "G": beta.n_clusters, "N": beta.n_observations,
                "B": beta.B, "minimum_detectable_beta1": minimum_detectable_beta1(beta.within_slopes),
                "within_slopes": beta.within_slopes.tolist(), "stride5": stride,
                "origin_overlap": ORIGIN_OVERLAP,
                "coverage_covariate": covariate,
                "coverage_restricted": None if covered is None else {
                    "beta1": covered.beta1, "headline_p": covered.headline_p,
                    "G": covered.n_clusters, "N": covered.n_observations}},
    }, amp


def _main_results(seed_avg: pl.DataFrame, keys: list[ModelKey], mcs: pl.DataFrame,
                  da_summary: pl.DataFrame) -> list[dict]:
    """Table 4: RelMSE and R2_oos with the SE across origins, MCS membership and DA."""
    main = seed_avg.filter(pl.col("pred_len") == PRED_LEN)
    membership = {row["model"]: row for row in mcs.to_dicts()}
    da = {(row["model"], row["k"]): row for row in da_summary.to_dicts()}
    rows = []
    for tag, k in keys:
        if (tag, k) == NAIVE:
            continue
        cell = main.filter((pl.col("model") == tag) & (pl.col("k") == k))
        by_origin = cell.group_by("origin").agg(
            pl.col("rel_mse").mean(), pl.col("r2_oos").mean(),
            pl.col("mse_seed_std").mean().alias("seed_std"))
        r2 = by_origin["r2_oos"].to_numpy()
        name = label((tag, k))
        rows.append({
            "model": name, "model_tag": tag, "k": k,
            "rel_mse": float(by_origin["rel_mse"].mean()), "r2_oos": float(r2.mean()),
            "se_across_origins": se_across(r2), "seed_std": float(by_origin["seed_std"].mean()),
            "n_origins": by_origin.height, "n_seeds": int(cell["n_seeds"].max()),
            "in_mcs_90": bool(membership.get(name, {}).get("in_mcs_90", False)),
            "in_mcs_75": bool(membership.get(name, {}).get("in_mcs_75", False)),
            "directional": da.get((tag, k)),
        })
    return rows


def build_report(artifacts: Path, bars: pl.DataFrame, features: pl.DataFrame, *,
                 roots: list[Path] | None = None, bootstrap_b: int = 9999, seed: int = 42,
                 log=print) -> ReportInputs:
    """Every number of the study from the complete 900-run grid.

    Raises:
        ValueError: If any run of the manifest is missing, or a run trained on a
            sample other than the fixed 11,500 windows.
    """
    roots = list(roots) if roots else [Path(artifacts)]
    run_ids = [c.run_id for c in manifest()]
    missing = sorted(set(run_ids) - completed_run_ids(roots))
    if missing:
        raise ValueError(f"the report needs all {len(run_ids)} runs; {len(missing)} are missing, "
                         f"e.g. {missing[:3]}")
    metadata = [load_meta(run_id, roots) for run_id in run_ids]
    if {int(m["n_train"]) for m in metadata} != {TRAIN_WINDOW_LIMIT}:
        raise ValueError(f"every run must train on {TRAIN_WINDOW_LIMIT} windows")
    windows = evaluation_windows(run_ids, roots)
    seed_avg = seed_average(gather_grid(run_ids, roots, windows=windows))
    log(f"report: {len(run_ids)} runs, {windows.height} common forecast times, "
        f"{seed_avg.height} seed-averaged cells")

    keff_tbl = keff_table(features)
    roll_pr, roll_r2 = rolling_pr(features, k=8), rolling_ols_r2(features, k=8)
    keys, absent = available_keys(list(COMPARISON_KEYS), roots)
    panel = build_panel(keys, roots, windows=windows)
    pairs = pair_matrix(panel, B=bootstrap_b, seed=seed)
    mcs = mcs_table(panel, B=bootstrap_b, seed=seed)
    da_summary = directional_accuracy_summary(directional_accuracy_table(run_ids, roots, windows=windows))
    questions, amp = _research_questions(seed_avg, keff_tbl, B=bootstrap_b, seed=seed)
    raw_scale = (raw_scale_table(seed_avg).group_by("model", "k", "origin")
                 .agg(pl.col("rmse_raw").mean()).group_by("model", "k")
                 .agg(pl.col("rmse_raw").mean(), pl.len().alias("n_origins")).sort("model", "k"))
    architecture = _architecture_section(run_ids, roots)
    numbers = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifacts_root": str(artifacts),
        "input_sha256": sorted({m["input_sha256"] for m in metadata}),
        "prediction_code_sha256": sorted({m["code_sha256"] for m in metadata}),
        "analysis_code_sha256": code_sha256(),
        "runs_complete": len(run_ids), "manifest_run_ids": run_ids,
        "inference_status": INFERENCE_STATUS,
        "dataset": _dataset_section(bars),
        "efficiency": efficiency_table(features).to_dicts(),
        "keff": _keff_section(features, keff_tbl),
        "keff_rolling": {
            "window_days": 90, "descriptive_only": True,
            "pr": {"min": float(roll_pr["pr"].min()), "max": float(roll_pr["pr"].max()),
                   "mean": float(roll_pr["pr"].mean())},
            "ols_r2": {"min": float(roll_r2["r2"].min()), "max": float(roll_r2["r2"].max()),
                       "mean": float(roll_r2["r2"].mean())},
        },
        "architecture": architecture,
        "main_results": _main_results(seed_avg, keys, mcs, da_summary),
        "contrasts": [{**paired_contrast(seed_avg, left, right), "claim": claim}
                      for left, right, claim in PAIRED_CONTRASTS],
        "comparisons": {"models": [label(key) for key in keys],
                        "absent": [label(key) for key in absent], "B": bootstrap_b,
                        "p_floor": 1.0 / (1 + bootstrap_b), "pairs": pairs.to_dicts(),
                        "mcs": mcs.to_dicts()},
        "directional_accuracy": da_summary.to_dicts(),
        "raw_scale": raw_scale.to_dicts(),
        **questions,
        "training_sample": {"windows": TRAIN_WINDOW_LIMIT, "equal_across_models": True},
        "optimization": {"runs_at_epoch_cap": sum(row["epochs_at_cap"] for row in architecture["cells"])},
    }
    return ReportInputs(numbers=numbers, seed_avg=seed_avg, amplification=amp,
                        rolling_pr=roll_pr, rolling_r2=roll_r2, da_summary=da_summary)


def build_paper_numbers(artifacts: Path, bars: pl.DataFrame, features: pl.DataFrame, **kwargs) -> dict:
    return build_report(artifacts, bars, features, **kwargs).numbers


# -- tables ---------------------------------------------------------------------------


def _table1(numbers: dict) -> str:
    data = numbers["dataset"]
    measured = data["measured"]
    rows = [[row["origin"], fmt(row["train_windows"], 0), fmt(row["loss_pct"], 2),
             fmt(row["worst_block_starts"], 0), "yes" if row["closed_form_agrees"] else "no"]
            for row in data["per_origin"]]
    note = (f"BTCUSDT spot 1\\,h, Binance. Window {data['window'][0][:10]} to {data['window'][1][:10]}, "
            f"end exclusive: {data['bars_expected']:,} expected bars, {data['bars_actual']:,} present, "
            f"{data['missing_bars']} missing in {data['gap_blocks']} blocks. "
            f"{measured['zero_volume_bars']} zero-volume, {measured['flat_bars']} with $H=L$ and "
            f"{measured['zero_trade_bars']} zero-trade bars; {measured['excluded_positions']} "
            f"calendar hours are excluded in total. Windows are counted segment by segment; "
            f"the closed form is an upper bound.")
    return tabular("Data and the per-origin window budget.", "tab:dataset",
                   ["Origin", "Train windows", "Loss (\\%)", "Worst test block", "Closed form agrees"],
                   rows, "lrrrc", note)


def _table2(numbers: dict) -> str:
    rows = [[tex_escape(str(row["span"])), fmt(row["n"], 0), fmt(row["adf_stat"], 2) + _star(row["adf_p"]),
             fmt(row["hurst"], 3),
             *[fmt(row.get(f"vr_{lag}"), 3) + _star(row.get(f"vr_p_{lag}")) for lag in (2, 4, 8, 16)]]
            for row in numbers["efficiency"]]
    note = ("Log-returns. ADF tests for a unit root; Hurst by rescaled range ($H\\approx0.5$: no "
            "long memory); Lo--MacKinlay variance ratio ($VR\\approx1$: consistent with a random "
            "walk). $^{*}$ marks $p<0.05$. The evidence is reported, not read as market efficiency.")
    return tabular("Market-efficiency diagnostics, full sample and per training sub-block.",
                   "tab:efficiency", ["Span", "$n$", "ADF", "Hurst", "$VR_2$", "$VR_4$", "$VR_8$",
                                      "$VR_{16}$"], rows, "lrrrrrrr", note)


def _table2b(numbers: dict) -> str:
    keff = numbers["keff"]
    rows = [[fmt(row["k"], 0), fmt(row["PR_raw"], 3) + " $\\pm$ " + fmt(row["PR_raw_sd"], 3),
             fmt(row["PR_windownorm"], 3), fmt(row["stable_rank"], 3), fmt(row["crosslag_share"], 3)]
            for row in keff["per_rung"]]
    gate = ("above the floor" if keff["gate_passed"]
            else "below the floor, so the value is disclosed and the ladder is not re-cut")
    note = (f"Measured per origin on that origin's 21-month training sub-block; $\\pm$ is the "
            f"standard deviation across origins. $corr(K, K_{{eff}}) = {fmt(keff['corr_k_keff'], 3)}$. "
            f"Gate: PR at $K=8$ on 2018-01 to 2020-01 is {fmt(keff['gate_pr_k8'], 3)} against a floor "
            f"of {fmt(keff['gate_floor'], 1)} fixed in advance, {gate}.")
    return tabular("Effective dimensionality per rung.", "tab:keff",
                   ["$K$", "PR (raw)", "PR (window-norm.)", "Stable rank", "Cross-lag share"],
                   rows, "rrrrr", note)


def _table3(numbers: dict) -> str:
    rows = [[MODEL_NAMES.get(c["model"], c["model"]), fmt(c["k"], 0), fmt(c["n_parameters"], 0),
             fmt(c["epochs_mean"], 2), fmt(c["epochs_max"], 0), fmt(c["epochs_at_cap"], 0),
             fmt(c["n_runs"], 0)] for c in numbers["architecture"]["cells"]]
    note = ("Hyperparameters follow the official iTransformer code, with $d_{model}=128$ and "
            "$d_{ff}=256$ for the sample size, identical at every rung. The vanilla Transformer uses "
            "the same code's defaults: two encoder layers, one decoder layer, GELU and a 48-hour "
            "start token. Ridge's $\\alpha$ is the only hyperparameter selected, on validation; its "
            "fit has no epochs. At cap counts runs that reached the 30-epoch budget. A parameter "
            "count does not measure effective capacity.")
    return tabular("Models, parameters and epochs at $H=24$.", "tab:hyperparameters",
                   ["Model", "$K$", "Params", "Epochs (mean)", "Max", "At cap", "Runs"],
                   rows, "lrrrrrr", note)


def _dda(row: dict | None, variant: str) -> str:
    if not row:
        return _MISSING
    return f"{100 * row[f'dda_{variant}']:+.1f} ({row[f'wins_{variant}']}/{row['n_origins']})"


def _table4(numbers: dict) -> str:
    rows = []
    for row in numbers["main_results"]:
        mcs = "90\\%, 75\\%" if row["in_mcs_75"] else "90\\%" if row["in_mcs_90"] else _MISSING
        rows.append([tex_escape(row["model"]), fmt(row["rel_mse"], 4),
                     fmt(row["r2_oos"], 4) + " $\\pm$ " + fmt(row["se_across_origins"], 4),
                     fmt(row["seed_std"], 6), mcs,
                     *[_dda(row["directional"], v) for v in ("1h", "24h", "cum")]])
    note = ("Common forecast times. Block RelMSE from seed-averaged squared error, equal block and "
            "origin weights; $\\pm$ is the SE across 15 origins. $\\Delta DA$ is directional accuracy "
            "minus each origin's majority-sign rate, in percentage points, with the origins that "
            "beat it; the rate uses test-period frequencies, so the comparison is conservative. "
            "Ridge is deterministic (seed std 0). All inference is diagnostic.")
    return tabular("Main results at $H=24$ across fifteen origins.", "tab:main",
                   ["Model", "RelMSE", "$R^2_{oos}$", "Seed std", "In MCS", "$\\Delta DA_{1h}$",
                    "$\\Delta DA_{24h}$", "$\\Delta DA_{cum}$"], rows, "lrrrcrrr", note)


def _table5(numbers: dict) -> str:
    rows = [[row["origin"], *[fmt(n, 0) for n in row["test_block_starts"]],
             fmt(100 * min(row["test_block_starts"]) / BLOCK_HOURS, 1)]
            for row in numbers["dataset"]["per_origin"]]
    note = ("Forecast origins surviving in each 30-day block, out of 720. An origin is lost only when "
            "a gap falls inside its 120-hour window. Survival depends on future gaps, so block "
            "coverage enters RQ2 as a covariate.")
    return tabular("Surviving forecast origins per test block.", "tab:coverage",
                   ["Origin", "B1", "B2", "B3", "B4", "B5", "B6", "Min cover (\\%)"],
                   rows, "lrrrrrrr", note)


def _table6(numbers: dict) -> str:
    rows = [[tex_escape(row["left"]), tex_escape(row["right"]), fmt(row["t_cluster"], 3),
             fmt(row["p_raw"], 4), fmt(row["p_romano_wolf"], 4), row.get("family", _MISSING),
             fmt(row.get("p_romano_wolf_family"), 4), fmt(row["T_min"], 0)]
            for row in numbers["comparisons"]["pairs"]]
    note = ("Pairwise forecast-loss contrasts on common targets; positive $t$ means the left model is "
            "worse. Romano--Wolf adjusts over all pairs and within each family (ladder, cross-model, "
            "vs-naive). Origins share training data, so these are diagnostics, not confirmatory tests.")
    return tabular("Pairwise forecast-loss diagnostics.", "tab:dm",
                   ["Left", "Right", "$t$", "$p_{raw}$", "$p_{RW}$", "Family", "$p_{RW}^{fam}$",
                    "$T_{min}$"], rows, "llrrrlrr", note)


_DIGITS: Final = {"1": "One", "2": "Two", "4": "Four", "8": "Eight"}


def render_tables(numbers: dict, out_dir: Path) -> list[Path]:
    """Write Tables 1-6 and the manuscript macros."""
    out_dir.mkdir(parents=True, exist_ok=True)
    builders = {"table1_dataset.tex": _table1, "table2_efficiency.tex": _table2,
                "table2b_keff.tex": _table2b, "table3_architecture.tex": _table3,
                "table4_main.tex": _table4, "table5_coverage.tex": _table5, "table6_dm.tex": _table6}
    written = []
    for name, builder in builders.items():
        path = out_dir / name
        path.write_text(builder(numbers), encoding="utf-8")
        written.append(path)
    main = {(row["model_tag"], row["k"]): row for row in numbers["main_results"]}
    macros = {"StudyRuns": fmt(numbers["runs_complete"], 0), "StudyOrigins": fmt(numbers["rq2"]["G"], 0),
              **{f"{tag.upper()}K{k}Rtwo": fmt(main[tag, k]["r2_oos"], 6)
                 for tag in MODEL_TAGS for k in K_LADDER},
              "BetaSlope": fmt(numbers["rq2"]["beta1"], 6), "BetaSE": fmt(numbers["rq2"]["cluster_se"], 6)}
    lines = ["% Generated from paper_numbers.json."]
    for name, value in macros.items():
        clean = "".join(_DIGITS.get(ch, ch) for ch in name)
        lines.append("\\newcommand{\\" + clean + "}{" + value + "}")
    (out_dir / "manuscript_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return written


# -- figures --------------------------------------------------------------------------


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("pdf", "png"):
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=200)
        paths.append(path)
    return paths


def _as_datetime(ms: np.ndarray) -> np.ndarray:
    return np.asarray(ms, dtype="int64").astype("datetime64[ms]")


def _figure1(inputs: ReportInputs, out_dir: Path) -> list[Path]:
    """The walk-forward design: every origin, then one origin resolved."""
    plt = _pyplot()
    fig, (top, low) = plt.subplots(2, 1, figsize=(7.6, 6.8), gridspec_kw={"height_ratios": [2.0, 0.8]})

    def year(moment) -> float:
        return moment.year + (moment.timetuple().tm_yday - 1) / 365.25

    for row, origin in enumerate(ORIGINS):
        y = len(ORIGINS) - row
        top.barh(y, year(origin.train_sub_end) - year(origin.train_start), left=year(origin.train_start),
                 height=0.62, color=SPLIT_COLOUR["train"])
        top.barh(y, year(origin.val_end) - year(origin.val_start), left=year(origin.val_start),
                 height=0.62, color=SPLIT_COLOUR["val"])
        for b, start, end in origin.blocks():
            top.barh(y, year(end) - year(start), left=year(start), height=0.62,
                     color=SPLIT_COLOUR["test"] if b % 2 else SPLIT_COLOUR["test_alt"],
                     edgecolor="#ffffff", lw=0.4)
    top.set_yticks(range(1, len(ORIGINS) + 1))
    top.set_yticklabels([o.label for o in reversed(ORIGINS)], fontsize=7.2)
    top.set_xlabel("calendar time (UTC)", fontsize=8.5)
    top.set_title(f"Fifteen origins, rolling {TRAIN_MONTHS}-month window, {ORIGIN_SPACING_MONTHS}-month "
                  f"spacing: consecutive origins share {100 * ORIGIN_OVERLAP:.1f}% of their training data",
                  fontsize=9)
    top.grid(axis="x", alpha=0.25, lw=0.5)
    origin = ORIGINS[0]

    def day(moment) -> float:
        return (moment - origin.train_start).total_seconds() / 86400.0

    low.barh(1.0, day(origin.train_sub_end) - day(origin.train_start), left=0, height=0.5,
             color=SPLIT_COLOUR["train"])
    low.barh(1.0, day(origin.val_end) - day(origin.val_start), left=day(origin.val_start), height=0.5,
             color=SPLIT_COLOUR["val"])
    for b, start, end in origin.blocks():
        low.barh(1.0, day(end) - day(start), left=day(start), height=0.5,
                 color=SPLIT_COLOUR["test"] if b % 2 else SPLIT_COLOUR["test_alt"], edgecolor="#ffffff")
        low.text((day(start) + day(end)) / 2, 1.0, str(b), ha="center", va="center", fontsize=6.5,
                 color="#ffffff")
    for boundary in (origin.train_sub_end, origin.val_end):
        low.barh(1.0, 22, left=day(boundary) - 11, height=0.62, color=SPLIT_COLOUR["purge"],
                 hatch="////", edgecolor="#5a2d00", lw=0.6)
    low.set_yticks([])
    low.set_xlabel(f"days since the training start, origin {origin.label}", fontsize=8.5)
    low.set_title("Train (21 months), purge, validation (3 months), purge, six 30-day test blocks; "
                  "the purge is drawn 22x wider than its 24 hours", fontsize=8.5)
    handles = [plt.Line2D([], [], lw=6, color=SPLIT_COLOUR[key], label=text) for key, text in (
        ("train", "training sub-block (scaler fitted here)"), ("val", "validation"),
        ("purge", "24-hour purge at both boundaries"), ("test", "test blocks 1-6"))]
    low.legend(handles=handles, fontsize=7, frameon=False, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, -1.0))
    fig.tight_layout()
    paths = _save(fig, out_dir, "figure1_walkforward")
    plt.close(fig)
    return paths


def _figure2(inputs: ReportInputs, out_dir: Path) -> list[Path]:
    """The three models side by side: what a token is and how the forecast leaves."""
    plt = _pyplot()
    columns = [
        ("iTransformer", "#1f4e79", ["window (B, 96, K)", "one token per variate:\nLinear(96 -> 128)",
                                     "encoder x2 over K tokens", "Linear(128 -> 24) per token",
                                     "read channel r"]),
        ("Transformer", "#c1121f", ["window (B, 96, K)", "one token per hour:\nConv1d(K -> 128) + position",
                                    "encoder x2 over 96 tokens;\ndecoder x1: last 48 h + 24 zeros",
                                    "Linear(128 -> K) per hour", "read channel r"]),
        ("Ridge", "#2e7d32", ["window (B, 96, K)", "flatten to 96K values", "one linear map with\nL2 penalty",
                              "24 outputs", "channel r only"]),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for col, (title, colour, steps) in enumerate(columns):
        x = 0.03 + col * 0.33
        ax.text(x + 0.14, 0.97, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color=colour)
        for row, text in enumerate(steps):
            y = 0.80 - row * 0.17
            ax.add_patch(plt.Rectangle((x, y), 0.28, 0.12, facecolor=colour, alpha=0.12 + 0.12 * (row % 2),
                                       edgecolor=colour, lw=0.8))
            ax.text(x + 0.14, y + 0.06, text, ha="center", va="center", fontsize=7.0)
    ax.text(0.5, -0.02, "All three read the same windows and are scored on the target channel r only.",
            ha="center", va="top", fontsize=7.5)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.06, 1.0)
    ax.axis("off")
    fig.tight_layout()
    paths = _save(fig, out_dir, "figure2_architecture")
    plt.close(fig)
    return paths


def _figure2b(inputs: ReportInputs, out_dir: Path) -> list[Path]:
    """Rolling participation ratio and in-window OLS R^2, descriptive only."""
    plt = _pyplot()
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.6), sharex=True)
    for axis, frame, column, colour, ylabel in (
            (axes[0], inputs.rolling_pr, "pr", "#22577a", "PR at $K=8$"),
            (axes[1], inputs.rolling_r2, "r2", "#c1121f", "in-window $R^2$")):
        axis.plot(_as_datetime(frame["window_end_ms"].to_numpy()), frame[column].to_numpy(), lw=1.0,
                  color=colour)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25, lw=0.5)
    axes[1].set_xlabel("window end (UTC)")
    fig.suptitle("90-day rolling participation ratio and OLS fit (descriptive only)", fontsize=10)
    fig.tight_layout()
    paths = _save(fig, out_dir, "figure2b_rolling")
    plt.close(fig)
    return paths


def _figure3(inputs: ReportInputs, out_dir: Path) -> list[Path]:
    """RQ2: the K=1 against K=8 gap by block, one line per origin, with the fit and MDE."""
    plt = _pyplot()
    amp = inputs.amplification
    rq2 = inputs.numbers["rq2"]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for _, part in amp.group_by(["origin"], maintain_order=True):
        part = part.sort("block")
        ax.plot(part["block"].to_numpy(), part["A"].to_numpy(), lw=0.8, alpha=0.55, marker="o", ms=2.5,
                color="#4a6fa5")
    blocks = np.arange(1, 7, dtype=float)
    intercept = float(amp["A"].mean()) - rq2["beta1"] * blocks.mean()
    ax.plot(blocks, intercept + rq2["beta1"] * blocks, lw=2.4, color="#c1121f",
            label=f"fitted $\\beta_1$ = {rq2['beta1']:+.6f}")
    mde = rq2["minimum_detectable_beta1"]
    ax.plot(blocks, intercept + mde * blocks, lw=1.6, ls="--", color="#333333",
            label=f"minimum detectable slope = {mde:+.6f}")
    ax.axhline(0.0, lw=0.8, color="#888888")
    ax.set_xlabel("test block $b$ (30 days each)")
    ax.set_ylabel("$A(i,b) = (MSE_{K1} - MSE_{K8}) / MSE_{K1}$")
    ax.set_title("The multivariate gap against model age, one line per origin", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    paths = _save(fig, out_dir, "figure3_gap_by_age")
    plt.close(fig)
    return paths


def _figure4(inputs: ReportInputs, out_dir: Path) -> list[Path]:
    """RelMSE per block for every model and rung; Naive-RW is the dashed line at 1."""
    plt = _pyplot()
    main = inputs.seed_avg.filter(pl.col("pred_len") == PRED_LEN)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for tag in MODEL_TAGS:
        for k in K_LADDER:
            cell = main.filter((pl.col("model") == tag) & (pl.col("k") == k))
            if cell.height == 0:
                continue
            by_block = cell.group_by("block").agg(pl.col("rel_mse").mean()).sort("block")
            ax.plot(by_block["block"].to_numpy(), by_block["rel_mse"].to_numpy(), marker="o", ms=3,
                    lw=1.1, color=MODEL_COLOUR[tag], ls=RUNG_STYLE[k], label=f"{MODEL_NAMES[tag]} K={k}")
    ax.axhline(1.0, lw=1.2, color="#000000", ls="--", label="Naive-RW")
    ax.set_xlabel("test block $b$ (30 days each)")
    ax.set_ylabel("RelMSE (lower is better)")
    ax.set_title("RelMSE per block; above the dashed line loses to Naive-RW", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=6.5, ncol=2, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    paths = _save(fig, out_dir, "figure4_relmse")
    plt.close(fig)
    return paths


def _figure5(inputs: ReportInputs, out_dir: Path) -> list[Path]:
    """Directional accuracy against the majority-sign baseline, three panels by variant."""
    plt = _pyplot()
    summary = inputs.da_summary
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.4), sharey=True)
    for axis, variant, title in zip(axes, ("1h", "24h", "cum"), ("first hour", "24th hour", "24-hour sum")):
        for tag in MODEL_TAGS:
            part = summary.filter(pl.col("model") == tag).sort("k")
            if part.height == 0:
                continue
            axis.errorbar(part["k"].to_numpy(), 100 * part[f"dda_{variant}"].to_numpy(),
                          yerr=100 * part[f"dda_{variant}_se"].to_numpy(), marker="o", ms=3.5, lw=1.2,
                          capsize=2.5, color=MODEL_COLOUR[tag], label=MODEL_NAMES[tag])
        axis.axhline(0.0, lw=1.0, color="#000000", ls="--")
        axis.set_xticks(list(K_LADDER))
        axis.set_xlabel("$K$")
        axis.set_title(f"DA, {title}", fontsize=9)
        axis.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("DA minus majority rate (pp)")
    axes[-1].legend(fontsize=7.5, frameon=False)
    fig.suptitle("Directional accuracy above the per-origin majority-sign rate; bars are SE across origins",
                 fontsize=9.5)
    fig.tight_layout()
    paths = _save(fig, out_dir, "figure5_directional")
    plt.close(fig)
    return paths


def render_figures(inputs: ReportInputs, out_dir: Path, log=print) -> list[Path]:
    """Write Figures 1, 2, 2b, 3, 4 and 5 as PDF and PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for builder in (_figure1, _figure2, _figure2b, _figure3, _figure4, _figure5):
        written.extend(builder(inputs, out_dir))
    log(f"report: {len(written)} figure files")
    return written

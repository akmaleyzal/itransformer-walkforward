"""Render the study's tables, figures and paper_numbers.json from saved runs.

Reads the 900-run grid output (``preds/`` and ``meta/``) with the package's
analysis code and never fits a model. --check compares a regenerated
paper_numbers.json with the one on disk, ignoring only its generation timestamp
and floating-point differences below FLOAT_RTOL.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from itransformer_btc.features import build_features  # noqa: E402
from itransformer_btc.report import (  # noqa: E402
    build_report,
    render_figures,
    render_tables,
)
from itransformer_btc.segments import load_bars, usable_mask  # noqa: E402

DEFAULT_ARTIFACTS = ROOT / "notebooks" / "outputs" / "grid_900" / "artifacts"
DEFAULT_OUT = ROOT / "notebooks" / "outputs" / "grid_900" / "paper"
DEFAULT_PARQUET = ROOT / "data" / "raw" / "BTCUSDT_1h.parquet"

#: Fields that change on every build and mean nothing to a reader.
VOLATILE = ("generated_utc",)

#: Relative tolerance for ``--check``. polars aggregates in parallel, so float32
#: means can move in the seventh significant digit between builds; structure is
#: still compared exactly.
FLOAT_RTOL = 1e-6


def resolve_parquet(argument: str | None) -> Path:
    """Argument, then ``ITBTC_PARQUET``, then the committed input parquet."""
    if argument:
        return Path(argument)
    from_env = os.environ.get("ITBTC_PARQUET")
    return Path(from_env) if from_env else DEFAULT_PARQUET


def stable(numbers: dict) -> dict:
    """The numbers without their volatile fields: what ``--check`` compares."""
    copy = json.loads(json.dumps(numbers, default=float))
    for field in VOLATILE:
        copy.pop(field, None)
    return copy


def first_difference(left, right, path: str = "") -> str | None:
    """The first place two reports disagree, or ``None``.

    Structure is compared exactly and floats within :data:`FLOAT_RTOL`; two NaNs
    count as equal, because Ridge's seed std is undefined on every build.
    """
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            missing = sorted(set(left) ^ set(right))
            return f"{path or '<root>'}: keys differ, e.g. {missing[:4]}"
        for key in left:
            found = first_difference(left[key], right[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} vs {len(right)}"
        for index, (a, b) in enumerate(zip(left, right)):
            found = first_difference(a, b, f"{path}[{index}]")
            if found:
                return found
        return None
    if isinstance(left, float) and isinstance(right, float):
        if left == right or (math.isnan(left) and math.isnan(right)):
            return None
        scale = max(abs(left), abs(right), 1e-30)
        if abs(left - right) / scale <= FLOAT_RTOL:
            return None
        return f"{path}: {left!r} vs {right!r}"
    if left != right:
        return f"{path}: {left!r} vs {right!r}"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the study's tables and figures.")
    parser.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--bootstrap-b", type=int, default=9_999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if regenerating would change paper_numbers.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    log = (lambda *_: None) if args.quiet else print
    artifacts = Path(args.artifacts)
    out_dir = Path(args.out)
    parquet = resolve_parquet(args.parquet)

    log(f"reading {parquet}")
    bars = usable_mask(load_bars(parquet))
    features = build_features(bars)
    log(f"bars {bars.height:,}  features {features.height:,}")

    inputs = build_report(artifacts, bars, features, bootstrap_b=args.bootstrap_b,
                          seed=args.seed, log=log)
    numbers_path = out_dir / "paper_numbers.json"
    rendered = stable(inputs.numbers)

    if args.check:
        if not numbers_path.exists():
            print(f"{numbers_path} is absent. Run: python tools/build_report.py")
            return 1
        current = stable(json.loads(numbers_path.read_text(encoding="utf-8")))
        difference = first_difference(current, rendered)
        if difference:
            print(f"{numbers_path} is stale against {artifacts}: {difference}\n"
                  f"Run: python tools/build_report.py")
            return 1
        log("report is current")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    numbers_path.write_text(json.dumps(inputs.numbers, indent=2, default=float), encoding="utf-8")
    log(f"wrote {numbers_path}")
    tables = render_tables(inputs.numbers, out_dir / "tables")
    log(f"wrote {len(tables)} tables to {out_dir / 'tables'}")
    render_figures(inputs, out_dir / "figures", log=log)

    # The panels the figures read, so a plot can be redone without re-aggregating.
    panels = out_dir / "panels"
    panels.mkdir(parents=True, exist_ok=True)
    for name, frame in (("seed_averaged_cells", inputs.seed_avg),
                        ("amplification_panel", inputs.amplification),
                        ("rolling_pr", inputs.rolling_pr),
                        ("rolling_ols_r2", inputs.rolling_r2),
                        ("directional_accuracy", inputs.da_summary)):
        frame.write_parquet(panels / f"{name}.parquet")
    log(f"wrote panels to {panels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

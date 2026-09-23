"""The report on a complete synthetic 900-run grid whose answers are known.

Ridge beats the iTransformer and the vanilla Transformer loses to it by
construction (``conftest.SKILL``), so the sign of every C1 and C2 contrast is
fixed in advance. Bars and features are the real input, so the data tables are
built exactly as in the study.
"""

from __future__ import annotations

import importlib.util
import math
import re
from pathlib import Path

import pytest

from itransformer_btc.features import build_features
from itransformer_btc.report import build_report, fmt, render_figures, render_tables, tabular
from itransformer_btc.runner import manifest
from itransformer_btc.segments import load_bars, usable_mask

from conftest import write_grid

ROOT = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "data" / "raw" / "BTCUSDT_1h.parquet"
#: ``nan`` as a token, so "Binance" does not trip it.
NON_VALUE = re.compile(r"(?<![A-Za-z])(nan|inf|None)(?![A-Za-z])", re.IGNORECASE)
#: Few bootstrap draws: these tests are about structure and signs, not p-values.
TEST_B = 99


def _generator():
    spec = importlib.util.spec_from_file_location("build_report_tool", ROOT / "tools" / "build_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fmt_never_emits_a_non_value() -> None:
    assert fmt(float("nan")) == fmt(float("inf")) == fmt(None) == "---"
    assert fmt(0.123456, 3) == "0.123"
    assert fmt(75094, 0) == "75,094"


def test_tabular_is_balanced_and_names_its_generator() -> None:
    out = tabular("Caption", "tab:x", ["A", "B"], [["1", "2"]], "rr", note="note")
    assert out.count(r"\begin{tabular}") == out.count(r"\end{tabular}") == 1
    assert out.count(r"\begin{table}") == out.count(r"\end{table}") == 1
    assert "tools/build_report.py" in out
    assert "itransformer_btc" not in out


@pytest.fixture(scope="module")
def built(full_grid, tmp_path_factory):
    if not PARQUET.exists():
        pytest.skip("the input parquet is not in this checkout")
    bars = usable_mask(load_bars(PARQUET))
    inputs = build_report(full_grid, bars, build_features(bars), bootstrap_b=TEST_B, log=lambda *_: None)
    out = tmp_path_factory.mktemp("report")
    tables = render_tables(inputs.numbers, out / "tables")
    figures = render_figures(inputs, out / "figures", log=lambda *_: None)
    return inputs, tables, figures


def test_every_section_is_present_and_finite(built) -> None:
    numbers = built[0].numbers
    for key in ("dataset", "efficiency", "keff", "keff_rolling", "architecture", "main_results",
                "contrasts", "comparisons", "directional_accuracy", "raw_scale", "rq1", "rq2"):
        assert numbers[key], key
    assert numbers["runs_complete"] == 900 and len(numbers["main_results"]) == 12

    def non_finite(value, path=""):
        if isinstance(value, dict):
            for k, v in value.items():
                yield from non_finite(v, f"{path}.{k}")
        elif isinstance(value, list):
            for i, v in enumerate(value):
                yield from non_finite(v, f"{path}[{i}]")
        elif isinstance(value, float) and not math.isfinite(value):
            yield path

    assert list(non_finite(numbers)) == []


def test_c1_and_c2_have_the_signs_built_into_the_grid(built) -> None:
    contrasts = built[0].numbers["contrasts"]
    assert [c["claim"] for c in contrasts] == ["C1"] * 4 + ["C2"] * 4
    for c in contrasts:
        # mean_diff is left minus right in RelMSE; the iTransformer is always left.
        assert c["left"].startswith("itr-") and c["n_origins"] == 15
        assert (c["mean_diff"] > 0) if c["claim"] == "C1" else (c["mean_diff"] < 0)


def test_ridge_is_deterministic_and_has_no_epochs(built) -> None:
    numbers = built[0].numbers
    for row in numbers["main_results"]:
        assert (row["seed_std"] == 0.0) == (row["model_tag"] == "rdg")
    for cell in numbers["architecture"]["cells"]:
        assert (cell["max_epochs"] == 0) == (cell["model"] == "rdg")


def test_directional_accuracy_reaches_table_4(built) -> None:
    rows = {(r["model"], r["k"]): r for r in built[0].numbers["directional_accuracy"]}
    assert len(rows) == 12 and all(r["n_origins"] == 15 and r["runs"] == 75 for r in rows.values())
    table4 = next(p for p in built[1] if p.name == "table4_main.tex").read_text(encoding="utf-8")
    assert len(re.findall(r"[+-]\d+\.\d \(\d+/15\)", table4)) == 36


def test_tables_are_balanced_latex_without_non_values(built) -> None:
    assert len(built[1]) == 7
    for path in built[1]:
        text = path.read_text(encoding="utf-8")
        assert text.count(r"\begin{tabular}") == text.count(r"\end{tabular}") == 1, path.name
        assert not NON_VALUE.search(text), path.name


def test_every_figure_is_written_as_pdf_and_png(built) -> None:
    stems = {p.stem for p in built[2]}
    assert len(built[2]) == 12
    assert {"figure3_gap_by_age", "figure5_directional"} <= stems


def test_a_short_grid_is_refused(tmp_path) -> None:
    write_grid(tmp_path, manifest()[:3])
    with pytest.raises(ValueError, match="needs all 900 runs"):
        build_report(tmp_path, None, None, log=lambda *_: None)


def test_generator_check_sees_drift_beyond_its_tolerance(built) -> None:
    tool = _generator()
    numbers = tool.stable(built[0].numbers)
    assert "generated_utc" not in numbers
    assert tool.first_difference(numbers, tool.stable(built[0].numbers)) is None
    moved = tool.stable(built[0].numbers)
    moved["rq2"]["beta1"] *= 1 + 1e-3
    assert tool.first_difference(numbers, moved) == (
        f".rq2.beta1: {numbers['rq2']['beta1']!r} vs {moved['rq2']['beta1']!r}")
    moved["rq2"]["beta1"] = numbers["rq2"]["beta1"] * (1 + 1e-9)
    assert tool.first_difference(numbers, moved) is None

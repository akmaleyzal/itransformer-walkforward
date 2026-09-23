"""Bootstrap and check notebooks/btc_walkforward_3model.ipynb.

The notebook is authoritative once it exists: edit it, save it, and export its
definition cells to ``src/itransformer_btc/`` with its final sync cell
(``tools/notebook_to_src.py``). This file writes the first version from
``src/`` (``--bootstrap``, refused when the notebook exists) and validates the
pair (``--check``).

Each module becomes one definition cell. Flattening removes only:

* intra-package imports, because every cell shares one kernel namespace;
* the ``if __name__ == "__main__":`` guard, because a cell's ``__name__`` is
  ``"__main__"`` and the guard would fire;
* the functions in :data:`FLATTEN_DROP_FUNCTIONS` (the command-line entry point);
* module-level imports, which the Library cell carries once for every module.

The upstream copies under ``vendor/thuml_iTransformer`` are written by
``%%writefile`` cells, byte for byte, and verified against their pinned sha256
before either model is built.

Usage::

    python tools/build_notebook.py --bootstrap   # write the notebook if absent
    python tools/build_notebook.py --check       # exit 1 if notebook and src/ disagree
"""

from __future__ import annotations

import argparse
import ast
import copy
import dataclasses
import hashlib
import json
import symtable
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "src" / "itransformer_btc"
NOTEBOOK = ROOT / "notebooks" / "btc_walkforward_3model.ipynb"
PKG_NAME = "itransformer_btc"
#: Where the ``%%writefile`` cells put the upstream copies, relative to the working directory.
VENDOR_TARGET = "vendor/thuml_iTransformer"
#: The published files at the pinned commit, as downloaded.
UPSTREAM_FIXTURES = ROOT / "tests" / "fixtures" / "thuml_iTransformer_c2426e68"

sys.path.insert(0, str(ROOT / "src"))
from itransformer_btc.upstream import (  # noqa: E402
    UPSTREAM_COMMIT,
    UPSTREAM_FILES,
    UPSTREAM_TRIMMED,
    permalink,
    vendor_root,
)

#: Execution order. Each cell is executed, so a cell may only use at module level
#: what an earlier cell defined; function bodies resolve names at call time.
MODULE_ORDER: tuple[str, ...] = (
    "config.py",
    "__init__.py",
    "segments.py",
    "windows.py",
    "budget.py",
    "features.py",
    "efficiency.py",
    "splits.py",
    "keff.py",
    "upstream.py",
    "model.py",
    "train.py",
    "baselines.py",
    "metrics.py",
    "comparisons.py",
    "runner.py",
    "report.py",
)

#: Definitions that cannot work in a notebook cell: the command-line entry point.
FLATTEN_DROP_FUNCTIONS: dict[str, tuple[str, ...]] = {
    "runner.py": ("_main",),
}


# -- flattening --------------------------------------------------------------


def _is_main_guard(node: ast.stmt) -> bool:
    """``if __name__ == "__main__":`` at module level, however it is spelled."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
    )


def _intra_package_import(node: ast.AST) -> bool:
    if isinstance(node, ast.ImportFrom):
        # A relative import has no parent package in a cell, so it counts too.
        return node.level > 0 or (node.module or "").split(".")[0] == PKG_NAME
    if isinstance(node, ast.Import):
        return any(a.name.split(".")[0] == PKG_NAME for a in node.names)
    return False


def _module_object_bindings(node: ast.AST) -> list[str]:
    """Names a dropped import would have bound to a module object.

    ``from itransformer_btc.metrics import f`` binds a function another cell
    defines, so dropping it is lossless. ``from itransformer_btc import metrics``
    binds a module no cell defines, so every ``metrics.f`` left behind would
    raise NameError when first reached.
    """
    bound: list[str] = []
    if isinstance(node, ast.ImportFrom):
        if node.level > 0 or (node.module or "") == PKG_NAME:
            for alias in node.names:
                if (PACKAGE / f"{alias.name}.py").exists():
                    bound.append(alias.asname or alias.name)
    elif isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name.split(".")[0] == PKG_NAME:
                bound.append(alias.asname or alias.name.split(".")[0])
    return bound


def unbound_global_reads(source: str, label: str, names: set[str]) -> set[str]:
    """Which of ``names`` the source reads as a global it never binds.

    A symbol-table question rather than a spelling one, so a local variable that
    shares a module's name is not reported.
    """
    hits: set[str] = set()
    stack = [symtable.symtable(source, label, "exec")]
    while stack:
        table = stack.pop()
        stack.extend(table.get_children())
        for sym in table.get_symbols():
            if (
                sym.get_name() in names
                and sym.is_referenced()
                and sym.is_global()
                and not (sym.is_assigned() or sym.is_imported())
            ):
                hits.add(sym.get_name())
    return hits


def _executable_source(source: str) -> str:
    """Source with docstrings stripped: what actually runs."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ) and ast.get_docstring(node):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(ast.fix_missing_locations(tree))


def flatten_module_source(name: str) -> str:
    """The module verbatim, minus intra-package imports, the guard and dropped functions.

    The result is compiled and re-parsed before it is returned, so a cell that
    would fail on a machine without the package fails here instead.
    """
    text = (PACKAGE / name).read_text(encoding="utf-8")
    tree = ast.parse(text, filename=name)

    spans: list[tuple[int, int]] = [
        (node.lineno, node.end_lineno or node.lineno)
        for node in ast.walk(tree)
        if _intra_package_import(node)
    ]
    module_objects = {
        binding
        for node in ast.walk(tree)
        if _intra_package_import(node)
        for binding in _module_object_bindings(node)
    }
    spans += [
        (node.lineno, node.end_lineno or node.lineno)
        for node in tree.body
        if _is_main_guard(node)
    ]

    unusable = FLATTEN_DROP_FUNCTIONS.get(name, ())
    found = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in unusable
    }
    absent = sorted(set(unusable) - found.keys())
    assert not absent, f"{name}: FLATTEN_DROP_FUNCTIONS names {absent}, which no longer exist"
    for node in found.values():
        first = min([d.lineno for d in node.decorator_list] + [node.lineno])
        spans.append((first, node.end_lineno or node.lineno))

    dropped = {n for lo, hi in spans for n in range(lo, hi + 1)}
    source = "".join(
        line
        for number, line in enumerate(text.splitlines(keepends=True), start=1)
        if number not in dropped
    )

    compile(source, f"<cell:{name}>", "exec")
    assert PKG_NAME not in _executable_source(source), (
        f"{name}: an executable reference to {PKG_NAME} survived flattening"
    )
    dangling = unbound_global_reads(source, f"<cell:{name}>", module_objects)
    assert not dangling, (
        f"{name}: {sorted(dangling)} is read after its import was dropped, and it named a "
        f"module, which no cell binds. Import the names instead: "
        f"`from {PKG_NAME}.{sorted(dangling)[0]} import <name>`."
    )
    return source


def _module_level_import_lines(source: str) -> set[int]:
    """1-based lines of every module-level import (function-local imports stay)."""
    out: set[int] = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return out


def flatten_module_body(name: str) -> str:
    """The flattened module without its module-level imports: the definition cell."""
    source = flatten_module_source(name)
    lines = source.splitlines(keepends=True)
    dropped = _module_level_import_lines(source)
    # The blank lines after an import block go with it; those before it stay.
    for number in sorted(dropped):
        after = number + 1
        while after <= len(lines) and not lines[after - 1].strip():
            dropped.add(after)
            after += 1
    return "".join(
        line for number, line in enumerate(lines, start=1) if number not in dropped
    )


def package_digest() -> str:
    """Byte for byte what ``code_sha256()`` returns for this source tree."""
    digest = hashlib.sha256()
    for path in sorted(PACKAGE.rglob("*.py"), key=lambda p: p.relative_to(PACKAGE).as_posix()):
        digest.update(path.relative_to(PACKAGE).as_posix().encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def library_cell() -> str:
    """Every module-level import the package makes, deduplicated and grouped."""
    plain: set[str] = {"import gc"}  # the grid cell frees memory between phases
    grouped: dict[str, set[tuple[str, str | None]]] = {}
    for name in MODULE_ORDER:
        for node in ast.parse(flatten_module_source(name)).body:
            if isinstance(node, ast.Import):
                plain.add(ast.unparse(node))
            elif isinstance(node, ast.ImportFrom):
                grouped.setdefault(node.module or "", set()).update(
                    (a.name, a.asname) for a in node.names
                )

    def render_from(module: str) -> str:
        names = ", ".join(f"{n} as {alias}" if alias else n for n, alias in sorted(grouped[module]))
        return f"from {module} import {names}"

    modules = sorted(m for m in grouped if m != "__future__")
    lines = (
        ([render_from("__future__")] if "__future__" in grouped else [])
        + sorted(plain)
        + [render_from(m) for m in modules]
    )
    return "\n".join(lines) + "\n"


def vendor_files() -> list[str]:
    """Upstream copies a ``%%writefile`` cell writes; the empty ``__init__.py`` come from setup."""
    return [path for path in UPSTREAM_FILES if (vendor_root() / path).stat().st_size > 0]


def vendor_cell(path: str) -> str:
    """``%%writefile`` plus the copy; IPython restores the final newline on write."""
    text = (vendor_root() / path).read_bytes().decode("utf-8")
    assert text.endswith("\n") and not text.endswith("\n\n"), f"{path} is not in canonical form"
    return f"%%writefile {VENDOR_TARGET}/{path}\n{text[:-1]}"


# -- layout --------------------------------------------------------------------

#: ``(gradient from, gradient to, accent, heading, body)`` per module.
MODULE_THEME: dict[str, tuple[str, str, str, str, str]] = {
    "config.py": ("#0b1021", "#14213d", "#8ecae6", "#8ecae6", "#a8c7d8"),
    "__init__.py": ("#101010", "#1c1c1c", "#9e9e9e", "#d0d0d0", "#a8a8a8"),
    "segments.py": ("#1a1200", "#2b1d00", "#ffb703", "#ffd60a", "#ffca7a"),
    "windows.py": ("#1a1200", "#2b1d00", "#fb8500", "#ffb703", "#ffca7a"),
    "budget.py": ("#231400", "#3a2200", "#f48c06", "#ffba08", "#ffd08a"),
    "features.py": ("#001a1a", "#002b2b", "#48cae4", "#90e0ef", "#ade8f4"),
    "efficiency.py": ("#002200", "#003300", "#7ae582", "#95d5b2", "#b7e4c7"),
    "splits.py": ("#001233", "#001845", "#4cc9f0", "#8ecae6", "#a9d6e5"),
    "keff.py": ("#150029", "#240046", "#9d4edd", "#e0aaff", "#c8a2e0"),
    "upstream.py": ("#1f1300", "#332000", "#e9c46a", "#f4d58d", "#f0dfb0"),
    "model.py": ("#150029", "#22003d", "#c77dff", "#e0aaff", "#cbb2e8"),
    "train.py": ("#1b0033", "#2d0052", "#bf5af2", "#e0aaff", "#cbb2e8"),
    "baselines.py": ("#0a1a12", "#0f2a1c", "#52b788", "#95d5b2", "#b7e4c7"),
    "metrics.py": ("#03071e", "#370617", "#e94560", "#f5a623", "#ffd6a5"),
    "comparisons.py": ("#2b0a00", "#3d1000", "#ff7b54", "#ffb4a2", "#ffd8c2"),
    "runner.py": ("#012a4a", "#013a63", "#48cae4", "#90e0ef", "#caf0f8"),
    "report.py": ("#001a0d", "#003317", "#52b788", "#95d5b2", "#b7e4c7"),
}

#: One line under each definition cell's heading.
MODULE_HEADER: dict[str, tuple[str, str]] = {
    "config.py": ("📐", "Kontrak data, protokol walk-forward, 15 origin, dan sumber tiap algoritma."),
    "__init__.py": ("🧾", "Nama publik paket."),
    "segments.py": ("✂️", "Gap dan bar tak layak memutus deret; tidak ada imputasi."),
    "windows.py": ("🪟", "Jendela sah divalidasi lewat timestamp, bukan indeks."),
    "budget.py": ("🧮", "Anggaran jendela per origin, dicocokkan persis."),
    "features.py": ("🔬", "Dua belas variat per-bar F1–F5 dan tangga K."),
    "efficiency.py": ("📉", "ADF, variance ratio, dan Hurst."),
    "splits.py": ("🗂️", "Split, purge, scaler dari data latih saja, dan tensor per origin."),
    "keff.py": ("📏", "Dimensionalitas efektif (participation ratio) per origin."),
    "upstream.py": ("🔐", "Verifikasi sha256 dan pemuatan terisolasi kode thuml/iTransformer."),
    "model.py": ("🧠", "Adapter iTransformer dan Transformer vanilla ke kelas upstream."),
    "train.py": ("🏋️", "Loop latih, checkpoint per epoch, digest kode, dan artefak run."),
    "baselines.py": ("🧷", "Ridge dari scikit-learn; α dipilih pada validation."),
    "metrics.py": ("🎯", "Metrik, akurasi arah, DM/HLN/Clark–West, β₁ dan bootstrap klaster."),
    "comparisons.py": ("⚖️", "Matriks pasangan, Romano–Wolf, dan Model Confidence Set."),
    "runner.py": ("🚀", "Manifes 900 run, resume, gerbang desain, eksekutor dua GPU, pilot."),
    "report.py": ("🖼️", "Tabel, figure, dan paper_numbers.json dari run tersimpan."),
}


@dataclasses.dataclass(frozen=True)
class Vendor:
    """One ``%%writefile`` cell for an upstream copy."""

    path: str


@dataclasses.dataclass(frozen=True)
class Step:
    """One orchestration cell.

    Attributes:
        step: Stable slug; tests and the artifact map address the cell by it.
        code: Cell body, or a ``{placeholder}`` that :func:`build` expands.
        title: One-line heading; ``None`` when the phase card directly above is
            the heading.
        reads: Path globs the cell reads, relative to the working directory.
        writes: Path globs the cell writes; empty means it only prints.
        guard: Run only when ``ANALYSIS_READY``, printing ``guard`` otherwise.
        role: ``step``, or ``library`` for the shared imports.
    """

    step: str
    code: str
    title: str | None = None
    emoji: str = ""
    blurb: str = ""
    theme: str = "__init__.py"
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    guard: str | None = None
    role: str = "step"

    def cell_metadata(self) -> dict:
        tag: dict = {"role": self.role}
        if self.role == "step":
            tag["step"] = self.step
            if self.reads:
                tag["reads"] = list(self.reads)
            if self.writes:
                tag["writes"] = list(self.writes)
            if self.guard is not None:
                tag["guarded_on"] = "ANALYSIS_READY"
        return {"itbtc": tag}


@dataclasses.dataclass(frozen=True)
class Phase:
    number: str
    title: str
    emoji: str
    blurb: str
    theme: str
    contents: tuple[str | Vendor | Step, ...] = ()


def _html_phase(phase: Phase) -> str:
    start, end, accent, head, body = MODULE_THEME[phase.theme]
    return (
        "##\n\n"
        f'<a id="section-{phase.number}"></a>\n\n'
        f'<div style="background: linear-gradient(135deg, {start}, {end}); border-left: 4px solid {accent}; '
        f'border-radius: 8px; padding: 10px 18px;">\n'
        f'  <h2 style="color: {head}; margin: 0 0 4px; font-size: 1.35em;">'
        f"{phase.emoji} {phase.number} · {phase.title}</h2>\n"
        f'  <p style="color: {body}; margin: 0; font-size: 0.95em;">{phase.blurb}</p>\n'
        "</div>"
    )


def _html_item(theme: str, emoji: str, title: str, blurb: str) -> str:
    start, end, accent, head, body = MODULE_THEME[theme]
    return (
        "###\n\n"
        f'<div style="background: linear-gradient(90deg, {start}, {end}); border-left: 3px solid {accent}; '
        f'border-radius: 6px; padding: 6px 14px;">'
        f'<h3 style="display: inline; color: {head}; font-size: 1em; margin: 0;">{emoji} {title}</h3> '
        f'<p style="display: inline; color: {body}; font-size: 0.9em; margin: 0;">· {blurb}</p>'
        "</div>"
    )


def _html_vendor(path: str) -> str:
    _, _, _, head, _ = MODULE_THEME["upstream.py"]
    raw = (UPSTREAM_FIXTURES / path).read_bytes().decode("utf-8")
    emptied = sum(1 for line in raw.splitlines() if line and not line.strip())
    if path in UPSTREAM_TRIMMED:
        kept, dropped = UPSTREAM_TRIMMED[path]
        state = f"hanya {', '.join(kept)} dipertahankan; impor {', '.join(dropped)} dihapus"
    elif path == "LICENSE":
        state = "lisensi repositori, disertakan bersama salinan"
    elif emptied:
        state = f"tidak diubah, kecuali {emptied} baris berisi spasi saja dikosongkan"
    else:
        state = "tidak diubah"
    link = (f'<a href="{permalink(path)}" style="color: {head};">'
            f"thuml/iTransformer@{UPSTREAM_COMMIT[:7]}</a>")
    return _html_item("upstream.py", "📦", f"<code>{path}</code>", f"{link} · MIT · {state}")


def _html_artifacts(step: Step) -> str:
    """The MENULIS / MEMBACA line under a step's heading, from its declared paths."""
    parts = []
    if step.writes:
        parts.append('<span style="color: #ffd166; font-weight: 600;">MENULIS</span> '
                     + " ".join(f"<code>{w}</code>" for w in step.writes))
    if step.reads:
        parts.append('<span style="color: #9ec5fe; font-weight: 600;">MEMBACA</span> '
                     + " ".join(f"<code>{r}</code>" for r in step.reads))
    if not parts:
        return ""
    return ('\n\n<div style="background: #0e0e12; border-left: 3px solid #6c757d; '
            'border-radius: 0 6px 6px 0; padding: 4px 14px; font-size: 0.82em; color: #cfcfcf;">'
            + " &nbsp;·&nbsp; ".join(parts) + "</div>")


def guarded(body: str, what: str) -> str:
    """Run ``body`` only once the grid is complete and the session has time to analyse it."""
    head = (
        "if not ANALYSIS_READY:\n"
        f'    print("{what}: skipped until the 900-run grid is complete and the session '
        'has time to analyse it.")\n'
        "else:\n"
    )
    return head + textwrap.indent(body, "    ")


def artifact_map_cell(cells: list[dict]) -> str:
    """The index of step cells and what each reads and writes, from the emitted cells."""
    rows = []
    for index, cell in enumerate(cells):
        tag = cell.get("metadata", {}).get("itbtc", {})
        if tag.get("role") == "step":
            rows.append((index, tag.get("step", "?"), list(tag.get("writes", [])),
                         list(tag.get("reads", []))))
    literal = "".join(f"    {row!r},\n" for row in rows)
    return (
        "_ARTIFACT_MAP = [\n"
        f"{literal}"
        "]\n"
        "\n"
        'print("ARTIFACT MAP: cell, step, writes, reads")\n'
        "for _i, _slug, _w, _r in _ARTIFACT_MAP:\n"
        '    print(f"  cell {_i:>3}  {_slug}")\n'
        "    for _p in _w:\n"
        '        print(f"            writes {_p}")\n'
        "    for _p in _r:\n"
        '        print(f"            reads  {_p}")\n'
    )


# -- step cells --------------------------------------------------------------

CODE_SETUP = r'''import os
import subprocess
import sys
import time
from pathlib import Path

# The Kaggle session wall runs from this cell, so the session budget does too.
SESSION_T0 = globals().get("SESSION_T0", time.perf_counter())

WEEKLY_GPU_HOURS_REMAINING = None  # from Kaggle's quota meter, before Run All
SESSION_ALREADY_USED_H = 0.0       # hours this session ran before the first cell
SESSION_LIMIT_H = 11.5             # below the 12-hour session wall
SAVE_RESERVE_H = 0.75              # kept free for saving the output
ANALYSIS_ONLY = False              # True renders a complete saved grid without training

ON_KAGGLE = Path("/kaggle/working").exists()
WORK = (Path("/kaggle/working") if ON_KAGGLE else Path.cwd()).resolve()
ARTIFACTS = WORK / "artifacts"
VENDOR = WORK / "vendor" / "thuml_iTransformer"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
for _package in ("layers", "model", "utils"):
    (VENDOR / _package).mkdir(parents=True, exist_ok=True)
    (VENDOR / _package / "__init__.py").write_text("", encoding="utf-8")
if Path.cwd().resolve() != WORK:
    os.chdir(WORK)


def ensure(module: str, pip_name: str | None = None) -> None:
    """Install a package only when the image lacks it; the image's versions win."""
    try:
        __import__(module)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or module])


for _module, _pip in (("polars", None), ("pyarrow", None), ("numpy", None), ("torch", None),
                      ("sklearn", "scikit-learn"), ("scipy", None), ("statsmodels", None),
                      ("arch", None), ("matplotlib", None)):
    ensure(_module, _pip)

if not ON_KAGGLE:
    raise RuntimeError("This notebook runs on Kaggle; locally, run the pytest suite instead.")
if not ANALYSIS_ONLY:
    if WEEKLY_GPU_HOURS_REMAINING is None:
        raise ValueError("Set WEEKLY_GPU_HOURS_REMAINING from Kaggle's quota meter, then Run All.")
    if not 0 <= float(WEEKLY_GPU_HOURS_REMAINING) <= 30 or not 0 <= SESSION_ALREADY_USED_H < 12:
        raise ValueError("Use a remaining quota in [0, 30] h and an elapsed session time in [0, 12) h.")
    import torch
    _gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(_gpus) != 2 or not all("T4" in name for name in _gpus):
        raise RuntimeError(f"Select the GPU T4 x2 accelerator before training. Detected: {_gpus}")


def looks_like_parquet(path: Path) -> bool:
    """True only for a file that starts and ends with the parquet magic ``PAR1``."""
    try:
        with path.open("rb") as handle:
            if handle.read(4) != b"PAR1":
                return False
            handle.seek(-4, 2)
            return handle.read(4) == b"PAR1"
    except OSError:
        return False


def find_parquet() -> Path:
    """Find BTCUSDT_1h.parquet under the inputs at any depth, data/raw/ copies first.

    Discovery is by file name, never by dataset slug, and the file is never
    downloaded here: a new download would be a different input.
    """
    patterns = ("data/raw/BTCUSDT_1h.parquet", "*/data/raw/BTCUSDT_1h.parquet",
                "BTCUSDT_1h.parquet", "*/BTCUSDT_1h.parquet", "*/*/BTCUSDT_1h.parquet")
    roots = [WORK, Path("/kaggle/input")] if ON_KAGGLE else [WORK, WORK.parent]
    rejected: list[str] = []

    def accept(candidate: Path):
        if candidate.is_file() and looks_like_parquet(candidate):
            return candidate.resolve()
        rejected.append(str(candidate))
        return None

    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for hit in sorted(root.glob(pattern)):
                if (found := accept(hit)) is not None:
                    return found
    for root in roots:
        if not root.exists():
            continue
        hits = [h for h in sorted(root.rglob("BTCUSDT_1h.parquet")) if h.is_file()]
        valid = [h for h in hits if looks_like_parquet(h)]
        rejected += [str(h) for h in hits if h not in valid]
        if valid:
            preferred = [h for h in valid if h.parent.name == "raw"]
            chosen = (preferred or valid)[0]
            if len(valid) > 1:
                print(f"note: {len(valid)} copies of BTCUSDT_1h.parquet under {root}; using {chosen}")
            return chosen.resolve()
    if rejected:
        raise FileNotFoundError(
            "found candidates but none passed the PAR1 magic check at both ends: "
            f"{rejected}. A truncated upload, a Git LFS pointer, or the wrong file under the right name.")
    raise FileNotFoundError(
        f"BTCUSDT_1h.parquet not found under {[str(r) for r in roots]}. Attach data/raw/ as a "
        "Kaggle Dataset; it is not downloaded here, because a new download is a different input.")


PARQUET = find_parquet()
os.environ["ITBTC_PARQUET"] = str(PARQUET)

import numpy as np
import polars as pl
import torch

print(f"work      {WORK}")
print(f"parquet   {PARQUET}  ({PARQUET.stat().st_size / 1e6:.1f} MB)")
print(f"polars {pl.__version__} | torch {torch.__version__} | numpy {np.__version__}")
print(f"CUDA devices: {torch.cuda.device_count()}")
for _i in range(torch.cuda.device_count()):
    _cap = torch.cuda.get_device_capability(_i)
    print(f"  cuda:{_i}  {torch.cuda.get_device_name(_i)}  sm_{_cap[0]}{_cap[1]}")
'''

CODE_DATA = r'''_digest = hashlib.sha256(PARQUET.read_bytes()).hexdigest()
if _digest != INPUT_SHA256:
    raise RuntimeError(f"{PARQUET.name} has sha256 {_digest[:12]}, not the study's input "
                       f"{INPUT_SHA256[:12]}; attach the original data/raw/BTCUSDT_1h.parquet.")

bars = usable_mask(load_bars(PARQUET))
print(f"bars {bars.height:,}  usable {int(bars['usable'].sum()):,}  unusable {int((~bars['usable']).sum())}")
print(bars.filter(~pl.col("usable")).select(["open_time", "zero_volume", "flat_bar", "zero_trades"]))

budgets = budget_table(bars)
drift = [(b.label, b.summary.break_runs, b.summary.excluded_positions, b.windows_measured,
          COMMITTED_TRAIN_BUDGET[b.label])
         for b in budgets
         if (b.summary.break_runs, b.summary.excluded_positions, b.windows_measured)
         != COMMITTED_TRAIN_BUDGET[b.label]]
assert not drift, f"window budget differs from docs/ORIGIN_WINDOW_BUDGET.md: {drift}"
print(pl.DataFrame([
    {"origin": b.label, "train_windows": b.windows_measured, "loss_pct": round(b.loss_pct, 2),
     "closed_form_agrees": b.closed_form_agrees,
     **{f"B{i}": n for i, n in enumerate(b.test_block_starts, start=1)}}
    for b in budgets
]))
'''

CODE_FEATURES = r'''features = build_features(bars)
print(f"feature frame {features.height:,} rows x {len(VARIATE_ORDER)} variates; "
      f"{int(bars['usable'].sum()) - features.height} rows dropped, one per segment")
for k in K_LADDER:
    print(f"  K={k:>2}: {ladder_columns(k)}")
print(features.select(VARIATE_ORDER).describe().filter(
    pl.col("statistic").is_in(["mean", "std", "min", "max"])))
_rs = features["log_rogers_satchell"]
print(f"log_rogers_satchell at the log(1e-9) floor: {int((_rs <= np.log(1e-9) + 1e-9).sum())} bars")
assert all(features.select([pl.col(c).is_finite().all() for c in VARIATE_ORDER]).row(0)), \
    "a variate is non-finite"
'''

CODE_KEFF = r'''gate = gate_pr(features, k=8)
print(gate_verdict(gate))

t0 = time.perf_counter()
keff_tbl = keff_table(features)
print(f"\nmeasured in {time.perf_counter() - t0:.0f}s on each origin's training sub-block")
print(keff_tbl.group_by("k").agg(
    pl.col("pr_raw").mean().alias("PR_raw"),
    pl.col("pr_raw").std().alias("PR_raw_sd"),
    pl.col("pr_window_norm").mean().alias("PR_windownorm"),
    pl.col("stable_rank_lookback").mean().alias("stable_rank"),
    pl.col("pr_lookback_ratio").mean().alias("crosslag_share"),
).sort("k"))
print(f"corr(K, K_eff) = {corr_k_keff(keff_tbl):.4f}")
keff_tbl.write_parquet(ARTIFACTS / "keff_table.parquet")
'''

CODE_UPSTREAM = r'''verify_upstream(VENDOR)
UPSTREAM = load_upstream(VENDOR)
print(f"{UPSTREAM_REPO} at {UPSTREAM_COMMIT} ({UPSTREAM_LICENSE})")
for _path in sorted(UPSTREAM_FILES):
    _state = "imports trimmed" if _path in UPSTREAM_TRIMMED else "unchanged"
    print(f"  {_path:32s} sha256 {UPSTREAM_FILES[_path][:12]}  {_state}")
print({tag: f"{cls.__module__}.{cls.__qualname__}" for tag, cls in UPSTREAM.items()})
'''

CODE_DIGEST = r'''# There is no git repository on Kaggle, so the package digest is pinned at export;
# it equals code_sha256() of the same source in a checkout.
CODE_SHA256_OVERRIDE = "{digest}"

_sentinels = ("ORIGINS", "__all__", "build_segments", "count_windows", "budget_table",
              "build_features", "efficiency_table", "build_origin_tensors", "keff_table",
              "load_upstream", "ITransformerConfig", "code_sha256", "RidgeConfig",
              "seed_average", "pair_matrix", "manifest", "build_report")
_missing = [name for name in _sentinels if name not in globals()]
assert not _missing, f"definition cells have not run: {_missing}; run the cells above in order"
assert "itransformer_btc" not in sys.modules, "an installed itransformer_btc was imported"

print(f"code_sha256 {code_sha256()}")
_status = {"copied": "copied from the official repository", "library": "imported and called",
           "own": "written here from the published method"}
print(f"\nsource of each component ({len(SOURCE_PROVENANCE)})")
for _prov in SOURCE_PROVENANCE:
    print(f"\n{_prov.component}\n  in         {_prov.module}\n"
          f"  status     {_prov.status} ({_status[_prov.status]})\n  reference  {_prov.reference}")
    if _prov.repo:
        print(f"  code       {_prov.repo} ({_prov.licence}, accessed {_prov.accessed})")
    if _prov.adapted:
        print(f"  adapted    {_prov.adapted}")
'''

CODE_INVARIANTS = r'''device = torch.device("cpu") if ANALYSIS_ONLY else visible_devices()[0]
print(f"device {device}")

set_seed(42)
probe = ITransformerConfig().build().to(device).eval()
base, scaled = scale_invariance_check(probe, torch.randn(64, SEQ_LEN, 8, device=device),
                                      torch.randn(64, PRED_LEN, device=device), c=100.0)
print(f"use_norm invariance: MSE(x) {base:.8f}, MSE(100x)/100^2 {scaled:.8f}")
assert abs(base - scaled) / base < 1e-3, "use_norm inactive: MSE(c x)/c^2 differs from MSE(x)"

for tag, cfg in (("itr", ITransformerConfig(dropout=0.0)), ("vtr", VanillaConfig(k=8, dropout=0.0))):
    set_seed(42)
    plumb = cfg.build().to(device).train()
    xs, ys = torch.randn(8, SEQ_LEN, 8, device=device), torch.randn(8, PRED_LEN, device=device)
    opt = torch.optim.Adam(plumb.parameters(), lr=1e-3)
    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(plumb.forecast_target(xs), ys)
        loss.backward()
        opt.step()
    print(f"{tag} one-batch overfit (dropout 0, 300 steps): {loss.item():.2e}")
    assert loss.item() < 1e-3, f"{tag} cannot overfit one batch; the training path is broken"

print(pl.DataFrame([{"K": k, "itr": ITransformerConfig().build().n_parameters(),
                     "vtr": VanillaConfig(k=k).build().n_parameters(),
                     "rdg": RidgeConfig(k=k).build().n_parameters()} for k in K_LADDER]))

naive = pl.DataFrame([
    {"origin": o.label, "mu_g": float(t.scaler.mean[0]), "sigma_g": float(t.scaler.std[0]),
     "mu_over_sigma": t.scaler.target_mu_over_sigma, "naive_rw_z": t.naive_rw_z, "n_train": len(t.train)}
    for o in ORIGINS
    for t in [build_origin_tensors(features, o, 1, train_window_limit=TRAIN_WINDOW_LIMIT,
                                   selection_seed=SELECTION_SEED)]
])
print(naive)
naive.write_parquet(ARTIFACTS / "naive_rw_by_origin.parquet")
'''

CODE_PILOT = r'''roots = discover_roots(ARTIFACTS)
SESSION_GUARD = BudgetGuard(
    0.0 if ANALYSIS_ONLY else min(float(WEEKLY_GPU_HOURS_REMAINING),
                                 max(0.0, SESSION_LIMIT_H - SESSION_ALREADY_USED_H)),
    SAVE_RESERVE_H, started_at=SESSION_T0)
DEVICES = [torch.device("cpu")] if ANALYSIS_ONLY else visible_devices()
print(f"devices {[str(d) for d in DEVICES]}; "
      f"{max(0.0, SESSION_GUARD.remaining_s) / 3600:.2f} h before the save reserve")

if ANALYSIS_ONLY:
    print("pilot skipped: rendering saved results only")
else:
    PILOT = pilot(features, devices=DEVICES, out_root=ARTIFACTS, roots=roots,
                  log=lambda msg: print(msg, flush=True))
    print(PILOT)
    (ARTIFACTS / "pilot.json").write_text(json.dumps(
        {"rows": list(PILOT.rows), "mean_wall_s": PILOT.mean_wall_s(),
         "design_digest": design_digest()}, indent=2), encoding="utf-8")
    print()
    print(session_plan(PILOT.mean_wall_s(), pending(manifest(), roots), devices=len(DEVICES),
                       session_left_h=max(0.0, SESSION_GUARD.remaining_s) / 3600,
                       usable_session_h=SESSION_LIMIT_H - SAVE_RESERVE_H,
                       weekly_left_h=float(WEEKLY_GPU_HOURS_REMAINING)))
    print(f"\ndesign digest {design_digest()}")
'''

CODE_GRID = r'''DESIGN_FREEZE_SHA256 = None  # the design digest printed by the pilot, once the design is frozen

for _name in ("probe", "plumb", "xs", "ys", "opt", "loss"):
    globals().pop(_name, None)
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

ALL = manifest()
roots = discover_roots(ARTIFACTS)
attached, available = resume_check(ALL, roots, ARTIFACTS)
print(f"manifest {len(ALL)} runs; attached from earlier sessions {attached}; complete here {available}")

try:
    FROZEN_DESIGN = require_frozen_design(DESIGN_FREEZE_SHA256, ALL)
except RuntimeError as exc:
    FROZEN_DESIGN = None
    print(f"GRID NOT STARTED: {exc}")

summary = None
if FROZEN_DESIGN and not ANALYSIS_ONLY:
    summary = execute_parallel(pending(ALL, roots), features, devices=DEVICES, out_root=ARTIFACTS,
                               roots=roots, guard=SESSION_GUARD,
                               log=lambda msg: print(msg, flush=True))
    print(summary)

left = pending(ALL, [ARTIFACTS])
GRID_COMPLETE = FROZEN_DESIGN is not None and not left
ANALYSIS_READY = GRID_COMPLETE and (ANALYSIS_ONLY or SESSION_GUARD.remaining_s > 3600)
status = {
    "code_sha256": code_sha256(), "input_sha256": _input_sha256()[0],
    "design_digest": design_digest(ALL), "design_frozen": FROZEN_DESIGN is not None,
    "manifest_runs": len(ALL), "grid_complete": GRID_COMPLETE, "analysis_ready": ANALYSIS_READY,
    "pending_run_ids": [c.run_id for c in left],
    "partial_checkpoints": sorted(p.name for p in (ARTIFACTS / "checkpoints").glob("*.pt")),
    "elapsed_session_h": (time.perf_counter() - SESSION_T0) / 3600 + SESSION_ALREADY_USED_H,
    "weekly_quota_entered_h": WEEKLY_GPU_HOURS_REMAINING,
    "summary": asdict(summary) if summary else None,
}
(ARTIFACTS / "session_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
print(f"grid complete: {GRID_COMPLETE} ({len(left)} pending); analysis ready: {ANALYSIS_READY}")
if not GRID_COMPLETE and FROZEN_DESIGN:
    print("Save Version, attach this output to the next session, update the quota, then Run All.")
if GRID_COMPLETE and not ANALYSIS_READY:
    print("Less than an hour left: set ANALYSIS_ONLY = True and render in a CPU session.")
'''

CODE_EVALUATE = r'''REPORT = build_report(ARTIFACTS, bars, features, roots=discover_roots(ARTIFACTS),
                      bootstrap_b=9_999, seed=42, log=lambda msg: print(msg, flush=True))
NUMBERS = REPORT.numbers
print(NUMBERS["inference_status"])
print(pl.DataFrame([
    {"model": r["model"], "RelMSE": r["rel_mse"], "R2_oos": r["r2_oos"], "SE": r["se_across_origins"],
     "seed_std": r["seed_std"], "origins": r["n_origins"], "MCS90": r["in_mcs_90"], "MCS75": r["in_mcs_75"]}
    for r in NUMBERS["main_results"]
]))
'''

CODE_CONTRASTS = r'''print(pl.DataFrame(NUMBERS["contrasts"]).select(
    "claim", "left", "right", "mean_diff", "se", "ci_low", "ci_high", "n_origins", "left_better"))
print("mean_diff = RelMSE(iTransformer) - RelMSE(other); negative means the iTransformer is better.")
print(pl.DataFrame(NUMBERS["comparisons"]["pairs"]).filter(pl.col("family") == "cross-model").select(
    "left", "right", "t_cluster", "p_raw", "p_romano_wolf_family", "T_min"))
'''

CODE_RQ = r'''rq1, rq2 = NUMBERS["rq1"], NUMBERS["rq2"]
print(pl.DataFrame(rq1["rung_effects"]))
print(f"RelMSE change K4 to K8 {rq1['delta_4_to_8']:+.6f}; K8 to K12 {rq1['delta_8_to_12']:+.6f}")
print(rq1["tost"])
print(f"J test: K plus K_eff p = {rq1['j_test_k_augmented_by_keff']['p']:.4f}; "
      f"K_eff plus K p = {rq1['j_test_keff_augmented_by_k']['p']:.4f}")
print(f"\nRQ2: beta1 {rq2['beta1']:+.6f} (cluster SE {rq2['cluster_se']:.6f}, G = {rq2['G']}); "
      f"wild bootstrap p {rq2['p_rademacher']:.4f} (Rademacher), {rq2['p_webb']:.4f} (Webb); "
      f"minimum detectable slope {rq2['minimum_detectable_beta1']:+.6f}")
print(pl.DataFrame(rq2["stride5"]))
print(f"coverage covariate: {rq2['coverage_covariate']}")
'''

CODE_DIRECTION = r'''print(pl.DataFrame(NUMBERS["directional_accuracy"]).select(
    "model", "k", "n_origins",
    *[f"{part}_{v}" for v in ("1h", "24h", "cum") for part in ("dda", "wins")]))
print("dda = DA minus each origin's majority-sign rate, from test-period frequencies, so the "
      "comparison is conservative; Pesaran-Timmermann counts (pt_*) are diagnostic only.")
'''

CODE_REPORT = r'''PAPER = WORK / "paper"
PAPER.mkdir(parents=True, exist_ok=True)
(PAPER / "paper_numbers.json").write_text(json.dumps(NUMBERS, indent=2, default=float), encoding="utf-8")
for _table in render_tables(NUMBERS, PAPER / "tables"):
    print(f"  {_table.name}")
render_figures(REPORT, PAPER / "figures", log=print)
(PAPER / "panels").mkdir(parents=True, exist_ok=True)
for _name, _frame in (("seed_averaged_cells", REPORT.seed_avg), ("amplification_panel", REPORT.amplification),
                      ("rolling_pr", REPORT.rolling_pr), ("rolling_ols_r2", REPORT.rolling_r2),
                      ("directional_accuracy", REPORT.da_summary)):
    _frame.write_parquet(PAPER / "panels" / f"{_name}.parquet")
print(f"wrote {PAPER}")
'''

CODE_SYNC = r'''# ============================================================================
# SINKRON KE src/ - NONAKTIF. Hapus '# ' pada empat baris terakhir untuk memakai.
# ============================================================================
#
# Menulis src/itransformer_btc/ dari sel definisi notebook yang SUDAH DISIMPAN,
# lalu memeriksa bahwa hasilnya mem-flatten kembali byte-identik; jika tidak,
# berkas dipulihkan. Sel vendor tidak pernah ditulis, hanya dicocokkan.
#
# Syarat: checkout lokal (Kaggle tidak membawa tools/ maupun src/). Impor baru
# ditambahkan di sel Library dan di metadata itbtc.projection_imports sel modulnya.
# Sesudahnya: python tools/build_notebook.py --check, lalu commit src/ dan
# notebook bersama.
#
# import subprocess, sys
# _sync = subprocess.run([sys.executable, "-X", "utf8", "tools/notebook_to_src.py",
#                         "notebooks/btc_walkforward_3model.ipynb"],
#                        capture_output=True, text=True, encoding="utf-8")
# print(_sync.stdout or _sync.stderr)
'''

PREDS = ("artifacts/preds/*.parquet", "artifacts/meta/*.json")
PARQUET_INPUT = ("data/raw/BTCUSDT_1h.parquet",)

MD_TITLE = """#

<div style="background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); border-radius: 12px; padding: 16px 24px;">
  <h1 style="color: #e0aaff; font-size: 1.9em; margin: 0 0 6px;">BTCUSDT 1 jam · iTransformer, Transformer, Ridge</h1>
  <p style="color: #ddd6fe; margin: 0 0 6px;">Walk-forward 15 origin × K ∈ {1, 4, 8, 12} × 5 seed × 3 model = 900 run pada horizon 24 jam, dibandingkan dengan Naive-RW. Kode arsitektur disalin dari repositori resmi pada commit terkunci; training berjalan di Kaggle GPU T4 × 2, satu run per GPU.</p>
  <p style="color: #ddd6fe; margin: 0;">Kaggle: lampirkan dataset <code>BTCUSDT_1h.parquet</code>, pilih <b>GPU T4 × 2</b>, isi <code>WEEKLY_GPU_HOURS_REMAINING</code> di sel setup, lalu <b>Save Version → Save &amp; Run All</b>.</p>
</div>
"""

PHASES: tuple[Phase, ...] = (
    Phase("01", "Persiapan lingkungan dan konfigurasi", "🔧",
          "Sesi Kaggle GPU T4 × 2, dataset input, impor bersama, dan konfigurasi penelitian.",
          "config.py", (
              Step("artifact_map", "{artifact_map}", "Peta artefak", "🗺️",
                   "Sel mana menulis apa, dari metadata tiap sel langkah.", "budget.py"),
              Step("setup", CODE_SETUP, "Setup sesi", "🧰",
                   "Kuota, 2 × T4, folder kerja, dan parquet input dicari di kedalaman berapa pun.",
                   "train.py", reads=PARQUET_INPUT),
              Step("library", "{library}", "Library", "📚",
                   "Semua impor tingkat modul, sekali untuk seluruh notebook.", role="library"),
              "config.py",
              "__init__.py",
          )),
    Phase("02", "Muat data dan audit kualitas", "📥",
          "Gap memutus deret tanpa imputasi; jendela divalidasi lewat timestamp; anggaran jendela dicocokkan per origin.",
          "segments.py", (
              "segments.py", "windows.py", "budget.py",
              Step("data", CODE_DATA, "Muat bar dan audit anggaran", "📊",
                   "sha256 input dicocokkan, bar tak layak ditandai, anggaran jendela per origin harus sama persis.",
                   "budget.py", reads=PARQUET_INPUT),
          )),
    Phase("03", "Feature engineering dan eksplorasi", "🧪",
          "Dua belas variat per-bar F1–F5 tanpa rolling window, dan diagnostik efisiensi pasar.",
          "features.py", (
              "features.py",
              Step("features", CODE_FEATURES, "Bangun frame fitur", "🔬",
                   "Satu baris per segmen jatuh karena r butuh close sebelumnya; setiap variat harus finite.",
                   "features.py"),
              "efficiency.py",
          )),
    Phase("04", "Split walk-forward, scaling, dan K_eff", "🪟",
          "21 bulan latih dan 3 bulan validasi, purge 24 jam di kedua batas, scaler dari data latih saja, K_eff sebelum training.",
          "splits.py", (
              "splits.py", "keff.py",
              Step("keff", CODE_KEFF, "Ukur K_eff", "📐",
                   "Gerbang PR pada rentang sebelum origin pertama, lalu K_eff per origin pada sub-blok latih.",
                   "keff.py", writes=("artifacts/keff_table.parquet",)),
          )),
    Phase("05", "Model, baseline, dan fungsi training", "🧠",
          "Kode arsitektur disalin dari thuml/iTransformer pada commit terkunci dan diverifikasi sha256; "
          "Ridge dari scikit-learn; satu loop latih untuk kedua Transformer.",
          "model.py", (
              "upstream.py",
              *(Vendor(path) for path in ("LICENSE", "layers/Embed.py", "layers/SelfAttention_Family.py",
                                          "layers/Transformer_EncDec.py", "utils/masking.py",
                                          "model/iTransformer.py", "model/Transformer.py")),
              Step("upstream", CODE_UPSTREAM, "Verifikasi dan muat kode upstream", "🔐",
                   "Setiap salinan harus cocok dengan sha256 yang dipin sebelum model dibangun.",
                   "upstream.py", reads=(f"{VENDOR_TARGET}/**",)),
              "model.py", "train.py", "baselines.py",
          )),
    Phase("06", "Persiapan evaluasi dan eksekutor", "⚙️",
          "Metrik, perbandingan antarmodel, manifes 900 run, eksekutor dua GPU, dan laporan; digest kode dicatat sebelum training.",
          "runner.py", (
              "metrics.py", "comparisons.py", "runner.py", "report.py",
              Step("code_digest", CODE_DIGEST, "Provenance kode", "🔏",
                   "Digest paket yang dipin, sumber tiap algoritma, dan cek bahwa semua sel definisi sudah berjalan.",
                   "train.py"),
          )),
    Phase("07", "Pemeriksaan sebelum training", "🛠️",
          "Invarian skala use_norm, overfit satu batch untuk kedua Transformer, jumlah parameter, dan Naive-RW per origin.",
          "efficiency.py", (
              Step("invariants", CODE_INVARIANTS, writes=("artifacts/naive_rw_by_origin.parquet",)),
          )),
    Phase("08", "Validasi pilot dan rencana sesi", "🛡️",
          "Tiga model × empat K pada validation origin pertama, satu run per GPU: cek teknis dan waktu, "
          "tanpa memilih apa pun; lalu rencana sesi dan digest desain untuk dibekukan.",
          "comparisons.py", (
              Step("pilot", CODE_PILOT, writes=("artifacts/validation/*.json", "artifacts/pilot.json")),
          )),
    Phase("09", "Training grid walk-forward", "🚀",
          "900 run, satu worker per GPU, resume otomatis dari output sesi sebelumnya; "
          "berjalan hanya setelah digest desain dibekukan di sel ini.",
          "splits.py", (
              Step("grid", CODE_GRID, reads=PARQUET_INPUT,
                   writes=("artifacts/preds/*.parquet", "artifacts/meta/*.json", "artifacts/weights/*.pt",
                           "artifacts/checkpoints/*.pt", "artifacts/session_status.json")),
          )),
    Phase("10", "Evaluasi model dan research questions", "📈",
          "C1 (iTransformer vs Ridge), C2 (iTransformer vs Transformer), RQ1, RQ2, dan akurasi arah dari "
          "prediksi tersimpan. Semua inferensi bersifat diagnostik: origin berbagi data latih.",
          "metrics.py", (
              Step("evaluate", CODE_EVALUATE, "Hasil utama", "📋",
                   "RelMSE dan R²_oos terhadap Naive-RW per model dan K, SE antar-origin, keanggotaan MCS.",
                   "metrics.py", reads=PREDS, guard="main results"),
              Step("contrasts", CODE_CONTRASTS, "C1 dan C2", "⚖️",
                   "Selisih RelMSE berpasangan per K dan uji keluarga cross-model.",
                   "comparisons.py", guard="C1 and C2"),
              Step("research_questions", CODE_RQ, "RQ1 dan RQ2", "🔢",
                   "Efek rung, TOST 8→12, uji-J K lawan K_eff; β₁ gap K1–K8 menurut umur model dengan MDE.",
                   "keff.py", guard="RQ1 and RQ2"),
              Step("direction", CODE_DIRECTION, "Akurasi arah", "🧭",
                   "DA-1h, DA-24h, dan DA-kumulatif dikurangi baseline mayoritas per origin.",
                   "features.py", guard="directional accuracy"),
          )),
    Phase("11", "Simpan hasil, tabel, dan figure", "💾",
          "paper_numbers.json, tabel LaTeX, figure, dan panel, semuanya dari prediksi tersimpan.",
          "report.py", (
              Step("report", CODE_REPORT, guard="tables and figures",
                   writes=("paper/paper_numbers.json", "paper/tables/*.tex", "paper/figures/*.pdf",
                           "paper/figures/*.png", "paper/panels/*.parquet")),
          )),
    Phase("12", "Lampiran — sinkronisasi lokal", "🔁",
          "Menulis src/ dari sel definisi notebook yang sudah disimpan; nonaktif, hanya untuk checkout lokal.",
          "__init__.py", (
              Step("sync_back", CODE_SYNC),
          )),
)


# -- assembly ----------------------------------------------------------------


def _lines(text: str) -> list[str]:
    """nbformat's source form: a list of lines, newlines kept."""
    return text.splitlines(keepends=True)


def _markdown(index: int, text: str) -> dict:
    return {"id": f"md-{index:02d}", "cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def _code(index: int, text: str, metadata: dict) -> dict:
    return {"id": f"code-{index:02d}", "cell_type": "code", "execution_count": None,
            "metadata": metadata, "outputs": [], "source": _lines(text)}


def program_digest(notebook: dict) -> str:
    """sha256 of every code cell's source, in order: what the notebook runs."""
    code = ["".join(c["source"]) for c in notebook["cells"] if c.get("cell_type") == "code"]
    return hashlib.sha256(json.dumps(code, ensure_ascii=False).encode("utf-8")).hexdigest()


def build() -> dict:
    """The whole notebook as an nbformat 4.5 dictionary."""
    laid_out = tuple(item for p in PHASES for item in p.contents if isinstance(item, str))
    if laid_out != MODULE_ORDER:
        raise SystemExit(f"PHASES lays out {laid_out}, but MODULE_ORDER is {MODULE_ORDER}")
    vendored = sorted(item.path for p in PHASES for item in p.contents if isinstance(item, Vendor))
    if vendored != sorted(vendor_files()):
        raise SystemExit(f"PHASES writes {vendored}, but the vendor copies are {sorted(vendor_files())}")

    cells: list[dict] = []

    def md(text: str) -> None:
        cells.append(_markdown(len(cells), text))

    def code(text: str, metadata: dict) -> None:
        cells.append(_code(len(cells), text, metadata))

    md(MD_TITLE + "\n**Daftar isi**\n\n" + "\n".join(
        f"- [{p.number} · {p.title}](#section-{p.number})" for p in PHASES))
    artifact_map_at = None
    for phase in PHASES:
        md(_html_phase(phase))
        for item in phase.contents:
            if isinstance(item, str):
                emoji, blurb = MODULE_HEADER[item]
                md(_html_item(item, emoji, f"<code>{item}</code>", blurb))
                code(flatten_module_body(item), {"itbtc": {"role": "module", "module": item}})
                continue
            if isinstance(item, Vendor):
                md(_html_vendor(item.path))
                code(vendor_cell(item.path), {"itbtc": {"role": "vendor", "path": item.path}})
                continue
            strip = _html_artifacts(item)
            if item.title is None:
                assert cells[-1]["cell_type"] == "markdown", f"{item.step} has no heading above it"
                cells[-1]["source"] = _lines("".join(cells[-1]["source"]) + strip)
            else:
                md(_html_item(item.theme, item.emoji, item.title, item.blurb) + strip)
            body = {"{library}": library_cell, "{artifact_map}": lambda: "# patched after layout\n"}.get(
                item.code, lambda: item.code.replace("{digest}", package_digest()))()
            if item.step == "artifact_map":
                artifact_map_at = len(cells)
            if item.guard is not None:
                body = guarded(body, item.guard)
            code(body, item.cell_metadata())
    assert artifact_map_at is not None, "no phase declares the artifact map step"
    cells[artifact_map_at]["source"] = _lines(artifact_map_cell(cells))

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook["metadata"]["itbtc_exported_program_sha256"] = program_digest(notebook)
    return notebook


def render(notebook: dict) -> str:
    """UTF-8 JSON with a trailing newline, the shape nbformat writes."""
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def _cell_key(cell: dict) -> tuple | None:
    """Stable identity for a code cell, independent of its position."""
    tag = cell.get("metadata", {}).get("itbtc")
    if not tag:
        return None
    role = tag.get("role")
    if role == "module":
        return ("module", tag.get("module"))
    if role == "vendor":
        return ("vendor", tag.get("path"))
    if role == "step":
        return ("step", tag.get("step"))
    if role == "library":
        return ("library",)
    return None


def carry_outputs(notebook: dict, previous: dict) -> tuple[int, int, int]:
    """Carry executed outputs from ``previous`` onto byte-identical cells of ``notebook``.

    Any change to the ordered program drops every output, because an unchanged
    cell can still depend on a changed one. Definition cells never keep outputs.

    Returns:
        ``(carried, dropped because the source changed, dropped because the cell is new)``.
    """
    program = lambda nb: ["".join(c["source"]) for c in nb.get("cells", [])
                          if c.get("cell_type") == "code"]
    if program(notebook) != program(previous):
        dropped = sum(bool(c.get("outputs")) for c in previous.get("cells", []))
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
        return 0, dropped, 0

    old: dict[tuple, dict] = {}
    by_source: dict[str, list[dict]] = {}
    for cell in previous.get("cells", []):
        if cell.get("cell_type") != "code" or not cell.get("outputs"):
            continue
        key = _cell_key(cell)
        if key is not None:
            old[key] = cell
        by_source.setdefault("".join(cell["source"]), []).append(cell)

    carried = changed = missing = 0
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        key = _cell_key(cell)
        if key is not None and key[0] == "module":
            continue
        match = old.get(key) if key is not None else None
        if match is not None and _lines("".join(match["source"])) != cell["source"]:
            changed += 1
            continue
        if match is None:
            same = by_source.get("".join(cell["source"]), [])
            if len(same) == 1:
                match = same[0]
        if match is None:
            missing += 1
            continue
        cell["outputs"] = copy.deepcopy(match["outputs"])
        cell["execution_count"] = match.get("execution_count")
        carried += 1
    return carried, changed, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap or check the three-model notebook.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="exit 1 if the notebook and src/ disagree")
    mode.add_argument("--bootstrap", action="store_true",
                      help="write the notebook from src/; refused when it already exists")
    args = parser.parse_args(argv)
    if args.check:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from notebook_to_src import check_projection
        return check_projection(NOTEBOOK)
    if NOTEBOOK.exists():
        print(f"{NOTEBOOK.relative_to(ROOT).as_posix()} exists and is authoritative: edit it and "
              "export with its sync cell.", file=sys.stderr)
        return 1
    notebook = build()
    NOTEBOOK.write_text(render(notebook), encoding="utf-8", newline="\n")
    print(f"wrote {NOTEBOOK.relative_to(ROOT).as_posix()}: {len(notebook['cells'])} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

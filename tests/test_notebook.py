"""notebooks/btc_walkforward_3model.ipynb: structure, projection, provenance and a CPU run.

The notebook carries every module as one definition cell, the upstream copies as
``%%writefile`` cells, and the orchestration steps between them. These tests pin
what a Kaggle session would otherwise discover hours in: a missing name, a cell
out of order, a vendor copy that differs from the pinned file, or a step that
fails on the real data.
"""

from __future__ import annotations

import __future__
import ast
import builtins
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import symtable
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "btc_walkforward_3model.ipynb"
GENERATOR = ROOT / "tools" / "build_notebook.py"
PARQUET = ROOT / "data" / "raw" / "BTCUSDT_1h.parquet"
MAX_CELLS = 90

#: Writes a package function performs on the step's behalf, so the path is not in its body.
INDIRECT_WRITES = {
    "grid": {"artifacts/preds/*.parquet", "artifacts/meta/*.json", "artifacts/weights/*.pt",
             "artifacts/checkpoints/*.pt"},
    "pilot": {"artifacts/validation/*.json"},
}
#: Globals read but never bound, on purpose: both sit behind a value the notebook sets first.
ALLOWED_UNBOUND = {
    ("train.py", "code_sha256", "__file__"),  # after the CODE_SHA256_OVERRIDE check
    ("upstream.py", "vendor_root", "__file__"),  # only when load_upstream was never called
}
#: Package names a step may rebind with this session's value.
ALLOWED_SHADOWS = {"ARTIFACTS", "CODE_SHA256_OVERRIDE"}
#: Cross-reference tokens that mean nothing to a reader of the notebook alone.
STYLE = re.compile(r"\b[DA]\d{2}[a-z]?\b|§\s?\d")


def _load_generator():
    spec = importlib.util.spec_from_file_location("build_notebook", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    return _load_generator()


@pytest.fixture(scope="module")
def notebook() -> dict:
    assert NOTEBOOK.exists(), f"{NOTEBOOK} is missing; run tools/build_notebook.py --bootstrap"
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _tag(cell: dict) -> dict:
    return cell.get("metadata", {}).get("itbtc", {})


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _code(notebook: dict) -> list[tuple[int, dict]]:
    return [(i, c) for i, c in enumerate(notebook["cells"]) if c["cell_type"] == "code"]


def _steps(notebook: dict) -> dict[str, tuple[int, str]]:
    return {_tag(c)["step"]: (i, _source(c)) for i, c in _code(notebook) if _tag(c).get("role") == "step"}


def _inherited_flags(source: str) -> int:
    """``__future__`` flags a cell leaves for later cells, as IPython accumulates them."""
    flags = 0
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            for alias in node.names:
                flags |= getattr(__future__, alias.name).compiler_flag
    return flags


# -- layout ----------------------------------------------------------------------


def test_notebook_is_nbformat_with_gpu_metadata(notebook: dict) -> None:
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] >= 5
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    assert notebook["metadata"]["accelerator"] == "GPU"
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))


def test_the_layout_is_compact_with_twelve_phases(notebook: dict) -> None:
    cells = notebook["cells"]
    assert len(cells) <= MAX_CELLS
    phases = [re.search(r'id="section-(\d\d)"', _source(c)).group(1)
              for c in cells if c["cell_type"] == "markdown" and 'id="section-' in _source(c)]
    assert phases == [f"{n:02d}" for n in range(1, 13)]
    title = _source(cells[0])
    assert all(f"(#section-{n})" in title for n in phases), "the table of contents misses a phase"
    for index, cell in _code(notebook):
        above = cells[index - 1]
        assert above["cell_type"] == "markdown", f"code cell {index} has no heading above it"


def test_every_module_has_exactly_one_cell_in_order(notebook: dict, generator) -> None:
    modules = [(_tag(c)["module"], _source(c)) for _, c in _code(notebook) if _tag(c).get("role") == "module"]
    assert [name for name, _ in modules] == list(generator.MODULE_ORDER)
    for name, body in modules:
        assert body == generator.flatten_module_body(name), f"{name}'s cell differs from src/"


def test_the_library_cell_carries_every_module_import(notebook: dict, generator) -> None:
    library = next(_source(c) for _, c in _code(notebook) if _tag(c).get("role") == "library")
    missing = [line for line in generator.library_cell().splitlines() if line not in library.splitlines()]
    assert not missing, f"the Library cell lacks {missing}"


def test_the_pinned_digests_match_the_source(notebook: dict, generator) -> None:
    pinned = re.findall(r'^CODE_SHA256_OVERRIDE = "([0-9a-f]{64})"', _steps(notebook)["code_digest"][1], re.M)
    assert pinned == [generator.package_digest()]
    assert notebook["metadata"]["itbtc_exported_program_sha256"] == generator.program_digest(notebook)


def test_the_generator_check_passes() -> None:
    result = subprocess.run([sys.executable, str(GENERATOR), "--check"], capture_output=True,
                            text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr or result.stdout


def test_the_notebook_map_is_current() -> None:
    result = subprocess.run([sys.executable, str(ROOT / "tools" / "notebook_map.py"), "--check"],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr or result.stdout


def test_no_definition_cell_carries_outputs(notebook: dict) -> None:
    assert not [i for i, c in _code(notebook) if _tag(c).get("role") == "module" and c.get("outputs")]


def test_markdown_and_code_carry_no_cross_reference_tokens(notebook: dict) -> None:
    offenders = [(i, m.group(0)) for i, c in enumerate(notebook["cells"])
                 if _tag(c).get("role") != "vendor"
                 for m in [STYLE.search(_source(c))] if m]
    assert not offenders, offenders


# -- upstream copies -----------------------------------------------------------------


def test_vendor_cells_write_the_pinned_copies_and_nothing_else(notebook: dict, generator) -> None:
    from itransformer_btc.upstream import UPSTREAM_FILES, vendor_root

    vendor = {_tag(c)["path"]: _source(c) for _, c in _code(notebook) if _tag(c).get("role") == "vendor"}
    assert sorted(vendor) == sorted(generator.vendor_files())
    for path, source in vendor.items():
        magic, body = source.split("\n", 1)
        assert magic == f"%%writefile vendor/thuml_iTransformer/{path}"
        written = (body + "\n").encode("utf-8")
        assert written == (vendor_root() / path).read_bytes()
        assert hashlib.sha256(written).hexdigest() == UPSTREAM_FILES[path]
    # The empty package markers are the setup cell's job.
    assert '"__init__.py").write_text("", encoding="utf-8")' in _steps(notebook)["setup"][1]


def test_ipython_writes_exactly_the_vendor_text(notebook: dict) -> None:
    """IPython hands ``%%writefile`` the cell body with one final newline."""
    transformer = pytest.importorskip("IPython.core.inputtransformer2").TransformerManager()
    from itransformer_btc.upstream import vendor_root

    for _, cell in _code(notebook):
        if _tag(cell).get("role") != "vendor":
            continue
        call = ast.parse(transformer.transform_cell(_source(cell))).body[0].value
        magic, target, body = (arg.value for arg in call.args)
        assert (magic, target) == ("writefile", f"vendor/thuml_iTransformer/{_tag(cell)['path']}")
        assert body.encode("utf-8") == (vendor_root() / _tag(cell)["path"]).read_bytes()


def test_writefile_and_sys_path_appear_only_where_allowed(notebook: dict) -> None:
    for index, cell in _code(notebook):
        source, tag = _source(cell), _tag(cell)
        if tag.get("role") == "vendor":
            continue
        assert not re.search(r"^[ \t]*(%|![^=])", source, re.M), f"cell {index} runs a magic or shell command"
        if "sys.path" in source:
            assert tag.get("module") == "upstream.py", f"cell {index} touches sys.path"


# -- the steps -------------------------------------------------------------------------


def test_setup_refuses_anything_but_two_t4(notebook: dict) -> None:
    setup = _steps(notebook)["setup"][1]
    assert "len(_gpus) != 2" in setup and '"T4" in name' in setup
    assert "WEEKLY_GPU_HOURS_REMAINING = None" in setup


def test_the_pilot_and_the_grid_use_every_visible_device(notebook: dict) -> None:
    steps = _steps(notebook)
    assert "else visible_devices()" in steps["pilot"][1] and "devices=DEVICES" in steps["pilot"][1]
    grid = steps["grid"][1]
    assert "devices=DEVICES" in grid and "guard=SESSION_GUARD" in grid


def test_the_grid_waits_for_a_frozen_design(notebook: dict) -> None:
    grid = _steps(notebook)["grid"][1]
    assert re.search(r"^DESIGN_FREEZE_SHA256 = None", grid, re.M)
    assert "require_frozen_design(DESIGN_FREEZE_SHA256" in grid
    assert "resume_check(ALL, roots, ARTIFACTS)" in grid


def test_analysis_steps_wait_for_a_complete_grid(notebook: dict) -> None:
    for slug in ("evaluate", "contrasts", "research_questions", "direction", "report"):
        assert _steps(notebook)[slug][1].startswith("if not ANALYSIS_READY:"), slug


def test_the_sync_cell_is_last_and_inert(notebook: dict) -> None:
    last = notebook["cells"][-1]
    assert _tag(last).get("step") == "sync_back"
    assert ast.parse(_source(last)).body == []
    assert "notebooks/btc_walkforward_3model.ipynb" in _source(last)


def _witness(pattern: str) -> str:
    name = pattern.rsplit("/", 1)[-1]
    if "*" not in name:
        return name
    parts = [p for p in pattern.split("/") if "*" not in p]
    return parts[-1] if parts else pattern


def test_declared_writes_are_evidenced_and_mapped(notebook: dict) -> None:
    steps = _steps(notebook)
    for slug, (index, source) in steps.items():
        tag = _tag(notebook["cells"][index])
        for pattern in tag.get("writes", []):
            assert not pattern.startswith("/") and ".." not in pattern, pattern
            if pattern not in INDIRECT_WRITES.get(slug, set()):
                assert _witness(pattern) in source, f"{slug} declares {pattern} but never names it"
    literal = ast.parse(steps["artifact_map"][1]).body[0].value
    rows = ast.literal_eval(literal)
    assert rows == [(i, s, _tag(notebook["cells"][i]).get("writes", []),
                     _tag(notebook["cells"][i]).get("reads", [])) for s, (i, _) in steps.items()]


# -- names ---------------------------------------------------------------------------


def _binds(sym: symtable.Symbol) -> bool:
    return sym.is_assigned() or sym.is_imported()


def _unbound_reads(source: str, label: str, defined: set[str]) -> list[tuple[str, str]]:
    out = []
    stack = [symtable.symtable(source, label, "exec")]
    while stack:
        table = stack.pop()
        stack.extend(table.get_children())
        for sym in table.get_symbols():
            if sym.is_referenced() and sym.is_global() and not _binds(sym) and sym.get_name() not in defined:
                out.append((table.get_name(), sym.get_name()))
    return out


def _assigned_names(source: str, label: str) -> set[str]:
    return {s.get_name() for s in symtable.symtable(source, label, "exec").get_symbols() if _binds(s)}


def _imported_names(source: str) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out |= {a.asname or a.name.split(".")[0] for a in node.names}
    return out


def test_every_name_the_notebook_reads_is_bound_first(notebook: dict) -> None:
    """Static NameError check: a cell may read only what the package or an earlier cell binds."""
    pytest.importorskip("torch")
    namespace: dict = {"__name__": "__main__"}
    library = next(_source(c) for _, c in _code(notebook) if _tag(c).get("role") == "library")
    flags = _inherited_flags(library)
    exec(compile(library, "<library>", "exec"), namespace)
    modules = [(i, _tag(c)["module"], _source(c)) for i, c in _code(notebook) if _tag(c).get("role") == "module"]
    for index, name, body in modules:
        exec(compile(body, f"<cell {index}: {name}>", "exec", flags=flags), namespace)
    defined = set(namespace) | set(dir(builtins))

    offenders = [(name, scope, symbol) for _, name, body in modules
                 for scope, symbol in _unbound_reads(body, name, defined)
                 if (name, scope, symbol) not in ALLOWED_UNBOUND]
    running = set(defined)
    for index, cell in _code(notebook):
        if _tag(cell).get("role") in ("module", "vendor"):
            continue
        source = _source(cell)
        running |= _assigned_names(source, f"cell {index}")
        offenders += [(f"cell {index}", s, n) for s, n in _unbound_reads(source, f"cell {index}", running)]
    assert not offenders, f"read but never bound: {sorted(offenders)}"


def test_no_step_rebinds_a_package_name(notebook: dict) -> None:
    package: set[str] = set()
    for _, cell in _code(notebook):
        if _tag(cell).get("role") == "module":
            package |= _assigned_names(_source(cell), _tag(cell)["module"])
    offenders = []
    for index, cell in _code(notebook):
        if _tag(cell).get("role") in ("module", "vendor"):
            continue
        source = _source(cell)
        bound = _assigned_names(source, f"cell {index}") - _imported_names(source)
        offenders += [(index, n) for n in sorted((bound & package) - ALLOWED_SHADOWS)]
    assert not offenders, f"steps rebind package names: {offenders}"


def test_every_cell_parses_as_python_3_11(notebook: dict) -> None:
    for index, cell in _code(notebook):
        source, tag = _source(cell), _tag(cell)
        if tag.get("role") == "vendor":
            if not tag["path"].endswith(".py"):
                continue
            source = source.split("\n", 1)[1]
        ast.parse(source, feature_version=(3, 11))


# -- find_parquet, as the setup cell defines it --------------------------------------

PARQUET_BYTES = b"PAR1" + bytes(64) + b"PAR1"


def _find_parquet(notebook: dict, input_root: Path, work: Path):
    setup = _steps(notebook)["setup"][1]
    source = "\n\n".join(ast.get_source_segment(setup, node) for node in ast.parse(setup).body
                         if getattr(node, "name", "") in {"looks_like_parquet", "find_parquet"})
    namespace = {"Path": Path, "WORK": work, "ON_KAGGLE": True, "print": lambda *a, **k: None}
    exec(source.replace('Path("/kaggle/input")', f"Path(r'{input_root}')"), namespace)
    return namespace["find_parquet"]


@pytest.mark.parametrize("layout", [
    "btc1h-raw/BTCUSDT_1h.parquet",
    "datasets/owner/btc1h-raw/BTCUSDT_1h.parquet",
    "btc1h-raw/data/raw/BTCUSDT_1h.parquet",
    "a/b/c/d/e/BTCUSDT_1h.parquet",
])
def test_find_parquet_is_depth_independent(notebook: dict, layout: str, tmp_path) -> None:
    target = tmp_path / "input" / layout
    target.parent.mkdir(parents=True)
    target.write_bytes(PARQUET_BYTES)
    (tmp_path / "working").mkdir()
    assert _find_parquet(notebook, tmp_path / "input", tmp_path / "working")() == target.resolve()


def test_find_parquet_prefers_data_raw(notebook: dict, tmp_path) -> None:
    canonical = tmp_path / "input" / "repo" / "data" / "raw" / "BTCUSDT_1h.parquet"
    stray = tmp_path / "input" / "repo" / "scratch" / "copies" / "BTCUSDT_1h.parquet"
    for path in (canonical, stray):
        path.parent.mkdir(parents=True)
        path.write_bytes(PARQUET_BYTES)
    (tmp_path / "working").mkdir()
    assert _find_parquet(notebook, tmp_path / "input", tmp_path / "working")() == canonical.resolve()


@pytest.mark.parametrize("blob", [b"not a parquet", b"PAR1" + bytes(32),
                                  b"version https://git-lfs.github.com/spec/v1\n", b""])
def test_find_parquet_refuses_anything_that_is_not_a_parquet(notebook: dict, blob: bytes, tmp_path) -> None:
    target = tmp_path / "input" / "btc1h-raw" / "BTCUSDT_1h.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(blob)
    (tmp_path / "working").mkdir()
    with pytest.raises(FileNotFoundError, match="PAR1 magic check"):
        _find_parquet(notebook, tmp_path / "input", tmp_path / "working")()


def test_find_parquet_says_what_to_attach_when_nothing_is_there(notebook: dict, tmp_path) -> None:
    (tmp_path / "input").mkdir()
    (tmp_path / "working").mkdir()
    with pytest.raises(FileNotFoundError, match="Attach data/raw/"):
        _find_parquet(notebook, tmp_path / "input", tmp_path / "working")()


# -- a run on the real input without a GPU --------------------------------------------


@pytest.fixture(scope="module")
def executed(notebook: dict, tmp_path_factory):
    """Every cell but setup and sync, in one namespace, with ANALYSIS_ONLY on the CPU.

    The setup cell's values are supplied as it would bind them; vendor cells are
    written as ``%%writefile`` writes them. Training and analysis stay behind their
    gates, which is the path a first Kaggle session takes up to the grid.
    """
    pytest.importorskip("torch")
    if not PARQUET.exists():
        pytest.skip("the input parquet is not in this checkout")
    work = tmp_path_factory.mktemp("work")
    vendor = work / "vendor" / "thuml_iTransformer"
    for package in ("layers", "model", "utils"):
        (vendor / package).mkdir(parents=True)
        (vendor / package / "__init__.py").write_text("", encoding="utf-8")
    (work / "artifacts").mkdir()
    namespace: dict = {"__name__": "__main__"}
    library = next(_source(c) for _, c in _code(notebook) if _tag(c).get("role") == "library")
    flags = _inherited_flags(library)
    exec(compile(library, "<library>", "exec"), namespace)
    namespace.update(
        os=os, Path=Path, SESSION_T0=time.perf_counter(), WEEKLY_GPU_HOURS_REMAINING=30.0,
        SESSION_ALREADY_USED_H=0.0, SESSION_LIMIT_H=11.5, SAVE_RESERVE_H=0.75, ANALYSIS_ONLY=True,
        ON_KAGGLE=False, WORK=work, ARTIFACTS=work / "artifacts", VENDOR=vendor, PARQUET=PARQUET)
    previous = os.environ.get("ITBTC_PARQUET")
    os.environ["ITBTC_PARQUET"] = str(PARQUET)
    # A Kaggle kernel has no itransformer_btc package; the provenance step checks that.
    package = {k: sys.modules.pop(k) for k in list(sys.modules)
               if k == "itransformer_btc" or k.startswith("itransformer_btc.")}
    # Setup moves into WORK, and train.py's ARTIFACTS default is relative to it.
    cwd = os.getcwd()
    os.chdir(work)
    try:
        for index, cell in _code(notebook):
            tag, source = _tag(cell), _source(cell)
            if tag.get("role") == "library" or tag.get("step") in ("setup", "sync_back"):
                continue
            if tag.get("role") == "vendor":
                magic, body = source.split("\n", 1)
                target = work / magic.split(" ", 1)[1]
                target.write_bytes((body + "\n").encode("utf-8"))
                continue
            exec(compile(source, f"<cell {index}>", "exec", flags=flags), namespace)
    finally:
        os.chdir(cwd)
        sys.modules.update(package)
        if previous is None:
            os.environ.pop("ITBTC_PARQUET", None)
        else:
            os.environ["ITBTC_PARQUET"] = previous
    return namespace, work


def test_the_steps_run_on_the_real_input_up_to_the_frozen_design_gate(executed, generator) -> None:
    namespace, work = executed
    assert set(namespace["UPSTREAM"]) == {"itr", "vtr"}
    assert namespace["code_sha256"]() == generator.package_digest()
    assert namespace["naive"].height == 15
    assert (work / "artifacts" / "keff_table.parquet").exists()
    assert namespace["FROZEN_DESIGN"] is None and namespace["GRID_COMPLETE"] is False
    status = json.loads((work / "artifacts" / "session_status.json").read_text(encoding="utf-8"))
    assert status["design_frozen"] is False and len(status["pending_run_ids"]) == 900
    assert status["design_digest"] == namespace["design_digest"]()

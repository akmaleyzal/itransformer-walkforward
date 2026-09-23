"""The return leg, notebook to src/, has to be lossless.

``tools/notebook_to_src.py`` is the only tool that writes ``src/``, and it writes
it from a file edited by hand. Rebuilding a module from its cell must reproduce
the file on disk byte for byte; nothing here writes to ``src/``.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "btc_walkforward_3model.ipynb"
REVERSE = ROOT / "tools" / "notebook_to_src.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reverse():
    return _load(REVERSE, "notebook_to_src")


@pytest.fixture(scope="module")
def generator(reverse):
    return reverse._load_generator()


@pytest.fixture(scope="module")
def notebook() -> dict:
    assert NOTEBOOK.exists(), f"{NOTEBOOK} is missing; run tools/build_notebook.py --bootstrap"
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_every_module_is_recoverable_from_the_notebook(
    reverse, generator, notebook: dict
) -> None:
    """No module may be missing its cells: recovery from nothing is truncation."""
    grouped = reverse.cells_by_module(notebook)
    missing = [m for m in generator.MODULE_ORDER if m not in grouped]
    assert not missing, f"the notebook carries no cells for {missing}"


def test_cells_rejoin_to_the_flattened_body(reverse, generator, notebook: dict) -> None:
    """Cells grouped through metadata.itbtc rejoin to each flattened body."""
    grouped = reverse.cells_by_module(notebook)
    for module in generator.MODULE_ORDER:
        assert "".join(grouped[module]) == generator.flatten_module_body(module), (
            f"{module}'s cells no longer rejoin to its flattened body"
        )


def test_rebuild_reproduces_every_module_byte_for_byte(
    reverse, generator, notebook: dict
) -> None:
    """An unchanged cell rebuilds its module byte for byte, so code_sha256 does not move."""
    grouped = reverse.cells_by_module(notebook)
    for module in generator.MODULE_ORDER:
        body = "".join(grouped[module])
        rebuilt = reverse.rebuild_module(generator, module, body)
        on_disk = (generator.PACKAGE / module).read_text(encoding="utf-8")
        assert rebuilt == on_disk, (
            f"rebuilding {module} from its cells does not reproduce the file on "
            f"disk. The import block or the trailing guard is landing in the "
            f"wrong place, and writing it would corrupt the module."
        )


def test_rebuilt_modules_still_parse(reverse, generator, notebook: dict) -> None:
    """A reconstruction that does not compile is the worst possible outcome."""
    grouped = reverse.cells_by_module(notebook)
    for module in generator.MODULE_ORDER:
        rebuilt = reverse.rebuild_module(generator, module, "".join(grouped[module]))
        ast.parse(rebuilt, filename=module)


def test_an_import_added_in_a_cell_is_detected(reverse) -> None:
    """A module-level import in a cell is refused; a function-local one is not."""
    body = '"""Docstring."""\n\nimport itertools\n\n\ndef f():\n    return 1\n'
    assert reverse._module_level_imports_in(body, "probe.py") == ["import itertools"]

    clean = '"""Docstring."""\n\n\ndef f():\n    import itertools\n    return 1\n'
    assert reverse._module_level_imports_in(clean, "probe.py") == [], (
        "a function-local import is not a module-level one: report._pyplot defers matplotlib"
    )


def test_the_notebook_carries_the_sync_cell_and_it_is_inert(notebook: dict) -> None:
    """The sync cell is last and fully commented: Kaggle has neither tools/ nor src/."""
    last = notebook["cells"][-1]
    assert last["cell_type"] == "code"
    assert last["metadata"]["itbtc"]["step"] == "sync_back"
    source = "".join(last["source"])
    assert ast.parse(source).body == [], "the sync cell executes something"
    assert "notebook_to_src.py" in source, "the sync cell does not name its tool"


def test_an_edited_vendor_cell_is_refused(reverse, generator, notebook: dict, tmp_path) -> None:
    """The upstream copies are compared, never written back from the notebook."""
    assert reverse.vendor_mismatches(generator, notebook) == []
    edited = json.loads(json.dumps(notebook))
    cell = next(c for c in edited["cells"] if c.get("metadata", {}).get("itbtc", {}).get("role") == "vendor")
    cell["source"] = cell["source"] + ["\n# local edit"]
    assert reverse.vendor_mismatches(generator, edited) == [cell["metadata"]["itbtc"]["path"]]
    path = tmp_path / "edited.ipynb"
    path.write_text(json.dumps(edited), encoding="utf-8")
    assert reverse.sync(path, dry_run=True) == 1

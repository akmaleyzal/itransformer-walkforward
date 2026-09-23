"""Export notebooks/btc_walkforward_3model.ipynb to its tested src projection.

Invoke through the notebook's final, locally enabled sync cell. The flattening
helpers of tools/build_notebook.py are reused to restore each module's imports
and to verify an exact round trip. Vendor cells are only compared with the
pinned upstream copies, never written back.

New imports belong in the notebook Library cell and module metadata
itbtc.projection_imports. Remove obsolete package imports through
projection_remove_imports. A successful export refreshes the artifact map, the
pinned package digest and the ordered program digest, which clears stale outputs.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import importlib.util
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "tools" / "build_notebook.py"


def _load_generator():
    """Import tools/build_notebook.py, whose flattening rules this file must not duplicate."""
    spec = importlib.util.spec_from_file_location("build_notebook", GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_notebook"] = module
    spec.loader.exec_module(module)
    return module


def cells_by_module(notebook: dict) -> dict[str, list[str]]:
    """Every definition cell's source, grouped by module, in notebook order."""
    grouped: dict[str, list[str]] = {}
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        tag = cell.get("metadata", {}).get("itbtc", {})
        if tag.get("role") == "module" or (
            tag.get("role") is None and tag.get("module")
        ):
            grouped.setdefault(tag["module"], []).append("".join(cell["source"]))
    return grouped


def dropped_runs(generator, module: str) -> dict[int, list[str]]:
    """Lines the flattening removed, keyed by where they sit in the body.

    Removed lines are module-level imports, intra-package imports wherever they
    live (some are inside functions), the ``__main__`` guard and dropped
    functions. Each removed run is anchored to the number of body lines before
    it, found by alignment: flattening only deletes, so the body is a
    subsequence of the file, and what the alignment does not consume was removed.

    Returns:
        ``{body_lines_before: [removed lines]}``. The key ``len(body)`` holds
        anything trailing the last kept line.
    """
    lines = (generator.PACKAGE / module).read_text(encoding="utf-8").splitlines(
        keepends=True
    )
    body = generator.flatten_module_body(module).splitlines(keepends=True)

    runs: dict[int, list[str]] = {}
    cursor = 0
    for line in lines:
        if cursor < len(body) and line == body[cursor]:
            cursor += 1
        else:
            runs.setdefault(cursor, []).append(line)
    if cursor != len(body):
        raise SystemExit(
            f"{module}: its flattened body is not a subsequence of the file on "
            f"disk, so the flattening is doing more than deleting lines and "
            f"this reconstruction cannot be trusted. Refusing rather than "
            f"guessing."
        )
    return runs


def _map_anchor(old: list[str], new: list[str], anchor: int) -> int:
    """Where ``anchor`` in the old body lands in the new one.

    A removed line sits between two body lines; editing the body moves them. The
    anchor is carried across on the unchanged stretches, which is what keeps an
    import block above the code it serves after the code below it has been
    edited.
    """
    if anchor >= len(old):
        return len(new)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, old, new, autojunk=False
    ).get_opcodes():
        if i1 <= anchor < i2:
            return j1 + (anchor - i1) if tag == "equal" else j1
    return len(new)


def rebuild_module(generator, module: str, body: str) -> str:
    """The full ``src/`` text for ``module`` given its flattened body.

    Every removed run goes back at its anchor, so an unchanged body reproduces
    the file byte for byte (and ``code_sha256`` does not move) and an edited one
    keeps its imports where they were.
    """
    runs = dropped_runs(generator, module)
    old = generator.flatten_module_body(module).splitlines(keepends=True)
    new = body.splitlines(keepends=True)

    insertions: dict[int, list[str]] = {}
    for anchor in sorted(runs):
        target = _map_anchor(old, new, anchor)
        insertions.setdefault(target, []).extend(runs[anchor])

    out: list[str] = []
    for index in range(len(new) + 1):
        out.extend(insertions.get(index, []))
        if index < len(new):
            out.append(new[index])
    return "".join(out)


def _module_level_imports_in(body: str, module: str) -> list[str]:
    """Import statements a cell body carries, which no cell body may."""
    return [
        ast.unparse(node)
        for node in ast.parse(body, filename=module).body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]


def vendor_mismatches(generator, notebook: dict) -> list[str]:
    """Upstream paths whose vendor cell differs from the pinned copy, or is missing."""
    cells = {}
    for cell in notebook["cells"]:
        tag = cell.get("metadata", {}).get("itbtc", {})
        if tag.get("role") == "vendor":
            cells[tag.get("path")] = "".join(cell["source"])
    expected = {path: generator.vendor_cell(path) for path in generator.vendor_files()}
    return sorted(str(path) for path in expected.keys() | cells.keys()
                  if cells.get(path) != expected.get(path))


def sync(notebook_path: Path, dry_run: bool = False) -> int:
    generator = _load_generator()
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    vendor = vendor_mismatches(generator, notebook)
    if vendor:
        print(f"refusing to write: vendor cells {vendor} differ from the pinned upstream copies. "
              "The upstream code is never edited in the notebook; restore those cells.",
              file=sys.stderr)
        return 1
    # The artifact map is derived from the notebook's own step metadata.
    for cell in notebook["cells"]:
        if cell.get("metadata", {}).get("itbtc", {}).get("step") == "artifact_map":
            source = "".join(cell["source"])
            node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "_ARTIFACT_MAP" for t in n.targets))
            rows = []
            for index, item in enumerate(notebook["cells"]):
                tag = item.get("metadata", {}).get("itbtc", {})
                if tag.get("role") == "step":
                    rows.append((index, tag["step"], tag.get("writes", []), tag.get("reads", [])))
            lines = source.splitlines(keepends=True)
            lines[node.lineno-1:node.end_lineno] = ["_ARTIFACT_MAP = [\n" + "".join(f"    {row!r},\n" for row in rows) + "]\n"]
            cell["source"] = "".join(lines).splitlines(keepends=True)
    grouped = cells_by_module(notebook)

    missing = [m for m in generator.MODULE_ORDER if m not in grouped]
    if missing:
        print(
            f"refusing to write: the notebook has no cells for {missing}; a module "
            f"without cells would be truncated to nothing. Restore them from version "
            f"history first.",
            file=sys.stderr,
        )
        return 1

    changed: list[str] = []
    for module in generator.MODULE_ORDER:
        body = "".join(grouped[module])

        offending = _module_level_imports_in(body, module)
        if offending:
            print(
                f"refusing to write {module}: its cells carry module-level "
                f"imports {offending}. Module imports live in the Library cell; "
                f"declare a package-only import in metadata.itbtc.projection_imports "
                f"or import inside the function.",
                file=sys.stderr,
            )
            return 1

        target = generator.PACKAGE / module
        original = target.read_text(encoding="utf-8")
        rebuilt = rebuild_module(generator, module, body)
        # Notebook metadata declares imports needed only by the exported package.
        # In the notebook, sibling functions already share the kernel namespace.
        extra = []
        remove_imports = set()
        for cell in notebook["cells"]:
            tag = cell.get("metadata", {}).get("itbtc", {})
            if tag.get("module") == module:
                extra.extend(tag.get("projection_imports", []))
                remove_imports.update(tag.get("projection_remove_imports", []))
        if remove_imports:
            lines = rebuilt.splitlines(keepends=True)
            for node in reversed(ast.parse(rebuilt).body):
                if isinstance(node, (ast.Import, ast.ImportFrom)) and ast.unparse(node) in remove_imports:
                    del lines[node.lineno - 1:node.end_lineno]
            rebuilt = "".join(lines)
        tree = ast.parse(rebuilt)
        existing = {ast.unparse(n) for n in tree.body
                    if isinstance(n, (ast.Import, ast.ImportFrom))}
        additions = []
        for statement in extra:
            parsed = ast.parse(statement).body
            if len(parsed) != 1 or not isinstance(parsed[0], (ast.Import, ast.ImportFrom)):
                raise ValueError(f"{module}: invalid projection import {statement!r}")
            normal = ast.unparse(parsed[0])
            if normal not in existing:
                additions.append(normal + "\n")
                existing.add(normal)
        if additions:
            # External imports must join the external block. Appending one after
            # package imports would swallow a notebook separator during flattening.
            end = max(n.end_lineno for n in tree.body
                      if isinstance(n, (ast.Import, ast.ImportFrom))
                      and not generator._intra_package_import(n))
            lines = rebuilt.splitlines(keepends=True)
            lines[end:end] = additions
            rebuilt = "".join(lines)
        ast.parse(rebuilt, filename=module)
        if rebuilt == original:
            continue
        if dry_run:
            changed.append(module)
            continue

        target.write_text(rebuilt, encoding="utf-8", newline="\n")
        # Re-flatten what was written; anything short of byte equality means an
        # import landed in the wrong place, so the file is restored.
        if generator.flatten_module_body(module) != body:
            mismatch = list(difflib.unified_diff(
                body.splitlines(), generator.flatten_module_body(module).splitlines(),
                fromfile="notebook", tofile="projection", n=2))
            target.write_text(original, encoding="utf-8", newline="\n")
            print(
                f"refusing to write {module}: re-flattening what was written "
                f"did not reproduce the cells byte for byte, so the round trip "
                f"is not sound here. The file has been restored unchanged.",
                file=sys.stderr,
            )
            print("\n".join(mismatch[:40]), file=sys.stderr)
            return 1
        changed.append(module)

    if not dry_run:
        digest = generator.package_digest()
        for cell in notebook["cells"]:
            if cell.get("metadata", {}).get("itbtc", {}).get("step") == "code_digest":
                source = "".join(cell["source"])
                source = re.sub(r"CODE_SHA256_OVERRIDE = ['\"][0-9a-f]{64}['\"]",
                                f'CODE_SHA256_OVERRIDE = "{digest}"', source)
                cell["source"] = source.splitlines(keepends=True)
        program = json.dumps(["".join(c["source"]) for c in notebook["cells"]
                              if c.get("cell_type") == "code"], ensure_ascii=False)
        program_digest = hashlib.sha256(program.encode("utf-8")).hexdigest()
        if notebook["metadata"].get("itbtc_exported_program_sha256") != program_digest:
            for cell in notebook["cells"]:
                if cell.get("cell_type") == "code":
                    cell["outputs"] = []
                    cell["execution_count"] = None
        notebook["metadata"]["itbtc_exported_program_sha256"] = program_digest
        notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                                 encoding="utf-8", newline="\n")

    if not changed:
        print("src/itransformer_btc/ already matches the notebook; nothing written")
        return 0
    verb = "would rewrite" if dry_run else "rewrote"
    print(f"{verb} {len(changed)} module(s): {', '.join(changed)}")
    if not dry_run:
        print(
            "now run: python tools/build_notebook.py --check\n"
            "and commit src/ and the notebook together -- they are one change."
        )
    return 0


def check_projection(notebook_path: Path) -> int:
    """Validate the notebook's module cells, vendor cells and pinned digests against src/."""
    generator = _load_generator()
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    grouped = cells_by_module(notebook)
    errors = [m for m in generator.MODULE_ORDER
              if "".join(grouped.get(m, [])) != generator.flatten_module_body(m)]
    errors += [f"vendor {path}" for path in vendor_mismatches(generator, notebook)]
    code = ["".join(c["source"]) for c in notebook["cells"]
            if c.get("cell_type") == "code"]
    for source in code:
        if not source.startswith("%%writefile "):
            ast.parse(source, feature_version=(3, 11))
    pinned = [s for s in code if re.search(r"^CODE_SHA256_OVERRIDE =", s, re.M)]
    if len(pinned) != 1 or generator.package_digest() not in pinned[0]:
        errors.append("pinned package digest")
    program = hashlib.sha256(json.dumps(code, ensure_ascii=False).encode("utf-8")).hexdigest()
    if notebook.get("metadata", {}).get("itbtc_exported_program_sha256") != program:
        errors.append("ordered notebook program digest")
    if errors:
        print(f"Projection is stale: {errors}. Save the notebook, then use its final sync cell.",
              file=sys.stderr)
        return 1
    print("Notebook source, ordered program digest and exported src projection agree")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write src/itransformer_btc/ back from the notebook"
    )
    parser.add_argument(
        "notebook",
        nargs="?",
        default=str(ROOT / "notebooks" / "btc_walkforward_3model.ipynb"),
        help="the notebook to read; defaults to the committed one",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report which modules would change and write nothing",
    )
    args = parser.parse_args(argv)

    path = Path(args.notebook)
    if not path.exists():
        raise SystemExit(f"{path} does not exist")
    return sync(path, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())

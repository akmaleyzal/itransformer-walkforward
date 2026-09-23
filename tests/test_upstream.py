"""The copied iTransformer files: provenance, derivation and isolated loading."""

from __future__ import annotations

import ast
import difflib
import hashlib
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from itransformer_btc.train import tree_digest
from itransformer_btc.upstream import (
    UPSTREAM_COMMIT,
    UPSTREAM_FILES,
    UPSTREAM_RAW_SHA256,
    UPSTREAM_TRIMMED,
    canonical,
    derive,
    load_upstream,
    permalink,
    vendor_root,
    verify_upstream,
)

FIXTURES = Path(__file__).parent / "fixtures" / "thuml_iTransformer_c2426e68"


def _classes(text: str) -> dict[str, str]:
    return {node.name: ast.get_source_segment(text, node)
            for node in ast.parse(text).body if isinstance(node, ast.ClassDef)}


def test_fixtures_are_the_published_bytes() -> None:
    for path, expected in UPSTREAM_RAW_SHA256.items():
        assert hashlib.sha256((FIXTURES / path).read_bytes()).hexdigest() == expected, path


@pytest.mark.parametrize("path", sorted(UPSTREAM_RAW_SHA256))
def test_each_copy_is_derived_from_the_published_file(path: str) -> None:
    derived = derive(path, (FIXTURES / path).read_text(encoding="utf-8"))
    assert canonical((vendor_root() / path).read_text(encoding="utf-8")) == derived
    assert hashlib.sha256(derived.encode("utf-8")).hexdigest() == UPSTREAM_FILES[path]


def test_whole_files_differ_only_in_whitespace() -> None:
    """Only the final newline and whitespace-only lines, which %%writefile empties, may change."""
    for path in UPSTREAM_RAW_SHA256:
        if path in UPSTREAM_TRIMMED:
            continue
        text = (FIXTURES / path).read_text(encoding="utf-8")
        raw = text.rstrip("\n").split("\n")
        copy = derive(path, text).rstrip("\n").split("\n")
        assert len(copy) == len(raw), path
        for ours, theirs in zip(copy, raw):
            assert ours == theirs or (ours == "" and not theirs.strip()), (path, theirs)


def test_the_trimmed_file_only_deletes_lines() -> None:
    for path, (keep, dropped) in UPSTREAM_TRIMMED.items():
        raw = canonical((FIXTURES / path).read_text(encoding="utf-8"))
        copy = derive(path, raw)
        opcodes = difflib.SequenceMatcher(None, raw.splitlines(), copy.splitlines(),
                                          autojunk=False).get_opcodes()
        assert {tag for tag, *_ in opcodes} <= {"equal", "delete"}
        assert set(_classes(copy)) == set(keep)
        assert all(_classes(copy)[name] == _classes(raw)[name] for name in keep)
        tree = ast.parse(copy)
        imported = {(n.module or "").split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom)}
        imported |= {a.name.split(".")[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names}
        assert not imported & set(dropped)


def test_permalinks_pin_the_commit() -> None:
    assert permalink("model/Transformer.py") == (
        f"https://github.com/thuml/iTransformer/blob/{UPSTREAM_COMMIT}/model/Transformer.py")


def test_loading_leaves_sys_modules_as_it_found_it() -> None:
    sentinel = types.ModuleType("utils")
    sys.modules["utils"] = sentinel
    try:
        models = load_upstream(vendor_root())
        assert sys.modules["utils"] is sentinel
        assert not any(name.split(".")[0] in ("layers", "model") for name in sys.modules)
    finally:
        sys.modules.pop("utils", None)
    assert models["itr"] is not models["vtr"]
    assert models["itr"].__name__ == models["vtr"].__name__ == "Model"


def _configs() -> SimpleNamespace:
    return SimpleNamespace(
        seq_len=96, pred_len=24, label_len=48, d_model=16, d_ff=32, e_layers=1, d_layers=1,
        n_heads=2, dropout=0.0, activation="gelu", output_attention=False, use_norm=True,
        embed="timeF", freq="h", factor=1, class_strategy="projection",
        channel_independence=False, enc_in=3, dec_in=3, c_out=3,
    )


def test_both_upstream_models_run_forward() -> None:
    models = load_upstream(vendor_root())
    x = torch.randn(2, 96, 3)
    assert models["itr"](_configs())(x, None, None, None).shape == (2, 24, 3)
    decoder_input = torch.cat([x[:, -48:, :], torch.zeros(2, 24, 3)], dim=1)
    assert models["vtr"](_configs())(x, None, decoder_input, None).shape == (2, 24, 3)


def test_a_changed_copy_is_refused_and_moves_the_code_digest(tmp_path: Path) -> None:
    package = tmp_path / "itransformer_btc"
    shutil.copytree(vendor_root().parents[1], package, ignore=shutil.ignore_patterns("__pycache__"))
    before = tree_digest(package)
    target = package / "vendor" / "thuml_iTransformer" / "model" / "Transformer.py"
    target.write_text(target.read_text(encoding="utf-8") + "# edit\n", encoding="utf-8")
    assert tree_digest(package) != before
    with pytest.raises(ValueError, match="differ"):
        verify_upstream(package / "vendor" / "thuml_iTransformer")

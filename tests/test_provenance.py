"""The provenance table and the module docstrings are two copies; bind them."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from itransformer_btc.config import SOURCE_PROVENANCE, Upstream

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "itransformer_btc"

#: ``copied``: the authors' code, unchanged. ``library``: imported and called.
#: ``own``: written here from the published description.
STATUSES = frozenset({"copied", "library", "own"})

#: Repositories checked at the stated revision.
VERIFIED_REPOS = {"https://github.com/thuml/iTransformer"}


def _module_source(name: str) -> str:
    return (PACKAGE / name).read_text(encoding="utf-8")


def _module_docstring(name: str) -> str:
    return ast.get_docstring(ast.parse(_module_source(name))) or ""


def test_every_row_names_a_module_that_exists() -> None:
    for row in SOURCE_PROVENANCE:
        assert (PACKAGE / row.module).is_file(), f"{row.component!r}: no {row.module}"


def test_status_is_one_of_the_declared_values() -> None:
    for row in SOURCE_PROVENANCE:
        assert row.status in STATUSES, f"{row.component!r} has status {row.status!r}"


def test_repo_url_appears_in_the_module_it_describes() -> None:
    for row in SOURCE_PROVENANCE:
        if row.repo:
            assert row.repo in _module_source(row.module), f"{row.repo} missing from {row.module}"


def test_a_row_without_a_repo_is_written_here() -> None:
    for row in SOURCE_PROVENANCE:
        if not row.repo:
            assert row.status == "own", f"{row.component!r} names no repository"


def test_third_party_rows_carry_a_licence_and_an_access_date() -> None:
    for row in SOURCE_PROVENANCE:
        if row.repo:
            assert row.licence, f"{row.component!r} names {row.repo} with no licence"
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", row.accessed), row.component


def test_verified_flag_matches_what_was_checked() -> None:
    assert {row.repo for row in SOURCE_PROVENANCE if row.verified} == VERIFIED_REPOS


def test_every_row_states_how_the_code_is_used() -> None:
    for row in SOURCE_PROVENANCE:
        assert row.adapted.strip(), f"{row.component!r} says nothing about its use"


@pytest.mark.parametrize("module", sorted({row.module for row in SOURCE_PROVENANCE}))
def test_module_docstring_carries_an_upstream_section(module: str) -> None:
    assert "Upstream" in _module_docstring(module), f"{module} has no Upstream section"


def test_every_model_in_the_grid_has_a_row() -> None:
    covered = " ".join(row.component for row in SOURCE_PROVENANCE)
    for name in ("ITransformerForecaster", "VanillaForecaster", "RidgeForecaster", "Naive-RW"):
        assert name in covered, f"{name} has no provenance row"


def test_rows_are_unique_per_component() -> None:
    components = [row.component for row in SOURCE_PROVENANCE]
    assert len(components) == len(set(components))


def test_upstream_is_immutable() -> None:
    row = SOURCE_PROVENANCE[0]
    assert isinstance(row, Upstream)
    with pytest.raises((AttributeError, TypeError)):
        row.repo = "https://example.invalid"  # type: ignore[misc]

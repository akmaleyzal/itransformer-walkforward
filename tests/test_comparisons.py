"""Table 6: the pair families, FWER control, the confidence set, and the panel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from itransformer_btc.comparisons import (
    FAMILY_ORDER,
    NAIVE,
    build_panel,
    cluster_bootstrap_t,
    differential,
    label,
    mcs_table,
    model_confidence_set,
    pair_family,
    pair_matrix,
    romano_wolf,
)
from itransformer_btc.runner import RunCell

from conftest import write_grid

ORIGINS_USED = (1, 2, 3)


@pytest.fixture(scope="module")
def small_grid(tmp_path_factory) -> Path:
    cells = [RunCell(arm, origin, k, 24, seed) for arm in ("main", "ridge", "vanilla")
             for origin in ORIGINS_USED for k in (1, 8) for seed in (42, 43)]
    return write_grid(tmp_path_factory.mktemp("small") / "artifacts", cells)


def test_families_follow_the_claims() -> None:
    assert pair_family(("itr", 8), NAIVE) == "vs-naive"
    assert pair_family(("itr", 1), ("itr", 8)) == "ladder"
    assert pair_family(("itr", 8), ("rdg", 8)) == "cross-model"
    assert pair_family(("vtr", 4), ("itr", 4)) == "cross-model"
    assert pair_family(("itr", 8), ("rdg", 4)) == "other"
    assert pair_family(("rdg", 1), ("rdg", 8)) == "other"
    assert FAMILY_ORDER == ("vs-naive", "ladder", "cross-model", "other")


def test_label_names_the_sentinel_readably() -> None:
    assert label(NAIVE) == "Naive-RW"
    assert label(("itr", 8)) == "itr-K8"


def test_romano_wolf_is_monotone_and_respects_its_floor() -> None:
    rng = np.random.default_rng(42)
    per_origin = rng.standard_normal((15, 6)) * 0.01
    per_origin[:, 0] += 0.05
    p = romano_wolf(per_origin, B=999, seed=7)
    assert p.shape == (6,) and p[0] == p.min()
    assert (p >= 1 / (1 + 999)).all() and (p <= 1.0).all()


def test_romano_wolf_controls_fwer_under_a_complete_null() -> None:
    rng = np.random.default_rng(1)
    p = romano_wolf(rng.standard_normal((15, 20)) * 0.01, B=999, seed=3)
    assert (p < 0.05).sum() <= 1


def test_romano_wolf_is_never_smaller_than_the_unadjusted_p() -> None:
    rng = np.random.default_rng(11)
    per_origin = rng.standard_normal((15, 8)) * 0.01
    per_origin[:, 2] += 0.08
    t_obs, t_boot = cluster_bootstrap_t(per_origin, B=999, seed=5)
    raw = np.array([(1 + int((np.abs(t_boot[:, j]) >= abs(t_obs[j])).sum())) / 1000
                    for j in range(per_origin.shape[1])])
    assert (romano_wolf(per_origin, B=999, seed=5) >= raw - 1e-12).all()


def test_model_confidence_set_drops_a_clearly_worse_model() -> None:
    rng = np.random.default_rng(2)
    losses = rng.standard_normal((15, 4)) * 0.001
    losses[:, 3] += 0.5
    keep = model_confidence_set(losses, alpha=0.10, B=999, seed=5)
    assert 3 not in keep and len(keep) >= 1


def test_model_confidence_set_keeps_everything_when_nothing_separates() -> None:
    rng = np.random.default_rng(4)
    assert len(model_confidence_set(rng.standard_normal((15, 5)) * 1e-9, alpha=0.10, B=999)) == 5


def test_build_panel_aligns_every_model_on_identical_targets(small_grid) -> None:
    keys = [("itr", 8), ("rdg", 8), ("vtr", 8), NAIVE]
    panel = build_panel(keys, [small_grid], origin_indices=ORIGINS_USED)
    for index in ORIGINS_USED:
        n = len(panel.y_true[index])
        assert n % 24 == 0 and all(len(panel.y_pred[key, index]) == n for key in keys)
        assert np.isclose(panel.y_pred[NAIVE, index].std(), 0.0), "Naive-RW is constant in z"


def test_a_missing_cell_fails_loudly(small_grid) -> None:
    with pytest.raises(FileNotFoundError, match="no run at origin"):
        build_panel([("itr", 8), ("itr", 12)], [small_grid], origin_indices=(1,))


def test_differential_changes_sign_when_the_pair_is_reversed(small_grid) -> None:
    panel = build_panel([("itr", 8), NAIVE], [small_grid], origin_indices=(1,))
    a = differential(panel, ("itr", 8), NAIVE, 1)
    np.testing.assert_array_equal(a, -differential(panel, NAIVE, ("itr", 8), 1))
    assert len(a) == len(panel.y_true[1]) // 24


def test_pair_matrix_labels_families_and_states_t_and_h(small_grid) -> None:
    keys = [("itr", 1), ("itr", 8), ("rdg", 8), ("vtr", 8), NAIVE]
    table = pair_matrix(build_panel(keys, [small_grid], origin_indices=ORIGINS_USED), B=199, seed=13)
    assert table.height == 10
    family = dict(zip(zip(table["left"], table["right"]), table["family"]))
    assert family["itr-K1", "itr-K8"] == "ladder"
    assert family["itr-K8", "rdg-K8"] == family["itr-K8", "vtr-K8"] == "cross-model"
    assert family["rdg-K8", "Naive-RW"] == "vs-naive"
    assert family["itr-K1", "rdg-K8"] == "other"
    assert (table["h"] == 24).all() and (table["G"] == 3).all()
    assert (table["p_romano_wolf"] >= 1 / 200).all()
    assert (table["p_romano_wolf_family"] <= table["p_romano_wolf"] + 1e-12).all()


def test_mcs_table_ranks_by_mean_loss(small_grid) -> None:
    keys = [("itr", 8), ("rdg", 8), ("vtr", 8), NAIVE]
    table = mcs_table(build_panel(keys, [small_grid], origin_indices=ORIGINS_USED), B=199, seed=17)
    assert table["rank"].to_list() == [1, 2, 3, 4]
    assert table["model"][0] == "rdg-K8", "Ridge is the best model by construction"
    assert (np.diff(table["mean_loss"].to_numpy()) > 0).all()
    assert table.filter(table["model"] == "Naive-RW")["mean_loss"].item() == pytest.approx(1.0)
    assert table["in_mcs_90"][0]

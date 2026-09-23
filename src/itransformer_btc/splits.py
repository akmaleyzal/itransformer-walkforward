"""Per-origin splits, the scaler, and the tensors the training loop slices.

Training and validation windows lie wholly inside their spans, so no target
crosses into the next split: the purge holds at both boundaries. A test window
is indexed by its forecast origin, the first target hour, so every hour of a test
block is an admissible origin and its lookback may reach back into validation,
which is information a forecaster has at that moment.

The scaler is fitted on the rows of the 21-month training sub-block and on
nothing else, at every origin.

Upstream:
    Rolling-origin evaluation after L. J. Tashman, Int. J. Forecast. 16(4), 2000,
    and C. Bergmeir and J. M. Benitez, Information Sciences 191, 2012. Purging
    after M. Lopez de Prado, Advances in Financial Machine Learning, Wiley, 2018,
    ch. 7, applied at both boundaries; no embargo, because every feature is
    per-bar. The scaler is written here: two lines of arithmetic kept inside the
    per-origin tensor build.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import polars as pl

from itransformer_btc.config import (
    PRED_LEN,
    SEQ_LEN,
    WINDOW_SPAN,
    OriginLike,
)
from itransformer_btc.features import TARGET_INDEX, ladder_columns
from itransformer_btc.segments import HOUR_MS

Semantics = Literal["contained", "origin"]


def window_starts(
    ts: np.ndarray, start: datetime, end: datetime, semantics: Semantics,
    span: int = WINDOW_SPAN, *, seq_len: int = SEQ_LEN,
) -> np.ndarray:
    """Input-start indices of the contiguous windows that belong to ``[start, end)``.

    ``"contained"`` keeps windows lying wholly inside the span (training and
    validation). ``"origin"`` keeps windows whose first target hour falls inside
    it (test). A forecast at t reads ``[t-L, t)`` and predicts ``[t, t+H)``.
    """
    if semantics not in ("contained", "origin") or span < 2:
        raise ValueError("invalid window semantics or span")
    if semantics == "origin" and not 0 < seq_len < span:
        raise ValueError("origin semantics require 0 < seq_len < span")
    ts = np.asarray(ts)
    if ts.ndim != 1 or np.any(np.diff(ts) <= 0) or np.any(ts % HOUR_MS != 0):
        raise ValueError("timestamps must be unique, increasing UTC hour boundaries")
    lo, hi = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    if hi <= lo:
        raise ValueError("window end must follow start")
    first = np.arange(max(0, len(ts) - span + 1), dtype=np.int64)
    contiguous = (ts[first + span - 1] - ts[first]) == (span - 1) * HOUR_MS
    if semantics == "contained":
        inside = (ts[first] >= lo) & (ts[first + span - 1] < hi)
    else:
        issued = ts[first + seq_len]
        inside = (issued >= lo) & (issued < hi)
    return first[contiguous & inside]


@dataclass(frozen=True, slots=True)
class Scaler:
    """Per-channel standardiser fitted on the training sub-block only.

    Under ``use_norm`` the iTransformer cancels any per-channel affine scaling, so
    for it this sets only the reporting scale; Ridge and the vanilla Transformer
    learn in this scale directly.
    """

    mean: np.ndarray
    std: np.ndarray
    columns: tuple[str, ...]

    @classmethod
    def fit(cls, values: np.ndarray, columns: list[str]) -> "Scaler":
        std = values.std(axis=0, ddof=0)
        if not np.all(np.isfinite(std)) or np.any(std <= 0):
            raise ValueError(
                f"degenerate channel std in the training sub-block: "
                f"{dict(zip(columns, std))}"
            )
        return cls(values.mean(axis=0), std, tuple(columns))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    @property
    def target_mu_over_sigma(self) -> float:
        """``mu_g / sigma_g`` on the target channel, the Naive-RW offset.

        A random walk predicts ``r = 0``. In scaler space that is
        ``-mu_g / sigma_g``, not 0, which would mean predicting the training drift.
        """
        return float(self.mean[TARGET_INDEX] / self.std[TARGET_INDEX])


@dataclass(frozen=True, slots=True)
class SplitTensors:
    """One split's standardised windows."""

    x: np.ndarray      # (n, L, K) float32 inputs
    y: np.ndarray      # (n, H)    float32 target channel
    ts: np.ndarray     # (n,) int64 forecast origin = first target bar open (UTC)

    def __len__(self) -> int:
        return len(self.ts)


@dataclass(frozen=True, slots=True)
class OriginTensors:
    """Everything one run consumes for an (origin, K) cell, already standardised."""

    origin: OriginLike
    k: int
    scaler: Scaler
    train: SplitTensors
    val: SplitTensors
    test_blocks: tuple[SplitTensors, ...]
    #: One-indexed label of each entry of ``test_blocks``.
    block_labels: tuple[int, ...]
    training_selection: dict | None = None

    @property
    def naive_rw_z(self) -> float:
        """The Naive-RW forecast in scaler space."""
        return -self.scaler.target_mu_over_sigma


def _gather(
    values: np.ndarray, starts: np.ndarray, ts: np.ndarray, seq_len: int, pred_len: int
) -> SplitTensors:
    """Slice windows out of a standardised array by index arithmetic, with no data loader."""
    if len(starts) == 0:
        return SplitTensors(
            x=np.empty((0, seq_len, values.shape[1]), np.float32),
            y=np.empty((0, pred_len), np.float32),
            ts=np.empty(0, np.int64),
        )
    rows = starts[:, None] + np.arange(seq_len)[None, :]
    tgt = starts[:, None] + seq_len + np.arange(pred_len)[None, :]
    return SplitTensors(
        x=values[rows].astype(np.float32, copy=False),
        y=np.ascontiguousarray(values[tgt, TARGET_INDEX].astype(np.float32, copy=False)),
        ts=ts[starts + seq_len],
    )


def build_origin_tensors(
    features: pl.DataFrame,
    origin: OriginLike,
    k: int,
    seq_len: int = SEQ_LEN,
    pred_len: int = PRED_LEN,
    train_window_limit: int | None = None,
    selection_seed: int = 1729,
) -> OriginTensors:
    """Build every split for one (origin, K) cell.

    The scaler is fitted on the purged training rows before any window is cut or
    subsampled, then applied to every split. With ``train_window_limit`` the
    training windows are drawn without replacement with ``selection_seed``, and
    the draw is recorded in ``training_selection``.

    Raises:
        ValueError: If the training split is empty, if a training target reaches
            validation (the purge failed), or if fewer windows exist than the
            limit asks for.
    """
    columns = ladder_columns(k)
    ts = features.get_column("ts_ms").to_numpy()
    values = features.select(columns).to_numpy()
    span = seq_len + pred_len

    train_idx = window_starts(ts, origin.train_start, origin.train_sub_end,
                              "contained", span)
    val_idx = window_starts(ts, origin.val_start, origin.val_end, "contained", span)
    if len(train_idx) == 0:
        raise ValueError(f"origin {origin.label}: empty training split")

    last_train_target = ts[train_idx[-1] + span - 1]
    if last_train_target >= int(origin.val_start.timestamp() * 1000):
        raise ValueError(
            f"origin {origin.label}: a training target reaches into validation "
            f"({last_train_target}); the purge did not hold"
        )

    scaler = Scaler.fit(values[train_idx[0] : train_idx[-1] + span], columns)
    scaled = scaler.transform(values)
    n_available = len(train_idx)
    if train_window_limit is not None:
        if train_window_limit < 1 or n_available < train_window_limit:
            raise ValueError(f"{origin.label}: {n_available} training windows < required {train_window_limit}")
        selected = np.random.default_rng(selection_seed).choice(n_available, train_window_limit, replace=False)
        train_idx = train_idx[np.sort(selected)]
    selection = {"available": n_available, "selected": len(train_idx),
                 "limit": train_window_limit, "seed": selection_seed,
                 "forecast_times_sha256": hashlib.sha256(ts[train_idx + seq_len].tobytes()).hexdigest(),
                 "scaler_fit": "all purged training rows before subsampling"}

    blocks = origin.blocks()
    return OriginTensors(
        origin=origin,
        k=k,
        scaler=scaler,
        train=_gather(scaled, train_idx, ts, seq_len, pred_len),
        val=_gather(scaled, val_idx, ts, seq_len, pred_len),
        test_blocks=tuple(
            _gather(scaled, window_starts(ts, lo, hi, "origin", span, seq_len=seq_len),
                    ts, seq_len, pred_len)
            for _, lo, hi in blocks
        ),
        block_labels=tuple(label for label, _, _ in blocks),
        training_selection=selection,
    )

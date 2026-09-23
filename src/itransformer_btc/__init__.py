"""Tested projection of notebooks/btc_walkforward_3model.ipynb.

Edit the notebook and export through its final cell; this package is what the
tests import and what ``code_sha256`` hashes.
"""

from __future__ import annotations

from itransformer_btc.config import (
    ORIGINS,
    PRED_LEN,
    SEQ_LEN,
    STARTS_LOST_PER_BREAK,
    WINDOW_SPAN,
    Origin,
    origin_grid,
)

__all__ = [
    "ORIGINS",
    "Origin",
    "PRED_LEN",
    "SEQ_LEN",
    "STARTS_LOST_PER_BREAK",
    "WINDOW_SPAN",
    "origin_grid",
]

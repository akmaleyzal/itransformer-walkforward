"""Ridge regression, the linear comparator on the same information.

Ridge maps the flattened ``K x 96`` lookback to the 24-step target with an L2
penalty, and C1 compares the iTransformer against it at every rung. Alpha is the
only hyperparameter selected anywhere in the study, on validation MSE. The fit
is closed-form and draws no random numbers, so the five seeds of a cell give
identical predictions.

Upstream:
    ``sklearn.linear_model.Ridge`` (https://github.com/scikit-learn/scikit-learn,
    BSD-3-Clause); F. Pedregosa et al., "Scikit-learn: Machine learning in
    Python," JMLR 12, 2011; A. E. Hoerl and R. W. Kennard, "Ridge regression:
    Biased estimation for nonorthogonal problems," Technometrics 12(1), 1970.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from itransformer_btc.config import PRED_LEN, SEQ_LEN
from itransformer_btc.metrics import assert_same_windows, load_predictions
from itransformer_btc.splits import OriginTensors
from itransformer_btc.train import SEED_LOCK, RunSpec, TrainOutcome, pick_device, set_seed

#: Alpha candidates, chosen on validation. The penalty is unnormalised, so the
#: scale that matters is ``diag(X'X)``, of the order of the training sample size.
RIDGE_ALPHAS: tuple[float, ...] = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)


@dataclass(frozen=True, slots=True)
class RidgeConfig:
    """L2-regularised linear map from the flattened window to the H-step target.

    ``k`` is a field because the weight matrix is ``(L*K, H)``. ``alpha`` is set by
    :meth:`fit`, and the fitted value is what enters ``meta['config']``.
    """

    seq_len: int = SEQ_LEN
    pred_len: int = PRED_LEN
    k: int = 8
    alphas: tuple[float, ...] = RIDGE_ALPHAS
    solver: str = "cholesky"
    alpha: float | None = None

    def build(self) -> "RidgeForecaster":
        return RidgeForecaster(self)

    def fit(
        self,
        tensors: OriginTensors,
        spec: RunSpec,
        *,
        device: torch.device | None = None,
    ) -> tuple["RidgeForecaster", "RidgeConfig", TrainOutcome]:
        """Fit one ``Ridge`` per alpha on the training windows, keep the best on validation.

        Inputs are cast to float64 before fitting. The intercept is fitted and not
        penalised, so the forecast is not shrunk toward the training drift.
        """
        from sklearn.linear_model import Ridge

        device = device or pick_device()
        started = time.perf_counter()
        with SEED_LOCK:
            set_seed(spec.seed, device)
            model = self.build().to(device)
        x_tr = tensors.train.x.reshape(len(tensors.train), -1).astype(np.float64)
        y_tr = tensors.train.y.astype(np.float64)
        x_va = tensors.val.x.reshape(len(tensors.val), -1).astype(np.float64)
        y_va = tensors.val.y.astype(np.float64)

        best = None
        for alpha in self.alphas:
            fitted = Ridge(alpha=alpha, fit_intercept=True, solver=self.solver).fit(x_tr, y_tr)
            val_mse = float(np.mean((fitted.predict(x_va) - y_va) ** 2))
            if best is None or val_mse < best[0]:
                best = (val_mse, float(alpha), fitted)
        if best is None:
            raise ValueError("no ridge alpha to select; `alphas` is empty")

        val_mse, alpha, fitted = best
        if len(self.alphas) > 1 and alpha in (self.alphas[0], self.alphas[-1]):
            warnings.warn(
                f"{spec.run_id}: ridge alpha {alpha:g} sits at the edge of "
                f"{self.alphas}; the grid may not bracket the optimum",
                stacklevel=2,
            )
        with torch.no_grad():
            model.weight.copy_(torch.from_numpy(fitted.coef_.T.astype(np.float32)))
            model.bias.copy_(torch.from_numpy(np.asarray(fitted.intercept_, dtype=np.float32)))
        train_mse = float(np.mean((fitted.predict(x_tr) - y_tr) ** 2))
        return (
            model,
            replace(self, alpha=alpha),
            TrainOutcome(
                run_id=spec.run_id,
                epochs_run=0,
                best_val_mse=val_mse,
                train_loss=train_mse,
                wall_time_s=time.perf_counter() - started,
                n_parameters=model.n_parameters(),
                device=str(device),
            ),
        )


class RidgeForecaster(nn.Module):
    """``y_hat = vec(x) @ W + b`` with the coefficients fitted by scikit-learn.

    ``W`` and ``b`` are buffers, not parameters, so no optimiser ever sees them.
    """

    def __init__(self, cfg: RidgeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer(
            "weight",
            torch.zeros(cfg.seq_len * cfg.k, cfg.pred_len, dtype=torch.float32),
        )
        self.register_buffer("bias", torch.zeros(cfg.pred_len, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        """``(B, L, K) -> (B, H)``."""
        return x.reshape(len(x), -1) @ self.weight + self.bias

    def forecast_target(self, x: Tensor) -> Tensor:
        return self(x)

    def n_parameters(self) -> int:
        return self.weight.numel() + self.bias.numel()


def assert_baseline_alignment(
    baseline_run_id: str, reference_run_id: str, roots: list[Path]
) -> None:
    """Refuse a comparator scored on windows other than its reference's.

    Test-window survival depends on future gaps, which cluster in stress, so a
    ratio over two different window sets would be biased, not noisy.

    Raises:
        ValueError: If the evaluated ``(block, timestamp)`` sets differ.
        FileNotFoundError: If either run is absent from ``roots``.
    """
    assert_same_windows(
        load_predictions(baseline_run_id, roots),
        load_predictions(reference_run_id, roots),
        f"{baseline_run_id} vs {reference_run_id}",
    )

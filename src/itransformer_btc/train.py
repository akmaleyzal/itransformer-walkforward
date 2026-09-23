"""Training loop, run identity, and the files every run leaves behind.

Each split is loaded onto the device once and batched by index slicing, with no
``Dataset`` or ``DataLoader``. Shuffling permutes an index tensor on the device.
Every run persists its raw predictions, its best weights and a metadata record
naming the code, input, configuration and environment that produced them.

Upstream:
    ``torch.optim.Adam`` (D. P. Kingma and J. Ba, ICLR 2015) at ``lr = 1e-4`` and
    ``torch.optim.lr_scheduler.StepLR`` halving every four epochs, from PyTorch
    (https://docs.pytorch.org/docs/stable/optim.html, BSD-3-Clause). Halving every
    epoch would make the 30-epoch cap unreachable. The loop around them is
    written here.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl
import torch
from torch import Tensor, nn

from itransformer_btc.splits import OriginTensors, SplitTensors
from itransformer_btc.upstream import UPSTREAM_COMMIT, UPSTREAM_REPO

ARTIFACTS: Path = Path("artifacts")

DEFAULT_PARQUET: Path = Path("data/raw/BTCUSDT_1h.parquet")

#: Environment variable naming the input parquet a process consumed.
INPUT_PARQUET_ENV: str = "ITBTC_PARQUET"

#: Code digest pinned by a launcher that has no package files to hash (the notebook).
CODE_SHA256_OVERRIDE: str | None = None

#: Serialises seeding and model construction, the only steps that share the CPU
#: generator across the per-GPU workers.
SEED_LOCK = threading.Lock()


def set_seed(seed: int, device: torch.device | None = None) -> None:
    """Seed Python, NumPy, the CPU generator and one CUDA device; force deterministic cuDNN.

    Seeding only the given device keeps two concurrent workers from resetting
    each other's CUDA stream.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.default_generator.manual_seed(seed)
    if torch.cuda.is_available():
        if device is not None and torch.device(device).type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        else:
            torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device() -> torch.device:
    """Prefer CUDA; fall back to CPU."""
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def supports_native_bf16(device: torch.device) -> bool:
    """True only on sm_80+; ``is_bf16_supported()`` also reports emulated support."""
    if device.type != "cuda":
        return False
    return torch.cuda.get_device_capability(device.index or 0)[0] >= 8


@dataclass(frozen=True, slots=True)
class RunSpec:
    """Run identity: ``{model}_o{origin:02d}_K{K:02d}_H{H:03d}_s{seed}``."""

    model: str
    origin_index: int
    k: int
    pred_len: int
    seed: int

    @property
    def run_id(self) -> str:
        return (
            f"{self.model}_o{self.origin_index:02d}_K{self.k:02d}"
            f"_H{self.pred_len:03d}_s{self.seed}"
        )


@dataclass(frozen=True, slots=True)
class TrainOutcome:
    """What one completed run produced, beyond its files."""

    run_id: str
    epochs_run: int
    best_val_mse: float
    train_loss: float
    wall_time_s: float
    n_parameters: int
    device: str


@dataclass(frozen=True, slots=True)
class TrainSchedule:
    """The optimisation budget shared by the two neural models."""

    max_epochs: int = 30
    patience: int = 5
    lr: float = 1e-4
    lr_halve_every: int = 4


class Architecture(Protocol):
    """What the trainer, the runner and the artifact writer need of a config.

    Behaviour lives in methods, not fields, because ``asdict(cfg)`` is written to
    every ``meta/*.json``.
    """

    seq_len: int
    pred_len: int

    def build(self) -> nn.Module:
        """A fresh, untrained module."""

    def fit(
        self,
        tensors: OriginTensors,
        spec: RunSpec,
        *,
        device: torch.device | None = None,
    ) -> tuple[nn.Module, "Architecture", TrainOutcome]:
        """Fit one cell; return the model, the resolved config and the outcome."""


class Forecaster(Protocol):
    """What the artifact writer needs of a fitted model."""

    cfg: Architecture

    def eval(self) -> "Forecaster":
        """Inference mode, dropout off."""

    def forecast_target(self, x: Tensor) -> Tensor:
        """``(B, L, K) -> (B, H)`` on the target channel."""

    def n_parameters(self) -> int:
        """Trainable parameters, or fitted coefficients for a closed-form model."""


def _to_device(split: SplitTensors, device: torch.device) -> tuple[Tensor, Tensor]:
    """Move one split's inputs and target channel to the device."""
    return (
        torch.from_numpy(split.x).to(device, non_blocking=True),
        torch.from_numpy(split.y).to(device, non_blocking=True),
    )


@torch.no_grad()
def _mean_loss(model: nn.Module, x: Tensor, y: Tensor, batch: int = 512) -> float:
    """Mean squared error over a split, batched to bound memory, synchronised once."""
    if len(x) == 0:
        return float("nan")
    model.eval()
    total = torch.zeros((), dtype=torch.float64, device=x.device)
    for i in range(0, len(x), batch):
        total += nn.functional.mse_loss(
            model.forecast_target(x[i : i + batch]), y[i : i + batch], reduction="sum"
        ).double()
    return float(total.item()) / (len(x) * int(np.prod(y.shape[1:])))


@torch.no_grad()
def predict(model: Forecaster, x: Tensor, batch: int = 512) -> np.ndarray:
    """The target channel's H-step forecasts for every window, batched."""
    model.eval()
    if len(x) == 0:
        return np.empty((0, model.cfg.pred_len), np.float32)
    return np.concatenate(
        [
            model.forecast_target(x[i : i + batch]).cpu().numpy()
            for i in range(0, len(x), batch)
        ]
    )


_TRAINING_CONTEXT = threading.local()


class SessionBudgetExhausted(RuntimeError):
    """A recoverable pause at a checkpoint boundary, not a failed run."""


class TrainingSession:
    """Per-worker checkpoint roots and a monotonic deadline."""

    def __init__(self, out_root: Path, roots: list[Path], deadline: float):
        self.state = (Path(out_root), [Path(r) for r in roots], deadline)

    def __enter__(self):
        self.previous = getattr(_TRAINING_CONTEXT, "state", None)
        _TRAINING_CONTEXT.state = self.state
        return self

    def __exit__(self, *exc):
        _TRAINING_CONTEXT.state = self.previous


def train_one(
    tensors: OriginTensors, spec: RunSpec, cfg: Architecture, *,
    device: torch.device | None = None, max_epochs: int | None = None,
    patience: int | None = None, lr: float | None = None,
    lr_halve_every: int | None = None, batch_size: int = 32,
) -> tuple[nn.Module, TrainOutcome]:
    """Train on one device with early stopping on validation MSE, checkpointing every epoch.

    A checkpoint holds the optimiser, scheduler, best weights, epoch, patience
    counter and device RNG, and is reused only when its identity (spec, config,
    code, input, schedule and training sample) matches. An epoch interrupted by
    the session deadline is replayed from its last boundary. Loss is accumulated
    on the device and checked once per epoch, so no step waits on the GPU.
    """
    device = device or pick_device()
    active = getattr(_TRAINING_CONTEXT, "state", None)
    if active is not None and time.perf_counter() >= active[2]:
        raise SessionBudgetExhausted(f"{spec.run_id}: session budget exhausted before fitting")
    protocol = cfg.schedule() if hasattr(cfg, "schedule") else TrainSchedule()
    max_epochs = protocol.max_epochs if max_epochs is None else max_epochs
    patience = protocol.patience if patience is None else patience
    lr = protocol.lr if lr is None else lr
    lr_halve_every = protocol.lr_halve_every if lr_halve_every is None else lr_halve_every
    if min(max_epochs, patience, lr_halve_every, batch_size) < 1 or lr <= 0:
        raise ValueError("positive training schedule and batch size required")
    if not len(tensors.train) or not len(tensors.val):
        raise ValueError("training and validation must be nonempty")
    with SEED_LOCK:
        set_seed(spec.seed, device)
        model = cfg.build().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.StepLR(optimiser, step_size=lr_halve_every, gamma=.5)
    x_tr, y_tr = _to_device(tensors.train, device)
    x_va, y_va = _to_device(tensors.val, device)
    best_val, best_state, stale = float("inf"), None, 0
    epochs_run, train_loss, prior_seconds = 0, float("nan"), 0.
    started = time.perf_counter()
    context = getattr(_TRAINING_CONTEXT, "state", None)
    checkpoint = None
    identity = None
    deadline = float("inf")
    if context is not None:
        out_root, roots, deadline = context
        checkpoint = out_root / "checkpoints" / f"{spec.run_id}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        identity = json.loads(json.dumps({
            "spec": asdict(spec), "config": asdict(cfg), "code": code_sha256(),
            "input": _input_sha256()[0], "torch": str(torch.__version__),
            "device_type": device.type, "batch_size": batch_size,
            "schedule": [max_epochs, patience, lr, lr_halve_every],
            "train_times": hashlib.sha256(tensors.train.ts.tobytes()).hexdigest(),
        }))
        for root in dict.fromkeys([out_root, *roots]):
            candidate = root / "checkpoints" / f"{spec.run_id}.pt"
            if not candidate.exists():
                continue
            try:
                saved = torch.load(candidate, map_location="cpu", weights_only=True)
            except (OSError, RuntimeError, EOFError) as exc:
                raise ValueError(f"{candidate}: unreadable training checkpoint") from exc
            if saved.get("identity") != identity:
                continue
            model.load_state_dict(saved["model"])
            optimiser.load_state_dict(saved["optimizer"])
            schedule.load_state_dict(saved["scheduler"])
            best_state, best_val = saved["best_state"], saved["best_val"]
            epochs_run, stale = saved["epoch"], saved["stale"]
            train_loss, prior_seconds = saved["train_loss"], saved["wall_time_s"]
            if device.type == "cuda":
                torch.cuda.set_rng_state(saved["rng"], device=device)
            else:
                torch.set_rng_state(saved["rng"])
            break

    def save_boundary():
        if checkpoint is None:
            return
        staging = checkpoint.with_suffix(".pt.tmp")
        torch.save({
            "identity": identity, "model": model.state_dict(),
            "optimizer": optimiser.state_dict(), "scheduler": schedule.state_dict(),
            "best_state": best_state, "best_val": best_val, "epoch": epochs_run,
            "stale": stale, "train_loss": train_loss,
            "wall_time_s": prior_seconds + time.perf_counter() - started,
            "rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state(),
        }, staging)
        staging.replace(checkpoint)

    # Epoch zero is recoverable even if the first epoch hits the deadline.
    if epochs_run == 0:
        save_boundary()
    for epoch in range(epochs_run + 1, max_epochs + 1):
        if stale >= patience:
            break
        if time.perf_counter() >= deadline:
            raise SessionBudgetExhausted(f"{spec.run_id}: resume from epoch {epochs_run}")
        model.train()
        order = torch.randperm(len(x_tr), device=device)
        running = torch.zeros((), dtype=torch.float64, device=device)
        for i in range(0, len(order), batch_size):
            if time.perf_counter() >= deadline:
                raise SessionBudgetExhausted(f"{spec.run_id}: resume from epoch {epochs_run}")
            idx = order[i:i+batch_size]
            optimiser.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model.forecast_target(x_tr[idx]), y_tr[idx])
            loss.backward()
            optimiser.step()
            running += loss.detach().double() * len(idx)
        schedule.step()
        epochs_run = epoch
        train_loss = float(running.item()) / len(x_tr)
        if not np.isfinite(train_loss):
            raise ValueError(f"{spec.run_id}: non-finite training loss in epoch {epoch}")
        val = _mean_loss(model, x_va, y_va)
        if not np.isfinite(val):
            raise ValueError(f"{spec.run_id}: non-finite validation loss")
        if val < best_val - 1e-9:
            best_val, stale = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        save_boundary()
    if best_state is None:
        raise ValueError(f"{spec.run_id}: no finite validation checkpoint")
    model.load_state_dict(best_state)
    return model, TrainOutcome(
        run_id=spec.run_id, epochs_run=epochs_run, best_val_mse=best_val,
        train_loss=train_loss, wall_time_s=prior_seconds + time.perf_counter()-started,
        n_parameters=model.n_parameters(), device=str(device),
    )


def scale_invariance_check(
    model: Forecaster, x: Tensor, y: Tensor, c: float = 100.0
) -> tuple[float, float]:
    """``(MSE(x), MSE(c x) / c^2)``: equal while instance normalisation is active.

    Scaling the input by ``c`` scales the denormalised forecast by ``c`` as well,
    so the invariant is ``MSE(c x) / c^2 == MSE(x)``, not ``MSE(c x) == MSE(x)``.
    """
    return _mean_loss(model, x, y), _mean_loss(model, x * c, y * c) / (c * c)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def code_sha256() -> str:
    """Digest of the package source, including the upstream copies.

    Off-repo (Kaggle) there is no git sha, so this names the code that ran.
    Line endings are normalised, so Windows and Linux checkouts agree.
    ``CODE_SHA256_OVERRIDE`` supplies the value where no files exist.
    """
    if CODE_SHA256_OVERRIDE is not None:
        return CODE_SHA256_OVERRIDE
    return tree_digest(Path(__file__).resolve().parent)


def tree_digest(root: Path) -> str:
    """sha256 over every ``*.py`` under ``root``, including the upstream copies.

    Files are taken in order of their relative POSIX path, a string order that is
    the same on Windows and Linux, and line endings are normalised to LF.
    """
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py"), key=lambda p: p.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def resolve_input_parquet(parquet: Path | str | None = None) -> Path:
    """The input parquet this process consumed: argument, then environment, then default."""
    if parquet is not None:
        return Path(parquet)
    from_env = os.environ.get(INPUT_PARQUET_ENV)
    return Path(from_env) if from_env else DEFAULT_PARQUET


def _input_sha256(parquet: Path | str | None = None) -> tuple[str, str]:
    """Hash the actual input bytes."""
    try:
        return hashlib.sha256(resolve_input_parquet(parquet).read_bytes()).hexdigest(), "file-digest"
    except OSError:
        return "unknown", "unresolved"


def environment(device: torch.device | None = None) -> dict:
    """Library versions, GPU and upstream commit, recorded with every run."""
    versions = {"python": platform.python_version(), "torch": str(torch.__version__),
                "numpy": np.__version__, "polars": pl.__version__}
    try:
        import sklearn
        versions["sklearn"] = sklearn.__version__
    except ImportError:
        versions["sklearn"] = None
    cuda = device is not None and torch.device(device).type == "cuda"
    return {
        **versions,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu": torch.cuda.get_device_name(device) if cuda else None,
        "upstream": f"{UPSTREAM_REPO}@{UPSTREAM_COMMIT}",
    }


def write_artifacts(
    model: Forecaster,
    tensors: OriginTensors,
    spec: RunSpec,
    cfg: Architecture,
    outcome: TrainOutcome,
    device: torch.device,
    root: Path = ARTIFACTS,
    requested_config: Architecture | None = None,
) -> tuple[Path, Path]:
    """Publish predictions and weights atomically, then the metadata that marks the run complete.

    ``meta`` is written last and carries the sha256 of both files, so a run is
    complete only when all three agree. The run's checkpoint is removed after.
    """
    (root / "preds").mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)

    frames = []
    for b, split in zip(tensors.block_labels, tensors.test_blocks):
        if len(split) == 0:
            continue
        x, _ = _to_device(split, device)
        pred = predict(model, x)
        n, h = pred.shape
        frames.append(
            pl.DataFrame(
                {
                    "block": np.full(n * h, b, dtype=np.int8),
                    "step": np.tile(np.arange(1, h + 1, dtype=np.int16), n),
                    "timestamp": np.repeat(split.ts, h),
                    "forecast_origin": np.repeat(split.ts, h),
                    "input_start": np.repeat(split.ts - cfg.seq_len * 3_600_000, h),
                    "target_timestamp": (split.ts[:, None] + np.arange(h) * 3_600_000).reshape(-1),
                    "y_true": split.y.reshape(-1),
                    "y_pred": pred.reshape(-1),
                }
            )
        )
    preds = (
        pl.concat(frames)
        if frames
        else pl.DataFrame(
            schema={
                "block": pl.Int8,
                "step": pl.Int16,
                "timestamp": pl.Int64,
                "forecast_origin": pl.Int64,
                "input_start": pl.Int64,
                "target_timestamp": pl.Int64,
                "y_true": pl.Float32,
                "y_pred": pl.Float32,
            }
        )
    )

    preds_path = root / "preds" / f"{spec.run_id}.parquet"
    meta_path = root / "meta" / f"{spec.run_id}.json"
    staging_preds = preds_path.with_suffix(".parquet.tmp")
    preds.write_parquet(staging_preds)
    staging_preds.replace(preds_path)

    weights_path = root / "weights" / f"{spec.run_id}.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    staging_weights = weights_path.with_suffix(".pt.tmp")
    torch.save(model.state_dict(), staging_weights)
    staging_weights.replace(weights_path)
    input_parquet = resolve_input_parquet()
    input_digest, input_provenance = _input_sha256(input_parquet)
    schedule = cfg.schedule() if hasattr(cfg, "schedule") else None
    meta = {
        "run_id": spec.run_id,
        "prediction_schema_version": 2,
        "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "torch_version": str(torch.__version__),
        "predictions_sha256": hashlib.sha256(preds_path.read_bytes()).hexdigest(),
        "timestamp_semantics": "forecast_origin",
        "forecast_origin_definition": "first target bar open, UTC",
        "evaluation_population": "surviving contiguous windows",
        "selection_time_ms": int(tensors.origin.test_start.timestamp() * 1000),
        "training_cutoff_ms": int(tensors.origin.train_sub_end.timestamp() * 1000),
        "latest_training_target_ms": int(tensors.train.ts.max() + (cfg.pred_len-1)*3600000),
        "spec": asdict(spec),
        "config": asdict(cfg),
        "requested_config": asdict(requested_config or cfg),
        "schedule": asdict(schedule) if schedule is not None else None,
        "origin": tensors.origin.label,
        "origin_index": tensors.origin.index,
        "block_labels": list(tensors.block_labels),
        "k": tensors.k,
        "variates": list(tensors.scaler.columns),
        "git_sha": _git_sha(),
        "code_sha256": code_sha256(),
        "input_parquet": str(input_parquet),
        "input_sha256": input_digest,
        "input_sha256_source": input_provenance,
        "environment": environment(device),
        "n_train": len(tensors.train),
        "training_selection": tensors.training_selection,
        "n_val": len(tensors.val),
        "n_test_per_block": [len(s) for s in tensors.test_blocks],
        "mu_g": float(tensors.scaler.mean[0]),
        "sigma_g": float(tensors.scaler.std[0]),
        "mu_over_sigma": tensors.scaler.target_mu_over_sigma,
        "naive_rw_z": tensors.naive_rw_z,
        "epochs_run": outcome.epochs_run,
        "best_val_mse": outcome.best_val_mse,
        "train_loss": outcome.train_loss,
        "wall_time_s": outcome.wall_time_s,
        "n_parameters": outcome.n_parameters,
        "n_allocated_parameters": sum(p.numel() for p in model.parameters()) if isinstance(model, nn.Module) else outcome.n_parameters,
        "loss_target": "target",
        "reached_epoch_cap": bool(schedule is not None and outcome.epochs_run >= schedule.max_epochs),
        "device": outcome.device,
        "status": "complete",
    }
    staging_meta = meta_path.with_suffix(".json.tmp")
    staging_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    staging_meta.replace(meta_path)
    (root / "checkpoints" / f"{spec.run_id}.pt").unlink(missing_ok=True)
    return preds_path, meta_path


def is_complete(
    run_id: str, root: Path = ARTIFACTS, *, strict: bool = False,
    cfg: Architecture | None = None, columns: tuple[str, ...] | None = None,
) -> bool:
    """Whether ``run_id`` is complete under ``root``.

    Loose: both files exist and ``meta`` says complete. Strict (resume): the
    input digest, code digest, requested config, schedule, variates, prediction
    schema and both file hashes must also match the current run request.
    """
    preds = root / "preds" / f"{run_id}.parquet"
    meta_path = root / "meta" / f"{run_id}.json"
    if not (preds.is_file() and meta_path.is_file()):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") != "complete":
            return False
        if not strict:
            return True
        digest, _ = _input_sha256()
        if digest == "unknown" or meta.get("input_sha256") != digest:
            return False
        if meta.get("code_sha256") != code_sha256() or meta.get("run_id") != run_id:
            return False
        if cfg is None or meta.get("requested_config") != json.loads(json.dumps(asdict(cfg))):
            return False
        schedule = asdict(cfg.schedule()) if hasattr(cfg, "schedule") else None
        if meta.get("schedule") != json.loads(json.dumps(schedule)):
            return False
        if columns is not None and meta.get("variates") != list(columns):
            return False
        required = {"block", "step", "timestamp", "forecast_origin", "input_start",
                    "target_timestamp", "y_true", "y_pred"}
        if meta.get("prediction_schema_version") != 2 or not required <= set(pl.read_parquet_schema(preds)):
            return False
        weights = root / "weights" / f"{run_id}.pt"
        if not weights.is_file() or meta.get("weights_sha256") != hashlib.sha256(weights.read_bytes()).hexdigest():
            return False
        if meta.get("predictions_sha256") != hashlib.sha256(preds.read_bytes()).hexdigest():
            return False
        return True
    except (OSError, ValueError, TypeError, pl.exceptions.PolarsError):
        return False

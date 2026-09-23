"""The 900-run manifest and the executor that walks it on one or two GPUs.

Three models, each at 15 origins x K in {1, 4, 8, 12} x 5 seeds at H = 24, run in
the order iTransformer, Ridge, vanilla Transformer. Runs are parallel across
devices, one worker thread per GPU pulling from a shared queue; batches are not
split. A run is complete only when its predictions, weights and metadata agree,
and resume accepts only outputs of the current code and input. The grid refuses
to start until the design digest has been frozen after the pilot.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import queue
import shutil
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch

from itransformer_btc.baselines import RidgeConfig, assert_baseline_alignment
from itransformer_btc.config import (
    K_LADDER,
    ORIGINS,
    PRED_LEN,
    SEEDS,
    SELECTION_SEED,
    SEQ_LEN,
    TRAIN_WINDOW_LIMIT,
    Origin,
)
from itransformer_btc.features import ladder_columns
from itransformer_btc.model import ITransformerConfig, VanillaConfig
from itransformer_btc.splits import OriginTensors, build_origin_tensors
from itransformer_btc.train import (
    ARTIFACTS,
    DEFAULT_PARQUET,
    INPUT_PARQUET_ENV,
    Architecture,
    RunSpec,
    SessionBudgetExhausted,
    TrainingSession,
    _input_sha256,
    code_sha256,
    is_complete,
    write_artifacts,
)

#: Arm name to the model tag in ``run_id``, in grid order.
ARM_MODEL_TAG: dict[str, str] = {"main": "itr", "ridge": "rdg", "vanilla": "vtr"}
ALL_ARMS: tuple[str, ...] = tuple(ARM_MODEL_TAG)

#: Defaults for the command-line entry point; the notebook sets its own deadline.
SESSION_BUDGET_H: float = 11.0
RESERVE_H: float = 0.5

#: Output folders that belong to one run.
RUN_FOLDERS: tuple[tuple[str, str], ...] = (("preds", ".parquet"), ("weights", ".pt"), ("meta", ".json"))


@dataclass(frozen=True, slots=True)
class RunCell:
    """One (model, origin, K, H, seed) cell of the grid."""

    arm: str
    origin_index: int
    k: int
    pred_len: int
    seed: int

    @property
    def model_tag(self) -> str:
        return ARM_MODEL_TAG[self.arm]

    @property
    def spec(self) -> RunSpec:
        return RunSpec(self.model_tag, self.origin_index, self.k, self.pred_len, self.seed)

    @property
    def run_id(self) -> str:
        return self.spec.run_id

    @property
    def tensor_key(self) -> tuple[int, int, int]:
        """All three models of a cell share windows, scaler and training sample."""
        return (self.origin_index, self.k, self.pred_len)

    def origin(self) -> Origin:
        return ORIGINS[self.origin_index - 1]

    def columns(self) -> tuple[str, ...]:
        return tuple(ladder_columns(self.k))

    def model_config(self) -> Architecture:
        if self.arm == "main":
            return ITransformerConfig(pred_len=self.pred_len)
        if self.arm == "ridge":
            return RidgeConfig(pred_len=self.pred_len, k=self.k)
        if self.arm == "vanilla":
            return VanillaConfig(pred_len=self.pred_len, k=self.k)
        raise ValueError(f"unknown arm {self.arm!r}")

    def reference_run_id(self) -> str:
        """The iTransformer run whose windows this cell must share."""
        return RunSpec(ARM_MODEL_TAG["main"], self.origin_index, self.k, self.pred_len, SEEDS[0]).run_id


def manifest(arms: tuple[str, ...] = ALL_ARMS) -> list[RunCell]:
    """Every run, in grid order: 300 per model, 900 for all three."""
    cells = [RunCell(arm, o.index, k, PRED_LEN, s)
             for arm in arms for o in ORIGINS for k in K_LADDER for s in SEEDS]
    if len({c.run_id for c in cells}) != len(cells):
        raise ValueError("manifest run ids are not unique")
    return cells


def discover_roots(working: Path = ARTIFACTS, inputs: Path = Path("/kaggle/input")) -> list[Path]:
    """The working directory, then every attached folder holding run outputs, at any depth."""
    roots = [Path(working)]
    marks = {"meta", "preds", "validation", "checkpoints"}
    if Path(inputs).exists():
        for dirpath, dirnames, _ in os.walk(inputs):
            if marks & set(dirnames):
                roots.append(Path(dirpath))
            dirnames[:] = [d for d in dirnames if d not in {*marks, "weights"}]
    seen: set[str] = set()
    return [r for r in roots if not (str(r) in seen or seen.add(str(r)))]


def completed_run_ids(roots: list[Path], code_digest: str = "") -> set[str]:
    """Run ids with predictions and a ``complete`` meta, optionally of one code vintage."""
    done: set[str] = set()
    for root in roots:
        meta_dir = Path(root) / "meta"
        if not meta_dir.is_dir():
            continue
        for meta_path in meta_dir.glob("*.json"):
            run_id = meta_path.stem
            if not (Path(root) / "preds" / f"{run_id}.parquet").exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if meta.get("status") != "complete":
                continue
            if code_digest and meta.get("code_sha256") != code_digest:
                continue
            done.add(run_id)
    return done


def pending(cells: list[RunCell], roots: list[Path]) -> list[RunCell]:
    """Cells without a strictly complete run in the first root that holds one."""
    todo = []
    for cell in cells:
        candidates = [root for root in roots
                      if (root / "preds" / f"{cell.run_id}.parquet").exists()
                      or (root / "meta" / f"{cell.run_id}.json").exists()]
        if not candidates or not is_complete(cell.run_id, candidates[0], strict=True,
                                             cfg=cell.model_config(), columns=cell.columns()):
            todo.append(cell)
    return todo


def consolidate_resume_outputs(cells: list[RunCell], roots: list[Path], out_root: Path) -> int:
    """Copy strictly complete runs from attached roots into ``out_root``; return how many.

    Copies, not links, so the next saved output is self-contained. Checkpoints and
    cached validation fits are copied too, so an interrupted run resumes mid-way.
    """
    out_root = Path(out_root)
    copied = 0
    for cell in cells:
        cfg = cell.model_config()
        if is_complete(cell.run_id, out_root, strict=True, cfg=cfg, columns=cell.columns()):
            continue
        for root in roots:
            root = Path(root)
            if root == out_root or not is_complete(cell.run_id, root, strict=True, cfg=cfg,
                                                    columns=cell.columns()):
                continue
            for folder, suffix in RUN_FOLDERS:
                source = root / folder / f"{cell.run_id}{suffix}"
                destination = out_root / folder / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                staging = destination.with_suffix(destination.suffix + ".tmp")
                shutil.copyfile(source, staging)
                staging.replace(destination)
            copied += 1
            break
    for root in roots:
        if Path(root) == out_root:
            continue
        for folder, pattern in (("checkpoints", "*.pt"), ("validation", "*.json")):
            for source in (Path(root) / folder).glob(pattern):
                destination = out_root / folder / source.name
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    staging = destination.with_suffix(destination.suffix + ".tmp")
                    shutil.copyfile(source, staging)
                    staging.replace(destination)
    return copied


def resume_check(cells: list[RunCell], roots: list[Path], out_root: Path) -> tuple[int, int]:
    """Carry attached runs forward and refuse to continue if any were left behind.

    Returns ``(attached, available)``: complete runs of this code vintage found in
    attached inputs, and strictly complete runs now in ``out_root``.

    Raises:
        RuntimeError: If an attached run of this vintage could not be carried
            forward, so the session would silently retrain it.
    """
    wanted = {c.run_id for c in cells}
    attached = completed_run_ids([r for r in roots if Path(r) != Path(out_root)], code_sha256()) & wanted
    consolidate_resume_outputs(cells, roots, out_root)
    available = {c.run_id for c in cells} - {c.run_id for c in pending(cells, [Path(out_root)])}
    missing = sorted(attached - available)
    if missing:
        raise RuntimeError(
            f"{len(missing)} attached runs of this code vintage failed the strict check "
            f"(first: {missing[0]}). Attach the complete output (preds, weights, meta) "
            f"of the previous session instead of retraining."
        )
    return len(attached), len(available)


def design_digest(cells: list[RunCell] | None = None) -> str:
    """sha256 of the code and the manifest: what the frozen design commits to."""
    run_ids = [c.run_id for c in (cells or manifest())]
    payload = json.dumps({"code": code_sha256(), "runs": run_ids}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require_frozen_design(frozen: str | None, cells: list[RunCell] | None = None) -> str:
    """Return the design digest, or refuse to run the grid before and after a design change.

    Raises:
        RuntimeError: If no digest has been frozen yet, or the current one differs.
    """
    digest = design_digest(cells)
    if frozen is None:
        raise RuntimeError(
            f"the design is not frozen. Run the pilot, record {digest} as the frozen "
            f"design digest, then start the grid."
        )
    if digest != frozen:
        raise RuntimeError(
            f"design digest {digest[:12]} differs from the frozen {frozen[:12]}: the code or "
            f"the manifest changed after the freeze."
        )
    return digest


class BudgetGuard:
    """A monotonic deadline, and the decision whether another run fits before it."""

    def __init__(self, budget_h: float = SESSION_BUDGET_H, reserve_h: float = RESERVE_H,
                 *, started_at: float | None = None) -> None:
        if not np.isfinite(budget_h + reserve_h) or min(budget_h, reserve_h) < 0:
            raise ValueError("budget and reserve must be finite, nonnegative hours")
        start = time.perf_counter() if started_at is None else started_at
        self.deadline = start + (budget_h - reserve_h) * 3600.0
        self.durations: list[float] = []

    def record(self, seconds: float) -> None:
        self.durations.append(seconds)

    @property
    def mean_run_s(self) -> float:
        return sum(self.durations) / len(self.durations) if self.durations else 120.0

    @property
    def remaining_s(self) -> float:
        return self.deadline - time.perf_counter()

    def may_start(self) -> bool:
        return self.remaining_s > max(120.0, 1.5 * max(self.durations[-20:], default=120.0))


class _TensorCache:
    """A small LRU of per-(origin, K) tensors, shared by the three models of a cell."""

    def __init__(self, features: pl.DataFrame, size: int = 2) -> None:
        self.features = features
        self.size = size
        self._store: OrderedDict[tuple, OriginTensors] = OrderedDict()

    def get(self, cell: RunCell) -> OriginTensors:
        key = cell.tensor_key
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        tensors = build_origin_tensors(
            self.features, cell.origin(), cell.k, seq_len=SEQ_LEN, pred_len=cell.pred_len,
            train_window_limit=TRAIN_WINDOW_LIMIT, selection_seed=SELECTION_SEED,
        )
        self._store[key] = tensors
        while len(self._store) > self.size:
            self._store.popitem(last=False)
        return tensors


@dataclass(frozen=True, slots=True)
class ExecutionSummary:
    completed: int
    skipped: int
    failed: int
    remaining: int
    wall_time_s: float
    mean_run_s: float

    def __str__(self) -> str:
        return (f"completed {self.completed}  skipped {self.skipped}  failed {self.failed}  "
                f"remaining {self.remaining}\nwall {self.wall_time_s / 3600:.2f} h  "
                f"mean run {self.mean_run_s:.1f} s")


def _run_cell(cell: RunCell, cache: _TensorCache, device: torch.device, out_root: Path,
              roots: list[Path], guard: BudgetGuard):
    """Fit one cell and write its files."""
    tensors = cache.get(cell)
    requested = cell.model_config()
    with TrainingSession(out_root, roots, guard.deadline):
        model, cfg, outcome = requested.fit(tensors, cell.spec, device=device)
    write_artifacts(model, tensors, cell.spec, cfg, outcome, device, root=out_root,
                    requested_config=requested)
    return tensors, outcome


def _check_alignment(cell: RunCell, roots: list[Path], log) -> None:
    """A comparator must be scored on the iTransformer's exact windows."""
    if cell.arm == "main":
        return
    try:
        assert_baseline_alignment(cell.run_id, cell.reference_run_id(), roots)
    except FileNotFoundError:
        log(f"  {cell.run_id}: window alignment unchecked, {cell.reference_run_id()} not on disk")


def visible_devices() -> list[torch.device]:
    if not torch.cuda.is_available():
        return [torch.device("cpu")]
    return [torch.device("cuda", i) for i in range(torch.cuda.device_count())]


def execute_parallel(cells: list[RunCell], features: pl.DataFrame, *,
                     devices: list[torch.device] | None = None, out_root: Path = ARTIFACTS,
                     roots: list[Path] | None = None, guard: BudgetGuard | None = None,
                     log=print) -> ExecutionSummary:
    """Run every pending cell, one worker per device, stopping cleanly at the deadline.

    A window misalignment between a comparator and the iTransformer stops the grid.
    """
    devices = devices or visible_devices()
    guard = guard or BudgetGuard()
    roots = list(dict.fromkeys([Path(out_root), *(roots or discover_roots(out_root))]))
    pending_ids = {c.run_id for c in pending(cells, roots)}
    queue_ = list(cells)
    cursor = 0
    completed = skipped = failed = 0
    state = threading.Lock()
    fatal: list[BaseException] = []
    started = time.perf_counter()
    log(f"workers on {[str(d) for d in devices]}")

    def take():
        nonlocal cursor
        with state:
            if fatal or cursor >= len(queue_) or not guard.may_start():
                return None
            cursor += 1
            return cursor, queue_[cursor - 1]

    def worker(device: torch.device) -> None:
        nonlocal completed, skipped, failed
        cache = _TensorCache(features)
        while (item := take()) is not None:
            position, cell = item
            if cell.run_id not in pending_ids:
                with state:
                    skipped += 1
                continue
            began = time.perf_counter()
            try:
                tensors, outcome = _run_cell(cell, cache, device, out_root, roots, guard)
            except SessionBudgetExhausted as exc:
                log(f"PAUSED: {exc}; save this output and attach it next session")
                break
            except Exception as exc:
                with state:
                    failed += 1
                log(f"[{position}/{len(queue_)}] {device} {cell.run_id} FAILED: {exc!r}")
                continue
            try:
                _check_alignment(cell, roots, log)
            except Exception as exc:
                with state:
                    fatal.append(exc)
                return
            elapsed = time.perf_counter() - began
            with state:
                guard.record(elapsed)
                completed += 1
            log(f"[{position}/{len(queue_)}] {device} {cell.run_id}  epochs={outcome.epochs_run}  "
                f"val={outcome.best_val_mse:.6f}  {elapsed:.1f}s  n_train={len(tensors.train)}")

    threads = [threading.Thread(target=worker, args=(d,), name=f"grid-{d}", daemon=True)
               for d in devices]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if fatal:
        raise RuntimeError("a comparator was scored on windows other than the iTransformer's") from fatal[0]
    if cursor < len(queue_):
        log(f"budget guard: {len(queue_) - cursor} cells unstarted; resume picks them up next session")
    return ExecutionSummary(completed=completed, skipped=skipped, failed=failed,
                            remaining=len(pending(queue_, roots)),
                            wall_time_s=time.perf_counter() - started, mean_run_s=guard.mean_run_s)


def validation_fit(tensors: OriginTensors, spec: RunSpec, cfg: Architecture, *,
                   device: torch.device, out_root: Path | None = None,
                   roots: list[Path] | None = None) -> dict:
    """Fit on the training sub-block and score on validation only, caching by identity.

    Nothing here reads a test block.
    """
    identity = json.loads(json.dumps({
        "spec": asdict(spec), "config": asdict(cfg), "code_sha256": code_sha256(),
        "input_sha256": _input_sha256()[0], "torch": str(torch.__version__),
        "device_type": device.type, "training_selection": tensors.training_selection,
    }))
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    destination = Path(out_root) / "validation" / f"{key}.json" if out_root else None
    if destination:
        for root in dict.fromkeys([Path(out_root), *(roots or [])]):
            path = root / "validation" / f"{key}.json"
            if path.exists():
                cached = json.loads(path.read_text(encoding="utf-8"))
                if cached.get("identity") == identity and np.isfinite(cached.get("val_mse", np.nan)):
                    return cached
    session = (TrainingSession(Path(out_root), list(roots or []), math.inf)
               if out_root else contextlib.nullcontext())
    with session:
        _, fitted, outcome = cfg.fit(tensors, spec, device=device)
    row = {"identity": identity, "val_mse": outcome.best_val_mse, "epochs_run": outcome.epochs_run,
           "wall_time_s": outcome.wall_time_s, "n_val": len(tensors.val),
           "config": asdict(fitted)}
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_suffix(".json.tmp")
        staging.write_text(json.dumps(row, indent=2), encoding="utf-8")
        staging.replace(destination)
        (Path(out_root) / "checkpoints" / f"{spec.run_id}.pt").unlink(missing_ok=True)
    return row


@dataclass(frozen=True, slots=True)
class PilotResult:
    """Validation MSE and wall time per model and rung at the first origin."""

    rows: tuple[dict, ...]

    def mean_wall_s(self) -> dict[str, float]:
        """Mean seconds per run, per model tag."""
        out: dict[str, list[float]] = {}
        for row in self.rows:
            out.setdefault(row["model"], []).append(row["wall_time_s"])
        return {tag: float(np.mean(v)) for tag, v in out.items()}

    def __str__(self) -> str:
        lines = ["model  K   val MSE    Naive-RW   epochs  seconds"]
        lines += [f"{r['model']:5s} {r['k']:2d}  {r['val_mse']:.6f}  {r['naive_val_mse']:.6f}  "
                  f"{r['epochs_run']:6d}  {r['wall_time_s']:7.1f}" for r in self.rows]
        return "\n".join(lines)


def pilot(features: pl.DataFrame, *, origin_index: int = 1, rungs: tuple[int, ...] = K_LADDER,
          seed: int = SEEDS[0], devices: list[torch.device] | None = None,
          out_root: Path | None = None, roots: list[Path] | None = None, log=print) -> PilotResult:
    """Fit every model at every rung on one origin's validation split and time it.

    An engineering check: every loss must be finite, and the timings plan the
    grid. Nothing is chosen from these numbers.
    """
    devices = devices or visible_devices()
    tasks = [RunCell(arm, origin_index, k, PRED_LEN, seed) for arm in ALL_ARMS for k in rungs]
    free: queue.Queue = queue.Queue()
    for device in devices:
        free.put(device)
    cache_lock = threading.Lock()
    cache = _TensorCache(features, size=len(rungs))

    def run(cell: RunCell) -> dict:
        device = free.get()
        try:
            with cache_lock:
                tensors = cache.get(cell)
            spec = RunSpec(f"pilot{cell.model_tag}", origin_index, cell.k, PRED_LEN, seed)
            row = validation_fit(tensors, spec, cell.model_config(), device=device,
                                 out_root=out_root, roots=roots)
        finally:
            free.put(device)
        naive = float(np.mean((tensors.val.y - tensors.naive_rw_z) ** 2))
        if not np.isfinite(row["val_mse"]):
            raise ValueError(f"pilot {spec.run_id}: non-finite validation MSE")
        log(f"pilot {spec.run_id} on {device}: val {row['val_mse']:.6f} (Naive-RW {naive:.6f}), "
            f"{row['wall_time_s']:.1f}s")
        return {"model": cell.model_tag, "k": cell.k, "val_mse": row["val_mse"],
                "naive_val_mse": naive, "epochs_run": row["epochs_run"],
                "wall_time_s": row["wall_time_s"]}

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        rows = tuple(pool.map(run, tasks))
    return PilotResult(rows=rows)


def session_plan(mean_wall_s: dict[str, float], cells: list[RunCell], *, devices: int,
                 session_left_h: float, usable_session_h: float,
                 weekly_left_h: float | None = None) -> str:
    """Hours and sessions the pending cells need, from the pilot's timings."""
    by_model: dict[str, int] = {}
    for cell in cells:
        by_model[cell.model_tag] = by_model.get(cell.model_tag, 0) + 1
    gpu_h = {tag: n * mean_wall_s.get(tag, float("nan")) / 3600 for tag, n in by_model.items()}
    wall_h = sum(gpu_h.values()) / max(1, devices)
    lines = [f"pending runs {by_model}",
             "GPU-hours per model " + ", ".join(f"{t} {h:.2f}" for t, h in gpu_h.items()),
             f"wall-clock on {devices} device(s): {wall_h:.2f} h; this session has {session_left_h:.2f} h left"]
    if wall_h > session_left_h:
        lines.append(f"sessions needed: {1 + math.ceil((wall_h - session_left_h) / usable_session_h)}")
    if weekly_left_h is not None:
        lines.append(f"weekly quota left {weekly_left_h:.1f} GPU-h against {wall_h * devices:.1f} needed")
    return "\n".join(lines)


def build_feature_frame(parquet: Path = DEFAULT_PARQUET) -> pl.DataFrame:
    """Load the input parquet and compute the twelve variates."""
    from itransformer_btc.features import build_features
    from itransformer_btc.segments import load_bars, usable_mask

    return build_features(usable_mask(load_bars(parquet)))


def _main(argv: list[str] | None = None) -> int:
    """Command-line grid for a source checkout; the notebook is the primary launcher."""
    parser = argparse.ArgumentParser(description="Run the 900-run grid from a checkout.")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--out", type=Path, default=ARTIFACTS)
    parser.add_argument("--arms", type=str, default=",".join(ALL_ARMS))
    parser.add_argument("--budget-h", type=float, default=SESSION_BUDGET_H)
    parser.add_argument("--reserve-h", type=float, default=RESERVE_H)
    parser.add_argument("--design", type=str, default=None, help="the frozen design digest")
    args = parser.parse_args(argv)
    os.environ[INPUT_PARQUET_ENV] = str(args.parquet)
    cells = manifest(tuple(a.strip() for a in args.arms.split(",") if a.strip()))
    require_frozen_design(args.design)
    features = build_feature_frame(args.parquet)
    roots = discover_roots(args.out)
    summary = execute_parallel(cells, features, out_root=args.out, roots=roots,
                               guard=BudgetGuard(args.budget_h, args.reserve_h),
                               log=lambda msg: print(msg, flush=True))
    print(summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

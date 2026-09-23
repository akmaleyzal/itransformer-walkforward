"""Run each test file in a fresh process, so memory held by one file is freed before the next.

Usage: ``python tools/run_tests.py [tests/test_x.py ...]``. Logs land in ``.tmp/test-logs/``.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
logs = root / ".tmp" / "test-logs"
logs.mkdir(parents=True, exist_ok=True)
results = []
files = [Path(p) for p in sys.argv[1:]] or sorted((root / "tests").glob("test_*.py"))
for path in files:
    started = time.perf_counter()
    command = [sys.executable, "-X", "utf8", "-m", "pytest", str(path), "-q", "--tb=short"]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, encoding="utf-8")
    (logs / (path.stem + ".log")).write_text(result.stdout + result.stderr, encoding="utf-8")
    row = {"file": path.name, "exit_code": result.returncode,
           "wall_seconds": round(time.perf_counter() - started, 2),
           "tail": (result.stdout + result.stderr).strip().splitlines()[-6:]}
    results.append(row)
    (logs / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(row), flush=True)
raise SystemExit(any(r["exit_code"] for r in results))

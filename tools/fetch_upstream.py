"""Download the pinned iTransformer files and compare them with the recorded digests.

    python tools/fetch_upstream.py              # verify what GitHub serves today
    python tools/fetch_upstream.py --write DIR  # and keep the raw files under DIR

The digests live in ``itransformer_btc.upstream``. For every file this checks the
raw bytes against the pin, rebuilds the study's copy with ``derive`` and compares
it with ``vendor/``, so anyone holding the commit hash can repeat the check.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from itransformer_btc.upstream import (  # noqa: E402
    UPSTREAM_COMMIT,
    UPSTREAM_FILES,
    UPSTREAM_RAW_SHA256,
    derive,
    raw_url,
    vendor_root,
)


def fetch(path: str) -> bytes:
    request = urllib.request.Request(raw_url(path), headers={"User-Agent": "btc-walkforward-provenance"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the copied iTransformer files against GitHub")
    parser.add_argument("--write", type=Path, help="also save the raw files under this directory")
    args = parser.parse_args(argv)
    bad = 0
    for path, expected in UPSTREAM_RAW_SHA256.items():
        raw = fetch(path)
        copy = derive(path, raw.decode("utf-8")).encode("utf-8")
        checks = {
            "raw": hashlib.sha256(raw).hexdigest() == expected,
            "copy": hashlib.sha256(copy).hexdigest() == UPSTREAM_FILES[path],
            "vendor": (vendor_root() / path).read_bytes().replace(b"\r\n", b"\n") == copy,
        }
        bad += not all(checks.values())
        verdict = "ok " if all(checks.values()) else "BAD"
        print(f"{verdict} {path}  " + "  ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in checks.items()))
        if args.write:
            target = args.write / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    total = len(UPSTREAM_RAW_SHA256)
    print(f"thuml/iTransformer@{UPSTREAM_COMMIT[:8]}: {total - bad}/{total} files match")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

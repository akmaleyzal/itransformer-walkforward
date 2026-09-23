"""Repository hygiene: what may be committed.

No third-party documents (papers are cited by DOI, never redistributed), and no
cross-reference tokens that point into documents this repository does not carry.
Upstream copies and Stage 1 data are byte-pinned and exempt from the text scan.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXEMPT = ("data/raw/", "tests/fixtures/", "src/itransformer_btc/vendor/")
DOCUMENTS = re.compile(r"\.(pdf|epub|djvu|docx?|pptx?)$", re.I)
TOKENS = re.compile(r"\b[DA]\d{2}[a-z]?\b|§\s?\d")


def _tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return [p for p in out.splitlines() if (ROOT / p).is_file()]


def test_no_document_files_are_committed() -> None:
    assert not [p for p in _tracked() if DOCUMENTS.search(p)]


def test_tracked_text_has_no_cross_reference_tokens() -> None:
    offenders = []
    for path in _tracked():
        if path.startswith(EXEMPT):
            continue
        try:
            text = (ROOT / path).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        offenders += [f"{path}:{text.count(chr(10), 0, m.start()) + 1}: {m.group(0)}"
                      for m in TOKENS.finditer(text)]
    assert not offenders, offenders

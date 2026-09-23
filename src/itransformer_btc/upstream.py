"""The authors' iTransformer code, copied unchanged and loaded on demand.

``itr`` and ``vtr`` are the ``Model`` classes of ``model/iTransformer.py`` and
``model/Transformer.py`` in the official iTransformer repository. The files sit
under ``vendor/thuml_iTransformer`` as published, normalised only in whitespace:
LF line endings, one final newline, and whitespace-only lines emptied, which is
the form a notebook ``%%writefile`` cell produces. ``layers/SelfAttention_Family.py`` keeps
only ``FullAttention`` and ``AttentionLayer``; its imports of ``reformer_pytorch``
and ``einops`` serve attention variants this study does not use, and are the
only lines removed. :func:`derive` states that rule as code, so a reader can
rebuild every copy from the published files.

Both files define a class named ``Model`` and import siblings as ``layers.*``,
``model.*`` and ``utils.*``. :func:`load_upstream` therefore imports them in a
private module namespace and leaves ``sys.modules`` as it found it.

Upstream:
    Y. Liu, T. Hu, H. Zhang, H. Wu, S. Wang, L. Ma, and M. Long, "iTransformer:
    Inverted transformers are effective for time series forecasting," ICLR 2024.
    Code: https://github.com/thuml/iTransformer (MIT), commit
    c2426e68ca13f74aaec08045c5c724d8ad328124.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import re
import sys
from pathlib import Path
from typing import Final

UPSTREAM_REPO: Final = "https://github.com/thuml/iTransformer"
UPSTREAM_COMMIT: Final = "c2426e68ca13f74aaec08045c5c724d8ad328124"
UPSTREAM_LICENSE: Final = "MIT"

#: sha256 of each file as GitHub serves it at the commit.
UPSTREAM_RAW_SHA256: Final[dict[str, str]] = {
    "LICENSE": "29d2a4c09fa577780522219dc248466977f77cd420354c5e9a0e86550be2b849",
    "layers/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "layers/Embed.py": "2de7e5049a696d320a23a8bdb6e08e823cf7fa3f246a0a49555fcd729c5c8398",
    "layers/SelfAttention_Family.py": "38ac67428d528b8e145e4944ca059b40629a8446c891142602d5aaadc38dfd9e",
    "layers/Transformer_EncDec.py": "985c31f4ae187b08afaa9663dc960a40544b2ec33ab55389ceab3930edc1ce02",
    "model/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "model/iTransformer.py": "6f34777a5a12c253a293f79e7e1fd5b6f79ac100b443951e051e269e7e2542db",
    "model/Transformer.py": "7a816cc3971b99ee21d87479f25cc15498607d34539c95e5d53889588653ff4b",
    "utils/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "utils/masking.py": "02f453e52cfaec65e34132923add80e7132f188706680fa86d7bd884976e09ef",
}

#: Files that keep a subset of their top-level definitions, and the imports dropped
#: with the rest. Every other file is copied whole.
UPSTREAM_TRIMMED: Final[dict[str, tuple[tuple[str, ...], tuple[str, ...]]]] = {
    "layers/SelfAttention_Family.py": (
        ("FullAttention", "AttentionLayer"),
        ("reformer_pytorch", "einops"),
    ),
}

#: sha256 of each copy under ``vendor/thuml_iTransformer``, as :func:`derive` builds it.
UPSTREAM_FILES: Final[dict[str, str]] = {
    "LICENSE": "29d2a4c09fa577780522219dc248466977f77cd420354c5e9a0e86550be2b849",
    "layers/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "layers/Embed.py": "91f50ec4e47a5fbce5c1a064614140b0d8ac1d01bb2500b0d7afe3d2f0a0885b",
    "layers/SelfAttention_Family.py": "d6643899c34a69d59b4ea4194944533f5d47d14b2ac4841e4a41047c4589b471",
    "layers/Transformer_EncDec.py": "985c31f4ae187b08afaa9663dc960a40544b2ec33ab55389ceab3930edc1ce02",
    "model/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "model/iTransformer.py": "48b424b0e3d77eec3610ca79f083b415c8ba96032ee598ad3d05383880e768e4",
    "model/Transformer.py": "7a816cc3971b99ee21d87479f25cc15498607d34539c95e5d53889588653ff4b",
    "utils/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "utils/masking.py": "02f453e52cfaec65e34132923add80e7132f188706680fa86d7bd884976e09ef",
}

#: Top-level package names the upstream files import each other through.
_NAMESPACES: Final = ("layers", "model", "utils")

#: The two classes, once :func:`load_upstream` has run.
_LOADED: dict[str, type] = {}


def permalink(path: str) -> str:
    """The file's page on GitHub at the pinned commit."""
    return f"{UPSTREAM_REPO}/blob/{UPSTREAM_COMMIT}/{path}"


def raw_url(path: str) -> str:
    """The file's raw bytes at the pinned commit."""
    return f"https://raw.githubusercontent.com/thuml/iTransformer/{UPSTREAM_COMMIT}/{path}"


def canonical(text: str) -> str:
    """LF line endings, whitespace-only lines emptied, one trailing newline.

    This is the form a file keeps after a notebook ``%%writefile`` cell: IPython
    dedents a cell before running it, which empties whitespace-only lines, and a
    cell cannot end without a newline. An empty file stays empty.
    """
    text = re.sub(r"^[ \t]+$", "", text.replace("\r\n", "\n"), flags=re.M)
    return text.rstrip("\n") + "\n" if text.strip() else ""


def derive(path: str, raw: str) -> str:
    """The copy of ``path`` this study uses, built from the published text.

    A whole file is only canonicalised. A trimmed file keeps its imports, minus
    the dropped modules, and the named top-level definitions with the blank lines
    that precede them. Lines are only ever deleted, never edited or added.
    """
    text = canonical(raw)
    if path not in UPSTREAM_TRIMMED:
        return text
    keep, dropped = UPSTREAM_TRIMMED[path]
    lines = text.splitlines(keepends=True)
    kept: set[int] = set()
    for node in ast.parse(text).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            roots = ([node.module or ""] if isinstance(node, ast.ImportFrom)
                     else [alias.name for alias in node.names])
            if not any(root.split(".")[0] in dropped for root in roots):
                kept.update(range(node.lineno - 1, node.end_lineno))
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in keep:
            first = node.lineno - 1
            while first > 0 and not lines[first - 1].strip():
                first -= 1
            kept.update(range(first, node.end_lineno))
    return canonical("".join(lines[i] for i in sorted(kept)))


def vendor_root() -> Path:
    """Where the copies live in a source checkout."""
    return Path(__file__).resolve().parent / "vendor" / "thuml_iTransformer"


def verify_upstream(root: Path) -> None:
    """Raise unless every copy under ``root`` matches its pinned digest."""
    bad = []
    for path, expected in UPSTREAM_FILES.items():
        file = Path(root) / path
        actual = (hashlib.sha256(canonical(file.read_text(encoding="utf-8")).encode("utf-8")).hexdigest()
                  if file.is_file() else "missing")
        if actual != expected:
            bad.append(f"{path}: {actual}")
    if bad:
        raise ValueError(f"upstream copies differ from commit {UPSTREAM_COMMIT[:8]}: {bad}")


def load_upstream(root: Path) -> dict[str, type]:
    """Verify the copies under ``root``, then import both ``Model`` classes.

    Any module already registered under ``layers``, ``model`` or ``utils`` is set
    aside and restored afterwards, and the upstream modules are unregistered, so
    no other library can later import them by accident.
    """
    root = Path(root)
    verify_upstream(root)
    owned = lambda name: name.split(".")[0] in _NAMESPACES  # noqa: E731
    saved = {name: module for name, module in sys.modules.items() if owned(name)}
    for name in saved:
        del sys.modules[name]
    sys.path.insert(0, str(root))
    try:
        loaded = {
            "itr": importlib.import_module("model.iTransformer").Model,
            "vtr": importlib.import_module("model.Transformer").Model,
        }
    finally:
        sys.path.remove(str(root))
        for name in [name for name in sys.modules if owned(name)]:
            del sys.modules[name]
        sys.modules.update(saved)
    _LOADED.update(loaded)
    return dict(loaded)


def upstream_model(name: str) -> type:
    """``"itr"`` or ``"vtr"``: the upstream class, loading the checkout's copies if needed."""
    if not _LOADED:
        load_upstream(vendor_root())
    return _LOADED[name]

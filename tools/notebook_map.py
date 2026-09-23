"""Render ``notebooks/btc_walkforward_3model.ipynb`` as a navigable Markdown map.

The notebook is the primary surface and ``src/`` its tested projection, but an
``.ipynb`` is not a format documentation tooling reads. This writes its shape:
the phases, the steps with the files each reads and writes, the module cells
and the upstream copies. Definition cells are exact projections of ``src/``, so
their code is named rather than repeated. Everything is read from the notebook
itself, never from the generator's constants.

Usage::

    python tools/notebook_map.py            # writes docs/NOTEBOOK_MAP.md
    python tools/notebook_map.py --check    # exit 1 if the file is stale
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "btc_walkforward_3model.ipynb"
OUTPUT = ROOT / "docs" / "NOTEBOOK_MAP.md"

_TAG = re.compile(r"<[^>]+>")
_HEADING = re.compile(r"<h([1-4])[^>]*>(.*?)</h\1>", re.S | re.I)
_MARKDOWN_HEADING = re.compile(r"^(#{1,4})[ \t]+(.+)$", re.M)
_PARAGRAPH = re.compile(r"<p[^>]*>(.*?)</p>", re.S | re.I)


def _plain(markup: str) -> str:
    """Strip the banner HTML down to the text a reader would see."""
    return " ".join(html.unescape(_TAG.sub("", markup)).split())


@dataclass
class Heading:
    """One banner: its level, its title, and its blurb if it carries one."""

    level: int
    title: str
    blurb: str = ""


@dataclass
class Entry:
    """One code cell, with the banner that introduces it."""

    index: int
    itbtc: dict
    heading: Heading | None

    @property
    def role(self) -> str:
        return str(self.itbtc.get("role", "?"))


@dataclass
class Phase:
    """One ``##`` phase of the notebook and every code cell beneath it."""

    title: str
    blurb: str
    entries: list[Entry] = field(default_factory=list)


def parse(notebook: Path) -> tuple[list[Phase], Counter]:
    """Walk the notebook once, grouping code cells under their phase banner.

    Args:
        notebook: Path to the ``.ipynb`` file.

    Returns:
        The phases in document order, and a count of cells by cell type.
    """
    doc = json.loads(notebook.read_text(encoding="utf-8"))
    phases: list[Phase] = []
    pending: Heading | None = None
    counts: Counter = Counter()

    for index, cell in enumerate(doc["cells"]):
        counts[cell["cell_type"]] += 1
        if cell["cell_type"] == "markdown":
            source = "".join(cell["source"])
            native = _MARKDOWN_HEADING.search(source)
            found = native or _HEADING.search(source)
            if not found:
                continue
            paragraphs = _PARAGRAPH.findall(source)
            heading = Heading(
                level=len(found.group(1)) if native else int(found.group(1)),
                title=_plain(found.group(2)),
                blurb=_plain(paragraphs[0]) if paragraphs else "",
            )
            if heading.level <= 2:
                # h1 is the notebook title and h2 opens a phase; both start a
                # new bucket, so the h1's bucket collects whatever precedes the
                # first real phase instead of it being dropped on the floor.
                phases.append(Phase(title=heading.title, blurb=heading.blurb))
                pending = None
            else:
                pending = heading
            continue

        itbtc = cell.get("metadata", {}).get("itbtc")
        if itbtc is None:
            continue
        if not phases:
            phases.append(Phase(title="(sebelum banner pertama)", blurb=""))
        phases[-1].entries.append(Entry(index=index, itbtc=itbtc, heading=pending))
        pending = None

    return phases, counts


def _steps(phases: list[Phase]) -> list[tuple[Phase, Entry]]:
    """Every producing cell, paired with the phase it sits in."""
    return [(p, e) for p in phases for e in p.entries if e.role == "step"]


def _modules(phases: list[Phase]) -> dict[str, list[tuple[Phase, Entry]]]:
    """Definition cells grouped by the module they project."""
    out: dict[str, list[tuple[Phase, Entry]]] = {}
    for phase in phases:
        for entry in phase.entries:
            if entry.role == "module":
                key = str(entry.itbtc.get("module", "?"))
                out.setdefault(key, []).append((phase, entry))
    return out


def _paths(entry: Entry, key: str) -> str:
    """Format a ``reads``/``writes`` list as inline code, or an em dash."""
    values = entry.itbtc.get(key) or []
    return ", ".join(f"`{v}`" for v in values) if values else "—"


def render(phases: list[Phase], counts: Counter) -> str:
    """Build the Markdown map. Prose is Indonesian, matching the notebook."""
    steps = _steps(phases)
    modules = _modules(phases)
    # phases[0] is the h1 title banner, not a phase of the pipeline.
    numbered = phases[1:] if phases else []
    lines: list[str] = []
    add = lines.append

    add("# Peta notebook — `notebooks/btc_walkforward_3model.ipynb`")
    add("")
    add(
        "**Digenerate oleh `tools/notebook_map.py`. Jangan disunting tangan** — "
        "jalankan ulang setelah notebook berubah."
    )
    add("")
    add(
        "Notebook ini adalah deliverable utama: setiap modul `src/itransformer_btc/` "
        "menjadi satu sel definisi, salinan kode resmi thuml/iTransformer ditulis oleh "
        "sel `%%writefile`, dan sel langkah menjalankan studi. `src/` adalah proyeksinya "
        "yang diuji (`tests/test_notebook.py`), sehingga peta ini menyebut modul alih-alih "
        "mengulang kodenya."
    )
    add("")
    add(
        f"- Sel: **{sum(counts.values())}** "
        f"({counts.get('code', 0)} kode, {counts.get('markdown', 0)} markdown)"
    )
    add(f"- Fase: **{len(numbered)}**")
    add(f"- Langkah produksi (`role: step`): **{len(steps)}**")
    add(
        f"- Modul yang diproyeksikan: **{len(modules)}** dalam "
        f"{sum(len(v) for v in modules.values())} sel definisi"
    )
    add("")

    add("## Peta artefak — sel mana menghasilkan apa")
    add("")
    add("Tiap sel langkah mendeklarasikan berkas yang dibaca dan ditulisnya.")
    add("")
    add("| Langkah | Sel | Fase | Membaca | Menulis |")
    add("| --- | --: | --- | --- | --- |")
    for phase, entry in steps:
        slug = entry.itbtc.get("step", "?")
        add(
            f"| `{slug}` | {entry.index} | {phase.title} | "
            f"{_paths(entry, 'reads')} | {_paths(entry, 'writes')} |"
        )
    add("")

    add("## Modul yang dibawa notebook")
    add("")
    add("Satu sel definisi per modul, dalam urutan eksekusi.")
    add("")
    add("| Modul | Sel | Fase |")
    add("| --- | --: | --- |")
    for module, pairs in modules.items():
        add(f"| `src/itransformer_btc/{module}` | {', '.join(str(e.index) for _, e in pairs)} | "
            f"{'; '.join(sorted({p.title for p, _ in pairs}))} |")
    add("")

    vendor = [(p, e) for p in phases for e in p.entries if e.role == "vendor"]
    add("## Salinan kode upstream")
    add("")
    add("Ditulis apa adanya oleh `%%writefile`, lalu diverifikasi terhadap sha256 yang dipin.")
    add("")
    add("| Berkas | Sel |")
    add("| --- | --: |")
    for _, entry in vendor:
        add(f"| `vendor/thuml_iTransformer/{entry.itbtc.get('path', '?')}` | {entry.index} |")
    add("")

    add("## Fase")
    add("")
    for phase in numbered:
        add(f"### {phase.title}")
        if phase.blurb:
            add("")
            add(phase.blurb)
        add("")
        if not phase.entries:
            add("_Tidak ada sel kode._")
            add("")
            continue
        for entry in phase.entries:
            label = entry.heading.title if entry.heading else ""
            if entry.role == "step":
                slug = entry.itbtc.get("step", "?")
                detail = f"**Langkah `{slug}`** (sel {entry.index})"
                if label:
                    detail += f" — {label}"
                reads, writes = _paths(entry, "reads"), _paths(entry, "writes")
                if writes != "—":
                    detail += f" · menulis {writes}"
                if reads != "—":
                    detail += f" · membaca {reads}"
            elif entry.role == "module":
                detail = f"Modul `src/itransformer_btc/{entry.itbtc.get('module', '?')}` (sel {entry.index})"
            elif entry.role == "vendor":
                detail = f"Salinan upstream `{entry.itbtc.get('path', '?')}` (sel {entry.index})"
            else:
                detail = f"`{entry.role}` (sel {entry.index})"
                if label:
                    detail += f" — {label}"
            add(f"- {detail}")
        add("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write the map, or verify the committed one is current."""
    parser = argparse.ArgumentParser(description="Render the notebook map.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if docs/NOTEBOOK_MAP.md differs from the notebook",
    )
    args = parser.parse_args(argv)

    phases, counts = parse(NOTEBOOK)
    rendered = render(phases, counts)
    relative = OUTPUT.relative_to(ROOT).as_posix()

    if args.check:
        if not OUTPUT.exists():
            print(f"{relative} is missing", file=sys.stderr)
            return 1
        if OUTPUT.read_text(encoding="utf-8") != rendered:
            print(
                f"{relative} is stale — run `python tools/notebook_map.py`",
                file=sys.stderr,
            )
            return 1
        print(f"{relative} is current")
        return 0

    OUTPUT.write_text(rendered, encoding="utf-8")
    print(
        f"wrote {relative} — {len(phases) - 1} fase, "
        f"{len(_steps(phases))} langkah, {len(_modules(phases))} modul"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

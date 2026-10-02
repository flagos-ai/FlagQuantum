"""Contract between the binding control sequence and the ADR that states it.

`ARCH-012` supersedes in part the control sequence in `AGENTS.md` § Current
Strategic Priority. Two files now describe one governing fact, and nothing else
in the repository notices when they disagree: reverting the `AGENTS.md` section
restores a sequence that forbids the parity programme, while the ADR and its
index row would keep reporting the programme as authorized.

These tests fail closed on that drift. They assert the section that must be
present, the superseded prohibition that must be absent, and the status
agreement between the ADR header and the decisions index.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
AGENTS = ROOT / "AGENTS.md"
ADR = ROOT / "docs/architecture/decisions/ARCH_012_CUDAQ_PARITY_CONTROL_SEQUENCE.md"
INDEX = ROOT / "docs/architecture/decisions/README.md"

SECTION_HEADING = "## Current Strategic Priority"

# The clause the owner authorized replacing. Its presence means the superseded
# sequence is back, whatever the ADR and the index say.
SUPERSEDED_PROHIBITION = "Do not open additional horizontal architecture tracks"

# The ownership constraint the replacement sequence carries.
NVIDIA_CONSTRAINT = "must not depend on NVIDIA-proprietary components"


def _collapsed(text: str) -> str:
    """Return `text` with every whitespace run collapsed to one space.

    The section is hard-wrapped at 88 columns, so a phrase it states may span a
    line break. Assertions read the collapsed form so that a reflow is not a
    contract change.
    """
    return re.sub(r"\s+", " ", text)


def _section() -> str:
    text = AGENTS.read_text(encoding="utf-8")
    start = text.index(SECTION_HEADING)
    end = text.find("\n## ", start + len(SECTION_HEADING))
    assert end != -1, f"{SECTION_HEADING} must be followed by another top-level section"
    return _collapsed(text[start:end])


def test_the_control_sequence_section_exists_and_is_not_the_last_section():
    assert SECTION_HEADING in AGENTS.read_text(encoding="utf-8")


def test_the_control_sequence_names_the_adr_that_states_it():
    """The named ADR must be the one on disk, resolved from the link itself."""
    section = _section()
    links = re.findall(r"\]\(([^)]+)\)", section)
    targets = [(ROOT / link).resolve() for link in links]

    assert ADR in targets, f"the section links {links}, not {ADR.name}"
    assert ADR.exists(), f"{ADR} is the authority the control sequence names"


def test_the_superseded_prohibition_is_gone():
    """The one assertion that distinguishes the replacement from the original."""
    section = _section()

    assert SUPERSEDED_PROHIBITION not in section
    assert NVIDIA_CONSTRAINT in section


def test_the_adr_header_status_is_approved():
    header = ADR.read_text(encoding="utf-8").splitlines()[2]

    assert header == "Status: Approved", header


def test_the_decisions_index_agrees_with_the_adr_header():
    """A status edit in one file and not the other is silent drift."""
    header = ADR.read_text(encoding="utf-8").splitlines()[2].removeprefix("Status: ")
    row = [
        line
        for line in INDEX.read_text(encoding="utf-8").splitlines()
        if "(ARCH_012" in line
    ]

    assert len(row) == 1, f"expected exactly one ARCH-012 index row, found {len(row)}"
    cells = [cell.strip() for cell in row[0].strip("|").split("|")]

    assert cells[2] == header, f"index says {cells[2]!r}, ADR header says {header!r}"
    assert header != "Proposed", "a proposal cannot govern the binding control sequence"


def test_the_replacement_sequence_states_its_five_clauses():
    section = _section()

    # Clauses 1, 4, and 5 carry over from the original sequence, so only their
    # opening is pinned; a reworded tail is not drift. Clauses 2 and 3 are the
    # two **admission gates** this ADR introduced -- the whole reason a second
    # track is now permitted -- so their substance is pinned in full. A gate
    # whose test can be satisfied by its own first line is not a gate.
    for opening in (
        "every round extends a proven vertical path",
        "legacy and transitional code is deleted or explicitly frozen in every round",
        "no parity claim is made without a generated parity-matrix entry",
    ):
        assert opening in section, opening

    for gate in (
        "a new horizontal abstraction is admitted only when it **replaces** an "
        "existing implementation behind an existing boundary, and admission "
        "requires a replacement test in which at least one implementation is "
        "swapped without modifying its consumers, per engineering decision "
        "principle 10",
        "two tracks may develop concurrently only when they do not share an "
        "unproven contract; where they would, the contract lands first on the "
        "integration branch together with a contract fake and a conformance "
        "test, and each track then synchronizes that baseline",
    ):
        assert gate in section, gate

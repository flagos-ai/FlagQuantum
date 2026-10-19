"""Guard the derived numbers in the CUDA-Q parity strategy document.

`docs/roadmap/CUDAQ_PARITY_STRATEGY.md` states counts that are read off
`contracts/cudaq-parity-matrix.toml`. A strategy document that quotes stale counts
is worse than one that quotes none, because the numbers are what a reader uses to
decide where to spend effort. These tests recompute every count from the contract
and fail when the prose drifts.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

from tools.parity_matrix import CONTRACT_PATH, load_inputs

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
STRATEGY = ROOT / "docs/roadmap/CUDAQ_PARITY_STRATEGY.md"


def _flowed() -> str:
    """The document with markup wrapping removed, so prose is matchable."""

    return re.sub(r"\s+", " ", STRATEGY.read_text(encoding="utf-8"))


@pytest.fixture()
def rows() -> list[tuple[str, dict]]:
    contract = load_inputs()[0]
    return [
        (domain["id"], capability)
        for domain in contract["domains"]
        for capability in domain["capabilities"]
    ]


def test_strategy_document_is_an_intent_document() -> None:
    text = STRATEGY.read_text(encoding="utf-8")
    assert (
        "future intent" in text[:1000].lower()
    ), "the strategy document must identify itself as future intent near the top"
    assert not re.search(
        r"\bfq\.[A-Za-z_]\w*", text
    ), "an intent document must not name public API symbols as if they existed"


def test_status_counts_match_the_contract(rows: list[tuple[str, dict]]) -> None:
    counts = Counter(capability["status"] for _, capability in rows)
    text = STRATEGY.read_text(encoding="utf-8")
    for status, count in counts.items():
        assert (
            f"| `{status}` | {count} |" in text
        ), f"the strategy document does not state {status} = {count}"
    assert sum(counts.values()) == len(rows)


def test_the_breadth_risk_repeats_the_unsupported_count(
    rows: list[tuple[str, dict]],
) -> None:
    """The risk section quotes the same number, so it drifts independently."""

    unsupported = sum(
        1 for _, capability in rows if capability["status"] == "unsupported"
    )
    assert f"{unsupported} `unsupported` rows invite a sprint" in _flowed()


def test_dependency_class_counts_match_the_contract(
    rows: list[tuple[str, dict]],
) -> None:
    counts = Counter(capability["dependency_class"] for _, capability in rows)
    text = STRATEGY.read_text(encoding="utf-8")
    for dependency_class, count in counts.items():
        assert (
            f"| `{dependency_class}` | {count} |" in text
        ), f"the strategy document does not state {dependency_class} = {count}"


def test_the_documented_replacement_obligation_is_the_whole_of_it(
    rows: list[tuple[str, dict]],
) -> None:
    """The A-class table is the programme's cost estimate, so it must be complete."""

    proprietary = {
        capability["id"]
        for _, capability in rows
        if capability["dependency_class"] == "A_nvidia_proprietary"
    }
    text = STRATEGY.read_text(encoding="utf-8")
    missing = sorted(name for name in proprietary if f"`{name}`" not in text)
    assert not missing, f"the strategy document omits proprietary rows: {missing}"
    assert len(proprietary) == 9


def test_the_contraction_path_rows_carry_a_verdict() -> None:
    """A surveyed project's outcome must be readable, not merely listed.

    The tensor-network contraction path is the one place § 5 has produced a
    decision, so the two rows that decide it must state which way they decided.
    An unmarked row reads as still-open, which is how a stale survey becomes a
    plan nobody owns.
    """

    text = STRATEGY.read_text(encoding="utf-8")
    section = text.split("## 5. What may be borrowed", 1)[1].split("## 6.", 1)[0]
    project_rows = [line for line in section.splitlines() if line.startswith("| ")][2:]
    notes = {}
    for line in project_rows:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        notes[cells[0]] = cells[3]

    assert "**Adopted**" in notes["cotengra"]
    assert "**Surveyed, not adopted.**" in notes["opt_einsum"]


def test_the_borrowable_count_matches_the_contract(
    rows: list[tuple[str, dict]],
) -> None:
    vendor_neutral = Counter(
        capability["status"]
        for _, capability in rows
        if capability["dependency_class"] == "B_open_neutral"
    )
    text = _flowed()
    unsupported = vendor_neutral["unsupported"]
    total = sum(vendor_neutral.values())
    assert (
        f"{unsupported} of the {total} `B_open_neutral` rows are `unsupported`" in text
    )


def test_every_borrowed_project_is_named_with_a_licence(
    rows: list[tuple[str, dict]],
) -> None:
    """§ 5 must not list a project without saying where its licence check lives."""

    text = STRATEGY.read_text(encoding="utf-8")
    section = text.split("## 5. What may be borrowed", 1)[1].split("## 6.", 1)[0]
    project_rows = [line for line in section.splitlines() if line.startswith("| ")][2:]
    assert project_rows, "the borrow table is missing"
    for line in project_rows:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        assert len(cells) >= 3, line
        assert cells[0] and cells[1] and cells[2], f"incomplete borrow row: {line}"
    assert "must be re-verified inside the adoption review" in _flowed()


def test_the_contract_path_in_the_document_exists() -> None:
    text = STRATEGY.read_text(encoding="utf-8")
    assert "../../contracts/cudaq-parity-matrix.toml" in text
    assert CONTRACT_PATH.is_file()

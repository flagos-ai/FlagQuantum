"""Conformance of the qubit-vocabulary ledger with the code it claims to record.

`tests/unit/test_census_wire_vocabulary.py` owns the scanner: what the census
counts and what it excludes. This file owns the gate, and asks the question the
whole program rests on -- is the ledger still an accurate record of the package,
and does the gate actually notice when it stops being one?

The second half of that sentence is the reason this file exists. A gate that
returns an empty tuple of errors for every input passes every check while
enforcing nothing, so every rule below is exercised twice: once against the real
contract, and once against a mutation of it that must be reported, by the rule
that owns it, naming the thing that was broken.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 and older
    import tomli as tomllib

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = _ROOT / "tools"
_CONTRACT_PATH = _ROOT / "contracts" / "qubit-vocabulary-contract.toml"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_CENSUS = _load("census_wire_vocabulary")
_GATE = _load("check_qubit_vocabulary")


def _contract() -> dict[str, Any]:
    return tomllib.loads(_CONTRACT_PATH.read_text(encoding="utf-8"))


def _errors(contract: dict[str, Any]) -> tuple[str, ...]:
    return _GATE.contract_errors(contract)


def _package(tmp_path: Path, files: dict[str, str]) -> Path:
    for relative, source in files.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
    return tmp_path / "flagquantum"


def _one(error_fragments: tuple[str, ...], needle: str) -> None:
    """Assert exactly one reported error mentions `needle`."""

    matches = [message for message in error_fragments if needle in message]
    assert len(matches) == 1, (needle, error_fragments)


def _some(error_fragments: tuple[str, ...], needle: str) -> None:
    """Assert at least one reported error mentions `needle`."""

    assert any(needle in message for message in error_fragments), (
        needle,
        error_fragments,
    )


# --------------------------------------------------------------------------- basics


def test_the_checked_in_contract_passes() -> None:
    assert _errors(_contract()) == ()


def test_the_gate_reports_progress_per_slice() -> None:
    rows = _GATE.slice_progress(_contract())
    assert [row[0] for row in rows] == [f"WQ-{index}" for index in range(2, 9)]
    assert all(retired == 0 for _, retired, _ in rows)
    assert sum(remaining for _, _, remaining in rows) == 341


def test_the_gate_is_wired_into_ci_and_pre_push() -> None:
    for path in (_GATE.CI_WORKFLOW, _GATE.PRE_PUSH):
        assert _GATE.SELF_PATH in path.read_text(encoding="utf-8")


def test_the_required_check_roster_is_unchanged_by_a_step() -> None:
    text = _GATE.CI_WORKFLOW.read_text(encoding="utf-8")
    assert "Verify the qubit-vocabulary ledger" in text
    assert "qubit-vocabulary" not in text.split("jobs:", 1)[0]


# ------------------------------------------------------------------- header rules


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "flagquantum_qubit_vocabulary_contract_v2"),
        ("maturity", "experimental"),
        ("gate_tool", "tools/check_something_else.py"),
        ("ir_version", "1.1"),
        ("ir_version_unchanged", False),
    ],
)
def test_a_changed_header_constant_is_reported(field: str, value: Any) -> None:
    contract = _contract()
    contract[field] = value
    _one(_errors(contract), field)


def test_a_changed_naming_root_is_reported() -> None:
    contract = _contract()
    contract["naming"]["root_wires"] = "wires"
    _one(_errors(contract), "naming roots")


@pytest.mark.parametrize(
    "field",
    [
        "serialized_keys",
        "environment_variables",
        "serialized_reason",
        "environment_reason",
    ],
)
def test_a_missing_exclusion_is_reported(field: str) -> None:
    contract = _contract()
    contract["exclusions"][field] = [] if field.endswith("s") else ""
    _one(_errors(contract), field)


@pytest.mark.parametrize(
    "field",
    ["public_attribute_names", "message_strings"],
)
def test_an_unmeasured_other_surface_is_reported(field: str) -> None:
    contract = _contract()
    contract["other_surfaces"][field] = 0
    _one(_errors(contract), field)


@pytest.mark.parametrize(
    "field",
    [
        "public_attribute_owner",
        "public_attribute_condition",
        "message_strings_owner",
        "message_strings_condition",
    ],
)
def test_an_unowned_other_surface_is_reported(field: str) -> None:
    contract = _contract()
    contract["other_surfaces"][field] = ""
    _one(_errors(contract), field)


def test_a_missing_boundary_definition_is_reported() -> None:
    contract = _contract()
    contract["boundary"]["definition"] = ""
    _one(_errors(contract), "definition")


def test_an_unexcluded_boundary_is_reported() -> None:
    contract = _contract()
    contract["boundary"]["private_excluded"] = False
    _one(_errors(contract), "must exclude private code")


# ------------------------------------------------------------------ ledger rules


def test_a_duplicated_ledger_entry_is_reported() -> None:
    contract = _contract()
    contract["ledger"]["canonical"].append(contract["ledger"]["canonical"][0])
    _some(_errors(contract), "repeats")


def test_a_ledger_that_disagrees_with_the_measured_total_is_reported() -> None:
    contract = _contract()
    contract["boundary"]["measured_canonical"] = 328
    _one(_errors(contract), "ledger length")


def test_an_alias_ledger_that_disagrees_with_the_measured_total_is_reported() -> None:
    contract = _contract()
    contract["aliases"]["declared"].pop()
    _one(_errors(contract), "alias ledger length")


def test_an_alias_without_a_removal_version_is_reported() -> None:
    contract = _contract()
    contract["aliases"]["removal_version"] = ""
    _one(_errors(contract), "removal version")


def test_a_retirement_entry_outside_the_baseline_is_reported() -> None:
    contract = _contract()
    contract["retirement"]["sites"] = ["flagquantum/nowhere.py::absent::wires"]
    _one(_errors(contract), "outside the baseline")


def test_a_baseline_line_deleted_without_a_retirement_entry_is_reported() -> None:
    """The rule that makes the ledger shrink-only: deleting a line is a defect.

    The deleted site is still in the code, so it is now a live `wire` parameter
    that no ledger entry covers. That is the right diagnosis: the line did not
    retire anything, it only stopped claiming the site.
    """

    contract = _contract()
    removed = contract["ledger"]["canonical"].pop()
    errors = _errors(contract)
    _one(errors, "ledger length")
    _some(errors, "outside the ledger")
    _some(errors, removed)


def test_a_fabricated_retirement_entry_is_reported() -> None:
    """A slice may not claim it renamed something that is still spelled `wire`."""

    contract = _contract()
    claimed = "flagquantum/circuit.py::Circuit.gate::wires"
    contract["retirement"]["sites"] = [claimed]
    errors = _errors(contract)
    _some(errors, "outside the ledger")
    _some(errors, claimed)


def test_a_rename_without_a_retirement_entry_is_reported(tmp_path: Path) -> None:
    """Renaming a site is not enough; the ledger has to record it as retired.

    The report truncates its list of sites, so the fixture renames the
    alphabetically first baseline identifier -- the one the truncation is
    guaranteed to name -- and asserts on it by name rather than trusting the
    count.
    """

    first = min(_contract()["ledger"]["canonical"])
    package = _package(
        tmp_path,
        {
            "flagquantum/algorithms/amplitude_estimation.py": (
                "def amplitude_estimation_circuit(n_counting_qubits):\n"
                "    return n_counting_qubits\n"
            )
        },
    )
    errors = _GATE.contract_errors(_contract(), package_root=package)
    _some(errors, "neither live nor retired")
    _some(errors, first)


def test_a_new_wire_parameter_outside_the_ledger_is_reported(tmp_path: Path) -> None:
    """The acceptance clause: a fixture that introduces a `wire` parameter fails."""

    package = _package(
        tmp_path,
        {
            "flagquantum/circuit.py": "def gate(self, name, wires):\n    return name, wires\n",
            "flagquantum/rogue.py": "def surprise(n_wires):\n    return n_wires\n",
        },
    )
    errors = _GATE.contract_errors(_contract(), package_root=package)
    _one(errors, "outside the ledger")
    assert "rogue.py::surprise::n_wires" in " ".join(errors)


def test_an_alias_forwarding_to_the_wrong_name_is_reported() -> None:
    """The ledger records the qubit name each alias forwards to, and checks it."""

    contract = _contract()
    row = next(
        entry
        for entry in contract["aliases"]["declared"]
        if entry["site"].endswith("::Z::wire")
    )
    row["replacement"] = "qubits"
    _some(_errors(contract), "must forward to")


def test_a_vacuous_boundary_is_reported(tmp_path: Path) -> None:
    package = _package(
        tmp_path,
        {"flagquantum/circuit.py": "def gate(self, wires):\n    return wires\n"},
    )
    errors = _GATE.contract_errors(_contract(), package_root=package)
    _one(errors, "boundary is vacuous")


def test_a_changed_private_bucket_is_reported(tmp_path: Path) -> None:
    package = _package(
        tmp_path,
        {
            "flagquantum/circuit.py": "def gate(self, wires):\n    return wires\n",
            "flagquantum/_hidden.py": "def helper(wire):\n    return wire\n",
        },
    )
    errors = _GATE.contract_errors(_contract(), package_root=package)
    _one(errors, "private bucket changed size")


# ------------------------------------------------------------------- slice rules


def test_a_duplicated_slice_path_is_reported() -> None:
    contract = _contract()
    contract["slices"].append(copy.deepcopy(contract["slices"][0]))
    errors = _errors(contract)
    assert any(
        "owner_team" in message or "owned by" in message or "repeat" in message
        for message in errors
    ), errors


def test_a_slice_count_that_disagrees_with_the_ledger_is_reported() -> None:
    contract = _contract()
    contract["slices"][0]["canonical_count"] += 1
    _one(_errors(contract), "canonical_count")


def test_a_slice_naming_a_missing_file_is_reported() -> None:
    contract = _contract()
    contract["slices"][0]["files"].append("flagquantum/not_a_real_module.py")
    errors = _errors(contract)
    assert any("names missing" in message for message in errors), errors


def test_an_unowned_baseline_path_is_reported() -> None:
    contract = _contract()
    dropped = contract["slices"][0]["files"].pop(0)
    errors = _errors(contract)
    assert any("unowned" in message for message in errors), (dropped, errors)


def test_a_stray_slice_path_is_reported() -> None:
    contract = _contract()
    empty = next(
        entry
        for entry in contract["slices"]
        if all("flagquantum" in str(path) for path in entry["files"])
    )
    empty["files"].append("flagquantum/__init__.py")
    errors = _errors(contract)
    assert any("no baseline site" in message for message in errors), errors


def test_a_slice_without_an_owner_is_reported() -> None:
    contract = _contract()
    contract["slices"][0]["owner_team"] = ""
    _one(_errors(contract), "owner_team")


def test_a_missing_verification_path_is_reported() -> None:
    contract = _contract()
    contract["verification"]["contract"] = "tests/unit/absent.py"
    _one(_errors(contract), "verification path")


def test_a_missing_reference_document_is_reported() -> None:
    contract = _contract()
    contract["proposal"] = "docs/development/absent.md"
    _one(_errors(contract), "reference")


# ----------------------------------------------------------- forbidden vocabulary


def test_a_forbidden_alternative_spelling_is_reported(tmp_path: Path) -> None:
    package = _package(
        tmp_path,
        {
            "flagquantum/circuit.py": (
                "def gate(self, qubit_indices):\n    return qubit_indices\n"
            )
        },
    )
    errors = _GATE.contract_errors(_contract(), package_root=package)
    _one(errors, "forbids the alternative spelling")


def test_the_forbidden_list_is_not_empty() -> None:
    assert _contract()["naming"]["forbidden"]


# --------------------------------------------------------------------- live facts


def test_the_ledger_is_the_scanner_output_at_the_baseline() -> None:
    contract = _contract()
    scanned = _CENSUS.census(_ROOT / "flagquantum")
    assert set(contract["ledger"]["canonical"]) == {
        site.identifier for site in scanned.canonical
    }


def test_the_existing_aliases_are_the_ones_the_scanner_finds() -> None:
    contract = _contract()
    scanned = _CENSUS.census(_ROOT / "flagquantum")
    declared = {
        str(row["site"]): str(row["replacement"])
        for row in contract["aliases"]["declared"]
    }
    assert declared == {site.identifier: site.replacement for site in scanned.aliases}

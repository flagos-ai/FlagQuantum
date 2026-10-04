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
import re
import sys
from functools import lru_cache
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


def _unretire_a_parameter(contract: dict[str, Any]) -> str:
    """Drop one retirement row and return the site it had retired.

    Every parameter is retired once the migration finishes, and that is the
    point of the program rather than a reason to delete the rule that proves a
    retirement is recorded. The fixture therefore un-retires the alphabetically
    first site in place of waiting for a slice to leave work behind, which is
    also why the two tests below mutate a copy instead of reading the live tree.
    """

    identifier = min(str(entry) for entry in contract["retirement"]["sites"])
    contract["retirement"]["sites"] = [
        entry for entry in contract["retirement"]["sites"] if str(entry) != identifier
    ]
    return identifier


def _unretire_a_definition(contract: dict[str, Any]) -> str:
    """Drop one definition retirement row and return the site it had retired.

    Same reason as `_unretire_a_parameter`: the migration ends with every name
    retired, and the rules that prove a retirement is recorded still have to be
    proved afterwards. The fixture reads the exemplar out of the contract instead
    of hard-coding one, because `WQ-7` and `WQ-8` each moved the literal the test
    used to name.
    """

    identifier = min(str(row["site"]) for row in contract["definition_ledger"]["sites"])
    contract["definition_retirement"]["sites"] = [
        entry
        for entry in contract["definition_retirement"]["sites"]
        if str(entry) != identifier
    ]
    return identifier


def _alias_parameters() -> list[str]:
    """The parameter spellings the contract keeps as forwarders, alphabetically."""

    contract = _contract()
    retired = {str(entry) for entry in contract["retirement"]["sites"]}
    alias = sorted(str(row["site"]) for row in contract["aliases"]["declared"])
    assert alias and not retired & set(alias)
    return alias


def _wire_named_package(tmp_path: Path, identifier: str) -> Path:
    """A one-file package in which `identifier` still carries its wire name.

    The inverse of `_renamed_package`, for the same reason: the file is copied
    out of the real tree and the spelling is put back, so a fixture that needs a
    live `wire` name does not depend on some slice having left one behind. The
    migration has none left, and a hard-coded stub would stop matching the module
    the ledger names the moment that module changes.
    """

    relative = identifier.split("::", 1)[0]
    name = identifier.rsplit("::", 1)[1]
    source = (_ROOT / relative).read_text(encoding="utf-8")
    replacement = _CENSUS.replacement_name(name)
    return _package(
        tmp_path,
        {relative: re.sub(rf"\b{re.escape(replacement)}\b", name, source)},
    )


def _unretired_attributes() -> list[str]:
    """The attributes the contract has not retired yet, alphabetically."""

    contract = _contract()
    retired = {str(row["site"]) for row in contract["attribute_retirement"]["sites"]}
    return sorted(set(map(str, contract["attribute_ledger"]["canonical"])) - retired)


def _renamed_package(tmp_path: Path, identifier: str) -> Path:
    """A one-file package in which `identifier` already carries its qubit name.

    The exemplar is read out of the contract and the file is copied out of the
    real tree rather than stubbed, because a hard-coded exemplar stops being a
    live site the moment the slice that owns it lands -- and the test then fails
    for the very reason it exists to detect.
    """

    relative = identifier.split("::", 1)[0]
    name = identifier.rsplit("::", 1)[1]
    source = (_ROOT / relative).read_text(encoding="utf-8")
    replacement = _CENSUS.replacement_name(name)
    return _package(
        tmp_path,
        {relative: re.sub(rf"\b{re.escape(name)}\b", replacement, source)},
    )


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


def test_the_gate_reports_docstring_progress() -> None:
    """A docstring is what `help()` prints, and no other ledger reads it.

    The four keyword ledgers read an AST signature or a documented keyword. A
    docstring is neither, so without this surface a package whose every parameter
    was renamed could still explain those parameters in the old vocabulary and the
    gate would report a finished migration.
    """

    contract = _contract()
    kept = sum(len(row.get("identifier", ())) for row in contract["docstring"]["keep"])
    scanned = _CENSUS.docstring_census(_ROOT / "flagquantum")
    assert scanned, "the docstring census found nothing to reconcile"
    assert kept == len(scanned) == int(contract["docstring"]["measured_tokens"])


def test_an_unrecorded_docstring_token_is_reported() -> None:
    """The default: a wire-named token in a docstring and no row for it.

    Dropping a row is the only way to reach this branch from the real contract --
    the gate reads the live package, so a fixture package cannot be substituted
    without leaving every other ledger reporting at once. The branch is what a
    docstring that was never migrated meets, and the census test above proves the
    reading it is compared against.
    """

    contract = _contract()
    dropped = contract["docstring"]["keep"].pop(0)
    container = dropped["identifier"][0].rsplit("::", 1)[0]
    errors = _errors(contract)
    _some(errors, "docstring publishes a wire-named token")
    assert any(container in message for message in errors), (container, errors)


def test_a_docstring_exemption_permitting_more_than_the_tree_shows_is_reported() -> (
    None
):
    """The row pins the whole multiset, so it cannot be widened to absorb a site."""

    contract = _contract()
    row = contract["docstring"]["keep"][0]
    container = row["identifier"][0].rsplit("::", 1)[0]
    row["identifier"] = [*row["identifier"], f"{container}::phantom_wire"]
    _some(_errors(contract), "is exempt")


def test_a_stale_docstring_exemption_is_reported() -> None:
    """A row whose container no longer shows the token is dead weight, not insurance."""

    contract = _contract()
    row = contract["docstring"]["keep"][0]
    row["identifier"] = [
        row["identifier"][0].rsplit("::", 1)[0] + "::a_token_that_is_not_there"
    ]
    _some(_errors(contract), "is exempt")
    contract = _contract()
    contract["docstring"]["keep"][0]["identifier"] = [
        "flagquantum/_api.py::a_definition_that_does_not_exist::n_wires"
    ]
    _some(_errors(contract), "is stale")


def test_a_docstring_exemption_without_a_reason_is_reported() -> None:
    contract = _contract()
    contract["docstring"]["keep"][0]["reason"] = ""
    _some(_errors(contract), "states no reason")


def test_a_docstring_exemption_without_an_identifier_is_reported() -> None:
    contract = _contract()
    contract["docstring"]["keep"][0]["identifier"] = []
    _some(_errors(contract), "missing an identifier")


def test_a_docstring_identifier_that_is_not_a_container_token_key_is_reported() -> None:
    """The key has to name a container, so a bare token cannot be an exemption."""

    contract = _contract()
    contract["docstring"]["keep"][0]["identifier"] = ["just_a_token"]
    _some(_errors(contract), "container::token")


def test_a_docstring_measurement_that_disagrees_with_the_scan_is_reported() -> None:
    contract = _contract()
    contract["docstring"]["measured_tokens"] = (
        int(contract["docstring"]["measured_tokens"]) + 1
    )
    _some(_errors(contract), "measured_tokens must equal the live scan")


@pytest.mark.parametrize("field", ["owner", "condition"])
def test_an_unowned_docstring_surface_is_reported(field: str) -> None:
    contract = _contract()
    contract["docstring"][field] = ""
    _some(_errors(contract), f"docstring {field} is unowned")


def test_the_docstring_surface_is_not_the_documentation_surface() -> None:
    """The two are separate ledgers, because the two reach a user differently.

    `[documentation]` reconciles a keyword argument a reader copies into a script;
    a documented keyword is a signature under another spelling, which is why an
    exemption there names a *file* and the whole multiset of spellings it may show.
    `[docstring]` reconciles prose attached to code that already exists, which is
    why an exemption here names a *container* and one token. Merging them would let
    a reworded docstring pay for a stale example.
    """

    contract = _contract()
    assert "docstring" in contract and "documentation" in contract
    assert "measured_tokens" in contract["docstring"]
    assert "measured_keywords" in contract["documentation"]
    assert "keep" in contract["docstring"]
    assert "exempt" in contract["documentation"]


def test_the_gate_reports_progress_per_slice() -> None:
    """Retirement moves a site from `remaining` to `retired`, never off the books.

    The ledger is the frozen baseline, so the two columns always add back up to
    the measured surface. A slice that could shrink the total would look finished
    while it still had sites to rename.
    """

    rows = _GATE.slice_progress(_contract())
    assert [row[0] for row in rows] == [f"WQ-{index}" for index in range(2, 9)]
    assert {row[0]: (row[1], row[2]) for row in rows}["WQ-2"] == (16, 0)
    assert sum(retired + remaining for _, retired, remaining in rows) == 341


def test_the_gate_reports_attribute_progress_per_slice() -> None:
    """The parameter count alone would call the migration finished too early.

    `MeasurementResult.wires` is on the user's screen and is not a parameter, so
    the gate reports a second total; the migration is finished only when both
    reach zero.
    """

    rows = _GATE.slice_progress(_contract(), "attribute")
    assert [row[0] for row in rows] == [f"WQ-{index}" for index in range(2, 9)]
    assert {row[0]: (row[1], row[2]) for row in rows}["WQ-2"] == (26, 2)
    assert sum(retired + remaining for _, retired, remaining in rows) == 122


def test_the_gate_reports_definition_progress_per_slice() -> None:
    """A renamed parameter inside `infer_n_qubits_from_dense_state` is not done.

    The name of a module-level public function or class is the third door: it is
    neither a parameter nor a class attribute, so without this surface the gate
    would report a finished migration while ten public names still say wire.
    """

    rows = _GATE.slice_progress(_contract(), "definition")
    assert [row[0] for row in rows] == [f"WQ-{index}" for index in range(2, 9)]
    assert sum(retired for _, retired, _ in rows) == 10
    assert sum(retired + remaining for _, retired, remaining in rows) == 10


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
    """The rule that makes the ledger a record of what was owed, not a live count.

    Deleting a line retires nothing: the site is still spelled `wire` in the code
    and now no baseline entry covers it. Both halves of the baseline are checked,
    because the migration ended with them in different states -- every canonical
    line is retired, so deleting one is caught by the line that retired it, while
    every alias line is still live code, so deleting one leaves a spelling on disk
    that the baseline no longer covers.
    """

    contract = _contract()
    removed = min(str(entry) for entry in contract["retirement"]["sites"])
    contract["ledger"]["canonical"].remove(removed)
    errors = _errors(contract)
    _one(errors, "ledger length")
    _some(errors, "retirement names sites outside the baseline")
    _some(errors, removed)

    contract = _contract()
    aliased = min(str(row["site"]) for row in contract["aliases"]["declared"])
    contract["aliases"]["declared"] = [
        row for row in contract["aliases"]["declared"] if str(row["site"]) != aliased
    ]
    errors = _errors(contract)
    _some(errors, "outside the ledger")
    _some(errors, aliased)


def test_a_fabricated_retirement_entry_is_reported() -> None:
    """A slice may not claim it renamed something that is still spelled `wire`.

    The fabricated entry is appended, not substituted: replacing the table would
    un-retire every landed site as well, and the fixture must fail for the one
    reason it names.
    """

    contract = _contract()
    # A live alias is the one kind of site that is deliberately still spelled
    # `wire`, so claiming it as retired is exactly the fabrication to report.
    claimed = _alias_parameters()[0]
    assert claimed not in contract["retirement"]["sites"]
    contract["retirement"]["sites"].append(claimed)
    errors = _errors(contract)
    _some(errors, "outside the ledger")
    _some(errors, claimed)


def test_a_rename_without_a_retirement_entry_is_reported(tmp_path: Path) -> None:
    """Renaming a site is not enough; the ledger has to record it as retired.

    The report truncates its list of sites, so the fixture un-retires the
    alphabetically first identifier -- the one the truncation is guaranteed to
    name -- and asserts on it by name rather than trusting the count. `WQ-5`
    landed the site this fixture used to hard-code, and `WQ-8` retired the last
    one, so it no longer reads the exemplar out of the live contract.
    """

    contract = _contract()
    first = _unretire_a_parameter(contract)
    package = _renamed_package(tmp_path, first)
    errors = _GATE.contract_errors(contract, package_root=package)
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


def test_a_forbidden_alternative_as_an_attribute_name_is_reported(
    tmp_path: Path,
) -> None:
    """A parameter scan cannot see a class field, so the rule is checked twice."""

    package = _package(
        tmp_path,
        {
            "flagquantum/algorithms/amplitude_estimation.py": (
                "def amplitude_estimation_circuit(n_counting_wires):\n"
                "    return n_counting_wires\n"
            ),
            "flagquantum/runtime/result.py": (
                "class MeasurementResult:\n    qubit_indices: tuple = ()\n"
            ),
        },
    )
    errors = _GATE.contract_errors(_contract(), package_root=package)
    _one(errors, "forbids the alternative spelling")


def test_the_forbidden_list_is_not_empty() -> None:
    assert _contract()["naming"]["forbidden"]


# ------------------------------------------------------------- attribute rules


def _attribute_rows(contract: dict[str, Any]) -> list[dict[str, Any]]:
    """The live exclusion list, so removing a row really removes it."""

    return contract["attribute_exclusions"]["sites"]


@lru_cache(maxsize=1)
def _declarations() -> dict[str, str]:
    """Each attribute's declaration kind, measured once for the whole file.

    The field/member split is recorded per slice and checked against the live
    scan, so a mutation that moves a name between lists has to keep those counts
    truthful. The kind does not change when a name moves, so it is read from the
    package once rather than re-derived per test.
    """

    scanned = _CENSUS.attribute_census(_ROOT / "flagquantum")
    return {
        site.identifier: site.declaration
        for site in (*scanned.ledgered, *scanned.excluded)
    }


def _recount_attributes(contract: dict[str, Any]) -> None:
    """Re-derive the per-slice attribute counts after a mutation."""

    names = [str(entry) for entry in contract["attribute_ledger"]["canonical"]]
    sites = [str(row["site"]) for row in _attribute_rows(contract)]
    declarations = _declarations()
    for entry in contract["slices"]:
        files = set(entry["files"])
        owned = [name for name in names if name.split("::", 1)[0] in files]
        entry["attribute_count"] = len(owned)
        for kind in _CENSUS.DECLARATION_KINDS:
            entry[f"attribute_{kind}_count"] = sum(
                1 for name in owned if declarations.get(name) == kind
            )
        entry["persisted_attribute_count"] = sum(
            1 for site in sites if site.split("::", 1)[0] in files
        )


def test_an_attribute_ledger_that_disagrees_with_the_measured_total_is_reported() -> (
    None
):
    contract = _contract()
    contract["attribute_ledger"]["measured_canonical"] = 80
    _one(_errors(contract), "attribute_ledger.measured_canonical")


def test_a_deleted_attribute_exclusion_is_reported() -> None:
    """The rule that pins the exact split, not only its size.

    Removing an exclusion also changes the size, so the assertion names the rule
    that owns the decision: the name now claims to be renameable while the code
    still writes it into a payload.
    """

    contract = _contract()
    removed = _attribute_rows(contract).pop(0)["site"]
    _recount_attributes(contract)
    errors = _errors(contract)
    _some(errors, "not excluded by name")
    _some(errors, removed)


def test_a_promoted_exclusion_is_reported() -> None:
    """Moving a payload key into the renameable list is the failure that matters.

    This is the shape of the real defect: a slice renames a field, the payload
    key changes, and every consumer of the serialized form breaks silently. The
    gate has to call it out at the moment the ledger stops excluding it.
    """

    contract = _contract()
    row = _attribute_rows(contract).pop(0)
    contract["attribute_ledger"]["canonical"].append(row["site"])
    contract["attribute_ledger"]["measured_canonical"] += 1
    contract["attribute_exclusions"]["measured_persisted"] -= 1
    contract["other_surfaces"]["public_attribute_ledgered"] += 1
    contract["other_surfaces"]["public_attribute_persisted"] -= 1
    _recount_attributes(contract)
    errors = _errors(contract)
    _some(errors, "neither live nor retired")
    _some(errors, row["site"])


def test_an_attribute_split_that_loses_a_name_is_reported() -> None:
    contract = _contract()
    contract["other_surfaces"]["public_attribute_persisted"] = 20
    _one(_errors(contract), "public_attribute_persisted")


def test_an_exclusion_for_a_class_that_stopped_serializing_is_reported(
    tmp_path: Path,
) -> None:
    """Exclusions are pinned from both sides, so a rule change cannot hide.

    The fixture keeps `Instruction` wire-named but removes the `asdict` call
    that made it a payload, so its exclusion is now stale and its field is a
    rename again. A gate that only checked "is the name still in the code" would
    see nothing wrong here.
    """

    package = _package(
        tmp_path,
        {
            "flagquantum/algorithms/amplitude_estimation.py": (
                "def amplitude_estimation_circuit(n_counting_wires):\n"
                "    return n_counting_wires\n"
            ),
            "flagquantum/core/ir.py": (
                "from dataclasses import dataclass\n"
                "\n"
                "\n"
                "@dataclass\n"
                "class Instruction:\n"
                "    wires: tuple = ()\n"
                "\n"
                "\n"
                "@dataclass\n"
                "class CircuitIR:\n"
                "    n_wires: int = 0\n"
                "    instructions: tuple[Instruction, ...] = ()\n"
            ),
        },
    )
    errors = _GATE.contract_errors(_contract(), package_root=package)
    _some(errors, "core/ir.py::CircuitIR::n_wires is stale")
    _some(errors, "outside the ledger")


def test_an_exclusion_for_a_class_that_still_serializes_is_not_reported(
    tmp_path: Path,
) -> None:
    """The other half of the same rule, so the previous test proves something.

    With `asdict(self)` present, the same two names are live payload keys, their
    exclusions match the scan, and the gate says nothing about them -- which is
    what makes the stale report in the previous test a real finding rather than a
    property of every fixture.
    """

    package = _package(
        tmp_path,
        {
            "flagquantum/algorithms/amplitude_estimation.py": (
                "def amplitude_estimation_circuit(n_counting_wires):\n"
                "    return n_counting_wires\n"
            ),
            "flagquantum/core/ir.py": (
                "from dataclasses import asdict, dataclass\n"
                "\n"
                "\n"
                "@dataclass\n"
                "class Instruction:\n"
                "    wires: tuple = ()\n"
                "\n"
                "\n"
                "@dataclass\n"
                "class CircuitIR:\n"
                "    n_wires: int = 0\n"
                "    instructions: tuple[Instruction, ...] = ()\n"
                "\n"
                "    def to_dict(self):\n"
                "        return asdict(self)\n"
            ),
        },
    )
    errors = _GATE.contract_errors(_contract(), package_root=package)
    assert not [
        message
        for message in errors
        if "core/ir.py::CircuitIR::n_wires is stale" in message
        or "core/ir.py::Instruction::wires is stale" in message
        or "core/ir.py::Instruction::wires" in message
    ], errors


def test_an_exclusion_whose_witness_changed_is_reported() -> None:
    """The reason has to stay true, not merely present."""

    contract = _contract()
    row = next(
        entry
        for entry in _attribute_rows(contract)
        if entry["site"].endswith("::Instruction::wires")
    )
    row["witness"] = "SomeOtherContract"
    _some(_errors(contract), "no longer matches the scan")


def test_an_attribute_rename_without_a_retirement_entry_is_reported(
    tmp_path: Path,
) -> None:
    """A rename is not done until the ledger records it.

    The report truncates its list of sites, so the fixture renames the
    alphabetically first attribute the ledger has not yet retired -- the one the
    truncation is guaranteed to name -- and asserts on it by name. `WQ-5` landed
    the site this fixture used to hard-code, so the exemplar comes out of the
    contract rather than out of a literal.
    """

    first = _unretired_attributes()[0]
    package = _renamed_package(tmp_path, first)
    errors = _GATE.contract_errors(_contract(), package_root=package)
    _some(errors, "neither live nor retired")
    _some(errors, first)


def test_a_retired_attribute_is_reported_when_it_still_reaches_a_payload() -> None:
    """Retiring an exclusion does not make the payload key go away.

    The fixture retires a name the census still measures as reaching a payload,
    so the checked-in exclusion and the retirement list contradict each other.
    The retirement row carries the declaration kind because a retired site is no
    longer in the live scan, and the per-slice counts still have to know what it
    was.
    """

    contract = _contract()
    row = _attribute_rows(_contract())[0]
    contract["attribute_retirement"] = {"sites": [dict(row)]}
    _some(_errors(contract), "still reaches a payload")


def test_an_attribute_retirement_outside_the_baseline_is_reported() -> None:
    contract = _contract()
    contract["attribute_retirement"] = {
        "sites": [
            {
                "site": "flagquantum/nowhere.py::Absent::wires",
                "replacement": "qubits",
                "declaration": "field",
            }
        ]
    }
    _one(_errors(contract), "outside the baseline")


def test_an_attribute_retirement_without_a_declaration_kind_is_reported() -> None:
    """The kind is what keeps the per-slice field/member/instance split stable.

    A retirement row that omits it would move its site out of the ledger without
    leaving behind which kind of declaration the slice discharged, so the counts
    would silently drift from the record of what the slice owed.
    """

    contract = _contract()
    contract["attribute_retirement"] = {
        "sites": [
            {
                "site": "flagquantum/core/ir.py::CircuitIR::n_wires",
                "replacement": "n_qubits",
            }
        ]
    }
    _some(_errors(contract), "missing declaration")


def test_an_attribute_retirement_that_is_not_a_rename_is_reported() -> None:
    contract = _contract()
    contract["attribute_retirement"] = {
        "sites": [
            {
                "site": "flagquantum/core/ir.py::CircuitIR::n_wires",
                "replacement": "width",
                "declaration": "field",
            }
        ]
    }
    _some(_errors(contract), "must be replaced by")


def test_an_attribute_retirement_with_an_unknown_declaration_kind_is_reported() -> None:
    contract = _contract()
    contract["attribute_retirement"] = {
        "sites": [
            {
                "site": "flagquantum/core/ir.py::CircuitIR::n_wires",
                "replacement": "n_qubits",
                "declaration": "attribute",
            }
        ]
    }
    _some(_errors(contract), "must name the declaration kind")


def test_an_attribute_alias_table_without_a_removal_version_is_reported() -> None:
    """An alias is a promise with an end date.

    A deprecation with no named removal version has no removal condition, so
    nothing can tell a scheduled migration from compatibility debt that was never
    retired. The version is a contract field for the same reason.
    """

    contract = _contract()
    contract["attribute_aliases"].pop("removal_version")
    _one(_errors(contract), "removal version")


def test_an_attribute_alias_that_is_not_a_rename_is_reported() -> None:
    contract = _contract()
    contract["attribute_aliases"] = {
        "removal_version": "0.4.0",
        "sites": [
            {
                "site": "flagquantum/core/ir.py::CircuitIR::n_wires",
                "replacement": "width",
            }
        ],
    }
    _some(_errors(contract), "must forward to")


def test_an_attribute_alias_without_a_live_declaration_is_reported(
    tmp_path: Path,
) -> None:
    """An alias must forward from somewhere.

    If the old spelling is neither declared nor reachable, the table claims a
    deprecation window that does not exist. This is the failure a slice would hit
    by deleting the forwarder while leaving its ledger row behind.
    """

    package = _package(
        tmp_path,
        {
            "flagquantum/algorithms/amplitude_estimation.py": (
                "from dataclasses import dataclass\n"
                "\n"
                "\n"
                "@dataclass\n"
                "class AmplitudeEstimationResult:\n"
                "    n_counting_qubits: int = 0\n"
            )
        },
    )
    contract = _contract()
    contract["attribute_aliases"] = {
        "removal_version": "0.4.0",
        "sites": [
            {
                "site": (
                    "flagquantum/algorithms/amplitude_estimation.py"
                    "::AmplitudeEstimationResult::n_counting_wires"
                ),
                "replacement": "n_counting_qubits",
            }
        ],
    }
    contract["other_surfaces"]["public_attribute_aliased"] = 1
    errors = _GATE.contract_errors(contract, package_root=package)
    _some(errors, "not a live attribute name")


def test_an_attribute_that_is_both_retired_and_aliased_is_reported() -> None:
    """The two tables answer the same question two ways, so they cannot overlap.

    A retired name is gone; an aliased name is still reachable. Recording both
    would let a slice read its own row as either fact, whichever the report
    needed.
    """

    contract = _contract()
    row = dict(_attribute_rows(_contract())[0])
    contract["attribute_retirement"] = {"sites": [row]}
    contract["attribute_aliases"] = {
        "removal_version": "0.4.0",
        "sites": [{"site": row["site"], "replacement": row["replacement"]}],
    }
    _some(_errors(contract), "retired and kept as an alias")


def test_an_attribute_alias_outside_the_baseline_is_reported() -> None:
    contract = _contract()
    contract["attribute_aliases"] = {
        "removal_version": "0.4.0",
        "sites": [
            {
                "site": "flagquantum/nowhere.py::Absent::wires",
                "replacement": "qubits",
            }
        ],
    }
    _some(_errors(contract), "outside the baseline")


def test_an_attribute_alias_count_that_disagrees_with_the_table_is_reported() -> None:
    contract = _contract()
    contract["other_surfaces"]["public_attribute_aliased"] += 1
    _one(_errors(contract), "public_attribute_aliased")


def test_an_attribute_alias_is_reported_as_remaining_work_not_as_progress() -> None:
    """A rename with a deprecation window is a migration in progress.

    The report is what a reviewer reads to decide whether a slice is finished.
    If an aliased name counted as retired, a slice could rename every attribute,
    ship forwarders for all of them, and report the surface as clear while every
    old spelling still resolves.
    """

    contract = _contract()
    marked = _unretired_attributes()[0]
    contract["attribute_aliases"] = {
        "removal_version": "0.4.0",
        "sites": [
            {
                "site": marked,
                "replacement": _CENSUS.replacement_name(marked.rsplit("::", 1)[1]),
            }
        ],
    }
    contract["other_surfaces"]["public_attribute_aliased"] = 1
    assert _errors(contract) == ()
    rows = _GATE.slice_progress(contract, "attribute")
    # The alias is the only site this fixture moves, and it moves nothing: the
    # retired column is exactly the retirement table, and the aliased site is
    # still counted in the remaining column.
    retired = len(contract["attribute_retirement"]["sites"])
    assert sum(done for _, done, _ in rows) == retired
    assert sum(left for _, _, left in rows) == (
        len(contract["attribute_ledger"]["canonical"]) - retired
    )


def test_the_report_names_the_aliases_apart_from_retirement(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A reader must be able to tell a shimmed rename from an untouched one.

    "remaining" alone cannot say whether the wire spelling is still on disk
    because nobody got to it or because a forwarder deliberately answers it. The
    report prints both numbers, on every surface that has an alias table.

    `WQ-8` exhausted both tables -- every canonical parameter and every canonical
    attribute is now either retired or a declared forwarder -- so the numbers are
    read out of the contract instead of being asserted as literals. The point is
    the split, and the split is exactly what the frozen tables say it is.
    """

    contract = _contract()
    _GATE._report(contract)
    lines = capsys.readouterr().out.splitlines()

    for surface, index, label, ledger, table in (
        ("parameter", 0, "baseline sites", "ledger", "canonical"),
        ("attribute", 8, "attribute sites", "attribute_ledger", "canonical"),
    ):
        total = len(contract[ledger][table])
        retired = len(
            {
                str(entry.get("site", "")) if isinstance(entry, dict) else str(entry)
                for entry in contract[
                    "retirement" if surface == "parameter" else "attribute_retirement"
                ]["sites"]
            }
        )
        aliased = len(_GATE.aliased_sites(contract, surface))
        assert aliased > 0
        assert (
            f"{retired} of {total} {label} retired, "
            f"{aliased} kept as deprecated aliases"
        ) in lines[index]
        # The alias is not progress, and how that shows differs by surface. A
        # parameter alias is declared outside the canonical ledger, so the
        # remaining column is zero while eleven forwarders are still live; an
        # attribute alias is drawn from the live ledger, so all five of them are
        # counted as remaining. Neither surface reports either as retired, which
        # is the fact a reviewer reads the report for.
        rows = _GATE.slice_progress(contract, surface)
        assert sum(done for _, done, _ in rows) == retired
        left = sum(left for _, _, left in rows)
        assert left + retired == total
        assert left == (0 if surface == "parameter" else aliased)

    assert (
        f"{len(contract['definition_ledger']['sites'])} definition names" in lines[16]
    )


def test_a_slice_attribute_count_that_disagrees_with_the_ledger_is_reported() -> None:
    contract = _contract()
    contract["slices"][0]["attribute_count"] += 1
    _one(_errors(contract), "attribute_count")


def test_a_slice_field_count_that_disagrees_with_the_ledger_is_reported() -> None:
    """The split by declaration kind is a recorded count, not a derived one.

    A field and a member carry different obligations: a field's name can only be
    a payload key through `asdict`, while a member's cannot be one that way at
    all. If the two were merged into one number, converting a field into a
    property would move the obligation without changing any count.
    """

    contract = _contract()
    contract["slices"][0]["attribute_field_count"] += 1
    _one(_errors(contract), "attribute_field_count")


def test_a_slice_member_count_that_disagrees_with_the_ledger_is_reported() -> None:
    contract = _contract()
    contract["slices"][0]["attribute_member_count"] += 1
    _one(_errors(contract), "attribute_member_count")


def test_a_slice_instance_count_that_disagrees_with_the_ledger_is_reported() -> None:
    """`self.name = ...` is a user-visible name, so it is a counted site.

    Without this count a slice could rename its fields, leave every
    `TextDrawer().wire_order` untouched, and still look finished: the field count
    would balance and nothing would name the instance attributes.
    """

    contract = _contract()
    contract["slices"][0]["attribute_instance_count"] += 1
    _one(_errors(contract), "attribute_instance_count")


def test_an_exclusion_without_a_declaration_kind_is_reported() -> None:
    contract = _contract()
    _attribute_rows(contract)[0]["declaration"] = ""
    _some(_errors(contract), "declaration")


def test_an_exclusion_whose_declaration_kind_changed_is_reported() -> None:
    """A field that becomes a property is not the same obligation any more."""

    contract = _contract()
    row = next(
        entry
        for entry in _attribute_rows(contract)
        if entry["site"].endswith("::Instruction::wires")
    )
    row["declaration"] = "member"
    _some(_errors(contract), "no longer matches the scan")


def test_a_slice_persisted_count_that_disagrees_with_the_ledger_is_reported() -> None:
    contract = _contract()
    contract["slices"][0]["persisted_attribute_count"] += 1
    _one(_errors(contract), "persisted_attribute_count")


def test_a_slice_definition_count_that_disagrees_with_the_ledger_is_reported() -> None:
    contract = _contract()
    contract["slices"][0]["definition_count"] += 1
    _one(_errors(contract), "definition_count")


def test_an_exclusion_without_a_reason_is_reported() -> None:
    contract = _contract()
    _attribute_rows(contract)[0]["reason"] = ""
    _some(_errors(contract), "reason")


def test_an_exclusion_without_a_removal_condition_is_reported() -> None:
    contract = _contract()
    _attribute_rows(contract)[0]["removal_condition"] = ""
    _some(_errors(contract), "removal_condition")


# ------------------------------------------------------------ definition rules


def _definition_rows(contract: dict[str, Any]) -> list[dict[str, Any]]:
    return contract["definition_ledger"]["sites"]


def test_a_missing_definition_ledger_is_reported() -> None:
    contract = _contract()
    contract["definition_ledger"]["sites"] = []
    _some(_errors(contract), "must ledger its definition names")


def test_a_definition_ledger_that_disagrees_with_the_measured_total_is_reported() -> (
    None
):
    contract = _contract()
    contract["definition_ledger"]["measured_canonical"] = 9
    _one(_errors(contract), "definition_ledger.measured_canonical")


def test_a_definition_total_that_disagrees_with_the_ledger_is_reported() -> None:
    contract = _contract()
    contract["other_surfaces"]["public_definition_names"] = 9
    _one(_errors(contract), "public_definition_names")


def test_a_duplicated_definition_entry_is_reported() -> None:
    contract = _contract()
    rows = _definition_rows(contract)
    rows.append(dict(rows[0]))
    contract["definition_ledger"]["measured_canonical"] += 1
    _some(_errors(contract), "definition ledger repeats")


def test_a_new_wire_named_public_definition_outside_the_ledger_is_reported(
    tmp_path: Path,
) -> None:
    """A function whose own name says wire is a site, not a parameter.

    The fixture declares a name the ledger does not contain. Neither the
    parameter ledger nor the attribute ledger can see it, so without this
    surface the gate would pass while the spelling is on the user's screen.
    """

    package = _package(
        tmp_path,
        {
            "flagquantum/algorithms/amplitude_estimation.py": (
                "def amplitude_estimation_circuit(n_counting_wires):\n"
                "    return n_counting_wires\n"
            ),
            "flagquantum/simulation/pauli.py": (
                "def count_wires_from_dense_state(state):\n" "    return state\n"
            ),
        },
    )
    errors = _GATE.contract_errors(_contract(), package_root=package)
    _some(errors, "public definitions outside the ledger")
    _some(errors, "count_wires_from_dense_state")


def test_a_definition_rename_without_a_retirement_entry_is_reported(
    tmp_path: Path,
) -> None:
    """A rename is not done until the ledger records it.

    The fixture un-retires the alphabetically first definition name in the ledger
    and renames it in a copy of its own file. The live scan no longer sees the old
    spelling, so the entry is neither live nor retired -- which is what stops a
    slice from renaming the code and editing the ledger away in the same commit.
    """

    contract = _contract()
    first = _unretire_a_definition(contract)
    assert first == (
        "flagquantum/benchmarking/statevector_cpu_paths.py"
        "::build_two_wire_diagonal_chain"
    )
    package = _renamed_package(tmp_path, first)
    errors = _GATE.contract_errors(contract, package_root=package)
    _some(errors, "neither live nor retired")
    _some(errors, first)


def test_a_fabricated_definition_retirement_is_reported(tmp_path: Path) -> None:
    """The other direction: claiming a rename that the code does not show.

    The fixture points the scan at an unmodified copy of the file, so the old
    spelling is genuinely still there and the retirement row is a lie.
    """

    contract = _contract()
    live = _unretire_a_definition(contract)
    contract["definition_retirement"]["sites"] = [live]
    package = _wire_named_package(tmp_path, live)
    errors = _GATE.contract_errors(contract, package_root=package)
    _some(errors, "still exist")
    _some(errors, live)


def test_a_definition_retirement_outside_the_baseline_is_reported() -> None:
    contract = _contract()
    contract["definition_retirement"] = {
        "sites": ["flagquantum/nowhere.py::absent_wires"]
    }
    _one(_errors(contract), "outside the baseline")


def test_a_definition_rename_that_disagrees_with_the_rule_is_reported(
    tmp_path: Path,
) -> None:
    """A ledger row may not name a replacement the substitution rule forbids.

    The rule check only runs on a row whose site is still live, so the fixture
    un-retires one and scans the unmodified file it names -- the site is on disk
    under its old spelling and the ledger disagrees with what the census would
    have proposed for it.
    """

    contract = _contract()
    live = _unretire_a_definition(contract)
    row = next(
        entry for entry in _definition_rows(contract) if str(entry["site"]) == live
    )
    row["replacement"] = "build_two_qubit_indices_diagonal_chain"
    package = _wire_named_package(tmp_path, live)
    _some(_GATE.contract_errors(contract, package_root=package), "must be replaced by")


def test_a_definition_without_a_kind_is_reported() -> None:
    contract = _contract()
    _definition_rows(contract)[0]["kind"] = ""
    _some(_errors(contract), "kind")


def test_the_definition_ledger_is_the_scanner_output_at_the_baseline() -> None:
    """The same measurement claim as the parameter ledger, one surface over.

    A definition name may only leave the live surface through a retirement row,
    so the ledger minus the retirements has to equal the scanner exactly and no
    retired name may still be found.
    """

    contract = _contract()
    scanned = _CENSUS.definition_census(_ROOT / "flagquantum")
    declared = {
        str(row["site"]): str(row["replacement"]) for row in _definition_rows(contract)
    }
    retired = {str(site) for site in contract["definition_retirement"]["sites"]}
    live = {site.identifier: site.replacement for site in scanned.ledgered}
    assert {site: declared[site] for site in set(declared) - retired} == live
    assert not retired & set(live)


# --------------------------------------------------------------------- live facts


def test_the_ledger_is_the_scanner_output_at_the_baseline() -> None:
    """The ledger is measured, not hand-written: it is the scan plus what landed.

    A ledger line may only disappear from the live surface through a retirement
    row, so the two together have to equal the scanner exactly. A missing
    retirement row puts a live site on the right-hand side only.
    """

    contract = _contract()
    scanned = _CENSUS.census(_ROOT / "flagquantum")
    assert set(contract["ledger"]["canonical"]) - set(
        contract["retirement"]["sites"]
    ) == {site.identifier for site in scanned.canonical}


def test_the_existing_aliases_are_the_ones_the_scanner_finds() -> None:
    contract = _contract()
    scanned = _CENSUS.census(_ROOT / "flagquantum")
    declared = {
        str(row["site"]): str(row["replacement"])
        for row in contract["aliases"]["declared"]
    }
    assert declared == {site.identifier: site.replacement for site in scanned.aliases}

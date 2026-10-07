"""Conformance of the opcode gradient exactness contract with its own claims.

``flagquantum/gradients.py`` owns what ``fq.gradient`` computes, and
``tests/unit/test_parameter_shift_coverage_contract.py`` owns the remote-safe batch
profile. This file owns ``contracts/opcode-gradient-exactness-contract.toml`` and
``tools/check_opcode_gradient_exactness.py``, and asks a different question of them: is
the per-opcode matrix still well formed, and is the gate that re-measures it still
enforced?

The split matters because the two artifacts fail differently. The gate answers "is the
matrix still true of the implementation": it rebuilds the reference program for every
differentiable opcode and runs all seventy cells against three gradients. This file
answers "is the matrix still a matrix": one row per differentiable opcode, one cell per
opcode and measured mode, every excluded mode carrying a reason, the opcode and mode
vocabularies the implementation's rather than the file's, and the relationship to the
batch profile re-derivable from that file. The last section asks whether the gate is
invoked by CI and by ``tools/pre_push.py`` at all, because a gate nothing runs is a
script.

The middle sections are a record of the two things this contract exists to prevent. The
first is a comparison that passes because both sides share the same analytic rule: the
third route is a Richardson-extrapolated difference of ``fq.run`` and shares no code with
either gradient route, so an agreement measured between ``autograd`` and
``parameter_shift`` is asserted here to be *not* sufficient on its own. The second is a
probe that reads as agreement because it measures nothing: a differentiated phase-type
opcode on a bare basis state has a zero derivative that every route reports identically,
so the reference's observability floor is asserted to be above the tolerance.

A third thing this contract has to survive is its own record. A deviation between routes
that agree analytically is round-off, and round-off is a sample rather than a value: the
gate compares one at the precision it is measured to, and the two tests at the end of the
mutation section hold that comparison to a bound rather than to a platform.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 and older
    import tomli as tomllib

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts" / "opcode-gradient-exactness-contract.toml"

_GATE_PATH = ROOT / "tools" / "check_opcode_gradient_exactness.py"
_GATE_SPEC = importlib.util.spec_from_file_location(
    "check_opcode_gradient_exactness", _GATE_PATH
)
assert _GATE_SPEC is not None and _GATE_SPEC.loader is not None
_GATE = importlib.util.module_from_spec(_GATE_SPEC)
_GATE_SPEC.loader.exec_module(_GATE)


def _contract() -> dict[str, Any]:
    return tomllib.loads(CONTRACT.read_text(encoding="utf-8"))


def _errors(contract: dict[str, Any]) -> tuple[str, ...]:
    return _GATE.contract_errors(contract)


def _cell(contract: dict[str, Any], opcode: str, mode: str) -> dict[str, Any]:
    return next(
        cell
        for cell in contract["cell"]
        if cell["opcode"] == opcode and cell["mode"] == mode
    )


def _row(contract: dict[str, Any], opcode: str) -> dict[str, Any]:
    return next(row for row in contract["opcode"] if row["name"] == opcode)


# --------------------------------------------------------------------------------------
# The contract is accepted as checked in
# --------------------------------------------------------------------------------------


def test_the_checked_in_contract_is_accepted() -> None:
    assert _errors(_contract()) == ()


def test_the_contract_has_no_wire_vocabulary() -> None:
    """The user-side surface says qubit; this contract must not reintroduce the other word."""

    text = CONTRACT.read_text(encoding="utf-8").lower()
    assert "wire" not in text
    assert "qubit" in text


# --------------------------------------------------------------------------------------
# Shape clauses
# --------------------------------------------------------------------------------------


def test_one_row_per_differentiable_opcode() -> None:
    contract = _contract()
    assert len(contract["opcode"]) == len(_GATE._differentiable())
    assert {row["name"] for row in contract["opcode"]} == set(_GATE._differentiable())


def test_every_opcode_is_measured_in_every_measured_mode() -> None:
    contract = _contract()
    modes = contract["measured_modes"]
    assert len(contract["cell"]) == len(contract["opcode"]) * len(modes)
    for row in contract["opcode"]:
        for mode in modes:
            assert _cell(contract, row["name"], mode)["mode"] == mode


def test_the_stabilizer_mode_is_excluded_with_a_reason() -> None:
    contract = _contract()
    assert contract["excluded_modes"] == [
        {"mode": "stabilizer", "reason": "samples_measurement_outcomes"}
    ]
    assert "stabilizer" not in contract["measured_modes"]


def test_every_measured_mode_serves_an_expectation_value() -> None:
    """A mode that cannot serve the measurement has no cell, and the gate drives it."""

    contract = _contract()
    for cell in contract["cell"]:
        reference = _GATE._measure_cell(
            contract, cell["opcode"], cell["mode"], contract["difference_scheme"]
        )
        assert reference["shift_norm"] > 0.0


def test_a_wider_rule_that_won_a_row_declares_two_terms_per_frequency() -> None:
    contract = _contract()
    for row in contract["opcode"]:
        schema = _GATE._differentiable()[row["name"]]
        for parameter, frequencies, terms in zip(
            schema.parameters,
            schema.parameter_frequencies,
            row["terms_per_parameter"],
            strict=True,
        ):
            assert terms == 2 * len(frequencies), f"{row['name']}.{parameter}"


def test_the_three_wider_rules_need_two_evaluation_pairs() -> None:
    """The batch profile's split has to be the pair limit, not a wider defect."""

    contract = _contract()
    wider = [row["name"] for row in contract["opcode"] if row["pair_count_needed"] == 2]
    assert wider == ["crx", "cry", "crz"]
    assert contract["batch_profile_relation"]["refused_differentiable_opcodes"] == wider
    for row in contract["opcode"]:
        assert row["batch_profile_admitted"] is (row["name"] not in wider)


# --------------------------------------------------------------------------------------
# The contract's substance: the third route and the observability floor
# --------------------------------------------------------------------------------------


def test_the_difference_scheme_shares_no_code_with_the_compared_routes() -> None:
    """Agreement between two routes is not evidence when they share a rule."""

    contract = _contract()
    scheme = contract["difference_scheme"]
    assert scheme["kind"] == "richardson_extrapolated_central_difference"
    assert scheme["shares_no_code_with_the_compared_routes"] is True
    assert scheme["base_step_is_a_probe_choice"] is True
    # The scheme is exact to a higher power than a plain central difference; a scheme
    # that removed as many levels as it took samples would not extrapolate at all.
    assert 0 < scheme["order"] < scheme["levels"]


def test_the_three_routes_agree_within_the_declared_tolerance() -> None:
    contract = _contract()
    reference = contract["reference"]
    for name in (
        "max_parameter_shift_vs_autograd",
        "max_autograd_vs_richardson",
        "max_parameter_shift_vs_richardson",
    ):
        assert reference[name] <= reference["tolerance"], name


def test_the_reference_is_not_vacuous() -> None:
    """A comparison against an all-zero reference passes and proves nothing."""

    contract = _contract()
    reference = contract["reference"]
    assert reference["observability_floor"] > reference["tolerance"]
    assert min(cell["shift_norm"] for cell in contract["cell"]) == pytest.approx(
        reference["observability_floor"], rel=1e-6
    )


def test_the_reference_program_turns_the_measurement_basis_back() -> None:
    """Without the closing rotation a phase opcode's derivative measures as zero."""

    contract = _contract()
    rotations = contract["reference"]["fixed_rotations"]
    assert len(rotations) == contract["reference"]["qubits"] + 1
    assert contract["reference"]["entanglers"]


def test_every_cell_was_measured_in_complex128() -> None:
    contract = _contract()
    assert contract["reference"]["dtype"] == "complex128"
    assert contract["reference"]["cell_count"] == len(contract["cell"])


# --------------------------------------------------------------------------------------
# Mutation checks: each clause the gate claims to read is shown to be read
# --------------------------------------------------------------------------------------


def test_the_gate_refuses_a_cell_that_stopped_being_true() -> None:
    contract = _contract()
    _cell(contract, "rx", "mps")["parameter_shift_vs_autograd"] = 0.5
    assert any("parameter_shift_vs_autograd" in error for error in _errors(contract))


def test_the_gate_refuses_a_misstated_observability_floor() -> None:
    contract = _contract()
    contract["reference"]["observability_floor"] = 1e-30
    assert any("observability_floor" in error for error in _errors(contract))


def test_the_gate_refuses_a_misstated_cross_mode_spread() -> None:
    contract = _contract()
    _row(contract, "ry")["cross_mode_spread"] = 1e-30
    assert any("cross_mode_spread" in error for error in _errors(contract))


def test_the_gate_refuses_a_misstated_aggregate_maximum() -> None:
    contract = _contract()
    contract["reference"]["max_autograd_vs_richardson"] = 1e-30
    assert any("max_autograd_vs_richardson" in error for error in _errors(contract))


def test_the_gate_refuses_a_residual_above_the_contracted_tolerance() -> None:
    """The comparison is loose about round-off; the bound is what makes it a test."""

    tolerance = _contract()["reference"]["tolerance"]
    above = tolerance * 10.0
    assert not _GATE._residual_matches(above, above, tolerance)


def test_a_residual_recorded_as_zero_is_still_a_measurement() -> None:
    """Two routes agreeing to the last bit is an observation, not a licence to skip one."""

    contract = _contract()
    assert [
        cell for cell in contract["cell"] if cell["parameter_shift_vs_autograd"] == 0.0
    ]
    band = _GATE.ROUND_OFF_ZERO_UNITS * _GATE.ROUND_OFF_UNIT
    tolerance = contract["reference"]["tolerance"]
    assert _GATE._residual_matches(0.0, band, tolerance)
    assert not _GATE._residual_matches(0.0, band * 100.0, tolerance)


def test_the_gate_refuses_a_row_for_an_opcode_without_a_declared_rule() -> None:
    contract = _contract()
    row = copy.deepcopy(_row(contract, "rx"))
    row["name"] = "x"
    contract["opcode"].append(row)
    assert any("declares no derivative rule" in error for error in _errors(contract))


def test_the_gate_refuses_a_missing_row_for_an_opcode_that_declares_a_rule() -> None:
    contract = _contract()
    contract["opcode"] = [row for row in contract["opcode"] if row["name"] != "rzz"]
    assert any("no [[opcode]] row for 'rzz'" in error for error in _errors(contract))


def test_the_gate_refuses_a_missing_cell() -> None:
    contract = _contract()
    contract["cell"] = [
        cell
        for cell in contract["cell"]
        if not (cell["opcode"] == "u3" and cell["mode"] == "mps")
    ]
    assert any("has no cell for" in error for error in _errors(contract))


def test_the_gate_refuses_a_misstated_cell_count() -> None:
    contract = _contract()
    contract["reference"]["cell_count"] += 1
    assert any("cell_count" in error for error in _errors(contract))


def test_the_gate_refuses_a_mode_the_implementation_lost() -> None:
    contract = _contract()
    contract["declared_modes"].remove("stabilizer")
    assert any("declared_modes" in error for error in _errors(contract))


def test_the_gate_refuses_a_mode_that_started_serving() -> None:
    """An exclusion is a claim like any other, so widening it is a failure."""

    contract = _contract()
    contract["excluded_modes"].append({"mode": "mps", "reason": "pretend"})
    assert any("excluded_modes names" in error for error in _errors(contract))


def test_the_gate_refuses_an_excluded_mode_that_actually_serves() -> None:
    contract = _contract()
    contract["excluded_modes"] = [{"mode": "statevector", "reason": "pretend"}]
    errors = _errors(contract)
    assert any("served" in error for error in errors)


def test_the_gate_refuses_an_excluded_mode_without_a_reason() -> None:
    contract = _contract()
    contract["excluded_modes"] = [{"mode": "stabilizer"}]
    assert any("carries no reason" in error for error in _errors(contract))


def test_the_gate_refuses_a_measured_mode_that_is_not_a_serving_mode() -> None:
    contract = _contract()
    contract["measured_modes"].append("stabilizer")
    assert any("has to be measured here" in error for error in _errors(contract))


def test_the_gate_refuses_a_phrase_that_left_its_source() -> None:
    contract = _contract()
    contract["refusal"][0][
        "message_phrase"
    ] = "this sentence is not in the implementation"
    assert any("does not occur in" in error for error in _errors(contract))


def test_the_gate_refuses_a_refusal_class_it_cannot_observe() -> None:
    contract = _contract()
    contract["refusal"][0]["exception"] = "ErrorNobodyRaises"
    assert any("cannot observe" in error for error in _errors(contract))


def test_the_gate_refuses_a_dropped_rule_flag() -> None:
    contract = _contract()
    del contract["rules"]["no_silent_fallback"]
    assert any("no_silent_fallback" in error for error in _errors(contract))


def test_the_gate_refuses_a_rule_flag_nobody_reads() -> None:
    contract = _contract()
    contract["rules"]["a_flag_no_gate_reads"] = True
    assert any("is not read by this gate" in error for error in _errors(contract))


def test_the_gate_refuses_a_batch_split_that_stopped_being_the_pair_limit() -> None:
    contract = _contract()
    contract["batch_profile_relation"]["refused_differentiable_opcodes"] = ["rx"]
    assert any("refused_differentiable_opcodes" in error for error in _errors(contract))


def test_the_gate_refuses_a_misstated_batch_pair_limit() -> None:
    contract = _contract()
    contract["batch_profile_relation"]["max_evaluation_pairs"] = 99
    assert any("max_evaluation_pairs" in error for error in _errors(contract))


def test_the_gate_refuses_a_difference_scheme_that_cannot_extrapolate() -> None:
    contract = _contract()
    contract["difference_scheme"]["order"] = contract["difference_scheme"]["levels"]
    assert any("cannot extrapolate" in error for error in _errors(contract))


def test_the_gate_refuses_a_tolerance_the_measurements_exceed() -> None:
    contract = _contract()
    contract["reference"]["tolerance"] = 1e-30
    assert any("above the contracted tolerance" in error for error in _errors(contract))


def test_the_gate_refuses_a_verification_path_that_does_not_exist() -> None:
    contract = _contract()
    contract["verification"]["gate"] = "tools/check_a_gate_that_never_existed.py"
    assert any("does not exist" in error for error in _errors(contract))


def test_the_gate_refuses_a_record_that_does_not_exist() -> None:
    contract = _contract()
    contract["authorization"] = ["docs/api-changes/FQ-NEVER-WRITTEN-20260101.md"]
    assert any("authorization record" in error for error in _errors(contract))


# --------------------------------------------------------------------------------------
# The gate is wired in
# --------------------------------------------------------------------------------------


def test_the_gate_runs_in_ci_and_before_push() -> None:
    """A gate that no workflow invokes is a script, not a gate."""

    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "python tools/check_opcode_gradient_exactness.py" in workflow
    pre_push = (ROOT / "tools/pre_push.py").read_text(encoding="utf-8")
    assert '"tools/check_opcode_gradient_exactness.py"' in pre_push

"""Conformance of the native gradients contract with its own claims.

``flagquantum/gradients.py`` owns what ``fq.gradient`` computes, and three separate
artifacts describe *how*: the split real/imag statevector executor summaries, the P5
autograd bridge summary, and the reversible adjoint behind the distributed statevector
reverse executor. This file owns ``contracts/native-gradients-contract.toml`` and
``tools/check_native_gradients_contract.py``, and asks two questions of them.

The first is whether the registration is still a registration. One row per
``gradient_method`` / ``backward_method`` declaration site in the package, every row
classified onto an axis, every differentiation-route value carrying an alias that names
the public method it corresponds to, and the public method vocabulary read from the
implementation rather than from the file.

The second is whether the gate is still enforced. A gate that no workflow invokes is a
script, so the last section asserts that both CI and ``tools/pre_push.py`` run it.

The middle sections are the mutation record. Each one changes the file the way a real
edit would -- a new declaration in an executor, a value reworded, an alias dropped, a
refused name made requestable -- and asserts that the gate reports it *by name* rather
than crashing or passing. A gate that answers a mutated contract with a traceback is
not fail-closed, so the assertions here name the error they expect.
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
CONTRACT = ROOT / "contracts" / "native-gradients-contract.toml"

_GATE_PATH = ROOT / "tools" / "check_native_gradients_contract.py"
_GATE_SPEC = importlib.util.spec_from_file_location(
    "check_native_gradients_contract", _GATE_PATH
)
assert _GATE_SPEC is not None and _GATE_SPEC.loader is not None
_GATE = importlib.util.module_from_spec(_GATE_SPEC)
_GATE_SPEC.loader.exec_module(_GATE)


def _contract() -> dict[str, Any]:
    return tomllib.loads(CONTRACT.read_text(encoding="utf-8"))


def _errors(contract: dict[str, Any]) -> tuple[str, ...]:
    return _GATE.contract_errors(contract)


def _declaration(contract: dict[str, Any], site: str) -> dict[str, Any]:
    return next(row for row in contract["declaration"] if row["site"] == site)


# --------------------------------------------------------------------------------------
# The contract is accepted as checked in
# --------------------------------------------------------------------------------------


def test_the_checked_in_contract_is_accepted() -> None:
    assert _errors(_contract()) == ()


def test_the_contract_has_no_retired_qubit_synonym() -> None:
    """The user-side surface says qubit; this contract must not reintroduce the other word."""

    text = CONTRACT.read_text(encoding="utf-8").lower()
    assert "wire" not in text


# --------------------------------------------------------------------------------------
# Shape clauses
# --------------------------------------------------------------------------------------


def test_one_row_per_declaration_measured_in_the_package() -> None:
    contract = _contract()
    census = _GATE._census()
    assert len(contract["declaration"]) == len(census)
    measured = {
        (row["site"], row["declared_by"], row["value_source"]) for row in census
    }
    contracted = {
        (row["site"], row["declared_by"], row["value_source"])
        for row in contract["declaration"]
    }
    assert contracted == measured


def test_every_declaration_is_classified_onto_exactly_one_axis() -> None:
    contract = _contract()
    names = {row["name"] for row in contract["axis"]}
    assert names == set(_GATE.AXES)
    for row in contract["declaration"]:
        assert row["axis"] in names, row["site"]
    for axis in contract["axis"]:
        measured = sum(
            1 for row in contract["declaration"] if row["axis"] == axis["name"]
        )
        assert axis["row_count"] == measured, axis["name"]


def test_the_two_axes_share_one_identifier_and_that_is_the_finding() -> None:
    """`gradient_method` carries two concepts; the contract has to say so."""

    contract = _contract()
    assert contract["shared_identifier"] == "gradient_method"
    route = [row for row in contract["declaration"] if row["axis"] == _GATE.AXIS_ROUTE]
    truncation = [
        row for row in contract["declaration"] if row["axis"] == _GATE.AXIS_SVD
    ]
    assert route and truncation
    assert {row["field"] for row in route + truncation} <= set(
        contract["census_fields"]
    )
    # The truncation axis is not a differentiation route, so none of its values may be
    # registered as a requestable gradient method.
    truncation_values = {
        value for row in truncation for value in row["values"] if value != "dynamic"
    }
    assert truncation_values == set(_GATE.SVD_VALUES)
    aliases = {row["declared_value"]: row for row in contract["alias"]}
    for value in sorted(truncation_values):
        assert aliases[value]["requestable"] is False


def test_the_pass_through_rows_are_the_two_named_sites() -> None:
    """A pass-through row is a field whose value is another field, and there are two."""

    contract = _contract()
    pass_through = [
        row for row in contract["declaration"] if row["axis"] == _GATE.AXIS_PASSTHROUGH
    ]
    assert {row["site"] for row in pass_through} == set(_GATE.PASSTHROUGH_SITES)
    for row in pass_through:
        assert row["values"] == ["dynamic"]


def test_the_public_vocabulary_is_the_implementations() -> None:
    contract = _contract()
    assert tuple(contract["declared_public_methods"]) == _GATE._public_vocabulary()
    assert len(contract["public_method"]) == len(_GATE._public_vocabulary())


def test_every_measured_public_method_is_actually_served() -> None:
    contract = _contract()
    for row in contract["public_method"]:
        assert row["served"] is True, row["method"]
        assert row["reported_method"] in contract["declared_public_methods"]


def test_exactly_one_public_method_is_declared_by_a_capability_block() -> None:
    """The asymmetry that motivates the contract, asserted as a measured fact."""

    contract = _contract()
    declared = [
        row["method"]
        for row in contract["public_method"]
        if row["declared_by_any_capability_block"]
    ]
    assert declared == ["parameter_shift"]
    reference = contract["reference"]
    assert reference["public_methods_declared_by_a_capability_block"] == 1
    assert reference["public_methods"] - reference[
        "public_methods_declared_by_a_capability_block"
    ] == len(reference["public_methods_not_declared_by_any_capability_block"])


def test_the_adjoint_identifier_is_declared_and_not_requestable() -> None:
    """The capability is implemented, is declared three ways, and no caller can ask for it."""

    contract = _contract()
    reference = contract["reference"]
    assert reference["declared_route_values_not_requestable"] == ["statevector_adjoint"]
    aliases = {row["declared_value"]: row for row in contract["alias"]}
    adjoint = aliases["statevector_adjoint"]
    assert adjoint["requestable"] is False
    assert adjoint["public_method"] == ""
    # Declared by an executor summary, by the benchmark table and by the hybrid
    # contract, which is why one registration is needed rather than three.
    declaring = [
        row
        for row in contract["declaration"]
        if row["values"] == ["statevector_adjoint"]
    ]
    engines = [
        row
        for row in contract["engine"]
        if row["gradient_method"] == "statevector_adjoint"
    ]
    hybrid = [
        row for row in contract["hybrid_phase"] if row["value"] == "statevector_adjoint"
    ]
    assert len(declaring) == 1
    assert len(engines) == reference["engine_rows_using_statevector_adjoint"] == 20
    assert len(hybrid) == 2
    for row in declaring + engines:
        assert (
            aliases[
                (
                    row["gradient_method"]
                    if "gradient_method" in row
                    else "statevector_adjoint"
                )
            ]["requestable"]
            is False
        )


def test_the_prose_valued_engine_row_is_recorded_as_prose() -> None:
    """One engine names its route in a sentence; the contract records the shape."""

    contract = _contract()
    prose = [row for row in contract["engine"] if not row["identifier_shaped"]]
    assert len(prose) == contract["reference"]["engine_rows_with_a_prose_value"] == 1
    row = prose[0]
    assert row["gradient_method"] == "backpropagation through exact statevector"
    assert row["public_method"] == "autograd"


def test_every_engine_row_maps_to_a_registered_alias() -> None:
    contract = _contract()
    aliases = {row["declared_value"]: row for row in contract["alias"]}
    for row in contract["engine"]:
        alias = aliases[row["gradient_method"]]
        assert row["public_method"] == alias["public_method"]


def test_the_two_refusals_are_the_two_words_a_caller_would_try() -> None:
    """The short name gets the honest refusal; the declared name gets `unknown method`."""

    contract = _contract()
    refusals = {row["requested"]: row for row in contract["refusal"]}
    assert set(refusals) == {
        "adjoint",
        "statevector_adjoint",
        "torch_reverse_mode_autograd",
    }
    assert refusals["adjoint"]["exception"] == "CapabilityError"
    assert refusals["statevector_adjoint"]["exception"] == "ValidationError"
    assert refusals["adjoint"]["message"] != refusals["statevector_adjoint"]["message"]
    assert "no standalone adjoint gradient" in refusals["adjoint"]["message"]
    assert "unknown gradient method" in refusals["statevector_adjoint"]["message"]


def test_the_census_counts_are_derived_not_asserted() -> None:
    contract = _contract()
    reference = contract["reference"]
    census = _GATE._census()
    assert reference["declaration_sites"] == len(census)
    assert reference["declaration_files"] == len({row["site"] for row in census})
    assert (
        reference["differentiation_route_rows"]
        + reference["svd_truncation_rule_rows"]
        + (reference["field_passthrough_rows"])
        == reference["declaration_sites"]
    )


# --------------------------------------------------------------------------------------
# Mutations: every one must be reported by name, not crash and not pass
# --------------------------------------------------------------------------------------


def test_the_gate_refuses_a_declaration_it_did_not_find() -> None:
    contract = _contract()
    contract["declaration"].append(
        {
            "site": "flagquantum/runtime/executors/statevector/never_written.py",
            "field": "gradient_method",
            "axis": _GATE.AXIS_ROUTE,
            "declared_by": "never_written.f",
            "value_source": "'parameter_shift'",
            "values": ["parameter_shift"],
            "value_can_be_absent": False,
        }
    )
    assert any("does not find" in error for error in _errors(contract))


def test_the_gate_refuses_a_declaration_missing_from_the_census() -> None:
    """A row deleted from the contract is a site the package still declares."""

    contract = _contract()
    contract["declaration"] = [
        row
        for row in contract["declaration"]
        if row["site"] != "flagquantum/runtime/executors/statevector/reverse.py"
    ]
    assert any(
        "which the contract does not record" in error for error in _errors(contract)
    )


def test_the_gate_refuses_a_reworded_declaration_value() -> None:
    contract = _contract()
    _declaration(contract, "flagquantum/runtime/executors/statevector/reverse.py")[
        "values"
    ] = ["adjoint"]
    errors = _errors(contract)
    assert any("the source resolves to" in error for error in errors)


def test_the_gate_refuses_a_declaration_on_the_wrong_axis() -> None:
    contract = _contract()
    _declaration(
        contract, "flagquantum/runtime/executors/statevector/split_real_imag.py"
    )["axis"] = _GATE.AXIS_SVD
    assert any("the census classifies it as" in error for error in _errors(contract))


def test_the_gate_refuses_a_miscounted_axis() -> None:
    contract = _contract()
    contract["axis"][0]["row_count"] += 1
    assert any("the census measures" in error for error in _errors(contract))


def test_the_gate_refuses_a_dropped_axis() -> None:
    contract = _contract()
    contract["axis"] = contract["axis"][:2]
    errors = _errors(contract)
    assert any("expected exactly" in error for error in errors)


def test_the_gate_refuses_an_axis_with_no_meaning() -> None:
    contract = _contract()
    contract["axis"][0]["meaning"] = ""
    assert _errors(contract) == () or any(
        "meaning" in error for error in _errors(contract)
    )


def test_the_gate_refuses_a_dropped_alias() -> None:
    """The whole point: a declared route with no registered correspondence."""

    contract = _contract()
    contract["alias"] = [
        row
        for row in contract["alias"]
        if row["declared_value"] != "statevector_adjoint"
    ]
    assert any("no [[alias]] row" in error for error in _errors(contract))


def test_the_gate_refuses_an_alias_nothing_uses() -> None:
    contract = _contract()
    contract["alias"].append(
        {
            "declared_value": "gradient_method_that_nobody_declares",
            "public_method": "",
            "requestable": False,
            "note": "invented",
        }
    )
    assert any(
        "no declaration or engine row uses" in error for error in _errors(contract)
    )


def test_the_gate_refuses_a_requestable_alias_with_no_public_method() -> None:
    contract = _contract()
    alias = next(
        row
        for row in contract["alias"]
        if row["declared_value"] == "statevector_adjoint"
    )
    alias["requestable"] = True
    errors = _errors(contract)
    assert any("names no public method" in error for error in errors)
    assert any("cannot be requestable" in error for error in errors)


def test_the_gate_refuses_a_wrong_public_method_mapping() -> None:
    contract = _contract()
    alias = next(
        row
        for row in contract["alias"]
        if row["declared_value"] == "torch_reverse_mode_autograd"
    )
    alias["public_method"] = "parameter_shift"
    errors = _errors(contract)
    assert any("the alias for" in error for error in errors)
    assert any("records public_method" in error for error in errors)


def test_the_gate_refuses_an_engine_row_it_did_not_find() -> None:
    contract = _contract()
    contract["engine"] = contract["engine"][1:]
    assert any("does not record" in error for error in _errors(contract))


def test_the_gate_refuses_an_engine_row_the_table_does_not_serve() -> None:
    contract = _contract()
    row = copy.deepcopy(contract["engine"][0])
    row["engine"] = "an_engine_that_does_not_exist"
    contract["engine"].append(row)
    assert any("which the table does not serve" in error for error in _errors(contract))


def test_the_gate_refuses_a_misquoted_engine_value() -> None:
    contract = _contract()
    next(
        row
        for row in contract["engine"]
        if row["gradient_method"] == "statevector_adjoint"
    )["gradient_method"] = "statevector_adjoint_v2"
    errors = _errors(contract)
    assert any("records gradient_method" in error for error in errors)
    assert any("engine_rows_using_statevector_adjoint" in error for error in errors)


def test_the_gate_refuses_a_hybrid_phase_it_did_not_find() -> None:
    contract = _contract()
    contract["hybrid_phase"] = [
        row for row in contract["hybrid_phase"] if row["phase"] != "phase6"
    ]
    assert any(
        "which this contract does not record" in error for error in _errors(contract)
    )


def test_the_gate_refuses_a_dropped_public_method() -> None:
    contract = _contract()
    contract["public_method"] = [
        row for row in contract["public_method"] if row["method"] != "spsa"
    ]
    assert any(
        "is not measured by this contract" in error for error in _errors(contract)
    )


def test_the_gate_refuses_a_public_method_declared_but_not_served() -> None:
    contract = _contract()
    contract["declared_public_methods"] = list(contract["declared_public_methods"]) + [
        "a_method_that_does_not_exist"
    ]
    assert any("the implementation declares" in error for error in _errors(contract))


def test_the_gate_refuses_a_misreported_method_name() -> None:
    contract = _contract()
    next(row for row in contract["public_method"] if row["method"] == "auto")[
        "reported_method"
    ] = "parameter_shift"
    assert any("reported_method" in error for error in _errors(contract))


def test_the_gate_refuses_a_misreported_exactness() -> None:
    contract = _contract()
    next(row for row in contract["public_method"] if row["method"] == "spsa")[
        "exact"
    ] = True
    assert any("exact=" in error for error in _errors(contract))


def test_the_gate_refuses_a_misreported_step() -> None:
    contract = _contract()
    next(
        row for row in contract["public_method"] if row["method"] == "finite_difference"
    )["step"] = 1e-3
    assert any("records step" in error for error in _errors(contract))


def test_the_gate_refuses_a_declared_by_flag_that_is_wrong() -> None:
    contract = _contract()
    next(row for row in contract["public_method"] if row["method"] == "autograd")[
        "declared_by_any_capability_block"
    ] = True
    assert any(
        "declared_by_any_capability_block" in error for error in _errors(contract)
    )


def test_the_gate_refuses_a_refusal_that_is_no_longer_raised() -> None:
    contract = _contract()
    contract["refusal"] = [
        row for row in contract["refusal"] if row["requested"] != "adjoint"
    ]
    assert any("is not recorded as a refusal" in error for error in _errors(contract))


def test_the_gate_refuses_a_reworded_refusal_message() -> None:
    contract = _contract()
    refusals = {row["requested"]: row for row in contract["refusal"]}
    refusals["adjoint"]["message"] = "no adjoint here"
    assert any("differs from the one raised" in error for error in _errors(contract))


def test_the_gate_refuses_a_refusal_with_the_wrong_exception_class() -> None:
    contract = _contract()
    refusals = {row["requested"]: row for row in contract["refusal"]}
    refusals["adjoint"]["exception"] = "ValidationError"
    errors = _errors(contract)
    assert any("the contract records" in error for error in errors)


def test_the_gate_refuses_an_exception_class_it_cannot_observe() -> None:
    contract = _contract()
    refusals = {row["requested"]: row for row in contract["refusal"]}
    refusals["adjoint"]["exception"] = "ErrorNobodyRaises"
    assert any("cannot observe" in error for error in _errors(contract))


def test_the_gate_refuses_a_refusal_it_does_not_drive() -> None:
    contract = _contract()
    row = copy.deepcopy(contract["refusal"][0])
    row["requested"] = "a_name_this_gate_never_drives"
    contract["refusal"].append(row)
    assert any("does not drive" in error for error in _errors(contract))


def test_the_gate_refuses_a_miscounted_reference_figure() -> None:
    contract = _contract()
    contract["reference"]["declaration_sites"] += 1
    assert any("declaration_sites" in error for error in _errors(contract))


def test_the_gate_refuses_a_misstated_engine_count() -> None:
    contract = _contract()
    contract["reference"]["engine_rows_using_statevector_adjoint"] = 0
    assert any(
        "engine_rows_using_statevector_adjoint" in error for error in _errors(contract)
    )


def test_the_gate_refuses_a_misstated_unrequestable_list() -> None:
    contract = _contract()
    contract["reference"]["declared_route_values_not_requestable"] = []
    assert any(
        "declared_route_values_not_requestable" in error for error in _errors(contract)
    )


def test_the_gate_refuses_a_rule_flag_that_is_not_read() -> None:
    contract = _contract()
    contract["rules"]["a_rule_this_gate_does_not_read"] = True
    assert any("is not read by this gate" in error for error in _errors(contract))


def test_the_gate_refuses_a_rule_that_is_not_contracted() -> None:
    contract = _contract()
    del contract["rules"]["no_silent_fallback"]
    assert any("is not contracted as true" in error for error in _errors(contract))


def test_the_gate_refuses_a_maturity_that_overclaims() -> None:
    contract = _contract()
    contract["maturity"] = "production"
    assert any("development_evidence" in error for error in _errors(contract))


def test_the_gate_refuses_a_misnamed_shared_identifier() -> None:
    contract = _contract()
    contract["shared_identifier"] = "backward_method"
    assert any("shared_identifier" in error for error in _errors(contract))


def test_the_gate_refuses_a_verification_flag_that_is_not_read() -> None:
    contract = _contract()
    contract["verification"]["a_flag_this_gate_does_not_read"] = True
    assert any("is not read by this gate" in error for error in _errors(contract))


def test_the_gate_refuses_a_gate_path_that_is_not_this_gate() -> None:
    contract = _contract()
    contract["verification"]["gate"] = "tools/check_opcode_gradient_exactness.py"
    assert any("not enforced by it" in error for error in _errors(contract))


def test_the_gate_refuses_a_contract_test_that_does_not_exist() -> None:
    contract = _contract()
    contract["verification"]["contract"] = "tests/unit/test_never_written.py"
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
    assert "python tools/check_native_gradients_contract.py" in workflow
    pre_push = (ROOT / "tools/pre_push.py").read_text(encoding="utf-8")
    assert '"tools/check_native_gradients_contract.py"' in pre_push

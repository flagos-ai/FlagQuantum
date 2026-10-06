"""Conformance of the gradient method x execution mode matrix with its own claims.

``tests/test_vector_derivatives.py`` and ``tests/integration/test_gradient_modes.py``
own what ``fq.gradient`` computes. This file owns
``contracts/gradient-methods-contract.toml`` and
``tools/check_gradient_methods_contract.py``, and asks a different question of them --
is the matrix still well formed, and is the gate that re-measures it still enforced?

That split matters because the two artifacts fail differently. The gate answers "is the
matrix still true of the implementation": it rebuilds the reference program and runs all
sixty combinations. This file answers "is the matrix still a matrix": one cell per pair,
every refusal resolvable, every recorded phrase present in the file it names, and the
two vocabularies the implementation's rather than the file's. The final section asks
whether the gate is invoked by CI and by ``tools/pre_push.py`` at all, because a gate
nothing runs is a script.

The middle section is a record of what the matrix refused to predict. Three groups of
cells in the plan's proposal were falsified by measurement -- a per-mode ``adjoint``
row, a distributed-mode column, and a separate noisy mode -- and each of those
falsifications is asserted here so that a later edit cannot quietly reintroduce the
proposal's shape.
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
CONTRACT = ROOT / "contracts" / "gradient-methods-contract.toml"

_GATE_PATH = ROOT / "tools" / "check_gradient_methods_contract.py"
_GATE_SPEC = importlib.util.spec_from_file_location(
    "check_gradient_methods_contract", _GATE_PATH
)
assert _GATE_SPEC is not None and _GATE_SPEC.loader is not None
_GATE = importlib.util.module_from_spec(_GATE_SPEC)
_GATE_SPEC.loader.exec_module(_GATE)


def _contract() -> dict[str, Any]:
    return tomllib.loads(CONTRACT.read_text(encoding="utf-8"))


def _cell(contract: dict[str, Any], key: str, method: str, mode: str) -> dict[str, Any]:
    return next(
        cell
        for cell in contract[key]
        if cell["method"] == method and cell["mode"] == mode
    )


# --------------------------------------------------------------------------------------
# The file is well formed
# --------------------------------------------------------------------------------------


def test_the_matrix_has_exactly_one_cell_per_method_and_mode_pair() -> None:
    contract = _contract()
    expected = {
        (method, mode)
        for method in contract["declared_methods"]
        for mode in contract["serving_modes"]
    }
    for key in ("cells", "noise_cells"):
        seen = [(cell["method"], cell["mode"]) for cell in contract[key]]
        assert sorted(seen) == sorted(expected)
        assert len(seen) == len(set(seen))


def test_the_two_axes_describe_the_same_pairs() -> None:
    """The noise axis is a second measurement of the same grid, not a different one.

    If the two axes ever drifted apart, a combination would be measured against noise in
    one axis and not the other, and the narrower noisy shape would be an artifact of
    which rows somebody happened to write.
    """

    contract = _contract()
    unnoisy = {(cell["method"], cell["mode"]) for cell in contract["cells"]}
    noisy = {(cell["method"], cell["mode"]) for cell in contract["noise_cells"]}
    assert unnoisy == noisy


def test_the_recorded_counts_are_the_counts_in_the_tables() -> None:
    contract = _contract()
    taxonomy = contract["taxonomy"]
    for key, count_key in (
        ("cells", "cell_count"),
        ("noise_cells", "noise_cell_count"),
    ):
        assert taxonomy[count_key] == len(contract[key])
    for key, count_key, verdict in (
        ("cells", "serving_cell_count", "serves"),
        ("cells", "refusing_cell_count", "refuses"),
        ("noise_cells", "noise_serving_cell_count", "serves"),
        ("noise_cells", "noise_refusing_cell_count", "refuses"),
    ):
        assert taxonomy[count_key] == sum(
            1 for cell in contract[key] if cell["verdict"] == verdict
        )


def test_the_noise_axis_is_strictly_narrower_than_the_unnoisy_axis() -> None:
    """The most consequential fact the file carries, asserted rather than described."""

    contract = _contract()
    unnoisy = {
        cell["mode"] for cell in contract["cells"] if cell["verdict"] == "serves"
    }
    noisy = {
        cell["mode"] for cell in contract["noise_cells"] if cell["verdict"] == "serves"
    }
    assert noisy < unnoisy
    assert noisy == {"auto", "density_matrix"}
    assert contract["noise"]["noise_narrows_the_reachable_mode_set"] is True


def test_every_cell_verdict_is_resolvable() -> None:
    contract = _contract()
    codes = [row["code"] for row in contract["refusals"]]
    assert len(codes) == len(set(codes))
    for key in ("cells", "noise_cells"):
        for cell in contract[key]:
            assert cell["verdict"] in {"serves", "refuses"}
            if cell["verdict"] == "serves":
                assert cell["reported_method"] in contract["declared_methods"]
                assert isinstance(cell["exact"], bool)
                assert isinstance(cell["loss_evaluations"], int)
            else:
                assert cell["refusal"] in codes


def test_the_spsa_rows_are_a_range_ordered_by_direction_count() -> None:
    """A statistical method is recorded as a range, and its cost as two per direction."""

    contract = _contract()
    assert contract["rules"]["statistical_method_reports_a_range_not_a_value"] is True
    directions = [row["directions"] for row in contract["accuracy"]["spsa"]]
    assert directions == sorted(directions)
    for row in contract["accuracy"]["spsa"]:
        assert row["loss_evaluations"] == 2 * row["directions"] + 1
        assert row["min_deviation"] <= row["median_deviation"] <= row["max_deviation"]


def test_the_verification_paths_exist() -> None:
    contract = _contract()
    verification = contract["verification"]
    assert (ROOT / verification["contract"]).is_file()
    assert (ROOT / verification["gate"]).is_file()
    for raw in contract["authorization"]:
        assert (ROOT / raw).is_file()


# --------------------------------------------------------------------------------------
# The matrix records measurements, not the plan's predictions
# --------------------------------------------------------------------------------------


def test_adjoint_is_refused_by_name_rather_than_given_a_row() -> None:
    """It never reaches a mode, so a per-mode row could not describe it."""

    contract = _contract()
    assert contract["refused_methods"] == ["adjoint"]
    assert "adjoint" not in contract["declared_methods"]
    for key in ("cells", "noise_cells"):
        assert all(cell["method"] != "adjoint" for cell in contract[key])
    row = next(
        item for item in contract["refusals"] if item["code"] == "no_standalone_adjoint"
    )
    assert row["exception"] == "CapabilityError"
    assert "adjoint" in row["trigger"]


def test_no_cell_uses_a_distributed_mode_name() -> None:
    """Those five names are refused before any gradient code could run."""

    contract = _contract()
    unreachable = set(contract["unreachable_modes"])
    assert unreachable
    assert unreachable.isdisjoint(contract["serving_modes"])
    for key in ("cells", "noise_cells"):
        assert all(cell["mode"] not in unreachable for cell in contract[key])


def test_noise_is_recorded_as_a_run_argument_and_not_as_a_mode() -> None:
    contract = _contract()
    assert contract["taxonomy"]["noise_is_not_a_mode"] is True
    assert contract["taxonomy"]["noise_enters_through_fq_run"] is True
    assert "noise" not in contract["serving_modes"]
    assert contract["noise"]["channel"] == "depolarizing"


def test_the_reported_method_is_the_method_that_ran() -> None:
    """``auto`` is a policy row: it reports what it resolved to, which is not itself."""

    contract = _contract()
    assert contract["taxonomy"]["reported_method_is_the_method_that_ran"] is True
    auto = _cell(contract, "cells", "auto", "statevector")
    assert auto["reported_method"] == contract["resolution"]["graph_present"]
    assert auto["reported_method"] != "auto"


# --------------------------------------------------------------------------------------
# The refusal vocabulary
# --------------------------------------------------------------------------------------


def test_every_refusal_names_an_exception_the_gate_can_observe() -> None:
    contract = _contract()
    for row in contract["refusals"]:
        assert row["exception"] in _GATE.EXCEPTIONS
        assert isinstance(row["message_phrase"], str) and row["message_phrase"]
        assert row["reachable_from"]


def test_every_phrase_occurs_in_the_source_it_names() -> None:
    contract = _contract()
    for row in contract["refusals"]:
        source = (ROOT / row["message_source"]).read_text(encoding="utf-8")
        assert row["message_phrase"] in source, row["code"]


def test_the_vocabulary_covers_both_fail_closed_classes() -> None:
    """A matrix that only recorded ``CapabilityError`` would hide its own guards."""

    contract = _contract()
    classes = {row["exception"] for row in contract["refusals"]}
    assert classes == {"CapabilityError", "ValidationError"}


def test_the_reframing_is_one_measured_asymmetry_with_its_instances() -> None:
    contract = _contract()
    assert len(contract["reframings"]) == 1
    row = contract["reframings"][0]
    assert row["verdict"] == "recorded_not_corrected"
    assert row["outer_exception"] == "CapabilityError"
    assert row["inner_exception"] == "ValidationError"
    assert row["applies_to"] == ["parameter_shift"]
    assert row["instances"]
    for instance in row["instances"]:
        assert instance["measured_refusing_modes"]
        assert instance["inner_phrase"]


# --------------------------------------------------------------------------------------
# The gate refuses a contract that no longer describes the implementation
# --------------------------------------------------------------------------------------
#
# Each mutation patches one clause and requires the gate to name it. The gate is not a
# schema check, so these are the clauses that would otherwise rot silently.


def _errors(contract: dict[str, Any]) -> tuple[str, ...]:
    return _GATE.contract_errors(copy.deepcopy(contract))


def test_the_gate_accepts_the_checked_in_contract() -> None:
    assert _errors(_contract()) == ()


def test_the_gate_refuses_a_dropped_cell() -> None:
    contract = _contract()
    contract["cells"].pop()
    assert any("are missing" in error for error in _errors(contract))


def test_the_gate_refuses_a_phantom_cell() -> None:
    contract = _contract()
    contract["cells"].append(copy.deepcopy(contract["cells"][0]))
    assert any("repeat" in error for error in _errors(contract))


def test_the_gate_refuses_a_restated_cell_count() -> None:
    contract = _contract()
    contract["taxonomy"]["cell_count"] = 29
    assert any("cell_count" in error for error in _errors(contract))


def test_the_gate_refuses_a_cell_that_misreports_which_method_ran() -> None:
    contract = _contract()
    _cell(contract, "cells", "autograd", "statevector")[
        "reported_method"
    ] = "parameter_shift"
    assert any("reports" in error for error in _errors(contract))


def test_the_gate_refuses_a_flipped_verdict() -> None:
    contract = _contract()
    _cell(contract, "cells", "parameter_shift", "stabilizer")["verdict"] = "serves"
    assert any("but measures" in error for error in _errors(contract))


def test_the_gate_refuses_a_noise_verdict_the_noise_axis_contradicts() -> None:
    contract = _contract()
    _cell(contract, "noise_cells", "autograd", "statevector")["verdict"] = "serves"
    assert any("noise_cells" in error for error in _errors(contract))


def test_the_gate_refuses_an_unlisted_refusal_code() -> None:
    contract = _contract()
    drop = next(
        row["code"]
        for row in contract["refusals"]
        if any(cell.get("refusal") == row["code"] for cell in contract["cells"])
    )
    contract["refusals"] = [row for row in contract["refusals"] if row["code"] != drop]
    assert any("unlisted refusal" in error for error in _errors(contract))


def test_the_gate_refuses_a_phrase_that_left_its_source() -> None:
    contract = _contract()
    contract["refusals"][0][
        "message_phrase"
    ] = "this sentence is not in the implementation"
    assert any("is not in" in error for error in _errors(contract))


def test_the_gate_refuses_a_rule_flag_nobody_reads() -> None:
    contract = _contract()
    contract["rules"]["a_flag_no_gate_reads"] = True
    assert any("is not read by this gate" in error for error in _errors(contract))


def test_the_gate_refuses_a_dropped_rule_flag() -> None:
    contract = _contract()
    del contract["rules"]["no_silent_fallback"]
    assert any("no_silent_fallback" in error for error in _errors(contract))


def test_the_gate_refuses_a_claim_that_noise_is_a_mode() -> None:
    contract = _contract()
    contract["taxonomy"]["noise_is_not_a_mode"] = False
    assert any("noise_is_not_a_mode" in error for error in _errors(contract))


def test_the_gate_refuses_a_declared_method_the_implementation_lost() -> None:
    contract = _contract()
    contract["declared_methods"].remove("spsa")
    assert any("declared_methods drifted" in error for error in _errors(contract))


def test_the_gate_refuses_a_mode_the_option_layer_rejects() -> None:
    contract = _contract()
    contract["serving_modes"].append("distributed_mps")
    assert any("serving_modes drifted" in error for error in _errors(contract))


def test_the_gate_refuses_an_unreachable_mode_that_became_reachable() -> None:
    contract = _contract()
    contract["unreachable_modes"].append("statevector")
    assert any("are now reachable" in error for error in _errors(contract))


def test_the_gate_refuses_a_reframing_that_no_longer_holds() -> None:
    contract = _contract()
    contract["reframings"][0]["instances"][0]["measured_refusing_modes"] = [
        "statevector"
    ]
    assert any("reframing" in error for error in _errors(contract))


def test_the_gate_refuses_a_misstated_reference_gradient() -> None:
    contract = _contract()
    contract["reference"]["reference_gradient"][0] += 0.5
    assert any("reference_gradient" in error for error in _errors(contract))


def test_the_gate_refuses_a_misstated_noisy_cross_method_spread() -> None:
    """A spread is a claim about agreement, so the gate re-measures rather than trusts."""

    contract = _contract()
    contract["noise"]["noisy_cross_method_max_spread"] = 1e-30
    assert any("cross_method_max_spread" in error for error in _errors(contract))


def test_the_gate_refuses_a_misstated_spsa_cost() -> None:
    contract = _contract()
    contract["accuracy"]["spsa"][0]["loss_evaluations"] = 4
    assert any("spsa" in error for error in _errors(contract))


def test_the_gate_refuses_a_verification_path_that_does_not_exist() -> None:
    contract = _contract()
    contract["verification"]["gate"] = "tools/check_a_gate_that_never_existed.py"
    assert any("does not exist" in error for error in _errors(contract))


# --------------------------------------------------------------------------------------
# The gate is wired in
# --------------------------------------------------------------------------------------


def test_the_gate_runs_in_ci_and_before_push() -> None:
    """A gate that no workflow invokes is a script, not a gate."""

    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "python tools/check_gradient_methods_contract.py" in workflow
    pre_push = (ROOT / "tools/pre_push.py").read_text(encoding="utf-8")
    assert '"tools/check_gradient_methods_contract.py"' in pre_push

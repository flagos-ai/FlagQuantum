"""Acceptance of the construction layer: one program, not one program per route.

`tests/unit/test_circuit_compose.py` and `tests/unit/test_circuit_adjoint.py` own what
composition and inversion mean. This file owns the acceptance claim in
`contracts/construction-acceptance-contract.toml` and asks it two questions:

1. Does a program `Circuit.compose` built have the same IR as the program a user writes
   by hand for the same placement? That claim is deterministic, so it is asserted
   exactly.
2. Do the execution modes agree on the result? That claim has two halves under one word,
   and keeping them apart is the point of this file. The two *construction routes* agree
   exactly. The *modes* do not: they compute in different arithmetic and land within
   complex64 resolution of each other rather than on each other. Recording the second as
   exact would be the failure this test exists to prevent.

The mode set is read from `ExecutionOptions` rather than restated, so a mode added later
fails the partition test instead of quietly going unmeasured.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import torch

import flagquantum as fq
from flagquantum.errors import CapabilityError, ValidationError
from flagquantum.runtime.execution import run_native
from flagquantum.runtime.options import _MODES, ExecutionOptions
from tools.check_construction_acceptance_contract import contract_errors

#: The exception classes the contract names. `fq` re-exports the error hierarchy
#: selectively, so the classes are read from their owning module rather than from the
#: root namespace, which is where a reader of the contract would look first.
REFUSAL_CLASSES: dict[str, type[BaseException]] = {
    "CapabilityError": CapabilityError,
    "ValidationError": ValidationError,
}

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts" / "construction-acceptance-contract.toml"

#: The two construction routes are the same program in different arithmetic, so they
#: must land on each other exactly. Measured at 0.000e+00 for every mode below.
EXACT = 0.0

#: The modes that do not share the reference mode's arithmetic land within complex64
#: resolution of it. The widest measured spread across density_matrix, mps, and
#: tensor_network is 2.980e-08 and the distributed member reaches 8.429e-08, against a
#: complex64 epsilon of 1.192e-07.
MODE_TOLERANCE = 1e-06


def _load_contract() -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 and older
        import tomli as tomllib

    return tomllib.loads(CONTRACT.read_text(encoding="utf-8"))


def _block(width: int) -> fq.Circuit:
    """The block the contract's placements place: a ladder of `h` then `cnot`."""

    block = fq.Circuit(width)
    for position in range(width):
        block.h(position)
        if position + 1 < width:
            block.cnot(position, position + 1)
    return block


def _hand_built(n_qubits: int, targets: tuple[int, ...]) -> fq.Circuit:
    """The same ladder emitted directly on the placed qubits, written by hand."""

    circuit = fq.Circuit(n_qubits)
    for offset, target in enumerate(targets):
        circuit.h(target)
        if offset + 1 < len(targets):
            circuit.cnot(target, targets[offset + 1])
    return circuit


def _signature(circuit: fq.Circuit) -> tuple[Any, ...]:
    return tuple(
        (
            instruction.name,
            tuple(instruction.wires),
            tuple(sorted((k, repr(v)) for k, v in dict(instruction.params).items())),
        )
        for instruction in circuit.to_ir().instructions
    )


def _contracted_placements() -> list[tuple[int, tuple[int, ...], dict[str, Any]]]:
    contract = _load_contract()
    rows: list[tuple[int, tuple[int, ...], dict[str, Any]]] = []
    for row in contract.get("placement", ()):
        targets = tuple(int(q) for q in row["qubits"])
        rows.append((int(row["n_qubits"]), targets, {"qubits": targets}))
    for row in contract.get("placement_map", ()):
        mapping = {int(k): int(v) for k, v in row["mapping"].items()}
        targets = tuple(mapping[k] for k in sorted(mapping))
        rows.append((int(row["n_qubits"]), targets, {"qubit_map": mapping}))
    return rows


def _contracted_controls() -> list[dict[str, Any]]:
    """The contracted control rows, read as written so a flag cannot be lost in a tuple."""

    return list(_load_contract().get("control_placement", ()))


def _control_routes(
    row: dict[str, Any],
) -> tuple[fq.Circuit, fq.Circuit, tuple[int, ...]]:
    """Build both routes for one contracted row and return them with its control qubits."""

    receiver_n_qubits = int(row["n_qubits"])
    targets = tuple(int(q) for q in row["qubits"])
    controls = tuple(int(q) for q in row["ctrl_qubits"])

    composed = fq.Circuit(receiver_n_qubits)
    composed.compose(_block(len(targets)), qubits=targets)
    composed_route = composed.control(
        int(row["n_controls"]), ctrl_qubits=list(controls)
    )
    hand_route = _hand_built(receiver_n_qubits, targets).control(
        int(row["n_controls"]), ctrl_qubits=list(controls)
    )
    return composed_route, hand_route, controls


def _served_modes() -> list[str]:
    return list(_load_contract().get("modes", {}).get("serving", ()))


def _probability_vector(circuit: fq.Circuit, mode: str) -> torch.Tensor:
    result = fq.run(
        circuit,
        options=ExecutionOptions(mode=mode),
        outputs=fq.OutputRequest("probabilities"),
    )
    return result.probabilities


# --------------------------------------------------------------------------- composed
# The placement claim: the composed program is the hand-built program.


def test_a_composed_program_has_the_hand_built_ir():
    """Every contracted placement is compared against the program written by hand."""

    rows = _contracted_placements()
    assert rows, "the contract lists no placement to measure"
    for n_qubits, targets, placement in rows:
        composed = fq.Circuit(n_qubits)
        composed.compose(_block(len(targets)), **placement)
        hand = _hand_built(n_qubits, targets)

        composed_ir, hand_ir = composed.to_ir(), hand.to_ir()
        assert composed_ir.to_dict() == hand_ir.to_dict(), placement
        assert composed_ir.content_hash == hand_ir.content_hash, placement
        assert composed_ir.n_wires == hand_ir.n_wires == n_qubits, placement
        assert _signature(composed) == _signature(hand), placement
        # The contracted count is the block's own shape, not a copied number.
        assert len(composed_ir.instructions) == 2 * len(targets) - 1, placement


def test_the_placement_forms_the_contract_lists_are_the_ones_measured():
    """`qubits` and `qubit_map` both appear, so neither form is measured alone."""

    forms = {tuple(sorted(placement)) for _, _, placement in _contracted_placements()}
    assert ("qubits",) in forms
    assert ("qubit_map",) in forms


def test_the_placements_in_the_contract_are_distinct_qubits_inside_the_circuit():
    for n_qubits, targets, placement in _contracted_placements():
        assert len(set(targets)) == len(targets), placement
        assert all(0 <= target < n_qubits for target in targets), placement


def test_a_placement_on_the_identity_target_is_measured():
    """The trivial placement is in the contract on purpose: it is where a claim passes
    for the wrong reason if the comparison is written the wrong way round."""

    composed = fq.Circuit(3)
    composed.compose(_block(2), qubits=(0, 1))
    hand = _hand_built(3, (0, 1))
    assert composed.to_ir().to_dict() == hand.to_ir().to_dict()


def test_an_adjoint_program_has_the_hand_built_inverse():
    """`Circuit.adjoint` measured the same way: reverse the order, negate the angles."""

    # The forward program is the placed block followed by three gates. The block is
    # three instructions, not two: a ladder of `h` and `cnot` over width w emits one `h`
    # per position and one `cnot` between neighbours, so `2 * w - 1`.
    forward = _hand_built(3, (0, 1))
    forward.ry(1, 0.3).cnot(1, 2).rz(2, -0.4)

    # The inverse written by hand: the same instructions, reversed, angles negated.
    hand = fq.Circuit(3)
    hand.rz(2, 0.4).cnot(1, 2).ry(1, -0.3)
    hand.h(1).cnot(0, 1).h(0)

    inverse = forward.adjoint()
    assert _signature(inverse) == _signature(hand)
    assert inverse.to_ir().to_dict() == hand.to_ir().to_dict()
    assert inverse.to_ir().content_hash == hand.to_ir().content_hash
    assert inverse.to_ir().n_wires == forward.to_ir().n_wires


def test_the_adjoint_of_a_composed_program_is_the_adjoint_of_the_hand_built_program():
    composed = fq.Circuit(3)
    composed.compose(_block(2), qubits=(0, 1))
    hand = _hand_built(3, (0, 1))
    assert composed.adjoint().to_ir().to_dict() == hand.adjoint().to_ir().to_dict()


# --------------------------------------------------------------------------- control
# The third construction member, measured on the same claim and not on its semantics.
# What `control` means and every way it refuses belong to the composition contract; the
# question here is narrower -- whether the construction layer produces one program.


def test_a_controlled_program_has_the_hand_built_ir():
    """`compose` then `control`, against the hand-built placement then `control`.

    The two routes would differ if `control` read anything about its receiver other than
    the receiver's instructions, so this is the measurement that says `compose` leaves no
    trace `control` can see and a user cannot reproduce by hand.
    """

    rows = _contracted_controls()
    assert rows, "the contract lists no control placement to measure"
    for row in rows:
        composed_route, hand_route, ctrl_qubits = _control_routes(row)
        label = (row["qubits"], row["n_controls"], row["ctrl_qubits"])

        composed_ir, hand_ir = composed_route.to_ir(), hand_route.to_ir()
        assert composed_ir.to_dict() == hand_ir.to_dict(), label
        assert composed_ir.content_hash == hand_ir.content_hash, label
        assert hand_ir.n_wires == composed_ir.n_wires, label
        assert _signature(composed_route) == _signature(hand_route), label

        # The width comes from the controls, which are added rather than taken from the
        # receiver, so it is one past the highest control qubit.
        assert composed_ir.n_wires == max(ctrl_qubits) + 1, label
        # And something was actually controlled: a route that returned its receiver would
        # still agree with a twin that also returned its receiver.
        receiver = _hand_built(int(row["n_qubits"]), tuple(row["qubits"])).to_ir()
        assert len(composed_ir.instructions) > len(receiver.instructions), label


def test_the_control_placements_in_the_contract_are_well_formed():
    """Controls must add qubits, so every one of them lies outside the receiver."""

    for row in _contracted_controls():
        receiver_n_qubits = int(row["n_qubits"])
        targets = tuple(int(q) for q in row["qubits"])
        controls = tuple(int(q) for q in row["ctrl_qubits"])
        assert len(set(targets)) == len(targets), row
        assert all(0 <= target < receiver_n_qubits for target in targets), row
        assert len(controls) == int(row["n_controls"]), row
        assert len(set(controls)) == len(controls), row
        assert all(qubit >= receiver_n_qubits for qubit in controls), row


def test_the_controlled_program_is_one_program_in_every_serving_mode():
    """The same claim once a mode serves it, on the row the contract pins it to.

    The contract does not use the widest row here, and says why: the two-control ladder
    is 57 instructions and reaches 8.941e-07 of spread, which is close enough to the
    declared 10 * eps(complex64) bound that asserting on it would make this test a
    statement about the platform's BLAS rather than about the construction layer.
    """

    flagged = [row for row in _contracted_controls() if row.get("mode_agreement")]
    assert flagged, "the contract marks no row for the mode half"
    # The contract's own reason for marking the rows it marks: a multi-control expansion
    # is the one whose spread approaches the declared bound, so the half that asserts
    # against that bound must not be measured on it.
    assert all(int(row["n_controls"]) == 1 for row in flagged)

    for row in flagged:
        composed_route, hand_route, ctrl_qubits = _control_routes(row)
        baseline = _probability_vector(composed_route, "statevector")
        for mode in _served_modes():
            left = _probability_vector(composed_route, mode)
            right = _probability_vector(hand_route, mode)
            # The two routes are the same program, so they agree exactly in every mode,
            # not merely within the mode tolerance.
            assert torch.equal(left, right), (mode, ctrl_qubits)
            spread = float((left - baseline).abs().max())
            assert spread <= MODE_TOLERANCE, (mode, ctrl_qubits, spread)


def test_a_controlled_program_is_refused_by_the_refused_mode_the_same_way():
    """The refusal is about the requested output, not about the gates in the program."""

    contract = _load_contract()["modes"]
    error = REFUSAL_CLASSES[contract["refusal_class"]]
    composed = fq.Circuit(3)
    composed.compose(_block(2), qubits=(2, 0))
    controlled = composed.control(1, ctrl_qubits=3)
    for mode in contract["refused"]:
        with pytest.raises(error, match=contract["refusal_phrase"]):
            _probability_vector(controlled, mode)


# ----------------------------------------------------------------------------- modes
# The result claim, in its two halves.


def test_every_mode_is_measured_exactly_once():
    """The contract's partition must equal the mode set the implementation offers."""

    contract = _load_contract()["modes"]
    serving, refused = set(contract["serving"]), set(contract["refused"])
    assert not serving & refused
    assert serving | refused == set(_MODES)


def test_a_mode_the_contract_calls_serving_answers_the_requested_output():
    circuit = _hand_built(3, (0, 1))
    for mode in _served_modes():
        vector = _probability_vector(circuit, mode)
        assert vector.shape == (1, 8), mode
        assert torch.allclose(vector.sum(dim=-1), torch.ones(1), atol=1e-6), mode


def test_a_mode_the_contract_calls_refused_fails_closed():
    contract = _load_contract()["modes"]
    error = REFUSAL_CLASSES.get(contract["refusal_class"])
    assert error is not None, contract["refusal_class"]
    circuit = _hand_built(3, (0, 1))
    for mode in contract["refused"]:
        with pytest.raises(error, match=contract["refusal_phrase"]):
            _probability_vector(circuit, mode)


def test_the_two_construction_routes_agree_exactly_in_every_serving_mode():
    """Half one: a composed program and its hand-built twin are the same program."""

    composed = fq.Circuit(3)
    composed.compose(_block(2), qubits=(0, 1))
    composed.ry(1, 0.3).cnot(1, 2).rz(2, -0.4)
    hand = _hand_built(3, (0, 1))
    hand.ry(1, 0.3).cnot(1, 2).rz(2, -0.4)

    assert composed.to_ir().to_dict() == hand.to_ir().to_dict()

    for mode in _served_modes():
        left = _probability_vector(composed, mode)
        right = _probability_vector(hand, mode)
        assert torch.equal(left, right), mode
        assert float((left - right).abs().max()) == EXACT, mode


def test_each_serving_mode_agrees_with_the_reference_within_the_declared_tolerance():
    """Half two: the modes agree within tolerance, and the tolerance is declared.

    This is the half that must not be written as exact. A test that asserted equality
    here would fail on mps and tensor_network for reasons that are arithmetic, not
    construction -- and the honest response to that is a declared tolerance, not a
    silently widened comparison.
    """

    contract = _load_contract()["result_identity"]
    reference = contract["reference_mode"]
    assert reference in _served_modes()

    composed = fq.Circuit(3)
    composed.compose(_block(2), qubits=(0, 1))
    composed.ry(1, 0.3).cnot(1, 2).rz(2, -0.4)

    baseline = _probability_vector(composed, reference)
    for mode in _served_modes():
        spread = float((_probability_vector(composed, mode) - baseline).abs().max())
        assert spread <= MODE_TOLERANCE, (mode, spread)
    # The reference agrees with itself exactly, so the tolerance above is not vacuous.
    assert float((baseline - baseline).abs().max()) == EXACT


def test_the_measured_spread_is_below_complex64_resolution():
    """The tolerance has a reason, and the reason is measurable rather than asserted."""

    composed = fq.Circuit(3)
    composed.compose(_block(2), qubits=(0, 1))
    composed.ry(1, 0.3).cnot(1, 2).rz(2, -0.4)
    baseline = _probability_vector(composed, "statevector")
    for mode in ("density_matrix", "mps", "tensor_network"):
        spread = float((_probability_vector(composed, mode) - baseline).abs().max())
        assert spread == 0.0 or spread < float(torch.finfo(torch.complex64).eps) * 10


# ----------------------------------------------------------------------- distributed
# The fourth mode, reached the way it is actually reachable.


def test_the_distributed_member_is_not_an_execution_options_mode():
    """The plan row named it as one. It is not, and the refusal is recorded."""

    contract = _load_contract()["distributed_member"]
    assert contract["reachable_through_execution_options"] is False
    error = REFUSAL_CLASSES[contract["execution_options_refusal_class"]]
    with pytest.raises(error, match=contract["execution_options_refusal_phrase"]):
        ExecutionOptions(mode="distributed_statevector")
    assert "distributed_statevector" not in _MODES


def test_neither_root_entry_point_accepts_a_world_size():
    """The member cannot be selected from `fq.run` or `fq.plan`, so the route is named."""

    import inspect

    contract = _load_contract()["distributed_member"]
    assert contract["root_run_accepts_world_size"] is False
    assert contract["root_plan_accepts_world_size"] is False
    assert "world_size" not in inspect.signature(fq.run).parameters
    assert "world_size" not in inspect.signature(fq.plan).parameters
    assert contract["route"] == "flagquantum.runtime.execution.run_native"


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_the_distributed_member_serves_the_composed_and_hand_built_program_alike(
    world_size,
):
    contract = _load_contract()["distributed_member"]
    composed = fq.Circuit(3)
    composed.compose(_block(2), qubits=(0, 1))
    composed.ry(1, 0.3).cnot(1, 2).rz(2, -0.4)
    hand = _hand_built(3, (0, 1))
    hand.ry(1, 0.3).cnot(1, 2).rz(2, -0.4)

    left, left_plan = run_native(
        composed,
        mode=contract["accepted_mode_argument"],
        return_plan=True,
        world_size=world_size,
    )
    right, _ = run_native(
        hand,
        mode=contract["accepted_mode_argument"],
        return_plan=True,
        world_size=world_size,
    )
    assert left_plan.is_distributed is True
    assert left_plan.world_size == world_size
    assert torch.equal(left.state, right.state)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_the_distributed_member_agrees_with_the_reference_within_the_declared_tolerance(
    world_size,
):
    composed = fq.Circuit(3)
    composed.compose(_block(2), qubits=(0, 1))
    composed.ry(1, 0.3).cnot(1, 2).rz(2, -0.4)

    baseline = fq.run(composed, options=ExecutionOptions(mode="statevector")).state
    result, _ = run_native(
        composed,
        mode="distributed_statevector",
        return_plan=True,
        world_size=world_size,
    )
    assert float((result.state - baseline).abs().max()) <= MODE_TOLERANCE


# ------------------------------------------------------------------ construction layer
# The census, which is what fails when the layer grows.


def test_the_construction_layer_covers_exactly_what_the_contract_covers():
    contract = _load_contract()["subject"]
    instance = fq.Circuit(2)
    for member in contract["covered"]:
        assert hasattr(instance, member.rsplit(".", 1)[-1]), member
    for member in contract["not_covered"]:
        assert not hasattr(instance, member.rsplit(".", 1)[-1]), member
    for member in _load_contract()["census"]["also_measured_absent"]:
        assert not hasattr(instance, member.rsplit(".", 1)[-1]), member


def test_the_census_is_a_partition_of_the_declared_family():
    """The gate agrees with the contract as written, so the mutations below start clean."""

    assert contract_errors(_load_contract()) == []


def test_the_census_rejects_a_member_deleted_from_the_family():
    """Deleting a row and its requirement together must not pass.

    This is the route the census could not see on its own. It re-checks the rows the
    contract still has, so removing a row and the requirement that named it left both
    the gate and this whole file green: the test function was still here, it had just
    stopped being required by anything. Measured on this slice before the partition
    check existed -- gate rc=0 and `26 passed`.
    """

    contract = copy.deepcopy(_load_contract())
    contract["subject"]["covered"] = [
        member
        for member in contract["subject"]["covered"]
        if member != "Circuit.control"
    ]
    del contract["verification"]["requirements"]["Circuit.control"]

    errors = contract_errors(contract)
    assert any(
        "is not the family" in error and "Circuit.control" in error for error in errors
    ), errors


def test_the_census_rejects_a_spelling_that_no_row_measures():
    """A declared operator spelling must be read by some row, or it is a blind spot."""

    contract = copy.deepcopy(_load_contract())
    contract["census"]["also_measured_absent"] = []

    errors = contract_errors(contract)
    assert any("operator_spellings declares" in error for error in errors), errors


def test_acceptance_added_no_public_surface():
    contract = _load_contract()
    assert contract["root_export_effect"] == "none"
    assert contract["ir_version_effect"] == "none"
    assert contract["authorization_required"] is False
    assert len(fq.__all__) == contract["measured_root_export_count"]
    from flagquantum.core.ir import IR_VERSION

    assert contract["measured_ir_version"] == IR_VERSION

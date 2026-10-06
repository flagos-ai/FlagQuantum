"""Conformance of the construction-time composition contract with the implementation.

`tests/unit/test_circuit_compose.py` and `tests/unit/test_circuit_adjoint.py` own the
semantics: which instruction each opcode becomes, which placement is refused, and what
the result executes to. This file owns the contract in
`contracts/circuit-composition-contract.toml` and asks one question of it -- is every
sentence in it still true of the code?

That is a different measurement. The contract is the repository's own list of the
refusal vocabulary, so it is checked against the implementation rather than trusted: a
phrase that no longer occurs in its source, a refusal whose exception class changed, a
declared rule name that the operator schema does not define, an operation that the
contract calls absent but that someone added, or an opcode census that no longer
supports a `reachable = false` row all fail here.
"""

from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import math
import pathlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.primitives import qft
from flagquantum.core import (
    ADJOINT_RULES,
    CONTROL_RULES,
    MAX_LADDER_LEVEL,
    OPERATOR_SCHEMAS,
)
from flagquantum.core.ir import IR_VERSION, Instruction, IRValidationError
from flagquantum.errors import CapabilityError, ValidationError

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 and older
    import tomli as tomllib

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "contracts" / "circuit-composition-contract.toml"

#: The exception class each contracted `exception` name denotes. A name absent here is
#: a contracted class this test does not know how to observe, which is a failure.
_EXCEPTIONS: dict[str, type[BaseException]] = {
    "TypeError": TypeError,
    "ValidationError": ValidationError,
    "CapabilityError": CapabilityError,
}


def _contract() -> dict[str, Any]:
    return tomllib.loads(CONTRACT.read_text(encoding="utf-8"))


def _refusals() -> list[dict[str, Any]]:
    refusals: list[dict[str, Any]] = _contract()["refusals"]
    return refusals


def _composable(width: int = 2) -> fq.Circuit:
    return fq.Circuit(width).h(0)


def _with_instruction(instruction: Instruction) -> fq.Circuit:
    """Return a two-qubit circuit holding one hand-built instruction."""

    circuit = fq.Circuit(2)
    circuit._instructions.append(instruction)
    return circuit


#: One call per contracted refusal that a user can reach. The keys must be exactly the
#: `reachable = true` codes; a row without a trigger would otherwise be asserted only by
#: its source text, and a trigger without a row would be an undocumented refusal.
_TRIGGERS: dict[str, Callable[[], object]] = {
    "both_placement_arguments": lambda: fq.Circuit(3).compose(
        _composable(), qubits=(0, 1), qubit_map={0: 0, 1: 1}
    ),
    "program_not_composable": lambda: fq.Circuit(3).compose(5),
    "qubit_label_not_an_integer": lambda: fq.Circuit(3).compose(
        _composable(), qubits=(1.0, 2)
    ),
    "placement_not_a_mapping": lambda: fq.Circuit(3).compose(
        _composable(), qubit_map=[1, 2]
    ),
    "batch_size_mismatch": lambda: fq.Circuit(2, bsz=3).compose(fq.Circuit(2)),
    "placement_incomplete": lambda: fq.Circuit(3).compose(_composable(), qubits=(1,)),
    "placement_map_incomplete": lambda: fq.Circuit(3).compose(
        _composable(), qubit_map={0: 0}
    ),
    "placement_map_unknown_source": lambda: fq.Circuit(3).compose(
        _composable(), qubit_map={0: 0, 1: 1, 7: 2}
    ),
    "target_qubit_repeated": lambda: fq.Circuit(3).compose(
        _composable(), qubits=(1, 1)
    ),
    "target_qubit_negative": lambda: fq.Circuit(3).compose(
        _composable(), qubits=(-1, 0)
    ),
    "target_qubit_out_of_range": lambda: fq.Circuit(3).compose(
        _composable(), qubits=(2, 3)
    ),
    "inverse_of_channel": lambda: fq.Circuit(1).depolarizing(0, 0.1).adjoint(),
    # `reset` with this metadata is what the dynamic runtime writes for a mid-circuit
    # reset, so the trigger is the real instruction rather than a synthetic one.
    "inverse_of_dynamic_operation": lambda: _with_instruction(
        Instruction("reset", (1,), metadata={"is_dynamic": True})
    ).adjoint(),
    "inverse_of_conditioned_gate": lambda: _with_instruction(
        Instruction("x", (1,), metadata={"conditions": (("c", 1),)})
    ).adjoint(),
    "inverse_of_undeclared_rule": lambda: _with_instruction(
        Instruction("bit_flip", (1,), params={"probability": 0.1})
    ).adjoint(),
    # A custom operation may carry any matrix, and one recorded as a nested sequence
    # rather than a tensor exposes no conjugate transpose.
    "inverse_of_matrix_without_conjugate_transpose": lambda: _with_instruction(
        Instruction("custom", (1,), matrix=((1.0, 0.0), (0.0, 1.0)))
    ).adjoint(),
    # `n_controls` is read before the control qubits are, so the count refusals are
    # reached without a valid control set and the label refusal is reached without a
    # valid count.
    "control_count_not_an_integer": lambda: fq.Circuit(2).control(
        2.5, ctrl_qubits=(2, 3)
    ),
    "control_count_not_positive": lambda: fq.Circuit(2).control(0, ctrl_qubits=()),
    "control_label_not_an_integer": lambda: fq.Circuit(2).control(
        1, ctrl_qubits=(2.0,)
    ),
    "control_count_mismatch": lambda: fq.Circuit(2).control(2, ctrl_qubits=(2,)),
    "control_qubit_repeated": lambda: fq.Circuit(2).control(2, ctrl_qubits=(2, 2)),
    "control_qubit_negative": lambda: fq.Circuit(2).control(1, ctrl_qubits=(-1,)),
    "control_qubit_inside_receiver": lambda: fq.Circuit(2).control(1, ctrl_qubits=(1,)),
    "control_of_channel": lambda: fq.Circuit(1)
    .depolarizing(0, 0.1)
    .control(1, ctrl_qubits=(1,)),
    # The same two hand-built instructions the adjoint triggers use, so the two surfaces
    # are refused for the same reason on the same input rather than on two similar ones.
    "control_of_dynamic_operation": lambda: _with_instruction(
        Instruction("reset", (1,), metadata={"is_dynamic": True})
    ).control(1, ctrl_qubits=(2,)),
    "control_of_conditioned_gate": lambda: _with_instruction(
        Instruction("x", (1,), metadata={"conditions": (("c", 1),)})
    ).control(1, ctrl_qubits=(2,)),
    "control_of_matrix": lambda: fq.Circuit(1)
    .any(0, unitary=torch.eye(2, dtype=torch.complex128))
    .control(1, ctrl_qubits=(1,)),
    "control_of_execution_metadata": lambda: _with_instruction(
        Instruction("h", (1,), metadata={"origin": "test"})
    ).control(1, ctrl_qubits=(2,)),
    # A noise-channel opcode recorded without its `is_channel` metadata reaches the
    # declaration itself rather than the channel branch, which is the one reachable
    # spelling of an opcode that declares no controlled form.
    "control_of_opcode_without_a_controlled_form": lambda: _with_instruction(
        Instruction("bit_flip", (1,), params={"probability": 0.1})
    ).control(1, ctrl_qubits=(2,)),
}


def test_contract_header_names_the_surface_and_its_authorization() -> None:
    contract = _contract()

    assert contract["schema"] == "flagquantum_circuit_composition_contract_v1"
    assert contract["maturity"] == "development_evidence"
    assert contract["surface"] == "construction_time_program_composition"
    assert contract["ir_version_effect"] == "none"
    assert contract["root_export_effect"] == "none"
    for field in (
        "implementation",
        "placement_implementation",
        "adjoint_rule_source",
        "control_rule_source",
        "control_expansion_implementation",
    ):
        assert (ROOT / contract[field]).is_file(), field
    for proposal in contract["authorization"]:
        assert (ROOT / proposal).is_file(), proposal


def test_the_contracted_operations_exist_and_the_absent_ones_do_not() -> None:
    scope = _contract()["scope"]

    for name in scope["provided"]:
        assert callable(getattr(fq.Circuit, name.split(".")[-1])), name
    for name in scope["not_provided"]:
        # The absence is contracted, so adding one of these is a contract change and
        # not an implementation detail.
        assert not hasattr(fq.Circuit, name.split(".")[-1]), name
    assert "no approved API change proposal" in scope["not_provided_reason"]


def test_placement_arguments_are_the_contracted_signature() -> None:
    placement = _contract()["placement"]
    signature = inspect.signature(fq.Circuit.compose)

    assert list(signature.parameters) == [
        "self",
        placement["program_argument"],
        *placement["arguments"],
    ]
    assert (
        signature.parameters[placement["program_argument"]].kind
        is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    assert all(
        signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in placement["arguments"]
    )
    assert all(
        signature.parameters[name].default is None for name in placement["arguments"]
    )
    assert placement["arguments_are_mutually_exclusive"] is True
    assert placement["mutates_receiver"] is True
    assert placement["returns"] == "receiver"


def test_the_placement_flags_hold_on_a_measured_composition() -> None:
    """Every `[placement]` flag is the user-visible result of one measurement.

    A flag no test reads is a comment, so each one is checked by composing a two-qubit
    program whose instructions are distinguishable by qubit and by parameter, and
    reading the emitted instructions back.
    """

    placement = _contract()["placement"]
    block = fq.Circuit(2).h(0).cnot(1, 0).rz(0, 0.7)
    receiver = fq.Circuit(4).x(2)

    # `existing_instructions_preserved`, `source_order_preserved` and
    # `occupied_target_qubits_are_legal`, in one measurement: qubit 2 already carried
    # `x`, that `x` is still first, the source follows in its own order, and the
    # source's own qubit 0 lands on the already-occupied qubit 2 rather than replacing
    # anything. The source is left untouched, which is why it can be composed again.
    receiver.compose(block, qubits=(2, 0))
    emitted = [(item.name, item.wires) for item in receiver.to_ir().instructions]
    assert emitted == [
        ("x", (2,)),
        ("h", (2,)),
        ("cx", (0, 2)),
        ("rz", (2,)),
    ]
    assert placement["occupied_target_qubits_are_legal"] is True
    assert placement["existing_instructions_preserved"] is True
    assert [item.wires for item in block.to_ir().instructions] == [(0,), (1, 0), (0,)]

    # `default_placement`: with neither argument the source lands on the low qubits.
    default = fq.Circuit(4).compose(fq.Circuit(2).h(0).cnot(0, 1))
    assert [(item.name, item.wires) for item in default.to_ir().instructions] == [
        ("h", (0,)),
        ("cx", (0, 1)),
    ]
    assert placement["default_placement"] == "identity"

    # `placement_is_total_over_source_width` is asserted through its refusal in the
    # parametrised refusal test; here it fixes that a complete mapping is accepted.
    assert placement["placement_is_total_over_source_width"] is True

    # `single_label_allowed_only_for_width_one`: a bare label works for width one and is
    # rejected by arity for width two, so it is not a general shorthand.
    assert fq.Circuit(4).compose(fq.Circuit(1).h(0), qubits=2).to_ir().instructions[
        0
    ].wires == (2,)
    with pytest.raises(ValidationError, match="qubits must name all"):
        fq.Circuit(4).compose(block, qubits=2)
    assert placement["single_label_allowed_only_for_width_one"] is True

    # `instruction_payload_preserved`: `rz` keeps its angle and a custom instruction
    # keeps its matrix and metadata, so composition is a rewrite and not a re-creation.
    custom = fq.Circuit(1).any(0, unitary=torch.eye(2, dtype=torch.complex128))
    custom._instructions[-1].metadata["origin"] = "test"
    carried = fq.Circuit(2).compose(fq.Circuit(1).rz(0, 0.25), qubits=1)
    carried.compose(custom, qubits=0)
    angle, matrix_instruction = carried.to_ir().instructions
    assert angle.params == {"theta": 0.25}
    assert torch.equal(matrix_instruction.matrix, torch.eye(2, dtype=torch.complex128))
    assert matrix_instruction.metadata == {"origin": "test"}
    assert placement["instruction_payload_preserved"] == [
        "name",
        "params",
        "matrix",
        "metadata",
    ]

    # `batch_size_must_match` is asserted through its refusal; a matching batch is
    # carried through rather than silently flattened.
    batched = fq.Circuit(2, bsz=3).compose(fq.Circuit(2, bsz=3).h(0))
    assert batched.bsz == 3
    assert placement["batch_size_must_match"] is True

    # `mutates_receiver` and `returns`: the receiver is the result, so the two are the
    # same object and the source is left alone.
    receiver2 = fq.Circuit(2)
    assert receiver2.compose(block) is receiver2
    assert placement["mutates_receiver"] is True
    assert placement["returns"] == "receiver"
    assert len(block.to_ir().instructions) == 3


def _instruction_fields(
    circuit: fq.Circuit,
) -> list[tuple[str, tuple[int, ...], dict[str, Any]]]:
    """Return the three fields a composed instruction must carry unchanged."""

    return [
        (item.name, item.wires, dict(item.params))
        for item in circuit.to_ir().instructions
    ]


def test_compose_expands_to_the_hand_built_program() -> None:
    """Expansion, instruction by instruction: the composed program is the hand-built one.

    `expansion_tests` in the contract names this test, and
    `tools/check_circuit_composition_contract.py` refuses a name that is not here. The
    measurement is exact equality of the emitted instruction lists, not a state
    comparison: two different programs can reach the same state, and the contract claims
    the emitted program.
    """

    block = fq.Circuit(2).h(0).cnot(0, 1).rz(1, 0.3)

    # A non-identity placement into a receiver that already carries an instruction, so
    # both the remapping and the append are inside one comparison.
    composed = fq.Circuit(4).x(2).compose(block, qubits=(2, 0))
    hand_built = fq.Circuit(4).x(2).h(2).cnot(2, 0).rz(0, 0.3)
    assert _instruction_fields(composed) == _instruction_fields(hand_built)
    assert composed.to_ir().instructions == hand_built.to_ir().instructions
    assert torch.allclose(composed.state(), hand_built.state())

    # The same comparison with the default placement, which is the identity map.
    default = fq.Circuit(2).compose(block)
    assert _instruction_fields(default) == _instruction_fields(block)
    assert default.to_ir().instructions == block.to_ir().instructions

    # The exit condition of this wave, with a library program instead of a hand one. The
    # plan writes `fq.gates.qft(3)`; that path does not exist, and the shipped program is
    # `flagquantum.algorithms.primitives.qft`. The expansion is written out by hand here
    # so the comparison does not run the same code it is checking.
    library = fq.Circuit(5).h(0).compose(qft(3), qubits=(1, 2, 3))
    expected = fq.Circuit(5).h(0)
    expected.h(1)
    expected.cphase(2, 1, theta=math.pi / 2)
    expected.cphase(3, 1, theta=math.pi / 4)
    expected.h(2)
    expected.cphase(3, 2, theta=math.pi / 2)
    expected.h(3)
    expected.swap(1, 3)
    assert _instruction_fields(library) == _instruction_fields(expected)
    assert library.to_ir().instructions == expected.to_ir().instructions
    assert torch.allclose(library.state(), expected.state())


def test_adjoint_expands_to_the_hand_built_inverse() -> None:
    """Expansion, instruction by instruction: the inverse is the reversed hand-built one.

    The contract also claims `returns = "new_circuit"` and
    `receiver_is_mutated = false`, so the comparison is made against a receiver that is
    read again afterwards rather than against the object the method returned.
    """

    block = fq.Circuit(2).h(0).cnot(0, 1).rz(1, 0.3)
    forward = _instruction_fields(block)

    inverse = block.adjoint()
    hand_built = fq.Circuit(2).rz(1, -0.3).cnot(0, 1).h(0)
    assert _instruction_fields(inverse) == _instruction_fields(hand_built)
    assert inverse.to_ir().instructions == hand_built.to_ir().instructions
    assert _instruction_fields(block) == forward

    # The inverse undoes the program: the two together are the identity on the state, in
    # the composed order `adjoint` then `block`. This is the numeric half of the same
    # claim and it is measured through `compose`, so it also covers placement.
    circuit = fq.Circuit(3, dtype=torch.complex128)
    circuit.compose(block.adjoint(), qubits=(0, 1))
    circuit.compose(block, qubits=(0, 1))
    initial = fq.Circuit(3, dtype=torch.complex128).state().detach()
    assert len(circuit.to_ir().instructions) == 6
    assert torch.allclose(circuit.state().detach(), initial, atol=1e-14)


def test_declared_adjoint_rules_are_the_operator_schema_vocabulary() -> None:
    adjoint = _contract()["adjoint"]

    # The vocabulary has one owner. Restating it in the contract is only safe while
    # this equality holds.
    assert list(ADJOINT_RULES) == adjoint["declared_opcode_rules"]
    assert adjoint["receiver_is_mutated"] is False
    assert adjoint["instruction_order"] == "reversed"
    assert adjoint["instruction_qubits"] == "unchanged"
    signature = inspect.signature(fq.Circuit.adjoint)
    assert list(signature.parameters) == ["self"]
    # The contract names the preserved attributes, so a rename must reach it. Only the
    # current name is contracted: `n_wires` is a deprecated alias and is not user-facing.
    assert "n_wires" not in adjoint["preserves"]
    for attribute in adjoint["preserves"]:
        assert hasattr(fq.Circuit(1), attribute), attribute


def test_the_adjoint_flags_hold_on_a_measured_inversion() -> None:
    """Every `[adjoint]` flag is the user-visible result of one measurement."""

    adjoint = _contract()["adjoint"]
    block = fq.Circuit(2).h(0).cnot(0, 1).rz(1, 0.7)
    forward = [(item.name, item.wires) for item in block.to_ir().instructions]

    inverse = block.adjoint()
    assert [(item.name, item.wires) for item in inverse.to_ir().instructions] == [
        ("rz", (1,)),
        ("cx", (0, 1)),
        ("h", (0,)),
    ]

    # `instruction_order` and `instruction_qubits`: the order is reversed and no qubit
    # is relabelled, so the inverse of a block acts on the same qubits it acted on.
    assert [wires for _, wires in forward] == [(0,), (0, 1), (1,)]
    assert [wires for _, wires in [(n, w) for n, w in forward]][::-1] == [
        (1,),
        (0, 1),
        (0,),
    ]
    assert adjoint["instruction_order"] == "reversed"
    assert adjoint["instruction_qubits"] == "unchanged"

    # `instruction_order_within_gate` is about the order of a gate's own qubits, which
    # is not reversed: only the order of the instructions is.
    assert inverse.to_ir().instructions[1].wires == (0, 1)
    assert adjoint["instruction_order_within_gate"] == "unchanged"

    # `receiver_is_mutated` and `returns`: the source stays usable, so a reusable block
    # can be inverted and appended again.
    assert [(item.name, item.wires) for item in block.to_ir().instructions] == forward
    assert inverse is not block
    assert adjoint["receiver_is_mutated"] is False
    assert adjoint["returns"] == "new_circuit"

    # `preserves`: the inverse is the same program shape, so it executes the same way.
    # The list is closed, so both dropping an entry and claiming one that is not kept
    # fail here: the attribute names are read from the contract, not restated.
    assert adjoint["preserves"] == ["n_qubits", "bsz", "device", "dtype"]
    shaped = fq.Circuit(3, bsz=2, device="cpu", dtype=torch.complex64).h(1)
    inverted = shaped.adjoint()
    assert inverted.to_ir().instructions != ()
    for attribute in adjoint["preserves"]:
        assert getattr(inverted, attribute) == getattr(shaped, attribute), attribute

    # `empty_circuit_result`: inverting nothing is nothing, and the width survives.
    empty = fq.Circuit(2).adjoint()
    assert empty.to_ir().instructions == ()
    assert empty.n_qubits == 2
    assert adjoint["empty_circuit_result"] == "empty_circuit"

    # `matrix_route_precedes_opcode_rule` and `matrix_route`: an opcode that is its own
    # inverse keeps its name and is inverted through the matrix it carries, because that
    # matrix is what executes. The rule for `h` would have copied the name forward with
    # the matrix unchanged, which is the wrong inverse here.
    angle = 0.7
    matrix = torch.tensor(
        [
            [complex(math.cos(-angle / 2), math.sin(-angle / 2)), 0j],
            [0j, complex(math.cos(angle / 2), math.sin(angle / 2))],
        ],
        dtype=torch.complex128,
    )
    instruction = _with_instruction(Instruction("h", (1,), matrix=matrix))
    conjugated = instruction.adjoint().to_ir().instructions[0]
    assert conjugated.name == "h"
    assert torch.allclose(conjugated.matrix, matrix.mH)
    assert not torch.allclose(conjugated.matrix, matrix)
    assert adjoint["matrix_route_precedes_opcode_rule"] is True
    assert adjoint["matrix_route"] == "conjugate_transpose"


def test_declared_control_rules_are_the_operator_schema_vocabulary() -> None:
    control = _contract()["control"]

    # The list is read from the implementation rather than restated here, so a rule that
    # is added to the schema but not to the contract fails in both directions.
    assert list(control["declared_opcode_rules"]) == list(CONTROL_RULES)
    assert control["absence_rule"] in CONTROL_RULES
    assert _contract()["control_rule_source"] == "flagquantum/core/operator_schema.py"
    assert list(_contract()["scope"]["provided"])[-1] == "Circuit.control"
    assert "Circuit.control" not in _contract()["scope"]["not_provided"]

    # Every declared rule is spoken by at least one opcode, and the only rule that is an
    # absence rather than a form is the one the four channels declare.
    used = {schema.control for schema in OPERATOR_SCHEMAS.values()}
    assert set(CONTROL_RULES) <= used
    undeclared = sorted(
        name
        for name, schema in OPERATOR_SCHEMAS.items()
        if schema.opcode == name and schema.control == control["absence_rule"]
    )
    assert all(OPERATOR_SCHEMAS[name].semantic_kind == "channel" for name in undeclared)


def test_the_control_flags_hold_on_a_measured_control() -> None:
    """Every `[control]` flag is the user-visible result of one measurement."""

    contract = _contract()
    control = contract["control"]
    block = fq.Circuit(2).h(0).cnot(0, 1)
    forward = [(item.name, item.wires) for item in block.to_ir().instructions]

    # `receiver_is_mutated` and `returns`: the receiver is read again afterwards, so a
    # control appended in place would show up here.
    widened = block.control(1, ctrl_qubits=(4,))
    assert [(item.name, item.wires) for item in block.to_ir().instructions] == forward
    assert widened is not block
    assert control["receiver_is_mutated"] is False
    assert control["returns"] == "new_circuit"

    # `instruction_order`, `instruction_qubits` and `instruction_order_within_gate`: the
    # receiver's own program is expanded instruction by instruction in its own order, and
    # no gate is relabelled. The expansion is therefore the concatenation of the two
    # instructions' own expansions, in the order the receiver wrote them.
    assert _instruction_fields(widened) == _instruction_fields(
        fq.Circuit(1).h(0).control(1, ctrl_qubits=(4,))
    ) + _instruction_fields(fq.Circuit(2).cnot(0, 1).control(1, ctrl_qubits=(4,)))
    assert control["instruction_order"] == "forward"
    assert control["instruction_qubits"] == "unchanged"
    assert control["instruction_order_within_gate"] == "unchanged"
    assert {4} <= {
        qubit for item in widened.to_ir().instructions for qubit in item.wires
    }
    assert 4 not in {
        qubit for item in block.to_ir().instructions for qubit in item.wires
    }
    assert {item.wires for item in widened.to_ir().instructions} <= {
        (0,),
        (4, 0),
        (4, 0, 1),
    }

    # `result_width`: the control qubit is added, so the width is one past the greatest
    # control qubit named rather than the receiver's width plus the count.
    assert control["result_width"] == "max(ctrl_qubits) + 1"
    assert widened.n_qubits == 5
    assert fq.Circuit(3).h(0).control(2, ctrl_qubits=(7, 9)).n_qubits == 10

    # `preserves`: `n_qubits` is deliberately absent from the list, and every attribute
    # that is listed is read from the contract rather than restated.
    assert control["preserves"] == ["bsz", "device", "dtype"]
    shaped = fq.Circuit(2, bsz=3, device="cpu", dtype=torch.complex64).h(1)
    shaped_control = shaped.control(1, ctrl_qubits=(2,))
    for attribute in control["preserves"]:
        assert getattr(shaped_control, attribute) == getattr(shaped, attribute)
    assert shaped_control.n_qubits != shaped.n_qubits

    # `receiver_input_state`: a receiver carrying an input state keeps it. Qubit 0 is the
    # most significant amplitude bit, so the added control qubit is the least significant
    # one and the receiver's amplitudes move to slot 0 of each pair.
    state = torch.tensor([[0.6, 0.8]], dtype=torch.complex128)
    carried = fq.Circuit(1, dtype=torch.complex128, inputs=state).control(
        1, ctrl_qubits=(1,)
    )
    assert torch.allclose(
        carried.state().detach(),
        torch.tensor([[0.6 + 0j, 0j, 0.8 + 0j, 0j]], dtype=torch.complex128),
        atol=1e-14,
    )
    assert "at |0>" in control["receiver_input_state"]

    # `ancilla_qubits` and the ladder shape: the expansion names no qubit outside the
    # receiver's own qubits and the controls, which is what zero ancillas means, and the
    # deepest measured rung is the bound the contract states.
    assert control["ancilla_qubits"] == 0
    assert control["ladder_control_count"] == "n_controls"
    assert control["ladder_level"] == "n_controls + arity - 1"
    assert control["max_ladder_level"] == MAX_LADDER_LEVEL
    assert control["ladder_depth_growth"] == "exponential"
    ladder = fq.Circuit(1).t(0).control(4, ctrl_qubits=(1, 2, 3, 4))
    assert len(ladder.to_ir().instructions) == 4 * 3**3 - 3
    assert {qubit for item in ladder.to_ir().instructions for qubit in item.wires} == {
        0,
        1,
        2,
        3,
        4,
    }

    # `zero_angles_are_emitted`: an opcode whose controlled form is a phase ladder with an
    # explicit angle still emits its own ladder, so the count is the same as for any other
    # angle. A version that elided a zero angle would shrink this list.
    zeroed = fq.Circuit(1).rz(0, 0.0).control(2, ctrl_qubits=(1, 2))
    assert control["zero_angles_are_emitted"] is True
    assert len(zeroed.to_ir().instructions) == len(
        fq.Circuit(1).rz(0, 0.37).control(2, ctrl_qubits=(1, 2)).to_ir().instructions
    )

    # `single_control_partner_source` and `max_ladder_level_source`: both tables are read
    # from the operator schema, so the contract cannot point at a copy.
    for field in ("single_control_partner_source", "max_ladder_level_source"):
        assert control[field] == "flagquantum/core/operator_schema.py"
    assert pathlib.Path(ROOT / control["single_control_partner_source"]).is_file()
    assert control["angle_handling"].startswith("scaled in place")

    # `emitted_opcodes_are_registered`: a rule that reached for an opcode the registry
    # does not declare would put a name into the IR that the IR cannot execute.
    for instruction in ladder.to_ir().instructions:
        schema = OPERATOR_SCHEMAS.get(instruction.name)
        assert schema is not None and schema.opcode == instruction.name


def test_control_expands_to_the_hand_built_ladder() -> None:
    """Expansion, instruction by instruction: the controlled program is the hand-built one.

    `expansion_tests` in the contract names this test, and
    `tools/check_circuit_composition_contract.py` refuses a name that is not here. Both
    halves of the claim are measured: the emitted instruction list is compared with the
    program a user would have written, exact and in order, and the result is then applied
    to a basis and compared with the diagonal the definition of a controlled gate names,
    because two different programs can reach the same state and an instruction listing
    cannot tell a correct ladder from a wrong one.
    """

    # The one-control shortcut: for these five opcodes the controlled form is the
    # registered partner, so the emitted program is a single instruction and the control
    # qubit is named first.
    for opcode, params, partner, wires in (
        ("x", {}, "cx", (2, 0)),
        ("cnot", {}, "ccx", (2, 0, 1)),
        ("swap", {}, "cswap", (2, 0, 1)),
        ("rz", {"theta": 0.7}, "crz", (2, 0)),
    ):
        width = 2 if opcode in {"cnot", "swap"} else 1
        builder = getattr(fq.Circuit(width), opcode)
        receiver = builder(*range(width), **params)
        emitted = receiver.control(1, ctrl_qubits=(2,))
        assert [(item.name, item.wires) for item in emitted.to_ir().instructions] == [
            (partner, wires)
        ], opcode

    # The ladder itself, written out by hand for the two shallowest rungs. `t` is a phase
    # gate, so its controlled form is the ladder with no basis change around it and the
    # angle is halved once per added control.
    one_control = fq.Circuit(1).t(0).control(1, ctrl_qubits=(1,))
    assert _instruction_fields(one_control) == _instruction_fields(
        fq.Circuit(2).cphase(1, 0, theta=math.pi / 4)
    )
    two_controls = fq.Circuit(1).t(0).control(2, ctrl_qubits=(1, 2))
    hand_built = (
        fq.Circuit(3)
        .h(2)
        .cphase(1, 2, theta=math.pi)
        .h(2)
        .cphase(2, 0, theta=-math.pi / 8)
        .h(2)
        .cphase(1, 2, theta=math.pi)
        .h(2)
        .cphase(2, 0, theta=math.pi / 8)
        .cphase(1, 0, theta=math.pi / 8)
    )
    assert _instruction_fields(two_controls) == _instruction_fields(hand_built)
    assert two_controls.to_ir().instructions == hand_built.to_ir().instructions

    # The emitted list is the receiver's own order, one instruction at a time: two
    # different gates in one receiver expand to their own forms in the order they were
    # written, which is what `instruction_order` claims.
    ordered = fq.Circuit(1).x(0).t(0).control(1, ctrl_qubits=(1,))
    assert _instruction_fields(ordered) == _instruction_fields(
        fq.Circuit(2).cx(1, 0).cphase(1, 0, theta=math.pi / 4)
    )

    # And the ladder computes the gate. The controlled program is placed after a
    # preparation on the same register, so both controls are exercised as well as the
    # target: qubit 0 is the most significant amplitude bit, which makes the register index
    # `q0 * 4 + q1 * 2 + q2`. `t` is the phase gate, so the phase lands only on the basis
    # state where both controls and the target are set, and the other three rows are the
    # near misses that must stay at unit amplitude. The reference is the basis state and
    # the phase the definition names, written here rather than read back from the expansion.
    controlled = fq.Circuit(1).t(0).control(2, ctrl_qubits=(1, 2))
    quarter = complex(math.cos(math.pi / 4), math.sin(math.pi / 4))
    for preparation, index, expected in (
        ((), 0, 1.0),
        ((1, 2), 3, 1.0),
        ((0, 1, 2), 7, quarter),
        ((0, 1), 6, 1.0),
    ):
        register = fq.Circuit(3, dtype=torch.complex128)
        for qubit in preparation:
            register.x(qubit)
        register.compose(controlled)
        reference = torch.zeros(8, dtype=torch.complex128)
        reference[index] = expected
        assert torch.allclose(
            register.state().detach()[0], reference, atol=1e-14
        ), preparation

    # `receiver_is_mutated`: the receiver is read again afterwards and still holds the
    # program it was written with, so the expansion copied rather than consumed it.
    receiver = fq.Circuit(1).t(0)
    before = _instruction_fields(receiver)
    receiver.control(2, ctrl_qubits=(1, 2))
    assert _instruction_fields(receiver) == before


def test_the_unreachable_adjoint_branch_is_unreachable_by_construction() -> None:
    """The `reachable = false` row is measured here rather than asserted in the table."""

    unreachable = [row for row in _refusals() if row["reachable"] is False]
    assert [row["code"] for row in unreachable] == ["inverse_of_unknown_opcode"]
    assert unreachable[0]["code"] not in _TRIGGERS

    # An unknown opcode carrying a matrix inverts through that matrix, so it never
    # reaches the "unknown opcode with no matrix" branch.
    custom = _with_instruction(
        Instruction("unregistered", (1,), matrix=torch.eye(2, dtype=torch.complex128))
    ).adjoint()
    assert [item.name for item in custom.to_ir().instructions] == ["unregistered"]

    # And an unknown opcode without a matrix cannot be built at all, which is what makes
    # the branch unreachable instead of merely untested.
    with pytest.raises(IRValidationError, match="unknown opcode"):
        Instruction("unregistered", (1,))


def test_composition_adds_no_ir_field_and_no_root_export() -> None:
    """The contract's central claim: construction-time, not a second IR.

    "Construction-time" is a claim about what the two operations emit, so it is measured
    that way. Both are run over a circuit whose instructions come from every route the
    IR supports, and every field of every instruction that comes out must already belong
    to the declared :class:`Instruction` fields. A new field would be an IR change, and
    an IR change moves the frozen ``IR_VERSION`` literal, which is why that pin is
    checked here as well rather than assumed.
    """

    contract = _contract()
    declared = {field for field in Instruction.__dataclass_fields__}
    baseline = json.loads(
        (ROOT / "contracts/public-api-v0.2-baseline.json").read_text(encoding="utf-8")
    )
    assert baseline["exports"]["IR_VERSION"]["value"]["value"] == IR_VERSION

    source = (
        fq.Circuit(2)
        .h(0)
        .cnot(0, 1)
        .any(0, unitary=torch.eye(2, dtype=torch.complex128))
    )
    emitted = [
        *source.compose(fq.Circuit(1).x(0), qubits=1).to_ir().instructions,
        *source.adjoint().to_ir().instructions,
    ]
    assert emitted
    for instruction in emitted:
        assert set(instruction.__dataclass_fields__) == declared
        # An instruction is either a registered opcode or a custom operation that
        # carries its own matrix, which is the only other route the IR describes.
        schema = OPERATOR_SCHEMAS.get(instruction.name)
        assert (schema is not None and schema.opcode == instruction.name) or (
            instruction.matrix is not None
        ), instruction.name

    assert contract["ir_version_effect"] == "none"
    assert contract["root_export_effect"] == "none"
    assert not [name for name in fq.__all__ if name in {"compose", "adjoint"}]


def test_the_refusal_vocabulary_is_unique_and_complete() -> None:
    contract = _contract()
    refusals = _refusals()
    codes = [row["code"] for row in refusals]
    phrases = [row["message_phrase"] for row in refusals]
    required = {
        "entry_point",
        "code",
        "exception",
        "message_phrase",
        "message_source",
        "reachable",
        "trigger",
    }

    assert len(codes) == len(set(codes)), "two refusals share a code"
    assert len(phrases) == len(set(phrases)), "two refusals share a message phrase"
    # The count is the contract's own length, so dropping a row is a change to this
    # number rather than a change nobody notices.
    assert len(refusals) == 30
    for row in refusals:
        assert required <= set(row), (row["code"], sorted(required - set(row)))
        assert row["entry_point"] in {"compose", "adjoint", "control"}, row["code"]
        assert row["exception"] in _EXCEPTIONS, row["exception"]
        assert (ROOT / row["message_source"]).is_file(), row["message_source"]
        if row["reachable"] is False:
            assert row.get("reachability_note"), row["code"]
    # The internal-only list is the contract explaining a gap in its own vocabulary: it
    # must name a real refusal that the contracted entry points cannot reach, so that the
    # list stays a statement about this surface instead of a parking space.
    for entry in contract["issue_codes"]["internal_only"]:
        assert entry["exception"] == "ValueError", entry
        source = (ROOT / entry["source"]).read_text(encoding="utf-8")
        assert entry["message_phrase"] in source, entry
        assert entry["owner"] in source, entry
        assert entry["reason"]


@pytest.mark.parametrize("code", sorted(_TRIGGERS))
def test_each_contracted_refusal_raises_its_contracted_class(code: str) -> None:
    row = next(item for item in _refusals() if item["code"] == code)
    expected = _EXCEPTIONS[row["exception"]]

    with pytest.raises(expected) as error:
        _TRIGGERS[code]()

    assert type(error.value).__name__ == row["exception"]
    assert row["message_phrase"] in str(error.value), row["code"]


def test_every_reachable_refusal_has_a_trigger_and_every_trigger_a_refusal() -> None:
    contracted = {row["code"] for row in _refusals() if row["reachable"] is True}
    unreachable = {row["code"] for row in _refusals() if row["reachable"] is False}

    assert contracted == set(_TRIGGERS)
    assert not contracted & unreachable
    assert unreachable == {"inverse_of_unknown_opcode"}
    assert {row["entry_point"] for row in _refusals() if row["reachable"]} == {
        "compose",
        "adjoint",
        "control",
    }


@pytest.mark.parametrize(
    "row",
    [row for row in _refusals()],
    ids=[row["code"] for row in _refusals()],
)
def test_every_contracted_phrase_still_occurs_in_its_source(
    row: Mapping[str, Any],
) -> None:
    # Every phrase is checked here, reachable or not. A reachable one is also raised by
    # the trigger test above; the unreachable row has no other check, because no user
    # path raises it -- yet the vocabulary is frozen either way, so its phrase must
    # remain in the code that would report it.
    source = (ROOT / row["message_source"]).read_text(encoding="utf-8")

    assert row["message_phrase"] in source, row["code"]


def test_the_opcode_census_supports_the_unreachable_adjoint_rows() -> None:
    """Every registered opcode is inverted, or is a channel refused by the first rule.

    The two `reachable = false` adjoint rows are only honest while this census holds:
    `inverse_of_undeclared_rule` needs an opcode whose `adjoint` names no single gate,
    and `inverse_of_unknown_opcode` needs one that the IR accepted without a matrix.
    Registering such an opcode makes one of them reachable, and this test fails so that
    the contract is updated instead of quietly going stale.
    """

    inverted: list[str] = []
    channels: list[str] = []
    for opcode, schema in sorted(OPERATOR_SCHEMAS.items()):
        if schema.opcode != opcode:
            continue
        params = dict.fromkeys(schema.parameters, 0.3)
        circuit = getattr(fq.Circuit(max(schema.arity, 1)), opcode)(
            *range(schema.arity), **params
        )
        try:
            circuit.adjoint()
        except CapabilityError as error:
            assert "noise channel" in str(error), opcode
            channels.append(opcode)
        else:
            inverted.append(opcode)

    census = _contract()["verification"]["opcode_census"]
    assert len(inverted) + len(channels) == len(OPERATOR_SCHEMAS)
    assert len(inverted) == 31 and len(channels) == 4, (inverted, channels)
    assert channels == sorted(
        name
        for name, schema in OPERATOR_SCHEMAS.items()
        if schema.opcode == name and schema.adjoint == "not_applicable"
    )
    assert f"{len(inverted)} of {len(OPERATOR_SCHEMAS)} registered opcodes invert" in (
        census
    )
    assert "the 4 refusals are the noise channels" in census


# --------------------------------------------------------------------------------------
# The gate: `tools/check_circuit_composition_contract.py`
# --------------------------------------------------------------------------------------
#
# The tests above ask whether the contract is true of the code. This section asks the
# other question: is the contract still enforced as a *gate*? The gate is what CI and
# `tools/pre_push.py` run, so a clause it stopped reading is a clause nobody checks. Each
# mutation below patches one clause and requires the gate to name it.

_GATE_PATH = ROOT / "tools" / "check_circuit_composition_contract.py"
_GATE_SPEC = importlib.util.spec_from_file_location(
    "check_circuit_composition_contract", _GATE_PATH
)
assert _GATE_SPEC is not None and _GATE_SPEC.loader is not None
_GATE = importlib.util.module_from_spec(_GATE_SPEC)
_GATE_SPEC.loader.exec_module(_GATE)


def test_the_composition_gate_accepts_the_checked_in_contract() -> None:
    assert _GATE.contract_errors(_contract()) == ()


def test_the_composition_gate_runs_in_ci_and_before_push() -> None:
    """A gate that no workflow invokes is a script, not a gate."""

    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "python tools/check_circuit_composition_contract.py" in workflow
    pre_push = (ROOT / "tools/pre_push.py").read_text(encoding="utf-8")
    assert '"tools/check_circuit_composition_contract.py"' in pre_push


def test_the_gate_and_this_file_agree_on_the_expansion_tests() -> None:
    """One expand-and-compare test per provided operation, and it exists in this file."""

    verification = _contract()["verification"]
    provided = list(_contract()["scope"]["provided"])
    assert list(verification["expansion_tests"]) == provided
    source = Path(__file__).read_text(encoding="utf-8")
    for operation, test in verification["expansion_tests"].items():
        assert test.startswith("test_"), operation
        assert f"def {test}(" in source, (operation, test)


def _drop_rule(contract: dict[str, Any]) -> None:
    contract["adjoint"]["declared_opcode_rules"].remove("adjoint_u2_angles")


def _claim_control(contract: dict[str, Any]) -> None:
    """Claim an operation the contract says is absent, which is the direction that matters.

    `Circuit.control` is provided, so the interesting mutation is the other one: a contract
    that promises an operation nobody wrote. `Circuit.power` is the contracted absence, and
    this mutation moves it into `provided`.
    """

    contract["scope"]["provided"].append("Circuit.power")
    contract["scope"]["not_provided"].remove("Circuit.power")


def _dropped_control_authorization(contract: dict[str, Any]) -> None:
    contract["control"][
        "authorization"
    ] = "docs/api-changes/FQ-CIRCUIT-POWER-20261021.md"


def _control_rule_dropped(contract: dict[str, Any]) -> None:
    contract["control"]["declared_opcode_rules"].remove("u_angle_ladder")


def _control_claims_ancillas(contract: dict[str, Any]) -> None:
    contract["control"]["ancilla_qubits"] = 2


def _control_mutates_receiver(contract: dict[str, Any]) -> None:
    contract["control"]["receiver_is_mutated"] = True


def _stale_ladder_ceiling(contract: dict[str, Any]) -> None:
    contract["control"]["max_ladder_level"] = 9


def _stale_ladder_formula(contract: dict[str, Any]) -> None:
    contract["control"]["ladder_instruction_bound"] = "3 ** k"


def _control_wire_spelling(contract: dict[str, Any]) -> None:
    contract["control"]["control_qubits_argument"] = "ctrl_wires"


def _control_preserves_width(contract: dict[str, Any]) -> None:
    contract["control"]["preserves"] = ["n_qubits", "bsz", "device", "dtype"]


def _stale_control_census(contract: dict[str, Any]) -> None:
    contract["verification"]["control_census"] = "30 of 35 registered opcodes"


def _no_trigger(contract: dict[str, Any]) -> None:
    row = next(item for item in contract["refusals"] if item["reachable"] is True)
    del row["trigger"]


def _no_reachability_note(contract: dict[str, Any]) -> None:
    row = next(item for item in contract["refusals"] if item["reachable"] is False)
    del row["reachability_note"]


def _second_unreachable_row(contract: dict[str, Any]) -> None:
    row = next(item for item in contract["refusals"] if item["reachable"] is True)
    row["reachable"] = False
    row["reachability_note"] = "claimed unreachable by this mutation"


def _unknown_reason(contract: dict[str, Any]) -> None:
    contract["refusals"][0]["exception"] = "RuntimeError"


def _stale_phrase(contract: dict[str, Any]) -> None:
    contract["refusals"][0]["message_phrase"] = "outside the wire range"


def _missing_phrase_source(contract: dict[str, Any]) -> None:
    contract["refusals"][0]["message_source"] = "flagquantum/core/absent.py"


def _duplicate_code(contract: dict[str, Any]) -> None:
    contract["refusals"][1]["code"] = contract["refusals"][0]["code"]


def _stale_census(contract: dict[str, Any]) -> None:
    contract["verification"]["opcode_census"] = "30 of 35 registered opcodes invert"


def _expansion_test_missing(contract: dict[str, Any]) -> None:
    del contract["verification"]["expansion_tests"]["Circuit.adjoint"]


def _expansion_test_unknown(contract: dict[str, Any]) -> None:
    contract["verification"]["expansion_tests"][
        "Circuit.compose"
    ] = "test_not_in_this_file"


def _absent_authorization(contract: dict[str, Any]) -> None:
    contract["authorization"].append("docs/api-changes/FQ-ABSENT-20990101.md")


def _wrong_delivery(contract: dict[str, Any]) -> None:
    contract["issue_codes"]["delivery"] = "prose"


def _mutated_matrix_route(contract: dict[str, Any]) -> None:
    contract["adjoint"]["matrix_route"] = "transpose"


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (_drop_rule, "declared adjoint rules drifted"),
        (_claim_control, "contracted operation 'Circuit.power' does not exist"),
        (_dropped_control_authorization, "circuit control authorization"),
        (_control_rule_dropped, "declared rules drifted from CONTROL_RULES"),
        (_control_claims_ancillas, "must contract zero ancillas"),
        (_control_mutates_receiver, "receiver_is_mutated must be contracted as False"),
        (_stale_ladder_ceiling, "ladder ceiling drifted"),
        (_stale_ladder_formula, "ladder_instruction_bound must be"),
        (_control_wire_spelling, "Circuit.control signature drifted"),
        (_control_preserves_width, "preserves must be exactly bsz, device and dtype"),
        (_stale_control_census, "control census drifted"),
        (_no_trigger, "is reachable with no trigger"),
        (_no_reachability_note, "is unreachable with no reachability note"),
        (_second_unreachable_row, "exactly one unreachable refusal"),
        (_unknown_reason, "names unknown 'RuntimeError'"),
        (_stale_phrase, "no longer occurs in"),
        (_missing_phrase_source, "phrase source"),
        (_duplicate_code, "refusal codes repeat"),
        (_stale_census, "opcode census drifted"),
        (_expansion_test_missing, "without an expansion test"),
        (_expansion_test_unknown, "does not exist"),
        (_absent_authorization, "authorization"),
        (_wrong_delivery, "delivery form drifted"),
        (_mutated_matrix_route, "matrix route must be conjugate_transpose"),
    ],
    ids=lambda item: getattr(item, "__name__", ""),
)
def test_the_composition_gate_refuses_a_mutated_contract(
    mutate: Callable[[dict[str, Any]], None], expected: str
) -> None:
    contract = copy.deepcopy(_contract())
    mutate(contract)
    errors = _GATE.contract_errors(contract)
    assert errors, f"{mutate.__name__} survived the gate"
    assert any(expected in error for error in errors), (mutate.__name__, errors)

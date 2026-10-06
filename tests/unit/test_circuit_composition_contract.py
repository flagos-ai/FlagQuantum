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
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.primitives import qft
from flagquantum.core import ADJOINT_RULES, OPERATOR_SCHEMAS, POWER_RULES
from flagquantum.core.ir import IR_VERSION, Instruction, IRValidationError
from flagquantum.core.operator_schema import MAX_POWER_REPEATS
from flagquantum.ecosystem.conformance import semantic_fingerprint
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
    "ValueError": ValueError,
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


def dense_operator(program: fq.Circuit) -> torch.Tensor:
    """Return the program's operator as a ``2**n_qubits`` square matrix.

    Built by running the program on each computational basis vector, so the matrix is the
    one the simulator executes rather than a second implementation of the same gates. A
    matrix is a different measurement from an instruction list, and the two are compared
    in the power tests precisely because a rewrite can be instruction-for-instruction
    different and still be the same operator.
    """

    n_qubits = program.n_qubits
    columns = []
    for basis in range(2**n_qubits):
        carrier = fq.Circuit(n_qubits, dtype=torch.complex128)
        carrier.compose(program)
        state = torch.zeros(2**n_qubits, dtype=torch.complex128)
        state[basis] = 1.0
        carrier._inputs = state
        columns.append(carrier.state().detach())
    return torch.stack(columns, dim=1)


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
    # `power`: three spellings of the same refusal, because the route matters. A float
    # has an obvious meaning that the IR cannot express, a string is not a count at all,
    # and `True` is a flag -- `int(True) == 1` would silently make it a power of one.
    "exponent_not_an_integer": lambda: fq.Circuit(1).rx(0, 0.3).power(2.5),
    "power_exceeds_instruction_limit": lambda: (
        fq.Circuit(1).h(0).power(MAX_POWER_REPEATS + 1)
    ),
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
        "power_rule_source",
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


def test_declared_power_rules_are_the_operator_schema_vocabulary() -> None:
    power = _contract()["power"]

    # The vocabulary has one owner. Restating it in the contract is only safe while
    # this equality holds.
    assert list(POWER_RULES) == power["declared_opcode_rules"]
    assert power["receiver_is_mutated"] is False
    assert power["instruction_order"] == "forward"
    assert power["instruction_qubits"] == "unchanged"
    assert power["empty_circuit_result"] == "empty_circuit"
    assert power["exponent_type"] == "integer"
    assert power["negative_exponent_route"] == "Circuit.adjoint"
    signature = inspect.signature(fq.Circuit.power)
    assert list(signature.parameters) == ["self", power["exponent_argument"]]
    # The exponent is required rather than defaulted: a default would have to be a
    # power, and there is no power that means "you forgot the argument".
    assert signature.parameters[power["exponent_argument"]].default is (
        inspect.Parameter.empty
    )
    # The contract names the preserved attributes, so a rename must reach it. Only the
    # current name is contracted: `n_wires` is a deprecated alias and is not user-facing.
    assert "n_wires" not in power["preserves"]
    for attribute in power["preserves"]:
        assert hasattr(fq.Circuit(1), attribute), attribute


def test_the_closed_form_opcodes_are_the_declared_rule_and_scale_exactly() -> None:
    """The one rewrite a power performs, measured rather than read off the declaration.

    `scale_single_parameter` claims `U(theta)^k == U(k * theta)`. The list of opcodes the
    rewrite may touch is contracted, and the claim is checked through each opcode's dense
    operator, so a gate that starts scaling or stops scaling is a contract change rather
    than a silent widening. The counterexample is asserted too: scaling a two-angle gate
    is not the same operator, which is why the rule is per-opcode.
    """

    power = _contract()["power"]
    closed = sorted(
        opcode
        for opcode, schema in OPERATOR_SCHEMAS.items()
        if schema.opcode == opcode and schema.power_rule == "scale_single_parameter"
    )
    assert closed == sorted(power["closed_form_opcodes"])

    for opcode in closed:
        schema = OPERATOR_SCHEMAS[opcode]
        assert len(schema.parameters) == 1, opcode
        name = schema.parameters[0]
        for exponent in (2, 3, 5):
            repeated = fq.Circuit(schema.arity, dtype=torch.complex128)
            getattr(repeated, opcode)(*range(schema.arity), **{name: 0.3})
            scaled = fq.Circuit(schema.arity, dtype=torch.complex128)
            getattr(scaled, opcode)(*range(schema.arity), **{name: 0.3 * exponent})
            program = dense_operator(repeated.power(exponent))
            assert torch.allclose(program, dense_operator(scaled), atol=1e-12), (
                opcode,
                exponent,
            )

    # The rewrite is not vacuously true. `u2` and `u3` declare several angles, so the
    # rule gives them the repetition; scaling all of their angles would be a different
    # operator, which is what this measures.
    for opcode, params in (
        ("u3", {"theta": 0.4, "phi": 0.9, "lbd": 1.3}),
        ("u2", {"phi": 0.9, "lbd": 1.3}),
    ):
        schema = OPERATOR_SCHEMAS[opcode]
        assert schema.power_rule == "repeat_instruction"
        once = fq.Circuit(1, dtype=torch.complex128)
        getattr(once, opcode)(0, **params)
        doubled = fq.Circuit(1, dtype=torch.complex128)
        getattr(doubled, opcode)(
            0, **{key: 2.0 * value for key, value in params.items()}
        )
        deviation = float(
            (dense_operator(once.power(2)) - dense_operator(doubled)).abs().max()
        )
        assert deviation > 1e-3, (opcode, deviation)


def test_the_power_flags_hold_on_a_measured_repetition() -> None:
    """Every `[power]` flag is the user-visible result of one measurement."""

    power = _contract()["power"]
    block = fq.Circuit(2).h(0).cnot(0, 1).rz(1, 0.7)
    forward = [(item.name, item.wires) for item in block.to_ir().instructions]

    # `instruction_order` and `instruction_qubits`: the program is repeated in order, and
    # no qubit is relabelled, so the power acts where the program acted.
    powered = block.power(2)
    assert [(item.name, item.wires) for item in powered.to_ir().instructions] == [
        *forward,
        *forward,
    ]
    assert power["instruction_order"] == "forward"
    assert power["instruction_qubits"] == "unchanged"
    assert power["instruction_order_within_gate"] == "unchanged"

    # `receiver_is_mutated` and `returns`: the source stays usable, so a reusable block
    # can be raised and appended again.
    assert [(item.name, item.wires) for item in block.to_ir().instructions] == forward
    assert powered is not block
    assert power["receiver_is_mutated"] is False
    assert power["returns"] == "new_circuit"

    # `preserves`: the power is the same program shape, so it executes the same way.
    shaped = fq.Circuit(3, bsz=2, device="cpu", dtype=torch.complex64).h(1)
    raised = shaped.power(2)
    assert raised.to_ir().instructions != ()
    assert power["preserves"] == ["n_qubits", "bsz", "device", "dtype"]
    for attribute in power["preserves"]:
        assert getattr(raised, attribute) == getattr(shaped, attribute), attribute

    # `empty_circuit_result`: the power of nothing is nothing, and the width survives.
    empty = fq.Circuit(2).power(3)
    assert empty.to_ir().instructions == ()
    assert empty.n_qubits == 2
    assert power["empty_circuit_result"] == "empty_circuit"

    # `matrix_route_precedes_opcode_rule` and `matrix_route`: a custom operation that
    # carries a matrix is repeated rather than rewritten, because the matrix is what
    # executes and the opcode declares no angle to scale.
    custom = fq.Circuit(1).any(0, unitary=torch.eye(2, dtype=torch.complex128))
    assert [item.name for item in custom.power(2).to_ir().instructions] == ["any"] * 2
    assert power["matrix_route_precedes_opcode_rule"] is True
    assert power["matrix_route"] == "repeat"

    # `single_instruction_rewrite_only`: a longer program is repeated even when one of its
    # gates would have been rewritten on its own, because the copies of that gate are
    # separated by the others and do not compose into one instruction.
    assert len(block.power(2).to_ir().instructions) == 6
    assert [
        item.name for item in fq.Circuit(1).rx(0, 0.3).power(2).to_ir().instructions
    ] == ["rx"]
    assert power["single_instruction_rewrite_only"] is True

    # `max_emitted_instructions`: the bound is the implementation's constant, read from
    # its own module rather than restated, and the refusal reports it.
    assert power["max_emitted_instructions"] == MAX_POWER_REPEATS
    assert (ROOT / power["max_emitted_instructions_source"]).is_file()
    with pytest.raises(ValueError, match=str(MAX_POWER_REPEATS)):
        block.power(MAX_POWER_REPEATS)


def test_power_expands_to_the_hand_built_repetition() -> None:
    """The expand-and-compare test the contract names for `Circuit.power`.

    A power is a construction-time rewrite, so the measurement is instruction by
    instruction against the program a user would have written by hand. The hand-built
    program is built with explicit repeated calls rather than with `power`, so the two
    sides share no implementation.
    """

    block = fq.Circuit(2).rx(0, 0.3).cnot(0, 1)

    by_power = block.power(2)
    by_hand = fq.Circuit(2).rx(0, 0.3).cnot(0, 1).rx(0, 0.3).cnot(0, 1)
    assert by_power.to_ir().to_dict()["instructions"] == (
        by_hand.to_ir().to_dict()["instructions"]
    )
    assert by_power.to_ir().n_wires == by_hand.to_ir().n_wires
    assert semantic_fingerprint(by_power) == semantic_fingerprint(by_hand)

    # The same circuit composed by hand out of a repeated block, which is the spelling a
    # user would reach for without `power`, reaches the same program.
    composed = fq.Circuit(2).compose(block).compose(block)
    assert composed.to_ir().to_dict()["instructions"] == (
        by_power.to_ir().to_dict()["instructions"]
    )

    # The single-instruction case is the one rewrite, and it is the same operator: one
    # scaled gate rather than two. A power of one is a copy, so the shortening does not
    # leak into the identity case.
    single = fq.Circuit(1).rx(0, 0.3).power(2)
    assert [(item.name, item.params) for item in single.to_ir().instructions] == [
        ("rx", {"theta": 0.6})
    ]
    assert [
        item.name for item in fq.Circuit(1).rx(0, 0.3).power(1).to_ir().instructions
    ] == ["rx"]
    assert fq.Circuit(1).rx(0, 0.3).power(1).to_ir().instructions[0].params == {
        "theta": 0.3
    }


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
    assert not [name for name in fq.__all__ if name in {"compose", "adjoint", "power"}]
    # `power` is a method, so it adds no root export and the count does not move. The
    # count itself is pinned by `tools/public_api_snapshot.py`; this is the same claim
    # read from the other side, that the family is a family of methods. It reads 37
    # here because `main` added `fq.density_matrix` to that frozen list while this
    # branch was in review; the composition family contributed none of them.
    assert len(fq.__all__) == 37


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
    assert len(refusals) == 19
    for row in refusals:
        assert required <= set(row), (row["code"], sorted(required - set(row)))
        assert row["entry_point"] in {"compose", "adjoint", "power"}, row["code"]
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
        "power",
    }
    # Every entry point that contracts an unreachable row is a claim the census test
    # below has to support, so the set is closed in both directions.
    assert {row["entry_point"] for row in _refusals() if not row["reachable"]} == {
        "adjoint"
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


def test_the_power_census_is_the_declared_rule_split() -> None:
    """The census the power rewrite reads, counted from the declaration.

    `power_rule` answers one of two rules for every registered opcode, and the census is
    that split. A rule that moved an opcode between the two forms would change how a
    power is written, so the count is contracted rather than described.
    """

    rules = {opcode: schema.power_rule for opcode, schema in OPERATOR_SCHEMAS.items()}
    closed = sorted(
        opcode for opcode, rule in rules.items() if rule == "scale_single_parameter"
    )
    repeated = sorted(
        opcode for opcode, rule in rules.items() if rule != "scale_single_parameter"
    )

    assert closed
    assert repeated
    assert len(closed) + len(repeated) == len(OPERATOR_SCHEMAS)
    assert len(closed) == 12 and len(repeated) == 23, (closed, repeated)
    # Every rule is one of the two declared names, so a third rule cannot appear without
    # this failing rather than quietly taking a default.
    assert set(rules.values()) <= set(POWER_RULES)
    # `scale_single_parameter` is exactly "a unitary that declares one parameter", which
    # is the derivation the property claims. It is checked here so that the property and
    # the rule list cannot drift.
    assert closed == sorted(
        opcode
        for opcode, schema in OPERATOR_SCHEMAS.items()
        if schema.unitary and len(schema.parameters) == 1
    )
    census = _contract()["verification"]["power_census"]
    assert f"{len(closed)} of {len(OPERATOR_SCHEMAS)} registered opcodes take" in census


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


#: The internal-only row's reader list, read as the measurement it claims to be.
_RELABELLING = _contract()["issue_codes"]["internal_only"][0]


def test_the_relabelling_rule_readers_are_measured_from_the_import_graph() -> None:
    """The recorded readers are exactly the package modules that import the rule."""

    source = _RELABELLING["source"]
    owner = _RELABELLING["owner"]
    module = source[: -len(".py")].replace("/", ".")

    measured = _GATE._package_readers(module, owner, root=ROOT)

    assert measured == sorted(_RELABELLING["consumers"])
    assert measured, "the relabelling rule must have at least one reader"


def test_the_relabelling_clause_refuses_a_tree_that_copied_the_rule(
    tmp_path: Path,
) -> None:
    """The clause is measured against a built tree, so each drift is shown to fail.

    Building the tree is the point: the clause's claim is about the shape of a package,
    and a fake package is the only way to show that it refuses a copy instead of only
    agreeing with the one checkout it was written in.
    """

    package = tmp_path / "flagquantum"
    (package / "core").mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "core" / "__init__.py").write_text("", encoding="utf-8")
    rule = (
        "def remap_qubits(qubits, mapping, *, owner='composition'):\n"
        "    return tuple(mapping[qubit] for qubit in qubits)\n"
    )
    (package / "core" / "qubit_mapping.py").write_text(rule, encoding="utf-8")
    reader = (
        "from .core.qubit_mapping import remap_qubits\n"
        "\n"
        "def place(qubits, mapping):\n"
        "    return remap_qubits(qubits, mapping, owner='cell')\n"
    )
    (package / "cell.py").write_text(reader, encoding="utf-8")
    contract = {
        "issue_codes": {
            "internal_only": [
                {
                    "source": "flagquantum/core/qubit_mapping.py",
                    "owner": "remap_qubits",
                    "consumers": ["flagquantum/cell.py"],
                    "retired_wording": "outside its mapping",
                }
            ]
        }
    }

    assert _GATE._relabelling_errors(contract, root=tmp_path) == []

    # A second module that reads the rule without being recorded is a second reader.
    (package / "other.py").write_text(reader, encoding="utf-8")
    errors = _GATE._relabelling_errors(contract, root=tmp_path)
    assert any("drifted from the import graph" in error for error in errors), errors
    assert any("flagquantum/other.py" in error for error in errors), errors

    # A reader that imports the rule but folds a failed lookup into its own refusal has
    # copied the rule, whether or not it still mentions the rule's name.
    (package / "other.py").unlink()
    (package / "cell.py").write_text(
        reader.replace(
            "    return remap_qubits(qubits, mapping, owner='cell')\n",
            "    try:\n"
            "        return remap_qubits(qubits, mapping, owner='cell')\n"
            "    except KeyError as error:\n"
            "        raise ValueError('a cell qubit is outside its mapping') from error\n",
        ),
        encoding="utf-8",
    )
    errors = _GATE._relabelling_errors(contract, root=tmp_path)
    assert any("kept the retired private wording" in error for error in errors), errors

    # A recorded reader that stopped reading the rule is a claim about code that is gone.
    (package / "cell.py").write_text(
        "def place(qubits, mapping):\n    return tuple(mapping[q] for q in qubits)\n",
        encoding="utf-8",
    )
    errors = _GATE._relabelling_errors(contract, root=tmp_path)
    assert any("never calls remap_qubits" in error for error in errors), errors

    # And a row that records no readers at all is not a measurement.
    recorded = contract["issue_codes"]["internal_only"][0].pop("consumers")
    errors = _GATE._relabelling_errors(contract, root=tmp_path)
    assert any("readers are not recorded" in error for error in errors), errors
    contract["issue_codes"]["internal_only"][0]["consumers"] = recorded


def _drop_rule(contract: dict[str, Any]) -> None:
    contract["adjoint"]["declared_opcode_rules"].remove("adjoint_u2_angles")


def _claim_control(contract: dict[str, Any]) -> None:
    contract["scope"]["provided"].append("Circuit.control")
    contract["scope"]["not_provided"].remove("Circuit.control")


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


def _drop_power_rule(contract: dict[str, Any]) -> None:
    contract["power"]["declared_opcode_rules"].remove("repeat_instruction")


def _drop_closed_form_opcode(contract: dict[str, Any]) -> None:
    contract["power"]["closed_form_opcodes"].remove("rzz")


def _claim_a_closed_form(contract: dict[str, Any]) -> None:
    contract["power"]["closed_form_opcodes"].append("u3")


def _stale_power_census(contract: dict[str, Any]) -> None:
    contract["verification"][
        "power_census"
    ] = "13 of 35 registered opcodes take the closed form"


def _mutated_power_route(contract: dict[str, Any]) -> None:
    contract["power"]["matrix_route"] = "scale_single_parameter"


def _renamed_exponent(contract: dict[str, Any]) -> None:
    contract["power"]["exponent_argument"] = "count"


def _stale_power_bound(contract: dict[str, Any]) -> None:
    contract["power"]["max_emitted_instructions"] = 1024


def _dropped_power_authorization(contract: dict[str, Any]) -> None:
    contract["authorization"] = [
        item for item in contract["authorization"] if "POWER" not in item
    ]


def _multi_instruction_rewrite(contract: dict[str, Any]) -> None:
    contract["power"]["single_instruction_rewrite_only"] = False


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (_drop_rule, "declared adjoint rules drifted"),
        (_claim_control, "contracted operation 'Circuit.control' does not exist"),
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
        (_drop_power_rule, "declared power rules drifted"),
        (_drop_closed_form_opcode, "closed-form opcodes drifted"),
        (_claim_a_closed_form, "closed-form opcodes drifted"),
        (_stale_power_census, "power census drifted"),
        (_mutated_power_route, "power matrix route must be repeat"),
        (_renamed_exponent, "Circuit.power parameters drifted"),
        (_stale_power_bound, "power bound drifted from MAX_POWER_REPEATS"),
        (_dropped_power_authorization, "authorization"),
        (_multi_instruction_rewrite, "single-instruction only"),
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

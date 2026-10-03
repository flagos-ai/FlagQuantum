"""Conformance tests for ``Circuit.adjoint`` and the opcode adjoint declaration.

The inverse of a program is only correct if the inverse of every opcode is correct, so
the tests here fall into three groups: the opcode table itself, the circuit rewrite that
reads it, and the refusal of the operations that have no unitary inverse.
"""

from __future__ import annotations

import math

import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms.primitives import qft
from flagquantum.compiler.pipeline import _SELF_INVERSE
from flagquantum.core import (
    ADJOINT_RULES,
    OPERATOR_SCHEMAS,
    Parameter,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.errors import CapabilityError
from flagquantum.simulation.gate_matrix import gate_matrix

pytestmark = pytest.mark.unit

#: Every invertible unitary opcode with one set of legal parameters.
_UNITARY_OPCODES: dict[str, tuple[int, dict[str, object]]] = {
    "i": (1, {}),
    "x": (1, {}),
    "y": (1, {}),
    "z": (1, {}),
    "h": (1, {}),
    "s": (1, {}),
    "sdg": (1, {}),
    "t": (1, {}),
    "tdg": (1, {}),
    "sx": (1, {}),
    "sxdg": (1, {}),
    "rx": (1, {"theta": 0.37}),
    "ry": (1, {"theta": 0.37}),
    "rz": (1, {"theta": 0.37}),
    "phase": (1, {"theta": 0.37}),
    "u1": (1, {"theta": 0.37}),
    "u2": (1, {"phi": 0.37, "lbd": 1.1}),
    "u3": (1, {"theta": 0.37, "phi": 0.4, "lbd": 1.1}),
    "cx": (2, {}),
    "cy": (2, {}),
    "cz": (2, {}),
    "swap": (2, {}),
    "crx": (2, {"theta": 0.37}),
    "cry": (2, {"theta": 0.37}),
    "crz": (2, {"theta": 0.37}),
    "cphase": (2, {"theta": 0.37}),
    "rxx": (2, {"theta": 0.37}),
    "ryy": (2, {"theta": 0.37}),
    "rzz": (2, {"theta": 0.37}),
    "ccx": (3, {}),
    "cswap": (3, {}),
}


def _append(circuit: fq.Circuit, other: fq.Circuit) -> fq.Circuit:
    """Append every instruction of ``other`` to ``circuit``, as a user would."""

    for instruction in other.to_ir().instructions:
        circuit.gate(
            instruction.name,
            instruction.wires,
            params=dict(instruction.params),
            matrix=instruction.matrix,
        )
    return circuit


def test_adjoint_declaration_is_defined_for_every_opcode() -> None:
    for schema in OPERATOR_SCHEMAS.values():
        assert schema.adjoint, schema.opcode
        if schema.adjoint in ADJOINT_RULES:
            continue
        partner = OPERATOR_SCHEMAS[schema.adjoint]
        assert partner.adjoint == schema.opcode, schema.opcode
        assert partner.arity == schema.arity, schema.opcode
        assert not partner.parameters, schema.opcode
        assert partner.unitary, schema.opcode


def test_every_unitary_opcode_declares_an_invertible_rule() -> None:
    for schema in OPERATOR_SCHEMAS.values():
        if not schema.unitary:
            assert schema.adjoint == "not_applicable", schema.opcode
            continue
        assert schema.adjoint != "not_applicable", schema.opcode
        assert schema.adjoint != "matrix_adjoint", schema.opcode


@pytest.mark.parametrize("opcode", sorted(_UNITARY_OPCODES))
def test_adjoint_of_one_opcode_restores_the_initial_state(opcode: str) -> None:
    arity, params = _UNITARY_OPCODES[opcode]
    qubits = tuple(range(arity))

    circuit = fq.Circuit(arity, dtype=torch.complex128)
    circuit.gate(opcode, qubits, params=params)
    _append(circuit, circuit.adjoint())

    initial = fq.Circuit(arity, dtype=torch.complex128).state().detach()
    assert torch.allclose(circuit.state().detach(), initial, atol=1e-14)


@pytest.mark.parametrize(
    ("opcode", "params"),
    [
        ("s", {}),
        ("t", {}),
        ("sx", {}),
        ("sdg", {}),
        ("tdg", {}),
        ("sxdg", {}),
        ("rx", {"theta": 0.37}),
        ("cphase", {"theta": 0.37}),
        ("u1", {"theta": 0.37}),
    ],
)
def test_adjoint_uses_the_declared_opcode_rule(
    opcode: str, params: dict[str, object]
) -> None:
    arity = OPERATOR_SCHEMAS[opcode].arity
    circuit = fq.Circuit(arity).gate(opcode, tuple(range(arity)), params=params)
    instructions = circuit.adjoint().to_ir().instructions
    assert len(instructions) == 1
    assert instructions[0].wires == tuple(range(arity))

    if params:
        assert instructions[0].name == opcode
        assert instructions[0].params == {
            name: -value for name, value in params.items()
        }
    else:
        assert instructions[0].name != opcode
        assert instructions[0].params == {}


def test_adjoint_rules_touch_the_declared_parameters_and_keep_the_rest() -> None:
    """An undeclared parameter carries through unchanged rather than disappearing.

    ``Instruction`` accepts a parameter the schema does not declare. The rule for an
    opcode is written against the parameters the schema declares, so it must not be
    read as permission to drop the others.
    """

    circuit = fq.Circuit(1).gate("rx", (0,), params={"theta": 0.37, "gamma": 0.25})
    adjoint = circuit.adjoint().to_ir().instructions[0]

    assert adjoint.params["theta"] == pytest.approx(-0.37)
    assert adjoint.params["gamma"] == pytest.approx(0.25)

    circuit = fq.Circuit(1).u3(0, 0.37, 0.4, 1.1)
    adjoint = circuit.adjoint().to_ir().instructions[0]
    assert adjoint.name == "u3"
    assert adjoint.params["theta"] == pytest.approx(-0.37)
    assert adjoint.params["phi"] == pytest.approx(-1.1)
    assert adjoint.params["lbd"] == pytest.approx(-0.4)


def test_u2_adjoint_uses_the_angle_rule_of_the_u2_convention() -> None:
    circuit = fq.Circuit(1).u2(0, 0.37, 1.1)
    adjoint = circuit.adjoint().to_ir().instructions[0]
    assert adjoint.name == "u2"
    assert adjoint.params["phi"] == pytest.approx(-1.1 - math.pi)
    assert adjoint.params["lbd"] == pytest.approx(-0.37 - math.pi)


def _matrix_of(instruction: Instruction) -> torch.Tensor:
    """Return the 2x2 matrix of a one-qubit instruction, in double precision.

    Double precision is not incidental: the naive rule's error is measured down to
    ``1e-16``, and the default single-precision dtype would report ``1e-07`` for the
    correct rule as well and make the comparison meaningless.
    """

    built = gate_matrix(
        instruction, bsz=1, device=torch.device("cpu"), dtype=torch.complex128
    )
    return built.reshape(2, 2)


@pytest.mark.parametrize(
    ("opcode", "params", "wrong_rule_error"),
    [
        ("u3", {"theta": 0.4, "phi": 0.9, "lbd": 1.3}, 5e-2),
        ("u2", {"phi": 0.9, "lbd": 1.3}, 5e-1),
    ],
)
def test_the_adjoint_rule_is_not_the_obvious_one_and_the_obvious_one_is_wrong(
    opcode: str, params: dict[str, float], wrong_rule_error: float
) -> None:
    """The measured error of the naive rule justifies a rule per convention.

    Negating every angle in place is the rule the other twelve single-angle rotations
    use. For ``u2`` and ``u3`` it composes to something far from the identity, so the
    declaration has to name a rule instead of a boolean, and this test fails if someone
    "simplifies" those two opcodes back onto ``negate_parameters``.
    """

    identity = torch.eye(2, dtype=torch.complex128)

    def residual(inverse_params: dict[str, float]) -> float:
        forward = _matrix_of(Instruction(opcode, (0,), params=params))
        inverse = _matrix_of(Instruction(opcode, (0,), params=inverse_params))
        return (inverse @ forward - identity).abs().max().item()

    circuit = fq.Circuit(1).gate(opcode, (0,), params=params)
    declared = dict(circuit.adjoint().to_ir().instructions[0].params)
    naive = {name: -value for name, value in params.items()}

    assert residual(declared) < 1e-14, declared
    assert residual(naive) > wrong_rule_error, naive


def test_adjoint_reverses_instruction_order_and_returns_a_new_circuit() -> None:
    circuit = fq.Circuit(2).h(0).s(0).cx(0, 1)

    inverse = circuit.adjoint()

    assert [item.name for item in inverse.to_ir().instructions] == ["cx", "sdg", "h"]
    assert [item.wires for item in inverse.to_ir().instructions] == [
        (0, 1),
        (0,),
        (0,),
    ]
    assert [item.name for item in circuit.to_ir().instructions] == ["h", "s", "cx"]


def test_adjoint_circuit_undoes_the_block_it_was_taken_from() -> None:
    circuit = (
        fq.Circuit(3, dtype=torch.complex128).h(0).rx(1, 0.9).cx(0, 1).cphase(1, 2, 0.3)
    )
    _append(circuit, circuit.adjoint())

    initial = fq.Circuit(3, dtype=torch.complex128).state().detach()
    assert torch.allclose(circuit.state().detach(), initial, atol=1e-14)


def test_adjoint_twice_restores_the_original_instruction_sequence() -> None:
    circuit = fq.Circuit(2).h(0).t(1).crx(0, 1, 0.25).swap(0, 1)
    names = [item.name for item in circuit.to_ir().instructions]
    assert [item.name for item in circuit.adjoint().adjoint().to_ir().instructions] == (
        names
    )


def test_adjoint_of_an_empty_circuit_is_empty() -> None:
    inverse = fq.Circuit(2).adjoint()
    assert inverse.to_ir().instructions == ()
    assert inverse.n_qubits == 2


def test_adjoint_preserves_width_batch_size_device_and_dtype() -> None:
    circuit = fq.Circuit(2, bsz=3, dtype=torch.complex128).ry(0, [0.1, 0.2, 0.3])
    inverse = circuit.adjoint()
    assert inverse.n_qubits == circuit.n_qubits
    assert inverse.bsz == 3
    assert inverse.device == circuit.device
    assert inverse.dtype == torch.complex128


def test_adjoint_negates_a_per_batch_angle_element_by_element() -> None:
    circuit = fq.Circuit(1, bsz=3).ry(0, [0.1, 0.2, 0.3])
    assert circuit.adjoint().to_ir().instructions[0].params["theta"] == [
        -0.1,
        -0.2,
        -0.3,
    ]


def test_adjoint_keeps_symbolic_parameters_symbolic() -> None:
    circuit = fq.Circuit(2)
    circuit.ry(0, Parameter("theta"))
    circuit.crx(0, 1, Parameter("theta") * 2)

    inverse = circuit.adjoint()

    assert inverse.parameter_names == ("theta",)
    assert inverse.is_parameterized()
    assert [item.name for item in inverse.to_ir().instructions] == ["crx", "ry"]


def test_adjoint_binds_to_the_same_values_as_the_forward_block() -> None:
    circuit = fq.Circuit(1, dtype=torch.complex128)
    circuit.ry(0, Parameter("theta"))
    inverse = circuit.adjoint().bind_parameters({"theta": 0.7})
    forward = circuit.bind_parameters({"theta": 0.7})

    assert torch.allclose(
        _append(forward, inverse).state().detach(),
        fq.Circuit(1, dtype=torch.complex128).state().detach(),
        atol=1e-14,
    )


def test_adjoint_inverts_a_custom_unitary_through_its_matrix() -> None:
    unitary = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.complex128)
    circuit = fq.Circuit(1, dtype=torch.complex128).any(0, unitary=unitary)

    _append(circuit, circuit.adjoint())

    initial = fq.Circuit(1, dtype=torch.complex128).state().detach()
    assert torch.allclose(circuit.state().detach(), initial, atol=1e-14)


def test_adjoint_of_a_custom_unitary_keeps_its_opcode_and_parameters() -> None:
    unitary = torch.eye(4, dtype=torch.complex128)
    instruction = Instruction("custom", (0, 1), matrix=unitary)
    circuit = fq.Circuit(2, dtype=torch.complex128)
    circuit._instructions.append(instruction)

    adjoint = circuit.adjoint().to_ir().instructions[0]

    assert adjoint.name == "custom"
    assert adjoint.wires == (0, 1)
    assert torch.allclose(adjoint.matrix, unitary.mH)


def test_adjoint_carries_the_metadata_of_the_instruction_it_inverts() -> None:
    instruction = Instruction(
        "custom",
        (0,),
        matrix=torch.eye(2, dtype=torch.complex128),
        metadata={"origin": "test"},
    )
    circuit = fq.Circuit(1, dtype=torch.complex128)
    circuit._instructions.append(instruction)

    assert circuit.adjoint().to_ir().instructions[0].metadata == {"origin": "test"}


@pytest.mark.parametrize(
    ("label", "instruction"),
    [
        (
            "a noise channel",
            Instruction(
                "bit_flip",
                (1,),
                params={"probability": 0.1},
                matrix=torch.eye(2, dtype=torch.complex128),
                metadata={"is_channel": True},
            ),
        ),
        (
            "a dynamic operation",
            Instruction("reset", (1,), metadata={"is_dynamic": True}),
        ),
        (
            "classically conditioned",
            Instruction("x", (1,), metadata={"conditions": (("c", 1),)}),
        ),
        (
            "does not expose a conjugate transpose",
            Instruction("custom", (1,), matrix=((1.0, 0.0), (0.0, 1.0))),
        ),
    ],
)
def test_adjoint_refuses_an_operation_without_a_unitary_inverse(
    label: str, instruction: Instruction
) -> None:
    circuit = fq.Circuit(2)
    circuit._instructions.append(instruction)

    with pytest.raises(CapabilityError) as error:
        circuit.adjoint()

    assert instruction.name in str(error.value)
    assert "(1,)" in str(error.value)
    assert label in str(error.value)


def test_adjoint_refuses_a_channel_reached_through_from_ir() -> None:
    ir = CircuitIR(
        n_wires=2,
        instructions=(
            Instruction("h", (0,)),
            Instruction(
                "bit_flip",
                (1,),
                params={"probability": 0.1},
                matrix=torch.eye(2, dtype=torch.complex128),
                metadata={"is_channel": True},
            ),
        ),
    )
    with pytest.raises(CapabilityError, match="noise channel"):
        fq.Circuit.from_ir(ir).adjoint()


def test_adjoint_of_the_qft_matches_the_recorded_inverse_transform() -> None:
    """``Circuit.adjoint`` reproduces the inverse the QFT builds by hand."""

    for n_qubits in range(1, 6):
        declared = [
            (item.name, item.wires, dict(item.params))
            for item in qft(n_qubits).adjoint().to_ir().instructions
        ]
        recorded = [
            (item.name, item.wires, dict(item.params))
            for item in qft(n_qubits, inverse=True).to_ir().instructions
        ]
        assert declared == recorded, n_qubits


def test_qft_adjoint_inverts_the_transform_on_a_state() -> None:
    circuit = fq.Circuit(3, dtype=torch.complex128)
    _append(circuit, qft(3))
    _append(circuit, circuit.adjoint())

    initial = fq.Circuit(3, dtype=torch.complex128).state().detach()
    assert torch.allclose(circuit.state().detach(), initial, atol=1e-14)


def test_compiler_self_inverse_transform_agrees_with_the_schema_declaration() -> None:
    """The compiler's self-inverse list may not name a gate the schema inverts."""

    declared = {
        schema.opcode
        for schema in OPERATOR_SCHEMAS.values()
        if schema.adjoint == "self_inverse"
    }
    assert declared >= _SELF_INVERSE, sorted(_SELF_INVERSE - declared)

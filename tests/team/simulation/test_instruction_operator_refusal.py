"""What an engine says when an instruction has no dense operator to execute.

`gate_matrix` is the one place a torch engine turns an IR instruction into a
dense operator, so it is the one place that can state this failure. Before this
contract, the opcode was a dictionary key: `barrier`, `measure`, `reset`, a
misspelled gate, and a channel whose declared parameters were never turned into
Kraus operators all escaped as `KeyError: 'barrier'`, which names neither what
was missing nor what to do about it, and which crosses the stable boundary as a
backend-native error that `contracts/errors-module-boundary-v1-candidate.json`
forbids. A barrier is admitted to the IR by design (`is_dynamic`), and
`fq.experimental.dynamic.run_dynamic` reaches `gate_matrix` with it, so that key
error was the whole report a user got for a legal program.

The tests are arranged so that removing the guard fails them for the right
reason: the two refused shapes assert the declared category and the two halves of
the report, and the reachability test goes through the public entry point rather
than through the helper. The accepted shapes are here to show the guard is not
vacuous -- a guard that refused a declared unitary would satisfy every refusal
test in this file and be useless.
"""

from __future__ import annotations

import math

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import get_operator_schema
from flagquantum.errors import CapabilityError, FlagQuantumError, ValidationError
from flagquantum.runtime.dynamic import run_dynamic
from flagquantum.runtime.dynamic.circuit import DynamicCircuit
from flagquantum.simulation.density_matrix import density_matrix_from_ir
from flagquantum.simulation.gate_matrix import gate_matrix

pytestmark = pytest.mark.unit

#: Directive opcodes the registry does not declare and no engine materializes.
#: Each is admitted to the IR for a different reason, which is why the list is
#: built from the instructions instead of from a vocabulary kept here.
_UNDECLARED = {
    "a directive that spans two operands": Instruction(
        "barrier", (0, 1), {}, None, {"is_dynamic": True}
    ),
    "a measurement": Instruction(
        "measure", (0,), {}, None, {"is_dynamic": True, "classical_bit": 0}
    ),
    "a reset": Instruction("reset", (0,), {}, None, {"is_dynamic": True}),
    "a misspelled gate": Instruction("hadamrd", (0,), {}, None, {"is_dynamic": True}),
}


def _matrix(instruction: Instruction) -> torch.Tensor:
    return gate_matrix(instruction, bsz=1, device="cpu", dtype=torch.complex128)


def test_a_declared_unitary_still_materializes_its_matrix() -> None:
    """The guard must not stand in front of the gates that do have operators."""

    x_matrix = _matrix(Instruction("x", (0,), {}, None, {}))
    assert torch.allclose(
        torch.as_tensor(x_matrix).reshape(2, 2),
        torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
    )
    rotation = torch.as_tensor(
        _matrix(Instruction("rx", (0,), {"theta": 0.3}, None, {}))
    ).reshape(2, 2)
    expected = torch.tensor(
        [
            [math.cos(0.15), -1j * math.sin(0.15)],
            [-1j * math.sin(0.15), math.cos(0.15)],
        ],
        dtype=torch.complex128,
    )
    assert torch.allclose(rotation, expected)


def test_an_instruction_carrying_its_own_matrix_is_never_refused() -> None:
    """A matrix on the instruction is the operator, whatever the opcode is called.

    This pins the branch order the guard was inserted behind. A barrier may carry
    a matrix, and then it is a gate that an engine executes; refusing it because
    its name is not in the registry would be a regression, not a clarification.
    """

    identity = torch.eye(4, dtype=torch.complex128)
    carried = Instruction("barrier", (0, 1), {}, identity, {"is_dynamic": True})
    assert torch.allclose(torch.as_tensor(_matrix(carried)).reshape(4, 4), identity)


@pytest.mark.parametrize(
    "label,instruction", list(_UNDECLARED.items()), ids=list(_UNDECLARED)
)
def test_an_undeclared_opcode_is_refused_as_a_capability(
    label: str, instruction: Instruction
) -> None:
    """The refusal is the declared category, not the dictionary key.

    `CapabilityError` is the errors-module boundary's category for a semantic
    capability the framework does not have, and it is the same type the
    stabilizer engine already raises for an instruction it cannot execute. It is
    asserted exactly rather than as a tolerance, because a tolerance is what let
    the key error survive.
    """

    with pytest.raises(CapabilityError) as refusal:
        _matrix(instruction)
    assert refusal.value.category == "capability"
    assert isinstance(refusal.value, FlagQuantumError)
    assert not isinstance(refusal.value, KeyError)
    message = str(refusal.value)
    # The report has to name the instruction and say why no operator exists; the
    # key error carried only the opcode, which is the defect this replaces.
    assert instruction.name in message
    assert "declares no operator" in message
    assert str(tuple(instruction.wires)) in message


def test_a_channel_without_operators_is_refused_by_its_declared_parameter() -> None:
    """A channel is a different repair, so it gets a different report.

    `bit_flip` is declared by the operator registry, so the opcode is not the
    mistake. Its instruction carries the declared probability and no Kraus
    operators, which is a materialization gap: the report has to name the
    parameter that still has to be turned into operators, or the user cannot tell
    this apart from a misspelled gate.
    """

    instruction = Instruction("bit_flip", (0,), {"probability": 0.1}, None, {})
    assert get_operator_schema("bit_flip") is not None
    with pytest.raises(CapabilityError) as refusal:
        _matrix(instruction)
    message = str(refusal.value)
    assert "carrying no Kraus operators" in message
    assert "'probability'" in message
    assert "declares no operator" not in message


def test_the_public_dynamic_entry_point_reports_the_same_refusal() -> None:
    """The defect was reachable by a user, so the repair is asserted from there.

    A barrier is admitted to the IR, so this program is legal and reaches
    `run_dynamic`; the engine has no operator for the directive and no path
    interprets it by name. `KeyError: 'barrier'` used to be the entire report.
    """

    program = CircuitIR(
        2,
        (
            Instruction("x", (0,), {}, None, {}),
            Instruction("barrier", (0, 1), {}, None, {"is_dynamic": True}),
            Instruction("x", (1,), {}, None, {}),
        ),
    )
    with pytest.raises(
        CapabilityError, match="declares no operator for opcode 'barrier'"
    ):
        run_dynamic(DynamicCircuit.from_ir(program), shots=8, seed=1)


def test_a_materialized_channel_is_refused_here_and_executed_by_its_own_engine() -> (
    None
):
    """A channel is not a dense operator, so the two answers have to differ.

    `fq.Circuit.gate` materializes a declared channel's parameters into Kraus
    operators before the instruction exists, and the density-matrix engine applies
    them. `gate_matrix` builds the dense operator of one *unitary* instruction and
    cannot build this one, so it says which operators it was handed and which
    engine applies them. Both halves are asserted, because a refusal that also
    broke the engine that does support channels would be worse than the key error.
    """

    circuit = fq.Circuit(1).gate("bit_flip", 0, probability=0.1)
    instruction = circuit._instructions[0]
    assert instruction.matrix is not None
    with pytest.raises(CapabilityError) as refusal:
        _matrix(instruction)
    assert "is a channel carrying" in str(refusal.value)
    assert "2 Kraus operator(s)" in str(refusal.value)
    density = density_matrix_from_ir(circuit.to_ir(), bsz=1)
    assert torch.allclose(
        torch.diagonal(density[0]).sum(), torch.tensor(1.0 + 0j).to(density.dtype)
    )


@pytest.mark.parametrize(
    "label,carried",
    (
        ("a 4-by-4 operator on one operand", torch.eye(4, dtype=torch.complex128)),
        ("a 3-by-3 operator", torch.eye(3, dtype=torch.complex128)),
        ("a string", "not a matrix"),
    ),
)
def test_a_carried_matrix_that_is_not_a_dense_operator_is_refused(
    label: str, carried: object
) -> None:
    """A carried matrix is checked as an operator, not reshaped and hoped for.

    `Circuit.gate` accepts a matrix for any opcode, so this shape is reachable from
    the public API; today the size mismatch escapes as a `RuntimeError` from
    `reshape` naming no instruction, and a string escapes as a `TypeError` from
    `torch.as_tensor`. The first is the framework's own validation category and the
    second is the wrong-Python-type category the errors-module boundary reserves.
    """

    instruction = Instruction("h", (0,), {}, carried, {})
    expected = TypeError if isinstance(carried, str) else ValidationError
    with pytest.raises(expected) as refusal:
        _matrix(instruction)
    message = str(refusal.value)
    assert "'h'" in message
    assert "operand" in message

"""Fixed singleton gates cannot shorten; supplied values still require validation."""

from unittest.mock import patch

import pytest
import torch

from flagquantum.compiler.optimization import two_qubit_optimization as folding
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS
from flagquantum.simulation.gate_matrix import gate_matrix

pytestmark = pytest.mark.unit
FIXED_GATES = tuple(
    name
    for name, schema in OPERATOR_SCHEMAS.items()
    if schema.unitary and schema.arity == 2 and not schema.parameters
)


@pytest.mark.parametrize("opcode", FIXED_GATES)
@pytest.mark.parametrize("wires", ((0, 1), (1, 0)))
def test_fixed_singleton_is_nonidentity_and_preserved(
    opcode: str, wires: tuple[int, int]
) -> None:
    gate = Instruction(opcode, wires, metadata={"origin": "input"})
    matrix = gate_matrix(
        gate, bsz=1, device=torch.device("cpu"), dtype=torch.complex128
    ).reshape(4, 4)
    assert not torch.allclose(matrix, torch.eye(4, dtype=torch.complex128))
    source = CircuitIR(2, (gate,))
    result = folding.collapse_two_qubit_blocks(source)
    assert result.instructions == (gate,)
    assert result.instructions[0] is gate


def test_many_interfering_fixed_singletons_do_not_construct_matrices() -> None:
    source = CircuitIR(
        3,
        tuple(
            Instruction("cx", (0, 1 if position % 2 else 2)) for position in range(2000)
        ),
    )
    with patch.object(
        folding, "_instruction_matrix", wraps=folding._instruction_matrix
    ) as reader:
        result = folding.collapse_two_qubit_blocks(source)
    assert result.instructions == source.instructions
    assert reader.call_count == 0


@pytest.mark.parametrize("opcode", FIXED_GATES)
def test_fixed_gate_with_identity_override_is_still_removed(opcode: str) -> None:
    matrix = torch.eye(4, dtype=torch.complex128)
    source = CircuitIR(2, (Instruction(opcode, (0, 1), matrix=matrix),))
    assert folding.collapse_two_qubit_blocks(source).instructions == ()
    matrix.copy_(
        gate_matrix(
            Instruction(opcode, (0, 1)),
            bsz=1,
            device=torch.device("cpu"),
            dtype=torch.complex128,
        ).reshape(4, 4)
    )
    assert folding.collapse_two_qubit_blocks(source).instructions == source.instructions


def test_parameterized_identity_and_batched_or_trainable_refusals_stay_checked() -> (
    None
):
    identity = CircuitIR(2, (Instruction("rzz", (0, 1), {"theta": 0.0}),))
    assert folding.collapse_two_qubit_blocks(identity).instructions == ()
    for angle in (torch.tensor([0.0, 0.2]), torch.tensor(0.0, requires_grad=True)):
        source = CircuitIR(2, (Instruction("rzz", (0, 1), {"theta": angle}),))
        assert (
            folding.collapse_two_qubit_blocks(source).instructions
            == source.instructions
        )
        assert source.instructions[0].params["theta"] is angle


def test_extra_parameters_and_routing_markers_keep_existing_validation() -> None:
    gate = Instruction("cx", (0, 1))
    assert isinstance(gate.params, dict)
    gate.params["extra"] = torch.tensor(0.0, requires_grad=True)
    source = CircuitIR(2, (gate, gate))
    assert folding.collapse_two_qubit_blocks(source).instructions == source.instructions
    gate.params.clear()
    assert folding.collapse_two_qubit_blocks(source).instructions == ()
    marked = Instruction("cx", (0, 1), metadata={"routing_phase": "forward"})
    routed = CircuitIR(2, (marked, marked))
    assert folding.collapse_two_qubit_blocks(routed).instructions == routed.instructions

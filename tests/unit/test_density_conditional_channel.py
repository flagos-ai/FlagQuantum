"""Conditional channels act only on the selected measurement branches."""

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.core.ir import Instruction
from flagquantum.noise.channels import amplitude_damping_channel
from flagquantum.simulation.density_matrix import density_matrix_from_ir

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("control", [0, 1])
def test_bell_branch_damping_preserves_inactive_weight(control: int) -> None:
    probability = 0.3
    channel = amplitude_damping_channel(
        torch.tensor(probability, dtype=torch.float64), dtype=torch.complex128
    )
    gates = [Instruction("h", (0,)), Instruction("cx", (0, 1))]
    if control == 0:
        gates.append(Instruction("x", (0,)))
    gates += [
        Instruction("measure", (0,), metadata={"is_dynamic": True, "classical_bit": 0}),
        Instruction(
            "kraus",
            (1,),
            matrix=channel.kraus,
            metadata={"is_channel": True, "conditions": ((0, control),)},
        ),
    ]
    circuit = Circuit(2, dtype=torch.complex128)
    circuit.append(FieldOperation(2, tuple(gates)))
    actual = density_matrix_from_ir(circuit).reshape(4, 4)
    expected = torch.zeros((4, 4), dtype=torch.complex128)
    expected[2 * control, 2 * control] = probability / 2
    expected[2 * control + 1, 2 * control + 1] = (1 - probability) / 2
    expected[2 * (1 - control), 2 * (1 - control)] = 0.5
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=0)

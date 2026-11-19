"""Channel leaves compose with typed calls and clean private workspace."""

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler import estimate_resources
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.component_graph import copy_component_program
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.density_matrix import density_matrix_from_ir

pytestmark = pytest.mark.unit


def test_channel_composition_releases_clean_work_and_counts_emitted_leaves() -> None:
    operators = (
        torch.diag(torch.tensor([1, 0.8], dtype=torch.complex128)),
        torch.tensor([[0, 0.6j], [0, 0]], dtype=torch.complex128),
    )
    registers = (Register("q", quantum_bit_type()),)
    leaf = bit_gate_function(
        "damp",
        registers,
        CircuitIR(
            1,
            (
                Instruction(
                    "kraus", (0,), matrix=operators, metadata={"is_channel": True}
                ),
            ),
        ),
    )
    builder = ComponentProgramBuilder(Signature(registers), functions=(leaf,))
    work = builder.allocate(dtype=quantum_bit_type())
    returned_work = builder.call_dict("damp", {"q": work})["q"]
    assert isinstance(returned_work, RegisterBinding)
    builder.free(returned_work)
    outputs = builder.call_dict("damp", builder.inputs)
    program = copy_component_program(builder.finalize(outputs))
    bound = bind_register_program(program, ((0,),), clean=(1,))
    assert [(event.kind, event.wires) for event in bound.events] == [
        ("allocate", (1,)),
        ("free", (1,)),
    ]
    assert estimate_resources(bound.circuit).operation_counts == {"kraus": 2}
    state = torch.tensor([1, 0, 1j, 0], dtype=torch.complex128) / 2**0.5
    circuit = Circuit(2, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(2, bound.circuit.instructions))
    density = density_matrix_from_ir(circuit).reshape(4, 4)
    expected = torch.zeros_like(density)
    for operator in operators:
        branch = torch.kron(operator, torch.eye(2, dtype=torch.complex128)) @ state
        expected += branch[:, None] @ branch.conj()[None, :]
    torch.testing.assert_close(density, expected, atol=1e-14, rtol=0)
    assert torch.count_nonzero(density[1::2]) == 0
    assert torch.count_nonzero(density[:, 1::2]) == 0

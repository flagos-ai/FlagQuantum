"""Execute owned matrix leaves through typed wrapping, expansion and binding."""

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
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register

pytestmark = pytest.mark.unit


def test_matrix_survives_graph_copy_and_append_with_clean_work() -> None:
    matrix = torch.tensor([[0, 1j], [1j, 0]], dtype=torch.complex128)
    registers = (Register("q", quantum_bit_type()),)
    leaf = bit_gate_function(
        "leaf",
        registers,
        CircuitIR(n_wires=1, instructions=(Instruction("any", (0,), matrix=matrix),)),
    )
    builder = ComponentProgramBuilder(Signature(registers), functions=(leaf,))
    work = builder.allocate(dtype=quantum_bit_type())
    forward_work = builder.call_dict("leaf", {"q": work})["q"]
    assert isinstance(forward_work, RegisterBinding)
    # A supplied exact inverse leaves the work clean and tests full global phase.
    inverse = bit_gate_function(
        "inverse",
        registers,
        CircuitIR(
            n_wires=1, instructions=(Instruction("any", (0,), matrix=matrix.mH),)
        ),
    )
    builder.define(inverse)
    restored_work = builder.call_dict("inverse", {"q": forward_work})["q"]
    assert isinstance(restored_work, RegisterBinding)
    builder.free(restored_work)
    q = builder.call_dict("leaf", builder.inputs)["q"]
    module = builder.finalize({"q": q})
    copied = copy_component_program(module)
    bound = bind_register_program(copied, ((0,),), clean=(1,))
    assert estimate_resources(bound.circuit).operation_counts == {"any": 3}
    assert len(bound.events) == 2
    state = torch.tensor([1, 0, 1j, 0], dtype=torch.complex128) / 2**0.5
    expected = torch.tensor([-1, 0, 1j, 0], dtype=torch.complex128) / 2**0.5
    circuit = Circuit(2, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(2, bound.circuit.instructions))
    torch.testing.assert_close(circuit.state().flatten(), expected, atol=2e-12, rtol=0)
    assert torch.count_nonzero(circuit.state().flatten()[1::2]) == 0
    # Physical consumers own fresh storage and cannot mutate semantic rows.
    assert bound.circuit.instructions[0].matrix is not None
    bound.circuit.instructions[0].matrix.zero_()
    rebound = bind_register_program(module, ((0,),), clean=(1,))
    assert rebound.circuit.instructions[0].matrix is not None
    torch.testing.assert_close(rebound.circuit.instructions[0].matrix, matrix)

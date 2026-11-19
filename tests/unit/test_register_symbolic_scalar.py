"""Nested register calls bind owned angles through the normal append entry."""

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
from flagquantum.core.parameters import Parameter

pytestmark = pytest.mark.unit


def test_nested_symbolic_calls_keep_clean_work_phase_and_gradient() -> None:
    theta = Parameter("theta")
    angle = 2 * theta - 0.125
    registers = (Register("q", quantum_bit_type()),)
    leaf = bit_gate_function(
        "rotate", registers, CircuitIR(1, (Instruction("rz", (0,), {"theta": angle}),))
    )
    inner = ComponentProgramBuilder(Signature(registers), (leaf,))
    nested = inner.finalize(inner.call_dict("rotate", inner.inputs))
    outer = ComponentProgramBuilder(
        Signature(registers), nested.functions, name="outer"
    )
    work = outer.allocate(dtype=quantum_bit_type())
    returned = outer.call_dict("rotate", {"q": work})["q"]
    assert isinstance(returned, RegisterBinding)
    outer.free(returned)
    program = copy_component_program(
        outer.finalize(outer.call_dict("main", outer.inputs))
    )
    bound = bind_register_program(program, ((0,),), clean=(1,), entry="outer")
    assert [(event.kind, event.wires) for event in bound.events] == [
        ("allocate", (1,)),
        ("free", (1,)),
    ]
    assert estimate_resources(bound.circuit).operation_counts == {"rz": 2}
    initial = torch.tensor([1, 0, 1j, 0], dtype=torch.complex128) / 2**0.5
    circuit = Circuit(2, inputs=initial, dtype=torch.complex128).append(
        FieldOperation(2, bound.circuit.instructions)
    )
    assert circuit.parameter_names == ("theta",)
    with pytest.raises(ValueError, match="Unbound circuit parameter"):
        circuit.state()
    value = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    actual = circuit.bind_parameters({"theta": value}).state().flatten()
    physical_angle = 2 * value - 0.125
    expected = (
        torch.stack(
            (torch.exp(-1j * physical_angle), value * 0j, value * 0j + 1j, value * 0j)
        )
        / 2**0.5
    )
    torch.testing.assert_close(actual, expected, atol=3e-15, rtol=0)
    assert torch.count_nonzero(actual[1::2]) == 0
    actual_grad = torch.autograd.grad(actual[0].real, value, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected[0].real, value)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=3e-15, rtol=0)

"""Literal register/constant addition paths keep source counterexamples explicit."""

from collections import Counter
from importlib import import_module
from typing import Any

import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.field_addition_decomposition import (
    field_addition_catalog_program,
)
from flagquantum import Circuit
from flagquantum.compiler.registers.finite_field.addition_component import (
    field_addition_decomposition_program,
    field_addition_program,
)
from flagquantum.compiler.registers.finite_field.addition_model import (
    field_add_constant_calls,
    field_addition_calls,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.finite_field_addition import field_add_classical_values
from flagquantum.simulation.finite_field_encoding import FiniteFieldEncoding
from tests.unit.test_field_multiplication import assert_gate_streams, reference_gates

np = pytest.importorskip("numpy")
pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "characteristic,degree", ((2, 1), (2, 3), (3, 1), (3, 2), (5, 1))
)
def test_actual_source_degree_prefix_gates_all_complex_states_and_separate_model(
    characteristic: int, degree: int
) -> None:
    from qualtran._infra.data_types import QGF
    from qualtran.bloqs.gf_arithmetic.gf2_addition import GF2Addition
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount

    source = GF2Addition(QGF(characteristic, degree))
    encoding = FiniteFieldEncoding(characteristic, degree)
    dtype = encoding.register_type()
    width = int(dtype.quantum_bits)
    program = field_addition_decomposition_program(dtype)
    assert verify_program_record(program) == ()
    logical = 2 * width
    wires = tuple(reversed(range(logical)))
    inputs = (wires[:width], wires[width:])
    bound = bind_register_program(program, inputs)
    assert bound.outputs == inputs and bound.events == ()
    expected_gates = tuple(
        Instruction("cx", (inputs[0][i], inputs[1][i])) for i in range(degree)
    )
    assert bound.circuit.instructions == expected_gates
    original, returned = reference_gates(
        source, dict(zip(("x", "y"), inputs, strict=True)), (), repair_adjoint=False
    )
    assert_gate_streams([(g.name, g.wires) for g in expected_gates], original)
    assert returned == dict(zip(("x", "y"), inputs, strict=True))
    histogram = Counter(g.name for g in expected_gates)
    assert histogram["cx"] == degree == get_cost_value(source, QECGatesCost()).clifford
    assert list(field_addition_calls(degree).values()) == [degree]
    assert get_cost_value(source, QubitCount()) == logical
    assert program == field_addition_catalog_program(
        "GF2Addition", Signature(program.functions[0].registers or ()), {}
    )
    # Complete complex states include every raw encoding, an entangled spectator
    # and a clean unused wire. Reversing both words only permutes source ports.
    initial = torch.zeros(1 << (logical + 2), dtype=torch.complex128)
    initial[::2] = torch.randn(
        1 << (logical + 1),
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(513),
    )
    initial /= initial.norm()
    original_operator = torch.as_tensor(
        source.decompose_bloq().flatten().tensor_contract(), dtype=torch.complex128
    )
    order = (*wires, logical, logical + 1)
    ordered = initial.reshape((2,) * (logical + 2)).permute(order)
    transformed = (original_operator @ ordered.reshape(1 << logical, 4)).reshape(
        (2,) * (logical + 2)
    )
    inverse_order = tuple(order.index(i) for i in range(logical + 2))
    expected = transformed.permute(inverse_order).reshape(-1)
    circuit = Circuit(logical + 2, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(logical + 2, expected_gates))
    torch.testing.assert_close(circuit.state().flatten(), expected, atol=2e-12, rtol=0)
    circuit.append(FieldOperation(logical + 2, expected_gates))
    torch.testing.assert_close(circuit.state().flatten(), initial, atol=2e-12, rtol=0)
    assert torch.count_nonzero(circuit.state().flatten()[1::2]) == 0
    if characteristic != 2:
        with pytest.raises(ValueError):
            field_addition_program(dtype)
        x, y = encoding.gf_type(1), encoding.gf_type(1)
        native_values = field_add_classical_values(encoding, x, y)
        original_values = source.on_classical_vals(x=x, y=y)
        assert native_values == (original_values["x"], original_values["y"])
        assert int(native_values[1]) == 2
        # The actual source quantum circuit leaves y=1 unchanged when the high
        # degree-prefix control bits of x=1 are zero. This is the source defect.
        assert degree < width
        column = (1 << width) | 1
        assert int(torch.argmax(original_operator[:, column].abs())) == column


@pytest.mark.parametrize("kind", ("zero_bool", "one_bool", "numpy", "field_scalar"))
def test_source_constant_integral_aliases_reach_actual_catalog_and_model(
    kind: str,
) -> None:
    from qualtran.bloqs.gf_arithmetic.gf2_add_k import GF2AddK
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value

    encoding = FiniteFieldEncoding(2, 3)
    constants: dict[str, Any] = {
        "zero_bool": False,
        "one_bool": True,
        "numpy": np.int64(5),
        "field_scalar": encoding.gf_type(5),
    }
    constant = constants[kind]
    source = GF2AddK(3, constant)
    dtype = encoding.register_type()
    signature = Signature.build_from_dtypes(x=dtype)
    program = field_addition_catalog_program("GF2AddK", signature, {"k": constant})
    assert program is not None
    bound = bind_register_program(program, ((2, 0, 1),))
    expected = tuple(
        Instruction("x", ((2, 0, 1)[i],)) for i, bit in enumerate(source._bits_k) if bit
    )
    assert bound.outputs == ((2, 0, 1),) and bound.circuit.instructions == expected
    assert bound.events == ()
    allocator_type = import_module(
        "qualtran.resource_counting._call_graph"
    ).SympySymbolAllocator
    assert isinstance(allocator_type, type)
    assert list(source.build_call_graph(allocator_type()).values()) == [len(expected)]
    if kind == "field_scalar":
        with pytest.raises(TypeError, match="unhashable"):
            get_cost_value(source, QECGatesCost())
    else:
        assert len(expected) == get_cost_value(source, QECGatesCost()).clifford
    assert list(field_add_constant_calls(3, constant).values()) == [len(expected)]
    assert (
        field_addition_catalog_program("GF2AddK", signature, {"k": sympy.Integer(0)})
        is None
    )
    assert list(field_add_constant_calls(3, sympy.Integer(0)).values()) == [3]

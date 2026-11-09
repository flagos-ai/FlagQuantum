"""Original numerical values and phase-sensitive native Pauli consumers."""

from itertools import product

import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_pauli_values import (
    cz_basis_phase,
    diagonal_classical_values,
    x_classical_value,
    z_basis_phase,
)
from tests.unit.test_classical_comparison import compare_calls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
VALUES = [
    0,
    1,
    2,
    -1,
    True,
    np.uint8(255),
    np.int8(127),
    np.array([0]),
    np.array([1]),
    np.array([0, 1]),
    sympy.Integer(1),
    sympy.Symbol("q"),
    None,
]


@pytest.mark.parametrize("value", VALUES)
def test_original_x_modulo_types_and_diagnostics(value):
    from qualtran.bloqs.basic_gates import XGate

    compare_calls(
        lambda q: tuple(XGate().on_classical_vals(q).values()),
        lambda q: (x_classical_value(q),),
        (value,),
    )


@pytest.mark.parametrize("value", VALUES)
def test_original_z_phase_exact_equality_and_ambiguous_truth(value):
    from qualtran.bloqs.basic_gates import ZGate

    compare_calls(
        lambda q: (ZGate().basis_state_phase(q),),
        lambda q: (z_basis_phase(q),),
        (value,),
    )


@pytest.mark.parametrize("control,target", list(product(VALUES, repeat=2)))
def test_original_cz_phase_short_circuit_and_types(control, target):
    from qualtran.bloqs.basic_gates import CZ

    compare_calls(
        lambda c, q: (CZ().basis_state_phase(c, q),),
        lambda c, q: (cz_basis_phase(c, q),),
        (control, target),
    )


@pytest.mark.parametrize("kind", ["z", "cz"])
def test_diagonal_named_values_preserve_order_aliases_extras_and_empty_mapping(kind):
    from qualtran.bloqs.basic_gates import CZ, ZGate

    source = ZGate() if kind == "z" else CZ()
    shared = np.array([0, 1])
    values = {"extra": shared, "q": shared, "q1": np.int8(1), "q2": None}
    expected = source.on_classical_vals(**values)
    actual = diagonal_classical_values(**values)
    assert list(actual) == list(expected) == list(values)
    assert actual is not values
    for key in values:
        assert actual[key] is expected[key] is values[key]
    assert diagonal_classical_values() == source.on_classical_vals() == {}


@pytest.mark.parametrize("kind", ["x", "z", "cz"])
def test_source_tensor_and_complete_complex_native_action_inverse_and_resources(kind):
    from dataclasses import asdict

    import torch
    from qualtran.bloqs.basic_gates import CZ, XGate, ZGate
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.circuit import Circuit
    from flagquantum.compiler._component_trace.qec_gate_cost import query_qec_gate_cost
    from flagquantum.compiler._component_trace.resource_generalization import (
        CliffordCall,
    )
    from flagquantum.compiler.registers.primitives.pauli_gate import pauli_gate_ir

    source = {"x": XGate(), "z": ZGate(), "cz": CZ()}[kind]
    arity = 2 if kind == "cz" else 1
    matrix = np.zeros((1 << arity, 1 << arity), dtype=np.complex128)
    for bits in range(1 << arity):
        if kind == "x":
            matrix[x_classical_value(bits), bits] = 1
        elif kind == "z":
            matrix[bits, bits] = z_basis_phase(bits)
        else:
            matrix[bits, bits] = cz_basis_phase(bits >> 1, bits & 1)
    np.testing.assert_allclose(source.tensor_contract(), matrix, rtol=0, atol=2e-12)
    grid = torch.arange(1, 9, dtype=torch.float64)
    initial = torch.complex(grid, 3 * grid - 5)
    initial /= initial.norm()
    expected = torch.zeros_like(initial)
    for left, right, spectator in product((0, 1), repeat=3):
        before = (left << 2) | (right << 1) | spectator
        after, phase = before, 1
        if kind == "x":
            after = (int(x_classical_value(left)) << 2) | (right << 1) | spectator
        elif kind == "z":
            phase = z_basis_phase(left)
        else:
            phase = cz_basis_phase(left, right)
        expected[after] = phase * initial[before]
    gates = (
        pauli_gate_ir("z", 1, controls=(0,), values=(1,))
        if kind == "cz"
        else pauli_gate_ir(kind, 0)
    )
    circuit = Circuit(3, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(3, gates))
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=2e-12)
    circuit.append(FieldOperation(3, gates))
    torch.testing.assert_close(circuit.state().flatten(), initial, rtol=0, atol=2e-12)
    for legacy in (False, True):
        model = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
        native = query_qec_gate_cost(CliffordCall(kind, arity), legacy_shims=legacy)
        assert (
            {name: count for name, count in asdict(native).items() if count}
            == model.asdict()
            == {"clifford": 1}
        )
        assert model.total_t_count() == model.measurement == 0

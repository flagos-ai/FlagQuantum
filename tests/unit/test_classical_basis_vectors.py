"""Source vector constructor/callback order and numerical ownership boundaries."""

import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_basis_vectors import (
    bit_basis_classical_values,
    integer_basis_classical_values,
)
from tests.unit.test_classical_comparison import compare_calls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "bit", [False, True, 2, -1, np.uint8(1), np.array([0, 1]), sympy.Symbol("b")]
)
@pytest.mark.parametrize(
    "width", [0, 1, 2, np.int64(1), sympy.Integer(1), sympy.Symbol("n")]
)
@pytest.mark.parametrize("state", [False, True])
@pytest.mark.parametrize("incoming", [None, 0, 1, np.array([0, 1])])
def test_original_z_vector_width_truthiness_and_callback_assertions(
    bit, width, state, incoming
):
    from qualtran.bloqs.basic_gates.z_basis import _ZVector

    compare_calls(
        lambda b, n, q: tuple(_ZVector(b, state, n).on_classical_vals(q=q).values()),
        lambda b, n, q: tuple(
            bit_basis_classical_values(b, width=n, state=state, value=q).values()
        ),
        (bit, width, incoming),
    )


@pytest.mark.parametrize(
    "value,width",
    [
        (0, 0),
        (0, 1),
        (1, 1),
        (2, 1),
        (-1, 3),
        (7, 3),
        (8, 3),
        (0, -1),
        (np.uint8(3), np.int64(2)),
        (0.5, 2),
        (np.array([1]), 2),
        (np.array([1, 2]), 3),
        (sympy.Integer(-1), 2),
        (-1, sympy.Integer(2)),
        (sympy.Symbol("x"), sympy.Symbol("n")),
    ],
)
@pytest.mark.parametrize("state", [False, True])
@pytest.mark.parametrize("incoming", [None, 0, 1, -1, np.array([0, 1])])
def test_original_integer_vector_validation_before_creation_or_consumption(
    value, width, state, incoming
):
    from qualtran.bloqs.basic_gates import IntEffect, IntState

    cls = IntState if state else IntEffect
    compare_calls(
        lambda v, n, q: tuple(cls(v, n).on_classical_vals(val=q).values()),
        lambda v, n, q: tuple(
            integer_basis_classical_values(v, n, incoming=q, state=state).values()
        ),
        (value, width, incoming),
    )


def test_creation_returns_exact_configured_object_and_effect_checks_before_discard():
    value = np.array([1])
    assert integer_basis_classical_values(value, 2)["val"] is value
    assert (
        integer_basis_classical_values(value, 2, incoming=np.array([1]), state=False)
        == {}
    )
    with pytest.raises(ValueError):
        integer_basis_classical_values(-1, 3, incoming=7)
    with pytest.raises(AssertionError):
        integer_basis_classical_values(1, 3, incoming=7)
    assert bit_basis_classical_values(2) == {"q": 1}


def test_expr_bypass_is_not_the_general_symbolic_protocol():
    class SymbolicInteger(int):
        def is_symbolic(self):
            return True

    from qualtran.bloqs.basic_gates import IntState

    value = SymbolicInteger(-1)
    with pytest.raises(ValueError):
        IntState(value, 3)
    with pytest.raises(ValueError):
        integer_basis_classical_values(value, 3)
    assert integer_basis_classical_values(sympy.Integer(-1), 3)["val"] == -1


@pytest.mark.parametrize("missing", ["sympy", "sympy_dependency"])
def test_optional_symbolic_import_does_not_hide_nested_dependency_failures(
    monkeypatch, missing
):
    import builtins

    original = builtins.__import__

    def import_module(name, *args, **kwargs):
        if name == "sympy":
            raise ModuleNotFoundError(name=missing)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    if missing == "sympy":
        assert integer_basis_classical_values(3, 2) == {"val": 3}
        with pytest.raises(ValueError):
            integer_basis_classical_values(-1, 2)
    else:
        with pytest.raises(ModuleNotFoundError) as error:
            integer_basis_classical_values(3, 2)
        assert error.value.name == missing


@pytest.mark.parametrize(
    "width,value", [(n, x) for n in (1, 2, 3) for x in range(1 << n)]
)
def test_raw_state_and_effect_values_bind_existing_complex_quantum_consumers(
    width, value
):
    import torch

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.circuit import Circuit
    from flagquantum.compiler.registers.layout.int_vector import (
        int_effect_bindings,
        int_state_ir,
    )
    from flagquantum.simulation.tensor_network.entrypoints import (
        build_tensor_network,
        tensor_network_basis_effect,
    )

    prepared = integer_basis_classical_values(value, width)["val"]
    assert type(prepared) is int
    spectator = torch.tensor([1j, 2 - 1j], dtype=torch.complex128)
    spectator /= spectator.norm()
    initial = torch.zeros(1 << (width + 1), dtype=torch.complex128)
    initial[:2] = spectator
    expected = torch.zeros_like(initial)
    expected[2 * prepared : 2 * prepared + 2] = spectator
    circuit = Circuit(width + 1, inputs=initial, dtype=torch.complex128)
    gates = int_state_ir(tuple(range(width)), prepared)
    circuit.append(FieldOperation(width + 1, gates))
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=2e-12)
    assert (
        integer_basis_classical_values(value, width, incoming=prepared, state=False)
        == {}
    )
    projected = tensor_network_basis_effect(
        build_tensor_network(circuit, dtype=torch.complex128),
        int_effect_bindings(tuple(range(width)), prepared),
    )
    torch.testing.assert_close(projected.flatten(), spectator, rtol=0, atol=2e-12)
    circuit.append(FieldOperation(width + 1, gates))
    torch.testing.assert_close(circuit.state().flatten(), initial, rtol=0, atol=2e-12)

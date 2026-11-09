"""Native numerical dimensions retain original symbolic descriptor behavior."""

import numpy as np
import pytest
import sympy

from flagquantum.core.value_parameters import is_symbolic_value
from flagquantum.simulation.value_dimensions import (
    SymbolicArrayShape,
    SymbolicSequenceLength,
    value_length,
    value_shape,
)
from tests.unit.test_classical_comparison import compare_calls, same_value

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (0,),
        (-1,),
        (2, 3),
        (sympy.Symbol("n"), 2),
        (np.int64(2),),
        [2, 3],
        None,
        np.array([2, 3]),
    ],
)
def test_original_array_constructor_tuple_guard_and_field_identity(shape):
    from qualtran.symbolics import Shaped

    compare_calls(
        lambda s: (Shaped(s).shape,), lambda s: (SymbolicArrayShape(s).shape,), (shape,)
    )
    if isinstance(shape, tuple):
        source, native = Shaped(shape), SymbolicArrayShape(shape)
        assert source.shape is native.shape is shape
        assert source.is_symbolic() is native.is_symbolic() is True
        assert is_symbolic_value(native) is True
        assert value_shape(native) is shape
        for obj in (source, native):
            with pytest.raises(AttributeError):
                obj.shape = ()


@pytest.mark.parametrize(
    "length",
    [0, 1, -1, 2.5, True, np.int64(3), sympy.Integer(2), sympy.Symbol("n"), None],
)
def test_original_sequence_length_is_not_builtin_len_or_a_coerced_integer(length):
    from qualtran.symbolics import HasLength, slen

    source, native = HasLength(length), SymbolicSequenceLength(length)
    assert source.n is native.length is length
    same_value(value_length(native), slen(source))
    assert is_symbolic_value(native) is source.is_symbolic() is True
    for obj in (source, native):
        with pytest.raises(TypeError):
            len(obj)


@pytest.mark.parametrize("shape", [(), (0,), (-1,), (2, 3), (sympy.Symbol("n"), 2)])
def test_original_shaped_length_uses_first_dimension_including_empty_failure(shape):
    from qualtran.symbolics import Shaped, slen

    compare_calls(
        lambda: (slen(Shaped(shape)),),
        lambda: (value_length(SymbolicArrayShape(shape)),),
        (),
    )


@pytest.mark.parametrize(
    "value", [[], [1, 2], "abc", range(3), np.zeros((2, 3)), np.array(1), None]
)
def test_original_ordinary_length_and_shape_diagnostics(value):
    from qualtran.symbolics import shape, slen

    compare_calls(lambda v: (slen(v),), lambda v: (value_length(v),), (value,))
    compare_calls(lambda v: (shape(v),), lambda v: (value_shape(v),), (value,))


def test_shape_descriptor_flows_to_existing_activation_and_both_controlled_consumers():
    from qualtran import CtrlSpec
    from qualtran.symbolics import Shaped

    from flagquantum.simulation.classical_control_activation import (
        classical_controls_active,
    )
    from flagquantum.simulation.classical_controlled_values import (
        controlled_basis_phase,
        controlled_classical_values,
    )

    shape = (2, 2)
    source, native = Shaped(shape), SymbolicArrayShape(shape)
    with pytest.raises(ValueError):
        CtrlSpec(cvs=source).is_active(np.zeros(shape, dtype=int))
    with pytest.raises(ValueError):
        classical_controls_active((np.zeros(shape, dtype=int),), (native,))

    def forbidden(**values):
        pytest.fail("Symbolic activation must reject before calling the child")

    values = {"ctrl": np.zeros(shape, dtype=int), "q": 0}
    with pytest.raises(ValueError):
        controlled_classical_values(values, ("ctrl",), ("q",), (native,), forbidden)
    with pytest.raises(ValueError):
        controlled_basis_phase(values, ("ctrl",), ("q",), (native,), forbidden)


@pytest.mark.parametrize("legacy", [False, True])
def test_shape_only_controls_use_existing_predicate_model_without_materializing_values(
    legacy,
):
    from dataclasses import asdict
    from math import prod

    from qualtran import CtrlSpec
    from qualtran.bloqs.mcmt.ctrl_spec_and import CtrlSpecAnd
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
    from qualtran.symbolics import Shaped

    from flagquantum.compiler.registers.control.control_predicate_resource_model import (
        control_predicate_resource_formula,
    )

    native_shape = SymbolicArrayShape((2, 2))
    source = CtrlSpecAnd(CtrlSpec(cvs=Shaped(native_shape.shape)))
    native = control_predicate_resource_formula(
        prod(native_shape.shape), legacy_shims=legacy
    )
    expected = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
    assert {
        name: count
        for name, count in asdict(native).items()
        if name != "logical_qubits" and count
    } == expected.asdict()
    assert native.logical_qubits == get_cost_value(source, QubitCount())


def test_concrete_matrix_controls_original_stream_all_branches_and_clean_inverse():
    from itertools import product

    import torch
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from examples.algorithms.numerical_value_dimensions import ShapedControlledX
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.simulation.classical_control_activation import (
        classical_controls_active,
    )
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    pattern = np.array([[0, 1], [1, 0]])
    source = XGate().controlled(CtrlSpec(cvs=pattern))
    circuit = DynamicCircuit(9)
    circuit.append(ShapedControlledX())
    gates = tuple(circuit.to_ir().instructions)
    original = reference_arithmetic_gates(
        source,
        {source.ctrl_reg_names[0]: (0, 1, 2, 3), "q": (4,)},
        (5, 6, 7),
        {5: 0, 6: 1, 7: 2},
    )
    assert_source(gates, original)
    grid = torch.arange(1, 65, dtype=torch.float64)
    weights = torch.complex(grid, 3 * grid - 5)
    weights /= weights.norm()
    initial = torch.zeros(512, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for word, target, spectator in product(range(16), (0, 1), (0, 1)):
        values = np.array([(word >> i) & 1 for i in (3, 2, 1, 0)]).reshape(2, 2)
        active = classical_controls_active((values,), (pattern,))
        before = (word << 5) | (target << 4) | spectator
        after = (word << 5) | ((target ^ active) << 4) | spectator
        initial[before] = weights[(word << 2) | (target << 1) | spectator]
        expected[after] = initial[before]
    for history in product((0, 1), repeat=3):
        actual = forced_branch(initial, gates, history)
        torch.testing.assert_close(actual, expected / 2**1.5, rtol=0, atol=2e-12)
        residual = actual.reshape(32, 8, 2)[:, 1:, :]
        torch.testing.assert_close(
            residual, torch.zeros_like(residual), rtol=0, atol=2e-12
        )
    for history in product((0, 1), repeat=6):
        torch.testing.assert_close(
            forced_branch(initial, gates + gates, history),
            initial / 8,
            rtol=0,
            atol=2e-12,
        )
    for legacy in (False, True):
        model = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
        assert model.total_t_count() == sum(g.name in ("t", "tdg") for g in gates) == 12
        assert model.measurement == sum(g.name == "measure" for g in gates) == 3
    assert get_cost_value(source, QubitCount()) == 8

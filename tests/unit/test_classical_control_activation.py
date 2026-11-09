"""Source activation short circuits and a shared measured quantum consumer."""

from itertools import product

import numpy as np
import pytest
import sympy

from flagquantum.simulation.classical_control_activation import (
    classical_controls_active,
)
from tests.unit.test_classical_comparison import compare_calls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
PATTERNS = [
    (np.array(0),),
    (np.array(1),),
    (np.array([0, 1]),),
    (np.array([[0, 1], [1, 0]]),),
    (np.array([], dtype=int),),
    (np.array(0), np.array([1, 0])),
]
VALUES = [
    (),
    (0,),
    (1,),
    (True,),
    (np.int8(1),),
    (0.0,),
    (None,),
    (np.array([0, 1]),),
    (np.array([[0, 1], [1, 0]]),),
    (np.array([], dtype=int),),
    (0, [1, 0]),
    (1, [1, 0]),
    (0, 7),
    (0, [1, 0], 1),
]


@pytest.mark.parametrize("patterns", PATTERNS)
@pytest.mark.parametrize("values", VALUES)
def test_original_bound_activation_shapes_types_and_first_mismatch(patterns, values):
    from qualtran import CtrlSpec, QBit

    source = CtrlSpec(qdtypes=(QBit(),) * len(patterns), cvs=patterns)
    compare_calls(
        lambda vals: (source.is_active(*vals),),
        lambda vals: (classical_controls_active(vals, patterns),),
        (values,),
    )


@pytest.mark.parametrize("width", [2, np.int64(2), sympy.Integer(2), sympy.Symbol("n")])
@pytest.mark.parametrize("values", [(), (0,), (1,), (np.array([1]),)])
def test_native_symbolic_parameters_preserve_original_dtype_guard_order(width, values):
    from qualtran import CtrlSpec, QUInt

    patterns = (np.array(1),)
    source = CtrlSpec(qdtypes=QUInt(width), cvs=1)
    compare_calls(
        lambda vals: (source.is_active(*vals),),
        lambda vals: (
            classical_controls_active(vals, patterns, symbolic_parameters=(width,)),
        ),
        (values,),
    )


def test_inactive_first_clause_does_not_touch_later_array_conversion():
    from qualtran import CtrlSpec, QBit

    class NeverArray:
        def __array__(self, *args, **kwargs):
            raise AssertionError("Later control must not be converted")

    patterns = (np.array(0), np.array([1, 0]))
    source = CtrlSpec(qdtypes=(QBit(), QBit()), cvs=patterns)
    assert source.is_active(1, NeverArray()) is False
    assert classical_controls_active((1, NeverArray()), patterns) is False
    for call in (
        lambda: source.is_active(0, 7),
        lambda: classical_controls_active((0, 7), patterns),
    ):
        with pytest.raises(ValueError):
            call()


def test_real_symbolic_classifier_replacement_precedes_value_count(monkeypatch):
    import flagquantum.simulation.classical_control_activation as module

    calls = []

    def classifier(*values):
        calls.append(values)
        return True

    monkeypatch.setattr(module, "is_symbolic_value", classifier)
    with pytest.raises(ValueError, match="symbolic"):
        classical_controls_active((), (np.array(1),), symbolic_parameters=("width",))
    assert calls == [("width",)]


def test_word_activation_original_stream_full_complex_branches_and_resource_counts():
    import torch
    from qualtran import CtrlSpec, QUInt
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.compiler.registers.primitives.pauli_gate import pauli_gate_ir
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    source = XGate().controlled(CtrlSpec(qdtypes=QUInt(2), cvs=2))
    gates = pauli_gate_ir(
        "x", 2, controls=(0, 1), values=(1, 0), workspace=(3,), classical_bits=(0,)
    )
    expected_stream = reference_arithmetic_gates(
        source, {source.ctrl_reg_names[0]: (0, 1), "q": (2,)}, (3,), {3: 0}
    )
    assert_source(gates, expected_stream)
    grid = torch.arange(1, 17, dtype=torch.float64)
    amplitudes = torch.complex(grid, 3 * grid - 5)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(32, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for word, target, spectator in product(range(4), (0, 1), (0, 1)):
        active = classical_controls_active(
            (word,), (np.array(2),), symbolic_parameters=(2,)
        )
        before = (word << 3) | (target << 2) | spectator
        after = (word << 3) | ((target ^ active) << 2) | spectator
        initial[before] = amplitudes[(word << 2) | (target << 1) | spectator]
        expected[after] = initial[before]
    for history in ((0,), (1,)):
        actual = forced_branch(initial, gates, history)
        torch.testing.assert_close(actual, expected / 2**0.5, rtol=0, atol=2e-12)
        residual = actual.reshape(8, 2, 2)[:, 1, :]
        torch.testing.assert_close(
            residual, torch.zeros_like(residual), rtol=0, atol=2e-12
        )
    for legacy in (False, True):
        cost = get_cost_value(source, QECGatesCost(legacy_shims=legacy))
        assert cost.total_t_count() == sum(g.name in ("t", "tdg") for g in gates) == 4
        assert cost.measurement == sum(g.name == "measure" for g in gates) == 1
    assert get_cost_value(source, QubitCount()) == 4

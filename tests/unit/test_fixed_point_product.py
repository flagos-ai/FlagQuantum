"""Original fixed-point layout, D5 components, and measured cleanup."""

from cmath import exp
from collections import Counter
from itertools import product
from math import sqrt

import pytest
import torch

from flagquantum.compiler.registers.and_compute import and_compute_ir, and_uncompute_ir
from flagquantum.compiler.registers.fixed_point_scaling import (
    fixed_point_product_ir,
    fixed_point_product_workspace_size,
)
from flagquantum.core.ir import CircuitIR
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_fixed_point_scaling import truncated
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def layout(d):
    size = fixed_point_product_workspace_size(d)
    offset = 3 * d
    return (
        tuple(range(d)),
        tuple(range(d, 2 * d)),
        tuple(range(2 * d, offset)),
        tuple(range(offset, offset + size)),
        tuple(range(offset + size, offset + 2 * size)),
        tuple(range(2 * size)),
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("d", [1, 2, 3, 4, 6])
def test_fixed_point_product_source_contract_children_and_all_histories(d):
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError, QFxp, QUInt
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd
    from qualtran.bloqs.arithmetic.multiplication import MultiplyTwoReals
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    bound = layout(d)
    left, right, result, loaded, carries, slots = bound
    source = MultiplyTwoReals(d)
    assert source.signature.get_right("result").dtype == QFxp(d, d)
    assert (
        source.signature.get_left("a").dtype
        == source.signature.get_left("b").dtype
        == QFxp(d, d)
    )
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()
    with pytest.raises(NotImplementedError):
        source.tensor_contract()
    assert source.on_classical_vals(a=0, b=0) is NotImplemented
    model = d * d - d - 1
    assert get_cost_value(source, QECGatesCost()).toffoli == model

    def word(value):
        return tuple(value >> i & 1 for i in reversed(range(d)))

    before, after = [], []
    clean = (0,) * (len(loaded) + len(carries))
    for a, b in product(range(1 << d), repeat=2):
        value = truncated(a, b, d)
        assert 0 <= a * b - (value << d) <= b.bit_count() * ((1 << d) - 1)
        prefix = word(a) + word(b)
        before.append(prefix + (0,) * d + clean)
        after.append(prefix + word(value) + clean)
    for adjoint in (False, True):
        gates = fixed_point_product_ir(*bound, adjoint=adjoint)
        if d == 1:
            assert gates == ()
        else:
            copy = (
                and_uncompute_ir(right[-2], left[0], result[-1], 0)
                if adjoint
                else and_compute_ir(right[-2], left[0], result[-1])
            )
            additions = []
            steps = tuple(enumerate(reversed(right[:-2]), start=2))
            for w, control in reversed(steps) if adjoint else steps:
                add = CAdd(QUInt(w), QUInt(w + 1))
                work = (*loaded[:w], *carries[:w])
                bits = (*slots[:w], *slots[len(loaded) : len(loaded) + w])
                additions.extend(
                    reference_arithmetic_gates(
                        add.adjoint() if adjoint else add,
                        {"ctrl": (control,), "a": left[:w], "b": result[-w - 1 :]},
                        work,
                        dict(zip(work, bits, strict=True)),
                    )
                )
            expected = (*additions, *copy) if adjoint else (*copy, *additions)
            assert_source(gates, expected)
        counts = Counter(g.name for g in gates)
        events = max(0, model - (0 if adjoint else 1))
        assert counts["measure"] == counts["reset"] == events
        assert counts["t"] + counts["tdg"] == 4 * max(0, model - adjoint)
        incoming, outgoing = (after, before) if adjoint else (before, after)
        assert certified_and_channel_map(gates, tuple(incoming)) == (
            tuple(outgoing),
            events,
        )


@pytest.mark.parametrize("d", [1, 2, 3, 4])
def test_fixed_point_product_complex_forward_and_erase(d):
    bound = layout(d)
    low = 2 * len(bound[3]) + 1
    width = 3 * d + low
    initial, expected = {}, {}
    for a, b, s in product(range(1 << d), range(1 << d), (0, 1)):
        i = (((a << d) | b) << (d + low)) | s
        j = i | (truncated(a, b, d) << low)
        initial[i] = expected[j] = exp(1j * (a - 0.31 * b + 0.2 * s)) / sqrt(
            2 ** (2 * d + 1)
        )
    assert truncated(7, 7, 3) == 4 and (7 * 7) >> 3 == 6
    assert truncated(4, 6, 3) == 3 and truncated(6, 4, 3) == 3
    for adjoint in (False, True):
        gates = fixed_point_product_ir(*bound, adjoint=adjoint)
        before, after = (expected, initial) if adjoint else (initial, expected)
        events = sum(g.name == "measure" for g in gates)
        if d <= 2:
            dense = torch.zeros(1 << width, dtype=torch.complex128)
            desired = torch.zeros_like(dense)
            for i, value in before.items():
                dense[i] = value
            for i, value in after.items():
                desired[i] = value
            for history in product((0, 1), repeat=events):
                torch.testing.assert_close(
                    forced_branch(dense, gates, history),
                    desired / sqrt(2**events),
                    atol=2e-12,
                    rtol=0,
                )
        else:
            for seed in (3, 19, 72):
                actual = execute_sparse_register(
                    CircuitIR(width, gates), before, seed=seed
                )
                assert all(
                    abs(actual.get(i, 0) - after.get(i, 0)) < 2e-12
                    for i in actual.keys() | after.keys()
                )


def test_fixed_point_product_append_round_trip():
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    bound = layout(2)
    initial = torch.zeros(64, dtype=torch.complex128)
    initial[0], initial[60] = 0.8, 0.6j
    expected = initial.clone()
    expected[60], expected[61] = 0, 0.6j
    circuit = DynamicCircuit(6, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(6, fixed_point_product_ir(*bound)))
    forward = run_dynamic(circuit, shots=8, seed=17, strategy="trajectory")
    torch.testing.assert_close(
        forward.final_states, expected.expand(8, -1), atol=2e-12, rtol=0
    )
    circuit.append(FieldOperation(6, fixed_point_product_ir(*bound, adjoint=True)))
    reverse = run_dynamic(circuit, shots=8, seed=17, strategy="trajectory")
    torch.testing.assert_close(
        reverse.final_states, initial.expand(8, -1), atol=2e-12, rtol=0
    )


def test_fixed_point_product_rejects_invalid_bindings():
    with pytest.raises(ValueError, match="positive input"):
        fixed_point_product_workspace_size(0)
    with pytest.raises(ValueError, match="equal-width"):
        fixed_point_product_ir((0,), (1, 2), (3,), (), (), ())

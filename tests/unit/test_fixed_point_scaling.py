"""Pinned fixed-point format, literal child gates and term-truncation oracle."""

from cmath import exp
from collections import Counter
from itertools import product
from math import sqrt

import pytest
import torch

from flagquantum.compiler.registers.and_compute import and_compute_ir, and_uncompute_ir
from flagquantum.compiler.registers.fixed_point_scaling import (
    scale_integer_by_real_ir,
    scale_integer_workspace_size,
)
from flagquantum.core.ir import CircuitIR
from tests.unit.algorithms.primitives.multiplexers.test_thc_qrom import _execute_sparse
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def layout(r, n):
    size = scale_integer_workspace_size(r, n)
    offset = 2 * r + n
    return (
        tuple(range(r)),
        tuple(range(r, r + n)),
        tuple(range(r + n, offset)),
        tuple(range(offset, offset + size)),
        tuple(range(offset + size, offset + 2 * size)),
        tuple(range(2 * size)),
    )


def truncated(real, integer, width):
    return sum(
        real >> (j + 1) for j in range(width) if integer & (1 << (width - 1 - j))
    )


@pytest.mark.parametrize(
    "r,n", [(1, 1), (4, 1), (2, 2), (3, 2), (4, 2), (4, 4), (6, 4)]
)
def test_scaling_pinned_layout_children_and_every_branch(r, n):
    from qualtran import DecomposeNotImplementedError, QUInt
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd
    from qualtran.bloqs.arithmetic.multiplication import ScaleIntByReal
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    bound = layout(r, n)
    real, integer, result, loaded, carries, slots = bound
    source = ScaleIntByReal(r, n)
    output = source.signature.get_right("result")
    assert output.bitsize == r and output.dtype.num_frac == r - n
    assert source.signature.get_left("real_in").dtype.num_frac == r
    assert source.signature.get_left("int_in").dtype == QUInt(n)
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()
    with pytest.raises(NotImplementedError):
        source.tensor_contract()
    assert source.on_classical_vals(real_in=0, int_in=0) is NotImplemented

    def word(value, width):
        return tuple((value >> i) & 1 for i in reversed(range(width)))

    incoming, outgoing = [], []
    clean = (0,) * (len(loaded) + len(carries))
    for a, b in product(range(1 << r), range(1 << n)):
        value = truncated(a, b, n)
        error = a * b - (value << n)
        assert 0 <= error <= b.bit_count() * ((1 << n) - 1)
        prefix = word(a, r) + word(b, n)
        incoming.append(prefix + (0,) * r + clean)
        outgoing.append(prefix + word(value, r) + clean)

    k = r - n
    pairs = tuple(zip(real[:k], result[n:], strict=True))
    source_model = get_cost_value(source, QECGatesCost())
    modeled = r * (2 * n - 1) - n**2
    assert source_model.toffoli == modeled
    for adjoint in (False, True):
        gates = scale_integer_by_real_ir(*bound, adjoint=adjoint)
        steps = tuple(enumerate(reversed(integer[:-1]), start=k + 1))
        ref = []
        for w, control in reversed(steps) if adjoint else steps:
            add = CAdd(QUInt(w), QUInt(w + 1))
            ref.extend(
                reference_arithmetic_gates(
                    add.adjoint() if adjoint else add,
                    {"ctrl": (control,), "a": real[:w], "b": result[-w - 1 :]},
                    (*loaded[:w], *carries[:w]),
                    dict(
                        zip(
                            (*loaded[:w], *carries[:w]),
                            (*slots[:w], *slots[len(loaded) : len(loaded) + w]),
                            strict=True,
                        )
                    ),
                )
            )
        if adjoint:
            for bit, out in reversed(pairs):
                ref.extend(and_uncompute_ir(integer[-1], bit, out, 0))
        else:
            copy = [
                gate
                for bit, out in pairs
                for gate in and_compute_ir(integer[-1], bit, out)
            ]
            ref = copy + ref
        assert_source(gates, tuple(ref))
        counts = Counter(g.name for g in gates)
        assert counts["t"] + counts["tdg"] == 4 * (modeled - (k if adjoint else 0))
        events = modeled - (0 if adjoint else k)
        assert counts["measure"] == counts["reset"] == events
        a, b = (outgoing, incoming) if adjoint else (incoming, outgoing)
        assert certified_and_channel_map(gates, tuple(a)) == (tuple(b), events)


@pytest.mark.parametrize("r,n", [(2, 1), (2, 2), (4, 2)])
def test_scaling_complex_amplitudes_with_coherent_spectator(r, n):
    bound = layout(r, n)
    width = 2 * r + n + len(bound[3]) + len(bound[4]) + 1
    remaining = len(bound[3]) + len(bound[4]) + 1
    initial, expected = {}, {}
    for a, b, s in product(range(1 << r), range(1 << n), (0, 1)):
        i = (((a << n) | b) << (r + remaining)) | s
        j = i | (truncated(a, b, n) << remaining)
        initial[i] = expected[j] = exp(1j * (a - 0.31 * b + 0.2 * s)) / sqrt(
            2 ** (r + n + 1)
        )
    if (r, n) == (4, 2):
        assert truncated(15, 3, 2) == 10 and (15 * 3) >> 2 == 11
    for adjoint in (False, True):
        gates = scale_integer_by_real_ir(*bound, adjoint=adjoint)
        before, after = (expected, initial) if adjoint else (initial, expected)
        events = sum(g.name == "measure" for g in gates)
        if r == 2:
            dense = torch.zeros(1 << width, dtype=torch.complex128)
            target = torch.zeros_like(dense)
            for i, value in before.items():
                dense[i] = value
            for i, value in after.items():
                target[i] = value
            for history in product((0, 1), repeat=events):
                torch.testing.assert_close(
                    forced_branch(dense, gates, history),
                    target / sqrt(2**events),
                    atol=2e-12,
                    rtol=0,
                )
        else:
            for seed in (3, 19, 72):
                actual = _execute_sparse(CircuitIR(width, gates), before, seed=seed)
                assert all(
                    abs(actual.get(i, 0) - after.get(i, 0)) < 2e-12
                    for i in actual.keys() | after.keys()
                )


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"integer": ()}, "integer width"),
        ({"result": ()}, "result and work"),
        ({"classical_bits": ()}, "pool slots"),
        ({"copy_bit": -1}, "pool slots"),
        ({"adjoint": 1}, "must be boolean"),
        ({"result": (0, 5)}, "distinct nonnegative"),
    ],
)
def test_scaling_rejects_invalid_binding(overrides, error):
    names = ("real", "integer", "result", "loaded", "carries", "classical_bits")
    bound = dict(zip(names, layout(2, 2), strict=True))
    with pytest.raises(ValueError, match=error):
        scale_integer_by_real_ir(**(bound | overrides))


def test_scaling_append_round_trip() -> None:
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    bound = layout(2, 1)
    initial = torch.zeros(32, dtype=torch.complex128)
    initial[0], initial[28] = 0.8, 0.6j
    target = initial.clone()
    target[28], target[29] = 0, 0.6j
    circuit = DynamicCircuit(5, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(5, scale_integer_by_real_ir(*bound)))
    forward = run_dynamic(circuit, shots=8, seed=17, strategy="trajectory")
    torch.testing.assert_close(
        forward.final_states, target.expand(8, -1), atol=2e-12, rtol=0
    )
    circuit.append(FieldOperation(5, scale_integer_by_real_ir(*bound, adjoint=True)))
    reverse = run_dynamic(circuit, shots=8, seed=17, strategy="trajectory")
    torch.testing.assert_close(
        reverse.final_states, initial.expand(8, -1), atol=2e-12, rtol=0
    )

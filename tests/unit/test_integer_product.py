"""Unsigned product layout, weighted terms, complex amplitudes and cleanup."""

from cmath import exp
from collections import Counter
from itertools import product
from math import sqrt

import pytest
import torch

from flagquantum.compiler.registers.and_compute import and_compute_ir
from flagquantum.compiler.registers.integer_product import (
    integer_product_ir,
    integer_product_requirements,
)
from flagquantum.core.ir import CircuitIR
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register

pytestmark = pytest.mark.unit


def layout(n, m):
    result_width, work_width = integer_product_requirements(n, m)
    result_stop = n + m + result_width
    return (
        tuple(range(n)),
        tuple(range(n, n + m)),
        tuple(range(n + m, result_stop)),
        tuple(range(result_stop, result_stop + work_width)),
    )


@pytest.mark.parametrize(
    "n,m", [(1, 1), (1, 4), (4, 1), (2, 3), (3, 2), (3, 3), (4, 6)]
)
def test_unsigned_product_source_contract_and_all_history_certificate(n, m):
    import qualtran
    from qualtran import Side
    from qualtran.bloqs.arithmetic.multiplication import PlusEqualProduct, Product
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    left, right, result, work = layout(n, m)
    source = Product(n, m)
    assert source.signature.get_left("a").bitsize == n
    assert source.signature.get_left("b").bitsize == m
    output = source.signature.get_right("result")
    assert output.bitsize == len(result) and output.side == Side.RIGHT
    with pytest.raises(qualtran.DecomposeNotImplementedError):
        source.decompose_bloq()
    with pytest.raises(NotImplementedError):
        source.tensor_contract()
    assert source.on_classical_vals(a=1, b=1) is NotImplemented
    gates = integer_product_ir(left, right, result, work, measurement_bit=7)
    prefix = tuple(
        gate
        for (a, b), target in zip(
            product(reversed(left), reversed(right)),
            work[: n * m],
            strict=True,
        )
        for gate in and_compute_ir(a, b, target)
    )
    assert gates[: len(prefix)] == prefix
    model = get_cost_value(source, QECGatesCost())
    counts = Counter(g.name for g in gates)
    assert len(work) == model.toffoli == 2 * n * m - max(n, m)
    assert counts["t"] + counts["tdg"] == 4 * len(work)
    assert counts["measure"] == counts["reset"] == len(work)
    assert all(g.metadata["classical_bit"] == 7 for g in gates if g.name == "measure")

    def word(value, bits):
        return tuple((value >> i) & 1 for i in reversed(range(bits)))

    original_arithmetic = PlusEqualProduct(n, m, len(result))
    before, after = [], []
    for a, b in product(range(1 << n), range(1 << m)):
        value = original_arithmetic.on_classical_vals(a=a, b=b, result=0)["result"]
        assert value == a * b
        seed = (1 << len(result)) - 1 if (a + b) % 2 else 0
        inp = word(a, n) + word(b, m)
        before.append(inp + word(seed, len(result)) + (0,) * len(work))
        after.append(inp + word(seed ^ value, len(result)) + (0,) * len(work))
    assert certified_and_channel_map(gates, tuple(before)) == (tuple(after), len(work))
    assert certified_and_channel_map(gates, tuple(after)) == (tuple(before), len(work))


@pytest.mark.parametrize("n,m", [(1, 1), (1, 2), (2, 2), (2, 3)])
def test_unsigned_product_complex_forward_and_erase(n, m):
    from qualtran.bloqs.arithmetic.multiplication import PlusEqualProduct

    left, right, result, work = layout(n, m)
    gates = integer_product_ir(left, right, result, work)
    original = PlusEqualProduct(n, m, len(result))
    width = n + m + len(result) + len(work) + 1
    before, after = {}, {}
    for a, b, spectator in product(range(1 << n), range(1 << m), (0, 1)):
        _, _, value = original.apply(a, b, 0)
        assert tuple(original.adjoint().apply(a, b, value)) == (a, b, 0)
        i = (((a << m) | b) << (width - n - m)) | spectator
        j = i | (value << (len(work) + 1))
        before[i] = after[j] = exp(1j * (a - 0.2 * b + 0.31 * spectator)) / sqrt(
            2 ** (n + m + 1)
        )

    for initial, expected in ((before, after), (after, before)):
        if len(work) <= 2:
            dense = torch.zeros(1 << width, dtype=torch.complex128)
            dense_expected = torch.zeros_like(dense)
            for i, value in initial.items():
                dense[i] = value
            for i, value in expected.items():
                dense_expected[i] = value
            for history in product((0, 1), repeat=len(work)):
                torch.testing.assert_close(
                    forced_branch(dense, gates, history),
                    dense_expected / sqrt(2 ** len(work)),
                    atol=2e-12,
                    rtol=0,
                )
        else:
            for seed in (3, 19, 72):
                actual = execute_sparse_register(
                    CircuitIR(width, gates), initial, seed=seed
                )
                assert all(
                    abs(actual.get(i, 0) - expected.get(i, 0)) < 2e-12
                    for i in actual.keys() | expected.keys()
                )


def test_unsigned_product_append_trajectory() -> None:
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    gates = integer_product_ir(*layout(1, 2))
    initial = torch.zeros(512, dtype=torch.complex128)
    initial[0], initial[7 << 6] = 0.8, 0.6j
    expected = initial.clone()
    expected[7 << 6] = 0
    expected[(7 << 6) | (3 << 2)] = 0.6j
    circuit = DynamicCircuit(9, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(9, gates))
    actual = run_dynamic(circuit, shots=8, seed=13, strategy="trajectory")
    torch.testing.assert_close(
        actual.final_states, expected.expand(8, -1), atol=2e-12, rtol=0
    )


@pytest.mark.parametrize(
    "left,right,result,work,bit,error",
    [
        ((), (0,), (), (), 0, "positive input"),
        ((0,), (1,), (), (), 0, "result and"),
        ((0,), (0,), (2, 3), (4,), 0, "distinct nonnegative"),
        ((0,), (1,), (2, 3), (4,), -1, "measurement slot"),
    ],
)
def test_unsigned_product_rejects_invalid_binding(
    left, right, result, work, bit, error
):
    with pytest.raises(ValueError, match=error):
        integer_product_ir(left, right, result, work, measurement_bit=bit)

"""Exact zero-based contiguous indexing and measured workspace cleanup."""

from cmath import exp
from collections import Counter
from itertools import product
from math import sqrt

import pytest
import torch

from flagquantum.compiler.registers.contiguous_index import (
    contiguous_index_ir,
    contiguous_index_requirements,
)
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register

pytestmark = pytest.mark.unit


def layout(n, out):
    work, slots = contiguous_index_requirements(n)
    return (
        tuple(range(n)),
        tuple(range(n, 2 * n)),
        tuple(range(2 * n, 2 * n + out)),
        tuple(range(2 * n + out, 2 * n + out + work)),
        tuple(range(slots)),
    )


def bits(word, width):
    return tuple(word >> j & 1 for j in reversed(range(width)))


def triangular(mu, nu, out):
    return (mu + nu * (nu + 1) // 2) % (1 << out)


@pytest.mark.parametrize("n,out", [(1, 1), (1, 2), (2, 5), (3, 3), (4, 8), (5, 10)])
def test_contiguous_index_all_inputs_occupied_results_and_histories(n, out):
    bound = layout(n, out)
    gates = contiguous_index_ir(*bound)
    counts = Counter(g.name for g in gates)
    events = 2 * (n * (n - 1) + 2 * (2 * n - 1))
    assert counts["measure"] == counts["reset"] == events
    assert counts["t"] + counts["tdg"] == 4 * events
    assert counts["ccx"] == 0
    assert len(bound[3]) == 2 * n + max(n * (n - 1), 2 * n - 1)
    before, after = [], []
    clean = (0,) * len(bound[3])
    for mu, nu in product(range(1 << n), repeat=2):
        prefix = bits(mu, n) + bits(nu, n)
        for old in (0, (1 << out) - 1, (2 * mu + nu) % (1 << out)):
            before.append(prefix + bits(old, out) + clean)
            after.append(prefix + bits(old ^ triangular(mu, nu, out), out) + clean)
    assert certified_and_channel_map(gates, tuple(before)) == (tuple(after), events)
    assert certified_and_channel_map(gates, tuple(after)) == (tuple(before), events)


@pytest.mark.parametrize("n,out", [(1, 2), (2, 3), (3, 6)])
def test_contiguous_index_append_complex_and_work_cleanup(n, out):
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit

    bound = layout(n, out)
    work = len(bound[3])
    gates = contiguous_index_ir(*bound)
    qubits = 2 * n + out + work + 1
    ir = DynamicCircuit(qubits).append(FieldOperation(qubits, gates)).to_ir()
    before, after = {}, {}
    for mu, nu, s in product(range(1 << n), range(1 << n), (0, 1)):
        prefix = ((mu << n) | nu) << (out + work + 1)
        occupied = (mu + nu) % (1 << out)
        i = prefix | (occupied << (work + 1)) | s
        j = i ^ (triangular(mu, nu, out) << (work + 1))
        before[i] = after[j] = exp(0.13j * (mu - nu) * (1 + s)) / sqrt(2 ** (2 * n + 1))
    for initial, expected in ((before, after), (after, before)):
        actual = execute_sparse_register(ir, initial, seed=71)
        assert all(
            abs(actual.get(i, 0) - expected.get(i, 0)) < 2e-12
            for i in actual.keys() | expected.keys()
        )
        if n == 1:
            state = torch.zeros(1 << qubits, dtype=torch.complex128)
            desired = torch.zeros_like(state)
            for i, amp in initial.items():
                state[i] = amp
            for i, amp in expected.items():
                desired[i] = amp
            for history in product((0, 1), repeat=4):
                torch.testing.assert_close(
                    forced_branch(state, gates, history),
                    desired / 4,
                    atol=2e-12,
                    rtol=0,
                )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n", [1, 3, 5])
def test_locked_classical_contract_model_and_missing_decomposition(n):
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.arithmetic.conversions.contiguous_index import ToContiguousIndex
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = ToContiguousIndex(n, 2 * n)
    for mu, nu in product(range(1 << n), repeat=2):
        assert source.on_classical_vals(mu=mu, nu=nu) == {
            "mu": mu,
            "nu": nu,
            "s": triangular(mu, nu, 2 * n),
        }
    assert get_cost_value(source, QECGatesCost()).toffoli == n * n + n - 1
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()
    with pytest.raises(TypeError, match="unexpected keyword argument 's'"):
        source.call_classically(mu=0, nu=0, s=0)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"nu": ()}, "positive input width"),
        ({"mu": ()}, "matching inputs"),
        ({"result": ()}, "nonempty result"),
        ({"workspace": ()}, "work qubits"),
        ({"classical_bits": (0,) * 3}, "distinct nonnegative slots"),
        ({"result": (0, 6, 7, 8)}, "distinct nonnegative wires"),
    ],
)
def test_contiguous_index_rejects_invalid_layout(change, match):
    names = ("mu", "nu", "result", "workspace", "classical_bits")
    bound = dict(zip(names, layout(2, 4), strict=True))
    with pytest.raises(ValueError, match=match):
        contiguous_index_ir(**(bound | change))

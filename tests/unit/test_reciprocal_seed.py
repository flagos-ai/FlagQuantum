"""Figure 4 reciprocal seed, negative controls and source resource boundary."""

from cmath import exp
from collections import Counter
from itertools import product
from math import sqrt

import pytest
import torch

from flagquantum.compiler.registers.reciprocal_seed import reciprocal_seed_ir
from flagquantum.core.ir import CircuitIR, Instruction
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def layout(m, f, b):
    n = m + f
    return (
        tuple(range(n)),
        tuple(range(n, n + b)),
        n + b,
        tuple(range(n + b + 1, n + b + m)),
        tuple(range(m - 1)),
    )


def seed_word(value, frac, precision):
    exponent = (value >> frac).bit_length()
    return 1 << (precision - exponent) if exponent else 0


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "m,f,b", [(1, 0, 1), (2, 0, 2), (3, 2, 3), (4, 3, 6), (8, 2, 8)]
)
def test_seed_figure_resources_and_all_histories(m, f, b):
    pytest.importorskip("qualtran")
    from qualtran import DecomposeNotImplementedError
    from qualtran.bloqs.arithmetic.multiplication import InvertRealNumber
    from qualtran.bloqs.basic_gates import CNOT, Toffoli, XGate
    from qualtran.bloqs.mcmt.multi_control_pauli import MultiControlX
    from qualtran.resource_counting import SympySymbolAllocator
    from qualtran.symbolics import HasLength

    source = InvertRealNumber(m + f, f)
    assert source.signature.get_left("a").dtype.num_frac == f
    assert source.signature.get_right("result").dtype.num_frac == f
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()
    if m + f < 3:
        with pytest.raises(ValueError, match="at least 3"):
            source.build_call_graph(SympySymbolAllocator())
    else:
        graph = source.build_call_graph(SympySymbolAllocator())
        assert graph[Toffoli()] == m - 1
        assert graph[CNOT()] == m + 1
        assert graph[XGate()] == graph[MultiControlX(cvs=HasLength(m))] == 1

    bound = layout(m, f, b)
    data, seed, flag, work, bits = bound
    gates = reciprocal_seed_ir(*bound, source_fraction_bits=f)
    reverse = reciprocal_seed_ir(*bound, source_fraction_bits=f, adjoint=True)
    # Figure 4 uses the flag as a negative control and restores it between targets.
    select = [
        Instruction("cx", (data[0], flag)),
        Instruction("cx", (data[0], seed[m - 1])),
    ]
    for i in range(1, m):
        q = seed[m - 1 - i]
        select.extend(
            (
                Instruction("x", (flag,)),
                Instruction("ccx", (data[i], flag, q)),
                Instruction("x", (flag,)),
                Instruction("cx", (q, flag)),
            )
        )
    assert gates[: len(select)] == tuple(select)
    assert gates[-1] == reverse[0] == Instruction("x", (flag,))
    assert reverse[-len(select) :] == tuple(reversed(select))
    cleanup = gates[len(select) : -1]
    assert reverse[1 : -len(select)] == cleanup
    assert_source(
        cleanup,
        reference_arithmetic_gates(
            MultiControlX((0,) * m),
            {"controls": seed[:m], "target": (flag,)},
            work,
            dict(zip(work, bits, strict=True)),
        ),
    )
    assert all(not set(g.wires) & set((*data[m:], *seed[m:])) for g in gates)
    counts = Counter(g.name for g in gates)
    assert counts["ccx"] == m - 1
    assert counts["t"] + counts["tdg"] == 4 * (m - 1)
    assert counts["measure"] == counts["reset"] == len(work) == m - 1

    def word(value, width):
        return tuple(value >> i & 1 for i in reversed(range(width)))

    before, after = [], []
    for value in range(1 << (m + f)):
        prefix = word(value, m + f)
        tail = (0,) * (m - 1 + 1)
        before.append(prefix + (0,) * b + tail)
        after.append(prefix + word(seed_word(value, f, b), b) + tail)
    assert certified_and_channel_map(gates, tuple(before)) == (tuple(after), m - 1)
    assert certified_and_channel_map(reverse, tuple(after)) == (tuple(before), m - 1)


@pytest.mark.parametrize("m,f,b", [(1, 1, 2), (3, 1, 3), (5, 2, 6)])
def test_seed_raw_branches_and_coherent_erasure(m, f, b):
    bound = layout(m, f, b)
    width = m + f + b + m + 1
    before, after = {}, {}
    for value, spectator in product(range(1 << (m + f)), (0, 1)):
        i = (value << (b + m + 1)) | spectator
        j = i | (seed_word(value, f, b) << (m + 1))
        before[i] = after[j] = exp(1j * (0.7 * value - 0.21 * spectator)) / sqrt(
            2 ** (m + f + 1)
        )
    for adjoint in (False, True):
        gates = reciprocal_seed_ir(*bound, source_fraction_bits=f, adjoint=adjoint)
        initial, expected = (after, before) if adjoint else (before, after)
        if m <= 3:
            dense = torch.zeros(1 << width, dtype=torch.complex128)
            desired = torch.zeros_like(dense)
            for i, value in initial.items():
                dense[i] = value
            for i, value in expected.items():
                desired[i] = value
            for history in product((0, 1), repeat=m - 1):
                torch.testing.assert_close(
                    forced_branch(dense, gates, history),
                    desired / sqrt(2 ** (m - 1)),
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


def test_seed_append_round_trip():
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    bound = layout(2, 0, 2)
    state = torch.zeros(64, dtype=torch.complex128)
    state[0], state[48] = 0.8, 0.6j
    expected = state.clone()
    expected[48], expected[52] = 0, 0.6j
    circuit = DynamicCircuit(6, inputs=state, dtype=torch.complex128)
    circuit.append(
        FieldOperation(6, reciprocal_seed_ir(*bound, source_fraction_bits=0))
    )
    forward = run_dynamic(circuit, shots=8, seed=17, strategy="trajectory")
    torch.testing.assert_close(
        forward.final_states, expected.expand(8, -1), atol=2e-12, rtol=0
    )
    circuit.append(
        FieldOperation(
            6, reciprocal_seed_ir(*bound, source_fraction_bits=0, adjoint=True)
        )
    )
    reverse = run_dynamic(circuit, shots=8, seed=17, strategy="trajectory")
    torch.testing.assert_close(
        reverse.final_states, state.expand(8, -1), atol=2e-12, rtol=0
    )


@pytest.mark.parametrize(
    "args,error",
    [
        ({"source_fraction_bits": 2}, "nonempty integer"),
        ({"adjoint": 1}, "must be boolean"),
        ({"seed": (2,)}, "fractional wire"),
        ({"workspace": ()}, "clean work"),
        ({"flag": 0}, "distinct nonnegative"),
    ],
)
def test_seed_rejects_invalid_bindings(args, error):
    names = ("source", "seed", "flag", "workspace", "classical_bits")
    bound = dict(zip(names, layout(2, 0, 2), strict=True)) | {"source_fraction_bits": 0}
    with pytest.raises(ValueError, match=error):
        reciprocal_seed_ir(**(bound | args))

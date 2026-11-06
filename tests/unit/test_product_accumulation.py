"""Pinned multiply-accumulate tensors and component streams with cleanup."""

from collections import Counter
from itertools import product
from math import sqrt

import pytest
import torch

from flagquantum.compiler.registers.integer_product import (
    integer_product_ir,
    integer_product_requirements,
)
from flagquantum.compiler.registers.product_accumulation import product_accumulation_ir
from flagquantum.core.ir import CircuitIR
from tests.unit.algorithms.primitives.multiplexers.test_thc_qrom import _execute_sparse
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def layout(n, m, width):
    _, work_width = integer_product_requirements(n, m)
    base = n + m + width
    return (
        tuple(range(n)),
        tuple(range(n, n + m)),
        tuple(range(n + m, base)),
        tuple(range(base, base + work_width)),
        tuple(range(base + work_width, base + work_width + width - 1)),
        tuple(range(width - 1)),
    )


@pytest.mark.parametrize(
    "n,m,width", [(1, 1, 2), (1, 2, 3), (2, 1, 4), (2, 2, 4), (2, 3, 5), (3, 2, 6)]
)
@pytest.mark.parametrize("adjoint", [False, True])
def test_accumulation_component_stream_and_full_classical_domain(n, m, width, adjoint):
    from qualtran import DecomposeNotImplementedError, QUInt
    from qualtran.bloqs.arithmetic.addition import Add
    from qualtran.bloqs.arithmetic.multiplication import PlusEqualProduct
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    left, right, result, work, carries, bits = layout(n, m, width)
    gates = product_accumulation_ir(
        left, right, result, work, carries, bits, adjoint=adjoint
    )
    total_wires = n + m + width + len(work) + len(carries)
    fake_result = tuple(range(total_wires, total_wires + 2 * max(n, m)))
    product_stream = integer_product_ir(left, right, fake_result, work)
    copy_indices = [
        i for i, gate in enumerate(product_stream) if gate.wires[-1] in fake_result
    ]
    live = tuple(product_stream[i].wires[0] for i in reversed(copy_indices))
    assert copy_indices == list(range(copy_indices[0], copy_indices[-1] + 1))
    add = Add(QUInt(len(live)), QUInt(width))
    addition = reference_arithmetic_gates(
        add.adjoint() if adjoint else add,
        {"a": live, "b": result},
        carries,
        dict(zip(carries, bits, strict=True)),
    )
    start, stop = copy_indices[0], copy_indices[-1] + 1
    assert gates[:start] == product_stream[:start]
    assert gates[-len(product_stream[stop:]) :] == product_stream[stop:]
    assert_source(gates[start : len(gates) - len(product_stream[stop:])], addition)

    source = PlusEqualProduct(n, m, width, is_adjoint=adjoint)
    with pytest.raises(DecomposeNotImplementedError):
        source.decompose_bloq()
    assert get_cost_value(source, QECGatesCost()).t == 8 * max(n, m) ** 2
    counts = Counter(g.name for g in gates)
    events = len(work) + len(carries)
    assert counts["t"] + counts["tdg"] == 4 * events
    assert counts["measure"] == counts["reset"] == events

    def word(value, bits):
        return tuple((value >> i) & 1 for i in reversed(range(bits)))

    incoming, outgoing = [], []
    for a, b, c in product(range(1 << n), range(1 << m), range(1 << width)):
        raw = source.on_classical_vals(a=a, b=b, result=c)["result"]
        assert tuple(source.adjoint().apply(a, b, raw)) == (a, b, c)
        prefix = word(a, n) + word(b, m)
        clean = (0,) * (len(work) + len(carries))
        incoming.append(prefix + word(c, width) + clean)
        outgoing.append(prefix + word(raw, width) + clean)
    assert certified_and_channel_map(gates, tuple(incoming)) == (
        tuple(outgoing),
        events,
    )


@pytest.mark.parametrize("n,m,width", [(1, 1, 2), (1, 2, 3), (2, 2, 4)])
@pytest.mark.parametrize("adjoint", [False, True])
def test_accumulation_original_tensor_and_complex_branches(n, m, width, adjoint):
    from qualtran.bloqs.arithmetic.multiplication import PlusEqualProduct

    bound = layout(n, m, width)
    gates = product_accumulation_ir(*bound, adjoint=adjoint)
    model = PlusEqualProduct(n, m, width, is_adjoint=adjoint)
    logical_bits = n + m + width
    work_bits = len(bound[3]) + len(bound[4])
    total_bits = logical_bits + work_bits + 1
    grid = torch.arange(1, (1 << logical_bits) * 2 + 1, dtype=torch.float64)
    state = torch.complex(grid.remainder(7) - 3, grid.remainder(5) - 2).reshape(-1, 2)
    state /= state.norm()
    expected = torch.as_tensor(model.tensor_contract().copy()) @ state

    def sparse(vector):
        return {
            ((i << (work_bits + 1)) | spectator): complex(vector[i, spectator])
            for i, spectator in product(range(1 << logical_bits), (0, 1))
            if vector[i, spectator] != 0
        }

    initial, target = sparse(state), sparse(expected)
    if work_bits == 2:
        dense = torch.zeros(1 << total_bits, dtype=torch.complex128)
        desired = torch.zeros_like(dense)
        for i, value in initial.items():
            dense[i] = value
        for i, value in target.items():
            desired[i] = value
        for history in product((0, 1), repeat=work_bits):
            torch.testing.assert_close(
                forced_branch(dense, gates, history),
                desired / sqrt(2**work_bits),
                atol=2e-12,
                rtol=0,
            )
    else:
        for seed in (4, 19, 73):
            actual = _execute_sparse(CircuitIR(total_bits, gates), initial, seed=seed)
            assert all(
                abs(actual.get(i, 0) - target.get(i, 0)) < 2e-12
                for i in actual.keys() | target.keys()
            )


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"result": (2,)}, "wide enough"),
        ({"product_work": ()}, "clean product"),
        ({"classical_bits": ()}, "one carry and slot"),
        ({"classical_bits": (-1,)}, "nonnegative classical"),
        ({"adjoint": 1}, "must be boolean"),
        ({"carries": (0,)}, "distinct nonnegative"),
    ],
)
def test_accumulation_rejects_invalid_binding(overrides, error):
    keys = ("left", "right", "result", "product_work", "carries", "classical_bits")
    bindings = dict(zip(keys, layout(1, 1, 2), strict=True))
    with pytest.raises(ValueError, match=error):
        product_accumulation_ir(**(bindings | overrides))


def test_product_accumulation_append_round_trip() -> None:
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    binding = layout(1, 1, 2)
    initial = torch.zeros(64, dtype=torch.complex128)
    initial[0], initial[60] = 0.8, 0.6j
    expected = initial.clone()
    expected[60], expected[48] = 0, 0.6j
    circuit = DynamicCircuit(6, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(6, product_accumulation_ir(*binding)))
    forward = run_dynamic(circuit, shots=8, seed=17, strategy="trajectory")
    torch.testing.assert_close(
        forward.final_states, expected.expand(8, -1), atol=2e-12, rtol=0
    )
    circuit.append(FieldOperation(6, product_accumulation_ir(*binding, adjoint=True)))
    reverse = run_dynamic(circuit, shots=8, seed=17, strategy="trajectory")
    torch.testing.assert_close(
        reverse.final_states, initial.expand(8, -1), atol=2e-12, rtol=0
    )

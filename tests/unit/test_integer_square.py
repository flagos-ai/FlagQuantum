"""Pinned Square tensors and reversible column-compression acceptance."""

from collections import Counter
from itertools import product
from math import sqrt

import pytest
import torch

from flagquantum.compiler.registers.arithmetic.integer_square import (
    integer_square_ir,
    integer_square_workspace_size,
)
from flagquantum.simulation.classical_square import square_classical_values
from tests.unit.measurement_checks import certified_and_channel_map, forced_branch

pytestmark = pytest.mark.unit


def square_layout(n: int) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(range(n)),
        tuple(range(n, 3 * n)),
        tuple(range(3 * n, 3 * n + integer_square_workspace_size(n))),
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n", range(1, 9))
def test_square_all_basis_inputs_all_history_certificate(n: int) -> None:
    pytest.importorskip("qualtran")
    import qualtran
    from qualtran.bloqs.arithmetic.multiplication import Square
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source, result, work = square_layout(n)
    gates = integer_square_ir(source, result, work, measurement_bit=7)
    bloq = Square(n)
    cost = n * (n - 1)
    counts = Counter(g.name for g in gates)
    assert len(work) == cost
    assert counts["t"] + counts["tdg"] == 4 * cost
    assert counts["measure"] == counts["reset"] == cost
    assert get_cost_value(bloq, QECGatesCost()).toffoli == cost
    assert get_cost_value(bloq.adjoint(), QECGatesCost()).toffoli == cost
    for reference in (bloq, bloq.adjoint()):
        with pytest.raises(qualtran.DecomposeNotImplementedError):
            reference.decompose_bloq()

    def word(value: int, bits: int) -> tuple[int, ...]:
        return tuple((value >> i) & 1 for i in reversed(range(bits)))

    blank = []
    squared = []
    for a in range(1 << n):
        expected = bloq.on_classical_vals(a=a)
        assert square_classical_values({"a": a}) == expected
        assert square_classical_values(expected, uncompute=True) == {"a": a}
        assert bloq.adjoint().on_classical_vals(**expected) == {"a": a}
        blank.append(word(a, n) + (0,) * (2 * n + len(work)))
        squared.append(word(a, n) + word(expected["result"], 2 * n) + (0,) * len(work))
    assert certified_and_channel_map(gates, tuple(blank)) == (tuple(squared), cost)
    assert certified_and_channel_map(gates, tuple(squared)) == (tuple(blank), cost)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n", [1, 2, 3])
def test_square_pinned_complex_tensors_and_cleanup(n: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.arithmetic.multiplication import Square

    source, result, work = square_layout(n)
    gates = integer_square_ir(source, result, work)
    # Two coherent spectator slices of the same pinned linear map.
    basis = torch.arange(1, 1 + (1 << n) * 2, dtype=torch.float64).reshape(-1, 2)
    logical = torch.complex(basis, basis.remainder(3) - 1)
    logical /= logical.norm()
    bloq = Square(n)
    tensor = torch.as_tensor(bloq.tensor_contract().copy())
    square = tensor @ logical
    numeric = torch.zeros_like(square)
    for a in range(1 << n):
        result_value = square_classical_values({"a": a})["result"]
        numeric[(a << (2 * n)) | int(result_value)] = logical[a]
    torch.testing.assert_close(numeric, square, rtol=0, atol=2e-12)
    inverse = torch.as_tensor(bloq.adjoint().tensor_contract().copy()) @ square
    torch.testing.assert_close(inverse, logical, atol=2e-12, rtol=0)
    embedded = torch.zeros_like(square)
    embedded[:: 1 << (2 * n)] = logical
    expanded = tuple(
        torch.zeros((1 << (3 * n), 1 << len(work), 2), dtype=torch.complex128)
        for _ in range(2)
    )
    expanded[0][:, 0] = embedded
    expanded[1][:, 0] = square
    events = len(work)
    histories = (
        tuple(product((0, 1), repeat=events))
        if n < 3
        else ((0,) * events, (1,) * events, tuple(i % 2 for i in range(events)))
    )
    for history in histories:
        for i in (0, 1):
            torch.testing.assert_close(
                forced_branch(expanded[i].reshape(-1), gates, history),
                expanded[1 - i].reshape(-1) / sqrt(2**events),
                atol=2e-12,
                rtol=0,
            )


@pytest.mark.parametrize(
    "source,result,work,slot,error",
    [
        ((), (), (), 0, "positive input"),
        ((0,), (1,), (), 0, "result and"),
        ((0,), (0, 1), (), 0, "distinct nonnegative"),
        ((0,), (1, 2), (), -1, "measurement slot"),
    ],
)
def test_square_rejects_invalid_bindings(source, result, work, slot, error) -> None:
    with pytest.raises(ValueError, match=error):
        integer_square_ir(source, result, work, measurement_bit=slot)


def test_append_square_then_erase_with_reused_measurement_slot() -> None:
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    source, result, work = square_layout(2)
    gates = integer_square_ir(source, result, work)
    initial = torch.zeros(256, dtype=torch.complex128)
    initial[0], initial[3 << 6] = 0.8, 0.6j
    expected = initial.clone()
    expected[3 << 6] = 0
    expected[(3 << 6) | (9 << 2)] = 0.6j
    circuit = DynamicCircuit(8, inputs=initial, dtype=torch.complex128)
    operation = FieldOperation(8, gates)
    circuit.append(operation)
    computed = run_dynamic(circuit, shots=8, seed=42, strategy="trajectory")
    torch.testing.assert_close(
        computed.final_states, expected.expand(8, -1), atol=2e-12, rtol=0
    )
    circuit.append(operation)
    erased = run_dynamic(circuit, shots=8, seed=42, strategy="trajectory")
    torch.testing.assert_close(
        erased.final_states, initial.expand(8, -1), atol=2e-12, rtol=0
    )

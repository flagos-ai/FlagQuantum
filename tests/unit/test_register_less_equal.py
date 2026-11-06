"""Unmodified comparison tree across equal and both unequal-width directions."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.less_equal import (
    less_equal_ir,
    less_equal_requirements,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit
CASES = [
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (8, 8),
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (1, 5),
    (5, 1),
    (2, 5),
    (5, 2),
    (3, 8),
    (8, 3),
]


def layout(
    m: int, n: int
) -> tuple[tuple[int, ...], tuple[int, ...], int, tuple[int, ...], tuple[int, ...]]:
    work, bits = less_equal_requirements(m, n)
    target = m + n
    return (
        tuple(range(m)),
        tuple(range(m, m + n)),
        target,
        tuple(range(target + 1, target + work + 1)),
        tuple(range(40, 40 + bits)),
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("m,n", CASES)
def test_less_equal_source_gates_work_measurements_and_model(m: int, n: int) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.arithmetic.comparison import LessThanEqual
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    x, y, t, w, b = layout(m, n)
    actual = less_equal_ir(x, y, t, w, b)
    measured = {
        g.wires[0]: g.metadata["classical_bit"] for g in actual if g.name == "measure"
    }
    assert len(measured) == len(b) and set(measured.values()) == set(b)
    source = LessThanEqual(m, n)
    expected = reference_arithmetic_gates(
        source, {"x": x, "y": y, "target": (t,)}, w, measured, cirq_style=True
    )
    assert_source(actual, expected)
    model = get_cost_value(source, QECGatesCost())
    counts = Counter(g.name for g in actual)
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * len(b)
    assert counts["measure"] == model.measurement == len(b)
    assert counts["reset"] == len(b)
    assert {wire for gate in actual for wire in gate.wires} & set(w) == set(w)


@pytest.mark.integration
@pytest.mark.parametrize(
    "m,n", [(1, 1), (2, 2), (3, 3), (1, 2), (2, 1), (1, 3), (3, 1), (1, 4), (4, 1)]
)
def test_less_equal_all_complex_input_target_and_work_cleanup(m: int, n: int) -> None:
    x, y, t, w, b = layout(m, n)
    gates = less_equal_ir(x, y, t, w, b)
    data = torch.arange(1 << (m + n + 1), dtype=torch.int64)
    source = data >> 1
    target = data & 1
    left = source >> n
    right = source & ((1 << n) - 1)
    mapped = (source << 1) | (target ^ (left <= right))
    coeff = data.to(torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    n_qubits = m + n + 1 + len(w)
    state = torch.zeros(1 << n_qubits, dtype=torch.complex128)
    state[data << len(w)] = amps
    expected = torch.zeros_like(state)
    expected[mapped << len(w)] = amps
    circuit = DynamicCircuit(n_qubits, inputs=state, dtype=torch.complex128).append(
        FieldOperation(n_qubits, gates)
    )
    result = run_dynamic(circuit, shots=256, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(256, -1),
        expected.expand(256, -1),
        rtol=0,
        atol=1e-12,
    )
    assert len(
        set(map(tuple, result.classical_bits[:, 40 : 40 + len(b)].tolist()))
    ) == 1 << len(b)
    restored = run_dynamic(
        circuit.append(FieldOperation(n_qubits, gates)),
        shots=8,
        seed=41,
        strategy="trajectory",
    )
    torch.testing.assert_close(
        restored.final_states.reshape(8, -1), state.expand(8, -1), rtol=0, atol=1e-12
    )


def test_less_equal_validation() -> None:
    for a, b in [(0, 1), (1, -1), (True, 2)]:
        with pytest.raises(ValueError, match="positive"):
            less_equal_requirements(a, b)
    x, y, t, w, b = layout(2, 3)
    with pytest.raises(ValueError, match="distinct"):
        less_equal_ir(x, y, x[0], w, b)
    with pytest.raises(ValueError, match="clean work"):
        less_equal_ir(x, y, t, w[:-1], b)
    with pytest.raises(ValueError, match="classical"):
        less_equal_ir(x, y, t, w, (-1,) * len(b))

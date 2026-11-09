"""Pinned constant comparison gates, coherent XOR target and clean flags."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.comparison.less_than import (
    less_than_constant_ir,
    less_than_constant_workspace_size,
    source_less_than_constant_cost,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit
CASES = [
    (n, t)
    for n in (1, 2, 3, 5, 8)
    for t in sorted({0, 1, (1 << n) // 2, (1 << n) - 1, 1 << n, (1 << n) + 7})
]


def layout(n: int, t: int) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    size = less_than_constant_workspace_size(n, t)
    return (
        tuple(range(n)),
        tuple(range(n + 1, n + 1 + size)),
        tuple(range(40, 40 + n)) if size else (),
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width,threshold", CASES)
def test_literal_source_comparison_gates_outputs_conditions_and_cost(
    width: int, threshold: int
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.arithmetic.comparison import LessThanConstant
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    x, work, bits = layout(width, threshold)
    source = LessThanConstant(width, threshold)
    assert_source_model(source_less_than_constant_cost(width, threshold), source)
    actual = less_than_constant_ir(x, threshold, width, work, bits)
    expected = reference_arithmetic_gates(
        source,
        {"x": x, "target": (width,)},
        work,
        dict(zip(work[1:], bits, strict=True)),
    )
    assert_source(actual, expected)
    model = get_cost_value(source, QECGatesCost())
    counts = Counter(g.name for g in actual)
    if work:
        assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * width
        assert counts["measure"] == model.measurement == width
        assert counts["reset"] == width
        assert counts["cx"] == 8 * width + threshold.bit_count()
        assert counts["x"] == 2 + 2 * threshold.bit_count()
    else:
        assert counts == {"x": 1}
        assert model.total_t_count() == model.measurement == 0
        assert model.clifford == 1


@pytest.mark.integration
@pytest.mark.parametrize(
    "width,threshold",
    [
        (n, t)
        for n in (1, 2, 3, 4)
        for t in sorted({0, 1, (1 << n) // 2, (1 << n) - 1, 1 << n, (1 << n) + 7})
    ],
)
def test_every_complex_input_target_spectator_and_clean_flags(
    width: int, threshold: int
) -> None:
    x, work, bits = layout(width, threshold)
    gates = less_than_constant_ir(x, threshold, width, work, bits)
    n_qubits = width + 1 + len(work) + 1
    data = torch.arange(1 << (width + 2), dtype=torch.int64)
    spectator = data & 1
    prefix = data >> 1
    value = prefix >> 1
    target = prefix & 1
    mapped = (value << 1) | (target ^ (value < threshold))
    coeff = data.to(torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << n_qubits, dtype=torch.complex128)
    state[(prefix << (len(work) + 1)) | spectator] = amps
    expected = torch.zeros_like(state)
    expected[(mapped << (len(work) + 1)) | spectator] = amps
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
    restored = run_dynamic(
        circuit.append(FieldOperation(n_qubits, gates)),
        shots=32,
        seed=41,
        strategy="trajectory",
    )
    torch.testing.assert_close(
        restored.final_states.reshape(32, -1), state.expand(32, -1), rtol=0, atol=1e-12
    )
    if bits:
        assert (
            len(set(map(tuple, result.classical_bits[:, 40 : 40 + width].tolist())))
            == 1 << width
        )


def test_comparison_rejects_invalid_inputs_and_workspace() -> None:
    for width in (0, -1, True):
        with pytest.raises(ValueError, match="width"):
            less_than_constant_workspace_size(width, 1)
    for threshold in (-1, True):
        with pytest.raises(ValueError, match="threshold"):
            less_than_constant_workspace_size(3, threshold)
    x, w, b = layout(3, 5)
    with pytest.raises(ValueError, match="distinct nonnegative quantum"):
        less_than_constant_ir(x, 5, 0, w, b)
    with pytest.raises(ValueError, match="clean work"):
        less_than_constant_ir(x, 5, 3, w[:-1], b)
    with pytest.raises(ValueError, match="classical slots"):
        less_than_constant_ir(x, 5, 3, w, (1, 1, 2))
    with pytest.raises(ValueError, match="classical slots"):
        less_than_constant_ir(x, 5, 3, w, (-1, 1, 2))

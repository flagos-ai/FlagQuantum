"""Pinned multi-control reduction with coherent targets and clean measured work."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.multi_control import (
    multi_and_ir,
    multi_control_x_ir,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit
PATTERNS = [
    v
    for n in range(9)
    for v in dict.fromkeys(((0,) * n, (1,) * n, tuple(i % 2 for i in range(n))))
]


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("values", PATTERNS)
def test_source_multi_control_x_gates_and_model(values: tuple[int, ...]) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.mcmt.multi_control_pauli import MultiControlX
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    n = len(values)
    ctrl = tuple(range(n))
    target = n
    work = tuple(range(n + 1, n + max(0, n - 1) + 1))
    bits = tuple(range(40, 40 + len(work)))
    actual = multi_control_x_ir(ctrl, values, target, work, bits)
    source = MultiControlX(values)
    inputs = {"target": (target,)}
    if ctrl:
        inputs["controls"] = ctrl
    expected = reference_arithmetic_gates(
        source, inputs, work, dict(zip(work, bits, strict=True))
    )
    assert_source(actual, expected)
    counts = Counter(g.name for g in actual)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * max(0, n - 1)
    assert counts["measure"] == model.measurement == max(0, n - 1)
    if n >= 2:
        assert counts["cx"] == 6 * (n - 1) + 1


@pytest.mark.parametrize("values", [v for v in PATTERNS if len(v) <= 4])
def test_all_control_target_amplitudes_work_cleanup_and_involution(
    values: tuple[int, ...],
) -> None:
    n = len(values)
    ctrl = tuple(range(n))
    target = n
    work = tuple(range(n + 1, n + max(0, n - 1) + 1))
    bits = tuple(range(40, 40 + len(work)))
    gates = multi_control_x_ir(ctrl, values, target, work, bits)
    n_qubits = n + 1 + len(work) + 1
    data = torch.arange(1 << (n + 2), dtype=torch.int64)
    prefix = data >> 1
    spectator = data & 1
    control = prefix >> 1
    z = prefix & 1
    pattern = sum(v << (n - i - 1) for i, v in enumerate(values))
    mapped = (control << 1) | (z ^ (control == pattern))
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
    result = run_dynamic(circuit, shots=128, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(128, -1),
        expected.expand(128, -1),
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
        assert len(
            set(map(tuple, result.classical_bits[:, 40 : 40 + len(work)].tolist()))
        ) == 1 << len(work)


@pytest.mark.parametrize("values", [(0, 1), (1, 0, 1), (0, 1, 0, 1), (1, 0, 1, 0, 1)])
def test_and_ladder_compute_has_correct_prefix_amplitudes(
    values: tuple[int, ...],
) -> None:
    n = len(values)
    controls = tuple(range(n))
    work = tuple(range(n, 2 * n - 1))
    gates = multi_and_ir(controls, values, work)
    state = torch.zeros(1 << (2 * n - 1), dtype=torch.complex128)
    expected = torch.zeros_like(state)
    amps = torch.tensor(
        [complex(i + 1, 2 * i - 3) for i in range(1 << n)], dtype=torch.complex128
    )
    amps /= amps.norm()
    state[:: 1 << (n - 1)] = amps
    for value in range(1 << n):
        equal = True
        flags = 0
        for i, bit in enumerate(values):
            equal = equal and ((value >> (n - i - 1)) & 1) == bit
            if i:
                flags = (flags << 1) | int(equal)
        expected[(value << (n - 1)) | flags] = amps[value]
    computed = Circuit(2 * n - 1, inputs=state, dtype=torch.complex128).append(
        FieldOperation(2 * n - 1, gates)
    )
    torch.testing.assert_close(computed.state().flatten(), expected, rtol=0, atol=1e-12)


def test_invalid_control_reduction_inputs() -> None:
    with pytest.raises(ValueError, match="zero/one"):
        multi_control_x_ir((0, 1), (1, True), 2, (3,), (40,))
    with pytest.raises(ValueError, match="distinct nonnegative"):
        multi_control_x_ir((0, 0), (1, 1), 2, (3,), (40,))
    with pytest.raises(ValueError, match="clean work"):
        multi_control_x_ir((0, 1), (1, 1), 2, (), ())
    with pytest.raises(ValueError, match="classical slots"):
        multi_control_x_ir((0, 1), (1, 1), 2, (3,), (-1,))
    with pytest.raises(
        ValueError, match="operands must be distinct from controls and work"
    ):
        multi_control_x_ir((0, 1), (1, 1), 0, (3,), (40,))
    with pytest.raises(ValueError, match="at least two"):
        multi_and_ir((0,), (1,), ())

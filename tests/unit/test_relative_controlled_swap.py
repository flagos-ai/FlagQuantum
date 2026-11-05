"""Literal relative-phase exchange parity against the locked source."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.relative_controlled_swap import (
    relative_controlled_swap_ir,
)
from flagquantum.core.ir import Instruction

pytestmark = pytest.mark.unit


def source_relative_swap(
    control: int, left: tuple[int, ...], right: tuple[int, ...]
) -> tuple[Instruction, ...]:
    import cirq
    from qualtran.bloqs.swap_network.cswap_approx import CSwapApprox

    op = CSwapApprox(len(left)).on(*map(cirq.LineQubit, (control, *left, *right)))
    names = {
        cirq.CNOT: "cx",
        cirq.H: "h",
        cirq.S: "s",
        cirq.S**-1: "sdg",
        cirq.T: "t",
        cirq.T**-1: "tdg",
    }
    return tuple(
        Instruction(names[gate.gate], tuple(q.x for q in gate.qubits))
        for gate in cirq.decompose(op, keep=lambda gate: gate.gate in names)
    )


def assert_complex_stream(
    actual: tuple[Instruction, ...],
    source: tuple[Instruction, ...],
    inverse: tuple[Instruction, ...],
    width: int,
) -> None:
    import cirq
    import numpy as np

    names = {
        "cx": cirq.CNOT,
        "h": cirq.H,
        "s": cirq.S,
        "sdg": cirq.S**-1,
        "t": cirq.T,
        "tdg": cirq.T**-1,
    }
    # Include a spectator and nonfactorized amplitudes over every basis column.
    amplitudes = torch.arange(1 << (width + 1), dtype=torch.float64)
    state = torch.complex(amplitudes + 1, (amplitudes % 7 - 3) * amplitudes)
    state /= state.norm()
    wires = cirq.LineQubit.range(width + 1)
    expected = (
        cirq.Simulator(dtype=np.complex128)
        .simulate(
            cirq.Circuit(
                names[g.name].on(*(wires[q] for q in g.wires)) for g in source
            ),
            qubit_order=wires,
            initial_state=state.numpy().copy(),
        )
        .final_state_vector
    )
    circuit = Circuit(width + 1, inputs=state, dtype=torch.complex128).append(
        FieldOperation(width + 1, actual)
    )
    torch.testing.assert_close(
        circuit.state().flatten(), torch.from_numpy(expected), rtol=0, atol=2e-12
    )
    circuit.append(FieldOperation(width + 1, inverse))
    torch.testing.assert_close(circuit.state().flatten(), state, rtol=0, atol=2e-12)


@pytest.mark.parametrize("width", [1, 2, 3, 5, 8])
def test_original_exchange_stream_and_cost(width: int) -> None:
    from qualtran.bloqs.swap_network.cswap_approx import CSwapApprox
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    left = tuple(range(1, width + 1))
    right = tuple(range(width + 1, 2 * width + 1))
    gates = relative_controlled_swap_ir(0, left, right)
    source = source_relative_swap(0, left, right)
    assert gates == source
    counts = Counter(gate.name for gate in gates)
    assert counts == {
        "t": 2 * width,
        "tdg": 2 * width,
        "h": 8 * width,
        "s": 4 * width,
        "sdg": 4 * width,
        "cx": 6 * width - 1,
    }
    model = get_cost_value(CSwapApprox(width), QECGatesCost())
    assert model.total_t_count() == 4 * width
    assert model.measurement == 0
    assert get_cost_value(CSwapApprox(width), QubitCount()) == 2 * width + 1
    assert_complex_stream(gates, source, gates, 2 * width + 1)


def test_rejects_missing_mismatched_and_aliased_words() -> None:
    for left, right in [((), ()), ((1,), (2, 3)), ((1,), (0,))]:
        with pytest.raises(ValueError):
            relative_controlled_swap_ir(0, left, right)

"""Compare the executable SF SELECT with the paper operator in every small branch."""

from itertools import product

import pytest
import torch

from flagquantum.algorithms.primitives.multiplexers.select_single_factorization import (
    SelectSingleFactorization,
)
from flagquantum.algorithms.primitives.single_factorization_resource_model import (
    source_single_factorization_select_cost,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import forced_branch

pytestmark = pytest.mark.unit


def _binding(size: int, additional_control: bool) -> SelectSingleFactorization:
    width = (size // 2 - 1).bit_length()
    spin = 2 * width
    start = spin + 3 + int(additional_control)
    work = SelectSingleFactorization.workspace_size(
        size, additional_control=additional_control
    )
    return SelectSingleFactorization(
        p=tuple(range(width)),
        q=tuple(range(width, spin)),
        spin=spin,
        succ_pq=spin + 1,
        succ_l=spin + 2,
        system=tuple(range(start, start + size)),
        workspace=tuple(range(start + size, start + size + work)),
        l_ne_zero=spin + 3 if additional_control else None,
    )


@pytest.mark.parametrize(
    "size,additional_control", list(product((2, 4), (False, True)))
)
def test_every_measurement_history(size, additional_control):
    op = _binding(size, additional_control)
    wires = op.workspace[-1] + 1
    gates = DynamicCircuit(wires).append(op).to_ir().instructions
    extra = int(additional_control)
    events = (0 if size == 2 else 2 * (size - 1)) + 1 + extra
    assert sum(gate.name == "measure" for gate in gates) == events
    assert sum(gate.name in {"t", "tdg"} for gate in gates) == 4 * events
    source = estimate_component_resources(source_single_factorization_select_cost(size))
    assert source.gates.toffoli == 2 * (size - 2)
    assert source.gates.measurement == 0

    state = torch.zeros(1 << wires, dtype=torch.complex128)
    expected = torch.zeros_like(state)
    data_width = size + len(op.workspace)
    flag_width = 2 + extra
    for p, q, spin, flags, data in product(
        range(size // 2),
        range(size // 2),
        range(2),
        range(1 << flag_width),
        range(1 << size),
    ):
        address = 2 * (p * (size // 2) + q) + spin
        prefix = ((address << flag_width) | flags) << data_width
        index = prefix | (data << len(op.workspace))
        amplitude = complex(index + 1, index % 7 - 3)
        state[index] = amplitude
        if flags == (1 << flag_width) - 1:
            i, j = sorted((2 * p + spin, 2 * q + spin))
            between = sum((data >> (size - 1 - k)) & 1 for k in range(i + 1, j))
            left, right = 1 << (size - 1 - i), 1 << (size - 1 - j)
            if p == q:
                amplitude *= 1 if data & left else -1
            else:
                amplitude *= (-1) ** between
                if p > q:
                    amplitude *= 1 if bool(data & left) != bool(data & right) else -1
                data ^= left | right
        expected[prefix | (data << len(op.workspace))] = amplitude

    norm = state.norm()
    state /= norm
    expected /= norm * 2 ** (events / 2)
    for history in product((0, 1), repeat=events):
        torch.testing.assert_close(
            forced_branch(state, gates, history), expected, atol=2e-12, rtol=0
        )


def test_repeated_append_allocates_distinct_feedback_slots():
    op = _binding(6, True)
    circuit = DynamicCircuit(op.workspace[-1] + 1)
    circuit.append(op)
    first = circuit.to_ir().instructions
    circuit.append(op)
    second = circuit.to_ir().instructions[len(first) :]
    first_slots = {
        gate.metadata["classical_bit"] for gate in first if gate.name == "measure"
    }
    second_slots = {
        gate.metadata["classical_bit"] for gate in second if gate.name == "measure"
    }
    assert first_slots == set(range(5))
    assert second_slots == set(range(5, 10))

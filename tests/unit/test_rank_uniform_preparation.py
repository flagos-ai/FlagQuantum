"""Verify the complete quantum rank-conditioned preparation and adjoint."""

from itertools import product
from math import acos, cos, sin, sqrt, tau

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.rank_uniform_preparation import (
    rank_rotation_word,
    rank_uniform_ir,
    rank_uniform_requirements,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_sparse_chemistry_prepare import _branch

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("rank", [1, 2, 3, 4, 5, 8, 17])
@pytest.mark.parametrize("bits", [1, 4, 10])
def test_rank_rotation_word_is_nearest_amplification_angle(rank, bits):
    capacity = 1 << (rank - 1).bit_length()
    exact = 2 * acos(sqrt(capacity / (4 * rank)))
    step = tau / (1 << bits)
    word = rank_rotation_word(rank, bits)
    assert abs(step * word - exact) <= step / 2


@pytest.mark.parametrize("width,bits", [(0, 1), (1, 2), (2, 4), (3, 5)])
def test_rank_signal_success_failure_and_adjoint(width, bits):
    work, feedback = rank_uniform_requirements(width)
    rank_wires = tuple(range(width))
    angle_wires = tuple(range(width, width + bits))
    index = tuple(range(width + bits, 2 * width + bits))
    rotation, success = 2 * width + bits, 2 * width + bits + 1
    scratch = tuple(range(success + 1, success + 1 + work))
    slots = tuple(range(7, 7 + feedback))
    wires = success + 1 + work
    streams = tuple(
        rank_uniform_ir(
            rank_wires,
            angle_wires,
            index,
            rotation,
            success,
            scratch,
            slots,
            adjoint=adjoint,
        )
        for adjoint in (False, True)
    )
    initial = {}
    expected = {}
    ranks = range(1, (1 << width) + 1)
    for rank in ranks:
        angle_word = rank_rotation_word(rank, bits)
        # Give some ranks a deliberately inaccurate word to exercise leakage.
        if rank % 3 == 0:
            angle_word ^= 1 << (bits - 1)
        prefix = ((rank - 1) << bits) | angle_word
        coefficient = complex(1, (-1) ** rank) / sqrt(2 * len(ranks))
        offset = prefix << (width + 2 + work)
        initial[offset] = coefficient
        capacity = 1 << (rank - 1).bit_length()
        theta = tau * angle_word / (1 << bits)
        rotations = (cos(theta / 2), sin(theta / 2))
        marked_weight = rank * rotations[0] ** 2 / capacity
        for address, ancilla in product(range(capacity), range(2)):
            active = int(address < rank and ancilla == 0)
            factor = 1 - 4 * marked_weight + 2 * active
            value = coefficient * rotations[ancilla] * factor / sqrt(capacity)
            if abs(value) > 1e-12:
                basis = offset | (((address << 2) | (ancilla << 1) | active) << work)
                expected[basis] = value
    assert sum(abs(value) ** 2 for value in expected.values()) == pytest.approx(1)
    for gates, before, after in (
        (streams[0], initial, expected),
        (streams[1], expected, initial),
    ):
        events = sum(gate.name == "measure" for gate in gates)
        t_count = sum(gate.name in {"t", "tdg"} for gate in gates)
        assert (events, work, t_count) == {
            0: (1, 2, 4),
            1: (6, 3, 24),
            2: (16, 7, 64),
            3: (26, 11, 104),
        }[width]
        assert sum(gate.name == "cry" for gate in gates) == 3 * bits
        histories = (
            product((0, 1), repeat=events)
            if events <= 8
            else (
                (pattern * events)[:events] for pattern in ((0,), (1,), (0, 1), (1, 0))
            )
        )
        for history in histories:
            assert _branch(
                gates, wires, history, before, normalize_measurements=True
            ) == pytest.approx(after, abs=2e-12)

    if width == 1:
        initial_vector = torch.zeros(1 << wires, dtype=torch.complex128)
        expected_vector = torch.zeros_like(initial_vector)
        for basis, value in initial.items():
            initial_vector[basis] = value
        for basis, value in expected.items():
            expected_vector[basis] = value
        circuit = DynamicCircuit(wires, inputs=initial_vector, dtype=torch.complex128)
        circuit.append(FieldOperation(wires, streams[0]))
        forward = run_dynamic(circuit, shots=8, seed=42, strategy="trajectory")
        torch.testing.assert_close(
            forward.final_states.reshape(8, -1),
            expected_vector.expand(8, -1),
            atol=2e-12,
            rtol=0,
        )
        circuit.append(FieldOperation(wires, streams[1]))
        reverse = run_dynamic(circuit, shots=8, seed=42, strategy="trajectory")
        torch.testing.assert_close(
            reverse.final_states.reshape(8, -1),
            initial_vector.expand(8, -1),
            atol=2e-12,
            rtol=0,
        )

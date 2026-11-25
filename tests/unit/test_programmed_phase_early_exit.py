"""Certify physical early exit through existing local guarded effects."""

from cmath import exp
from itertools import product
from math import pi

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from examples.algorithms.programmed_phase_catalog import ProgrammedRotationExample
from flagquantum.compiler.registers.phase.programmed_phase import (
    programmed_phase_rotation_ir,
)
from flagquantum.compiler.registers.phase.programmed_phase_resource_model import (
    ProgrammedRotationModel,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from flagquantum.simulation.density_matrix import density_matrix_from_ir
from tests.unit.measurement_checks import forced_branch

pytestmark = pytest.mark.unit


def initial_state() -> torch.Tensor:
    state: torch.Tensor = torch.tensor(
        [1, 2j, 0, 0, -0.5j, 1 + 2j, 0, 0], dtype=torch.complex128
    )
    state /= state.norm()
    return state


@pytest.mark.parametrize("rounds", [1, 2, 3, 4])
@pytest.mark.parametrize("correct", [False, True])
def test_early_exit_preserves_every_complex_branch(rounds: int, correct: bool) -> None:
    slots = (5, 2, 7, 1)[:rounds]
    padded = programmed_phase_rotation_ir(
        0, 1, slots, exponent=-0.375, apply_final_correction=correct
    )
    early = programmed_phase_rotation_ir(
        0, 1, slots, exponent=-0.375, apply_final_correction=correct, early_exit=True
    )
    ir = CircuitIR(3, early)
    assert CircuitIR.from_json(ir.to_json()) == ir
    for history in product((0, 1), repeat=rounds):
        actual = forced_branch(initial_state(), early, history)
        expected = forced_branch(initial_state(), padded, history)
        torch.testing.assert_close(actual, expected, rtol=0, atol=2e-12)
        assert torch.count_nonzero(actual.reshape(2, 2, 2)[:, 1, :]) == 0


@pytest.mark.parametrize("strategy", ["trajectory", "batched"])
@pytest.mark.parametrize("correct", [False, True])
def test_append_skips_measurements_after_success(strategy: str, correct: bool) -> None:
    rounds, shots = 3, 256
    model = ProgrammedRotationModel(
        0.125, rounds=rounds, apply_final_correction=correct
    )
    circuit = DynamicCircuit(3, inputs=initial_state(), dtype=torch.complex128)
    circuit.append(ProgrammedRotationExample(model))
    result = run_dynamic(circuit, shots=shots, seed=42, strategy=strategy)
    histories = result.classical_bits.tolist()
    assert set(map(tuple, histories)) == {(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)}
    attempts = sum(min(sum(history) + 1, rounds) for history in histories)
    assert result.statistics["measurement_count"] == attempts < rounds * shots
    for history, actual in zip(histories, result.final_states, strict=True):
        failures = sum(history)
        phase = exp(1j * pi * (2**failures - 1) * 0.125)
        expected = initial_state() * phase
        net = 0.125 if failures < rounds or correct else -(2**rounds - 1) * 0.125
        expected[4:] *= exp(1j * pi * net)
        torch.testing.assert_close(actual.reshape(-1), expected, rtol=0, atol=2e-12)


@pytest.mark.parametrize("correct", [False, True])
def test_density_channel_matches_padded_execution(correct: bool) -> None:
    circuits = []
    for early_exit in (False, True):
        gates = programmed_phase_rotation_ir(
            0,
            1,
            (0, 1, 2),
            exponent=0.125,
            apply_final_correction=correct,
            early_exit=early_exit,
        )
        circuit = DynamicCircuit(3, inputs=initial_state(), dtype=torch.complex128)
        circuits.append(circuit.append(FieldOperation(3, gates)))
    torch.testing.assert_close(
        density_matrix_from_ir(circuits[0]),
        density_matrix_from_ir(circuits[1]),
        rtol=0,
        atol=2e-12,
    )

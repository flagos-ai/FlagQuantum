"""The original 32-by-32 toroidal walk retains its complete quantum state."""

import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.tutorials.advanced_tutorials.discrete_quantum_walk_2d.discrete_time_quantum_walk import (
    main as walk,
)
from examples.synthesis.tutorials.advanced_tutorials.discrete_quantum_walk_2d.discrete_time_quantum_walk import (
    physics,
)
from examples.synthesis.tutorials.advanced_tutorials.discrete_quantum_walk_2d.discrete_time_quantum_walk.comparisons.circuit import (
    reference_circuit,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError

pytestmark = pytest.mark.integration
SOURCE = (
    Path(__file__).resolve().parents[2]
    / "tests/fixtures/classiq_qmod/corpus/original/tutorials/advanced_tutorials/discrete_quantum_walk_2d/discrete_time_quantum_walk.qmod"
)


def check(program: synth.Program, expected: torch.Tensor) -> None:
    vector = expected.flatten()
    actual = fq.run(program)
    complete = actual.statevector().reshape(4096, -1)
    torch.testing.assert_close(complete[:, 0], vector, atol=1e-12, rtol=0)
    assert float(complete[:, 1:].abs().square().sum()) < 1e-22
    marginal = vector.abs().square().reshape(32, 32, 4).sum(dim=2)
    probabilities = program.probabilities(actual, registers=("x", "y"))
    for x in range(32):
        for y in range(32):
            assert probabilities.get((x, y), 0) == pytest.approx(
                float(marginal[x, y]), abs=1e-12, rel=0
            )


@pytest.mark.parametrize("steps", [0, 1, 2, 3])
def test_original_native_and_independent_full_state(steps: int) -> None:
    expected = fq.run(reference_circuit(steps)).statevector()
    check(walk.build_program(steps), expected)
    check(synth.synthesize(SOURCE, classical_inputs={"t": steps}), expected)


@pytest.mark.parametrize("variant", [0, 1])
@pytest.mark.parametrize("steps", [0, 1, 2, 3])
def test_wraparound_and_signed_coherent_inputs(variant: int, steps: int) -> None:
    if variant == 0:
        x = tuple(1.0 if word == 31 else 0.0 for word in range(32))
        y = tuple(1.0 if word == 0 else 0.0 for word in range(32))
        coin = (0.5, -0.5, 0.5, -0.5)
    else:
        x = tuple((-1.0 if word % 2 else 1.0) / math.sqrt(32) for word in range(32))
        y = tuple(
            1 / math.sqrt(2) if word == 0 else -1 / math.sqrt(2) if word == 31 else 0.0
            for word in range(32)
        )
        coin = (0.6, 0.0, -0.8, 0.0)
    expected = fq.run(reference_circuit(steps, x, y, coin)).statevector()
    check(walk.build_program(steps, x, y, coin), expected)


def test_coin_is_retained_environment_not_clean_work() -> None:
    program = walk.build_program(1)
    result = fq.run(program).statevector().reshape(1024, 4, -1)
    assert float(result[:, 1:, 0].abs().square().sum()) > 0.5
    resources = program.report["resources"]
    work = program.report["workspace_qubits"]
    assert isinstance(resources, ResourceEstimate) and isinstance(work, int)
    assert resources.n_qubits == 12 + work
    assert tuple(program.registers) == ("x", "y")


@pytest.mark.parametrize("steps", [-1, 17, True])
def test_invalid_step_count_refuses(steps: int) -> None:
    with pytest.raises(ValueError, match="step count"):
        walk.build_program(steps)


def test_invalid_amplitude_width_refuses() -> None:
    with pytest.raises(ValueError, match="amplitudes"):
        walk.build_program(1, (1.0, 0.0), physics.POSITION_AMPLITUDES)


@pytest.mark.parametrize(
    "constraint", ["qubits", "workspace", "operations", "depth", "memory"]
)
def test_walk_actual_resource_limits_refuse(constraint: str) -> None:
    classical: dict[str, float | list[float]] = {
        "t": 2,
        "x_amplitudes": list(physics.POSITION_AMPLITUDES),
        "y_amplitudes": list(physics.POSITION_AMPLITUDES),
        "coin_amplitudes": list(physics.COIN_AMPLITUDES),
    }
    with pytest.raises((synth.SynthesisError, HybridVerificationError, PlanningError)):
        if constraint == "qubits":
            synth.synthesize(walk.main, classical_inputs=classical, max_qubits=12)
        elif constraint == "workspace":
            synth.synthesize(walk.main, classical_inputs=classical, max_workspace=0)
        elif constraint == "operations":
            synth.synthesize(walk.main, classical_inputs=classical, max_operations=1)
        elif constraint == "depth":
            synth.synthesize(walk.main, classical_inputs=classical, max_depth=1)
        else:
            synth.synthesize(
                walk.main, classical_inputs=classical, memory_limit_bytes=16
            )

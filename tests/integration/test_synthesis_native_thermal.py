"""Complete native thermal channels match independent Circuit/Kraus evolution."""

import cmath
import math

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_state_preparation.gibbs.quantum_thermal_state_preparation.comparisons.circuit import (
    kraus_operators,
    reference_circuit,
    reference_density,
)
from examples.synthesis.algorithms.quantum_state_preparation.gibbs.quantum_thermal_state_preparation.main import (
    build_program,
    main,
)
from examples.synthesis.algorithms.quantum_state_preparation.gibbs.quantum_thermal_state_preparation.physics import (
    GAUSSIAN,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_qmod_thermal_reset import _original
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.integration
DTYPE = torch.complex128


def initial_state(amplitudes: tuple[float, ...], phase: float) -> torch.Tensor:
    vector = torch.tensor(amplitudes, dtype=DTYPE)
    vector[1::2] *= cmath.exp(1j * phase)
    return vector


@pytest.mark.parametrize(
    "rounds,delta,amplitudes,phase",
    [
        (50, 0.1, (1.0, 0.0, 0.0, 0.0), 0.0),
        (2, 0.2, (1.0, 0.0, 0.0, 0.0), 0.0),
        (1, 0.1, (0.5, -0.5, 0.5, -0.5), 0.37),
        (1, 0.2, (0.5, 0.5, -0.5, -0.5), -0.61),
        (1, 0.0, (0.5, -0.5, 0.5, 0.5), 0.29),
    ],
)
def test_complete_channel_and_all_reset_environment(
    rounds: int,
    delta: float,
    amplitudes: tuple[float, ...],
    phase: float,
) -> None:
    program = build_program(rounds, delta, amplitudes, phase)
    assert program.execution.state_mode == "density_matrix"
    assert program.registers == {
        "state": (0, 1),
        "boltz": (2,),
        "bohr": (3, 4, 5, 6),
        "jump": (7, 8),
    }
    operators = kraus_operators(delta)
    completeness = torch.einsum("kji,kjl->il", operators.conj(), operators)
    torch.testing.assert_close(
        completeness, torch.eye(4, dtype=DTYPE), atol=1e-12, rtol=0
    )
    expected = reference_density(operators, rounds, initial_state(amplitudes, phase))
    actual = fq.run(program).state
    assert isinstance(actual, torch.Tensor)
    full = torch.zeros((1024, 1024), dtype=DTYPE)
    full[::256, ::256] = expected
    torch.testing.assert_close(actual.reshape(1024, 1024), full, atol=1e-10, rtol=0)
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_qubits == 10
    assert resources.channel_count == 8 * rounds
    assert program.report["workspace_qubits"] == 0
    if rounds == 50 and phase == 0:
        assert resources.n_operations == 57700
        original = synth.synthesize(_original())
        original_state = fq.run(original).state
        assert isinstance(original_state, torch.Tensor)
        torch.testing.assert_close(
            original_state.reshape(4, 512, 4, 512)[:, 0, :, 0],
            expected,
            atol=1e-10,
            rtol=0,
        )
        original_resources = original.report["resources"]
        assert isinstance(original_resources, ResourceEstimate)
        assert original_resources.channel_count == 400
        assert original_resources.n_operations == 22200


@pytest.mark.parametrize("basis", range(-1, 4))
def test_native_encoded_block_preserves_absolute_phase_and_bit_order(
    basis: int,
) -> None:
    source = """from flagquantum.synth import *
from examples.synthesis.algorithms.quantum_state_preparation.gibbs.quantum_thermal_state_preparation.main import block
@qfunc
def main(weights:list[float],times:list[float],state:Output[QArray[QBit]],boltz:Output[QBit],bohr:Output[QArray[QBit]],jump:Output[QArray[QBit]]):
    allocate(2,state)
    allocate(boltz)
    allocate(4,bohr)
    allocate(2,jump)
    PREPARE
    block(weights,times,state,boltz,bohr,jump)
"""
    preparation = (
        "H(state[0]);RZ(0.37,state[0])"
        if basis < 0
        else "\n    ".join(f"X(state[{bit}])" for bit in range(2) if basis >> bit & 1)
    )
    program = synth.synthesize(
        program_definition(source.replace("PREPARE", preparation)),
        classical_inputs={
            "weights": list(GAUSSIAN),
            "times": [-0.2 * math.pi * 2**k for k in range(4)],
        },
        max_workspace=0,
    )
    circuit = fq.Circuit(9, dtype=DTYPE)
    if basis >= 0:
        for bit in range(2):
            if basis >> bit & 1:
                circuit.x(1 - bit)
    else:
        circuit.h(1).rz(1, theta=0.37)
    expected = fq.run(circuit.compose(reference_circuit())).statevector()
    torch.testing.assert_close(
        fq.run(program).statevector(), expected, atol=1e-11, rtol=0
    )


@pytest.mark.parametrize(
    "qubits,operations,memory",
    [(9, None, None), (None, 50000, None), (None, None, 1024**2)],
)
def test_full_program_budgets_refuse(
    qubits: int | None, operations: int | None, memory: int | None
) -> None:
    with pytest.raises(
        (PlanningError, HybridVerificationError), match="budget|limit|memory|infeasible"
    ):
        synth.synthesize(
            main,
            classical_inputs={
                "prepare_initial": 0,
                "initial": [1, 0, 0, 0],
                "initial_phase": 0.0,
                "rounds": 50,
                "weights": list(GAUSSIAN),
                "times": [-0.2 * math.pi * 2**k for k in range(4)],
                "theta": math.asin(2 * math.sqrt(0.1)),
            },
            max_qubits=qubits,
            max_operations=operations,
            memory_limit_bytes=memory,
        )


@pytest.mark.parametrize(
    "rounds,delta",
    [(0, 0.1), (51, 0.1), (True, 0.1), (50, -0.1), (50, 0.26), (50, float("nan"))],
)
def test_classical_validation_refuses(rounds: int, delta: float) -> None:
    with pytest.raises(ValueError):
        build_program(rounds, delta)

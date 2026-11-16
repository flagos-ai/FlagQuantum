"""Native OAA preserves the source's Pauli order, full block and global phase."""

from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.amplitude_amplification_and_estimation.oblivious_amplitude_amplification.comparisons.circuit import (
    amplification_matrix,
    reference_circuit,
)
from examples.synthesis.algorithms.amplitude_amplification_and_estimation.oblivious_amplitude_amplification.main import (
    build_program,
    main,
)
from examples.synthesis.algorithms.amplitude_amplification_and_estimation.oblivious_amplitude_amplification.physics import (
    AMPLITUDES,
)
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
APP = "examples.synthesis.algorithms.amplitude_amplification_and_estimation.oblivious_amplitude_amplification"


def test_unchanged_original_native_and_independent_state() -> None:
    original = (
        ROOT
        / "tests/fixtures/classiq_qmod/corpus/original/algorithms/amplitude_amplification_and_estimation/oblivious_amplitude_amplification/oblivious_amplitude_amplification.qmod"
    )
    expected = fq.run(reference_circuit()).statevector()
    for program in (synth.synthesize(original), build_program()):
        result = fq.run(program)
        torch.testing.assert_close(result.statevector(), expected, atol=1e-12, rtol=0)
        assert program.report["workspace_qubits"] == 0
        probabilities = program.probabilities(result, registers=("data", "block"))
        for data in range(4):
            for block in range(4):
                assert probabilities.get((data, block), 0) == pytest.approx(
                    float(expected.flatten()[4 * data + block].abs().square()),
                    abs=1e-12,
                    rel=0,
                )


@pytest.mark.parametrize("repetitions", [0, 1, 2])
@pytest.mark.parametrize("variant", [0, 1])
def test_nonzero_real_data_models(repetitions: int, variant: int) -> None:
    values = ((0.3, -0.4, 0.5, -0.7), (-0.2, 0.6, 0.4, 0.5))[variant]
    norm = sum(v * v for v in values) ** 0.5
    amplitudes = tuple(v / norm for v in values)
    program = build_program(repetitions, amplitudes)
    torch.testing.assert_close(
        fq.run(program).statevector(),
        fq.run(reference_circuit(repetitions, amplitudes)).statevector(),
        atol=1e-12,
        rtol=0,
    )
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("basis", range(16))
def test_every_data_and_nonzero_block_column(basis: int) -> None:
    amplitudes = [float(i == basis) for i in range(16)]
    entry = definition(
        "data:Output[QArray[QBit]],block:Output[QArray[QBit]]",
        "allocate(2,block)\nallocate(2,data)\n"
        + f"inplace_prepare_amplitudes({amplitudes!r},0,(block,data))\nencode(block,data)\nzero_phase(block)\nreflect(block,data)",
        "from " + APP + ".operations import encode,zero_phase,reflect\n",
    )
    program = synth.synthesize(entry, max_workspace=0)
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(),
        amplification_matrix()[:, basis],
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("repetitions", [-1, 17, True])
def test_invalid_repetition_count(repetitions: int) -> None:
    with pytest.raises(ValueError, match="0..16"):
        build_program(repetitions)


@pytest.mark.parametrize("budget", ["operations", "width", "depth", "memory"])
def test_actual_program_budget_is_enforced(budget: str) -> None:
    bindings: dict[str, float | list[float]] = {
        "repetitions": 1,
        "amplitudes": list(AMPLITUDES),
    }
    with pytest.raises((HybridVerificationError, synth.SynthesisError, PlanningError)):
        if budget == "operations":
            synth.synthesize(main, classical_inputs=bindings, max_operations=1)
        elif budget == "width":
            synth.synthesize(main, classical_inputs=bindings, max_qubits=3)
        elif budget == "depth":
            synth.synthesize(main, classical_inputs=bindings, max_depth=1)
        else:
            synth.synthesize(main, classical_inputs=bindings, memory_limit_bytes=16)

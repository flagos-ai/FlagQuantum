"""Original counting, Grover and QMC states match native and Circuit programs."""

import importlib
import math
from fractions import Fraction
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.qpe_reference import dense_qpe
from flagquantum import synth
from flagquantum.compiler.resource_estimation import ResourceEstimate
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
CASES = (
    (
        "algorithms.amplitude_amplification_and_estimation.quantum_counting.quantum_counting_qpe",
        "algorithms/amplitude_amplification_and_estimation/quantum_counting/quantum_counting_qpe.qmod",
        "phase_var",
        5,
        False,
    ),
    (
        "algorithms.amplitude_amplification_and_estimation.qmc_user_defined",
        "algorithms/amplitude_amplification_and_estimation/qmc_user_defined/qmc_user_defined.qmod",
        "phase_result",
        3,
        True,
    ),
    (
        "tutorials.technology_demonstrations.QPE.qpe_for_grover_operator",
        "tutorials/technology_demonstrations/QPE/qpe_for_grover_operator/qpe_for_grover_operator.qmod",
        "phase_result",
        5,
        False,
    ),
)


@pytest.mark.parametrize("module,original,port,bits,signed", CASES)
def test_complete_state_absolute_phase_and_retained_data(
    module: str, original: str, port: str, bits: int, signed: bool
) -> None:
    entry = importlib.import_module("examples.synthesis." + module + ".main")
    comparison = importlib.import_module(
        "examples.synthesis." + module + ".comparisons.circuit"
    )
    native = entry.build_program()
    source = synth.synthesize(
        ROOT / "tests/fixtures/classiq_qmod/corpus/original" / original
    )
    expected = fq.run(comparison.reference_circuit()).statevector().flatten()
    # Long Grover decompositions accumulate floating-point roundoff; no phase alignment.
    for program in (source, native):
        result = fq.run(program)
        work = program.report["workspace_qubits"]
        assert isinstance(work, int)
        complete = torch.zeros((expected.numel(), 1 << work), dtype=torch.complex128)
        complete[:, 0] = expected
        torch.testing.assert_close(
            result.statevector().flatten(), complete.flatten(), atol=1e-11, rtol=0
        )
        dirty = result.statevector().reshape(expected.numel(), 1 << work)[:, 1:]
        assert float(dirty.abs().square().sum()) < 1e-22
        resources = program.report["resources"]
        assert isinstance(resources, ResourceEstimate)
        assert resources.n_qubits == int(expected.numel()).bit_length() - 1 + work
    assert native.report["workspace_qubits"] == 0
    result = fq.run(native)
    marginal = expected.reshape(1 << bits, -1).abs().square().sum(dim=1)
    raw = native.probabilities(result, registers=(port,))
    numeric = native.numeric_probabilities(result, registers=(port,))
    for word in range(1 << bits):
        integer = word - (1 << bits) if signed and word >= (1 << (bits - 1)) else word
        assert raw.get((word,), 0) == pytest.approx(
            float(marginal[word]), abs=1e-11, rel=0
        )
        assert numeric.get((Fraction(integer, 1 << bits),), 0) == pytest.approx(
            float(marginal[word]), abs=1e-11, rel=0
        )


@pytest.mark.parametrize("module,original,port,bits,signed", CASES)
@pytest.mark.parametrize("variant", [0, 1])
def test_original_operator_on_nonzero_coherent_inputs(
    module: str, original: str, port: str, bits: int, signed: bool, variant: int
) -> None:
    comparison = importlib.import_module(
        "examples.synthesis." + module + ".comparisons.circuit"
    )
    matrix, _, _ = comparison.operators()
    dimension = matrix.shape[0]
    values = tuple(
        (-1) ** (index + variant) * (index + 1 + variant) for index in range(dimension)
    )
    amplitudes = tuple(value / math.hypot(*values) for value in values)
    if "quantum_counting" in module:
        setup = "a:QNum[2]=QNum()\nb:QNum[2]=QNum()\nallocate(a)\nallocate(b)"
        target, call, cleanup = "(b,a)", "grover(a,b)", "drop(a)\ndrop(b)"
    elif "qmc_user_defined" in module:
        setup = "state:QArray[QBit]=QArray()\nallocate(4,state)"
        target, call, cleanup = "state", "grover(state)", "drop(state)"
    else:
        setup = "ind:QBit=QBit()\nx:QNum[4]=QNum()\nallocate(ind)\nallocate(x)"
        target, call, cleanup = "(x,ind)", "grover(x,ind)", "drop(x)\ndrop(ind)"
    program = synth.synthesize(
        definition(
            "p:Output[QNum[2,False,2]]",
            setup
            + f"\ninplace_prepare_amplitudes({amplitudes!r},0,{target})\nallocate(p)\nqpe(lambda:{call},p)\n"
            + cleanup,
            "from examples.synthesis." + module + ".main import grover\n",
        ),
        max_workspace=0,
    )
    identity = torch.eye(dimension, dtype=torch.complex128)
    vector = torch.tensor(amplitudes, dtype=torch.complex128)
    difference = identity[:, 0] - vector
    preparation = (
        identity
        - 2
        * torch.outer(difference, difference.conj())
        / difference.abs().square().sum()
    )
    expected = fq.run(dense_qpe(matrix, preparation, 2)).statevector().flatten()
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(), expected, atol=1e-11, rtol=0
    )

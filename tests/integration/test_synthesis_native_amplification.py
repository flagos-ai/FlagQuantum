"""Native counting, circle and loss programs retain phase-exact amplification."""

import importlib
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.compiler.resource_estimation import ResourceEstimate
from tests.integration.test_synthesis_native_exact_numbers import definition

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
CASES = (
    (
        "algorithms.amplitude_amplification_and_estimation.quantum_counting",
        "algorithms/amplitude_amplification_and_estimation/quantum_counting/quantum_counting.qmod",
    ),
    (
        "algorithms.amplitude_amplification_and_estimation.quantum_counting.quantum_counting_iqae",
        "algorithms/amplitude_amplification_and_estimation/quantum_counting/quantum_counting_iqae.qmod",
    ),
    (
        "tutorials.basics.qmci_pi_estimation",
        "tutorials/basic_tutorials/qmci_pi_estimation/qmci_pi_estimation.qmod",
    ),
    (
        "applications.finance.value_at_risk",
        "applications/finance/value_at_risk/value_at_risk.qmod",
    ),
)


@pytest.mark.parametrize("module,original", CASES)
@pytest.mark.parametrize("k", [0, 1, 2])
def test_original_native_independent_state_and_named_marginal(
    module: str, original: str, k: int
) -> None:
    entry = importlib.import_module("examples.synthesis." + module + ".main")
    reference = importlib.import_module(
        "examples.synthesis." + module + ".comparisons.circuit"
    )
    native = entry.build_program(k)
    source = synth.synthesize(
        ROOT / "tests/fixtures/classiq_qmod/corpus/original" / original,
        classical_inputs={"k": k},
    )
    expected = fq.run(reference.reference_circuit(k)).statevector().flatten()
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
        expected_marginal = expected.reshape(2, -1).abs().square().sum(dim=1)
        probabilities = program.probabilities(result, registers=("indicator",))
        for word in range(2):
            assert probabilities.get((word,), 0) == pytest.approx(
                float(expected_marginal[word]), abs=1e-11, rel=0
            )
    assert native.report["workspace_qubits"] == 0


@pytest.mark.parametrize("module,original", CASES)
@pytest.mark.parametrize("word", [1, -1])
def test_preparation_preserves_nonzero_indicator_xor_and_all_data(
    module: str, original: str, word: int
) -> None:
    entry = importlib.import_module("examples.synthesis." + module + ".main")
    reference = importlib.import_module(
        "examples.synthesis." + module + ".comparisons.circuit"
    )
    matrix = reference.preparation_matrix()
    dimension = matrix.shape[0]
    basis = dimension // 2 + 1 if word == 1 else dimension - 1
    amplitudes = [0.0] * dimension
    amplitudes[basis] = 1.0
    if "value_at_risk" in module:
        setup = "asset:QNum[7]=QNum()\nallocate(asset)\nallocate(indicator)"
        target, call, cleanup = (
            "(asset,indicator)",
            "prepare(probabilities,asset,indicator)",
            "drop(asset)",
        )
        ports = "probabilities:list[float],indicator:Output[QBit]"
        bindings: dict[str, float | list[float]] = {
            "probabilities": list(entry.PROBABILITIES)
        }
    else:
        width = 1 if "qmci" in module else 2
        first, second = ("x", "y") if width == 1 else ("a", "b")
        setup = f"{first}:QNum[{width}]=QNum()\n{second}:QNum[{width}]=QNum()\nallocate({first})\nallocate({second})\nallocate(indicator)"
        target, call, cleanup = (
            f"({second},{first},indicator)",
            f"prepare({first},{second},indicator)",
            f"drop({first})\ndrop({second})",
        )
        ports, bindings = "indicator:Output[QBit]", {}
    program = synth.synthesize(
        definition(
            ports,
            setup
            + f"\ninplace_prepare_amplitudes({amplitudes!r},0,{target})\n{call}\n"
            + cleanup,
            "from examples.synthesis." + module + ".main import prepare\n",
        ),
        classical_inputs=bindings,
        max_workspace=0,
    )
    expected = matrix[:, basis]
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(), expected, atol=1e-11, rtol=0
    )
    # A reset of the marker would lose a full unitary column even if zero input passed.
    assert math.isclose(float(expected.abs().square().sum()), 1.0, abs_tol=1e-12)


@pytest.mark.parametrize("module,original", CASES)
@pytest.mark.parametrize("k", [-1, 17, True])
def test_iteration_bound_rejects_invalid_values(
    module: str, original: str, k: int
) -> None:
    entry = importlib.import_module("examples.synthesis." + module + ".main")
    with pytest.raises(ValueError, match="0..16"):
        entry.build_program(k)


@pytest.mark.parametrize("module,original", CASES)
def test_real_operation_budget_cannot_be_bypassed(module: str, original: str) -> None:
    entry = importlib.import_module("examples.synthesis." + module + ".main")
    bindings: dict[str, float | list[float]] = {"k": 1}
    if "value_at_risk" in module:
        bindings = {"k": 1, "probabilities": list(entry.PROBABILITIES)}
    with pytest.raises((HybridVerificationError, synth.SynthesisError)):
        synth.synthesize(entry.main, classical_inputs=bindings, max_operations=1)

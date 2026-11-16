"""The original GQSP phase model preserves complex blocks and boundary loops."""

from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_primitives.gqsp.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.algorithms.quantum_primitives.gqsp.main import (
    build_program,
    main,
)
from examples.synthesis.algorithms.quantum_primitives.gqsp.physics import PHASES
from flagquantum import synth
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.compiler.resource_estimation import ResourceEstimate
from flagquantum.errors import PlanningError

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def test_unchanged_original_native_and_independent_full_states() -> None:
    original = (
        ROOT
        / "tests/fixtures/classiq_qmod/corpus/original/algorithms/quantum_primitives/gqsp/gqsp.qmod"
    )
    expected = fq.run(reference_circuit()).statevector()
    for program in (synth.synthesize(original), build_program()):
        result = fq.run(program)
        torch.testing.assert_close(result.statevector(), expected, atol=1e-12, rtol=0)
        assert program.report["workspace_qubits"] == 0
        resources = program.report["resources"]
        assert isinstance(resources, ResourceEstimate)
        assert resources.n_qubits == 8
        marginal = expected.reshape(128, 2).abs().square().sum(dim=0)
        distribution = program.probabilities(result, registers=("ind",))
        for bit in range(2):
            assert distribution.get((bit,), 0) == pytest.approx(
                float(marginal[bit]), abs=1e-12, rel=0
            )


@pytest.mark.parametrize("negative_power", [0, 3, 7])
@pytest.mark.parametrize("initial_indicator", [0, 1])
@pytest.mark.parametrize("variant", [0, 1])
def test_full_complex_blocks_with_nonzero_phase_models(
    negative_power: int, initial_indicator: int, variant: int
) -> None:
    phases = tuple(
        (
            theta + 0.11 * (variant + 1),
            phi - 0.17 * (variant + 1),
            lam + 0.23 * (variant + 1),
        )
        for theta, phi, lam in PHASES
    )
    program = build_program(negative_power, initial_indicator, phases)
    expected = fq.run(
        reference_circuit(negative_power, initial_indicator, phases)
    ).statevector()
    torch.testing.assert_close(
        fq.run(program).statevector(), expected, atol=1e-12, rtol=0
    )
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize(
    "power,marker", [(-1, 0), (8, 0), (True, 0), (3, -1), (3, 2), (3, True)]
)
def test_invalid_bounds_are_refused(power: int, marker: int) -> None:
    with pytest.raises(ValueError):
        build_program(power, marker)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_phase_models_are_refused(value: float) -> None:
    phases = ((value, PHASES[0][1], PHASES[0][2]), *PHASES[1:])
    with pytest.raises((ValueError, synth.SynthesisError)):
        build_program(phases=phases)


@pytest.mark.parametrize("budget", ["operations", "width", "depth", "memory"])
def test_actual_program_budgets_are_enforced(budget: str) -> None:
    bindings: dict[str, float | list[float]] = {
        "negative_power": 3,
        "initial_indicator": 0,
        "theta": [r[0] for r in PHASES],
        "phi": [r[1] for r in PHASES],
        "lam": [r[2] for r in PHASES],
    }
    with pytest.raises((HybridVerificationError, synth.SynthesisError, PlanningError)):
        if budget == "operations":
            synth.synthesize(main, classical_inputs=bindings, max_operations=1)
        elif budget == "width":
            synth.synthesize(main, classical_inputs=bindings, max_qubits=7)
        elif budget == "depth":
            synth.synthesize(main, classical_inputs=bindings, max_depth=1)
        else:
            synth.synthesize(main, classical_inputs=bindings, memory_limit_bytes=16)

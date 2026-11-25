"""Native discrete log preserves raw modular powers and exact fractional labels."""

from fractions import Fraction
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.number_theory_and_cryptography.discrete_log.comparisons.circuit import (
    analytic_state,
    reference_circuit,
)
from examples.synthesis.algorithms.number_theory_and_cryptography.discrete_log.main import (
    build_program,
    main,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]


def test_untouched_original_native_and_independent_state_and_resources() -> None:
    source = (
        ROOT
        / "tests/fixtures/classiq_qmod/corpus/original/algorithms/number_theory_and_cryptography/discrete_log/discrete_log.qmod"
    )
    original = synth.synthesize(source)
    native = build_program()
    reference = reference_circuit()
    compare_states(original, reference, ("alpha", "beta", "func_res"))
    compare_states(native, reference, ("alpha", "beta", "func_res"))
    assert native.report["workspace_qubits"] == 2
    assert tuple(map(len, native.registers.values())) == (5, 5, 4)
    costs = native.report["resources"]
    assert isinstance(costs, ResourceEstimate)
    assert (costs.n_qubits, costs.n_operations, costs.depth) == (16, 1068, 872)


@pytest.mark.parametrize("element,generator", [(3, 7), (2, 6), (5, 11)])
def test_nonzero_parameters_match_circuit_and_two_axis_analytic_fft(
    element: int, generator: int
) -> None:
    program = build_program(element, generator)
    circuit = reference_circuit(element, generator)
    expected = analytic_state(element, generator)
    torch.testing.assert_close(
        fq.run(circuit).statevector().flatten(), expected, atol=1e-12, rtol=0
    )
    compare_states(program, circuit, ("alpha", "beta", "func_res"))
    result = fq.run(program)
    raw = dict(program.probabilities(result, registers=("alpha", "beta")))
    numeric = dict(program.numeric_probabilities(result, registers=("alpha", "beta")))
    assert numeric == {
        (Fraction(a, 32), Fraction(b, 32)): probability
        for (a, b), probability in raw.items()
    }
    expected_marginal = expected.reshape(32, 32, 16).abs().square().sum(dim=-1)
    for alpha in range(32):
        for beta in range(32):
            assert raw.get((alpha, beta), 0.0) == pytest.approx(
                float(expected_marginal[alpha, beta]), abs=1e-12
            )


@pytest.mark.parametrize("element,generator", [(13, 7), (3, 13), (0, 7), (3, -1)])
def test_noninvertible_or_nonpositive_bases_refuse(
    element: int, generator: int
) -> None:
    with pytest.raises(HybridVerificationError):
        build_program(element, generator)


@pytest.mark.parametrize(
    "constraint", ["workspace", "width", "depth", "operations", "memory"]
)
def test_actual_workspace_and_resource_budgets_refuse(constraint: str) -> None:
    bindings = {"element": 3, "generator": 7}
    with pytest.raises((synth.SynthesisError, HybridVerificationError, PlanningError)):
        if constraint == "workspace":
            synth.synthesize(main, classical_inputs=bindings, max_workspace=1)
        elif constraint == "width":
            synth.synthesize(main, classical_inputs=bindings, max_qubits=13)
        elif constraint == "depth":
            synth.synthesize(main, classical_inputs=bindings, max_depth=1)
        elif constraint == "operations":
            synth.synthesize(main, classical_inputs=bindings, max_operations=10)
        else:
            synth.synthesize(main, classical_inputs=bindings, memory_limit_bytes=16)

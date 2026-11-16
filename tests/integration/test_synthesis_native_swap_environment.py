"""The native swap test preserves a dirty environment and the overlap marginal."""

from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.quantum_primitives.swap_test.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.algorithms.quantum_primitives.swap_test.main import (
    RAW_FIRST,
    RAW_SECOND,
    build_program,
    main,
    normalize,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.ecosystem import InteropConversionError
from flagquantum.errors import PlanningError

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
CASE = "algorithms/quantum_primitives/swap_test/swap_test.qmod"


def check(first: tuple[float, ...], second: tuple[float, ...]) -> None:
    program = build_program(first, second)
    result = fq.run(program)
    state = result.statevector().flatten()
    expected = fq.run(reference_circuit(first, second)).statevector().flatten()
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)
    a = torch.tensor(normalize(first), dtype=torch.complex128)
    b = torch.tensor(normalize(second), dtype=torch.complex128)
    ab, ba = torch.outer(a, b), torch.outer(b, a)
    analytic = torch.stack(((ab + ba) / 2, (ab - ba) / 2))
    torch.testing.assert_close(state, analytic.flatten(), atol=1e-12, rtol=0)
    overlap = float(torch.vdot(a, b).abs().square())
    actual = dict(program.probabilities(result, registers=("test",)))
    assert actual.get((0,), 0) == pytest.approx((1 + overlap) / 2, abs=1e-12)
    assert actual.get((1,), 0) == pytest.approx((1 - overlap) / 2, abs=1e-12)
    assert program.report["workspace_qubits"] == 0
    assert tuple(program.registers) == ("test",)
    assert float(state.reshape(2, 64)[:, 1:].abs().square().sum()) > 0
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_qubits == 7


def test_raw_original_refusal_and_approved_full_environment() -> None:
    with pytest.raises(InteropConversionError, match="normalized"):
        synth.synthesize(ROOT / "tests/fixtures/classiq_qmod/corpus/original" / CASE)
    approved = synth.synthesize(
        ROOT / "tests/fixtures/classiq_qmod/corpus/approved" / CASE
    )
    native = build_program()
    torch.testing.assert_close(
        fq.run(approved).statevector(), fq.run(native).statevector(), atol=1e-12, rtol=0
    )
    for program in (approved, native):
        resources = program.report["resources"]
        assert isinstance(resources, ResourceEstimate)
        assert resources.n_qubits == 7
    check(RAW_FIRST, RAW_SECOND)


@pytest.mark.parametrize(
    "first,second",
    [
        ((0.6, -0.8, 0, 0, 0, 0, 0, 0), (0, 0, 0.8, -0.6, 0, 0, 0, 0)),
        (
            (-0.3, 0.4, 0.1, -0.2, 0.5, 0.7, -0.6, 0.8),
            (0.1, -0.8, 0.6, 0.2, -0.9, 0.4, 0.3, -0.7),
        ),
        (RAW_FIRST, RAW_FIRST),
    ],
)
def test_nonzero_parameters_bit_order_phase_and_overlap(
    first: tuple[float, ...], second: tuple[float, ...]
) -> None:
    check(first, second)


@pytest.mark.parametrize("budget", ["width", "operations", "depth", "memory"])
def test_actual_budgets_and_raw_non_normalized_input_refuse(budget: str) -> None:
    bindings = {
        "first": list(normalize(RAW_FIRST)),
        "second": list(normalize(RAW_SECOND)),
    }
    with pytest.raises((HybridVerificationError, PlanningError, synth.SynthesisError)):
        if budget == "width":
            synth.synthesize(main, classical_inputs=bindings, max_qubits=6)
        elif budget == "operations":
            synth.synthesize(main, classical_inputs=bindings, max_operations=1)
        elif budget == "depth":
            synth.synthesize(main, classical_inputs=bindings, max_depth=1)
        else:
            synth.synthesize(main, classical_inputs=bindings, memory_limit_bytes=16)


def test_unmodified_rounded_vectors_are_not_silently_normalized() -> None:
    with pytest.raises(HybridVerificationError, match="normalized"):
        synth.synthesize(
            main,
            classical_inputs={"first": list(RAW_FIRST), "second": list(RAW_SECOND)},
        )

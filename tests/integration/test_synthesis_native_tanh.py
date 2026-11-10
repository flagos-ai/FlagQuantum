"""Native piecewise arithmetic retains every original segment and raw output bit."""

from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.tutorials.advanced_tutorials.linear_approximation_of_tanh.tanh import (
    main as tanh,
)
from examples.synthesis.tutorials.advanced_tutorials.linear_approximation_of_tanh.tanh.comparisons.circuit import (
    reference_circuit,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.compiler._hybrid.verifier import HybridVerificationError
from flagquantum.errors import PlanningError

pytestmark = pytest.mark.integration
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/corpus/original/tutorials/advanced_tutorials/linear_approximation_of_tanh/tanh.qmod"
)


def check(program: synth.Program, expected: torch.Tensor) -> None:
    actual = fq.run(program).statevector().reshape(256, -1)
    torch.testing.assert_close(actual[:, 0], expected.flatten(), atol=1e-12, rtol=0)
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22


def test_original_native_and_independent_full_state() -> None:
    expected = fq.run(reference_circuit()).statevector()
    program = tanh.build_program()
    check(program, expected)
    check(synth.synthesize(SOURCE, max_workspace=2, max_qubits=10), expected)
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert (resources.n_qubits, resources.n_operations, resources.depth) == (10, 60, 48)
    assert program.report["workspace_qubits"] == 2


@pytest.mark.parametrize("initial", [0, 1, 5, 10, 15])
@pytest.mark.parametrize("repeats", [1, 2])
def test_nonzero_target_and_inverse(initial: int, repeats: int) -> None:
    check(
        tanh.build_program(initial, repeats),
        fq.run(reference_circuit(initial, repeats)).statevector(),
    )


@pytest.mark.parametrize(
    "slopes,intercepts",
    [
        ((0.5, 0.75, -0.25, 0.375), (0.03125, -0.0625, 0.1875, 0.25)),
        ((-0.125, 0.25, 0.625, -0.5), (-0.03125, 0.0625, -0.125, 0.875)),
    ],
)
def test_different_nonzero_piecewise_models(
    slopes: tuple[float, ...], intercepts: tuple[float, ...]
) -> None:
    check(
        tanh.build_program(5, 1, slopes, intercepts),
        fq.run(reference_circuit(5, 1, slopes, intercepts)).statevector(),
    )


def test_invalid_precision_refuses() -> None:
    with pytest.raises(HybridVerificationError, match="binary rationals"):
        tanh.build_program(slopes=(1.0, 0.9, 0.8125, 0.625))


@pytest.mark.parametrize(
    "kind", ["qubits", "workspace", "operations", "depth", "memory"]
)
def test_actual_resource_budgets_refuse(kind: str) -> None:
    classical: dict[str, float | list[float]] = {
        "slopes": list(tanh.SLOPES),
        "intercepts": list(tanh.INTERCEPTS),
        "initial": 0,
        "repeats": 1,
    }
    with pytest.raises((synth.SynthesisError, PlanningError)):
        if kind == "qubits":
            synth.synthesize(tanh.main, classical_inputs=classical, max_qubits=8)
        elif kind == "workspace":
            synth.synthesize(tanh.main, classical_inputs=classical, max_workspace=0)
        elif kind == "operations":
            synth.synthesize(tanh.main, classical_inputs=classical, max_operations=1)
        elif kind == "depth":
            synth.synthesize(tanh.main, classical_inputs=classical, max_depth=1)
        else:
            synth.synthesize(
                tanh.main, classical_inputs=classical, memory_limit_bytes=16
            )

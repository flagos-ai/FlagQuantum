"""Closed source callbacks become explicit native algorithm helper composition."""

from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.foundational.simon import main as simon
from examples.synthesis.algorithms.foundational.simon.comparisons import (
    circuit as simon_ref,
)
from examples.synthesis.algorithms.search_and_optimization.qaoa.qaoa_max_cut import (
    main as maxcut,
)
from examples.synthesis.algorithms.search_and_optimization.qaoa.qaoa_max_cut.comparisons import (
    circuit as cut_ref,
)
from examples.synthesis.compare_states import compare_states
from examples.synthesis.functions.state_preparation.prepare_exponential_state import (
    main as exponential,
)
from examples.synthesis.functions.state_preparation.prepare_exponential_state.comparisons import (
    circuit as exp_ref,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "tests/fixtures/classiq_qmod/corpus/original"
SIMON = CORPUS / "algorithms/foundational/simon/simon.qmod"
MAXCUT = CORPUS / "algorithms/search_and_optimization/QAOA/qaoa_max_cut.qmod"


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_simon_original_and_native_complete_joint_state(
    objective: Literal["width", "depth"],
) -> None:
    native = synth.synthesize(simon.main, objective=objective, max_workspace=0)
    source = synth.synthesize(SIMON, objective=objective)
    reference = simon_ref.reference_circuit()
    compare_states(native, reference, ("x", "res"))
    compare_states(source, reference, ("x", "res"))
    resources = native.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_qubits == 10
    assert native.report["workspace_qubits"] == 0


@pytest.mark.parametrize("objective", ["width", "depth"])
@pytest.mark.parametrize("binding", [0, 1])
def test_maxcut_original_and_native_complete_layers(
    objective: Literal["width", "depth"], binding: int
) -> None:
    params = cut_ref.BINDINGS[binding]
    native = synth.synthesize(
        maxcut.main,
        classical_inputs={"params": params},
        objective=objective,
        max_workspace=0,
    )
    source = synth.synthesize(
        MAXCUT, classical_inputs={"params": params}, objective=objective
    )
    reference = cut_ref.reference_circuit(params)
    compare_states(native, reference, ("v",))
    compare_states(source, reference, ("v",))
    resources = native.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_qubits == 5
    assert native.report["workspace_qubits"] == 0


def test_simon_all_inputs_and_two_to_one_secret() -> None:
    values = [simon_ref.value(x) for x in range(32)]
    assert len(set(values)) == 16
    for x in range(32):
        assert values[x] == min(x, x ^ 6) == values[x ^ 6]
        assert values.count(values[x]) == 2


def test_all_maxcut_assignments_preserve_six_edge_normalization() -> None:
    for basis in range(32):
        bits = tuple((basis >> bit) & 1 for bit in range(5))
        expected = (
            (bits[0] ^ bits[1])
            + (bits[0] ^ bits[2])
            + (bits[1] ^ bits[2])
            + (bits[1] ^ bits[3])
            + (bits[2] ^ bits[4])
            + (bits[3] ^ bits[4])
        )
        assert cut_ref.cost(bits) == expected / 6


def test_maxcut_bindings_and_zero_angle_uniform_state() -> None:
    compare_states(
        maxcut.build_program((0.0,) * 8), cut_ref.reference_circuit((0.0,) * 8), ("v",)
    )
    with pytest.raises(ValueError, match="parameters"):
        maxcut.build_program((0.1,))
    with pytest.raises(ValueError, match="finite"):
        maxcut.build_program((float("nan"),) * 8)


@pytest.mark.parametrize("algorithm", ["simon", "maxcut"])
@pytest.mark.parametrize(
    "budget", ["max_qubits", "max_operations", "max_depth", "memory_limit_bytes"]
)
def test_complete_native_algorithm_budget_refusals(algorithm: str, budget: str) -> None:
    width = 10 if algorithm == "simon" else 5
    entry = simon.main if algorithm == "simon" else maxcut.main
    bindings = None if algorithm == "simon" else {"params": cut_ref.BINDINGS[0]}
    with pytest.raises(ValueError):
        synth.synthesize(
            entry,
            classical_inputs=bindings,
            max_qubits=width - 1 if budget == "max_qubits" else None,
            max_operations=1 if budget == "max_operations" else None,
            max_depth=1 if budget == "max_depth" else None,
            memory_limit_bytes=1 if budget == "memory_limit_bytes" else None,
        )


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_original_fresh_exponential_state(objective: Literal["width", "depth"]) -> None:
    native = synth.synthesize(
        exponential.main,
        classical_inputs={"probabilities": exponential.PROBABILITIES},
        objective=objective,
        max_workspace=0,
    )
    source = synth.synthesize(
        CORPUS
        / "functions/qmod_library_reference/classiq_open_library/special_state_preparations/prepare_exponential_state.qmod",
        objective=objective,
    )
    circuit = exp_ref.reference_circuit()
    compare_states(native, circuit, ("x",))
    compare_states(source, circuit, ("x",))
    resources = native.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    assert resources.n_qubits == 5
    assert native.report["workspace_qubits"] == 0


def test_all_exponential_probabilities_and_absolute_amplitudes() -> None:
    program = exponential.build_program()
    state = fq.run(program).statevector().flatten()
    expected = torch.softmax(-0.1 * torch.arange(32, dtype=torch.float64), dim=0)
    torch.testing.assert_close(
        state, expected.sqrt().to(torch.complex128), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(state.abs().square(), expected, atol=1e-12, rtol=0)
    assert sum(exponential.PROBABILITIES) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "budget", ["max_qubits", "max_operations", "max_depth", "memory_limit_bytes"]
)
def test_fresh_exponential_program_budget_refusals(budget: str) -> None:
    with pytest.raises(ValueError):
        synth.synthesize(
            exponential.main,
            classical_inputs={"probabilities": exponential.PROBABILITIES},
            max_qubits=4 if budget == "max_qubits" else None,
            max_operations=1 if budget == "max_operations" else None,
            max_depth=1 if budget == "max_depth" else None,
            memory_limit_bytes=1 if budget == "memory_limit_bytes" else None,
        )

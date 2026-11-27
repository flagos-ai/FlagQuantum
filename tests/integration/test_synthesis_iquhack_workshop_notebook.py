"""Verify the workshop's complete native arithmetic and state preparation."""

import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.community.hackathons.iquhack_2025.workshop.overview import (
    workflow,
)
from flagquantum.errors import PlanningError

pytestmark = pytest.mark.integration


def test_unsigned_and_signed_full_complex_states() -> None:
    arithmetic = workflow.arithmetic()
    assert (arithmetic.qubits, arithmetic.operations) == (14, 69)
    assert arithmetic.state_error < 3e-12 and arithmetic.workspace_probability == 0
    values, signed = workflow.fixed_point()
    assert values == (5, -7.5, -2.5)
    assert signed.qubits == 14 and signed.state_error < 3e-12


def test_seeded_random_distribution_and_complete_qft() -> None:
    pytest.importorskip("numpy")
    probabilities = workflow.random_probabilities()
    assert len(probabilities) == 32 and math.isclose(math.fsum(probabilities), 1)
    prepared, distance = workflow.prepare_random(probabilities)
    assert prepared.qubits == 5 and prepared.state_error < 3e-12
    assert distance <= 0.01
    checked, output, sample = workflow.exponential_qft()
    assert len(output) == 32 and math.isclose(math.fsum(output), 1)
    assert checked.qubits == 5 and checked.state_error < 3e-12
    assert 0 <= sample < 32


def test_fractional_square_and_joint_counts() -> None:
    checked, sample = workflow.amplitude_assignment()
    assert checked.qubits == 6 and checked.state_error < 3e-12
    assert sum(count for _, count in sample.joint) == 2048
    assert sum(count for _, count in sample.marginal) == 2048
    marginal = dict(sample.marginal)
    assert len(marginal) == 32
    for word in range(32):
        x = word / 32
        assert marginal[x] == sum(
            count for (value, _), count in sample.joint if value == x
        )
    assert math.isclose(sample.expected_indicator_probability, 0.18470048904418945)


def test_all_original_workflow_stages_unexecuted() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/community/hackathons/iquhack_2025/workshop/overview/walkthrough.ipynb"
    )
    cells = json.loads(path.read_text())["cells"]
    code = [cell for cell in cells if cell["cell_type"] == "code"]
    assert len(code) == 16
    assert all(
        cell["execution_count"] is None and cell["outputs"] == [] for cell in code
    )


def test_signed_native_execution_preserves_compact_default() -> None:
    from examples.synthesis.community.hackathons.iquhack_2025.workshop.overview.quantum import (
        signed_addition,
    )
    from flagquantum import synth

    program = synth.synthesize(
        signed_addition,
        max_qubits=20,
        max_workspace=6,
        max_operations=1000,
        memory_limit_bytes=1024**3,
    )
    resources = program.report["resources"]
    assert resources.n_qubits == 20
    assert resources.n_operations == 47
    assert resources.depth == 23
    assert program.report["workspace_qubits"] == 6
    result = fq.run(program)
    assert dict(
        program.numeric_probabilities(result, registers=("x", "y", "result"))
    ) == {(5, -7.5, -2.5): 1.0}
    state = result.statevector().flatten()
    expected = torch.zeros_like(state)
    basis = sum(
        ((word >> shift) & 1) << (resources.n_qubits - 1 - wire)
        for name, word in (("x", 5), ("y", 17), ("result", 59))
        for shift, wire in enumerate(reversed(program.registers[name]))
    )
    expected[basis] = 1
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)
    # Unconstrained emission keeps its fewer-instruction ripple implementation.
    for limits in ({"max_qubits": 20, "max_workspace": 6, "max_operations": 46},):
        with pytest.raises(synth.SynthesisError):
            synth.synthesize(
                signed_addition,
                **{"memory_limit_bytes": 1024**3, **limits},
            )
    with pytest.raises(PlanningError, match="execution memory budget exceeded"):
        synth.synthesize(
            signed_addition,
            max_qubits=20,
            max_workspace=6,
            memory_limit_bytes=1024**2,
        )


@pytest.mark.parametrize("qubits,workspace", [(14, 0), (19, 6), (20, 5)])
def test_signed_native_execution_at_original_circuit_width(
    qubits: int, workspace: int
) -> None:
    from examples.synthesis.community.hackathons.iquhack_2025.workshop.overview.quantum import (
        signed_addition,
    )
    from flagquantum import synth

    program = synth.synthesize(
        signed_addition,
        max_qubits=qubits,
        max_workspace=workspace,
        max_operations=89,
        memory_limit_bytes=1024**2,
    )
    resources = program.report["resources"]
    assert (resources.n_qubits, resources.n_operations, resources.depth) == (14, 89, 51)
    assert program.report["workspace_qubits"] == 0
    result = fq.run(program)
    distribution = dict(
        program.numeric_probabilities(result, registers=("x", "y", "result"))
    )
    assert set(distribution) == {(5, -7.5, -2.5)}
    assert distribution[(5, -7.5, -2.5)] == pytest.approx(1, abs=1e-12)
    state = result.statevector().flatten()
    expected = torch.zeros_like(state)
    basis = sum(
        ((word >> shift) & 1) << (13 - wire)
        for name, word in (("x", 5), ("y", 17), ("result", 59))
        for shift, wire in enumerate(reversed(program.registers[name]))
    )
    expected[basis] = 1
    torch.testing.assert_close(state, expected, atol=1e-12, rtol=0)


@pytest.mark.parametrize(
    "limits",
    [
        {"max_qubits": 13, "max_workspace": 0},
        {"max_operations": 88},
        {"basis": ("h", "cx", "phase", "swap", "x")},
        {"topology": "line"},
    ],
)
def test_tight_signed_request_keeps_budgets_and_phase_safe_target(
    limits: dict[str, int | str | tuple[str, ...]],
) -> None:
    from examples.synthesis.community.hackathons.iquhack_2025.workshop.overview.quantum import (
        signed_addition,
    )
    from flagquantum import synth

    with pytest.raises(synth.SynthesisError):
        synth.synthesize(
            signed_addition,
            **{"max_qubits": 14, "max_workspace": 0, **limits},
            memory_limit_bytes=1024**2,
        )


def test_tight_signed_within_phase_restores_all_local_work() -> None:
    from flagquantum import synth
    from tests.team.compiler.test_construction_python_program import program_definition

    entry = program_definition(
        "from flagquantum.synth import *\n"
        "@qfunc\n"
        "def main(x:Output[QNum[3]], y:Output[QNum[5,True,1]]):\n"
        "    allocate(x)\n    allocate(y)\n"
        "    hadamard_transform(x)\n    hadamard_transform(y)\n"
        "    with within():\n        score:QNum[6,True,1] = x+y\n"
        "    with apply():\n        phase(score,0.37)\n"
    )
    program = synth.synthesize(
        entry, max_qubits=14, max_workspace=6, memory_limit_bytes=1024**2
    )
    state = fq.run(program).statevector().reshape(256, 64)
    indices = torch.arange(256)
    x, raw_y = indices // 32, indices % 32
    y = torch.where(raw_y >= 16, raw_y - 32, raw_y).to(torch.float64) / 2
    expected = torch.exp(0.37j * (x + y)) / 16
    torch.testing.assert_close(state[:, 0], expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        state[:, 1:], torch.zeros_like(state[:, 1:]), atol=1e-12, rtol=0
    )

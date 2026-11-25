"""Check states, cleanup, finite shots, and symbol reuse from the complete tutorial."""

import json
from fractions import Fraction
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.tutorials.basics.quantum_programming.qmod_tutorial_part2.overview import (
    quantum,
    reference,
    workflow,
)
from flagquantum import synth

pytestmark = pytest.mark.integration


def test_all_native_teaching_states_and_workspace() -> None:
    for entry, expected, names, width, workspace, angles in (
        ("product_phase", reference.phase_state(), ("x", "y"), 4, 0, True),
        (
            "solution_oracle",
            reference.phase_state(oracle=True),
            ("a", "b"),
            4,
            0,
            False,
        ),
        (
            "local_comparison",
            reference.comparison_state(),
            ("x", "res"),
            7,
            4,
            False,
        ),
        (
            "fourier_add",
            reference.addition_state(controlled=False),
            ("x", "y"),
            7,
            0,
            True,
        ),
        (
            "controlled_fourier_add",
            reference.addition_state(controlled=True),
            ("ctrl", "x", "y"),
            8,
            0,
            False,
        ),
        ("pauli_evolution", reference.trotter_state(), ("qarr",), 4, 0, False),
    ):
        program, result, checked = workflow.run_native(
            entry,
            expected,
            names,
            max_qubits=width,
            max_workspace=workspace,
            angles=angles,
        )
        assert checked.error < 1e-12
        assert checked.workspace_probability == 0
        counts = workflow.read_counts(program, result, names, seed=12)
        expected_words = dict(program.probabilities(result, registers=names))
        assert sum(counts.values()) == 1_000
        assert set(counts) == set(expected_words)

    with pytest.raises(synth.SynthesisError, match="max_qubits"):
        workflow.run_native(
            "controlled_fourier_add",
            reference.addition_state(controlled=True),
            ("ctrl", "x", "y"),
            max_qubits=7,
        )


def test_observables_phase_reading_and_general_callback() -> None:
    value, error = workflow.observe_plus()
    assert value == pytest.approx(2)
    assert error < 1e-12
    samples, error = workflow.read_callback()
    assert set(samples) == {(word,) for word in range(8)}
    assert sum(samples.values()) == 1_000
    assert error < 1e-12
    circuit = quantum.apply_to_all(fq.Circuit(3, dtype=torch.complex128), fq.Circuit.x)
    assert fq.run(circuit).statevector().abs().argmax().item() == 7
    distribution, checked = workflow.read_qpe()
    assert distribution == {(Fraction(1, 4),): pytest.approx(1.0)}
    assert checked.error < 1e-12


def test_one_template_survives_three_distinct_bindings() -> None:
    template = quantum.parameterized_ansatz()
    identity = template.to_ir().content_hash
    rows = workflow.read_bindings(template)
    assert [row[0] for row in rows] == list(workflow.BINDINGS)
    assert all(
        sum(counts.values()) == 1_000 and error < 1e-12 for _, counts, error in rows
    )
    assert rows[0][1] != rows[1][1]
    value, error = workflow.observe_binding(template)
    assert value == pytest.approx(0.8623724356957945)
    assert error < 1e-12
    assert template.parameter_names == ("alpha", "beta")
    assert template.to_ir().content_hash == identity


def test_notebook_preserves_all_source_stages_without_saved_outputs() -> None:
    overview = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/tutorials/basics/quantum_programming/qmod_tutorial_part2/overview"
    )
    cells = json.loads((overview / "walkthrough.ipynb").read_text())["cells"]
    proof = json.loads((overview / "provenance.json").read_text())
    stages = [cell for cell in cells if cell["cell_type"] == "code"]
    assert len(stages) == len(proof["reviewed_code_cells"]) == 23
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in stages
    )
    assert [item["index"] for item in proof["reviewed_code_cells"]] == [
        2,
        5,
        8,
        12,
        16,
        19,
        23,
        25,
        29,
        32,
        36,
        39,
        44,
        46,
        48,
        51,
        53,
        56,
        58,
        61,
        63,
        66,
        68,
    ]

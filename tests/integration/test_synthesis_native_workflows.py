"""Native workflows cover notebook steps beyond source-file compilation."""

import ast
import runpy
from pathlib import Path
from typing import Literal

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.foundational.bernstein_vazirani.comparisons.circuit import (
    circuit_reference,
)
from examples.synthesis.algorithms.foundational.bernstein_vazirani.main import (
    build_program,
)
from examples.synthesis.algorithms.foundational.bernstein_vazirani.main import (
    run_example as run_bernstein_vazirani,
)
from examples.synthesis.algorithms.foundational.deutsch_jozsa.main import classify
from examples.synthesis.algorithms.foundational.deutsch_jozsa.main import (
    run_example as run_deutsch_jozsa,
)
from examples.synthesis.community.basic_examples.superposition.main import run_example
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.basics.quantum_programming.classiq_overview_tutorial.main import (
    run_example as run_overview,
)
from examples.synthesis.tutorials.basics.quantum_programming.synthesis_tutorial.comparisons.circuit import (
    reference_circuit as arithmetic_reference,
)
from examples.synthesis.tutorials.basics.quantum_programming.synthesis_tutorial.main import (
    build_program as build_arithmetic,
)
from examples.synthesis.tutorials.basics.quantum_programming.synthesis_tutorial.main import (
    run_example as run_synthesis,
)
from flagquantum.compiler import ResourceEstimate

pytestmark = pytest.mark.integration


def test_superposition_notebook_build_draw_and_named_readout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_example()
    diagram, probabilities = capsys.readouterr().out.rstrip().rsplit("\n", 1)
    assert diagram.strip()
    assert ast.literal_eval(probabilities) == pytest.approx(
        {(word,): 1 / 8 for word in range(8)}, abs=1e-12
    )


@pytest.mark.parametrize("secret", [13, 22])
def test_hidden_string_build_draw_sample_and_recover(
    secret: int, capsys: pytest.CaptureFixture[str]
) -> None:
    run_bernstein_vazirani(secret)
    output = capsys.readouterr().out
    assert f"Secret integer: {secret}" in output
    assert ast.literal_eval(output.split("Counts: ", 1)[1]) == {(secret,): 1000}


@pytest.mark.parametrize("secret", range(32))
def test_every_five_bit_secret_and_absolute_phase(secret: int) -> None:
    program = build_program(secret)
    result = fq.run(program)
    reference = fq.run(circuit_reference(secret)).statevector()
    torch.testing.assert_close(result.statevector(), reference, atol=1e-12, rtol=0)
    probabilities = dict(program.probabilities(result, registers=("x",)))
    assert probabilities == pytest.approx({(secret,): 1.0}, abs=1e-12)
    assert program.report["workspace_qubits"] == 0


@pytest.mark.parametrize("secret", [-1, 32, True, 1.5])
def test_hidden_string_rejects_invalid_secrets(secret: object) -> None:
    with pytest.raises(ValueError, match="five-bit integer"):
        build_program(secret)  # type: ignore[arg-type]


def test_deutsch_jozsa_both_predicates_draw_sample_and_decide(
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_deutsch_jozsa()
    output = capsys.readouterr().out
    assert "simple balanced" in output
    assert "complex balanced" in output
    counts = [
        ast.literal_eval(line.removeprefix("Counts: "))
        for line in output.splitlines()
        if line.startswith("Counts: ")
    ]
    assert counts == [{(8,): 1000}, {(4,): 1000}]


def test_deutsch_jozsa_classical_decision_checks_the_zero_mass() -> None:
    assert classify({(0,): 1.0}) == "constant"
    assert classify({(2,): 0.5, (5,): 0.5}) == "balanced"
    with pytest.raises(ValueError, match="promise"):
        classify({(0,): 0.5, (1,): 0.5})


@pytest.mark.parametrize(
    ("example", "expected"),
    [
        ("hadamard_transform", {(value,): 1 / 8 for value in range(8)}),
        ("state_preparation/prepare_bell_state", {(1,): 0.5, (2,): 0.5}),
        ("state_preparation/prepare_ghz_state", {(0,): 0.5, (31,): 0.5}),
    ],
)
def test_state_preparation_notebook_workflows(
    example: str,
    expected: dict[tuple[int, ...], float],
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = Path(__file__).resolve().parents[2] / "examples/synthesis/functions"
    runpy.run_path(str(path / example / "main.py"), run_name="__main__")
    assert ast.literal_eval(capsys.readouterr().out) == pytest.approx(
        expected, abs=1e-12
    )


def test_arithmetic_overview_draws_executes_and_samples_named_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_overview()
    output = capsys.readouterr().out
    diagram, readout = output.split("Probabilities: ", 1)
    assert diagram.strip()
    probabilities, counts = readout.split("\nCounts: ", 1)
    expected = {(x, x * x + 1): 1 / 8 for x in range(8)}
    assert ast.literal_eval(probabilities) == pytest.approx(expected, abs=1e-12)
    observed = ast.literal_eval(counts)
    assert observed.keys() == expected.keys()
    assert sum(observed.values()) == 1000
    assert all(80 < count < 170 for count in observed.values())


@pytest.mark.parametrize("objective", ["width", "depth"])
def test_both_synthesis_objectives_preserve_complete_arithmetic_state(
    objective: Literal["width", "depth"],
) -> None:
    program = build_arithmetic(objective)
    compare_states(program, arithmetic_reference(), ("x", "y"))
    result = fq.run(program)
    assert dict(program.probabilities(result, registers=("x", "y"))) == pytest.approx(
        {(x, x * x + 1): 1 / 8 for x in range(8)}, abs=1e-12
    )


def test_synthesis_workflow_draws_and_reports_both_actual_objectives(
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_synthesis()
    output = capsys.readouterr().out
    objectives: tuple[Literal["width", "depth"], ...] = ("width", "depth")
    for objective in objectives:
        report = next(
            line
            for line in output.splitlines()
            if line.startswith(objective + " resources: ")
        )
        measured = ast.literal_eval(report.split("resources: ", 1)[1])
        resources = build_arithmetic(objective).report["resources"]
        assert isinstance(resources, ResourceEstimate)
        assert measured == {"width": resources.n_qubits, "depth": resources.depth}
    readouts = [
        ast.literal_eval(line) for line in output.splitlines() if line.startswith("{(")
    ]
    assert len(readouts) == 2
    for probabilities in readouts:
        assert probabilities == pytest.approx(
            {(x, x * x + 1): 1 / 8 for x in range(8)}, abs=1e-12
        )

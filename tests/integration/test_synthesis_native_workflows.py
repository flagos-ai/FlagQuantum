"""Native workflows cover notebook steps beyond source-file compilation."""

import ast

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

"""Original tutorial teaching stages, actual observations and native ownership."""

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

import flagquantum as fq
from examples.synthesis.tutorials.basics.quantum_programming.qmod_tutorial_part1.overview.workflow import (
    PROBABILITIES,
    build_program,
    initialization_refusal,
    sample,
    signed_distribution,
    verify,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "name",
    (
        "bell",
        "ghz",
        "signed_ghz",
        "distributions",
        "corrected_initialization",
        "rotations",
        "parameterized",
        "layer",
        "arithmetic",
        "accumulation",
        "conditional",
    ),
)
def test_complete_absolute_states_and_clean_workspace(name: str) -> None:
    row = verify(name)
    assert row.maximum_error < 1e-12
    assert row.dirty_workspace_probability < 1e-24


@pytest.mark.parametrize("alpha,beta", ((0.3, 0.2), (-0.17, 0.41)))
def test_nonzero_parameter_bindings_are_full_turns(alpha: float, beta: float) -> None:
    assert verify("parameterized", alpha=alpha, beta=beta).maximum_error < 1e-12


def test_signed_ghz_is_zero_and_minus_one() -> None:
    distribution = signed_distribution(build_program("signed_ghz"))
    assert dict(distribution) == pytest.approx(
        {(Fraction(0),): 0.5, (Fraction(-1),): 0.5}
    )


def test_original_probability_vector_and_ten_thousand_actual_draws() -> None:
    program = build_program("distributions")
    values = program.probabilities(fq.run(program), registers=("a", "b"))
    assert dict(values) == pytest.approx(
        {
            (a, b): pa * pb
            for a, pa in enumerate(PROBABILITIES)
            for b, pb in enumerate(PROBABILITIES)
        }
    )
    observations = sample(program, shots=10000)
    assert sum(row.count for row in observations) == 10000
    assert observations == sample(program, shots=10000)
    assert observations != sample(program, shots=10000, seed=20)
    assert all(row.probability == row.count / 10000 for row in observations)


def test_nonzero_accumulator_and_conditional_marking() -> None:
    for name, expected in (
        ("accumulation", {(y, 4 + y): 0.25 for y in range(4)}),
        ("conditional", {(x, int(x == 5)): 0.125 for x in range(8)}),
        (
            "arithmetic",
            {(x, y, x + y, x * y): 1 / 16 for x in range(4) for y in range(4)},
        ),
    ):
        program = build_program(name)
        assert dict(
            program.probabilities(fq.run(program), registers=tuple(program.registers))
        ) == pytest.approx(expected)
    assert "construction.register.uninitialized" in initialization_refusal()


@pytest.mark.parametrize("shots,seed", ((0, 0), (True, 0), (100, -1), (100, True)))
def test_invalid_sampling_controls_fail_closed(shots: int, seed: int) -> None:
    with pytest.raises(ValueError, match="shots|seed"):
        sample(build_program("bell"), shots=shots, seed=seed)


def test_all_original_code_cells_have_executable_native_stages() -> None:
    directory = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/tutorials/basics/quantum_programming/qmod_tutorial_part1/overview"
    )
    proof = json.loads((directory / "provenance.json").read_text())
    notebook = json.loads((directory / "walkthrough.ipynb").read_text())
    assert [row["index"] for row in proof["reviewed_code_cells"]] == [
        2,
        5,
        9,
        12,
        16,
        19,
        22,
        25,
        28,
        31,
        34,
        38,
        43,
        45,
        48,
        50,
        53,
        55,
        57,
        60,
        62,
        64,
    ]
    for row in proof["reviewed_code_cells"]:
        cell = notebook["cells"][row["native_code_cell"]]
        assert cell["cell_type"] == "code" and "".join(cell["source"]).strip()
    assert len(proof["original_sha256"]) == len(hashlib.sha256(b"").hexdigest())
    text = (directory / "workflow.py").read_text()
    assert "source_format" not in text and "classiq import" not in text

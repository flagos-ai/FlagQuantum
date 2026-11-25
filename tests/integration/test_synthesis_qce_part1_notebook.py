"""Complete QCE part I arithmetic, signed values, phases and source replay stages."""

import cmath
import json
import math
from fractions import Fraction
from pathlib import Path

import pytest

from examples.synthesis.tutorials.workshops.algo_design_qce_tutorial.algo_design_qce_tutorial_part_i.overview.workflow import (
    STAGES,
    build_program,
    distribution,
    numeric_attributes,
    phase_table,
    sample,
    verify,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("name", STAGES)
def test_all_native_stages_keep_absolute_states_and_zero_workspace(name: str) -> None:
    check = verify(name)
    assert check.maximum_state_error < 2e-11
    assert check.dirty_workspace_probability < 1e-22


def test_integer_and_fractional_output_formats_are_inferred() -> None:
    integer = build_program("integer_sum")
    fractional = build_program("fractional_sum")
    first, second = numeric_attributes(integer, "c"), numeric_attributes(
        fractional, "c"
    )
    assert (first.size, first.signed, first.fraction_digits) == (4, False, 0)
    assert (second.size, second.signed, second.fraction_digits) == (5, False, 2)
    assert dict(distribution(integer)) == pytest.approx(
        {
            (Fraction(a), Fraction(b), Fraction(3 * a + b)): 1 / 16
            for a in range(4)
            for b in range(4)
        }
    )
    assert dict(distribution(fractional)) == pytest.approx(
        {
            (Fraction(a, 2), Fraction(b, 4), Fraction(6 * a + b, 4)): 1 / 16
            for a in range(4)
            for b in range(4)
        }
    )


def test_signed_ghz_and_strict_fractional_boundary_are_decoded() -> None:
    assert dict(distribution(build_program("signed_ghz"))) == pytest.approx(
        {(Fraction(0),): 0.5, (Fraction(-1),): 0.5}
    )
    expected = {
        (
            Fraction(raw - 8 if raw >= 4 else raw, 4),
            Fraction(int((raw - 8 if raw >= 4 else raw) < 2)),
        ): 1
        / 8
        for raw in range(8)
    }
    assert dict(distribution(build_program("signed_condition"))) == pytest.approx(
        expected
    )
    assert expected[(Fraction(1, 2), Fraction(0))] == 1 / 8
    assert expected[(Fraction(-1), Fraction(1))] == 1 / 8


def test_all_sixteen_phase_entries_include_global_phase_and_word_order() -> None:
    rows = phase_table(build_program("product_phase"))
    assert len(rows) == 16
    for x, y, amplitude in rows:
        assert amplitude == pytest.approx(
            cmath.exp(1j * math.pi * x * y / 8) / 4, abs=1e-12
        )
    with pytest.raises(ValueError, match="two two-bit"):
        phase_table(build_program("bell"))


def test_fourier_square_preserves_modular_wraparound() -> None:
    values = distribution(build_program("fourier_square"))
    assert dict(values) == pytest.approx(
        {(Fraction(x), Fraction(x * x % 16)): 1 / 8 for x in range(8)}
    )
    assert values[(Fraction(7), Fraction(1))] == pytest.approx(1 / 8)


def test_grover_marks_original_six_bit_equation_without_injected_counts() -> None:
    program = build_program("grover")
    values = distribution(program)
    marked = sum(p for (a, b, c), p in values.items() if 3 * a + b + 2 * c == 9)
    count = sum(
        3 * a + b + 2 * c == 9 for a in range(4) for b in range(4) for c in range(4)
    )
    theta = math.asin(math.sqrt(count / 64))
    assert marked == pytest.approx(math.sin(5 * theta) ** 2, abs=1e-12)
    first = sample(program, shots=1000, seed=19)
    assert first == sample(program, shots=1000, seed=19)
    assert first != sample(program, shots=1000, seed=20)
    assert sum(row.count for row in first) == 1000
    assert all(row.frequency == row.count / 1000 for row in first)


@pytest.mark.parametrize(
    "shots,seed", ((0, 0), (True, 0), (100, -1), (100, True), (100, 2**63))
)
def test_sampling_limits_fail_before_execution(shots: int, seed: int) -> None:
    with pytest.raises(ValueError, match="shots|seed"):
        sample(build_program("bell"), shots=shots, seed=seed)


def test_every_original_cell_and_full_solution_replays_are_mapped() -> None:
    directory = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/tutorials/workshops/algo_design_qce_tutorial/algo_design_qce_tutorial_part_i/overview"
    )
    proof = json.loads((directory / "provenance.json").read_text())
    notebook = json.loads((directory / "walkthrough.ipynb").read_text())
    assert [row["index"] for row in proof["reviewed_code_cells"]] == [
        4,
        7,
        11,
        14,
        20,
        24,
        28,
        32,
        37,
        41,
        45,
        47,
        50,
        52,
        54,
        56,
        58,
        60,
    ]
    for row in proof["reviewed_code_cells"]:
        assert notebook["cells"][row["native_code_cell"]]["cell_type"] == "code"
    text = "\n".join("".join(c["source"]) for c in notebook["cells"])
    assert text.count('show_stage("') == 17
    assert "fractional_sum_solution = synth.synthesize(" in text
    assert "classiq import" not in text

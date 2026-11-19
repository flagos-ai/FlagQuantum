"""Source-sized HHL retains absolute states,finite precision and real postselection."""

import json
import math
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.tutorials.workshops.hhl_workshop.overview.ansatz import (
    build_program,
    load_b,
)
from examples.synthesis.tutorials.workshops.hhl_workshop.overview.physics import (
    AMPLITUDES,
    ANGLES,
    MATRIX,
    QPE_SIZE,
    UNITARY,
    B,
)
from examples.synthesis.tutorials.workshops.hhl_workshop.overview.workflow import (
    extract_solution,
    plot_solution,
    validate_problem,
    verify_program,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate

pytestmark = pytest.mark.integration


def test_original_matrix_vector_and_finite_precision_are_preserved() -> None:
    validate_problem(MATRIX, B)
    assert MATRIX.shape == (4, 4) and QPE_SIZE == 4
    torch.testing.assert_close(
        B,
        torch.tensor([1.0, 2.0, 4.0, 3.0], dtype=torch.float64) / math.sqrt(30),
        atol=1e-15,
        rtol=0,
    )
    eig = torch.linalg.eigvalsh(MATRIX)
    assert bool((eig > 0).all()) and bool((eig < 1).all())
    recomputed = torch.matrix_exp(2j * math.pi * MATRIX)
    torch.testing.assert_close(
        torch.tensor(UNITARY, dtype=torch.complex128), recomputed, atol=1e-15, rtol=0
    )


def test_preparation_full_complex_state_matches_original_normalized_vector() -> None:
    p = synth.synthesize(
        load_b, classical_inputs={"amplitudes": list(AMPLITUDES)}, max_workspace=0
    )
    row = verify_program(p, preparation_only=True)
    assert row.maximum_state_error < 1e-12 and row.dirty_workspace_probability < 1e-22
    state = fq.run(p).statevector()
    assert isinstance(state, torch.Tensor)
    torch.testing.assert_close(
        state.flatten(), B.to(torch.complex128), atol=1e-12, rtol=0
    )


def test_all_reciprocal_amplitudes_use_fractional_values_and_zero_word() -> None:
    assert len(ANGLES) == 16 and ANGLES[0] == 0
    for word in range(1, 16):
        assert math.sin(ANGLES[word] / 2) == pytest.approx(
            (1 / 16) / (word / 16), abs=1e-15
        )


def test_hhl_absolute_full_state_and_clean_compiler_work() -> None:
    p = build_program()
    row = verify_program(p)
    assert row.maximum_state_error < 2e-11 and row.dirty_workspace_probability < 1e-22
    assert tuple(p.registers) == ("res", "phase_", "indicator")
    assert tuple(len(w) for w in p.registers.values()) == (2, 4, 1)
    resource = p.report["resources"]
    assert isinstance(resource, ResourceEstimate)
    assert resource.n_qubits == 7


def test_actual_postselection_rounding_fidelity_and_retained_phase_leakage() -> None:
    p = build_program()
    solution = extract_solution(p)
    probabilities = p.probabilities(fq.run(p), registers=("res", "phase_", "indicator"))
    selected = sum(
        prob
        for (_, phase, indicator), prob in probabilities.items()
        if phase == 0 and indicator == 1
    )
    indicator = sum(
        prob for (_, _, indicator), prob in probabilities.items() if indicator == 1
    )
    assert solution.postselection_probability == pytest.approx(selected, abs=1e-14)
    assert solution.indicator_probability == pytest.approx(indicator, abs=1e-14)
    assert solution.retained_phase_leakage_probability == pytest.approx(
        indicator - selected, abs=1e-14
    )
    assert 0.09 < selected < 0.11 and solution.retained_phase_leakage_probability > 1e-5
    expected = torch.tensor(
        [1.43559, 2.56952, 3.26897, 2.48819], dtype=torch.complex128
    )
    torch.testing.assert_close(solution.rounded_quantum, expected, atol=1e-12, rtol=0)
    assert solution.fidelity == pytest.approx(0.9999806280936835, abs=1e-10)
    assert solution.fidelity < 1
    assert solution.magnitude_relative_distance_percent == 2.5
    assert 0.02 < solution.complex_relative_distance < 0.03


def test_final_plot_contains_actual_classical_and_quantum_components(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    p = plot_solution(extract_solution(build_program()), tmp_path)
    assert p.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") and p.stat().st_size > 10000


@pytest.mark.parametrize(
    "kind", ("shape", "asymmetric", "zero", "one", "nan", "vector")
)
def test_invalid_original_problem_admission_fails_closed(kind: str) -> None:
    a = MATRIX.clone()
    b = B.clone()
    if kind == "shape":
        a = a[:3]
    elif kind == "asymmetric":
        a[0, 1] += 0.1
    elif kind == "zero":
        a = torch.zeros_like(a)
    elif kind == "one":
        a = torch.eye(4, dtype=torch.float64)
    elif kind == "nan":
        a[0, 0] = math.nan
    else:
        b *= 2
    with pytest.raises(ValueError):
        validate_problem(a, b)


def test_preparation_cannot_be_misread_as_hhl_solution() -> None:
    p = synth.synthesize(load_b, classical_inputs={"amplitudes": list(AMPLITUDES)})
    with pytest.raises(ValueError, match="source res"):
        extract_solution(p)


def test_hhl_budget_refuses_less_than_original_width() -> None:
    from examples.synthesis.tutorials.workshops.hhl_workshop.overview.ansatz import main

    with pytest.raises(ValueError):
        synth.synthesize(
            main, classical_inputs={"amplitudes": list(AMPLITUDES)}, max_qubits=6
        )


def test_all_twenty_one_source_stages_and_magnitude_caveat_are_retained() -> None:
    directory = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/tutorials/workshops/hhl_workshop/overview"
    )
    proof = json.loads((directory / "provenance.json").read_text())
    n = json.loads((directory / "walkthrough.ipynb").read_text())
    assert [r["index"] for r in proof["reviewed_code_cells"]] == [
        5,
        6,
        10,
        12,
        14,
        16,
        19,
        22,
        24,
        26,
        28,
        30,
        31,
        33,
        34,
        35,
        37,
        39,
        40,
        42,
        43,
    ]
    for row in proof["reviewed_code_cells"]:
        assert n["cells"][row["native_code_cell"]]["cell_type"] == "code"
    text = "\n".join("".join(c["source"]) for c in n["cells"])
    assert "not a physical global-phase correction" in text
    assert "classiq import" not in text and "QmodLoader" not in text

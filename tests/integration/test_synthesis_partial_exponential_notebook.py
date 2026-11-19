"""Fresh partial exponential workflows retain exact-state and refusal contracts."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.finance.autocallable_options.partial_exponential_state_preparation import (
    main as retained,
)
from examples.synthesis.applications.finance.autocallable_options.partial_exponential_state_preparation.overview.model import (
    make_case,
    source_cases,
)
from examples.synthesis.applications.finance.autocallable_options.partial_exponential_state_preparation.overview.workflow import (
    prepare_distribution,
    prepare_interval,
)

np = pytest.importorskip("numpy")
pytestmark = pytest.mark.integration


@pytest.mark.parametrize("rate", (0.25, 0.5, 0.75))
@pytest.mark.parametrize("interval", ((15, 29, 0), (21, 31, -18)))
def test_complete_nonzero_cases_match_absolute_physical_states_and_clean_work(
    rate: float,
    interval: tuple[int, int, int],
) -> None:
    lower, upper, shift = interval
    case = make_case(rate, lower, upper, shift)
    prepared = prepare_interval(case)
    assert prepared.resources.n_qubits <= 8
    assert prepared.check.state_error < 1e-11 and prepared.check.named_error < 1e-11
    assert prepared.check.workspace_error < 1e-11
    assert prepared.check.private_attenuation_one_probability is not None
    assert prepared.check.private_attenuation_one_probability < 1e-22
    assert prepared.check.interval_leakage < 1e-22
    assert prepared.check.source_amplitude_error < 0.01
    expected = np.zeros(32)
    values = np.exp(rate * np.arange(lower + shift, upper + shift + 1))
    expected[lower + shift : upper + shift + 1] = values / values.sum()
    np.testing.assert_allclose(prepared.probabilities, expected, atol=2e-12, rtol=0)


@pytest.mark.parametrize("rate", (-0.25, -0.5, -0.75))
def test_negative_rate_lower_interval_and_positive_shift(rate: float) -> None:
    case = make_case(rate, 0, 10, 3)
    result = prepare_interval(case)
    assert result.check.state_error < 1e-11
    assert result.check.interval_leakage < 1e-22
    assert result.check.source_amplitude_error < 0.01


@pytest.mark.parametrize("rate", (-0.5, 0.5, 0.75))
def test_fresh_base_distribution_matches_independent_normalized_weights(
    rate: float,
) -> None:
    amplitudes = tuple(retained.exponential_amplitudes(-rate, 5))
    result = prepare_distribution(amplitudes)
    weights = np.exp(rate * np.arange(32))
    np.testing.assert_allclose(
        result.probabilities, weights / weights.sum(), atol=1e-14, rtol=0
    )
    assert result.check.state_error < 1e-11
    assert result.check.private_attenuation_one_probability is None


def test_full_interval_zero_round_case_retains_the_full_exponential_state() -> None:
    case = make_case(0.5, 0, 31)
    assert case.observed_overlap == 1 and case.iterations == 0 and case.beta == 0
    result = prepare_interval(case)
    assert result.check.state_error < 1e-11
    assert result.check.private_attenuation_one_probability is not None
    assert result.check.private_attenuation_one_probability < 1e-22


@pytest.mark.parametrize("lower,upper,shift", ((15, 29, 0), (21, 31, -18)))
def test_original_approximate_hints_are_refused_and_not_silently_accepted(
    lower: int,
    upper: int,
    shift: int,
) -> None:
    case = make_case(0.5, lower, upper, shift)
    assert abs(case.declared_overlap - case.observed_overlap) > 1e-12
    with pytest.raises(ValueError, match="supplied overlap disagrees"):
        make_case(0.5, lower, upper, shift, supplied_overlap=case.declared_overlap)
    accepted = make_case(
        0.5, lower, upper, shift, supplied_overlap=case.observed_overlap
    )
    assert accepted == case


def test_final_workflow_matches_existing_approved_native_counterpart() -> None:
    _, case = source_cases()
    actual = prepare_interval(case)
    retained_program = retained.build_program(-case.rate, case.observed_overlap)
    torch.testing.assert_close(
        fq.run(actual.program).statevector(),
        fq.run(retained_program).statevector(),
        atol=1e-11,
        rtol=0,
    )
    with pytest.raises(ValueError, match="success amplitude"):
        retained.build_program(success_amplitude=case.declared_overlap)


def test_original_small_interval_remains_classical_and_hint_difference_is_visible() -> (
    None
):
    small = make_case(0.5, 3, 13)
    _, shifted = source_cases()
    assert small.iterations > shifted.iterations == 1
    assert small.declared_overlap != small.observed_overlap
    assert len(small.amplitudes) == 32


@pytest.mark.parametrize("rate", (0, True, float("nan"), float("inf"), 1.1))
def test_invalid_rate_is_rejected_before_quantum_construction(rate: float) -> None:
    with pytest.raises(ValueError, match="bounded rate"):
        make_case(rate, 15, 29)


@pytest.mark.parametrize(
    "lower,upper,shift", ((-1, 4, 0), (4, 4, 0), (4, 32, 0), (True, 5, 0), (2, 5, 32))
)
def test_invalid_interval_and_shift_are_rejected(
    lower: int, upper: int, shift: int
) -> None:
    with pytest.raises(ValueError, match="five-bit interval"):
        make_case(0.5, lower, upper, shift)


def test_nonwrapping_plot_profile_rejects_before_execution() -> None:
    case = make_case(0.5, 15, 29, -20)
    with pytest.raises(ValueError, match="non-wrapping"):
        prepare_interval(case)


def test_all_seven_source_stages_are_mapped_and_notebook_is_unexecuted() -> None:
    root = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/applications/finance/autocallable_options/partial_exponential_state_preparation/overview"
    )
    proof = json.loads((root / "provenance.json").read_text())
    notebook = json.loads((root / "walkthrough.ipynb").read_text())
    assert tuple(row["index"] for row in proof["reviewed_code_cells"]) == (
        4,
        5,
        7,
        8,
        10,
        12,
        14,
    )
    assert all(len(row["sha256"]) == 64 for row in proof["reviewed_code_cells"])
    assert all(
        not c.get("outputs") for c in notebook["cells"] if c["cell_type"] == "code"
    )

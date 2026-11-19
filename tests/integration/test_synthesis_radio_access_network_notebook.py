"""Fresh original antenna data, original cost sign and all sixteen quantum bits."""

import json
import math
from pathlib import Path

import pytest
import torch

from examples.synthesis.applications.telecom.radio_access_network.radio_access_network_positioning_antennas.overview.model import (
    assignment,
    fresh_coefficients,
    residual_penalty,
)
from examples.synthesis.applications.telecom.radio_access_network.radio_access_network_positioning_antennas.overview.quantum import (
    build_program,
    main,
)
from examples.synthesis.applications.telecom.radio_access_network.radio_access_network_positioning_antennas.overview.reference import (
    reference_circuit,
)
from examples.synthesis.applications.telecom.radio_access_network.radio_access_network_positioning_antennas.overview.workflow import (
    cost_table,
    read_distribution,
    run_workflow,
    sample_distribution,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth

np = pytest.importorskip("numpy")
pytestmark = pytest.mark.integration
BINDINGS = (
    (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08),
    (-0.17, 0.23, -0.31, 0.41, -0.53, 0.67, -0.79, 0.83),
)


def test_fresh_legacy_seed_data_preserves_global_rng_and_source_initialization() -> (
    None
):
    state = np.random.get_state()
    try:
        np.random.seed(7)
        expected_next = np.random.RandomState(7).random_sample()
        coefficients = fresh_coefficients()
        assert coefficients == tuple(
            float(v) for v in np.random.RandomState(0).rand(1, 13)[0]
        )
        assert float(np.random.random_sample()) == expected_next
        assert len(coefficients) == 13
    finally:
        np.random.set_state(state)


def test_all_joint_costs_preserve_maximization_and_three_binary_slack_bits() -> None:
    coefficients = fresh_coefficients()
    table = cost_table(coefficients)
    feasible = 0
    zero = 0
    for word in range(8192):
        bits = tuple((word >> i) & 1 for i in range(13))
        value = assignment(word, coefficients)
        expected_score = sum(c * b for c, b in zip(coefficients, bits, strict=True))
        legal = (
            sum(bits) <= 7
            and bits[1] + bits[3] + bits[4] == 1
            and bits[0] + bits[5] == 1
            and bits[6] + bits[9] == 1
        )
        assert value.score == pytest.approx(expected_score)
        assert value.feasible == legal
        feasible += legal
        for slack in range(8):
            penalty = (
                (sum(bits) + slack - 7) ** 2
                + (bits[1] + bits[3] + bits[4] - 1) ** 2
                + (bits[0] + bits[5] - 1) ** 2
                + (bits[6] + bits[9] - 1) ** 2
            )
            assert residual_penalty(word, slack) == penalty
            assert float(table[word | (slack << 13)]) == pytest.approx(
                -expected_score + 3 * penalty
            )
            zero += penalty == 0
    assert feasible == zero == 684
    assert float(table.mean()) == pytest.approx(58.5 - sum(coefficients) / 2)
    assert float(table.min()) == pytest.approx(-5.048594866139186)


@pytest.mark.parametrize("parameters", BINDINGS)
def test_two_nonzero_full_complex_states_use_fresh_native_costs(
    parameters: tuple[float, ...],
) -> None:
    coefficients = fresh_coefficients()
    program = build_program(parameters, coefficients)
    compare_states(program, reference_circuit(parameters, coefficients), ("x", "slack"))
    assert program.report["workspace_qubits"] == 0


def test_actual_zero_angle_uniform_state_and_joint_sample_order() -> None:
    coefficients = fresh_coefficients()
    probabilities = read_distribution(build_program((0.0,) * 8, coefficients))
    assert torch.allclose(
        probabilities, torch.full_like(probabilities, 1 / 2**16), atol=1e-14, rtol=0
    )
    observed = sample_distribution(probabilities, coefficients, shots=3000, seed=7)
    assert observed == sample_distribution(
        probabilities, coefficients, shots=3000, seed=7
    )
    assert sum(row.count for row in observed) == 3000
    for row in observed:
        assert row.assignment.selected == tuple(
            i for i in range(13) if row.assignment.word & (1 << i)
        )
        assert row.cost == pytest.approx(-row.assignment.score + 3 * row.penalty)


def test_infeasible_joint_samples_remain_actual_observations() -> None:
    probabilities = torch.zeros(2**16, dtype=torch.float64)
    probabilities[-1] = 1
    rows = sample_distribution(probabilities, fresh_coefficients(), shots=100, seed=11)
    assert len(rows) == 1 and rows[0].assignment.word == 8191 and rows[0].slack == 7
    assert (
        rows[0].count == 100
        and rows[0].penalty == 175
        and not rows[0].assignment.feasible
    )


@pytest.mark.parametrize(
    "parameters", ((0.1,), (True,) * 8, (math.nan,) * 8, (math.inf,) * 8)
)
def test_invalid_angles_refuse(parameters: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="eight finite"):
        build_program(parameters, fresh_coefficients())


@pytest.mark.parametrize("coefficients", ((0.1,), (True,) * 13, (math.nan,) * 13))
def test_invalid_coefficients_refuse(coefficients: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="thirteen finite"):
        build_program(BINDINGS[0], coefficients)


def test_joint_width_budget_cannot_drop_the_modeled_slack_bits() -> None:
    with pytest.raises(ValueError, match="qubit|width|budget"):
        synth.synthesize(
            main,
            classical_inputs={
                "params": BINDINGS[0],
                "coefficients": fresh_coefficients(),
            },
            max_qubits=15,
            max_workspace=0,
        )


def test_actual_bounded_search_has_no_classical_answer_substitution() -> None:
    result = run_workflow(maxiter=1, shots=150, seed=7)
    assert result.trace and result.optimizer.iterations <= 1
    assert all(math.isfinite(row.cost) for row in result.trace)
    assert sum(row.count for row in result.samples) == 150
    assert result.coefficients == fresh_coefficients()


def test_all_ten_source_stages_are_mapped_and_unexecuted() -> None:
    root = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/applications/telecom/radio_access_network/radio_access_network_positioning_antennas/overview"
    )
    proof = json.loads((root / "provenance.json").read_text())
    notebook = json.loads((root / "walkthrough.ipynb").read_text())
    assert tuple(row["index"] for row in proof["reviewed_code_cells"]) == (
        3,
        4,
        7,
        10,
        12,
        14,
        16,
        18,
        20,
        21,
    )
    assert all(
        not c.get("outputs") for c in notebook["cells"] if c["cell_type"] == "code"
    )

"""Source gradient sweeps preserve signed phase,real counts and clean auxiliary."""

import json
import math
from pathlib import Path

import pytest

from examples.synthesis.algorithms.quantum_primitives.gradient_estimation.overview.ansatz import (
    auxiliary_phase,
    build_program,
    phase_only,
)
from examples.synthesis.algorithms.quantum_primitives.gradient_estimation.overview.workflow import (
    L_VALUES,
    M_VALUES,
    Model,
    Observation,
    build_two_dimensional,
    phase_table,
    sample,
    success_rate,
    verify_auxiliary,
    verify_program,
    verify_two_dimensional,
)
from flagquantum import synth

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "model",
    (
        Model(0, 0.5, 0.25),
        Model(0.5, 0.25, 0.1, window=2),
        Model(0.5, 0.25, 0.1, window=0.2),
        Model(0, -0.5, 0.25),
        Model(0, 0.55, 0.25),
        Model(0.6, 0.25, 0.1, window=0.1),
        Model(0.6, 0.25, 0.1, window=1),
    ),
)
@pytest.mark.parametrize("transform", (False, True))
def test_signed_polynomial_absolute_states(model: Model, transform: bool) -> None:
    p = build_program(model.coefficients, inverse_fourier=transform)
    check = verify_program(p, model, inverse_fourier=transform)
    assert (
        check.maximum_state_error < 2e-11 and check.dirty_workspace_probability < 1e-22
    )


def test_auxiliary_is_real_modular_kickback_and_returns_to_zero() -> None:
    p = synth.synthesize(auxiliary_phase, max_workspace=0)
    check = verify_auxiliary(p)
    assert (
        check.maximum_state_error < 2e-11 and check.dirty_workspace_probability < 1e-22
    )
    model = Model(0, 0.5, 0.25)
    assert [r.unwrapped_phase for r in phase_table(p, model)] == pytest.approx(
        [x / 4 for x in range(-4, 4)], abs=1e-5
    )


@pytest.mark.parametrize("parameter,values", (("m", M_VALUES), ("l", L_VALUES)))
def test_all_source_sweep_points_have_independent_absolute_references(
    parameter: str, values: tuple[float, ...]
) -> None:
    assert len(values) == (17 if parameter == "m" else 9)
    for value in values:
        model = (
            Model(0.6, 0.25, 0.1, window=0.1, m=value)
            if parameter == "m"
            else Model(0.6, 0.25, 0.1, window=value, m=1)
        )
        p = build_program(model.coefficients)
        check = verify_program(p, model, inverse_fourier=True)
        assert (
            check.maximum_state_error < 2e-11
            and check.dirty_workspace_probability < 1e-22
        )


def test_actual_small_and_large_window_success_thresholds() -> None:
    small = Model(0.6, 0.25, 0.1, window=0.1)
    large = Model(0.6, 0.25, 0.1, window=1)
    good = success_rate(
        sample(build_program(small.coefficients), seed=3002), small, (0.25,)
    )
    bad = success_rate(
        sample(build_program(large.coefficients), seed=3003), large, (0.25,)
    )
    assert good.shots == bad.shots == 1000
    assert good.rate > 0.9 and bad.rate < 0.5
    assert good.majority_gradient == (0.25,) and good.majority_correct
    assert good.rate == good.successes / 1000 and bad.rate == bad.successes / 1000


def test_two_dimensional_full_state_and_joint_readout() -> None:
    p = build_two_dimensional()
    check = verify_two_dimensional(p)
    assert (
        check.maximum_state_error < 2e-11 and check.dirty_workspace_probability < 1e-22
    )
    result = success_rate(
        sample(p, seed=3400), Model(0.6, 0.25, 0.1, window=0.1), (0.25, -0.5)
    )
    assert result.rate > 0.9 and result.majority_gradient == (0.25, -0.5)


def test_noninteger_gradient_uses_actual_reproducible_draws() -> None:
    model = Model(0, 0.55, 0.25)
    p = build_program(model.coefficients)
    first = sample(p, seed=3001)
    assert first == sample(p, seed=3001) and first != sample(p, seed=3002)
    assert sum(row.count for row in first) == 1000 and len(first) > 1
    assert all(row.frequency == row.count / 1000 for row in first)


def test_signed_decoding_and_strict_success_boundaries() -> None:
    model = Model(0, 0, 0)
    assert success_rate((Observation((0,), 100, 1.0),), model, (0.125,)).successes == 0
    assert (
        success_rate((Observation((0,), 100, 1.0),), model, (0.124,)).successes == 100
    )
    assert success_rate(
        (Observation((-2,), 100, 1.0),), model, (-0.5,)
    ).majority_gradient == (-0.5,)
    assert success_rate((), model, (0.25,)).rate == 0
    with pytest.raises(ValueError, match="dimensions"):
        success_rate((Observation((0,), 100, 1.0),), model, (0.25, -0.5))
    with pytest.raises(ValueError, match="tolerance"):
        success_rate((), model, (0.25,), tolerance=0)


@pytest.mark.parametrize(
    "window,m",
    (
        (0, 2),
        (0.1, 0),
        (-1, 2),
        (0.1, -2),
        (math.nan, 2),
        (0.1, math.inf),
        (True, 2),
        (0.1, True),
    ),
)
def test_invalid_classical_models_are_refused(window: float, m: float) -> None:
    with pytest.raises(ValueError):
        Model(0.6, 0.25, 0.1, window=window, m=m)


def test_budget_does_not_reduce_the_original_coordinate_width() -> None:
    with pytest.raises(ValueError):
        synth.synthesize(
            phase_only,
            classical_inputs={"coefficients": [0.0, 0.25, 2.0]},
            max_qubits=2,
        )


def test_all_source_stages_and_helper_provenance_are_present() -> None:
    directory = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/algorithms/quantum_primitives/gradient_estimation/overview"
    )
    proof = json.loads((directory / "provenance.json").read_text())
    n = json.loads((directory / "walkthrough.ipynb").read_text())
    assert [r["index"] for r in proof["reviewed_code_cells"]] == [
        3,
        13,
        15,
        20,
        24,
        26,
        30,
        33,
        35,
        39,
        41,
        44,
        46,
        49,
    ]
    assert proof["reviewed_dependencies"][0]["path"].endswith(
        "gradient_estimation_helpers.py"
    )
    for row in proof["reviewed_code_cells"]:
        assert n["cells"][row["native_code_cell"]]["cell_type"] == "code"
    text = "\n".join("".join(c["source"]) for c in n["cells"])
    assert (
        "gradient_estimation_helpers import" not in text
        and "classiq import" not in text
    )

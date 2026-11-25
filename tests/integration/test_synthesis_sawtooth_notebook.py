"""Full original kicked-map workflow, signed readout and historical data honesty."""

import hashlib
import json
from pathlib import Path

import pytest

from examples.synthesis.applications.physical_systems.quantum_chaos.quantum_sawtooth_map.main import (
    main,
)
from examples.synthesis.applications.physical_systems.quantum_chaos.quantum_sawtooth_map.overview.workflow import (
    DISPLAY_MOMENTA,
    TIMESTEPS,
    Observation,
    display_probabilities,
    historical_observations,
    localization_parameters,
    probabilities,
    run_workflow,
    sample,
    verify,
)
from flagquantum import synth
from flagquantum.compiler.basis_conversion import BasisConversionError

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("kicks", (0, 1, 2, 4, 6, 8))
def test_absolute_native_and_exact_target_states(kicks: int) -> None:
    row = verify(kicks)
    assert row.maximum_native_error < 3e-11
    assert row.maximum_target_error < 3e-11
    assert row.dirty_workspace_probability == 0.0
    assert row.resources.n_qubits == 3
    assert set(row.resources.operation_counts) <= {"cx", "u3"}
    assert row.resources.operation_counts.get("cx", 0) == 28 * kicks
    assert row.resources.depth == 1 + 56 * kicks
    values = probabilities(row.program)
    assert sum(values) == pytest.approx(1.0)
    if kicks:
        assert values[2] > 0.85  # Numeric signed momentum -2, not raw bitstring 2.
    else:
        assert values == pytest.approx((0, 0, 1, 0, 0, 0, 0, 0))


def test_original_times_shots_and_localization() -> None:
    rows = run_workflow()
    assert tuple(row.evolution.kicks for row in rows) == (2, 4, 6, 8)
    for row in rows:
        assert sum(r.count for r in row.observations) == 1000
        assert tuple(r.momentum for r in row.observations) == tuple(range(-4, 4))
        assert next(r.probability for r in row.observations if r.momentum == -2) > 0.85
    bound, length = localization_parameters()
    assert bound > 0.1 and length < 8


def test_actual_seeded_sampling_cannot_inject_localized_counts() -> None:
    values = (1.0,) + (0.0,) * 7
    rows = sample(values, shots=101, seed=7)
    assert rows == sample(values, shots=101, seed=7)
    assert rows[0].momentum == -4 and rows[0].count == 101
    assert rows[2].momentum == -2 and rows[2].count == 0
    assert sum(row.count for row in rows) == 101


@pytest.mark.parametrize("kicks", TIMESTEPS)
def test_historical_assets_are_original_eight_state_measurements(kicks: int) -> None:
    rows = historical_observations(kicks)
    assert sum(row.count for row in rows) == 1000
    assert {row.momentum for row in rows} == set(range(-4, 4))
    assert all(row.probability == row.count / 1000 for row in rows)
    assert (
        next(
            row.probability for row in historical_observations(2) if row.momentum == -2
        )
        == 0.706
    )


def test_extra_display_point_is_zero_not_a_ninth_quantum_state() -> None:
    rows = tuple(Observation(p, 125, 0.125) for p in range(-4, 4))
    assert tuple(range(-4, 5)) == DISPLAY_MOMENTA
    assert display_probabilities(rows) == (0.125,) * 8 + (0.0,)


@pytest.mark.parametrize(
    "values", ((1.0,), (float("nan"),) * 8, (-1.0, 2.0) + (0.0,) * 6, (0.0,) * 8)
)
def test_invalid_probabilities(values: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="eight normalized"):
        sample(values)


@pytest.mark.parametrize("shots,seed", ((0, 0), (True, 0), (100, -1), (100, True)))
def test_invalid_sampling_controls(shots: int, seed: int) -> None:
    with pytest.raises(ValueError, match="shots|seed"):
        sample((0.125,) * 8, shots=shots, seed=seed)


@pytest.mark.parametrize("kicks", (-1, 17, True))
def test_invalid_kick_binding(kicks: int) -> None:
    with pytest.raises(ValueError, match="zero and sixteen"):
        verify(kicks)


@pytest.mark.parametrize("kicks", (0, 1, 3, True))
def test_nonhistorical_time_is_refused(kicks: int) -> None:
    with pytest.raises(ValueError, match="historical kicks"):
        historical_observations(kicks)


def test_exact_conversion_budget_is_enforced() -> None:
    with pytest.raises(BasisConversionError, match="max_added_operations"):
        verify(2, max_added_operations=1)


@pytest.mark.parametrize("limit", ("qubits", "operations", "memory"))
def test_original_resource_budget_refusals(limit: str) -> None:
    with pytest.raises(ValueError, match="budget|limit|memory|candidate"):
        synth.synthesize(
            main,
            classical_inputs={"num_kicks": 8},
            max_qubits=2 if limit == "qubits" else 3,
            max_workspace=0,
            max_operations=1 if limit == "operations" else 2048,
            memory_limit_bytes=1 if limit == "memory" else 64 * 1024**2,
        )


def test_all_source_stages_assets_and_frozen_parent() -> None:
    directory = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/applications/physical_systems/quantum_chaos/quantum_sawtooth_map/overview"
    )
    proof = json.loads((directory / "provenance.json").read_text())
    assert [row["index"] for row in proof["reviewed_code_cells"]] == [
        7,
        12,
        14,
        16,
        18,
        19,
        21,
        22,
        24,
        25,
        27,
        28,
        29,
        30,
    ]
    assert (
        hashlib.sha256((directory.parent / "main.py").read_bytes()).hexdigest()
        == proof["frozen_quantum_entry_sha256"]
    )
    for asset in proof["historical_assets"]:
        assert asset["kind"] == "historical_upstream_hardware_measurement"
        assert (
            hashlib.sha256((directory / asset["path"]).read_bytes()).hexdigest()
            == asset["sha256"]
        )
    notebook = json.loads((directory / "walkthrough.ipynb").read_text())
    code = "\n".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
    )
    assert "classiq" not in code and ".qmod" not in code
    assert "RUN_ON_HARDWARE = False" in code

"""Original full-size walks retain entangled coin, phases and actual sampling."""

import json
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.tutorials.advanced_tutorials.discrete_quantum_walk_2d.discrete_time_quantum_walk.overview.workflow import (
    build_program,
    reference_ring,
    run_workflow,
    sample,
    verify,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("dimension", (1, 2))
@pytest.mark.parametrize("steps", (0, 1, 2, 20))
def test_full_absolute_states_include_retained_coin(dimension: int, steps: int) -> None:
    # Parameterized tests use the concrete Literal alternatives at each call.
    row = verify(1, steps) if dimension == 1 else verify(2, steps)
    assert row.maximum_error < 1e-12
    assert row.dirty_workspace_probability < 1e-22
    assert tuple(row.program.registers) == (("x",) if dimension == 1 else ("x", "y"))
    assert tuple(len(q) for q in row.program.registers.values()) == (
        (6,) if dimension == 1 else (5, 5)
    )


def test_three_original_actual_draw_batches_and_coordinate_order() -> None:
    rows = run_workflow()
    assert [(e.dimension, e.steps) for e, _ in rows] == [(1, 1), (1, 20), (2, 20)]
    for evolution, observations in rows:
        assert sum(row.count for row in observations) == 1000
        assert all(row.probability == row.count / 1000 for row in observations)
        assert all(len(row.position) == evolution.dimension for row in observations)
        assert all(
            0 <= coordinate < (64 if evolution.dimension == 1 else 32)
            for row in observations
            for coordinate in row.position
        )
    assert {r.position for r in rows[0][1]} == {(31,), (33,)}
    assert all(r.position[0] % 2 == 0 for r in rows[1][1])
    assert all((r.position[0] + r.position[1]) % 2 == 0 for r in rows[2][1])


def test_one_step_coin_is_retained_and_entangled() -> None:
    row = verify(1, 1)
    state = fq.run(row.program).statevector().reshape(64, 2, -1)[:, :, 0]
    density = state @ state.conj().T
    assert float(torch.trace(density @ density).real) == pytest.approx(0.5)


def test_sampling_is_actual_reproducible_draws_not_injected_counts() -> None:
    program = build_program(1, 20)
    first = sample(program, shots=1000, seed=19)
    assert first == sample(program, shots=1000, seed=19)
    assert first != sample(program, shots=1000, seed=20)


@pytest.mark.parametrize("steps", (-1, 21, True))
def test_step_budget_is_checked_before_synthesis(steps: int) -> None:
    with pytest.raises(ValueError, match="0..20"):
        build_program(1, steps)


@pytest.mark.parametrize(
    "shots,seed", ((0, 0), (True, 0), (100, -1), (100, True), (100, 2**63))
)
def test_sampling_budget_and_seed_refusals(shots: int, seed: int) -> None:
    with pytest.raises(ValueError, match="shots|seed"):
        sample(build_program(1, 0), shots=shots, seed=seed)


def test_reference_shift_wraps_both_ring_boundaries() -> None:
    # The pure shift matrix embedded in each step maps0->63 for coin0 and
    # 63->0 for coin1. Its full64-position domain also verifies both directions.
    circuit = reference_ring(1)
    matrix = circuit.to_ir().instructions[-1].matrix
    assert matrix is not None
    for coin in range(2):
        for position in range(64):
            column = coin * 64 + position
            expected = coin * 64 + (position + (1 if coin else -1)) % 64
            assert complex(matrix[expected, column]) == 1
            assert float(matrix[:, column].abs().square().sum()) == 1


def test_every_original_notebook_stage_is_mapped() -> None:
    root = Path(__file__).resolve().parents[2]
    directory = (
        root
        / "examples/synthesis/tutorials/advanced_tutorials/discrete_quantum_walk_2d/discrete_time_quantum_walk/overview"
    )
    proof = json.loads((directory / "provenance.json").read_text())
    notebook = json.loads((directory / "walkthrough.ipynb").read_text())
    assert [row["index"] for row in proof["reviewed_code_cells"]] == [
        2,
        3,
        4,
        5,
        6,
        7,
        9,
        10,
        11,
        12,
    ]
    for row in proof["reviewed_code_cells"]:
        assert notebook["cells"][row["native_code_cell"]]["cell_type"] == "code"

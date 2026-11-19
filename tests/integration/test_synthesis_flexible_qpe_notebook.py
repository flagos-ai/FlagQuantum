"""Original flexible-QPE stage preservation and shared-environment full-state acceptance."""

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")

import numpy as np

from examples.synthesis.tutorials.advanced_tutorials.high_level_modeling_flexible_qpe.overview.ansatz import (
    build_program,
    exact_matrix,
    repetition_table,
    source_eigenpair,
    source_matrix,
)
from examples.synthesis.tutorials.advanced_tutorials.high_level_modeling_flexible_qpe.overview.workflow import (
    Check,
    verify_and_sample,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", params=(False, True))
def check(request: pytest.FixtureRequest) -> Check:
    comparison = bool(request.param)
    return verify_and_sample(
        build_program(comparison), comparison=comparison, seed=4100 + comparison
    )


def test_original_full_physical_state_and_named_marginal(check: Check) -> None:
    assert check.state_error < 2e-11
    assert check.named_error < 2e-11
    assert check.resources.n_qubits == (13 if check.comparison else 7)
    assert check.resources.channel_count == 0
    assert check.resources.n_operations > 1000
    assert sum(row.count for row in check.outcomes) == 1024
    if check.comparison:
        assert check.exact_overlap is not None
        assert 0.94 < check.exact_overlap < 0.98
        assert check.observed_overlap is not None
        assert abs(check.observed_overlap - check.exact_overlap) < 0.05
    else:
        assert check.most_likely_phase == 22 / 32
        assert max(check.outcomes, key=lambda row: row.probability).word == 22


def test_fresh_source_eigendecomposition_and_index_convention() -> None:
    eigenvalue, vector = source_eigenpair()
    assert eigenvalue == pytest.approx(0.7031971279434318, abs=2e-14)
    source_vector = np.array(
        [
            -0.08272059133944247,
            -0.41789453693845247,
            0.9027098719984156,
            -0.060302129001003825,
        ]
    )
    np.testing.assert_allclose(vector, source_vector, atol=2e-14, rtol=0)
    np.testing.assert_allclose(
        source_matrix() @ source_vector, eigenvalue * source_vector, atol=2e-14, rtol=0
    )
    matrix = np.asarray(exact_matrix())
    np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(4), atol=2e-14, rtol=0)
    np.testing.assert_allclose(
        matrix @ source_vector,
        np.exp(2j * np.pi * eigenvalue) * source_vector,
        atol=2e-14,
        rtol=0,
    )


def test_original_conditional_ceiling_schedule() -> None:
    counts = repetition_table()
    assert tuple(counts[p] for p in (1, 2, 4, 8, 16)) == (4, 4, 4, 12, 32)
    assert counts[3] == counts[4] == 4
    assert counts[5] == 6
    assert counts[7] == 10


@pytest.mark.parametrize(
    "arguments",
    ({"shots": 0}, {"shots": True}, {"shots": 100001}, {"seed": -1}, {"seed": True}),
)
def test_readout_validation_precedes_execution(arguments: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="seed|shots"):
        verify_and_sample(
            build_program(),
            shots=arguments.get("shots", 1024),
            seed=arguments.get("seed", 4100),
        )


def test_source_stages_are_all_mapped_to_executable_native_cells() -> None:
    directory = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/tutorials/advanced_tutorials/high_level_modeling_flexible_qpe/overview"
    )
    provenance = json.loads((directory / "provenance.json").read_text())
    notebook = json.loads((directory / "walkthrough.ipynb").read_text())
    assert [row["index"] for row in provenance["reviewed_code_cells"]] == [
        3,
        6,
        10,
        14,
        17,
        20,
        21,
        23,
        24,
        25,
        29,
        31,
        32,
        34,
        35,
        37,
    ]
    assert len(provenance["original_sha256"]) == hashlib.sha256().digest_size * 2
    assert all(
        notebook["cells"][row["native_code_cell"]]["cell_type"] == "code"
        for row in provenance["reviewed_code_cells"]
    )

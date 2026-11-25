"""Native Oracle Workshop keeps XOR,absolute phases and clean borrowed auxiliaries."""

import json
import math
from pathlib import Path
from typing import Annotated

import pytest
import torch

import flagquantum as fq
from examples.synthesis.tutorials.workshops.oracle_workshop.oracles_workshop.overview.ansatz import (
    oracle,
)
from examples.synthesis.tutorials.workshops.oracle_workshop.oracles_workshop.overview.workflow import (
    STAGES,
    Observation,
    build_program,
    classify_deutsch_jozsa,
    declared_amplitudes,
    phase_table,
    sample,
    verify,
)
from flagquantum import synth
from flagquantum.synth import (
    H,
    Output,
    QBit,
    QNum,
    X,
    allocate,
    hadamard_transform,
    qfunc,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("name", STAGES)
def test_full_complex_states_and_clean_hidden_work(name: str) -> None:
    row = verify(name)
    assert row.maximum_state_error < 2e-11
    assert row.dirty_workspace_probability < 1e-22


@qfunc
def nonzero_mark(
    x: Output[Annotated[QNum, 1]], y: Output[Annotated[QNum, 2]], z: Output[QBit]
) -> None:
    allocate(x)
    allocate(y)
    allocate(z)
    hadamard_transform(x)
    hadamard_transform(y)
    X(z)
    oracle(x, y, z)


@qfunc
def coherent_mark(
    x: Output[Annotated[QNum, 1]], y: Output[Annotated[QNum, 2]], z: Output[QBit]
) -> None:
    allocate(x)
    allocate(y)
    allocate(z)
    hadamard_transform(x)
    hadamard_transform(y)
    H(z)
    oracle(x, y, z)


def test_predicate_xor_handles_nonzero_target_without_overwriting() -> None:
    p = synth.synthesize(nonzero_mark)
    assert dict(p.probabilities(fq.run(p), registers=("x", "y", "z"))) == pytest.approx(
        {(x, y, 1 ^ int(2 * x + y == 4)): 1 / 8 for x in range(2) for y in range(4)}
    )


def test_predicate_xor_preserves_superposed_mark_state() -> None:
    p = synth.synthesize(coherent_mark)
    actual = declared_amplitudes(p)
    torch.testing.assert_close(
        actual[:, 0],
        torch.full((16,), 0.25, dtype=torch.complex128),
        atol=1e-12,
        rtol=0,
    )
    assert float(actual[:, 1:].abs().square().sum()) < 1e-22


def test_auxiliary_and_direct_phase_tables_preserve_eight_absolute_signs() -> None:
    tables = [
        phase_table(build_program(name))
        for name in ("auxiliary_kickback", "direct_kickback")
    ]
    for table in tables:
        assert len(table) == 8
        for row in table:
            marked = 2 * row.x + row.y == 4
            assert row.solution == marked
            assert row.amplitude == pytest.approx(
                (-1 if marked else 1) / math.sqrt(8), abs=1e-12
            )
            assert row.magnitude == pytest.approx(1 / math.sqrt(8))
    assert sum(row.solution for row in tables[0]) == 1
    with pytest.raises(ValueError, match="one-bit"):
        phase_table(build_program("grover"))


def test_negative_difference_is_not_a_modular_grover_solution() -> None:
    p = build_program("grover")
    values = p.probabilities(fq.run(p), registers=("x", "y"))
    expected = math.sin(5 * math.asin(math.sqrt(2 / 16))) ** 2
    assert sum(v for (x, y), v in values.items() if x - y == 2) == pytest.approx(
        expected, abs=1e-12
    )
    assert values[(0, 2)] < values[(2, 0)]
    assert values[(1, 3)] < values[(3, 1)]


def test_actual_deutsch_jozsa_readout_is_balanced() -> None:
    observations = sample(build_program("deutsch_jozsa"), shots=1000, seed=19)
    assert sum(row.count for row in observations) == 1000
    assert classify_deutsch_jozsa(observations) == "balanced"
    assert {row.values for row in observations} == {(1,)}


def test_classifier_does_not_invent_a_promise_result() -> None:
    assert classify_deutsch_jozsa((Observation((0,), 10, 1.0),)) == "constant"
    assert (
        classify_deutsch_jozsa((Observation((0,), 5, 0.5), Observation((1,), 5, 0.5)))
        == "undetermined"
    )
    assert classify_deutsch_jozsa(()) == "undetermined"
    with pytest.raises(ValueError, match="one named word"):
        classify_deutsch_jozsa((Observation((0, 1), 10, 1.0),))


def test_actual_sampling_is_reproducible_and_not_injected() -> None:
    p = build_program("grover")
    first = sample(p, shots=1000, seed=19)
    assert first == sample(p, shots=1000, seed=19)
    assert first != sample(p, shots=1000, seed=20)
    assert all(row.frequency == row.count / 1000 for row in first)


@pytest.mark.parametrize(
    "shots,seed", ((0, 0), (True, 0), (100, -1), (100, True), (100, 2**63))
)
def test_invalid_draw_controls_refused(shots: int, seed: int) -> None:
    with pytest.raises(ValueError, match="shots|seed"):
        sample(build_program("deutsch_jozsa"), shots=shots, seed=seed)


def test_all_original_stages_and_both_kickback_methods_are_present() -> None:
    directory = (
        Path(__file__).resolve().parents[2]
        / "examples/synthesis/tutorials/workshops/oracle_workshop/oracles_workshop/overview"
    )
    proof = json.loads((directory / "provenance.json").read_text())
    notebook = json.loads((directory / "walkthrough.ipynb").read_text())
    assert [row["index"] for row in proof["reviewed_code_cells"]] == [
        8,
        9,
        12,
        13,
        16,
        17,
        25,
        26,
        27,
        33,
        35,
        36,
        41,
        42,
        44,
        45,
        48,
        49,
        50,
    ]
    for row in proof["reviewed_code_cells"]:
        assert notebook["cells"][row["native_code_cell"]]["cell_type"] == "code"
    text = "\n".join("".join(c["source"]) for c in notebook["cells"])
    assert "auxiliary_program = synth.synthesize(" in text
    assert "direct_program = synth.synthesize(" in text
    assert text.count('check_program("') == 8
    assert "classiq import" not in text

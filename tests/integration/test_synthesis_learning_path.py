"""Execute the teaching path and independently check its quantum meanings."""

import math
from pathlib import Path
from typing import cast

import pytest
import torch

import flagquantum as fq
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler import final_layout
from tests.integration.synthesis_notebook_support import execute_code_cells

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
LESSONS = ROOT / "examples/synthesis/tutorials/learning_path"


def run_lesson(name: str) -> dict[str, object]:
    namespace, _ = execute_code_cells(LESSONS / f"{name}.ipynb")
    return cast(dict[str, object], namespace)


def test_bell_relative_phase_and_named_readout() -> None:
    namespace = run_lesson("01_states")
    program = cast(synth.Program, namespace["program"])
    reference = fq.Circuit(2, dtype=torch.complex128).h(0).cx(0, 1).z(1)
    compare_states(program, reference, ("pair",))
    assert namespace["outcomes"] == pytest.approx({(0,): 0.5, (3,): 0.5})


def test_bell_runtime_counts_are_seeded_and_separate_from_exact_readout() -> None:
    namespace = run_lesson("01_states")
    program = cast(synth.Program, namespace["program"])
    counts = cast(dict[str, int], namespace["shot_counts"])
    assert sum(counts.values()) == 4096
    assert set(counts) == {"00", "11"}
    repeated = fq.run(
        program.circuit,
        outputs=fq.counts(qubits=program.registers["pair"], name="pair"),
        options=fq.ExecutionOptions(shots=4096, seed=37),
    )
    assert repeated.measurement("pair").value[0] == counts
    other = fq.run(
        program.circuit,
        outputs=fq.counts(qubits=program.registers["pair"], name="pair"),
        options=fq.ExecutionOptions(shots=4096, seed=91),
    )
    assert other.counts[0] != counts
    assert counts["00"] / 4096 == pytest.approx(0.5, abs=0.05)
    assert namespace["outcomes"] == pytest.approx({(0,): 0.5, (3,): 0.5})
    # A measured IR is a distinct result identity; it must not impersonate the
    # original sealed synthesis execution.
    with pytest.raises((TypeError, ValueError)):
        program.probabilities(repeated, registers=("pair",))
    with pytest.raises(TypeError, match="execution overrides"):
        fq.run(program, outputs=fq.counts(), shots=4096)


def test_computation_preserves_every_input_and_cleans_workspace() -> None:
    namespace = run_lesson("02_computation")
    program = cast(synth.Program, namespace["program"])
    expected = torch.zeros(256, dtype=torch.complex128)
    for x in range(4):
        for y in range(4):
            expected[(x << 6) | (y << 4) | (x * x + y)] = 0.25
    # Householder preparation is independent of synth arithmetic lowering.
    zero = torch.zeros_like(expected)
    zero[0] = 1
    direction = (zero - expected) / torch.linalg.vector_norm(zero - expected)
    matrix = torch.eye(256, dtype=torch.complex128) - 2 * torch.outer(
        direction, direction.conj()
    )
    reference = fq.Circuit(8, dtype=torch.complex128).any(*range(8), matrix=matrix)
    compare_states(program, reference, ("x", "y", "result"))
    assert namespace["outcomes"] == pytest.approx(
        {(x, y, x * x + y): 1 / 16 for x in range(4) for y in range(4)}
    )


def test_predicate_phase_is_visible_through_interference() -> None:
    namespace = run_lesson("03_predicate_phase")
    reference = fq.Circuit(2, dtype=torch.complex128).h(0).h(1)
    reference.any(
        0,
        1,
        matrix=torch.diag(torch.tensor([-1, -1, 1, 1], dtype=torch.complex128)),
    ).h(0).h(1)
    compare_states(cast(synth.Program, namespace["program"]), reference, ("x",))
    assert namespace["outcomes"] == pytest.approx({(2,): 1.0})


def test_compute_use_uncompute_retains_phase_and_erases_temporary() -> None:
    namespace = run_lesson("04_uncomputation")
    values = torch.tensor(
        [x * x + y for x in range(4) for y in range(4)], dtype=torch.float64
    )
    reference = fq.Circuit(4, dtype=torch.complex128)
    for wire in range(4):
        reference.h(wire)
    reference.any(*range(4), matrix=torch.diag(torch.exp(0.3j * values)))
    compare_states(cast(synth.Program, namespace["program"]), reference, ("x", "y"))


def test_topology_preserves_coherent_control_and_bounds() -> None:
    namespace = run_lesson("05_constraints")
    reference = fq.Circuit(5, dtype=torch.complex128)
    for wire in range(4):
        reference.h(wire)
    permutation = torch.eye(32, dtype=torch.complex128)
    permutation[30, 30] = permutation[31, 31] = 0
    permutation[30, 31] = permutation[31, 30] = 1
    reference.any(*range(5), matrix=permutation)
    programs = cast(dict[str, synth.Program], namespace["programs"])
    for label, program in programs.items():
        state = fq.run(program).statevector().flatten()
        logical = (*program.registers["controls"], *program.registers["target"])
        if label == "line":
            logical = final_layout(program.circuit).map_qubits(logical)
        hidden = tuple(q for q in range(program.circuit.n_wires) if q not in logical)
        ordered = (
            state.reshape((2,) * program.circuit.n_wires)
            .permute((*logical, *hidden))
            .reshape(32, -1)
        )
        torch.testing.assert_close(
            ordered[:, 0], fq.run(reference).statevector().flatten(), atol=1e-12, rtol=0
        )
        assert float(ordered[:, 1:].abs().square().sum()) < 1e-24
        assert program.circuit.n_wires <= 6
        if label == "line":
            assert all(
                abs(op.wires[0] - op.wires[1]) == 1
                for op in program.circuit.instructions
                if op.name == "cx"
            )


def test_variational_objective_records_actual_quantum_energies() -> None:
    namespace = run_lesson("06_variational")
    evaluations = cast(list[tuple[float, float]], namespace["evaluations"])
    assert len(evaluations) > 2
    for angle, energy in evaluations:
        assert energy == pytest.approx(
            math.cos(angle) + 0.1 * math.sin(angle), abs=1e-12
        )
    assert min(value for _, value in evaluations) == pytest.approx(
        -math.sqrt(1.01), abs=1e-10
    )
    runtime = cast(list[tuple[float, float, float]], namespace["runtime_evaluations"])
    assert [angle for angle, _, _ in runtime] == [0.31, -0.47, 1.13]
    for angle, energy, gradient in runtime:
        assert energy == pytest.approx(
            math.cos(angle) + 0.1 * math.sin(angle), abs=1e-12
        )
        assert gradient == pytest.approx(
            -math.sin(angle) + 0.1 * math.cos(angle), abs=1e-12
        )


@pytest.mark.parametrize(
    ("relative", "registers", "reference"),
    [
        (
            "functions/standard_gates/overview",
            ("control_bit", "target"),
            fq.Circuit(2, dtype=torch.complex128).h(0).cx(0, 1),
        ),
        (
            "functions/unitary",
            ("x",),
            fq.Circuit(2, dtype=torch.complex128)
            .h(0)
            .h(1)
            .any(
                0,
                1,
                matrix=torch.diag(
                    torch.tensor([1, 1, -1j, 1j], dtype=torch.complex128)
                ),
            ),
        ),
    ],
)
def test_supplementary_coherent_examples_expose_gate_effects(
    relative: str, registers: tuple[str, ...], reference: fq.Circuit
) -> None:
    namespace, _ = execute_code_cells(
        ROOT / "examples/synthesis" / relative / "walkthrough.ipynb"
    )
    program = cast(synth.Program, namespace["coherent_program"])
    compare_states(program, reference, registers)

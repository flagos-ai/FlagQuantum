"""Independent acceptance for three native versions of commented walk references."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.quantum_walk.reference_circuit import (
    permutation,
    reference_circuit,
)
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.quantum_walk.reference_programs import (
    catalyst_commented_walk,
    qiskit_tket_walk,
)
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.quantum_walk.reference_rotation_basis import (
    rotation_basis,
)
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.quantum_walk.reference_workflows import (
    Source,
    Workflow,
    logical_state,
    run_workflow,
)
from flagquantum import synth
from flagquantum.compiler.basis_conversion import BasisConversionError
from flagquantum.errors import PlanningError
from tests.integration.synthesis_notebook_support import execute_code_cells

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "examples/synthesis/tutorials/technology_demonstrations/classiq_paper/quantum_walk"
)
NAMES: dict[Source, str] = {
    "qiskit": "qiskit_discrete_quantum_walk",
    "tket": "tket_discrete_quantum_walk",
    "catalyst": "pennylane_catalyst_discrete_quantum_walk",
}


@pytest.fixture(scope="module")
def workflows() -> dict[Source, Workflow]:
    return {source: run_workflow(source) for source in NAMES}


@pytest.mark.parametrize("source", ("qiskit", "catalyst"))
def test_all_128_columns_native_target_and_independent_circuit(
    source: Source, workflows: dict[Source, Workflow]
) -> None:
    workflow = workflows[source]
    program = workflow.program
    logical = tuple(wire for name in ("x", "coin") for wire in program.registers[name])
    reference = reference_circuit(catalyst=source == "catalyst").to_ir()
    matrix = reference.instructions[-1].matrix
    assert isinstance(matrix, torch.Tensor)
    # Evaluate all columns of the independent Circuit operator H then P.
    # Runtime execution of that Circuit is checked separately on complex input.
    for word in range(128):
        prefix = fq.Circuit(7, dtype=torch.complex128)
        for wire in range(7):
            if (word >> (6 - wire)) & 1:
                prefix.x(wire)
        instructions = prefix.to_ir().instructions
        expected = (matrix[:, word & ~1] + (-1) ** (word & 1) * matrix[:, word | 1]) / (
            2**0.5
        )
        native_prefix = tuple(
            replace(operation, wires=tuple(logical[wire] for wire in operation.wires))
            for operation in instructions
        )
        for ir in (program.circuit, workflow.executable):
            actual = logical_state(
                replace(ir, instructions=native_prefix + ir.instructions), program
            )
            torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)


def test_two_source_pipelines_share_the_verified_artifact(
    workflows: dict[Source, Workflow],
) -> None:
    qiskit, tket = workflows["qiskit"], workflows["tket"]
    assert qiskit.program.circuit.content_hash == tket.program.circuit.content_hash
    assert qiskit.executable.content_hash == tket.executable.content_hash
    assert qiskit.resources == tket.resources


@pytest.mark.parametrize("source", tuple(NAMES))
def test_complex_coherence_absolute_phase_and_actual_basis_cost(
    source: Source, workflows: dict[Source, Workflow]
) -> None:
    workflow = workflows[source]
    reference = reference_circuit(catalyst=source == "catalyst").to_ir()
    prefix = fq.Circuit(7, dtype=torch.complex128)
    for wire in range(7):
        prefix.h(wire).rz(wire, theta=(wire + 1) * 0.217)
    prefix.cx(0, 6)
    instructions = prefix.to_ir().instructions
    expected = (
        fq.run(replace(reference, instructions=instructions + reference.instructions))
        .statevector()
        .flatten()
    )
    logical = tuple(
        wire for name in ("x", "coin") for wire in workflow.program.registers[name]
    )
    native_prefix = tuple(
        replace(operation, wires=tuple(logical[wire] for wire in operation.wires))
        for operation in instructions
    )
    for ir in (workflow.program.circuit, workflow.executable):
        actual = logical_state(
            replace(ir, instructions=native_prefix + ir.instructions), workflow.program
        )
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
    gates = {"cx", "rz", "ry", "rx"} if source == "catalyst" else {"cx", "u3"}
    assert {
        instruction.name for instruction in workflow.executable.instructions
    } <= gates
    counts = workflow.resources.operation_counts
    assert counts["cx"] == sum(
        instruction.name == "cx" for instruction in workflow.executable.instructions
    )
    assert sum(counts.values()) == len(workflow.executable.instructions)
    assert workflow.resources.n_qubits <= 16 and workflow.resources.depth > 0
    assert workflow.synthesis_seconds >= 0 and workflow.lowering_seconds >= 0
    assert (
        workflow.elapsed_seconds
        >= workflow.synthesis_seconds + workflow.lowering_seconds
    )
    probabilities = dict(
        workflow.program.probabilities(
            fq.run(workflow.program), registers=("x", "coin")
        )
    )
    positions = {(1, 1), (3, 0)} if source == "catalyst" else {(1, 1), (63, 0)}
    assert set(probabilities) == positions
    assert all(
        value == pytest.approx(0.5, abs=1e-12) for value in probabilities.values()
    )


def test_source_loop_difference_and_explicit_phase_compensation(
    workflows: dict[Source, Workflow],
) -> None:
    a, b = permutation(catalyst=False), permutation(catalyst=True)
    assert sum(left != right for left, right in zip(a, b, strict=True)) == 32
    workflow = workflows["catalyst"]
    correction = workflow.executable.instructions[-1]
    assert correction.name == "rz"
    assert (
        correction.metadata["phase_compensation"] == "input_zero_output_zero_workspace"
    )
    logical = {wire for wires in workflow.program.registers.values() for wire in wires}
    assert set(correction.wires).isdisjoint(logical)
    actual = logical_state(workflow.executable, workflow.program)
    uncorrected = replace(
        workflow.executable, instructions=workflow.executable.instructions[:-1]
    )
    without = logical_state(uncorrected, workflow.program)
    assert float((actual - without).abs().max()) > 0.1
    torch.testing.assert_close(
        actual.abs().square(), without.abs().square(), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("source", tuple(NAMES))
def test_real_notebook_inline_program_and_clean_source(source: Source) -> None:
    path = BASE / NAMES[source] / "overview/walkthrough.ipynb"
    notebook = json.loads(path.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and not cell["outputs"]
    namespace, count = execute_code_cells(path)
    assert count == 3
    entry = namespace["walk"]
    assert isinstance(entry, type(qiskit_tket_walk))
    program = synth.synthesize(entry, max_qubits=16, max_workspace=9)
    expected = (
        fq.run(reference_circuit(catalyst=source == "catalyst")).statevector().flatten()
    )
    torch.testing.assert_close(
        logical_state(program.circuit, program), expected, atol=1e-11, rtol=0
    )


@pytest.mark.parametrize("source", tuple(NAMES))
def test_conversion_budget_refuses(source: Source) -> None:
    with pytest.raises(BasisConversionError):
        run_workflow(source, max_added_operations=0)


def test_rotation_expansion_budget_and_missing_clean_work_refuse(
    workflows: dict[Source, Workflow],
) -> None:
    with pytest.raises(ValueError, match="expansion"):
        rotation_basis(workflows["catalyst"].program, max_added_operations=1000)
    from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.quantum_walk.classiq_discrete_quantum_walk.overview.workflow import (
        fourier_program,
    )

    with pytest.raises(ValueError, match="clean work"):
        rotation_basis(fourier_program())


@pytest.mark.parametrize("catalyst", (False, True))
@pytest.mark.parametrize("budget", ("width", "operations", "depth", "memory"))
def test_synthesis_budgets_refuse(catalyst: bool, budget: str) -> None:
    entry = catalyst_commented_walk if catalyst else qiskit_tket_walk
    with pytest.raises((synth.SynthesisError, PlanningError)):
        if budget == "width":
            synth.synthesize(entry, max_qubits=6)
        elif budget == "operations":
            synth.synthesize(entry, max_operations=1)
        elif budget == "depth":
            synth.synthesize(entry, max_depth=1)
        else:
            synth.synthesize(entry, memory_limit_bytes=16)

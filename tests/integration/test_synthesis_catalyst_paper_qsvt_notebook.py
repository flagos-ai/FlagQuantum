"""Catalyst QSVT reconstruction: positive controls, big-endian addition and absolute phase."""

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.catalyst_qsvt.overview.comparisons.circuit import (
    reference_circuit,
    source_modular_adder,
)
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.catalyst_qsvt.overview.program import (
    PHASES,
    build_program,
    main,
)
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.catalyst_qsvt.overview.workflow import (
    Workflow,
    run_workflow,
)
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.classiq_qsvt.overview.workflow import (
    logical_state,
)
from flagquantum import synth
from flagquantum.compiler.basis_conversion import BasisConversionError
from flagquantum.errors import PlanningError
from tests.integration.synthesis_notebook_support import execute_code_cells

pytestmark = pytest.mark.integration
DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "examples/synthesis/tutorials/technology_demonstrations/classiq_paper/qsvt/catalyst_qsvt/overview"
)


@pytest.fixture(scope="module")
def workflow() -> Workflow:
    return run_workflow()


@pytest.mark.parametrize("constant", (2, 127))
def test_all_big_endian_adder_columns(constant: int) -> None:
    actual = source_modular_adder(constant)
    for word in range(128):
        column = torch.zeros(128, dtype=torch.complex128)
        column[(word + constant) % 128] = 1
        torch.testing.assert_close(actual[:, word], column, atol=1e-14, rtol=0)


def assert_prefix(
    workflow: Workflow, phases: tuple[float, ...], prefix: fq.Circuit
) -> None:
    reference = reference_circuit(phases).to_ir()
    instructions = prefix.to_ir().instructions
    expected = (
        fq.run(replace(reference, instructions=instructions + reference.instructions))
        .statevector()
        .flatten()
    )
    logical = tuple(
        w
        for name in ("block", "data", "qsvt_aux")
        for w in workflow.program.registers[name]
    )
    mapped = tuple(
        replace(op, wires=tuple(logical[w] for w in op.wires)) for op in instructions
    )
    for ir in (workflow.program.circuit, workflow.executable):
        actual = logical_state(
            replace(ir, instructions=mapped + ir.instructions), workflow.program
        )
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=0)


@pytest.mark.parametrize("word", (0, 1023, 126))
def test_zero_nonzero_and_boundary_inputs(workflow: Workflow, word: int) -> None:
    prefix = fq.Circuit(10, dtype=torch.complex128)
    for wire in range(10):
        if word & (1 << (9 - wire)):
            prefix.x(wire)
    assert_prefix(workflow, PHASES, prefix)


def coherent_prefix() -> fq.Circuit:
    prefix = fq.Circuit(10, dtype=torch.complex128)
    for wire in range(10):
        prefix.h(wire).rz(wire, 0.217 * (wire + 1))
    return prefix.cx(0, 9).cx(2, 8)


def test_complex_coherence_absolute_phase_clean_work_and_actual_costs(
    workflow: Workflow,
) -> None:
    assert_prefix(workflow, PHASES, coherent_prefix())
    assert workflow.resources.n_qubits <= 11
    work = workflow.program.report["workspace_qubits"]
    assert isinstance(work, int) and work <= 1
    assert {op.name for op in workflow.executable.instructions} <= {"cx", "u3"}
    assert sum(workflow.resources.operation_counts.values()) == len(
        workflow.executable.instructions
    )
    assert workflow.resources.depth > 0
    assert (
        workflow.elapsed_seconds
        >= workflow.synthesis_seconds + workflow.conversion_seconds
        > 0
    )
    probabilities = dict(
        workflow.program.probabilities(
            fq.run(workflow.program), registers=("qsvt_aux",)
        )
    )
    expected = (
        fq.run(reference_circuit())
        .statevector()
        .reshape(512, 2)
        .abs()
        .square()
        .sum(dim=0)
    )
    actual = torch.tensor(
        [probabilities.get((bit,), 0.0) for bit in range(2)], dtype=torch.float64
    )
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=0)


def test_positive_controls_and_big_endian_word_are_observable() -> None:
    from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.qiskit_qsvt.overview.comparisons.circuit import (
        reference_circuit as classiq_reference,
    )

    left = fq.run(reference_circuit()).statevector().flatten()
    right = fq.run(classiq_reference()).statevector().flatten()
    assert float((left - right).abs().max()) > 0.01


def test_second_nonzero_phase_binding() -> None:
    phases = (0.17, -0.29, 0.43, -0.61)
    alternative = run_workflow(phases=phases)
    assert_prefix(alternative, phases, coherent_prefix())


@pytest.mark.parametrize("budget", ("width", "operations", "depth", "memory"))
def test_native_resource_budgets_refuse(budget: str) -> None:
    # Keep calls explicit so each public keyword remains statically checked.
    with pytest.raises((synth.SynthesisError, PlanningError)):
        if budget == "width":
            synth.synthesize(
                main, classical_inputs={"phases": list(PHASES)}, max_qubits=9
            )
        elif budget == "operations":
            synth.synthesize(
                main, classical_inputs={"phases": list(PHASES)}, max_operations=1
            )
        elif budget == "depth":
            synth.synthesize(
                main, classical_inputs={"phases": list(PHASES)}, max_depth=1
            )
        else:
            synth.synthesize(
                main, classical_inputs={"phases": list(PHASES)}, memory_limit_bytes=16
            )


def test_target_conversion_budget_refuses() -> None:
    with pytest.raises(BasisConversionError):
        run_workflow(max_added_operations=0)


@pytest.mark.parametrize(
    "phases",
    (
        (0.1, 0.2, 0.3),
        (0.1, math.inf, 0.3, 0.4),
        (0.1, math.nan, 0.3, 0.4),
        (True, 0.2, 0.3, 0.4),
    ),
)
def test_phase_contract_refuses(phases: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="four finite real"):
        build_program(phases=phases)


def test_notebook_executes_inline_qfunc_and_keeps_source_outputs_empty() -> None:
    path = DIRECTORY / "walkthrough.ipynb"
    notebook = json.loads(path.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and not cell["outputs"]
    namespace, count = execute_code_cells(path)
    assert count == 3
    actual = namespace["workflow"]
    assert isinstance(actual, Workflow)
    assert actual.program.report["function"] == "catalyst_reference_qsvt"


def test_literal_zero_controlled_reflection_all_64_columns() -> None:
    """The target is the first data wire, matching the original source."""
    permutation = torch.eye(64, dtype=torch.complex128)
    permutation[:, 0] = 0
    permutation[:, 32] = 0
    permutation[32, 0] = 1
    permutation[0, 32] = 1
    reflection = fq.Circuit(6, dtype=torch.complex128)
    reflection.x(0).h(0).any(*range(6), matrix=permutation).h(0).x(0)
    body = reflection.to_ir()
    for word in range(64):
        prefix = fq.Circuit(6, dtype=torch.complex128)
        for wire in range(6):
            if word & (1 << (5 - wire)):
                prefix.x(wire)
        actual = (
            fq.run(
                replace(
                    body, instructions=prefix.to_ir().instructions + body.instructions
                )
            )
            .statevector()
            .flatten()
        )
        expected = torch.zeros(64, dtype=torch.complex128)
        expected[word] = -1 if word == 0 else 1
        torch.testing.assert_close(actual, expected, atol=1e-13, rtol=0)

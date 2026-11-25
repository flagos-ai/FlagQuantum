"""Full-state, exact-basis and source-workflow acceptance for native paper QSVT."""

import cmath
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.classiq_qsvt.overview.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.classiq_qsvt.overview.exact_basis import (
    exact_rules,
    rz_to_two_u3,
)
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.classiq_qsvt.overview.program import (
    PHASES,
    build_program,
    main,
)
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.classiq_qsvt.overview.workflow import (
    Workflow,
    logical_state,
    run_workflow,
)
from flagquantum import synth
from flagquantum.compiler.basis_conversion import BasisConversionError, convert_basis
from flagquantum.core.ir import Instruction
from flagquantum.errors import PlanningError
from tests.integration.synthesis_notebook_support import execute_code_cells

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = (
    ROOT
    / "examples/synthesis/tutorials/technology_demonstrations/classiq_paper/qsvt/classiq_qsvt/overview"
)


@pytest.fixture(scope="module")
def workflow() -> Workflow:
    return run_workflow()


@pytest.mark.parametrize(
    "angle",
    (-123.45, -math.tau, -math.pi, -0.001, 0.0, math.pi / 7, 1.9, math.tau, 100.0),
)
@pytest.mark.parametrize("state", ("zero", "one", "complex"))
def test_two_u3_rz_preserves_each_column_coherence_phase_and_metadata(
    angle: float, state: str
) -> None:
    prefix = fq.Circuit(1, dtype=torch.complex128)
    if state == "one":
        prefix.x(0)
    elif state == "complex":
        prefix.h(0).rz(0, 0.37)
    before = fq.run(prefix).statevector().flatten()
    expected = before * torch.tensor(
        [cmath.exp(-0.5j * angle), cmath.exp(0.5j * angle)], dtype=torch.complex128
    )
    source = Instruction(
        "rz", (0,), params={"theta": angle}, metadata={"source": "paper-qsvt"}
    )
    translated = rz_to_two_u3(source)
    assert all(op.name == "u3" and op.metadata == source.metadata for op in translated)
    ir = prefix.to_ir()
    actual = (
        fq.run(replace(ir, instructions=ir.instructions + translated))
        .statevector()
        .flatten()
    )
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=0)


def assert_input_matches_reference(
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


@pytest.mark.parametrize("block,data,aux", ((7, 63, 1), (0, 63, 0)))
def test_nonzero_block_boundary_data_and_auxiliary_inputs(
    workflow: Workflow, block: int, data: int, aux: int
) -> None:
    prefix = fq.Circuit(10, dtype=torch.complex128)
    word = (block << 7) | (data << 1) | aux
    for wire in range(10):
        if word & (1 << (9 - wire)):
            prefix.x(wire)
    assert_input_matches_reference(workflow, PHASES, prefix)


def coherent_prefix() -> fq.Circuit:
    prefix = fq.Circuit(10, dtype=torch.complex128)
    for wire in range(10):
        prefix.h(wire).rz(wire, 0.217 * (wire + 1))
    prefix.cx(0, 9).cx(2, 8)
    return prefix


def test_complex_coherence_absolute_phase_and_source_target_costs(
    workflow: Workflow,
) -> None:
    assert_input_matches_reference(workflow, PHASES, coherent_prefix())
    cost = workflow.resources
    assert cost.n_qubits <= 11
    work = workflow.program.report["workspace_qubits"]
    assert isinstance(work, int) and work <= 1
    assert {op.name for op in workflow.executable.instructions} <= {"cx", "u3"}
    assert cost.operation_counts["cx"] == sum(
        op.name == "cx" for op in workflow.executable.instructions
    )
    assert sum(cost.operation_counts.values()) == len(workflow.executable.instructions)
    assert (
        cost.depth > 0
        and workflow.synthesis_seconds > 0
        and workflow.conversion_seconds > 0
    )
    assert (
        workflow.elapsed_seconds
        >= workflow.synthesis_seconds + workflow.conversion_seconds
    )
    probabilities = dict(
        workflow.program.probabilities(
            fq.run(workflow.program), registers=("qsvt_aux",)
        )
    )
    assert sum(probabilities.values()) == pytest.approx(1, abs=1e-10)
    expected_state = fq.run(reference_circuit()).statevector().reshape(512, 2)
    expected_marginal = expected_state.abs().square().sum(dim=0)
    actual_marginal = torch.tensor(
        [probabilities.get((bit,), 0.0) for bit in range(2)], dtype=torch.float64
    )
    torch.testing.assert_close(actual_marginal, expected_marginal, atol=1e-10, rtol=0)


def test_second_nonzero_phase_sequence_uses_all_four_bindings() -> None:
    phases = (0.17, -0.29, 0.43, -0.61)
    alternative = run_workflow(phases=phases)
    assert_input_matches_reference(alternative, phases, coherent_prefix())
    reference = fq.run(reference_circuit(PHASES)).statevector().flatten()
    actual = logical_state(alternative.program.circuit, alternative.program)
    assert float((actual - reference).abs().max()) > 0.01


def test_notebook_executes_its_inline_program_and_keeps_source_clean() -> None:
    path = DIRECTORY / "walkthrough.ipynb"
    notebook = json.loads(path.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and not cell["outputs"]
    namespace, count = execute_code_cells(path)
    assert count == 3
    result = namespace["workflow"]
    assert isinstance(result, Workflow)
    assert result.program.report["function"] == "paper_qsvt"
    assert result.resources.n_qubits <= 11


@pytest.mark.parametrize("budget", ("width", "operations", "depth", "memory"))
def test_actual_synthesis_budgets_refuse(budget: str) -> None:
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


def test_target_conversion_budget_and_missing_exact_rule_refuse(
    workflow: Workflow,
) -> None:
    with pytest.raises(BasisConversionError):
        convert_basis(
            workflow.program.circuit,
            gates=("cx", "u3"),
            rules=exact_rules(),
            max_added_operations=0,
        )
    from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.quantum_walk.classiq_discrete_quantum_walk.overview.exact_basis import (
        exact_rules as walk_rules,
    )

    converted = convert_basis(
        workflow.program.circuit,
        gates=("cx", "u3"),
        rules=walk_rules(),
        max_added_operations=200_000,
    )
    torch.testing.assert_close(
        logical_state(converted.program, workflow.program),
        logical_state(workflow.program.circuit, workflow.program),
        atol=1e-10,
        rtol=0,
    )
    with pytest.raises(BasisConversionError, match="rz"):
        convert_basis(
            fq.Circuit(1).rz(0, theta=0.37),
            gates=("cx", "x"),
            rules={},
        )


@pytest.mark.parametrize(
    "phases",
    (
        (0.1, 0.2, 0.3),
        (0.1, 0.2, 0.3, 0.4, 0.5),
        (0.1, math.nan, 0.3, 0.4),
        (0.1, math.inf, 0.3, 0.4),
        (True, 0.2, 0.3, 0.4),
    ),
)
def test_phase_contract_refuses_before_compilation(phases: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="four finite real"):
        build_program(phases=phases)


def test_feasible_zero_workspace_budget_is_respected() -> None:
    program = synth.synthesize(
        main, classical_inputs={"phases": list(PHASES)}, max_qubits=11, max_workspace=0
    )
    assert program.report["workspace_qubits"] == 0
    assert program.circuit.n_wires == 10
    expected = fq.run(reference_circuit()).statevector().flatten()
    torch.testing.assert_close(
        logical_state(program.circuit, program), expected, atol=1e-10, rtol=0
    )

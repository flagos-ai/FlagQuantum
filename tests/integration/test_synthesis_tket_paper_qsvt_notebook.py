"""Literal Tket-reference QSVT: box order, half-turn rotations and absolute phase."""

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.classiq_qsvt.overview.workflow import (
    logical_state,
)
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.tket_qsvt.overview.comparisons.circuit import (
    reference_circuit,
    source_fourier_adder,
)
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.tket_qsvt.overview.program import (
    PHASES,
    build_program,
    classical_inputs,
    main,
)
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.tket_qsvt.overview.workflow import (
    Workflow,
    run_workflow,
)
from flagquantum import synth
from flagquantum.compiler.basis_conversion import BasisConversionError
from flagquantum.errors import PlanningError
from tests.integration.synthesis_notebook_support import execute_code_cells

pytestmark = pytest.mark.integration
DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "examples/synthesis/tutorials/technology_demonstrations/classiq_paper/qsvt/tket_qsvt/overview"
)


@pytest.fixture(scope="module")
def workflow() -> Workflow:
    return run_workflow()


@pytest.mark.parametrize("constant", (2, 127))
def test_literal_half_turn_adder_all_columns_and_nonmodular_contract(
    constant: int,
) -> None:
    actual = source_fourier_adder(constant)
    torch.testing.assert_close(
        actual.mH @ actual, torch.eye(128, dtype=torch.complex128), atol=1e-12, rtol=0
    )
    modular = torch.zeros_like(actual)
    for physical in range(128):
        word = int(f"{physical:07b}"[::-1], 2)
        encoded = int(f"{(word + constant) % 128:07b}"[::-1], 2)
        modular[encoded, physical] = 1
    assert float((actual - modular).abs().max()) > 0.1


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
    assert workflow.resources.n_qubits == 10
    work = workflow.program.report["workspace_qubits"]
    assert isinstance(work, int) and work == 0
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


def test_literal_box_mapping_is_observable() -> None:
    # A whole-wire permutation would also move projectors; only U boxes are remapped.
    literal = reference_circuit().to_ir()
    permutation = (3, 4, 5, 6, 7, 8, 0, 1, 2, 9)
    direct = replace(
        literal,
        instructions=tuple(
            (
                replace(op, wires=tuple(permutation[w] for w in op.wires))
                if len(op.wires) != 4
                else op
            )
            for op in literal.instructions
        ),
    )
    left = fq.run(literal).statevector().flatten()
    right = fq.run(direct).statevector().flatten()
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
            synth.synthesize(main, classical_inputs=classical_inputs(), max_qubits=9)
        elif budget == "operations":
            synth.synthesize(
                main, classical_inputs=classical_inputs(), max_operations=1
            )
        elif budget == "depth":
            synth.synthesize(main, classical_inputs=classical_inputs(), max_depth=1)
        else:
            synth.synthesize(
                main, classical_inputs=classical_inputs(), memory_limit_bytes=16
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
    assert actual.program.report["function"] == "tket_reference_qsvt"


def test_literal_reflection_phase_identity_all_64_columns() -> None:
    from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.tket_qsvt.overview.comparisons.circuit import (
        source_reflection,
    )

    h = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128) * 2**-0.5
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    h0 = torch.kron(h, torch.eye(32, dtype=torch.complex128))
    x0 = torch.kron(x, torch.eye(32, dtype=torch.complex128))
    h5 = torch.kron(torch.eye(32, dtype=torch.complex128), h)
    diagonal = torch.eye(64, dtype=torch.complex128)
    diagonal[1, 1] = -1
    expected = source_reflection()
    torch.testing.assert_close(
        x0 @ h0 @ h5 @ diagonal @ h5 @ h0 @ x0, expected, atol=1e-14, rtol=0
    )
    assert float((expected - torch.diag(torch.diag(expected))).abs().max()) > 0.1


@pytest.mark.parametrize("name", ("s", "sdg", "ch"))
def test_exact_phase_and_controlled_h_rules_all_columns(name: str) -> None:
    from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.tket_qsvt.overview.exact_basis import (
        exact_rules,
    )
    from flagquantum.compiler.basis_conversion import convert_basis

    source = fq.Circuit(2, dtype=torch.complex128)
    if name == "s":
        source.s(1)
    elif name == "sdg":
        source.sdg(1)
    else:
        source.ch(0, 1)
    native = source.to_ir()
    target = convert_basis(native, gates=("cx", "u3"), rules=exact_rules()).program
    for word in range(4):
        prefix = fq.Circuit(2, dtype=torch.complex128)
        for wire in range(2):
            if word & (1 << (1 - wire)):
                prefix.x(wire)
        instructions = prefix.to_ir().instructions
        expected = fq.run(
            replace(native, instructions=instructions + native.instructions)
        ).statevector()
        actual = fq.run(
            replace(target, instructions=instructions + target.instructions)
        ).statevector()
        torch.testing.assert_close(actual, expected, atol=1e-13, rtol=0)


@pytest.mark.parametrize("name", ("rx", "rz"))
@pytest.mark.parametrize(
    "angle", (0.0, math.pi / 2, math.pi, 2 * math.pi, -math.pi / 2, 0.37)
)
def test_exact_source_rotation_preparation_all_columns(name: str, angle: float) -> None:
    from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.qsvt.tket_qsvt.overview.exact_basis import (
        exact_rules,
        preserve_rotation_phases,
    )
    from flagquantum.compiler.basis_conversion import convert_basis
    from flagquantum.core.ir import Instruction

    native = replace(
        fq.Circuit(1, dtype=torch.complex128).to_ir(),
        instructions=(Instruction(name, (0,), params={"theta": angle}),),
    )
    target = convert_basis(
        preserve_rotation_phases(native), gates=("cx", "u3"), rules=exact_rules()
    ).program
    for word in range(2):
        prefix = fq.Circuit(1, dtype=torch.complex128)
        if word:
            prefix.x(0)
        before = prefix.to_ir().instructions
        expected = fq.run(
            replace(native, instructions=before + native.instructions)
        ).statevector()
        actual = fq.run(
            replace(target, instructions=before + target.instructions)
        ).statevector()
        torch.testing.assert_close(actual, expected, atol=1e-13, rtol=0)

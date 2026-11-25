"""Independent full-state acceptance of the complete six-bit walk Notebook."""

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.quantum_walk.classiq_discrete_quantum_walk.main import (
    main as fourier_main,
)
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.quantum_walk.classiq_discrete_quantum_walk.overview.arithmetic_candidate import (
    main as arithmetic_main,
)
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.quantum_walk.classiq_discrete_quantum_walk.overview.exact_basis import (
    exact_rules,
)
from examples.synthesis.tutorials.technology_demonstrations.classiq_paper.quantum_walk.classiq_discrete_quantum_walk.overview.workflow import (
    Workflow,
    run_workflow,
)
from flagquantum import synth
from flagquantum.compiler.basis_conversion import BasisConversionError, convert_basis
from flagquantum.compiler.resource_estimation import estimate_resources
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.errors import PlanningError
from tests.integration.synthesis_notebook_support import execute_code_cells

pytestmark = pytest.mark.integration
DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "examples/synthesis/tutorials/technology_demonstrations/classiq_paper/quantum_walk/classiq_discrete_quantum_walk/overview"
)


@pytest.fixture(scope="module")
def workflow() -> Workflow:
    return run_workflow()


def logical_state(ir: CircuitIR, program: synth.Program) -> torch.Tensor:
    logical = tuple(wire for name in ("x", "coin") for wire in program.registers[name])
    width = estimate_resources(program.circuit).n_qubits
    hidden = tuple(wire for wire in range(width) if wire not in logical)
    vector = fq.run(ir).statevector().flatten()
    ordered = vector.reshape((2,) * width).permute((*logical, *hidden)).reshape(128, -1)
    assert float(ordered[:, 1:].abs().square().sum()) < 1e-24
    state = ordered[:, 0]
    assert isinstance(state, torch.Tensor)
    return state


@pytest.mark.parametrize("candidate_name", ("fourier", "arithmetic"))
def test_every_logical_input_and_both_ir_bases(
    workflow: Workflow, candidate_name: str
) -> None:
    candidate = next(row for row in workflow.candidates if row.name == candidate_name)
    program = candidate.program
    logical = tuple(wire for name in ("x", "coin") for wire in program.registers[name])
    for position in range(64):
        for coin in range(2):
            prefix = tuple(
                Instruction("x", (wire,))
                for i, wire in enumerate(logical)
                if ((2 * position + coin) >> (6 - i)) & 1
            )
            expected = torch.zeros(128, dtype=torch.complex128)
            expected[2 * ((position + 1) % 64)] = 1 / math.sqrt(2)
            expected[2 * ((position - 1) % 64) + 1] = (-1) ** coin / math.sqrt(2)
            for ir in (program.circuit, candidate.conversion.program):
                actual = logical_state(
                    replace(ir, instructions=prefix + ir.instructions), program
                )
                torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)


def test_complex_coherent_input_and_actual_cost_selection(workflow: Workflow) -> None:
    for candidate in workflow.candidates:
        program = candidate.program
        logical = tuple(
            wire for name in ("x", "coin") for wire in program.registers[name]
        )
        prefix = fq.Circuit(candidate.resources.n_qubits, dtype=torch.complex128)
        for i, wire in enumerate(logical):
            prefix.h(wire)
            prefix.rz(wire, theta=0.13 * (i + 1))
        prefix.cx(logical[0], logical[-1])
        initial = logical_state(prefix.to_ir(), program)
        expected = torch.zeros_like(initial)
        for position in range(64):
            for coin in range(2):
                amplitude = initial[2 * position + coin] / math.sqrt(2)
                expected[2 * ((position + 1) % 64)] += amplitude
                expected[2 * ((position - 1) % 64) + 1] += (-1) ** coin * amplitude
        for ir in (program.circuit, candidate.conversion.program):
            actual = logical_state(
                replace(ir, instructions=prefix.to_ir().instructions + ir.instructions),
                program,
            )
            torch.testing.assert_close(actual, expected, atol=1e-11, rtol=0)
        resources = candidate.resources
        assert resources.n_qubits <= 12 and resources.depth > 0
        actual_cx = sum(
            instruction.name == "cx"
            for instruction in candidate.conversion.program.instructions
        )
        assert resources.operation_counts["cx"] == actual_cx
        assert {
            instruction.name
            for instruction in candidate.conversion.program.instructions
        } <= {"cx", "u3"}
        assert candidate.synthesis_seconds >= 0 and candidate.conversion_seconds >= 0
    ordered = sorted(
        workflow.candidates,
        key=lambda row: (
            row.resources.operation_counts["cx"],
            row.resources.depth,
            row.resources.n_qubits,
            row.name,
        ),
    )
    assert workflow.selected is ordered[0]
    assert len({row.resources.n_qubits for row in workflow.candidates}) == 2


IDENTITIES: list[tuple[str, int, tuple[int, ...], float | None]] = [
    (name, 1, (0,), None) for name in ("h", "x", "t", "tdg")
]
IDENTITIES += [
    ("phase", 1, (0,), angle) for angle in (0.0, math.pi, -math.pi, 0.17, -0.41)
]
IDENTITIES += [
    ("cphase", 2, (0, 1), angle) for angle in (0.0, math.pi, -math.pi, 0.17, -0.41)
]
IDENTITIES += [
    ("cphase", 2, (1, 0), 0.17),
    ("swap", 2, (0, 1), None),
    ("ccx", 3, (0, 1, 2), None),
]


@pytest.mark.parametrize("opcode,width,wires,angle", IDENTITIES)
def test_exact_rule_all_operator_columns(
    opcode: str, width: int, wires: tuple[int, ...], angle: float | None
) -> None:
    for word in range(1 << width):
        prefix = fq.Circuit(width, dtype=torch.complex128)
        for wire in range(width):
            if (word >> (width - 1 - wire)) & 1:
                prefix.x(wire)
        ir = prefix.to_ir()
        operation = Instruction(
            opcode, wires, params={} if angle is None else {"theta": angle}
        )
        source = replace(ir, instructions=ir.instructions + (operation,))
        converted = convert_basis(source, gates=("cx", "u3"), rules=exact_rules())
        torch.testing.assert_close(
            fq.run(source).statevector(),
            fq.run(converted.program).statevector(),
            atol=1e-12,
            rtol=0,
        )


@pytest.mark.parametrize(
    "opcode,width,wires",
    [("phase", 1, (0,)), ("cphase", 2, (0, 1)), ("cphase", 2, (1, 0))],
)
def test_phase_gradient_metadata_and_conversion_budget(
    opcode: str, width: int, wires: tuple[int, ...]
) -> None:
    theta = torch.tensor(0.217, dtype=torch.float64, requires_grad=True)
    prefix = fq.Circuit(width, dtype=torch.complex128)
    for wire in range(width):
        prefix.h(wire)
    ir = prefix.to_ir()
    source = replace(
        ir,
        instructions=ir.instructions
        + (
            Instruction(
                opcode, wires, params={"theta": theta}, metadata={"marker": "retained"}
            ),
        ),
    )
    converted = convert_basis(source, gates=("cx", "u3"), rules=exact_rules())
    a, b = (
        fq.run(source).statevector().flatten(),
        fq.run(converted.program).statevector().flatten(),
    )
    weights = torch.arange(1, a.numel() + 1, dtype=torch.float64)
    fa, fb = (weights * (a.real + 0.37 * a.imag)).sum(), (
        weights * (b.real + 0.37 * b.imag)
    ).sum()
    ga = torch.autograd.grad(fa, theta, retain_graph=True)[0]
    gb = torch.autograd.grad(fb, theta)[0]
    torch.testing.assert_close(ga, gb, atol=1e-12, rtol=0)
    assert all(
        instruction.metadata.get("marker") == "retained"
        for instruction in converted.program.instructions[len(ir.instructions) :]
    )
    if opcode == "cphase":
        with pytest.raises(BasisConversionError):
            convert_basis(
                source, gates=("cx", "u3"), max_added_operations=0, rules=exact_rules()
            )


def test_notebook_inline_qfunc_and_source_outputs() -> None:
    notebook = json.loads((DIRECTORY / "walkthrough.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and not cell["outputs"]
    namespace, count = execute_code_cells(DIRECTORY / "walkthrough.ipynb")
    assert count == 4
    entry = namespace["walk"]
    assert isinstance(entry, type(arithmetic_main))
    program = synth.synthesize(entry, max_qubits=12, max_workspace=5)
    expected = torch.zeros(128, dtype=torch.complex128)
    expected[2] = expected[127] = 1 / math.sqrt(2)
    torch.testing.assert_close(
        logical_state(program.circuit, program), expected, atol=1e-11, rtol=0
    )


@pytest.mark.parametrize("budget", ("width", "operations", "depth", "memory"))
def test_synthesis_budgets_refuse(budget: str) -> None:
    with pytest.raises((synth.SynthesisError, PlanningError)):
        if budget == "width":
            synth.synthesize(fourier_main, max_qubits=6)
        elif budget == "operations":
            synth.synthesize(fourier_main, max_operations=1)
        elif budget == "depth":
            synth.synthesize(fourier_main, max_depth=1)
        else:
            synth.synthesize(fourier_main, memory_limit_bytes=16)

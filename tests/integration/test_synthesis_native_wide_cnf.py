"""Native wide CNF preserves the full original and independent coherent phases."""

import ast
import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import NoReturn, TypedDict

import pytest
import torch

import flagquantum as fq
from examples.synthesis.tutorials.technology_demonstrations.oracle_generation.three_sat_oracles.comparisons.circuit import (
    CLAUSES,
    marked,
    product_signature,
    reference_circuit,
)
from examples.synthesis.tutorials.technology_demonstrations.oracle_generation.three_sat_oracles.main import (
    build_program,
    main,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate
from flagquantum.compiler._construction import cnf_phase
from flagquantum.compiler._construction.cnf_phase import Clause, Location
from flagquantum.compiler._construction.frontend_python import _QuantumFunction
from flagquantum.compiler._construction.selection import select_register_program
from flagquantum.compiler._hybrid.model import Value
from flagquantum.core.ir import Instruction
from flagquantum.ecosystem._qmod.source import load_qmod
from flagquantum.runtime.mps_result import native_mps
from flagquantum.simulation.mps.entrypoints import run_mps
from flagquantum.simulation.mps.state import MPSState
from tests.integration.test_synthesis_qmod_wide_cnf import _original
from tests.team.compiler.test_construction_python_program import program_definition

pytestmark = pytest.mark.integration
SOURCE = (
    Path(__file__).resolve().parents[2]
    / "examples/synthesis/tutorials/technology_demonstrations/oracle_generation/three_sat_oracles/main.py"
)


def test_complete_original_clause_identity_and_public_exact_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _original().read_bytes()
    clauses = tuple(
        tuple(map(int, clause))
        for clause in re.findall(
            r"\(\(qbv\[(\d+)\] or \(not qbv\[(\d+)\]\)\) or qbv\[(\d+)\]\)",
            raw.decode(),
        )
    )
    assert clauses == CLAUSES and len(clauses) == 67
    parsed = ast.parse(SOURCE.read_text())
    entry = next(node for node in parsed.body if isinstance(node, ast.FunctionDef))
    call = entry.body[1]
    assert isinstance(call, ast.Expr) and isinstance(call.value, ast.Call)
    condition = call.value.args[0]
    assert isinstance(condition, ast.BoolOp) and len(condition.values) == 67
    # Verify every native literal against the independent, pinned truth table.
    native_clauses = []
    for clause in condition.values:
        assert isinstance(clause, ast.BoolOp) and isinstance(clause.op, ast.Or)
        a, negative, c = clause.values
        assert isinstance(negative, ast.UnaryOp) and isinstance(negative.op, ast.Not)
        indices = []
        for literal in (a, negative.operand, c):
            assert isinstance(literal, ast.Subscript)
            assert isinstance(literal.slice, ast.Constant)
            indices.append(literal.slice.value)
        native_clauses.append(tuple(indices))
    assert tuple(native_clauses) == CLAUSES

    def forbid_dense(*args: object, **kwargs: object) -> NoReturn:
        pytest.fail("wide CNF materialized an exponential statevector")

    monkeypatch.setattr(MPSState, "to_statevector", forbid_dense)
    program = build_program()
    assert program.report["execution_profile"] == "exact_computational_basis_mps"
    assert program.report["workspace_qubits"] == 83
    assert program.circuit.n_wires == 150
    assert len(program.circuit.instructions) == 1033
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate) and resources.depth == 629
    result = fq.run(program)
    actual = native_mps(result)
    assert actual is not None and actual.max_bond == 1
    assert not actual.truncation_records
    assert product_signature(actual.tensors) == ((0,) * 150, -1 + 0j)
    assert result.plan and result.plan.state_bytes == 4800
    for enabled in ((), (29,), tuple(range(67))):
        # Also exercise a false clause, not only the original all-zero input.
        prefix = tuple(Instruction("x", (66 - bit,)) for bit in enabled)
        state = run_mps(
            replace(
                program.circuit, instructions=(*prefix, *program.circuit.instructions)
            )
        )
        expected = run_mps(reference_circuit(enabled))
        assert product_signature(state.tensors) == product_signature(expected.tensors)
        bits, phase = product_signature(state.tensors)
        assert bits[67:] == (0,) * 83
        assert phase == (-1 if marked(enabled) else 1)


def definition(
    predicate: str, *, prepare: str = "hadamard_transform(q)", width: int = 4
) -> _QuantumFunction:
    return program_definition(
        "from flagquantum.synth import *\n@qfunc\ndef main(q:Output[QArray[QBit]]):\n"
        f"    allocate({width},q)\n"
        + (f"    {prepare}\n" if prepare else "")
        + f"    control({predicate},lambda:phase(1,3.141592653589793))\n"
    )


@pytest.mark.parametrize("syntax", ["logical", "bitwise"])
@pytest.mark.parametrize("width", [4, 13])
def test_every_coherent_amplitude_tautology_duplicates_and_clean_scratch(
    syntax: str,
    width: int,
) -> None:
    predicate = (
        "(q[0] or not q[1] or q[0]) and (q[2] or q[1] or q[3]) and (q[3] or not q[3])"
    )
    if syntax == "bitwise":
        predicate = "(q[0]|~q[1]|q[0]) & (q[2]|q[1]|q[3]) & (q[3]|~q[3])"
    for bit in range(4, width):
        predicate += (
            f" and (q[{bit}] or not q[{bit}])"
            if syntax == "logical"
            else f" & (q[{bit}]|~q[{bit}])"
        )
    prepare = "\n    ".join(f"H(q[{bit}])" for bit in range(4))
    program = synth.synthesize(definition(predicate, prepare=prepare, width=width))
    actual = fq.run(program).statevector().reshape(1 << width, -1)
    reference = fq.Circuit(width, dtype=torch.complex128)
    for bit in range(4):
        reference.h(width - 1 - bit)
    signs = []
    for word in range(16):
        q0, q1, q2, q3 = (bool(word >> bit & 1) for bit in range(4))
        signs.append(-1 if (q0 or not q1) and (q2 or q1 or q3) else 1)
    reference.any(
        *range(width - 4, width),
        matrix=torch.diag(torch.tensor(signs, dtype=torch.complex128)),
    )
    torch.testing.assert_close(
        actual[:, 0], fq.run(reference).statevector().flatten(), atol=1e-12, rtol=0
    )
    assert float(actual[:, 1:].abs().square().sum()) < 1e-24


def test_all_tautologies_preserve_absolute_global_phase() -> None:
    program = synth.synthesize(definition("(q[0] or not q[0]) and (q[1] or not q[1])"))
    state = fq.run(program).statevector().flatten()
    torch.testing.assert_close(
        state, torch.full((16,), -0.25, dtype=torch.complex128), atol=1e-12, rtol=0
    )
    assert program.report["workspace_qubits"] == 0


def test_existing_small_bitwise_predicates_keep_their_wider_clause_support() -> None:
    program = synth.synthesize(
        definition("(q[0]|q[1]|q[2]|q[3]) & q[0]"), max_workspace=0
    )
    expected = torch.tensor(
        [-0.25 if word & 1 else 0.25 for word in range(16)], dtype=torch.complex128
    )
    torch.testing.assert_close(
        fq.run(program).statevector().flatten(), expected, atol=1e-12, rtol=0
    )


class _Limits(TypedDict, total=False):
    max_qubits: int
    max_workspace: int
    max_operations: int
    max_depth: int
    memory_limit_bytes: int


@pytest.mark.parametrize(
    "limits",
    [
        dict(max_qubits=149),
        dict(max_workspace=82),
        dict(max_operations=1032),
        dict(max_depth=628),
        dict(memory_limit_bytes=4799),
    ],
)
def test_actual_emitted_resources_refuse_before_execution(limits: _Limits) -> None:
    with pytest.raises(ValueError, match="budget|limit|requires|constraints"):
        synth.synthesize(main, **limits)


@pytest.mark.parametrize(
    "predicate",
    [
        "(q[4] or q[0]) and q[1]",
        "(q[0] or q[1] or q[2] or q[3]) and q[0]",
        " and ".join(["q[0]"] * 81),
    ],
)
def test_index_literal_and_clause_limits_refuse(predicate: str) -> None:
    with pytest.raises(ValueError):
        synth.synthesize(definition(predicate))


@pytest.mark.parametrize("body", ["", "allocate(4,q)\n    drop(q)"])
def test_tautology_cannot_erase_uninitialized_or_dropped_reads(body: str) -> None:
    entry = program_definition(
        "from flagquantum.synth import *\n@qfunc\ndef main(q:Output[QArray[QBit]]):\n"
        + (f"    {body}\n" if body else "")
        + "    control((q[0] or not q[0]) and (q[1] or not q[1]),lambda:phase(1,3.141592653589793))\n"
    )
    with pytest.raises(ValueError):
        synth.synthesize(entry)


def test_both_frontends_use_the_same_replaceable_clause_emitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = cnf_phase.emit_cnf_phase
    seen: list[int] = []

    def replacement(
        clauses: tuple[Clause, ...],
        scratch: Value,
        *,
        gate: Callable[[Location, str], None],
        controlled_x: Callable[[tuple[Location, ...]], None],
    ) -> None:
        seen.append(len(clauses))
        original(clauses, scratch, gate=gate, controlled_x=controlled_x)

    monkeypatch.setattr(cnf_phase, "emit_cnf_phase", replacement)
    native = build_program()
    source = load_qmod(_original())
    selected = select_register_program(
        source.program, source.functions, include_boolean=True
    ).selected
    assert selected is not None and selected.target is not None
    assert seen == [67, 67]
    assert native.circuit.instructions == selected.target.submitted.instructions

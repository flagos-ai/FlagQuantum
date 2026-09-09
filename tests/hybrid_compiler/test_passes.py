from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from flagquantum.compiler._hybrid import (
    HybridProgram,
    HybridVerificationError,
    Region,
    analyze_program,
    capture_source,
    lower_dynamic_program,
    run_pass_pipeline,
    scalar_type,
    specialize_and_lower,
)

pytestmark = pytest.mark.unit

STATIC_SOURCE = """
def optimized(theta):
    wire = 0 + 1
    dead = 4 + 5
    qp.RX(theta, wires=wire)
    return qp.expval(qp.PauliZ(1))
"""

CONTROL_FLOW_SOURCE = """
def simplified():
    wire = 0
    if True:
        wire = 1
        qp.X(wires=wire)
    else:
        wire = 2
        qp.X(wires=wire)
    for layer in range(0):
        wire = 3
        qp.X(wires=wire)
    qp.H(wires=wire)
    return qp.expval(qp.PauliZ(1))
"""

DYNAMIC_SOURCE = """
def optimized_feedback():
    wire = 0 + 1
    dead = 4 + 5
    if False:
        qp.X(wires=2)
    for layer in range(0):
        qp.X(wires=3)
    bit = qp.measure(wires=wire)
    if bit:
        qp.X(wires=0)
    return bit
"""


def _static_program() -> HybridProgram:
    return capture_source(STATIC_SOURCE, (scalar_type("float64"),))


def _operation_names(program: HybridProgram) -> tuple[str, ...]:
    names: list[str] = []

    def visit_region(region: Region) -> None:
        for block in region.blocks:
            for operation in block.operations:
                names.append(operation.name)
                for nested in operation.regions:
                    visit_region(nested)

    visit_region(program.body)
    return tuple(names)


def test_default_pipeline_folds_constants_and_removes_dead_constants() -> None:
    program = _static_program()

    result = run_pass_pipeline(program)

    assert result.source_identity == program.semantic_identity
    assert result.optimized_identity == result.program.semantic_identity
    assert result.changed is True
    assert tuple(record.name for record in result.records) == (
        "constant_fold",
        "structured_control_flow_simplification",
        "dead_constant_elimination",
    )
    assert result.records[0].changed is True
    assert result.records[1].changed is False
    assert result.records[2].input_operation_count == 9
    assert result.records[2].output_operation_count == 4
    assert analyze_program(result.program).operation_count == 4
    assert tuple(
        operation.name for operation in result.program.body.blocks[0].operations
    ) == (
        "arith.constant",
        "quantum.rx",
        "quantum.expectation",
        "program.return",
    )


def test_optimized_and_unoptimized_static_lowering_are_equivalent() -> None:
    program = _static_program()
    theta = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)

    optimized = specialize_and_lower(program, (theta,))
    baseline = specialize_and_lower(program, (theta,), optimize=False)

    assert optimized.bind() == baseline.bind()
    assert optimized.program_identity == program.semantic_identity
    assert optimized.optimized_program_identity != optimized.program_identity
    assert tuple(record.name for record in optimized.optimization_records) == (
        "constant_fold",
        "structured_control_flow_simplification",
        "dead_constant_elimination",
    )
    assert baseline.optimized_program_identity == baseline.program_identity
    assert baseline.optimization_records == ()
    optimized.bind().instructions[0].params["theta"].backward()
    assert theta.grad is not None
    assert theta.grad.item() == pytest.approx(1.0)


def test_default_pipeline_is_deterministic_and_idempotent() -> None:
    first = run_pass_pipeline(_static_program())
    repeated = run_pass_pipeline(_static_program())
    second = run_pass_pipeline(first.program)

    assert first == repeated
    assert second.optimized_identity == first.optimized_identity
    assert all(record.changed is False for record in second.records)


def test_constant_branches_and_empty_loops_preserve_carried_ssa_values() -> None:
    program = capture_source(CONTROL_FLOW_SOURCE, ())

    result = run_pass_pipeline(program)
    optimized = specialize_and_lower(program, ())
    baseline = specialize_and_lower(program, (), optimize=False)

    operations = result.program.body.blocks[0].operations
    assert analyze_program(program).operation_count == 19
    assert analyze_program(result.program).operation_count == 5
    assert all(operation.name not in {"scf.if", "scf.for"} for operation in operations)
    assert tuple((item.name, item.wires) for item in optimized.bind().instructions) == (
        ("x", (1,)),
        ("h", (1,)),
    )
    assert optimized.bind() == baseline.bind()


def test_each_pass_boundary_rejects_an_invalid_program() -> None:
    class InvalidPass:
        name = "remove_terminator"

        def run(self, program, analysis):
            del analysis
            entry = program.body.blocks[0]
            body = replace(
                program.body, blocks=(replace(entry, operations=entry.operations[:-1]),)
            )
            return replace(program, body=body)

    with pytest.raises(HybridVerificationError, match="terminator"):
        run_pass_pipeline(_static_program(), (InvalidPass(),))


def test_pipeline_rejects_invalid_passes_and_unbounded_sequences() -> None:
    class WrongResultPass:
        name = "wrong_result"

        def run(self, program, analysis):
            del program, analysis
            return object()

    with pytest.raises(TypeError, match="must return HybridProgram"):
        run_pass_pipeline(_static_program(), (WrongResultPass(),))
    with pytest.raises(TypeError, match="name and run method"):
        run_pass_pipeline(_static_program(), (object(),))
    with pytest.raises(ValueError, match="32-pass limit"):
        run_pass_pipeline(_static_program(), (WrongResultPass(),) * 33)


def test_dynamic_lowering_uses_the_same_verified_optimization_stage() -> None:
    program = capture_source(DYNAMIC_SOURCE, ())
    normalized = run_pass_pipeline(program).program

    optimized = lower_dynamic_program(program)
    baseline = lower_dynamic_program(program, optimize=False)

    assert _operation_names(normalized).count("scf.if") == 1
    assert "scf.for" not in _operation_names(normalized)
    assert optimized.circuit == baseline.circuit
    assert optimized.program_identity == program.semantic_identity
    assert optimized.optimized_program_identity != optimized.program_identity
    assert tuple(record.name for record in optimized.optimization_records) == (
        "constant_fold",
        "structured_control_flow_simplification",
        "dead_constant_elimination",
    )
    assert baseline.optimization_records == ()

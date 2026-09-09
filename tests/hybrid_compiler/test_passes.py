from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from flagquantum.compiler._hybrid import (
    BoundedLoopUnrollPass,
    HybridProgram,
    HybridVerificationError,
    Region,
    SpecializationError,
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

UNROLL_SOURCE = """
def bounded_unroll(theta, delta):
    angle = theta
    wire = 0
    for layer in range(3):
        angle = angle + delta
        wire = (wire + 1) % 2
        qp.RX(angle, wires=wire)
    return qp.expval(qp.PauliZ(0))
"""

DYNAMIC_UNROLL_SOURCE = """
def bounded_dynamic_unroll():
    for wire in range(3):
        qp.X(wires=wire)
    bit = qp.measure(wires=0)
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
        "bounded_loop_unroll",
        "constant_fold",
        "structured_control_flow_simplification",
        "dead_constant_elimination",
    )
    assert result.records[0].changed is False
    assert result.records[1].changed is True
    assert result.records[2].changed is False
    assert result.records[3].input_operation_count == 9
    assert result.records[3].output_operation_count == 4
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
        "bounded_loop_unroll",
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


def test_bounded_loop_unroll_preserves_parameters_gradients_and_budget() -> None:
    program = capture_source(
        UNROLL_SOURCE,
        (scalar_type("float64"), scalar_type("float64")),
    )
    theta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    delta = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)

    normalized = run_pass_pipeline(program)
    optimized = specialize_and_lower(program, (theta, delta))
    baseline = specialize_and_lower(program, (theta, delta), optimize=False)

    assert "scf.for" not in _operation_names(normalized.program)
    unroll_record = normalized.records[0]
    assert unroll_record.name == "bounded_loop_unroll"
    assert unroll_record.preexpanded_iterations == 3
    assert normalized.preexpanded_iterations == 3
    assert (
        run_pass_pipeline(normalized.program).optimized_identity
        == normalized.optimized_identity
    )
    assert tuple((item.name, item.wires) for item in optimized.bind().instructions) == (
        ("rx", (1,)),
        ("rx", (0,)),
        ("rx", (1,)),
    )
    assert tuple((item.name, item.wires) for item in baseline.bind().instructions) == (
        ("rx", (1,)),
        ("rx", (0,)),
        ("rx", (1,)),
    )
    optimized_parameters = torch.stack(
        [item.params["theta"] for item in optimized.bind().instructions]
    )
    baseline_parameters = torch.stack(
        [item.params["theta"] for item in baseline.bind().instructions]
    )
    assert torch.equal(optimized_parameters, baseline_parameters)
    optimized_parameters.sum().backward()
    assert theta.grad is not None and theta.grad.item() == pytest.approx(3.0)
    assert delta.grad is not None and delta.grad.item() == pytest.approx(6.0)
    with pytest.raises(SpecializationError, match="control.unroll_limit"):
        specialize_and_lower(program, (theta, delta), max_unrolled_iterations=2)


def test_unroll_budget_preserves_larger_loops_for_existing_lowering() -> None:
    program = capture_source(
        UNROLL_SOURCE,
        (scalar_type("float64"), scalar_type("float64")),
    )

    iteration_limited = run_pass_pipeline(
        program,
        (BoundedLoopUnrollPass(max_iterations=2),),
    )
    operation_limited = run_pass_pipeline(
        program,
        (BoundedLoopUnrollPass(max_expanded_operations=2),),
    )

    assert "scf.for" in _operation_names(iteration_limited.program)
    assert "scf.for" in _operation_names(operation_limited.program)
    assert iteration_limited.preexpanded_iterations == 0
    assert operation_limited.preexpanded_iterations == 0
    with pytest.raises(ValueError, match="max_iterations must be positive"):
        BoundedLoopUnrollPass(max_iterations=0)
    with pytest.raises(ValueError, match="max_expanded_operations must be positive"):
        BoundedLoopUnrollPass(max_expanded_operations=0)


def test_dynamic_unroll_preserves_circuit_and_original_iteration_limit() -> None:
    program = capture_source(DYNAMIC_UNROLL_SOURCE, ())

    optimized = lower_dynamic_program(program)
    baseline = lower_dynamic_program(program, optimize=False)

    assert optimized.circuit == baseline.circuit
    assert optimized.circuit.metadata["hybrid_dynamic_unrolled_iterations"] == 3
    assert baseline.circuit.metadata["hybrid_dynamic_unrolled_iterations"] == 3
    with pytest.raises(SpecializationError, match="control.unroll_limit"):
        lower_dynamic_program(program, max_unrolled_iterations=2)


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
        "bounded_loop_unroll",
        "constant_fold",
        "structured_control_flow_simplification",
        "dead_constant_elimination",
    )
    assert baseline.optimization_records == ()

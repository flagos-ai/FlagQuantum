from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
import torch

import flagquantum as fq
from flagquantum._compiler.testing.differential import (
    DifferentialStatus,
    execute_differential,
)

pytestmark = pytest.mark.unit


def _run_statevector(ir: fq.CircuitIR) -> fq.ExecutionResult:
    return fq.run(ir, options=fq.ExecutionOptions(mode="statevector"))


@pytest.mark.parametrize(
    ("dtype", "atol"),
    [("complex64", 1e-6), ("complex128", 1e-12)],
)
def test_state_expectation_dtype_and_device_parity(dtype: str, atol: float) -> None:
    source = fq.CircuitIR(
        2,
        (
            fq.Instruction("h", (0,)),
            fq.Instruction("cx", (0, 1)),
            fq.Instruction("rz", (1,), {"theta": 0.37}),
        ),
        dtype=dtype,
        measurements=(fq.MeasurementNode("expectation_z", (1, 0)),),
    )

    execution = execute_differential(
        source,
        _run_statevector,
        requested_backend="pytorch",
        requested_mode="statevector",
    )
    legacy = execution.legacy_result
    candidate = execution.candidate_result

    assert execution.report.ok
    assert execution.report.source_content_hash == source.content_hash
    assert execution.report.candidate_content_hash == source.content_hash
    assert execution.report.legacy.dtype == dtype
    assert execution.report.candidate.dtype == dtype
    assert execution.report.legacy.device == execution.report.candidate.device == "cpu"
    assert execution.report.legacy.mode == execution.report.candidate.mode
    torch.testing.assert_close(legacy.state, candidate.state, atol=atol, rtol=0)
    torch.testing.assert_close(
        legacy.expectation(), candidate.expectation(), atol=atol, rtol=0
    )


def test_seeded_measurement_and_result_ordering_are_exact() -> None:
    source = fq.CircuitIR(
        2,
        (fq.Instruction("h", (0,)), fq.Instruction("cx", (0, 1))),
        measurements=(
            fq.MeasurementNode(
                "sample", (1, 0), shots=32, metadata={"seed": 17, "name": "s"}
            ),
            fq.MeasurementNode(
                "counts", (0, 1), shots=32, metadata={"seed": 17, "name": "c"}
            ),
            fq.MeasurementNode("probabilities", (1, 0), metadata={"name": "p"}),
        ),
    )

    execution = execute_differential(source, _run_statevector)
    legacy = execution.legacy_result
    candidate = execution.candidate_result

    assert execution.report.ok
    assert execution.report.legacy.result_ordering == (
        ("sample", (1, 0), "s"),
        ("counts", (0, 1), "c"),
        ("probabilities", (1, 0), "p"),
    )
    assert execution.report.legacy.result_ordering == (
        execution.report.candidate.result_ordering
    )
    assert torch.equal(legacy.measurement("s").value, candidate.measurement("s").value)
    assert legacy.measurement("c").value == candidate.measurement("c").value
    torch.testing.assert_close(
        legacy.measurement("p").value, candidate.measurement("p").value
    )


def test_fixed_seed_property_circuits_have_state_parity() -> None:
    generator = random.Random(20260902)
    single_qubit = ("h", "x", "y", "z")
    for _ in range(24):
        instructions: list[fq.Instruction] = []
        for _ in range(generator.randint(1, 12)):
            if generator.random() < 0.25:
                first = generator.randrange(3)
                second = (first + generator.randrange(1, 3)) % 3
                instructions.append(fq.Instruction("cx", (first, second)))
            elif generator.random() < 0.35:
                instructions.append(
                    fq.Instruction(
                        "rx",
                        (generator.randrange(3),),
                        {"theta": generator.uniform(-3.0, 3.0)},
                    )
                )
            else:
                instructions.append(
                    fq.Instruction(
                        generator.choice(single_qubit), (generator.randrange(3),)
                    )
                )
        source = fq.CircuitIR(3, tuple(instructions))

        execution = execute_differential(source, _run_statevector)

        assert execution.report.ok
        torch.testing.assert_close(
            execution.legacy_result.state,
            execution.candidate_result.state,
            atol=1e-6,
            rtol=0,
        )


def test_custom_matrix_and_asymmetric_wire_order_lower_from_quantum_ir() -> None:
    cnot = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=torch.complex64,
    )
    source = fq.CircuitIR(
        2,
        (
            fq.Instruction("x", (1,)),
            fq.Instruction("custom_cnot", (1, 0), matrix=cnot),
        ),
    )

    execution = execute_differential(source, _run_statevector)

    assert execution.report.ok
    torch.testing.assert_close(
        execution.legacy_result.state,
        execution.candidate_result.state,
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        execution.candidate_result.state,
        torch.tensor([[0, 0, 0, 1]], dtype=torch.complex64),
    )


def test_import_and_execution_failures_are_reported_without_fallback() -> None:
    unsupported = fq.CircuitIR(
        1,
        (fq.Instruction("x", (0,), metadata={"is_dynamic": True}),),
    )
    calls = 0

    def must_not_run(_: fq.CircuitIR) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("unsupported source must not execute")

    rejected = execute_differential(unsupported, must_not_run)

    assert rejected.report.status is DifferentialStatus.IMPORT_FAILED
    assert rejected.report.diagnostics
    assert calls == 0

    source = fq.CircuitIR(1, (fq.Instruction("h", (0,)),))

    def fail_candidate(program: fq.CircuitIR) -> object:
        if program is not source:
            raise RuntimeError("candidate failed")
        return SimpleNamespace(
            state=torch.ones(1, 2, dtype=torch.complex64),
            measurements=(),
            runtime={"mode": "statevector", "backend": "pytorch", "fallback": False},
        )

    failed = execute_differential(source, fail_candidate)

    assert failed.report.status is DifferentialStatus.EXECUTION_FAILED
    assert failed.report.legacy.failure is None
    assert failed.report.legacy.fallback is False
    assert failed.report.candidate.failure == "RuntimeError: candidate failed"

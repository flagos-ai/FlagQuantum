"""ISSUE-081 generated correctness, mutation, fuzz and watchdog contracts."""

import importlib.util
import json
import math
import sys

import pytest
import torch

import flagquantum as fq
import flagquantum.backends as fqb
from flagquantum.compiler.operator_lowering import DEFAULT_LOWERING_REGISTRY
from flagquantum.core import OPERATOR_SCHEMAS, CircuitIR
from flagquantum.simulation import matrices
from flagquantum.testing import (
    FailureArtifact,
    PhaseAwareWatchdog,
    ProgressSnapshot,
    TolerancePolicy,
    certification_matrix,
    execute_certification_case,
    generate_circuit_ir,
)

pytestmark = pytest.mark.unit


def test_every_supported_operator_backend_pair_has_a_versioned_case():
    actual = {(case.backend, case.operator) for case in certification_matrix()}
    expected = {
        (backend, operator)
        for backend, values in DEFAULT_LOWERING_REGISTRY.manifest()["backends"].items()
        for operator in values["supported"]
    }
    assert actual == expected
    assert all(
        case.tolerance_version == "tolerance_v1" for case in certification_matrix()
    )


def test_every_certification_case_executes_its_backend_surface():
    cases = certification_matrix()
    if importlib.util.find_spec("jax") is None:
        cases = tuple(case for case in cases if case.backend != "jax")
    results = [execute_certification_case(case) for case in cases]
    assert results
    assert all(result.executed and result.passed for result in results)


@pytest.mark.parametrize("seed", range(8))
def test_generated_ir_is_deterministic_normalized_and_round_trips(seed):
    first = generate_circuit_ir(seed=seed, depth=10)
    second = generate_circuit_ir(seed=seed, depth=10)
    assert first.content_hash == second.content_hash
    assert CircuitIR.from_json(first.to_json()) == first
    state = fq.Circuit.from_ir(first).state()
    assert torch.allclose(torch.linalg.vector_norm(state), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(state.abs().square().sum(), torch.tensor(1.0), atol=1e-5)


def test_dense_mps_and_tensor_network_are_differentially_equal():
    ir = generate_circuit_ir(seed=811, depth=7)
    dense = fq.Circuit.from_ir(ir).state()
    mps = fqb.run_mps(ir, max_bond=None, cutoff=0.0).to_statevector()
    tensor_network = fqb.run_tensor_network(ir).state()
    assert torch.allclose(mps, dense, atol=1e-5, rtol=1e-5)
    assert torch.allclose(tensor_network, dense, atol=1e-5, rtol=1e-5)


def test_all_fixed_unitary_matrices_are_unitary():
    for opcode, schema in OPERATOR_SCHEMAS.items():
        matrix = matrices.GATE_MAT_DICT.get(opcode)
        if not schema.unitary or not isinstance(matrix, torch.Tensor):
            continue
        identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype)
        assert torch.allclose(matrix.mH @ matrix, identity, atol=1e-6), opcode


def test_gradient_matches_parameter_shift_and_is_deterministic():
    theta = torch.tensor(0.31, requires_grad=True)
    value = fq.Circuit(1).ry(0, theta).expectation_z(0)
    value.backward()
    expected = -math.sin(float(theta.detach()))
    assert theta.grad is not None
    assert float(theta.grad) == pytest.approx(expected, abs=1e-5)
    assert float(value.detach()) == pytest.approx(math.cos(0.31), abs=1e-5)


def test_tolerance_is_versioned_by_dtype_method_depth_and_approximation():
    policy = TolerancePolicy()
    assert policy.accepts(
        dtype="complex64", method="exact", depth=16, approximation="none"
    )
    assert not policy.accepts(
        dtype="complex128", method="exact", depth=16, approximation="none"
    )
    assert not policy.accepts(
        dtype="complex64", method="mps", depth=16, approximation="none"
    )
    assert not policy.accepts(
        dtype="complex64", method="exact", depth=17, approximation="none"
    )
    assert not policy.accepts(
        dtype="complex64", method="exact", depth=16, approximation="truncated"
    )


def test_failure_artifact_retains_minimal_reproducer():
    ir = generate_circuit_ir(seed=42, depth=2)
    artifact = FailureArtifact.capture(
        seed=42, backend="pytorch", property_name="norm", ir=ir, error=ValueError("bad")
    )
    payload = json.loads(artifact.to_json())
    assert payload["seed"] == 42 and payload["backend"] == "pytorch"
    assert CircuitIR.from_dict(payload["ir"]) == ir
    assert payload["environment"]["python"]


def test_failure_artifact_delta_debugs_to_one_instruction():
    ir = generate_circuit_ir(seed=43, depth=6)
    failing_name = ir.instructions[-1].name
    artifact = FailureArtifact.capture(
        seed=43,
        backend="pytorch",
        property_name="synthetic",
        ir=ir,
        error=ValueError("bad"),
        fails=lambda candidate: any(
            item.name == failing_name for item in candidate.instructions
        ),
    )
    minimized = CircuitIR.from_dict(json.loads(artifact.to_json())["ir"])
    assert len(minimized.instructions) == 1
    assert minimized.instructions[0].name == failing_name


def test_real_watchdog_terminates_stalled_process_and_writes_cleanup_diagnosis(
    tmp_path,
):
    from tools.run_with_watchdog import run

    diagnostics = tmp_path / "watchdog"
    status = run(
        [
            "--phase",
            "execute",
            "--stall-seconds",
            "0.2",
            "--timeout-seconds",
            "2",
            "--diagnostics",
            str(diagnostics),
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ]
    )
    payload = json.loads((diagnostics / "watchdog.json").read_text())
    assert status == 124
    assert payload["reason"] == "no_progress_timeout"
    assert payload["cleanup_required"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: {**data, "version": "999"},
        lambda data: {**data, "n_wires": 0},
        lambda data: {
            **data,
            "instructions": [{"name": "unknown", "wires": [0], "params": {}}],
        },
        lambda data: {**data, "unexpected": True},
    ],
)
def test_ir_parser_fuzz_mutations_fail_closed(mutation):
    payload = generate_circuit_ir(seed=7, depth=1).to_dict()
    with pytest.raises((ValueError, TypeError)):
        CircuitIR.from_dict(mutation(payload))


def _snapshot(time, **overrides):
    values = {
        "timestamp": time,
        "phase": "execute",
        "operation": "all_reduce",
        "completed_units": 0,
        "memory_bytes": 1024,
        "collective": "all_reduce",
        "rank": 1,
    }
    values.update(overrides)
    return ProgressSnapshot(**values)


def test_watchdog_classifies_collective_stall_and_requires_cleanup():
    watchdog = PhaseAwareWatchdog(stall_seconds=5)
    assert watchdog.observe(_snapshot(0)) is None
    diagnosis = watchdog.observe(_snapshot(6))
    assert diagnosis.cause == "collective_participant_or_transport_stall"
    assert diagnosis.last_operation == "all_reduce"
    assert diagnosis.cleanup_required


def test_watchdog_allows_bounded_compile_and_checkpoint_without_false_positive():
    watchdog = PhaseAwareWatchdog(
        stall_seconds=5, bounded_phase_seconds={"compile": 30, "checkpoint": 20}
    )
    assert watchdog.observe(_snapshot(0, phase="compile", collective="none")) is None
    assert watchdog.observe(_snapshot(25, phase="compile", collective="none")) is None
    assert watchdog.observe(_snapshot(26, phase="execute", completed_units=1)) is None


@pytest.mark.parametrize(
    ("overrides", "cause"),
    [
        ({"collective": "none", "phase": "input"}, "stalled_input"),
        (
            {"collective": "none", "memory_bytes": 2048},
            "memory_growth_without_progress",
        ),
        (
            {"collective": "none", "useful_work_launched": False},
            "memory_owner_without_useful_work",
        ),
        (
            {"collective": "none", "operation": "backward"},
            "rank_desynchronization_or_no_progress",
        ),
    ],
)
def test_watchdog_fault_matrix(overrides, cause):
    watchdog = PhaseAwareWatchdog(stall_seconds=2)
    initial = dict(overrides)
    if cause == "memory_growth_without_progress":
        initial["memory_bytes"] = 1024
    assert watchdog.observe(_snapshot(0, **initial)) is None
    assert watchdog.observe(_snapshot(3, **overrides)).cause == cause

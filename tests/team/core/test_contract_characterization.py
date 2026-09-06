from __future__ import annotations

import json

import pytest

from flagquantum.core._artifacts import ArtifactKind, ProgramArtifact
from flagquantum.core.contracts import (
    CapabilityContract,
    ContractVersionError,
    EstimatedResources,
    RequestedExecution,
    RuntimePlanContract,
    UnknownContractFieldError,
)
from flagquantum.core.ir import (
    CircuitIR,
    Instruction,
    IRSerializationError,
    IRValidationError,
)
from flagquantum.errors import ValidationError
from flagquantum.runtime.options import ExecutionOptions
from flagquantum.runtime.result import ExecutionResult

pytestmark = pytest.mark.unit


def test_circuit_ir_characterizes_canonical_round_trip_and_strict_reads() -> None:
    ir = CircuitIR(
        n_wires=2,
        instructions=(Instruction("h", (0,)), Instruction("cx", (0, 1))),
        metadata={"z": 2, "a": 1},
    )
    reordered = CircuitIR(
        n_wires=2,
        instructions=ir.instructions,
        metadata={"a": 1, "z": 2},
    )

    restored = CircuitIR.from_json(ir.to_json())

    assert restored.to_dict() == ir.to_dict()
    assert restored.content_hash == ir.content_hash == reordered.content_hash
    with pytest.raises(IRSerializationError, match="unknown field"):
        CircuitIR.from_dict({**ir.to_dict(), "future": True})
    with pytest.raises(IRValidationError, match="unsupported IR version"):
        CircuitIR.from_dict({**ir.to_dict(), "version": "2.0"})


def test_program_artifact_characterizes_hash_round_trip_and_strict_reads() -> None:
    ir = CircuitIR(n_wires=1, instructions=(Instruction("h", (0,)),))
    artifact = ProgramArtifact(
        kind=ArtifactKind.CIRCUIT,
        payload=ir.to_dict(),
        producer="characterization",
        metadata={"z": 2, "a": 1},
    )
    reordered = ProgramArtifact(
        kind=ArtifactKind.CIRCUIT,
        payload=ir.to_dict(),
        producer="characterization",
        metadata={"a": 1, "z": 2},
    )

    restored = ProgramArtifact.from_dict(json.loads(json.dumps(artifact.to_dict())))

    assert restored == artifact
    assert restored.content_hash == artifact.content_hash == reordered.content_hash
    with pytest.raises(ValueError, match="unknown program artifact field"):
        ProgramArtifact.from_dict({**artifact.to_dict(), "future": True})
    with pytest.raises(ValueError, match="unsupported artifact envelope version"):
        ProgramArtifact.from_dict({**artifact.to_dict(), "version": "2.0"})


def test_core_runtime_plan_characterizes_nested_strictness_and_hashing() -> None:
    plan = RuntimePlanContract(
        requested=RequestedExecution(world_size=2),
        estimated=EstimatedResources(memory_bytes=1024, communication_bytes=128),
        capability=CapabilityContract(supports_distributed=True),
        plan_id="plan-characterization",
    )
    payload = json.loads(plan.to_json())

    restored = RuntimePlanContract.from_dict(payload)

    assert restored == plan
    assert restored.content_hash == plan.content_hash
    with pytest.raises(UnknownContractFieldError, match="future"):
        RuntimePlanContract.from_dict({**payload, "future": True})
    nested = dict(payload["requested"])
    nested["version"] = "2.0"
    with pytest.raises(ContractVersionError, match="unsupported requested_execution"):
        RuntimePlanContract.from_dict({**payload, "requested": nested})


def test_execution_options_characterizes_versioned_strict_round_trip() -> None:
    options = ExecutionOptions(
        mode="statevector",
        precision="complex128",
        require_gradients=True,
    )
    payload = options.to_dict()

    assert ExecutionOptions.from_dict(payload) == options
    with pytest.raises(ValidationError, match="unknown execution options"):
        ExecutionOptions.from_dict({**payload, "future": True})
    with pytest.raises(ValidationError, match="unsupported execution options version"):
        ExecutionOptions.from_dict({**payload, "version": "2.0"})


def test_execution_result_summary_keeps_core_fields_authoritative() -> None:
    result = ExecutionResult(
        metrics={"elapsed_seconds": 0.25},
        provenance={"producer": "characterization"},
        runtime={"schema": "runtime-owned-value", "world_size": 1},
    )

    summary = result.summary()
    diagnostics = result.diagnostics()

    assert summary["schema"] == "flagquantum.execution_result.summary"
    assert summary["version"] == "1.0"
    assert summary["world_size"] == 1
    assert summary["runtime_projection_conflicts"] == ("schema",)
    assert diagnostics == {
        "schema": "flagquantum.execution_diagnostics",
        "version": "1.0",
        "metrics": {"elapsed_seconds": 0.25},
        "provenance": {"producer": "characterization"},
        "runtime": {"schema": "runtime-owned-value", "world_size": 1},
        "compatibility": {},
    }

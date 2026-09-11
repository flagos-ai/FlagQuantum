from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import torch

from flagquantum.compiler.native_gate_legalization import (
    NativeGateLegalizationError,
    legalize_native_gates,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.target_capabilities import (
    CapabilityFact,
    CapabilityScope,
    EvidenceLevel,
    EvidenceReference,
    FactExposure,
    FactSource,
    SupportStatus,
    TargetCapabilitySnapshot,
    TargetIdentity,
)
from flagquantum.simulation.statevector.local import run_local_statevector

pytestmark = pytest.mark.integration

_NOW = datetime(2026, 9, 10, 8, 0, tzinfo=timezone.utc)
_SCOPE = CapabilityScope(device_ids=("native:0",))
_SOURCE = FactSource(kind="phase26_test", ref="phase26-evidence")


def _snapshot(native_gates: tuple[object, ...]) -> TargetCapabilitySnapshot:
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id="phase26-target",
            target_class="test",
            provider="flagquantum.test",
            provider_version="1",
            target_revision="1",
            environment_id="phase26-environment",
        ),
        scope=_SCOPE,
        captured_at=_NOW.isoformat(),
        valid_until=(_NOW + timedelta(hours=1)).isoformat(),
        facts=(
            CapabilityFact(
                name="gates.native",
                value=native_gates,
                support_status=SupportStatus.VERIFIED,
                fact_exposure=FactExposure.DECLARED,
                source=_SOURCE,
            ),
        ),
        evidence_refs=(
            EvidenceReference(
                evidence_id="phase26-evidence",
                sha256="b" * 64,
                level=EvidenceLevel.BASIC,
                scope=_SCOPE,
            ),
        ),
    )


def _basis_snapshot() -> TargetCapabilitySnapshot:
    return _snapshot(
        (
            {"name": "h", "parameters": ()},
            {"name": "rz", "parameters": ("theta",)},
            {"name": "s", "parameters": ()},
            {"name": "sdg", "parameters": ()},
        )
    )


def _state(program: CircuitIR) -> torch.Tensor:
    return run_local_statevector(
        program,
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.complex128,
    )


def test_parameterized_rotations_decompose_to_evidenced_native_basis() -> None:
    theta = torch.tensor(0.31, dtype=torch.float64, requires_grad=True)
    phi = torch.tensor(-0.27, dtype=torch.float64, requires_grad=True)
    metadata = {"conditions": ((0, 1),)}
    source = CircuitIR(
        1,
        (
            Instruction("rx", (0,), params={"theta": theta}, metadata=metadata),
            Instruction("ry", (0,), params={"theta": phi}, metadata=metadata),
        ),
        dtype="complex128",
    )
    result = legalize_native_gates(
        source, snapshot=_basis_snapshot(), evaluated_at=_NOW
    )

    assert tuple(item.name for item in result.program.instructions) == (
        "h",
        "rz",
        "h",
        "sdg",
        "h",
        "rz",
        "h",
        "s",
    )
    assert result.program.instructions[1].params["theta"] is theta
    assert result.program.instructions[5].params["theta"] is phi
    assert all(item.metadata == metadata for item in result.program.instructions)
    assert result.decompositions[0].replacement_opcodes == ("h", "rz", "h")
    assert result.decompositions[1].replacement_opcodes == (
        "sdg",
        "h",
        "rz",
        "h",
        "s",
    )
    torch.testing.assert_close(_state(result.program), _state(source))


def test_native_program_preserves_the_exact_circuit_instance() -> None:
    source = CircuitIR(1, (Instruction("h", (0,)),), dtype="complex128")
    snapshot = _snapshot(("h",))
    first = legalize_native_gates(source, snapshot=snapshot, evaluated_at=_NOW)
    second = legalize_native_gates(source, snapshot=snapshot, evaluated_at=_NOW)

    assert first.program is source
    assert first.source_program is source
    assert first.changed is False
    assert first.decompositions == ()
    assert first.legalization_identity == second.legalization_identity


def test_descriptor_parameter_mismatch_uses_verified_decomposition() -> None:
    theta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    source = CircuitIR(
        1,
        (Instruction("rx", (0,), params={"theta": theta}),),
        dtype="complex128",
    )
    snapshot = _snapshot(
        (
            {"name": "rx", "parameters": ()},
            {"name": "h", "parameters": ()},
            {"name": "rz", "parameters": ("theta",)},
        )
    )
    result = legalize_native_gates(source, snapshot=snapshot, evaluated_at=_NOW)

    assert tuple(item.name for item in result.program.instructions) == ("h", "rz", "h")
    assert result.changed is True


def test_missing_decomposition_basis_fails_closed() -> None:
    source = CircuitIR(
        1,
        (Instruction("rx", (0,), params={"theta": 0.2}),),
        dtype="complex128",
    )
    with pytest.raises(
        NativeGateLegalizationError, match="requires unsupported native gate 'rz'"
    ):
        legalize_native_gates(
            source,
            snapshot=_snapshot(("h",)),
            evaluated_at=_NOW,
        )


def test_separate_native_descriptor_variants_do_not_merge_parameters() -> None:
    source = CircuitIR(
        1,
        (
            Instruction(
                "u2",
                (0,),
                params={"phi": 0.1, "lbd": 0.2},
            ),
        ),
        dtype="complex128",
    )
    snapshot = _snapshot(
        (
            {"name": "u2", "parameters": ("phi",)},
            {"name": "u2", "parameters": ("lbd",)},
        )
    )
    with pytest.raises(NativeGateLegalizationError, match="no verified decomposition"):
        legalize_native_gates(source, snapshot=snapshot, evaluated_at=_NOW)


def test_decomposition_expansion_limit_fails_closed() -> None:
    source = CircuitIR(1, (Instruction("x", (0,)),), dtype="complex128")
    with pytest.raises(NativeGateLegalizationError, match="max_added_operations"):
        legalize_native_gates(
            source,
            snapshot=_snapshot(("h", "z")),
            evaluated_at=_NOW,
            max_added_operations=1,
        )


def test_stale_or_missing_native_gate_evidence_fails_closed() -> None:
    source = CircuitIR(1, (Instruction("h", (0,)),), dtype="complex128")
    with pytest.raises(NativeGateLegalizationError, match="snapshot_stale"):
        legalize_native_gates(
            source,
            snapshot=_snapshot(("h",)),
            evaluated_at=_NOW + timedelta(hours=2),
        )

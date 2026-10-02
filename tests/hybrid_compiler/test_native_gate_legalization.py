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


def _z_sx_snapshot(*extra: object) -> TargetCapabilitySnapshot:
    """An IBM-style basis: a z-rotation, `sx`, and whatever else is passed."""

    return _snapshot(
        (
            {"name": "rz", "parameters": ("theta",)},
            {"name": "sx", "parameters": ()},
        )
        + extra
    )


def _one_qubit_group() -> tuple[Instruction, ...]:
    """One instruction of every declared single-qubit unitary opcode."""

    from flagquantum.core.operator_schema import OPERATOR_SCHEMAS

    angles = {"theta": 0.7137, "phi": -0.4211, "lbd": 1.9073}
    return tuple(
        Instruction(
            name,
            (0,),
            params={item: angles[item] for item in schema.parameters},
        )
        for name, schema in OPERATOR_SCHEMAS.items()
        if schema.unitary and schema.arity == 1
    )


def test_a_z_rotation_and_sx_basis_reaches_the_whole_one_qubit_group() -> None:
    """The basis the plan calls a gap now legalizes every single-qubit opcode."""

    group = _one_qubit_group()
    source = CircuitIR(1, group, dtype="complex128")
    result = legalize_native_gates(source, snapshot=_z_sx_snapshot(), evaluated_at=_NOW)

    # Non-vacuity: `rz` is the basis's own z-rotation and `sx` is native, so 18
    # declared opcodes yield exactly 16 records, one per rewritten opcode.
    assert len(group) == 18
    rewritten = tuple(item.source_opcode for item in result.decompositions)
    assert len(rewritten) == 16
    assert set(rewritten) == {
        "i",
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
        "sxdg",
        "rx",
        "ry",
        "phase",
        "u1",
        "u2",
        "u3",
    }
    assert {item.name for item in result.program.instructions} <= {"rz", "sx"}
    # `i` is the one empty replacement: a legal answer that removes a gate.
    assert [
        item.source_opcode
        for item in result.decompositions
        if not item.replacement_opcodes
    ] == ["i"]
    assert max(len(item.replacement_opcodes) for item in result.decompositions) == 5


def test_the_legalized_state_matches_up_to_one_global_phase() -> None:
    """A z-rotation and `sx` basis cannot match `h` exactly, only up to a phase.

    The phase is a real, measurable property of the basis: this test pins both
    that the states agree up to a phase and that the raw difference is not zero,
    so a future change that starts preserving the phase cannot pass silently
    either.
    """

    source = CircuitIR(
        2,
        (
            Instruction("h", (0,)),
            Instruction("cx", (0, 1)),
            Instruction("rx", (0,), params={"theta": 0.7137}),
            Instruction("u3", (1,), params={"theta": -1.2, "phi": 0.4, "lbd": 2.1}),
            Instruction("sdg", (0,)),
            Instruction("y", (1,)),
            Instruction("cx", (1, 0)),
            Instruction("t", (1,)),
        ),
        dtype="complex128",
    )
    result = legalize_native_gates(
        source,
        snapshot=_z_sx_snapshot({"name": "cx", "parameters": ()}),
        evaluated_at=_NOW,
    )
    assert result.changed is True

    original = _state(source).reshape(-1)
    legalized = _state(result.program).reshape(-1)
    overlap = torch.vdot(original, legalized)
    assert float(torch.abs(overlap)) == pytest.approx(1.0, abs=1e-12)
    assert float(torch.max(torch.abs(legalized - original))) > 0.5
    assert float(torch.abs(torch.angle(overlap))) > 1e-3


def test_an_exact_rewrite_still_wins_over_the_general_synthesis() -> None:
    """Where the target's basis allows an exact form, the exact form is used."""

    source = CircuitIR(1, (Instruction("x", (0,)),), dtype="complex128")
    exact = legalize_native_gates(
        source, snapshot=_snapshot(("h", "z")), evaluated_at=_NOW
    )
    synthesized = legalize_native_gates(
        source, snapshot=_z_sx_snapshot(), evaluated_at=_NOW
    )

    assert tuple(item.name for item in exact.program.instructions) == ("h", "z", "h")
    assert tuple(item.name for item in synthesized.program.instructions) == (
        "sx",
        "sx",
    )
    assert _state(source).shape == _state(synthesized.program).shape


def test_a_trainable_rotation_stays_in_the_autograd_graph() -> None:
    """A trainable angle is never treated as a branch-selecting constant."""

    theta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    source = CircuitIR(
        1, (Instruction("ry", (0,), params={"theta": theta}),), dtype="complex128"
    )
    result = legalize_native_gates(source, snapshot=_z_sx_snapshot(), evaluated_at=_NOW)

    carried = [
        instruction.params["theta"]
        for instruction in result.program.instructions
        if instruction.name == "rz"
    ]
    assert carried, "the general form must keep a z-rotation carrying theta"
    assert any(
        isinstance(angle, torch.Tensor) and angle.requires_grad for angle in carried
    )


def test_a_rotation_only_basis_is_no_weaker_than_an_sx_basis() -> None:
    """`RX(pi/2)` is `SX`, so a basis of rotations alone reaches the same group.

    Before Euler synthesis this basis legalized 2 of the 18 declared single-qubit
    unitary opcodes: its own `rz` and `rx`.
    """

    group = _one_qubit_group()
    source = CircuitIR(1, group, dtype="complex128")
    result = legalize_native_gates(
        source,
        snapshot=_snapshot(
            (
                {"name": "rz", "parameters": ("theta",)},
                {"name": "rx", "parameters": ("theta",)},
                {"name": "cz", "parameters": ()},
            )
        ),
        evaluated_at=_NOW,
    )

    assert len(group) == 18
    assert len(result.decompositions) == 16
    assert {item.name for item in result.program.instructions} <= {"rz", "rx"}
    # Every `rx` is either the source `rx` the basis already carries at its own
    # angle, or a pi/2 pulse the synthesis inserted. No other angle is legal.
    assert {
        float(item.params["theta"])
        for item in result.program.instructions
        if item.name == "rx"
    } == {0.7137, 1.5707963267948966}


def test_a_basis_without_a_z_rotation_keeps_the_old_behavior() -> None:
    """No z-rotation means no synthesis: the previous fail-closed path stands."""

    source = CircuitIR(1, (Instruction("y", (0,)),), dtype="complex128")
    with pytest.raises(NativeGateLegalizationError, match="no verified decomposition"):
        legalize_native_gates(source, snapshot=_snapshot(("h", "t")), evaluated_at=_NOW)

    # A rotation reaches the hand-written rewrite, which then names the one gate
    # the target is missing rather than silently synthesizing past it.
    rotation = CircuitIR(
        1, (Instruction("ry", (0,), params={"theta": 0.4}),), dtype="complex128"
    )
    with pytest.raises(
        NativeGateLegalizationError, match="requires unsupported native gate 'sdg'"
    ):
        legalize_native_gates(
            rotation, snapshot=_snapshot(("h", "t")), evaluated_at=_NOW
        )

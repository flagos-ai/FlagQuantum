from __future__ import annotations

import math
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
from flagquantum.simulation.gate_matrix import gate_matrix
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

    # A rotation reaches the equivalence table, which then names a gate the
    # target is missing rather than silently synthesizing past it. The gate it
    # names is the first unsupported *leaf*, not the source opcode: `ry` becomes
    # `sdg h rz h s`, `sdg` itself expands to `s s s`, and `s` is the primitive
    # this target does not publish.
    rotation = CircuitIR(
        1, (Instruction("ry", (0,), params={"theta": 0.4}),), dtype="complex128"
    )
    with pytest.raises(
        NativeGateLegalizationError, match="requires unsupported native gate 's'"
    ):
        legalize_native_gates(
            rotation, snapshot=_snapshot(("h", "t")), evaluated_at=_NOW
        )


def _two_qubit_group() -> tuple[Instruction, ...]:
    """One matrix-carrying instruction per declared two-qubit unitary opcode.

    A named two-qubit gate that is not native cannot be legalized here, because
    its matrix would have to come from `flagquantum.simulation`. A caller that
    holds the matrix already, which is what a two-qubit synthesis pass produces,
    can be legalized.
    """

    from flagquantum.core.operator_schema import OPERATOR_SCHEMAS

    angles = {"theta": 0.7137}
    group: list[Instruction] = []
    for name, schema in sorted(OPERATOR_SCHEMAS.items()):
        if not schema.unitary or schema.arity != 2:
            continue
        params = {item: angles[item] for item in schema.parameters}
        probe = Instruction(name, (0, 1), params=params)
        matrix = gate_matrix(
            probe, bsz=1, device=torch.device("cpu"), dtype=torch.complex128
        ).reshape(-1, 4, 4)[0]
        group.append(
            Instruction(
                name,
                (0, 1),
                params=params,
                matrix=matrix.tolist(),
                metadata={"conditions": ((0, 1),)},
            )
        )
    return tuple(group)


def test_a_z_rotation_sx_and_cx_basis_reaches_the_whole_two_qubit_group() -> None:
    """Every declared two-qubit unitary reaches a `rz`/`sx`/`cx` target.

    Before KAK synthesis this basis legalized 1 of the 11 declared two-qubit
    unitary opcodes, and only when the source carried no matrix at all.
    """

    group = _two_qubit_group()
    source = CircuitIR(2, group, dtype="complex128")
    result = legalize_native_gates(
        source,
        snapshot=_z_sx_snapshot({"name": "cx", "parameters": ()}),
        evaluated_at=_NOW,
    )

    # Non-vacuity: 11 declared arity-2 unitaries, one of which is `cx` itself.
    assert len(group) == 11
    assert {item.name for item in group} == {
        "cphase",
        "crx",
        "cry",
        "crz",
        "cx",
        "cy",
        "cz",
        "rxx",
        "ryy",
        "rzz",
        "swap",
    }
    rewritten = {item.source_opcode for item in result.decompositions}
    assert rewritten == {item.name for item in group} - {"cx"}
    assert {item.name for item in result.program.instructions} <= {"rz", "sx", "cx"}
    assert result.program.instructions[0].metadata == {"conditions": ((0, 1),)}

    # The pinned entangler cost of each source opcode. `cx`, `cz`, and `cy` are
    # controlled gates, locally equivalent to the entangler; `swap` is the most
    # expensive point in the chamber; the remaining seven are controllizations
    # of a rotation.
    per_opcode = {
        item.source_opcode: sum(1 for name in item.replacement_opcodes if name == "cx")
        for item in result.decompositions
    }
    assert per_opcode == {
        "cphase": 2,
        "crx": 2,
        "cry": 2,
        "crz": 2,
        "cy": 1,
        "cz": 1,
        "rxx": 2,
        "ryy": 2,
        "rzz": 2,
        "swap": 3,
    }
    assert sum(per_opcode.values()) == 19


def test_the_two_qubit_legalized_state_matches_up_to_one_global_phase() -> None:
    """The entangler basis cannot carry the phase of an arbitrary unitary."""

    probe = CircuitIR(
        2,
        (Instruction("swap", (0, 1)), Instruction("h", (0,))),
        dtype="complex128",
    )
    matrix = gate_matrix(
        Instruction("swap", (0, 1)),
        bsz=1,
        device=torch.device("cpu"),
        dtype=torch.complex128,
    ).reshape(4, 4)
    with_matrix = CircuitIR(
        2,
        (
            Instruction("swap", (0, 1), matrix=matrix.tolist()),
            Instruction("h", (0,)),
        ),
        dtype="complex128",
    )
    snapshot = _z_sx_snapshot({"name": "cx", "parameters": ()})
    result = legalize_native_gates(with_matrix, snapshot=snapshot, evaluated_at=_NOW)

    assert all(item.matrix is None for item in result.program.instructions)
    # `swap` is the KAK record; `h` is not native in this basis either, so it is
    # rewritten by one-qubit synthesis. Both are needed to produce a phase.
    assert [item.source_opcode for item in result.decompositions] == ["swap", "h"]
    entanglers = [
        name for name in result.decompositions[0].replacement_opcodes if name == "cx"
    ]
    assert len(entanglers) == 3, "`swap` is the far corner of the Weyl chamber"

    before = _state(probe)
    after = _state(result.program)
    overlap = torch.vdot(before.reshape(-1), after.reshape(-1))
    assert float(torch.abs(overlap)) == pytest.approx(1.0, abs=1e-12)
    assert float(torch.max(torch.abs(before - after))) > 0.5


def test_a_matrix_too_wide_for_the_entangler_basis_fails_closed() -> None:
    """Three wires need a decomposition this module does not have."""

    matrix = torch.eye(8, dtype=torch.complex128).tolist()
    source = CircuitIR(
        3,
        (Instruction("blob", (0, 1, 2), matrix=matrix),),
        dtype="complex128",
    )
    with pytest.raises(NativeGateLegalizationError, match="has no native contract"):
        legalize_native_gates(
            source,
            snapshot=_z_sx_snapshot({"name": "cx", "parameters": ()}),
            evaluated_at=_NOW,
        )


def test_a_matrix_without_an_entangler_in_the_basis_fails_closed() -> None:
    """No entangler, no two-qubit synthesis: the previous refusal stands."""

    matrix = gate_matrix(
        Instruction("cz", (0, 1)),
        bsz=1,
        device=torch.device("cpu"),
        dtype=torch.complex128,
    ).reshape(4, 4)
    source = CircuitIR(
        2,
        (Instruction("cz", (0, 1), matrix=matrix.tolist()),),
        dtype="complex128",
    )

    # `rz` and `sx` but no `cx` or `cz`.
    with pytest.raises(NativeGateLegalizationError, match="has no native contract"):
        legalize_native_gates(source, snapshot=_z_sx_snapshot(), evaluated_at=_NOW)
    # `cx` but no z-rotation.
    with pytest.raises(NativeGateLegalizationError, match="has no native contract"):
        legalize_native_gates(
            source,
            snapshot=_snapshot(("h", "t", {"name": "cx", "parameters": ()})),
            evaluated_at=_NOW,
        )


def test_a_two_wire_matrix_keeps_the_hand_written_rules_out_of_its_way() -> None:
    """A matrix on `cx` is not native work; a matrix on `swap` is KAK work."""

    cx_matrix = gate_matrix(
        Instruction("cx", (0, 1)),
        bsz=1,
        device=torch.device("cpu"),
        dtype=torch.complex128,
    ).reshape(4, 4)
    source = CircuitIR(
        2,
        (Instruction("cx", (0, 1), matrix=cx_matrix.tolist()),),
        dtype="complex128",
    )
    snapshot = _z_sx_snapshot({"name": "cx", "parameters": ()})
    result = legalize_native_gates(source, snapshot=snapshot, evaluated_at=_NOW)

    # `cx` is native, so the matrix-carrying instruction is kept untouched
    # rather than being re-synthesized into a longer form.
    assert result.decompositions == ()
    assert len(result.program.instructions) == 1
    assert result.program.instructions[0].matrix is not None


def test_a_target_whose_only_two_qubit_gate_is_a_rotation_reaches_the_group() -> None:
    """An ion-trap shape: `rz`, `rx`, and `rzz` at pi/2 and nothing else.

    This is the basis W8-03 opens. It publishes no controlled gate at all, so
    before the entangler table carried the interaction rotations it legalized one
    of the eleven declared two-qubit unitaries. The rotation is applied at pi/2,
    which is the only angle at which the family sits on its supercontrolled Weyl
    point.
    """

    rotation_only = _snapshot(
        (
            {"name": "rz", "parameters": ("theta",)},
            {"name": "rx", "parameters": ("theta",)},
            {"name": "rzz", "parameters": ("theta",)},
        )
    )
    group = _two_qubit_group()
    source = CircuitIR(2, group, dtype="complex128")
    result = legalize_native_gates(source, snapshot=rotation_only, evaluated_at=_NOW)

    assert len(group) == 11
    # Non-vacuity: this basis really has no controlled gate and no `sx`.
    assert {item.name for item in result.program.instructions} <= {"rz", "rx", "rzz"}
    assert "cx" not in {item.name for item in result.program.instructions}
    assert {item.source_opcode for item in result.decompositions} == {
        item.name for item in group
    } - {"rzz"}

    per_opcode = {
        item.source_opcode: sum(1 for name in item.replacement_opcodes if name == "rzz")
        for item in result.decompositions
    }
    assert per_opcode == {
        "cphase": 2,
        "crx": 2,
        "cry": 2,
        "crz": 2,
        "cx": 1,
        "cy": 1,
        "cz": 1,
        "rxx": 2,
        "ryy": 2,
        "swap": 3,
    }
    # One fewer than the 20 the whole group costs over an entangler the basis has
    # to synthesize: `rzz` is native here, so it is kept instead of decomposed and
    # produces no record at all. The count above is therefore over ten opcodes,
    # which is why it is pinned as a literal rather than derived from the sum.
    assert sum(per_opcode.values()) == 18
    assert len(result.decompositions) == 10

    # Every entangler the synthesis emits carries the pi/2 angle, and every other
    # gate it emits is a rotation the target published. `rzz` is dropped from the
    # source first, because the basis publishes `rzz` natively and keeps it, so
    # its own 0.7137 would otherwise be read as a synthesized angle.
    without_rzz = tuple(item for item in group if item.name != "rzz")
    synthesized = legalize_native_gates(
        CircuitIR(2, without_rzz, dtype="complex128"),
        snapshot=rotation_only,
        evaluated_at=_NOW,
    )
    entanglers = 0
    for instruction in synthesized.program.instructions:
        if instruction.name == "rzz":
            entanglers += 1
            assert instruction.params["theta"] == pytest.approx(math.pi / 2)
        else:
            assert instruction.name in {"rz", "rx"}
        assert instruction.matrix is None
        assert len(instruction.wires) == 1 or instruction.name == "rzz"
    assert entanglers == sum(per_opcode.values()) == 18


def test_a_rotation_entangler_is_preferred_only_when_no_cheaper_one_exists() -> None:
    """A target publishing both `cx` and `rzz` keeps `cx`.

    The entangler table is ordered so the choice is a fixed function of the
    published gate set rather than of a dictionary order, and a parameter-free
    entangler always wins over one that has to be applied at an angle.
    """

    both = _snapshot(
        (
            {"name": "rz", "parameters": ("theta",)},
            {"name": "rx", "parameters": ("theta",)},
            {"name": "cx", "parameters": ()},
            {"name": "rzz", "parameters": ("theta",)},
        )
    )
    result = legalize_native_gates(
        CircuitIR(2, _two_qubit_group(), dtype="complex128"),
        snapshot=both,
        evaluated_at=_NOW,
    )
    emitted = {
        name for item in result.decompositions for name in item.replacement_opcodes
    }
    assert "cx" in emitted
    # `rzz` does appear in the program, but only as the source `rzz` the basis
    # publishes natively; the synthesis never reaches for it.
    assert "rzz" not in emitted


def test_a_rotation_entangler_declared_without_its_angle_fails_closed() -> None:
    """A target that publishes `rzz` as a fixed gate is refused, not miscompiled.

    The descriptor language can say which parameter names a target accepts but
    not which angle it applies, so a target that accepts no `theta` cannot
    receive the pi/2 rotation the synthesis needs. The legalizer must name the
    gate it cannot emit rather than emit an instruction the target never offered.
    """

    fixed = _snapshot(
        (
            {"name": "rz", "parameters": ("theta",)},
            {"name": "rx", "parameters": ("theta",)},
            {"name": "rzz", "parameters": ()},
        )
    )
    with pytest.raises(
        NativeGateLegalizationError, match="requires unsupported native gate 'rzz'"
    ):
        legalize_native_gates(
            CircuitIR(2, _two_qubit_group(), dtype="complex128"),
            snapshot=fixed,
            evaluated_at=_NOW,
        )


def _named_two_qubit_group() -> tuple[Instruction, ...]:
    """One **named** instruction per declared two-qubit unitary opcode.

    No matrix: this is the caller that names a gate and has nothing else. It is
    the path the equivalence table opens, and it must not need
    `flagquantum.simulation` to be legalized.
    """

    from flagquantum.core.operator_schema import OPERATOR_SCHEMAS

    return tuple(
        Instruction(
            name,
            (0, 1),
            params=dict.fromkeys(schema.parameters, 0.7137),
        )
        for name, schema in sorted(OPERATOR_SCHEMAS.items())
        if schema.unitary and schema.arity == 2
    )


def test_a_z_rotation_sx_and_cx_basis_reaches_the_named_two_qubit_group() -> None:
    """Every declared two-qubit unitary reaches a `rz`/`sx`/`cx` target by name.

    Before the equivalence table this basis legalized 2 of the 11 named
    two-qubit opcodes -- `cx` itself, and `swap` through a hand-written rule --
    and refused the other nine, because turning a named gate into the entangler
    basis through KAK would have needed a matrix from `flagquantum.simulation`.
    """

    group = _named_two_qubit_group()
    # Non-vacuity: this is a claim about the whole group, not a chosen subset.
    assert len(group) == 11
    assert all(item.matrix is None for item in group)
    result = legalize_native_gates(
        CircuitIR(2, group, dtype="complex128"),
        snapshot=_z_sx_snapshot(
            {"name": "x", "parameters": ()}, {"name": "cx", "parameters": ()}
        ),
        evaluated_at=_NOW,
    )
    assert len(result.decompositions) == 10
    assert {item.source_opcode for item in result.decompositions} == {
        "cphase",
        "crx",
        "cry",
        "crz",
        "cy",
        "cz",
        "rxx",
        "ryy",
        "rzz",
        "swap",
    }
    # No leaf may be a gate the basis never published, and nothing may survive
    # with a matrix attached, which would mean the table was bypassed.
    assert {item.name for item in result.program.instructions} <= {"rz", "sx", "cx"}
    assert all(item.matrix is None for item in result.program.instructions)


def test_every_named_two_qubit_translation_reproduces_its_state() -> None:
    """The strongest check available: run both sides and compare the states.

    Thirteen of the table's entries are exact and `cphase` is not, so the
    comparison is up to one global phase for all of them. A mixed two-wire
    program is used rather than the bare gate, because a single gate on `|00>`
    would not exercise the entangling phase the equivalence is about.
    """

    basis = _z_sx_snapshot(
        {"name": "x", "parameters": ()}, {"name": "cx", "parameters": ()}
    )
    for source in _named_two_qubit_group():
        program = CircuitIR(
            2,
            (
                Instruction("h", (0,)),
                Instruction("x", (1,)),
                Instruction(source.name, (0, 1), params=dict(source.params)),
            ),
            dtype="complex128",
        )
        result = legalize_native_gates(program, snapshot=basis, evaluated_at=_NOW)
        assert result.changed is True, source.name
        expected = _state(program)
        actual = _state(result.program)
        overlap = torch.vdot(expected.flatten(), actual.flatten())
        assert abs(abs(overlap) - 1.0) < 1e-12, source.name
        torch.testing.assert_close(
            actual,
            overlap * expected,
            rtol=0.0,
            atol=1e-12,
        )


def test_a_basis_without_a_z_rotation_reaches_only_the_identities() -> None:
    """`clifford-t` publishes `h`, `s`, `t` and `cx`, so four names get through.

    `cx` is native, `swap` is three `cx`, `cz` is a `cx` under two `h`, and `cy`
    is `sdg cx s` whose `sdg` is `s s s`. Every other name needs a z-rotation to
    build an interaction rotation from, and the refusal has to name a gate that
    is really missing rather than the source opcode.

    Qiskit's `BasisTranslator` reaches these same four names into this same
    basis, and reaches `cy` through the same three-`s` expansion of `sdg`; see
    `benchmarks/compiler_basis_translation.py` for the measured comparison.
    """

    group = _named_two_qubit_group()
    basis = _snapshot(
        (
            {"name": "h", "parameters": ()},
            {"name": "s", "parameters": ()},
            {"name": "t", "parameters": ()},
            {"name": "cx", "parameters": ()},
        )
    )
    reached: list[str] = []
    missing: dict[str, str] = {}
    for item in group:
        try:
            legalize_native_gates(
                CircuitIR(2, (item,), dtype="complex128"),
                snapshot=basis,
                evaluated_at=_NOW,
            )
        except NativeGateLegalizationError as error:
            missing[item.name] = str(error)
            continue
        reached.append(item.name)
    assert sorted(reached) == ["cx", "cy", "cz", "swap"]
    assert len(missing) == 7
    # A failure names the native gate the target is missing, not the source gate.
    assert all("requires unsupported native gate" in text for text in missing.values())

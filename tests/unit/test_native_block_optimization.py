"""Native baseline retention, bounded synthesis and independent operator checks."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
import torch

from flagquantum.compiler.lowering import native_block_optimization as blocks
from flagquantum.compiler.lowering.native_gate_legalization import (
    NativeGateLegalizationResult,
    legalize_native_gates,
)
from flagquantum.compiler.optimization import _cz_blocks
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.parameters import Parameter
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

pytestmark = pytest.mark.unit
_NOW = datetime(2026, 11, 19, tzinfo=timezone.utc)


def _snapshot(
    gates: tuple[object, ...] = ("rx", "rz", "cz")
) -> TargetCapabilitySnapshot:
    scope = CapabilityScope(device_ids=("native-block-test",))
    return TargetCapabilitySnapshot(
        target_identity=TargetIdentity(
            target_id="native-block-test",
            target_class="test",
            provider="test",
            provider_version="1",
            target_revision="1",
            environment_id="offline",
        ),
        scope=scope,
        captured_at=_NOW.isoformat(),
        valid_until=(_NOW + timedelta(hours=1)).isoformat(),
        facts=(
            CapabilityFact(
                name="gates.native",
                value=gates,
                support_status=SupportStatus.VERIFIED,
                fact_exposure=FactExposure.DECLARED,
                source=FactSource(kind="test", ref="native-block-evidence"),
            ),
        ),
        evidence_refs=(
            EvidenceReference(
                evidence_id="native-block-evidence",
                sha256="a" * 64,
                level=EvidenceLevel.BASIC,
                scope=scope,
            ),
        ),
    )


def _source(pair: tuple[int, int] = (0, 1)) -> CircuitIR:
    return CircuitIR(2, (Instruction("swap", pair), Instruction("cx", pair[::-1])))


def _stage(
    source: CircuitIR, *, budget: int = 256
) -> tuple[blocks.NativeBlockOptimizationResult, NativeGateLegalizationResult]:
    snapshot = _snapshot()
    native = legalize_native_gates(
        source, snapshot=snapshot, evaluated_at=_NOW, max_added_operations=budget
    )
    result = blocks.optimize_native_blocks(
        native, snapshot=snapshot, evaluated_at=_NOW, max_added_operations=budget
    )
    assert result is not None
    blocks.verify_native_block_optimization(
        result,
        native,
        snapshot=snapshot,
        evaluated_at=_NOW,
        max_added_operations=budget,
    )
    return result, native


def _operator(program: CircuitIR) -> torch.Tensor:
    size = 1 << program.n_wires
    result = torch.eye(size, dtype=torch.complex128)
    for instruction in program.instructions:
        local_size = 1 << len(instruction.wires)
        local = gate_matrix(
            instruction, bsz=1, device=torch.device("cpu"), dtype=torch.complex128
        ).reshape(local_size, local_size)
        embedded = torch.zeros_like(result)
        for column in range(size):
            local_column = sum(
                ((column >> (program.n_wires - qubit - 1)) & 1)
                << (len(instruction.wires) - position - 1)
                for position, qubit in enumerate(instruction.wires)
            )
            for local_row in range(local_size):
                row = column
                for position, qubit in enumerate(instruction.wires):
                    bit = (local_row >> (len(instruction.wires) - position - 1)) & 1
                    mask = 1 << (program.n_wires - qubit - 1)
                    row = (row & ~mask) | bit * mask
                embedded[row, column] = local[local_row, local_column]
        result = embedded @ result
    return result


def _equivalent(source: CircuitIR, output: CircuitIR) -> None:
    expected, actual = _operator(source), _operator(output)
    phase = torch.trace(actual.conj().T @ expected) / len(expected)
    assert float(abs(abs(phase) - 1)) <= 1e-11
    torch.testing.assert_close(expected, phase * actual, atol=1e-11, rtol=0)


@pytest.mark.parametrize("pair", [(0, 1), (1, 0)])
def test_actual_native_baseline_reduces_four_cz_to_two(pair: tuple[int, int]) -> None:
    result, native = _stage(_source(pair))
    assert result.source_program is native.program
    assert sum(item.name == "cz" for item in native.program.instructions) == 4
    assert sum(item.name == "cz" for item in result.program.instructions) == 2
    assert result.rewrites[0].input_instruction_indices == tuple(range(28))
    assert result.rewrites[0].max_abs_residual <= 1e-11
    _equivalent(_source(pair), result.program)


def test_identity_cancellation_retains_empty_output_evidence() -> None:
    source = CircuitIR(2, (Instruction("cz", (0, 1)), Instruction("cz", (1, 0))))
    result, _ = _stage(source)
    assert result.program.instructions == ()
    assert result.rewrites[0].output_instruction_indices == ()
    _equivalent(source, result.program)


def test_disjoint_spectators_and_interleaved_blocks_preserve_ownership() -> None:
    source = CircuitIR(
        4,
        (
            Instruction("cz", (0, 2)),
            Instruction("cz", (1, 3)),
            Instruction("rz", (2,), {"theta": 0.3}),
            Instruction("cz", (2, 0)),
            Instruction("cz", (3, 1)),
        ),
    )
    result, _ = _stage(source)
    assert len(result.rewrites) == 2
    _equivalent(source, result.program)


@pytest.mark.parametrize(
    "held",
    [
        Instruction("rx", (0,), {"theta": Parameter("theta")}),
        Instruction("rx", (0,), {"theta": torch.tensor(0.3, requires_grad=True)}),
        Instruction("rx", (0,), {"theta": torch.tensor([0.2, 0.3])}),
        Instruction("rz", (0,), {"theta": 0.2}, metadata={"conditions": ((0, 1),)}),
        Instruction("cz", (0, 1), metadata={"routing_phase": "forward"}),
        Instruction("cz", (0, 1), matrix=torch.eye(4, dtype=torch.complex128)),
    ],
)
def test_ineligible_members_remain_exact_objects(held: Instruction) -> None:
    result, native = _stage(
        CircuitIR(2, (Instruction("cz", (0, 1)), held, Instruction("cz", (0, 1))))
    )
    assert result.program is native.program
    assert any(item is held for item in result.program.instructions)
    assert result.rewrites == ()


def test_unknown_spectator_annotation_is_global_collection_boundary() -> None:
    source = CircuitIR(
        3,
        (
            Instruction("cz", (0, 1)),
            Instruction("rz", (2,), {"theta": 0.1}, metadata={"classical_write": "x"}),
            Instruction("cz", (1, 0)),
        ),
    )
    result, native = _stage(source)
    assert result.program is native.program
    assert result.rewrites == ()


@pytest.mark.parametrize(
    "gates",
    [
        ("rx", "rz", "cx"),
        ("rx", "rz", "cz", "cx"),
        ("rx", "rz", "cz", "ccx"),
        ({"name": "rx", "parameters": ()}, "rz", "cz"),
    ],
)
def test_nonqualifying_targets_do_not_have_a_block_stage(
    gates: tuple[object, ...],
) -> None:
    snapshot = _snapshot(gates)
    native = legalize_native_gates(
        CircuitIR(2, ()), snapshot=snapshot, evaluated_at=_NOW
    )
    assert (
        blocks.optimize_native_blocks(native, snapshot=snapshot, evaluated_at=_NOW)
        is None
    )


@pytest.mark.parametrize(
    "reason,candidate",
    [
        ("synthesis_unavailable", None),
        ("not_lower_cz_cost", (Instruction("cz", (0, 1)),) * 4),
        ("residual_rejected", (Instruction("cz", (0, 1)),)),
        (
            "instruction_growth_rejected",
            (Instruction("rz", (0,), {"theta": 0.0}),) * 61,
        ),
    ],
)
def test_rejected_proposals_keep_baseline_and_report_reason(
    reason: str,
    candidate: tuple[Instruction, ...] | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _cz_blocks, "synthesize_two_qubit", lambda *args, **kwargs: candidate
    )
    result, native = _stage(_source())
    assert result.program is native.program
    assert result.rejection_counts[reason] == 1
    assert sum(result.rejection_counts.values()) == 1


def test_combined_budget_rejects_a_candidate_that_fits_the_block_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot()
    native = legalize_native_gates(_source(), snapshot=snapshot, evaluated_at=_NOW)
    valid = _cz_blocks._candidate(native.program.instructions, (0, 1))
    assert valid is not None
    padded = valid + (Instruction("rz", (0,), {"theta": 0.0}),) * 10
    assert len(padded) <= len(native.program.instructions) + 32
    monkeypatch.setattr(
        _cz_blocks, "synthesize_two_qubit", lambda *args, **kwargs: padded
    )
    result, original = _stage(_source(), budget=26)
    assert result.program is original.program
    assert result.rejection_counts["instruction_growth_rejected"] == 1


def _reseal(
    result: blocks.NativeBlockOptimizationResult,
) -> blocks.NativeBlockOptimizationResult:
    return replace(
        result,
        optimization_identity=blocks._identity(
            result.source_program,
            result.program,
            result.target_snapshot_id,
            result.rewrites,
            result.rejection_counts,
        ),
    )


@pytest.mark.parametrize(
    "tamper", ["operator", "phase", "cost", "membership", "metadata"]
)
def test_resealed_numerical_or_lineage_corruption_is_rejected(tamper: str) -> None:
    result, native = _stage(_source())
    rewrite = result.rewrites[0]
    if tamper == "operator":
        first, *rest = result.program.instructions
        result = replace(
            result,
            program=replace(
                result.program,
                instructions=(replace(first, params={"theta": 1.0}), *rest),
            ),
        )
    elif tamper == "phase":
        result = replace(
            result,
            rewrites=(
                replace(
                    rewrite,
                    phase_real=-rewrite.phase_real,
                    phase_imag=-rewrite.phase_imag,
                ),
            ),
        )
    elif tamper == "cost":
        result = replace(result, rewrites=(replace(rewrite, baseline_cz_count=3),))
    elif tamper == "membership":
        result = replace(
            result,
            rewrites=(
                replace(
                    rewrite,
                    input_instruction_indices=rewrite.input_instruction_indices[1:],
                ),
            ),
        )
    else:
        result = replace(
            result, program=replace(result.program, metadata={"altered": True})
        )
    with pytest.raises(blocks.NativeBlockOptimizationError):
        blocks.verify_native_block_optimization(
            _reseal(result), native, snapshot=_snapshot(), evaluated_at=_NOW
        )


def test_equivalent_baseline_copy_is_not_the_retained_instance() -> None:
    result, native = _stage(_source())
    result = replace(result, source_program=replace(result.source_program))
    with pytest.raises(blocks.NativeBlockOptimizationError, match="baseline instance"):
        blocks.verify_native_block_optimization(
            result, native, snapshot=_snapshot(), evaluated_at=_NOW
        )


def test_long_stream_bounds_each_rewrite() -> None:
    source = CircuitIR(2, (Instruction("cz", (0, 1)), Instruction("cz", (1, 0))) * 100)
    result, _ = _stage(source)
    assert all(
        len(rewrite.input_instruction_indices) <= 64 for rewrite in result.rewrites
    )
    _equivalent(source, result.program)


@pytest.mark.parametrize(
    "budget,exception", [(True, TypeError), (1.5, TypeError), (-1, ValueError)]
)
def test_invalid_budgets_fail_before_search(
    budget: object, exception: type[Exception]
) -> None:
    snapshot = _snapshot()
    native = legalize_native_gates(_source(), snapshot=snapshot, evaluated_at=_NOW)
    with pytest.raises(exception):
        blocks.optimize_native_blocks(
            native, snapshot=snapshot, evaluated_at=_NOW, max_added_operations=budget
        )


def test_stage_cannot_rescue_an_over_budget_native_baseline() -> None:
    snapshot = _snapshot()
    native = legalize_native_gates(_source(), snapshot=snapshot, evaluated_at=_NOW)
    with pytest.raises(
        blocks.NativeBlockOptimizationError, match="baseline already exceeds"
    ):
        blocks.optimize_native_blocks(
            native, snapshot=snapshot, evaluated_at=_NOW, max_added_operations=25
        )


def test_no_improvement_keeps_program_and_counters_are_immutable() -> None:
    result, native = _stage(CircuitIR(2, (Instruction("cz", (0, 1)),)))
    assert result.program is native.program
    assert result.rejection_counts["not_lower_cz_cost"] == 1
    with pytest.raises(TypeError):
        result.rejection_counts["not_lower_cz_cost"] = 0


@pytest.mark.parametrize(
    "counts",
    [
        {},
        {"unknown": 0},
        {
            "not_lower_cz_cost": False,
            "synthesis_unavailable": 0,
            "residual_rejected": 0,
            "instruction_growth_rejected": 0,
        },
    ],
)
def test_closed_rejection_counter_schema(counts: dict[str, int]) -> None:
    result, _ = _stage(_source())
    with pytest.raises(blocks.NativeBlockOptimizationError, match="rejection counters"):
        replace(result, rejection_counts=counts)


def test_stale_hash_identity_and_target_are_rejected() -> None:
    result, native = _stage(_source())
    with pytest.raises(blocks.NativeBlockOptimizationError, match="source hash"):
        replace(result, source_content_hash="0" * 64)
    with pytest.raises(blocks.NativeBlockOptimizationError, match="identity is stale"):
        blocks.verify_native_block_optimization(
            replace(result, optimization_identity="0" * 64),
            native,
            snapshot=_snapshot(),
            evaluated_at=_NOW,
        )
    with pytest.raises(blocks.NativeBlockOptimizationError, match="another snapshot"):
        blocks.verify_native_block_optimization(
            replace(result, target_snapshot_id="0" * 64),
            native,
            snapshot=_snapshot(),
            evaluated_at=_NOW,
        )
    with pytest.raises(blocks.NativeBlockOptimizationError, match="another snapshot"):
        blocks.optimize_native_blocks(
            replace(native, target_snapshot_id="0" * 64),
            snapshot=_snapshot(),
            evaluated_at=_NOW,
        )


@pytest.mark.parametrize(
    "change", ["duplicate", "input_range", "output_range", "orphan", "output_basis"]
)
def test_resealed_partition_errors_are_rejected(change: str) -> None:
    result, native = _stage(_source())
    rewrite = result.rewrites[0]
    if change == "duplicate":
        result = replace(result, rewrites=(rewrite, rewrite))
    elif change == "input_range":
        result = replace(
            result,
            rewrites=(replace(rewrite, input_instruction_indices=tuple(range(1, 29))),),
        )
    elif change == "output_range":
        result = replace(
            result,
            rewrites=(
                replace(rewrite, output_instruction_indices=tuple(range(1, 24))),
            ),
        )
    elif change == "orphan":
        result = replace(
            result,
            program=replace(
                result.program,
                instructions=(
                    *result.program.instructions,
                    Instruction("rz", (0,), {"theta": 0.0}),
                ),
            ),
        )
    else:
        result = replace(
            result,
            program=replace(
                result.program,
                instructions=(Instruction("h", (0,)), *result.program.instructions[1:]),
            ),
        )
    with pytest.raises(blocks.NativeBlockOptimizationError):
        blocks.verify_native_block_optimization(
            _reseal(result), native, snapshot=_snapshot(), evaluated_at=_NOW
        )


def test_resealed_corrupt_native_baseline_is_not_accepted_as_proof() -> None:
    result, native = _stage(_source())
    corrupted = replace(native, decompositions=())
    with pytest.raises(
        blocks.NativeBlockOptimizationError, match="baseline failed replay"
    ):
        blocks.verify_native_block_optimization(
            result, corrupted, snapshot=_snapshot(), evaluated_at=_NOW
        )


@pytest.mark.parametrize(
    "spectator",
    [
        Instruction("rx", (0,), {"theta": 0.3}),
        Instruction("rz", (2,), {"theta": 0.3}, metadata={"classical_write": "x"}),
    ],
)
def test_forged_cancellation_cannot_cross_a_dependency(spectator: Instruction) -> None:
    source = CircuitIR(
        3, (Instruction("cz", (0, 1)), spectator, Instruction("cz", (1, 0)))
    )
    result, native = _stage(source)
    rewrite = blocks.BlockRewriteEvidence(
        input_instruction_indices=(0, 2),
        output_instruction_indices=(),
        physical_qubits=(0, 1),
        baseline_cz_count=2,
        replacement_cz_count=0,
        equivalence="unitary_up_to_global_phase",
        phase_real=1.0,
        phase_imag=0.0,
        max_abs_residual=0.0,
    )
    forged = _reseal(
        replace(
            result,
            program=replace(native.program, instructions=(spectator,)),
            rewrites=(rewrite,),
        )
    )
    with pytest.raises(
        blocks.NativeBlockOptimizationError, match="ordering dependency"
    ):
        blocks.verify_native_block_optimization(
            forged, native, snapshot=_snapshot(), evaluated_at=_NOW
        )


def test_resealed_equivalent_output_must_obey_the_combined_budget() -> None:
    result, native = _stage(_source())
    padded = (
        *result.program.instructions,
        *((Instruction("rz", (0,), {"theta": 0.0}),) * 10),
    )
    rewrite = replace(
        result.rewrites[0], output_instruction_indices=tuple(range(len(padded)))
    )
    forged = _reseal(
        replace(
            result,
            program=replace(result.program, instructions=padded),
            rewrites=(rewrite,),
        )
    )
    with pytest.raises(blocks.NativeBlockOptimizationError, match="combined budget"):
        blocks.verify_native_block_optimization(
            forged,
            native,
            snapshot=_snapshot(),
            evaluated_at=_NOW,
            max_added_operations=26,
        )


def test_resealed_output_membership_cannot_point_past_the_program() -> None:
    result, native = _stage(_source())
    length = len(result.program.instructions)
    rewrite = replace(
        result.rewrites[0], output_instruction_indices=tuple(range(length, 2 * length))
    )
    forged = _reseal(replace(result, rewrites=(rewrite,)))
    with pytest.raises(
        blocks.NativeBlockOptimizationError, match="output index is out of range"
    ):
        blocks.verify_native_block_optimization(
            forged, native, snapshot=_snapshot(), evaluated_at=_NOW
        )


def test_an_unchanged_instruction_cannot_be_replaced_by_an_equal_copy() -> None:
    result, native = _stage(CircuitIR(2, (Instruction("cz", (0, 1)),)))
    forged = _reseal(
        replace(
            result,
            program=replace(
                result.program, instructions=(replace(result.program.instructions[0]),)
            ),
        )
    )
    with pytest.raises(
        blocks.NativeBlockOptimizationError, match="unchanged native instruction"
    ):
        blocks.verify_native_block_optimization(
            forged, native, snapshot=_snapshot(), evaluated_at=_NOW
        )

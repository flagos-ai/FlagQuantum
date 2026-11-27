"""Approved v4 lineage budgets preserve all other evidence constraints."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from flagquantum.core import _compilation_evidence_v4 as v4
from flagquantum.core._compilation_evidence import (
    read_compilation_evidence_bundle_json,
)

pytestmark = pytest.mark.unit
_DIGEST = "1" * 64


def _plan(native_count: int, output_count: int = 0) -> v4.PhysicalPlanEvidenceV4:
    # Each adjacent CZ pair cancels. Optional RX(0) outputs are identity too.
    # Evidence records carry structure, not angles or an equivalence proof.
    native = tuple(
        v4.NativeInputInstructionEvidence(
            index, index // 2, index // 2, index % 2, "source", "cz", (0, 1), (0, 1)
        )
        for index in range(native_count)
    )
    blocks = []
    outputs = []
    for start in range(0, native_count, 64):
        stop = min(start + 64, native_count)
        output_start = len(outputs)
        output_stop = min(output_start + 64, output_count)
        block_index = len(blocks)
        for index in range(output_start, output_stop):
            outputs.append(
                v4.PhysicalInstructionEvidenceV4(
                    index,
                    None,
                    block_index,
                    index - output_start,
                    "rx",
                    (0,),
                    index,
                    (index - 1,) if index else (),
                    ("wire",) if index else (),
                )
            )
        blocks.append(
            v4.BlockRewriteEvidence(
                tuple(range(start, stop)),
                tuple(range(output_start, output_stop)),
                (0, 1),
                stop - start,
                0,
                "unitary_up_to_global_phase",
                1.0,
                0.0,
                0.0,
            )
        )
    assert len(outputs) == output_count
    return v4.PhysicalPlanEvidenceV4(
        source_circuit_hash=_DIGEST,
        physical_circuit_hash=_DIGEST,
        target_snapshot_id=_DIGEST,
        topology_identity=None,
        coupling=None,
        initial_logical_to_physical=(0, 1),
        pre_restore_logical_to_physical=(0, 1),
        final_logical_to_physical=(0, 1),
        mapping_transitions=(),
        instructions=tuple(outputs),
        topology_legalization_identity=None,
        native_gate_legalization_identity=_DIGEST,
        schedule_identity=_DIGEST,
        schedule_depth=output_count,
        maximum_parallel_width=int(bool(output_count)),
        critical_path=tuple(range(output_count)),
        native_input_circuit_hash=_DIGEST,
        native_input_instructions=native,
        block_rewrites=tuple(blocks),
        native_block_optimization_identity=_DIGEST,
    )


def _bundle(plan: v4.PhysicalPlanEvidenceV4) -> v4.CompilationEvidenceBundleV4:
    return v4.CompilationEvidenceBundleV4(
        producer="core-lineage-capacity-test",
        source={
            "source_artifact_identity": _DIGEST,
            "circuit_artifact_identity": _DIGEST,
            "binding_identity": None,
            "source_circuit_hash": _DIGEST,
            "final_circuit_hash": _DIGEST,
        },
        target={"snapshot_id": _DIGEST, "target_legalization_identity": _DIGEST},
        physical_plan=plan,
        output={
            "profile": "openqasm3",
            "payload_sha256": _DIGEST,
            "emission_identity": _DIGEST,
            "conformance_identity": _DIGEST,
            "executable_artifact_identity": _DIGEST,
            "artifact_compilation_identity": _DIGEST,
        },
    )


@pytest.mark.parametrize(
    "native_count,output_count", [(4096, 0), (8192, 0), (4096, 4096)]
)
def test_exact_capacity_roundtrips(native_count: int, output_count: int) -> None:
    plan = _plan(native_count, output_count)
    assert len(plan.native_input_instructions) == native_count
    assert (
        sum(
            len(block.input_instruction_indices) + len(block.output_instruction_indices)
            for block in plan.block_rewrites
        )
        == native_count + output_count
    )
    restored = v4.PhysicalPlanEvidenceV4.from_dict(
        plan.to_dict(), target_snapshot_id=_DIGEST
    )
    assert restored == plan
    bundle = _bundle(plan)
    assert read_compilation_evidence_bundle_json(bundle.to_json()) == bundle


def test_native_record_overflow_rejected_before_record_reconstruction() -> None:
    boundary = _plan(8192)
    # One unchanged RX(0) tail exceeds only the native count: memberships stay
    # at 8192, outputs at one, and topology/source ancestry stays below 4096.
    tail = v4.NativeInputInstructionEvidence(
        8192, 4095, 4095, 2, "source", "rx", (0, 1), (0,)
    )
    output = v4.PhysicalInstructionEvidenceV4(
        0, 8192, None, None, "rx", (0,), 0, (), ()
    )
    with pytest.raises(
        ValueError, match="native_input_instructions exceeds record limit 8192"
    ):
        replace(
            boundary,
            native_input_instructions=boundary.native_input_instructions + (tail,),
            instructions=(output,),
            schedule_depth=1,
            maximum_parallel_width=1,
            critical_path=(0,),
            plan_identity="",
        )
    raw = boundary.to_dict()
    raw["native_input_instructions"].append(tail.to_dict())
    raw.update(
        instructions=[output.to_dict()],
        schedule_depth=1,
        maximum_parallel_width=1,
        critical_path=[0],
        plan_identity="",
    )
    with pytest.raises(
        ValueError, match="native_input_instructions exceeds record limit 8192"
    ):
        v4.PhysicalPlanEvidenceV4.from_dict(raw, target_snapshot_id=_DIGEST)
    plan = _plan(2)
    oversized = (plan.native_input_instructions[0],) * 8193
    with pytest.raises(
        ValueError, match="native_input_instructions exceeds record limit 8192"
    ):
        replace(plan, native_input_instructions=oversized, plan_identity="")
    payload = _bundle(plan).to_dict()
    # Invalid members prove rejection occurs before any record is reconstructed.
    payload["physical_plan"]["native_input_instructions"] = [None] * 8193
    with pytest.raises(
        ValueError, match="native_input_instructions exceeds record limit 8192"
    ):
        v4.PhysicalPlanEvidenceV4.from_dict(
            payload["physical_plan"], target_snapshot_id=_DIGEST
        )
    with pytest.raises(
        ValueError, match="native_input_instructions exceeds record limit 8192"
    ):
        read_compilation_evidence_bundle_json(json.dumps(payload))


def test_membership_overflow_is_independent_and_rejected_before_reconstruction() -> (
    None
):
    # 4098 CZ inputs and 4095 RX(0) outputs satisfy every independent count cap.
    # Their 8193 aggregate memberships exceed only the approved membership cap.
    with pytest.raises(
        ValueError, match="nested block memberships exceed record limit"
    ):
        _plan(4098, 4095)
    plan = _plan(4098, 4094)
    output = v4.PhysicalInstructionEvidenceV4(
        4094, None, 64, 0, "rx", (0,), 4094, (4093,), ("wire",)
    )
    larger = replace(plan.block_rewrites[-1], output_instruction_indices=(4094,))
    with pytest.raises(
        ValueError, match="nested block memberships exceed record limit"
    ):
        replace(
            plan,
            block_rewrites=plan.block_rewrites[:-1] + (larger,),
            instructions=plan.instructions + (output,),
            schedule_depth=4095,
            critical_path=tuple(range(4095)),
            plan_identity="",
        )
    payload = _bundle(plan).to_dict()
    raw = payload["physical_plan"]
    raw["block_rewrites"][-1] = larger.to_dict()
    raw["instructions"].append(output.to_dict())
    raw["schedule_depth"] = 4095
    raw["critical_path"] = list(range(4095))
    # Invalid member proves rejection precedes typed record reconstruction.
    raw["native_input_instructions"][0] = None
    with pytest.raises(
        ValueError, match="nested block memberships exceed record limit"
    ):
        v4.PhysicalPlanEvidenceV4.from_dict(raw, target_snapshot_id=_DIGEST)
    with pytest.raises(
        ValueError, match="nested block memberships exceed record limit"
    ):
        read_compilation_evidence_bundle_json(json.dumps(payload))


@pytest.mark.parametrize(
    "field", ["instructions", "mapping_transitions", "block_rewrites"]
)
def test_unrelated_record_caps_remain_4096(field: str) -> None:
    plan = _plan(2)
    with pytest.raises(ValueError, match=f"{field} exceeds record limit 4096"):
        replace(plan, **{field: (None,) * 4097, "plan_identity": ""})
    payload = plan.to_dict()
    payload[field] = [None] * 4097
    with pytest.raises(ValueError, match=f"{field} exceeds record limit 4096"):
        v4.PhysicalPlanEvidenceV4.from_dict(payload, target_snapshot_id=_DIGEST)


@pytest.mark.parametrize(
    "field", ["source_instruction_index", "topology_instruction_index"]
)
def test_expanded_native_capacity_does_not_expand_ancestry_indices(field: str) -> None:
    plan = _plan(8192)
    bad = replace(plan.native_input_instructions[-1], **{field: 4096})
    with pytest.raises(ValueError, match="native ancestry exceeds record index limit"):
        replace(
            plan,
            native_input_instructions=plan.native_input_instructions[:-1] + (bad,),
            plan_identity="",
        )


def test_per_block_caps_remain_unchanged() -> None:
    plan = _plan(128)
    block = plan.block_rewrites[0]
    with pytest.raises(ValueError, match="block inputs exceeds record limit 64"):
        replace(block, input_instruction_indices=tuple(range(65)))
    with pytest.raises(ValueError, match="block outputs exceeds record limit 96"):
        replace(block, output_instruction_indices=tuple(range(97)))

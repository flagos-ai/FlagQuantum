"""Strict v4 structural acceptance, distinct from quantum equivalence proof."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from typing import Any

import pytest

from flagquantum.core import _compilation_evidence_v4 as v4
from flagquantum.core._compilation_evidence import (
    CompilationEvidenceBundle,
    CouplingEvidence,
    MappingTransitionEvidence,
    _canonical_bytes,
    _coupling_identity,
    read_compilation_evidence_bundle_json,
)
from flagquantum.experimental.artifacts import (
    dump_compilation_evidence,
    load_compilation_evidence,
)

pytestmark = pytest.mark.unit
_DIGEST = "1" * 64


def _bundle(*, empty_output: bool = False) -> v4.CompilationEvidenceBundleV4:
    # CZ squared is identity. The independent spectator RX stays unchanged;
    # no Compiler synthesis, matrix engine, or output evaluator constructs this.
    native = (
        v4.NativeInputInstructionEvidence(0, 0, 0, 0, "source", "cz", (0, 1), (0, 1)),
        v4.NativeInputInstructionEvidence(1, 1, 1, 0, "source", "cz", (0, 1), (0, 1)),
    )
    outputs = ()
    if not empty_output:
        native += (
            v4.NativeInputInstructionEvidence(2, 2, 2, 0, "source", "rx", (2,), (2,)),
        )
        outputs = (
            v4.PhysicalInstructionEvidenceV4(0, 2, None, None, "rx", (2,), 0, (), ()),
        )
    plan = v4.PhysicalPlanEvidenceV4(
        source_circuit_hash=_DIGEST,
        physical_circuit_hash="2" * 64,
        target_snapshot_id=_DIGEST,
        topology_identity=None,
        coupling=None,
        initial_logical_to_physical=(0, 1, 2),
        pre_restore_logical_to_physical=(0, 1, 2),
        final_logical_to_physical=(0, 1, 2),
        mapping_transitions=(),
        instructions=outputs,
        topology_legalization_identity=None,
        native_gate_legalization_identity=_DIGEST,
        schedule_identity=_DIGEST,
        schedule_depth=0 if empty_output else 1,
        maximum_parallel_width=0 if empty_output else 1,
        critical_path=() if empty_output else (0,),
        native_input_circuit_hash="3" * 64,
        native_input_instructions=native,
        block_rewrites=(
            v4.BlockRewriteEvidence(
                (0, 1), (), (0, 1), 2, 0, "unitary_up_to_global_phase", 1.0, 0.0, 0.0
            ),
        ),
        native_block_optimization_identity=_DIGEST,
    )
    return v4.CompilationEvidenceBundleV4(
        producer="core-structural-test",
        source={
            "source_artifact_identity": _DIGEST,
            "circuit_artifact_identity": _DIGEST,
            "binding_identity": None,
            "source_circuit_hash": _DIGEST,
            "final_circuit_hash": "2" * 64,
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


def _unsealed() -> dict[str, Any]:
    payload = copy.deepcopy(_bundle().to_dict())
    payload["bundle_identity"] = ""
    payload["physical_plan"]["plan_identity"] = ""
    return payload


@pytest.mark.parametrize("empty_output", [False, True])
def test_read_only_artifact_view_preserves_v4_identity_and_canonical_json(
    empty_output: bool,
) -> None:
    original = _bundle(empty_output=empty_output)
    encoded = original.to_json()
    view = load_compilation_evidence(encoded)
    assert view.version == "4.0"
    assert view.producer == original.producer
    assert view.identity == original.bundle_identity
    assert view.to_dict() == original.to_dict()
    assert view.to_json() == encoded
    assert dump_compilation_evidence(view) == encoded


def test_read_only_artifact_view_rejects_tampered_v4_identity() -> None:
    payload = _bundle().to_dict()
    payload["producer"] = "tampered-producer"
    with pytest.raises(ValueError, match="identity"):
        load_compilation_evidence(json.dumps(payload))


def _read(payload: dict[str, Any]) -> v4.CompilationEvidenceBundleV4:
    result = read_compilation_evidence_bundle_json(json.dumps(payload))
    assert isinstance(result, v4.CompilationEvidenceBundleV4)
    return result


@pytest.mark.parametrize("empty_output", [False, True])
def test_cancellation_keeps_complete_input_lineage_even_when_no_output_remains(
    empty_output: bool,
) -> None:
    original = _bundle(empty_output=empty_output)
    decoded = read_compilation_evidence_bundle_json(original.to_json())
    assert isinstance(decoded, v4.CompilationEvidenceBundleV4)
    assert decoded.to_json() == original.to_json()
    assert decoded.physical_plan.block_rewrites[0].output_instruction_indices == ()
    assert decoded.physical_plan.block_rewrites[0].input_instruction_indices == (0, 1)
    assert len(decoded.physical_plan.instructions) == (0 if empty_output else 1)
    with pytest.raises(ValueError, match="unknown physical plan evidence field"):
        CompilationEvidenceBundle.from_dict(original.to_dict())


def test_plan_and_bundle_identities_bind_native_baseline_and_stage() -> None:
    original = _bundle()
    for name in (
        "native_input_circuit_hash",
        "native_block_optimization_identity",
        "source_circuit_hash",
        "target_snapshot_id",
    ):
        changed = replace(
            original.physical_plan, **{name: "4" * 64, "plan_identity": ""}
        )
        assert changed.plan_identity != original.physical_plan.plan_identity
        if name not in {"source_circuit_hash", "target_snapshot_id"}:
            bundle = replace(original, physical_plan=changed, bundle_identity="")
            assert bundle.bundle_identity != original.bundle_identity


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("source",),
        ("target",),
        ("output",),
        ("physical_plan",),
        ("physical_plan", "native_input_instructions", 0),
        ("physical_plan", "block_rewrites", 0),
        ("physical_plan", "instructions", 0),
    ],
)
@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_every_record_is_closed(path: tuple[str | int, ...], mutation: str) -> None:
    payload = _unsealed()
    record = payload
    for part in path:
        record = record[part]
    if mutation == "unknown":
        record["unexpected"] = None
    else:
        del record[next(iter(record))]
    with pytest.raises(ValueError, match="(missing|unknown)"):
        _read(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("input_instruction_indices", []),
        ("input_instruction_indices", [1, 0]),
        ("input_instruction_indices", [0, 0]),
        ("input_instruction_indices", [0, 3]),
        ("input_instruction_indices", list(range(65))),
        ("input_instruction_indices", [True]),
        ("output_instruction_indices", [0, 2]),
        ("output_instruction_indices", [1]),
        ("output_instruction_indices", list(range(97))),
        ("physical_qubits", [0, 0]),
        ("physical_qubits", [0]),
        ("physical_qubits", [0, 3]),
        ("baseline_cz_count", 1),
        ("baseline_cz_count", True),
        ("replacement_cz_count", 2),
        ("replacement_cz_count", -1),
        ("equivalence", "exact"),
        ("phase_real", 0.0),
        ("phase_real", True),
        ("phase_real", float("nan")),
        ("phase_imag", float("inf")),
        ("max_abs_residual", -1e-12),
        ("max_abs_residual", 1.01e-11),
    ],
)
def test_resealed_block_corruption_is_rejected(field: str, value: object) -> None:
    payload = _unsealed()
    payload["physical_plan"]["block_rewrites"][0][field] = value
    with pytest.raises((TypeError, ValueError)):
        _read(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("instruction_index", 1),
        ("native_input_instruction_index", 1),
        ("native_input_instruction_index", 3),
        ("native_input_instruction_index", True),
        ("block_rewrite_index", 0),
        ("block_replacement_ordinal", 0),
        ("opcode", "cz"),
        ("opcode", "unknown"),
        ("physical_qubits", [1]),
        ("physical_qubits", [3]),
        ("layer", 1),
        ("predecessors", [0]),
        ("dependency_kinds", ["wire"]),
    ],
)
def test_resealed_output_corruption_is_rejected(field: str, value: object) -> None:
    payload = _unsealed()
    payload["physical_plan"]["instructions"][0][field] = value
    with pytest.raises((TypeError, ValueError)):
        _read(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("instruction_index", 3),
        ("source_instruction_index", True),
        ("source_instruction_index", 4096),
        ("topology_instruction_index", 0),
        ("native_replacement_ordinal", 1),
        ("origin", "fabricated"),
        ("opcode", "cx"),
        ("opcode", "unknown"),
        ("source_logical_qubits", [3]),
        ("physical_qubits", [0, 3]),
    ],
)
def test_resealed_native_input_corruption_is_rejected(
    field: str, value: object
) -> None:
    payload = _unsealed()
    payload["physical_plan"]["native_input_instructions"][1][field] = value
    with pytest.raises((TypeError, ValueError)):
        _read(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mapping_transition_space", "final_program"),
        ("block_rewrites", []),
        ("block_rewrites", None),
        ("initial_logical_to_physical", [1, 0, 2]),
        ("final_logical_to_physical", [1, 0, 2]),
        ("pre_restore_logical_to_physical", [1, 0, 2]),
        ("native_input_circuit_hash", "invalid"),
        ("native_block_optimization_identity", None),
        ("schedule_depth", 2),
        ("maximum_parallel_width", 2),
        ("critical_path", []),
    ],
)
def test_resealed_plan_corruption_is_rejected(field: str, value: object) -> None:
    payload = _unsealed()
    payload["physical_plan"][field] = value
    with pytest.raises((TypeError, ValueError)):
        _read(payload)


def test_overlap_and_missing_input_cannot_be_hidden_by_resealing() -> None:
    payload = _unsealed()
    payload["physical_plan"]["block_rewrites"] *= 2
    with pytest.raises(ValueError, match="disjoint"):
        _read(payload)
    payload = _unsealed()
    payload["physical_plan"]["native_input_instructions"].append(
        {
            **payload["physical_plan"]["native_input_instructions"][2],
            "instruction_index": 3,
            "topology_instruction_index": 3,
        }
    )
    with pytest.raises(ValueError, match="complete"):
        _read(payload)


def test_an_unrelated_spectator_can_appear_between_block_members() -> None:
    payload = _unsealed()
    plan = payload["physical_plan"]
    native = plan["native_input_instructions"]
    native[1], native[2] = native[2], native[1]
    for index, item in enumerate(native):
        item["instruction_index"] = item["topology_instruction_index"] = index
    plan["block_rewrites"][0]["input_instruction_indices"] = [0, 2]
    plan["instructions"][0]["native_input_instruction_index"] = 1
    assert _read(payload).physical_plan.block_rewrites[0].input_instruction_indices == (
        0,
        2,
    )
    native[1]["physical_qubits"] = [0]
    plan["instructions"][0]["physical_qubits"] = [0]
    with pytest.raises(ValueError, match="intervening operation"):
        _read(payload)


def test_routing_transitions_stay_in_topology_space_after_empty_block_rewrite() -> None:
    original = _bundle()
    plan = original.physical_plan
    coupling = CouplingEvidence(3, ((0, 1), (1, 2)))
    native = tuple(
        replace(item, origin="routing_swap_decomposition") if index < 2 else item
        for index, item in enumerate(plan.native_input_instructions)
    )
    # This synthetic codec example checks replay only. Actual SWAP decomposition
    # and operator equivalence must be verified from programs by Compiler.
    transitions = (
        MappingTransitionEvidence(0, 0, "forward", (0, 1), (0, 1, 2), (1, 0, 2)),
        MappingTransitionEvidence(1, 1, "final_restore", (0, 1), (1, 0, 2), (0, 1, 2)),
    )
    routed = replace(
        plan,
        coupling=coupling,
        topology_identity=_coupling_identity(coupling),
        topology_legalization_identity=_DIGEST,
        native_input_instructions=native,
        mapping_transitions=transitions,
        pre_restore_logical_to_physical=(1, 0, 2),
        plan_identity="",
    )
    bundle = replace(original, physical_plan=routed, bundle_identity="")
    assert _read(bundle.to_dict()).physical_plan.mapping_transitions == transitions
    for replacement in (
        transitions[::-1],
        transitions[:1],
        (replace(transitions[0], routed_instruction_index=2), transitions[1]),
        (replace(transitions[0], layout_after=(0, 1, 2)), transitions[1]),
    ):
        with pytest.raises(ValueError):
            replace(routed, mapping_transitions=replacement, plan_identity="")


def test_duplicate_keys_and_byte_limit_apply_to_v4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = _bundle().to_json()
    duplicate = text.replace('"phase_real":1.0', '"phase_real":1.0,"phase_real":1.0')
    assert duplicate != text
    with pytest.raises(ValueError, match="duplicate"):
        read_compilation_evidence_bundle_json(duplicate)
    monkeypatch.setattr(v4, "_MAX_BUNDLE_BYTES", 1)
    with pytest.raises(ValueError, match="maximum UTF-8 bytes"):
        _bundle()


def test_nested_membership_budget_applies_before_record_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _unsealed()
    monkeypatch.setattr(v4, "_MAX_RECORDS", 1)
    with pytest.raises(ValueError, match="nested block memberships"):
        _read(payload)


def test_input_mutation_cannot_change_typed_evidence() -> None:
    payload = _bundle().to_dict()
    restored = _read(payload)
    expected = restored.to_json()
    payload["physical_plan"]["block_rewrites"][0]["input_instruction_indices"].append(2)
    payload["source"]["source_circuit_hash"] = "0" * 64
    assert restored.to_json() == expected
    with pytest.raises(TypeError):
        restored.source["source_circuit_hash"] = "0" * 64
    assert json.loads(_canonical_bytes(restored.to_dict())) == restored.to_dict()


def _reducing_bundle() -> v4.CompilationEvidenceBundleV4:
    from pathlib import Path

    fixture = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "docs/api-changes/examples/compiler-cz-block-lineage-v4.json"
        ).read_text()
    )
    native = tuple(
        v4.NativeInputInstructionEvidence(
            index,
            0 if index < 21 else 1,
            0 if index < 21 else 1,
            index if index < 21 else index - 21,
            "native_decomposition",
            item["opcode"],
            (0, 1) if index < 21 else (1, 0),
            tuple(item["physical_qubits"]),
        )
        for index, item in enumerate(fixture["native_input_instructions"])
    )
    outputs = []
    last_on_wire = {}
    for index, item in enumerate(fixture["proposed_final_instructions"]):
        wires = tuple(item["physical_qubits"])
        previous = tuple(
            sorted({last_on_wire[wire] for wire in wires if wire in last_on_wire})
        )
        layer = 1 + max((outputs[pred].layer for pred in previous), default=-1)
        outputs.append(
            v4.PhysicalInstructionEvidenceV4(
                index,
                None,
                0,
                index,
                item["opcode"],
                wires,
                layer,
                previous,
                ("wire",) if previous else (),
            )
        )
        for wire in wires:
            last_on_wire[wire] = index
    depth = 1 + max(item.layer for item in outputs)
    current = min(
        (item for item in outputs if item.layer == depth - 1),
        key=lambda item: item.instruction_index,
    )
    path = [current.instruction_index]
    while current.predecessors:
        index = min(current.predecessors, key=lambda pred: (-outputs[pred].layer, pred))
        path.append(index)
        current = outputs[index]
    from collections import Counter

    original = _bundle()
    plan = replace(
        original.physical_plan,
        initial_logical_to_physical=(0, 1),
        pre_restore_logical_to_physical=(0, 1),
        final_logical_to_physical=(0, 1),
        native_input_instructions=native,
        instructions=tuple(outputs),
        block_rewrites=(
            v4.BlockRewriteEvidence.from_dict(fixture["proposed_block_rewrite"]),
        ),
        schedule_depth=depth,
        maximum_parallel_width=max(Counter(item.layer for item in outputs).values()),
        critical_path=tuple(reversed(path)),
        plan_identity="",
    )
    return replace(original, physical_plan=plan, bundle_identity="")


def test_approved_swap_reversed_cx_example_roundtrips_with_collective_ancestry() -> (
    None
):
    original = _reducing_bundle()
    decoded = _read(original.to_dict())
    assert decoded.to_json() == original.to_json()
    block = decoded.physical_plan.block_rewrites[0]
    assert (block.baseline_cz_count, block.replacement_cz_count) == (4, 2)
    assert len(decoded.physical_plan.native_input_instructions) == 28
    assert len(decoded.physical_plan.instructions) == 23
    assert all(
        item.native_input_instruction_index is None
        for item in decoded.physical_plan.instructions
    )
    # The collective input set reaches both source gates. No output invents a
    # single source instruction for the SWAP/CX fusion.
    assert {
        decoded.physical_plan.native_input_instructions[index].source_instruction_index
        for index in block.input_instruction_indices
    } == {0, 1}


@pytest.mark.parametrize(
    "field,value",
    [
        ("block_rewrite_index", 1),
        ("block_replacement_ordinal", 0),
        ("native_input_instruction_index", 0),
        ("opcode", "ry"),
        ("predecessors", []),
        ("dependency_kinds", []),
        ("layer", 0),
    ],
)
def test_nonempty_rewrite_rejects_resealed_reference_and_schedule_tampering(
    field: str, value: object
) -> None:
    payload = _reducing_bundle().to_dict()
    payload["bundle_identity"] = payload["physical_plan"]["plan_identity"] = ""
    payload["physical_plan"]["instructions"][1][field] = value
    with pytest.raises(ValueError):
        _read(payload)


def test_changed_native_ancestry_and_unchanged_output_both_bind_identity() -> None:
    payload = _bundle().to_dict()
    payload["physical_plan"]["native_input_instructions"][2][
        "source_instruction_index"
    ] = 3
    with pytest.raises(ValueError, match="plan_identity"):
        _read(payload)
    payload = _reducing_bundle().to_dict()
    payload["physical_plan"]["block_rewrites"][0]["max_abs_residual"] = 1e-12
    with pytest.raises(ValueError, match="plan_identity"):
        _read(payload)


def test_direct_construction_and_reader_apply_the_same_membership_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _reducing_bundle()
    monkeypatch.setattr(v4, "_MAX_RECORDS", 50)
    with pytest.raises(ValueError, match="nested block memberships"):
        replace(original.physical_plan, plan_identity="")
    with pytest.raises(ValueError, match="nested block memberships"):
        _read(original.to_dict())


@pytest.mark.parametrize(
    "field,value",
    [
        ("native_input_instruction_index", None),
        ("dependency_kinds", ["unknown"]),
        ("dependency_kinds", ["wire", "wire"]),
    ],
)
def test_output_cannot_drop_ownership_or_forge_dependency_kinds(
    field: str, value: object
) -> None:
    payload = _unsealed()
    payload["physical_plan"]["instructions"][0][field] = value
    with pytest.raises(ValueError):
        _read(payload)


@pytest.mark.parametrize(
    "section,field",
    [
        ("source", "source_circuit_hash"),
        ("source", "final_circuit_hash"),
        ("target", "snapshot_id"),
    ],
)
def test_resealed_bundle_rejects_stale_plan_handoff(section: str, field: str) -> None:
    payload = _unsealed()
    if section == "target":
        # The target is encoded once in the envelope; its binding to the plan
        # is the retained plan identity, not a duplicated target field.
        payload["physical_plan"][
            "plan_identity"
        ] = _bundle().physical_plan.plan_identity
    payload[section][field] = "9" * 64
    failure = "plan_identity" if section == "target" else "lineage is inconsistent"
    with pytest.raises(ValueError, match=failure):
        _read(payload)


def test_output_digest_cannot_change_without_a_new_bundle_identity() -> None:
    payload = _bundle().to_dict()
    payload["output"]["payload_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="bundle_identity"):
        _read(payload)
    payload["schema"] = "unrecognized"
    with pytest.raises(ValueError, match="schema"):
        _read(payload)
    with pytest.raises(ValueError, match="version"):
        replace(_bundle(), version="3.0")


def test_block_growth_is_bounded_even_when_its_cz_cost_improves() -> None:
    block = _bundle().physical_plan.block_rewrites[0]
    with pytest.raises(ValueError, match="additional instruction budget"):
        replace(block, output_instruction_indices=tuple(range(35)))
    assert (
        len(
            replace(
                block, output_instruction_indices=tuple(range(34))
            ).output_instruction_indices
        )
        == 34
    )


def test_layout_cannot_allocate_workspace_or_hide_missing_topology_evidence() -> None:
    original = _bundle().physical_plan
    for layout in ((), (0, 2, 3)):
        with pytest.raises(ValueError, match="dense identity"):
            replace(
                original,
                initial_logical_to_physical=layout,
                final_logical_to_physical=layout,
                plan_identity="",
            )
    with pytest.raises(ValueError, match="dense permutation"):
        replace(original, pre_restore_logical_to_physical=(0, 0, 2), plan_identity="")
    with pytest.raises(ValueError, match="absent topology"):
        replace(original, topology_legalization_identity=_DIGEST, plan_identity="")
    coupling = CouplingEvidence(3, ((1, 2),))
    with pytest.raises(ValueError, match="topology identities"):
        replace(
            original,
            coupling=coupling,
            topology_identity="8" * 64,
            topology_legalization_identity=_DIGEST,
            plan_identity="",
        )
    with pytest.raises(ValueError, match="native input violates coupling"):
        replace(
            original,
            coupling=coupling,
            topology_identity=_coupling_identity(coupling),
            topology_legalization_identity=_DIGEST,
            plan_identity="",
        )


def test_rewrite_cz_claim_is_checked_against_final_records() -> None:
    payload = _reducing_bundle().to_dict()
    payload["bundle_identity"] = payload["physical_plan"]["plan_identity"] = ""
    payload["physical_plan"]["block_rewrites"][0]["replacement_cz_count"] = 1
    with pytest.raises(ValueError, match="replacement CZ count"):
        _read(payload)


def test_block_output_cannot_be_extra_or_unclaimed() -> None:
    payload = _unsealed()
    payload["physical_plan"]["instructions"][0].update(
        {
            "native_input_instruction_index": None,
            "block_rewrite_index": 0,
            "block_replacement_ordinal": 0,
        }
    )
    with pytest.raises(ValueError, match="not covered by its block"):
        _read(payload)


def test_bundle_byte_budget_includes_the_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _bundle()
    plan_size = len(_canonical_bytes(original.physical_plan.to_dict()))
    monkeypatch.setattr(v4, "_MAX_BUNDLE_BYTES", plan_size + 1)
    with pytest.raises(ValueError, match="compilation evidence exceeds maximum"):
        replace(original, bundle_identity="")


def test_a_validly_resealed_plan_cannot_reverse_shared_wire_outputs() -> None:
    plan = _bundle().physical_plan
    native = plan.native_input_instructions + (
        v4.NativeInputInstructionEvidence(3, 3, 3, 0, "source", "rz", (2,), (2,)),
    )
    outputs = (
        v4.PhysicalInstructionEvidenceV4(0, 3, None, None, "rz", (2,), 0, (), ()),
        v4.PhysicalInstructionEvidenceV4(
            1, 2, None, None, "rx", (2,), 1, (0,), ("wire",)
        ),
    )
    with pytest.raises(ValueError, match="shared wire"):
        replace(
            plan,
            native_input_instructions=native,
            instructions=outputs,
            schedule_depth=2,
            critical_path=(0, 1),
            plan_identity="",
        )


def test_schedule_cannot_omit_a_shared_wire_edge_by_rewriting_layer_summary() -> None:
    original = _reducing_bundle().physical_plan
    changed = list(original.instructions)
    changed[1] = replace(changed[1], predecessors=(), dependency_kinds=(), layer=0)
    with pytest.raises(ValueError, match="shared-wire dependency"):
        replace(original, instructions=tuple(changed), plan_identity="")


def test_scalar_range_failure_is_a_validation_error() -> None:
    block = _bundle().physical_plan.block_rewrites[0]
    with pytest.raises(ValueError, match="finite scalar range"):
        replace(block, phase_real=10**400)


def test_direct_bundle_construction_checks_target_and_plan_independently() -> None:
    original = _bundle()
    with pytest.raises(ValueError, match="lineage is inconsistent"):
        replace(
            original,
            target={**original.target, "snapshot_id": "9" * 64},
            bundle_identity="",
        )


@pytest.mark.parametrize(
    "field",
    [
        "initial_logical_to_physical",
        "pre_restore_logical_to_physical",
        "final_logical_to_physical",
    ],
)
def test_layout_arrays_cannot_exceed_the_record_budget(field: str) -> None:
    plan = _bundle().physical_plan
    with pytest.raises(ValueError, match="record limit"):
        replace(plan, **{field: tuple(range(4097)), "plan_identity": ""})

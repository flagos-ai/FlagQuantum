"""Native baseline retention, bounded synthesis and independent operator checks."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from inspect import getattr_static
from operator import setitem

import pytest
import torch

from flagquantum.compiler.lowering import native_block_optimization as blocks
from flagquantum.compiler.lowering.native_gate_legalization import (
    NativeGateLegalizationError,
    NativeGateLegalizationResult,
    legalize_native_gates,
)
from flagquantum.compiler.optimization import _cz_blocks
from flagquantum.compiler.optimization.two_qubit_synthesis import synthesize_two_qubit
from flagquantum.compiler.routing import CouplingMap
from flagquantum.compiler.topology import topology_legalization as routing
from flagquantum.core._compilation_evidence_v4 import BlockRewriteEvidence
from flagquantum.core.ir import CircuitIR, Instruction, MeasurementNode, ObservableNode
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


def _routed_stage(
    source: CircuitIR,
    *,
    strategy: str = "persistent_layout",
) -> tuple[
    blocks.NativeBlockOptimizationResult,
    NativeGateLegalizationResult,
    routing.TopologyLegalizationResult,
    CouplingMap,
]:
    snapshot = _snapshot()
    coupling = CouplingMap.line(source.n_wires)
    topology = routing.legalize_circuit_topology(
        source, coupling_map=coupling, snapshot=snapshot, strategy=strategy
    )
    native = legalize_native_gates(
        topology.program, snapshot=snapshot, evaluated_at=_NOW
    )
    result = blocks.optimize_native_blocks(
        native,
        snapshot=snapshot,
        evaluated_at=_NOW,
        topology=topology,
        coupling_map=coupling,
    )
    assert result is not None
    blocks.verify_native_block_optimization(
        result,
        native,
        snapshot=snapshot,
        evaluated_at=_NOW,
        topology=topology,
        coupling_map=coupling,
    )
    return result, native, topology, coupling


@pytest.mark.parametrize(
    "strategy",
    ["restore_after_each_gate", "persistent_layout", "sabre", "sabre_layout", "auto"],
)
def test_verified_routing_allows_native_blocks_and_preserves_layout(
    strategy: str,
) -> None:
    source = CircuitIR(
        3,
        (
            Instruction("swap", (0, 2)),
            Instruction("cx", (2, 0)),
            Instruction("cx", (1, 2)),
        ),
    )
    result, native, topology, coupling = _routed_stage(source, strategy=strategy)
    assert result.rewrites
    assert topology.inserted_swap_count > 0
    assert sum(item.name == "cz" for item in result.program.instructions) < sum(
        item.name == "cz" for item in native.program.instructions
    )
    assert all(
        len(item.wires) != 2 or coupling.has_edge(*item.wires)
        for item in result.program.instructions
    )
    _equivalent(topology.program, result.program)
    _equivalent_after_placement(
        source, result.program, topology.initial_logical_to_physical
    )
    assert topology.final_logical_to_physical == tuple(range(source.n_wires))


def _equivalent_after_placement(
    source: CircuitIR, output: CircuitIR, initial_layout: tuple[int, ...]
) -> None:
    # An initial placement permutes logical input into physical slots. Checking
    # every input basis column detects errors hidden by an all-zero input.
    size = 1 << source.n_wires
    placement = torch.zeros((size, size), dtype=torch.complex128)
    for column in range(size):
        row = sum(
            ((column >> (source.n_wires - logical - 1)) & 1)
            << (source.n_wires - physical - 1)
            for logical, physical in enumerate(initial_layout)
        )
        placement[row, column] = 1
    expected, actual = _operator(source), _operator(output) @ placement
    phase = torch.trace(actual.conj().T @ expected) / size
    assert float(abs(abs(phase) - 1)) <= 1e-11
    torch.testing.assert_close(expected, phase * actual, atol=1e-11, rtol=0)


def _routing_control(case_index: int) -> CircuitIR:
    rng = random.Random(2026112201)
    for index in range(case_index + 1):
        width = 3 if index < 75 else 4
        instructions = []
        for step in range(8 + index % 13):
            if step % 3 == 0:
                pair = tuple(rng.sample(range(width), 2))
                instructions.append(Instruction(rng.choice(("cx", "cz", "swap")), pair))
            else:
                instructions.append(
                    Instruction(
                        rng.choice(("rx", "ry", "rz")),
                        (rng.randrange(width),),
                        {"theta": rng.uniform(-3.2, 3.2)},
                    )
                )
    return CircuitIR(width, tuple(instructions))


@pytest.mark.parametrize("case_index", (0, 75, 77, 86, 89, 94, 95, 99))
@pytest.mark.parametrize(
    "strategy", ("auto", "sabre", "sabre_layout", "persistent_layout")
)
def test_routing_and_native_synthesis_preserve_full_operator_or_refuse_budget(
    case_index: int, strategy: str
) -> None:
    source = _routing_control(case_index)
    try:
        result, _, topology, coupling = _routed_stage(source, strategy=strategy)
    except NativeGateLegalizationError as error:
        # These inspected cases may exceed the unchanged baseline budget.
        # A better route may make them pass; a new refusal remains a failure.
        assert strategy == "persistent_layout"
        assert case_index in (77, 86, 89, 94, 95, 99)
        assert str(error) == "native-gate decomposition exceeds max_added_operations"
        return

    _equivalent(topology.program, result.program)
    _equivalent_after_placement(
        source, result.program, topology.initial_logical_to_physical
    )
    assert topology.final_logical_to_physical == tuple(range(source.n_wires))
    assert all(
        len(item.wires) != 2 or coupling.has_edge(*item.wires)
        for item in result.program.instructions
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {"conditions": ((0, 1),)},
        {"user_annotation": "retain"},
        {"source_instruction_index": 10, "routing_strategy": "sabre"},
        {"routing_phase": "forward", "logical_wires": (0,)},
    ],
)
def test_original_annotations_cannot_be_certified_as_router_metadata(
    metadata: dict[str, object],
) -> None:
    source = CircuitIR(
        2,
        (
            Instruction("cz", (0, 1)),
            Instruction("rz", (0,), {"theta": 0.2}, metadata=metadata),
            Instruction("cz", (1, 0)),
        ),
    )
    result, native, _, _ = _routed_stage(source)
    assert result.rewrites == ()
    assert result.program is native.program


def test_original_disjoint_annotation_remains_global_boundary_after_routing() -> None:
    source = CircuitIR(
        3,
        (
            Instruction("cz", (0, 1)),
            Instruction(
                "rz", (2,), {"theta": 0.2}, metadata={"user_annotation": "retain"}
            ),
            Instruction("cz", (1, 0)),
        ),
    )
    result, native, _, _ = _routed_stage(source)
    assert result.program is native.program


@pytest.mark.parametrize(
    "change",
    [
        "angle",
        "order",
        "logical_qubits",
        "source_index",
        "missing",
        "duplicate",
        "restore",
    ],
)
def test_resealed_tampered_routing_is_rejected_before_metadata_admission(
    change: str,
) -> None:
    source = CircuitIR(
        3,
        (
            Instruction("rx", (0,), {"theta": 0.2}),
            Instruction("cx", (0, 2)),
            Instruction("rz", (0,), {"theta": 0.4}),
        ),
    )
    snapshot = _snapshot()
    coupling = CouplingMap.line(3)
    topology = routing.legalize_circuit_topology(
        source, coupling_map=coupling, snapshot=snapshot, strategy="persistent_layout"
    )
    instructions = list(topology.program.instructions)
    source_positions = [
        index
        for index, item in enumerate(instructions)
        if "layout_mapped" in item.metadata
    ]
    first, last = source_positions[0], source_positions[-1]
    if change == "angle":
        instructions[first] = replace(instructions[first], params={"theta": 0.3})
    elif change == "order":
        instructions[first], instructions[last] = (
            instructions[last],
            instructions[first],
        )
    elif change == "logical_qubits":
        instructions[first] = replace(
            instructions[first],
            metadata=dict(instructions[first].metadata) | {"logical_wires": (2,)},
        )
    elif change == "source_index":
        instructions[first] = replace(
            instructions[first],
            metadata=dict(instructions[first].metadata)
            | {"source_instruction_index": 2},
        )
    elif change == "missing":
        del instructions[first]
    elif change == "duplicate":
        instructions.insert(first, instructions[first])
    else:
        restore = next(
            index
            for index, item in enumerate(instructions)
            if item.metadata.get("routing_phase") == "final_restore"
        )
        del instructions[restore]
    altered = replace(topology.program, instructions=tuple(instructions))
    topology = replace(
        topology,
        program=altered,
        routed_instruction_count=len(instructions),
        legalization_identity=routing._legalization_identity(
            source, altered, snapshot, topology.topology_identity, topology.strategy
        ),
    )
    native = legalize_native_gates(altered, snapshot=snapshot, evaluated_at=_NOW)
    with pytest.raises(blocks.NativeBlockOptimizationError, match="routing replay"):
        blocks.optimize_native_blocks(
            native,
            snapshot=snapshot,
            evaluated_at=_NOW,
            topology=topology,
            coupling_map=coupling,
        )


def test_routing_proof_is_required_again_during_rewrite_verification() -> None:
    result, native, topology, coupling = _routed_stage(_source())
    assert result.rewrites
    with pytest.raises(
        blocks.NativeBlockOptimizationError, match="operator equivalence"
    ):
        blocks.verify_native_block_optimization(
            result, native, snapshot=_snapshot(), evaluated_at=_NOW
        )
    with pytest.raises(blocks.NativeBlockOptimizationError, match="routing replay"):
        blocks.verify_native_block_optimization(
            result,
            native,
            snapshot=_snapshot(),
            evaluated_at=_NOW,
            topology=replace(topology, final_logical_to_physical=(1, 0)),
            coupling_map=coupling,
        )


@pytest.mark.parametrize(
    "change",
    [
        "missing_map",
        "missing_topology",
        "copied_source",
        "directed",
        "allocated",
        "wrong_map",
        "baseline",
    ],
)
def test_routing_admission_requires_the_complete_bound_stage_chain(change: str) -> None:
    _, native, topology, coupling = _routed_stage(_source())
    provided_topology: routing.TopologyLegalizationResult | None = topology
    provided_map: CouplingMap | None = coupling
    if change == "missing_map":
        provided_map = None
    elif change == "missing_topology":
        provided_topology = None
    elif change == "copied_source":
        native = replace(native, source_program=replace(native.source_program))
    elif change == "directed":
        provided_topology = replace(topology, direction_semantics="directed_cx")
    elif change == "allocated":
        provided_topology = replace(topology, allocation_identity="a" * 64)
    elif change == "wrong_map":
        provided_map = CouplingMap(2, ())
    else:
        native = replace(native, decompositions=())
    with pytest.raises(blocks.NativeBlockOptimizationError):
        blocks.optimize_native_blocks(
            native,
            snapshot=_snapshot(),
            evaluated_at=_NOW,
            topology=provided_topology,
            coupling_map=provided_map,
        )


@pytest.mark.parametrize(
    "theta",
    [
        Parameter("theta"),
        torch.tensor(0.3, requires_grad=True),
        torch.tensor([0.2, 0.3]),
    ],
)
def test_verified_routing_does_not_admit_symbolic_trainable_or_batched_values(
    theta: object,
) -> None:
    source = CircuitIR(
        2,
        (
            Instruction("cz", (0, 1)),
            Instruction("rz", (0,), {"theta": theta}),
            Instruction("cz", (0, 1)),
        ),
    )
    result, native, _, _ = _routed_stage(source)
    assert result.program is native.program
    assert result.rewrites == ()
    assert result.program.instructions[1].params["theta"] is theta


@pytest.mark.parametrize("routed", [False, True])
def test_native_decomposition_does_not_erase_a_custom_matrix_boundary(
    routed: bool,
) -> None:
    source = CircuitIR(
        2,
        (
            Instruction("cz", (0, 1)),
            Instruction("blob", (0, 1), matrix=_operator(_source()).tolist()),
            Instruction("cz", (0, 1)),
        ),
    )
    if routed:
        result, native, _, _ = _routed_stage(source)
    else:
        result, native = _stage(source)
    assert native.decompositions
    assert result.program is native.program
    assert result.rewrites == ()


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
        ("rx", "rz", "cz", "unknown_operation"),
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


@pytest.mark.parametrize("extra", [("measure",), ("reset",), ("measure", "reset")])
def test_nonunitary_target_operations_do_not_disable_cz_optimization(
    extra: tuple[str, ...],
) -> None:
    snapshot = _snapshot(("rx", "rz", "cz", *extra))
    source = replace(
        _source(), measurements=(MeasurementNode("samples", (0, 1), shots=100),)
    )
    native = legalize_native_gates(source, snapshot=snapshot, evaluated_at=_NOW)
    result = blocks.optimize_native_blocks(native, snapshot=snapshot, evaluated_at=_NOW)
    assert result is not None
    blocks.verify_native_block_optimization(
        result, native, snapshot=snapshot, evaluated_at=_NOW
    )
    assert sum(item.name == "cz" for item in result.program.instructions) == 2
    assert result.program.measurements == source.measurements
    _equivalent(_source(), result.program)


@pytest.mark.parametrize("opcode", ["bit_flip", "depolarizing"])
@pytest.mark.parametrize("qubit", [0, 2])
def test_nonunitary_operations_remain_boundaries_on_any_qubit(
    opcode: str, qubit: int
) -> None:
    held = Instruction(opcode, (qubit,), {"probability": 0.1})
    source = CircuitIR(3, (Instruction("cz", (0, 1)), held, Instruction("cz", (1, 0))))
    snapshot = _snapshot(("rx", "rz", "cz", opcode))
    native = legalize_native_gates(source, snapshot=snapshot, evaluated_at=_NOW)
    result = blocks.optimize_native_blocks(native, snapshot=snapshot, evaluated_at=_NOW)
    assert result is not None
    assert result.program is native.program
    assert result.program.instructions[1] is held
    assert result.rewrites == ()
    blocks.verify_native_block_optimization(
        result, native, snapshot=snapshot, evaluated_at=_NOW
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
    combined_limit = len(native.source_program.instructions) + 26
    assert len(valid) <= combined_limit
    padding = combined_limit + 1 - len(valid)
    padded = valid + (Instruction("rz", (0,), {"theta": 0.0}),) * padding
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


@pytest.mark.parametrize(
    "alter",
    [
        pytest.param(lambda program: replace(program, n_wires=3), id="wire-count"),
        pytest.param(
            lambda program: replace(program, dtype="complex128"), id="precision"
        ),
        pytest.param(lambda program: replace(program, shape=(2, 2)), id="shape"),
        pytest.param(
            lambda program: replace(
                program, observables=(ObservableNode("z", (1,), coefficient=-0.5),)
            ),
            id="observable",
        ),
        pytest.param(
            lambda program: replace(
                program, measurements=(MeasurementNode("samples", (1,), shots=37),)
            ),
            id="measurement",
        ),
        pytest.param(
            lambda program: replace(
                program, metadata={"purpose": {"calibration": (0.3, 0.7)}}
            ),
            id="nested-metadata",
        ),
    ],
)
def test_resealed_noninstruction_changes_are_rejected(
    alter: Callable[[CircuitIR], CircuitIR],
) -> None:
    result, native = _stage(_source())
    changed = alter(result.program)
    forged = _reseal(replace(result, program=changed))
    with pytest.raises(
        blocks.NativeBlockOptimizationError, match="noninstruction semantics"
    ):
        blocks.verify_native_block_optimization(
            forged, native, snapshot=_snapshot(), evaluated_at=_NOW
        )


def test_native_rewrites_preserve_noninstruction_tensor_semantics() -> None:
    coefficient = torch.tensor(0.41, dtype=torch.float64)
    calibration = torch.tensor([0.2, 0.8], dtype=torch.float64)
    source = replace(
        _source(),
        dtype="complex128",
        shape=(2, 2),
        observables=(ObservableNode("z", (0,), coefficient=coefficient),),
        measurements=(MeasurementNode("samples", (1, 0), shots=37),),
        metadata={"calibration": calibration, "parameter": Parameter("readout")},
    )
    result, native = _stage(source)
    assert result.rewrites
    _equivalent(source, result.program)
    for program in (source, native.program, result.program):
        payload = program.to_dict()
        payload.pop("instructions")
        expected = source.to_dict()
        expected.pop("instructions")
        assert payload == expected
    # Re-sealing the identity cannot legitimize a different observable or
    # calibration tensor, even though instruction operators are unchanged.
    for changed in (
        replace(
            result.program,
            observables=(ObservableNode("z", (0,), coefficient=coefficient + 0.1),),
        ),
        replace(
            result.program,
            metadata=dict(result.program.metadata) | {"calibration": calibration + 0.1},
        ),
    ):
        with pytest.raises(
            blocks.NativeBlockOptimizationError, match="noninstruction semantics"
        ):
            blocks.verify_native_block_optimization(
                _reseal(replace(result, program=changed)),
                native,
                snapshot=_snapshot(),
                evaluated_at=_NOW,
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
    # Deliberately cross the typed boundary to exercise runtime input validation.
    unchecked: Callable[..., blocks.NativeBlockOptimizationResult | None] = (
        blocks.optimize_native_blocks
    )
    with pytest.raises(exception):
        unchecked(
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
    # Attempt the forbidden mutation through Python's runtime assignment boundary.
    attempt_assignment: Callable[..., None] = setitem
    with pytest.raises(TypeError):
        attempt_assignment(result.rejection_counts, "not_lower_cz_cost", 0)


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
    rewrite = BlockRewriteEvidence(
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
    combined_limit = len(native.source_program.instructions) + 26
    assert len(result.program.instructions) <= combined_limit
    padding = combined_limit + 1 - len(result.program.instructions)
    padded = (
        *result.program.instructions,
        *((Instruction("rz", (0,), {"theta": 0.0}),) * padding),
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


def test_no_rewrite_hash_reuse_keeps_mutation_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    angle = torch.tensor(0.31, dtype=torch.float64)
    source = CircuitIR(2, (Instruction("rx", (0,), {"theta": angle}),))
    snapshot = _snapshot()
    native = legalize_native_gates(source, snapshot=snapshot, evaluated_at=_NOW)
    descriptor = getattr_static(CircuitIR, "content_hash")
    assert isinstance(descriptor, property)
    original_hash = descriptor.fget
    assert original_hash is not None
    calls: list[CircuitIR] = []

    def observed_hash(program: CircuitIR) -> str:
        calls.append(program)
        value = original_hash(program)
        assert isinstance(value, str)
        return value

    monkeypatch.setattr(CircuitIR, "content_hash", property(observed_hash))
    result = blocks.optimize_native_blocks(native, snapshot=snapshot, evaluated_at=_NOW)
    assert result is not None and result.program is native.program
    # Includes independent baseline replay and source/result construction checks.
    assert sum(program is native.program for program in calls) == 7
    blocks.verify_native_block_optimization(
        result, native, snapshot=snapshot, evaluated_at=_NOW
    )
    angle.add_(0.5)
    with pytest.raises(
        blocks.NativeBlockOptimizationError, match="source hash is stale"
    ):
        replace(result)
    with pytest.raises(blocks.NativeBlockOptimizationError):
        blocks.verify_native_block_optimization(
            result, native, snapshot=snapshot, evaluated_at=_NOW
        )
    fresh_native = legalize_native_gates(source, snapshot=snapshot, evaluated_at=_NOW)
    fresh = blocks.optimize_native_blocks(
        fresh_native, snapshot=snapshot, evaluated_at=_NOW
    )
    assert fresh is not None
    assert fresh.source_content_hash != result.source_content_hash
    assert fresh.optimization_identity != result.optimization_identity


def _nonreducing_cz_source() -> CircuitIR:
    return CircuitIR(
        2,
        (
            Instruction("cz", (0, 1)),
            Instruction("rx", (0,), {"theta": 0.37}),
            Instruction("rz", (1,), {"theta": -0.28}),
            Instruction("cz", (1, 0)),
        ),
    )


def test_nonreducing_cz_blocks_keep_resource_rejection_precedence() -> None:
    source = _nonreducing_cz_source()
    product = _cz_blocks._product(source.instructions, (0, 1))
    assert product is not None
    proposal = synthesize_two_qubit(
        product, qubits=(0, 1), entangler="cz", z_rotation="rz", pulse_opcode="rx"
    )
    assert proposal is not None
    count = sum(item.name == "cz" for item in proposal)
    assert count == 2
    bound = count + 10 * (count + 1)
    for budget in (0, len(proposal) - 1, len(proposal), bound, 256):
        reasons: list[str] = []
        assert (
            _cz_blocks._candidate(
                source.instructions,
                (0, 1),
                max_output_operations=budget,
                on_rejection=reasons.append,
            )
            is None
        )
        assert reasons == [
            (
                "instruction_growth_rejected"
                if budget < len(proposal)
                else "not_lower_cz_cost"
            )
        ]


def test_general_cz_raw_form_respects_its_instruction_bound() -> None:
    for seed in range(12):
        rng = random.Random(seed)
        instructions: list[Instruction] = []
        for _ in range(4):
            instructions.extend(
                (
                    Instruction("rx", (0,), {"theta": rng.uniform(-2, 2)}),
                    Instruction("rz", (1,), {"theta": rng.uniform(-2, 2)}),
                    Instruction("cz", (1, 0)),
                    Instruction("rx", (1,), {"theta": rng.uniform(-2, 2)}),
                    Instruction("rz", (0,), {"theta": rng.uniform(-2, 2)}),
                )
            )
        source = CircuitIR(2, tuple(instructions))
        proposal = synthesize_two_qubit(
            _operator(source).tolist(),
            qubits=(0, 1),
            entangler="cz",
            z_rotation="rz",
            pulse_opcode="rx",
        )
        assert proposal is not None
        count = sum(item.name == "cz" for item in proposal)
        assert count <= 3
        assert len(proposal) <= count + 10 * (count + 1)
        _equivalent(source, CircuitIR(2, proposal))


@pytest.mark.parametrize(
    ("instructions", "maximum_czs"),
    [
        pytest.param(
            (
                Instruction("cz", (0, 1)),
                Instruction("rx", (0,), {"theta": torch.pi}),
                Instruction("cz", (0, 1)),
            ),
            0,
            id="refocused-pi-x",
        ),
        pytest.param(
            (
                Instruction("rz", (0,), {"theta": 0.173}),
                Instruction("cz", (0, 1)),
                Instruction("rx", (0,), {"theta": torch.pi}),
                Instruction("cz", (0, 1)),
                Instruction("rz", (1,), {"theta": 0.379}),
            ),
            0,
            id="refocused-pi-x-with-frame",
        ),
        pytest.param(
            (
                Instruction("cz", (0, 1)),
                Instruction("rx", (0,), {"theta": torch.pi / 2}),
                Instruction("cz", (0, 1)),
            ),
            1,
            id="quarter-turn-pauli-product",
        ),
        pytest.param(
            (
                Instruction("cz", (0, 1)),
                Instruction("rx", (0,), {"theta": 0.173}),
                Instruction("cz", (0, 1)),
                Instruction("rx", (1,), {"theta": 0.379}),
                Instruction("cz", (0, 1)),
            ),
            2,
            id="mixed-wire-two-cz-reduction",
        ),
    ],
)
def test_reducing_cz_blocks_preserve_complete_operators(
    instructions: tuple[Instruction, ...], maximum_czs: int
) -> None:
    source = CircuitIR(2, instructions)
    proposal = _cz_blocks._candidate(
        source.instructions, (0, 1), max_output_operations=256
    )
    assert proposal is not None
    assert sum(item.name == "cz" for item in proposal) <= maximum_czs
    _equivalent(source, CircuitIR(2, proposal))


def test_native_matrix_reads_preserve_the_operand_position_of_shared_wires() -> None:
    source = CircuitIR(
        3,
        (
            Instruction("rz", (1,), {"theta": 0.37}),
            Instruction("cz", (0, 1)),
            Instruction("rx", (1,), {"theta": 0.29}),
            Instruction("cz", (1, 0)),
            Instruction("rz", (1,), {"theta": 0.48}),
            Instruction("cz", (1, 2)),
            Instruction("rx", (1,), {"theta": 0.91}),
            Instruction("cz", (2, 1)),
        ),
    )
    result, _ = _stage(source)
    _equivalent(source, result.program)


@pytest.mark.parametrize("storage", ("tensor", "mapping"))
def test_rewritten_blocks_recheck_mutable_angles_in_later_analyses(
    storage: str,
) -> None:
    angle = torch.tensor(0.31, dtype=torch.float64)
    rotation = Instruction(
        "rx", (0,), {"theta": angle if storage == "tensor" else 0.31}
    )
    source = replace(_source(), instructions=(*_source().instructions, rotation))
    result, native = _stage(source)
    assert result.rewrites
    if storage == "tensor":
        angle.add_(0.5)
    else:
        assert isinstance(rotation.params, dict)
        rotation.params["theta"] = 0.81
    with pytest.raises(blocks.NativeBlockOptimizationError):
        blocks.verify_native_block_optimization(
            result, native, snapshot=_snapshot(), evaluated_at=_NOW
        )
    fresh, _ = _stage(source)
    assert fresh.source_content_hash != result.source_content_hash
    _equivalent(source, fresh.program)

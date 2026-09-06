from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
import torch

import flagquantum as fq
from flagquantum._compiler.exporters.circuit_ir import export_transformed_circuit_ir
from flagquantum._compiler.offline_deployment import (
    OfflineCompilationStatus,
    OfflineStaticTarget,
    OfflineTextFormat,
    compile_offline_static,
)
from flagquantum._compiler.passes.placement_routing import DirectedCouplingGraph
from flagquantum._compiler.pipeline_cache import BoundedPipelineCache, CacheDisposition
from flagquantum._compiler.testing.differential import lower_module_for_differential
from flagquantum.compiler import optimize

pytestmark = pytest.mark.unit


def _source() -> fq.CircuitIR:
    return fq.CircuitIR(
        3,
        (
            fq.Instruction("h", (0,)),
            fq.Instruction("h", (0,)),
            fq.Instruction(
                "rx",
                (1,),
                {"theta": 0.125},
                metadata={"source": "first-rotation"},
            ),
            fq.Instruction("rx", (1,), {"theta": 0.25}),
            fq.Instruction("cx", (0, 2), metadata={"source": "entangler"}),
        ),
        dtype="complex128",
        metadata={"source": "compiler-golden-path"},
    )


def _target(*, calibration_identity: str = "a" * 64) -> OfflineStaticTarget:
    return OfflineStaticTarget(
        DirectedCouplingGraph(
            3,
            ((0, 1), (1, 0), (1, 2), (2, 1)),
        ),
        calibration_identity,
    )


def _compile(source: fq.CircuitIR, *, cache: BoundedPipelineCache | None = None):
    result = compile_offline_static(source, _target(), cache=cache)
    assert result.ok, result.diagnostics
    assert result.source_artifact is not None
    assert result.module is not None
    assert result.execution is not None
    assert result.execution.pipeline is not None
    assert result.execution.identity is not None
    return result


def _assert_state_equivalent(actual: fq.CircuitIR, expected: fq.CircuitIR) -> None:
    actual_state = fq.run(actual).state.reshape(-1)
    expected_state = fq.run(expected).state.reshape(-1)
    pivot = int(torch.argmax(torch.abs(expected_state)).item())
    phase = actual_state[pivot] / expected_state[pivot]
    torch.testing.assert_close(actual_state, phase * expected_state, atol=1e-12, rtol=0)


def test_static_pipeline_is_deterministic_for_identical_inputs() -> None:
    source = _source()
    cache = BoundedPipelineCache(max_entries=2)

    first = _compile(source, cache=cache)
    second = _compile(source, cache=cache)

    assert first.execution.cache_disposition is CacheDisposition.MISS
    assert second.execution.cache_disposition is CacheDisposition.HIT
    assert first.execution.identity == second.execution.identity
    assert first.module.canonical() == second.module.canonical()
    assert first.emissions == second.emissions


def test_static_pipeline_matches_existing_compiler_semantics() -> None:
    source = _source()
    candidate = _compile(source)
    lowered = export_transformed_circuit_ir(
        candidate.source_artifact,
        candidate.module,
    )

    assert lowered.ok, lowered.diagnostics
    assert lowered.circuit_ir is not None
    stable = optimize(source)
    assert lowered.circuit_ir.metadata == stable.metadata
    candidate_metadata = tuple(
        item.metadata for item in lowered.circuit_ir.instructions
    )
    assert all(
        instruction.metadata in candidate_metadata
        for instruction in stable.instructions
    )
    _assert_state_equivalent(lowered.circuit_ir, source)
    _assert_state_equivalent(lowered.circuit_ir, stable)


def test_static_pipeline_fails_closed_without_partial_artifacts() -> None:
    invalid = compile_offline_static(object(), _target())
    unsupported = compile_offline_static(
        replace(
            _source(),
            measurements=(fq.MeasurementNode("sample", (0,), shots=8),),
        ),
        _target(),
    )

    assert invalid.status is OfflineCompilationStatus.INVALID_INPUT
    assert invalid.source_artifact is None
    assert invalid.module is None
    assert invalid.execution is None
    assert invalid.emissions == ()
    assert invalid.diagnostics

    assert unsupported.status is OfflineCompilationStatus.UNSUPPORTED_WITH_DIAGNOSTICS
    assert unsupported.source_artifact is None
    assert unsupported.module is None
    assert unsupported.execution is None
    assert unsupported.emissions == ()
    assert unsupported.diagnostics


def test_static_pipeline_preserves_opaque_top_level_metadata() -> None:
    source = replace(_source(), metadata={"application_tag": "chemistry"})

    assert optimize(source).metadata == source.metadata

    candidate = _compile(source)
    lowered = lower_module_for_differential(
        candidate.source_artifact,
        candidate.module,
    )

    assert lowered.ok, lowered.diagnostics
    assert lowered.circuit_ir is not None
    assert lowered.circuit_ir.metadata == source.metadata


def test_static_pipeline_matches_opaque_instruction_metadata_behavior() -> None:
    source = fq.CircuitIR(
        3,
        (
            fq.Instruction(
                "rx",
                (0,),
                {"theta": 0.5},
                metadata={"application_tag": "kept"},
            ),
            fq.Instruction("rx", (1,), {"theta": 0.125}, metadata={"step": 1}),
            fq.Instruction("rx", (1,), {"theta": 0.25}, metadata={"step": 2}),
            fq.Instruction("x", (2,), metadata={"pair": "first"}),
            fq.Instruction("x", (2,), metadata={"pair": "second"}),
        ),
    )
    stable = optimize(source)
    candidate = _compile(source)
    lowered = lower_module_for_differential(
        candidate.source_artifact,
        candidate.module,
    )

    assert lowered.ok, lowered.diagnostics
    assert lowered.circuit_ir is not None
    assert tuple(item.name for item in lowered.circuit_ir) == tuple(
        item.name for item in stable
    )
    assert tuple(item.metadata for item in lowered.circuit_ir) == tuple(
        item.metadata for item in stable
    )


def test_static_pipeline_binds_source_pipeline_target_and_emission_identities() -> None:
    source = _source()
    result = _compile(source)
    identity = result.execution.identity
    source_artifact = result.source_artifact

    assert source_artifact.source_content_hash == source.content_hash
    assert source_artifact.internal_program_identity == (
        source_artifact.imported.internal_program_identity
    )
    assert identity.source_identity == source.content_hash
    assert (
        identity.input_program_identity
        == source_artifact.imported.module.program_identity
    )
    assert identity.pipeline_digest == result.execution.pipeline.pipeline_digest
    for _, emission in result.emissions:
        assert emission.text is not None
        assert (
            emission.content_hash == hashlib.sha256(emission.text.encode()).hexdigest()
        )

    changed_target = compile_offline_static(
        source,
        _target(calibration_identity="b" * 64),
        output_formats=(OfflineTextFormat.OPENQASM2,),
    )
    assert changed_target.ok
    assert changed_target.execution is not None
    assert changed_target.execution.identity is not None
    assert changed_target.execution.identity.source_identity == identity.source_identity
    assert changed_target.execution.identity.input_program_identity == (
        identity.input_program_identity
    )
    assert changed_target.execution.identity.digest != identity.digest

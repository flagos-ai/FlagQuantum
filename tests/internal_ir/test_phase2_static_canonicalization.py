from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import flagquantum as fq
from flagquantum._compiler.exporters.circuit_ir import seal_circuit_ir_round_trip
from flagquantum._compiler.ir.modules import Block, QuantumModule, Region
from flagquantum._compiler.passes.manager import PassManager
from flagquantum._compiler.passes.static_canonicalization import (
    CancelSelfInverseOperationsPass,
    MergeAdjacentRotationsPass,
    RemoveIdentityOperationsPass,
    phase2_batch_a_passes,
)
from flagquantum._compiler.testing.differential import (
    lower_module_for_differential,
)
from flagquantum.compiler import simple_compile

pytestmark = pytest.mark.unit


def _compile_candidate(
    source: fq.CircuitIR,
) -> tuple[object, fq.CircuitIR]:
    sealed = seal_circuit_ir_round_trip(source)
    assert sealed.ok and sealed.artifact is not None
    pipeline = PassManager(phase2_batch_a_passes()).run(sealed.artifact.imported.module)
    assert pipeline.ok, pipeline.diagnostics
    lowered = lower_module_for_differential(sealed.artifact, pipeline.module)
    assert lowered.ok and lowered.circuit_ir is not None
    assert lowered.binding_identity_preserved is True
    return pipeline, lowered.circuit_ir


def test_batch_a_matches_legacy_structure_across_disjoint_wires() -> None:
    source = fq.CircuitIR(
        2,
        (
            fq.Instruction("i", (0,)),
            fq.Instruction("x", (0,)),
            fq.Instruction("h", (1,)),
            fq.Instruction("x", (0,)),
            fq.Instruction("rx", (0,), {"theta": 0.1}),
            fq.Instruction("z", (1,)),
            fq.Instruction("rx", (0,), {"theta": 0.2}),
            fq.Instruction("ry", (1,), {"theta": 0.0}),
        ),
    )

    pipeline, candidate = _compile_candidate(source)
    legacy = simple_compile(source)

    assert candidate.instructions == legacy.instructions
    assert tuple(item.name for item in candidate.instructions) == ("h", "rx", "z")
    assert candidate.instructions[1].params["theta"] == pytest.approx(0.3)
    assert pipeline.module.program_identity != (
        seal_circuit_ir_round_trip(source).artifact.internal_program_identity
    )


def test_fused_pipeline_is_exactly_equivalent_to_independent_passes() -> None:
    source = fq.CircuitIR(
        3,
        (
            fq.Instruction("i", (0,)),
            fq.Instruction("h", (1,)),
            fq.Instruction("h", (1,)),
            fq.Instruction("rx", (2,), {"theta": 0.1}),
            fq.Instruction("rx", (2,), {"theta": 0.2}),
            fq.Instruction("x", (0,)),
        ),
    )
    sealed = seal_circuit_ir_round_trip(source)
    assert sealed.ok and sealed.artifact is not None

    independent = PassManager(phase2_batch_a_passes()).run(
        sealed.artifact.imported.module
    )
    fused = PassManager(phase2_batch_a_passes(fused=True)).run(
        sealed.artifact.imported.module
    )
    assert independent.ok and fused.ok
    assert fused.module == independent.module
    assert fused.module.canonical() == independent.module.canonical()
    assert fused.module.program_identity == independent.module.program_identity

    independent_lowered = lower_module_for_differential(
        sealed.artifact, independent.module
    )
    fused_lowered = lower_module_for_differential(sealed.artifact, fused.module)
    assert independent_lowered.ok and fused_lowered.ok
    assert fused_lowered.circuit_ir == independent_lowered.circuit_ir


@pytest.mark.parametrize(
    "compiler_pass",
    (
        RemoveIdentityOperationsPass(),
        CancelSelfInverseOperationsPass(),
        MergeAdjacentRotationsPass(),
    ),
)
def test_each_batch_a_pass_is_idempotent(compiler_pass: object) -> None:
    source = fq.CircuitIR(
        2,
        (
            fq.Instruction("i", (0,)),
            fq.Instruction("x", (0,)),
            fq.Instruction("h", (1,)),
            fq.Instruction("x", (0,)),
            fq.Instruction("rz", (1,), {"theta": 0.1}),
            fq.Instruction("rz", (1,), {"theta": 0.2}),
        ),
    )
    artifact = seal_circuit_ir_round_trip(source).artifact

    first = PassManager((compiler_pass,)).run(artifact.imported.module)
    second = PassManager((compiler_pass,)).run(first.module)

    assert first.ok and second.ok
    assert second.module is first.module
    assert second.pass_results[0].changed is False


def test_trainable_rotation_merge_preserves_forward_and_both_gradients() -> None:
    theta = torch.tensor(0.2, requires_grad=True)
    phi = torch.tensor(-0.1, requires_grad=True)
    source = fq.CircuitIR(
        2,
        (
            fq.Instruction("ry", (0,), {"theta": theta}),
            fq.Instruction("h", (1,)),
            fq.Instruction("ry", (0,), {"theta": phi}),
        ),
        measurements=(fq.MeasurementNode("expectation_z", (0,)),),
    )

    _, candidate = _compile_candidate(source)
    legacy = simple_compile(source)
    legacy_loss = fq.Circuit.from_ir(legacy).expectation_z(0).sum()
    candidate_loss = fq.Circuit.from_ir(candidate).expectation_z(0).sum()
    legacy_gradients = torch.autograd.grad(legacy_loss, (theta, phi), retain_graph=True)
    candidate_gradients = torch.autograd.grad(candidate_loss, (theta, phi))

    torch.testing.assert_close(legacy_loss, candidate_loss)
    for legacy_gradient, candidate_gradient in zip(
        legacy_gradients, candidate_gradients, strict=True
    ):
        torch.testing.assert_close(legacy_gradient, candidate_gradient)
    assert candidate.instructions[0].params["theta"].requires_grad


def test_zero_initialized_trainable_rotation_is_not_removed() -> None:
    theta = torch.tensor(0.0, requires_grad=True)
    source = fq.CircuitIR(
        1,
        (fq.Instruction("rx", (0,), {"theta": theta}),),
    )

    _, candidate = _compile_candidate(source)

    assert len(candidate.instructions) == 1
    assert candidate.instructions[0].params["theta"] is theta


@pytest.mark.parametrize(
    ("dtype", "atol"),
    (("complex64", 1e-6), ("complex128", 1e-12)),
)
def test_batch_a_state_and_expectation_parity(dtype: str, atol: float) -> None:
    source = fq.CircuitIR(
        2,
        (
            fq.Instruction("h", (0,)),
            fq.Instruction("h", (0,)),
            fq.Instruction("rx", (1,), {"theta": 0.17}),
            fq.Instruction("rx", (1,), {"theta": 0.23}),
            fq.Instruction("cx", (1, 0)),
        ),
        dtype=dtype,
        measurements=(fq.MeasurementNode("expectation_z", (1, 0)),),
    )

    _, candidate = _compile_candidate(source)
    legacy_result = fq.run(simple_compile(source))
    candidate_result = fq.run(candidate)

    torch.testing.assert_close(
        legacy_result.state, candidate_result.state, atol=atol, rtol=0
    )
    torch.testing.assert_close(
        legacy_result.expectation(),
        candidate_result.expectation(),
        atol=atol,
        rtol=0,
    )


def test_batch_a_fails_closed_for_non_single_block_module() -> None:
    source = fq.CircuitIR(1, (fq.Instruction("x", (0,)),))
    artifact = seal_circuit_ir_round_trip(source).artifact
    block = artifact.imported.module.body.blocks[0]
    unsupported = QuantumModule(Region((block, Block())))

    result = PassManager((RemoveIdentityOperationsPass(),)).run(unsupported)

    assert not result.ok
    assert result.module is unsupported
    assert "one block is required" in result.diagnostics[0].message


def test_optimized_lowering_rejects_unsealed_source_provenance() -> None:
    source = fq.CircuitIR(1, (fq.Instruction("x", (0,)),))
    artifact = seal_circuit_ir_round_trip(source).artifact
    operation = replace(
        artifact.imported.module.body.blocks[0].operations[0], location=None
    )
    module = QuantumModule(
        Region(
            (Block(artifact.imported.module.body.blocks[0].arguments, (operation,)),)
        )
    )

    lowered = lower_module_for_differential(artifact, module)

    assert not lowered.ok
    assert "source provenance" in lowered.diagnostics[0].message

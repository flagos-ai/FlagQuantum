from __future__ import annotations

import math

import pytest
import torch

import flagquantum as fq
from flagquantum._compiler.exporters.circuit_ir import seal_circuit_ir_round_trip
from flagquantum._compiler.passes.manager import PassManager
from flagquantum._compiler.passes.static_canonicalization import (
    StaticCanonicalizationPass,
)
from flagquantum._compiler.passes.target_decomposition import (
    UNIVERSAL_RX_RY_RZ_CX_V1,
    DecomposeToTargetGateSetPass,
)
from flagquantum._compiler.testing.differential import lower_module_for_differential
from flagquantum.core.operator_schema import OPERATOR_SCHEMAS

pytestmark = pytest.mark.unit


def _decompose(source: fq.CircuitIR):
    sealed = seal_circuit_ir_round_trip(source)
    assert sealed.ok and sealed.artifact is not None
    pipeline = PassManager(
        (StaticCanonicalizationPass(), DecomposeToTargetGateSetPass())
    ).run(sealed.artifact.imported.module)
    assert pipeline.ok, pipeline.diagnostics
    lowered = lower_module_for_differential(sealed.artifact, pipeline.module)
    assert lowered.ok and lowered.circuit_ir is not None
    return sealed.artifact, pipeline, lowered.circuit_ir


def _assert_state_equal_up_to_global_phase(
    actual: torch.Tensor, expected: torch.Tensor, *, atol: float = 1e-11
) -> None:
    actual = actual.reshape(-1)
    expected = expected.reshape(-1)
    pivot = int(torch.argmax(torch.abs(expected)).item())
    phase = actual[pivot] / expected[pivot]
    torch.testing.assert_close(actual, phase * expected, atol=atol, rtol=0)


def test_profile_covers_every_static_unitary_after_batch_a_identity_removal() -> None:
    expected = {
        f"quantum.{name}"
        for name, schema in OPERATOR_SCHEMAS.items()
        if schema.unitary and name != "i"
    }

    assert set(UNIVERSAL_RX_RY_RZ_CX_V1.certified_source_operations) == expected
    assert set(UNIVERSAL_RX_RY_RZ_CX_V1.native_operations) == {
        "quantum.rx",
        "quantum.ry",
        "quantum.rz",
        "quantum.cx",
    }


@pytest.mark.parametrize(
    ("name", "wires", "params"),
    (
        ("x", (0,), {}),
        ("y", (0,), {}),
        ("z", (0,), {}),
        ("h", (0,), {}),
        ("s", (0,), {}),
        ("sdg", (0,), {}),
        ("t", (0,), {}),
        ("tdg", (0,), {}),
        ("sx", (0,), {}),
        ("sxdg", (0,), {}),
        ("phase", (0,), {"theta": 0.31}),
        ("u1", (0,), {"theta": -0.27}),
        ("u2", (0,), {"phi": 0.2, "lbd": -0.4}),
        ("u3", (0,), {"theta": 0.3, "phi": -0.2, "lbd": 0.5}),
        ("cy", (0, 1), {}),
        ("cz", (0, 1), {}),
        ("swap", (0, 1), {}),
        ("crx", (0, 1), {"theta": 0.37}),
        ("cry", (0, 1), {"theta": -0.23}),
        ("crz", (0, 1), {"theta": 0.41}),
        ("cphase", (0, 1), {"theta": -0.29}),
        ("rxx", (0, 1), {"theta": 0.19}),
        ("ryy", (0, 1), {"theta": -0.33}),
        ("rzz", (0, 1), {"theta": 0.47}),
        ("ccx", (0, 1, 2), {}),
        ("cswap", (0, 1, 2), {}),
    ),
)
def test_each_certified_rewrite_matches_source_state(
    name: str, wires: tuple[int, ...], params: dict[str, float]
) -> None:
    n_wires = max(wires) + 1
    preparation = tuple(
        fq.Instruction("ry", (wire,), {"theta": 0.2 + wire}) for wire in range(n_wires)
    )
    source = fq.CircuitIR(
        n_wires,
        (*preparation, fq.Instruction(name, wires, params)),
        dtype="complex128",
    )

    _, _, candidate = _decompose(source)
    assert set(item.name for item in candidate.instructions) <= {
        name.removeprefix("quantum.")
        for name in UNIVERSAL_RX_RY_RZ_CX_V1.native_operations
    }
    _assert_state_equal_up_to_global_phase(
        fq.run(candidate).state,
        fq.run(source).state,
    )


def test_target_decomposition_is_idempotent_and_identity_deterministic() -> None:
    source = fq.CircuitIR(
        2,
        (
            fq.Instruction("h", (0,)),
            fq.Instruction("rzz", (0, 1), {"theta": 0.25}),
        ),
    )
    artifact, first, _ = _decompose(source)
    second = PassManager((DecomposeToTargetGateSetPass(),)).run(first.module)

    assert second.ok
    assert second.module is first.module
    assert second.pass_results[0].changed is False
    assert first.module.program_identity == (
        PassManager((StaticCanonicalizationPass(), DecomposeToTargetGateSetPass()))
        .run(artifact.imported.module)
        .module.program_identity
    )


def test_trainable_parameter_rewrite_preserves_expectation_and_gradient() -> None:
    theta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    source = fq.CircuitIR(
        2,
        (
            fq.Instruction("ry", (0,), {"theta": math.pi / 3}),
            fq.Instruction("rx", (1,), {"theta": math.pi / 5}),
            fq.Instruction("rxx", (0, 1), {"theta": theta}),
        ),
        dtype="complex128",
        measurements=(fq.MeasurementNode("expectation_z", (0,)),),
    )
    _, _, candidate = _decompose(source)

    source_loss = fq.Circuit.from_ir(source).expectation_z(0).sum()
    source_gradient = torch.autograd.grad(source_loss, theta, retain_graph=True)[0]
    candidate_loss = fq.Circuit.from_ir(candidate).expectation_z(0).sum()
    candidate_gradient = torch.autograd.grad(candidate_loss, theta)[0]

    torch.testing.assert_close(candidate_loss, source_loss, atol=1e-6, rtol=0)
    torch.testing.assert_close(candidate_gradient, source_gradient, atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    ("name", "wires", "parameter_values"),
    (
        ("phase", (0,), {"theta": 0.31}),
        ("u1", (0,), {"theta": -0.27}),
        ("u2", (0,), {"phi": 0.2, "lbd": -0.4}),
        ("u3", (0,), {"theta": 0.3, "phi": -0.2, "lbd": 0.5}),
        ("crx", (0, 1), {"theta": 0.37}),
        ("cry", (0, 1), {"theta": -0.23}),
        ("crz", (0, 1), {"theta": 0.41}),
        ("cphase", (0, 1), {"theta": -0.29}),
        ("rxx", (0, 1), {"theta": 0.19}),
        ("ryy", (0, 1), {"theta": -0.33}),
        ("rzz", (0, 1), {"theta": 0.47}),
    ),
)
def test_every_parameterized_rewrite_preserves_trainable_gradients(
    name: str,
    wires: tuple[int, ...],
    parameter_values: dict[str, float],
) -> None:
    parameters = {
        key: torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for key, value in parameter_values.items()
    }
    n_wires = max(wires) + 1
    source = fq.CircuitIR(
        n_wires,
        (
            *(
                fq.Instruction("ry", (wire,), {"theta": 0.31 + wire})
                for wire in range(n_wires)
            ),
            fq.Instruction(name, wires, parameters),
            fq.Instruction("h", (wires[-1],)),
        ),
        dtype="complex128",
    )
    _, _, candidate = _decompose(source)
    ordered_parameters = tuple(parameters.values())

    source_loss = fq.Circuit.from_ir(source).expectation_z(wires[-1]).sum()
    source_gradients = torch.autograd.grad(
        source_loss, ordered_parameters, retain_graph=True
    )
    candidate_loss = fq.Circuit.from_ir(candidate).expectation_z(wires[-1]).sum()
    candidate_gradients = torch.autograd.grad(candidate_loss, ordered_parameters)

    torch.testing.assert_close(candidate_loss, source_loss, atol=1e-6, rtol=0)
    for candidate_gradient, source_gradient in zip(
        candidate_gradients, source_gradients, strict=True
    ):
        torch.testing.assert_close(
            candidate_gradient,
            source_gradient,
            atol=1e-6,
            rtol=0,
        )


@pytest.mark.parametrize("unsupported", ("bit_flip", "custom"))
def test_unsupported_target_operation_fails_closed(unsupported: str) -> None:
    instruction = (
        fq.Instruction("bit_flip", (0,))
        if unsupported == "bit_flip"
        else fq.Instruction("private_u", (0,), matrix=torch.eye(2))
    )
    source = fq.CircuitIR(1, (instruction,))
    sealed = seal_circuit_ir_round_trip(source)
    assert sealed.ok and sealed.artifact is not None

    result = PassManager((DecomposeToTargetGateSetPass(),)).run(
        sealed.artifact.imported.module
    )

    assert not result.ok
    assert result.module is sealed.artifact.imported.module
    assert "no certified decomposition" in result.diagnostics[0].message

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path

import pytest
import torch

import flagquantum as fq
from flagquantum._compiler.bindings import RuntimeBindingRef
from flagquantum._compiler.importers.circuit_ir import import_circuit_ir
from flagquantum._compiler.passes.placement_routing import DirectedCouplingGraph
from flagquantum._compiler.target_capabilities import (
    ArtifactFormat,
    ArtifactProfile,
    GateCapability,
    MeasurementResult,
    ParameterConstraint,
    TargetCapabilities,
    TargetClass,
)
from flagquantum._compiler.target_ir import TargetIR, TargetOperation
from flagquantum._compiler.target_legalization import (
    TargetLegalizationStatus,
    legalize_quantum_module,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]


def _target(
    *,
    gates: tuple[GateCapability, ...] | None = None,
    edges: tuple[tuple[int, int], ...] = ((0, 1), (1, 0), (1, 2), (2, 1)),
    parameter_binding: bool = True,
) -> TargetCapabilities:
    return TargetCapabilities(
        target_class=TargetClass.LOCAL_RUNTIME,
        logical_qubit_capacity=3,
        physical_qubit_capacity=3,
        native_gates=gates
        or (
            GateCapability("h"),
            GateCapability("rx"),
            GateCapability("ry"),
            GateCapability("rz"),
            GateCapability("cx"),
        ),
        measurement_results=(
            MeasurementResult.STATE,
            MeasurementResult.EXPECTATION,
        ),
        artifact_profiles=(ArtifactProfile(ArtifactFormat.RUNTIME_PLAN, "1.0"),),
        topology=DirectedCouplingGraph(3, edges),
        supports_parameter_binding=parameter_binding,
        maximum_program_operations=10000,
        maximum_shots=1000,
    )


def _import(source: fq.CircuitIR):
    result = import_circuit_ir(source)
    assert result.ok and result.imported is not None
    return result.imported


def _native_source(theta: object = 0.37) -> fq.CircuitIR:
    return fq.CircuitIR(
        3,
        (
            fq.Instruction("h", (0,)),
            fq.Instruction("cx", (0, 1)),
            fq.Instruction("rx", (1,), {"theta": theta}),
            fq.Instruction("rz", (2,), {"theta": -0.21}),
        ),
        dtype="complex128",
    )


def test_legalization_materializes_typed_target_ir_and_identity() -> None:
    imported = _import(_native_source())
    target = _target()
    result = legalize_quantum_module(
        imported.module,
        target,
        required_results=(MeasurementResult.STATE,),
    )

    assert result.ok and result.target_ir is not None
    assert result.status is TargetLegalizationStatus.LEGALIZED
    assert result.target_ir.source_program_identity == imported.module.program_identity
    assert result.target_ir.target_capability_fingerprint == (
        target.semantic_fingerprint
    )
    assert result.target_ir.logical_to_physical == (0, 1, 2)
    assert tuple(item.operation for item in result.target_ir.operations) == (
        "h",
        "cx",
        "rx",
        "rz",
    )
    assert len(result.target_ir.target_program_identity) == 64


def test_target_ir_is_immutable_and_debug_location_is_non_semantic() -> None:
    imported = _import(_native_source())
    target_ir = legalize_quantum_module(imported.module, _target()).target_ir
    assert target_ir is not None
    first = target_ir.operations[0]
    changed_location = replace(first, location=replace(first.location, line=99))
    changed = replace(
        target_ir, operations=(changed_location, *target_ir.operations[1:])
    )

    assert changed.target_program_identity == target_ir.target_program_identity
    with pytest.raises(FrozenInstanceError):
        target_ir.logical_to_physical = (2, 1, 0)


def test_layout_and_directed_wire_order_are_explicit() -> None:
    imported = _import(_native_source())
    target = _target(edges=((2, 0), (0, 2), (0, 1), (1, 0)))
    result = legalize_quantum_module(
        imported.module,
        target,
        logical_to_physical=(2, 0, 1),
    )

    assert result.ok and result.target_ir is not None
    assert result.target_ir.operations[1].physical_qubits == (2, 0)
    rejected = legalize_quantum_module(
        imported.module,
        _target(edges=((0, 2), (0, 1), (1, 0))),
        logical_to_physical=(2, 0, 1),
    )
    assert not rejected.ok
    assert "directed target edge 2->0" in rejected.diagnostics[0].message


def test_unsupported_gate_result_capacity_and_layout_fail_closed() -> None:
    imported = _import(_native_source())
    gates = tuple(item for item in _target().native_gates if item.operation != "h")
    cases = (
        legalize_quantum_module(imported.module, _target(gates=gates)),
        legalize_quantum_module(
            imported.module,
            replace(
                _target(),
                measurement_results=(MeasurementResult.STATE,),
            ),
            required_results=(MeasurementResult.EXPECTATION,),
        ),
        legalize_quantum_module(
            imported.module, _target(), logical_to_physical=(0, 0, 1)
        ),
        legalize_quantum_module(imported.module, _target(), logical_to_physical=(0, 1)),
    )

    assert all(not item.ok and item.target_ir is None for item in cases)
    assert all(item.diagnostics for item in cases)


def test_shot_and_program_limits_fail_closed_and_affect_identity() -> None:
    imported = _import(_native_source())
    accepted = legalize_quantum_module(
        imported.module,
        _target(),
        required_results=(MeasurementResult.STATE,),
        requested_shots=100,
    )
    changed = legalize_quantum_module(
        imported.module,
        _target(),
        required_results=(MeasurementResult.STATE,),
        requested_shots=101,
    )
    excessive_shots = legalize_quantum_module(
        imported.module, _target(), requested_shots=1001
    )
    excessive_program = legalize_quantum_module(
        imported.module,
        replace(_target(), maximum_program_operations=3),
    )
    unknown_shot_limit = legalize_quantum_module(
        imported.module,
        replace(_target(), maximum_shots=None),
        requested_shots=1,
    )
    unknown_program_limit = legalize_quantum_module(
        imported.module,
        replace(_target(), maximum_program_operations=None),
    )

    assert accepted.ok and accepted.target_ir is not None
    assert changed.ok and changed.target_ir is not None
    assert accepted.target_ir.target_program_identity != (
        changed.target_ir.target_program_identity
    )
    assert all(
        not result.ok
        for result in (
            excessive_shots,
            excessive_program,
            unknown_shot_limit,
            unknown_program_limit,
        )
    )


def test_static_parameter_domain_is_checked_without_normalization() -> None:
    constrained = (
        GateCapability("h"),
        GateCapability("cx"),
        GateCapability("rz"),
        GateCapability(
            "rx", (ParameterConstraint("theta", minimum=-1.0, maximum=1.0),)
        ),
    )
    accepted = legalize_quantum_module(
        _import(_native_source(0.5)).module,
        _target(gates=constrained),
    )
    rejected = legalize_quantum_module(
        _import(_native_source(1.5)).module,
        _target(gates=constrained),
    )

    assert accepted.ok
    assert not rejected.ok
    assert "outside target domain" in rejected.diagnostics[0].message


def test_dynamic_parameter_binding_is_preserved_or_rejected_explicitly() -> None:
    theta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    imported = _import(_native_source(theta))
    accepted = legalize_quantum_module(imported.module, _target())

    assert accepted.ok and accepted.target_ir is not None
    reference = accepted.target_ir.operations[2].attributes["theta"]
    assert isinstance(reference, RuntimeBindingRef)
    assert imported.bindings[reference.slot] is theta
    unsupported = legalize_quantum_module(
        imported.module, _target(parameter_binding=False)
    )
    assert not unsupported.ok
    constrained = replace(
        _target(),
        native_gates=tuple(
            (
                GateCapability(
                    "rx", (ParameterConstraint("theta", minimum=-1.0, maximum=1.0),)
                )
                if item.operation == "rx"
                else item
            )
            for item in _target().native_gates
        ),
    )
    unprovable = legalize_quantum_module(imported.module, constrained)
    assert not unprovable.ok
    assert "cannot be proven" in unprovable.diagnostics[0].message


def test_static_state_and_trainable_gradient_are_preserved() -> None:
    source = _native_source()
    imported = _import(source)
    target_ir = legalize_quantum_module(imported.module, _target()).target_ir
    assert target_ir is not None
    restored = fq.CircuitIR(
        3,
        tuple(
            fq.Instruction(item.operation, item.physical_qubits, dict(item.attributes))
            for item in target_ir.operations
        ),
        dtype=source.dtype,
    )
    torch.testing.assert_close(
        fq.run(restored).statevector(), fq.run(source).statevector(), atol=1e-12, rtol=0
    )

    theta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    trainable = _native_source(theta)
    imported = _import(trainable)
    target_ir = legalize_quantum_module(imported.module, _target()).target_ir
    assert target_ir is not None
    restored_instructions = []
    for item in target_ir.operations:
        params = {}
        for name, value in item.attributes.items():
            params[name] = (
                imported.bindings[value.slot]
                if isinstance(value, RuntimeBindingRef)
                else value
            )
        restored_instructions.append(
            fq.Instruction(item.operation, item.physical_qubits, params)
        )
    restored = fq.CircuitIR(3, tuple(restored_instructions), dtype="complex128")
    source_loss = fq.Circuit.from_ir(trainable).expectation_z(1).sum()
    target_loss = fq.Circuit.from_ir(restored).expectation_z(1).sum()
    source_gradient = torch.autograd.grad(source_loss, theta, retain_graph=True)[0]
    target_gradient = torch.autograd.grad(target_loss, theta)[0]
    torch.testing.assert_close(target_loss, source_loss, atol=1e-12, rtol=0)
    torch.testing.assert_close(target_gradient, source_gradient, atol=1e-12, rtol=0)


def test_target_identity_changes_with_target_layout_source_and_order() -> None:
    first_import = _import(_native_source())
    first = legalize_quantum_module(first_import.module, _target()).target_ir
    assert first is not None
    target_changed = legalize_quantum_module(
        first_import.module,
        replace(_target(), supports_noise=True),
    ).target_ir
    layout_changed = legalize_quantum_module(
        first_import.module,
        _target(edges=((2, 0), (0, 2), (0, 1), (1, 0))),
        logical_to_physical=(2, 0, 1),
    ).target_ir
    source_changed = legalize_quantum_module(
        _import(
            fq.CircuitIR(
                3,
                tuple(reversed(_native_source().instructions)),
                dtype="complex128",
            )
        ).module,
        _target(),
    ).target_ir

    assert target_changed is not None and layout_changed is not None
    assert source_changed is not None
    identities = {
        item.target_program_identity
        for item in (first, target_changed, layout_changed, source_changed)
    }
    assert len(identities) == 4


def test_target_identity_ignores_python_hash_seed() -> None:
    script = """
from flagquantum import CircuitIR, Instruction
from flagquantum._compiler.importers.circuit_ir import import_circuit_ir
from flagquantum._compiler.target_capabilities import *
from flagquantum._compiler.target_legalization import legalize_quantum_module
source = CircuitIR(1, (Instruction('rx', (0,), {'theta': 0.25}),))
module = import_circuit_ir(source).imported.module
target = TargetCapabilities(
    TargetClass.LOCAL_RUNTIME, 1, 1, (GateCapability('rx'),),
    (MeasurementResult.STATE,), (ArtifactProfile(ArtifactFormat.RUNTIME_PLAN, '1.0'),),
    maximum_program_operations=10,
)
print(legalize_quantum_module(module, target).target_ir.target_program_identity)
"""
    values = []
    for seed in (1, 8675309):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = str(seed)
        values.append(
            subprocess.run(
                [sys.executable, "-c", script],
                cwd=ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
        )
    assert values[0] == values[1]


def test_target_ir_remains_private() -> None:
    for name in (
        "TargetIR",
        "TargetOperation",
        "TargetLegalizationResult",
        "legalize_quantum_module",
    ):
        assert not hasattr(fq, name)


def test_target_ir_schema_excludes_provider_and_execution_identity() -> None:
    forbidden = {
        "provider",
        "backend_id",
        "credential",
        "token",
        "url",
        "job_id",
        "queue",
        "price",
    }
    schema_fields = {item.name for item in fields(TargetIR)} | {
        item.name for item in fields(TargetOperation)
    }
    assert schema_fields.isdisjoint(forbidden)
    with pytest.raises(ValueError, match="exactly match"):
        TargetOperation("h", (0,), {"credential": "secret"})

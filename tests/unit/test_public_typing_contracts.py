"""Behavioral and static conformance for the approved public typing changes."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from datetime import datetime, timedelta, timezone

import pytest
import torch

from flagquantum.circuit import Circuit
from flagquantum.compute.cpu_target_capabilities import (
    CPUCapabilityObservation,
    CPUCapabilityProbe,
    _ObservedCPUProbe,
    cpu_platform_to_target_capability_snapshot,
)
from flagquantum.core.target_capabilities import (
    CapabilityScope,
    EvidenceLevel,
    EvidenceReference,
)
from flagquantum.noise import NoiseModel, bit_flip_channel
from flagquantum.runtime.noise_registry import noisy_density_matrix
from flagquantum.runtime.planner import analyze, plan_runtime_selection
from flagquantum.runtime.routing import StrictExecutionScope

pytestmark = pytest.mark.unit


@dataclass
class _MutableProbe:
    observation: CPUCapabilityObservation
    target_id: str = "cpu-test"
    provider_version: str = "test"
    target_revision: str = "1"
    environment_id: str = "test"
    source_ref: str = "observation"
    target_class_source_ref: str = "declaration"

    def observe(self) -> CPUCapabilityObservation:
        return self.observation


def test_mutable_and_frozen_probes_produce_identical_snapshots() -> None:
    mutable = _MutableProbe(
        CPUCapabilityObservation(available=True, device_count=1, device_ids=("cpu:0",))
    )
    frozen = _ObservedCPUProbe(
        observation=mutable.observation,
        target_id=mutable.target_id,
        provider_version=mutable.provider_version,
        target_revision=mutable.target_revision,
        environment_id=mutable.environment_id,
        source_ref=mutable.source_ref,
        target_class_source_ref=mutable.target_class_source_ref,
    )
    # These assignments also check structural compatibility under strict mypy.
    probes: tuple[CPUCapabilityProbe, CPUCapabilityProbe] = (mutable, frozen)
    evidence = tuple(
        EvidenceReference(
            evidence_id=name,
            sha256=digest * 64,
            level=level,
            scope=CapabilityScope(device_ids=("cpu:0",)),
        )
        for name, digest, level in (
            ("observation", "a", EvidenceLevel.OBSERVABLE),
            ("declaration", "b", EvidenceLevel.BASIC),
        )
    )
    snapshots = tuple(
        cpu_platform_to_target_capability_snapshot(
            probe=probe,
            captured_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
            ttl=timedelta(minutes=30),
            evidence_refs=evidence,
        )
        for probe in probes
    )
    assert snapshots[0] == snapshots[1]
    with pytest.raises(FrozenInstanceError):
        setattr(frozen, "target_revision", "2")
    mutable.target_revision = "2"
    assert probes[0].target_revision == "2"


def test_execution_scope_propagates_the_original_exception() -> None:
    error = RuntimeError("sentinel execution failure")
    with pytest.raises(RuntimeError) as caught:
        with StrictExecutionScope():
            raise error
    assert caught.value is error


def test_circuit_inspection_matches_runtime_delegates() -> None:
    circuit = Circuit(2).gate("h", 0).gate("cx", (0, 1))
    assert circuit.analysis() == analyze(circuit.to_ir())
    assert circuit.runtime_plan() == plan_runtime_selection(
        circuit.to_ir(), bsz=circuit.bsz
    )


@pytest.mark.parametrize("with_noise", [False, True])
def test_circuit_density_matches_runtime_delegate(with_noise: bool) -> None:
    circuit = Circuit(1).gate("x", 0)
    model = NoiseModel().add("x", bit_flip_channel(1.0)) if with_noise else None
    actual = circuit.noisy_density_matrix(model)
    torch.testing.assert_close(actual, noisy_density_matrix(circuit, model))
    expected = torch.zeros_like(actual)
    basis = 0 if with_noise else 1
    expected[..., basis, basis] = 1
    torch.testing.assert_close(actual, expected)

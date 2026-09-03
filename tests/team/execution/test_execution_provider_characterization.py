"""Characterize the pre-contract Execution Provider boundary.

These tests intentionally describe the current behavior.  They are migration
evidence for the Core-owned Execution Provider contract, not an endorsement of
the gaps that they expose.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import pytest

import flagquantum as fq
import flagquantum.backends as fqb
import flagquantum.deployment as fqd
from flagquantum.extensions import ProviderExtension
from flagquantum.runtime.result import ExecutionResult
from flagquantum.runtime.target_execution import TargetExecutionResult

pytestmark = pytest.mark.unit


def _local_package(*, provider: str = "local", shots: int = 8):
    backend = fq.CloudBackendProfile(
        provider=provider,
        name="characterization-target",
        n_wires=2,
        is_simulator=provider == "local",
    )
    return fqd.create_deployment_package(
        fq.Circuit(2).x(0), backend=backend, shots=shots
    )


def test_current_local_target_and_deployment_results_are_distinct() -> None:
    circuit = fq.Circuit(2).x(0)

    runtime_result = fq.run(circuit)
    target_result = fqb.run_target(
        circuit,
        target="few_amplitudes",
        bitstrings=("00", "10"),
        mode="tensor_network",
    )
    deployment_result = fqd.LocalSimulatorProvider().run(_local_package())

    assert isinstance(runtime_result, ExecutionResult)
    assert isinstance(target_result, TargetExecutionResult)
    assert isinstance(deployment_result, fqd.DeploymentResult)
    assert not isinstance(target_result, ExecutionResult)
    assert not isinstance(deployment_result, ExecutionResult)
    assert target_result.summary()["full_state_materialized"] is False
    assert deployment_result.counts == {"10": 8}


def test_local_simulator_exercises_submission_status_result_identity_chain() -> None:
    provider = fqd.LocalSimulatorProvider()
    package = _local_package(shots=16)

    handle = provider.submit(package)
    result = provider.fetch_result(handle)

    assert provider.query_status(handle) == "Finished"
    assert result.shots == sum(result.counts.values()) == 16
    assert result.metadata["simulated"] is True
    assert (
        result.metadata["deployment_artifact_sha256"]
        == handle.payload["deployment_artifact_sha256"]
        == package.metadata["deployment_artifact_sha256"]
    )
    assert fqd.validate_deployment_result(result) is result


def test_deployment_result_validation_only_seals_identity_and_shot_accounting() -> None:
    result = fqd.LocalSimulatorProvider().run(_local_package())
    proposed_provider_evidence = {
        "duration_ms",
        "calibration_version",
        "noise_model_version",
        "counts_bit_order",
        "physical_qubits",
    }

    assert proposed_provider_evidence.isdisjoint(result.metadata)
    with_extra_native_metadata = replace(
        result,
        metadata={**result.metadata, "provider_native_state": "COMPLETED"},
    )
    assert (
        fqd.validate_deployment_result(with_extra_native_metadata)
        is with_extra_native_metadata
    )
    with pytest.raises(fqd.DeploymentPackageIdentityError, match="shots do not match"):
        fqd.validate_deployment_result(replace(result, shots=result.shots + 1))


def test_base_provider_run_checks_status_once_instead_of_polling() -> None:
    class PendingProvider(fqd.QuantumProvider):
        provider = "pending-characterization"

        def __init__(self) -> None:
            self.status_calls = 0
            self.fetch_calls = 0

        def submit(self, package):
            return fqd.ProviderTaskHandle(
                self.provider,
                "pending-1",
                package.backend.name,
                fqd.build_submission_receipt(package),
            )

        def query_status(self, handle):
            self.status_calls += 1
            return "Running"

        def fetch_result(self, handle):
            self.fetch_calls += 1
            raise AssertionError("a non-terminal task must not be fetched")

    provider = PendingProvider()

    with pytest.raises(RuntimeError, match="ended with status 'Running'"):
        provider.run(_local_package(provider=provider.provider))

    assert provider.status_calls == 1
    assert provider.fetch_calls == 0


def test_cancel_and_result_are_not_shared_by_both_provider_surfaces() -> None:
    assert "cancel" not in fqd.QuantumProvider.__dict__
    assert "cancel" in fqd.QuafuProvider.__dict__
    assert "submit" in ProviderExtension.__dict__
    assert "status" in ProviderExtension.__dict__
    assert "result" not in ProviderExtension.__dict__
    assert "cancel" not in ProviderExtension.__dict__


def test_generic_http_decoder_accepts_untyped_count_width_and_rounds_values() -> None:
    class Transport:
        def post_json(
            self,
            url: str,
            payload: Mapping[str, Any],
            headers: Mapping[str, str],
            timeout: float,
        ) -> Mapping[str, Any]:
            return {"task_id": "remote-1"}

        def get_json(
            self,
            url: str,
            headers: Mapping[str, str],
            timeout: float,
        ) -> Mapping[str, Any]:
            if url.endswith("/status"):
                return {"status": "Finished"}
            return {"counts": {"0": 1.6, "10": 2.4}}

        def post_form(
            self,
            url: str,
            payload: Mapping[str, Any],
            headers: Mapping[str, str],
            timeout: float,
        ) -> Mapping[str, Any]:
            raise AssertionError("form transport is unused")

    provider = fqd.HttpQuantumProvider(
        provider="remote-characterization",
        base_url="https://provider.invalid",
        transport=Transport(),
    )

    result = provider.run(_local_package(provider=provider.provider, shots=4))

    assert result.counts == {"0": 2, "10": 2}
    assert result.shots == 4
    assert "counts_bit_order" not in result.metadata
    assert "counts_width" not in result.metadata

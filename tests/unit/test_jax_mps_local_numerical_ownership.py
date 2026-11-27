"""Defining-owner replacements; collective fakes are not hardware evidence."""

from types import SimpleNamespace

import numpy as np
import pytest

from flagquantum.runtime.executors.jax.mps import backward, canonicalization
from flagquantum.runtime.executors.jax.mps import pullbacks as runtime_pullbacks
from flagquantum.simulation.mps.jax import pullbacks as numerics

pytestmark = pytest.mark.unit


def test_cpu_backward_uses_simulation_site_response_without_jax(monkeypatch):
    calls = []

    def replacement(parameters):
        calls.append(parameters.copy())
        return np.ones((2, 2)), np.asarray([2.0, 3.0])

    def refuse_jax():
        raise AssertionError("CPU development execution must not admit JAX")

    monkeypatch.setattr(numerics, "_numpy_mps_ry_site_response", replacement)
    monkeypatch.setattr(backward, "_require_jax", refuse_jax)
    summary = backward._execute_minimal_mps_sharded_backward(
        [0.2, 0.4], execution_backend="cpu"
    )
    np.testing.assert_array_equal(calls, [[0.2, 0.4]])
    assert summary["value"] == 6.0
    assert summary["claim_evidence_type"] == "development_smoke"


def test_cpu_backward_uses_simulation_weighted_gradient(monkeypatch):
    received = []

    def replacement(parameters, remote):
        received.append((parameters.copy(), remote.copy()))
        return np.asarray([11.0, 13.0])

    monkeypatch.setattr(numerics, "_numpy_mps_ry_weighted_gradient", replacement)
    summary = backward._execute_minimal_mps_sharded_backward(
        [0.2, 0.4], execution_backend="cpu"
    )
    assert summary["gradient"] == (11.0, 13.0)
    np.testing.assert_array_equal(received[0][0], [0.2, 0.4])
    np.testing.assert_allclose(received[0][1], np.cos([0.4, 0.2]))


def test_owner_vjp_uses_simulation_scores_before_cotangent_exchange(monkeypatch):
    pytest.importorskip("jax")
    received = []
    original = runtime_pullbacks.jax_mps_owner_local_vjp

    def record(parameter, cotangent, *, gate_kind):
        received.append(float(cotangent))
        return original(parameter, cotangent, gate_kind=gate_kind)

    monkeypatch.setattr(
        numerics,
        "_numpy_mps_owner_local_scores",
        lambda parameters: np.asarray([5.0, 7.0]),
    )
    monkeypatch.setattr(runtime_pullbacks, "jax_mps_owner_local_vjp", record)
    summary = runtime_pullbacks._execute_minimal_mps_owner_rank_parameter_vjp(
        [0.2, 0.4]
    )
    assert received == [7.0, 5.0]
    assert summary["value"] == 35.0
    assert summary["scalability_claim_allowed"] is False


class _ReadyArray(np.ndarray):
    def block_until_ready(self):
        return self


def _collective_contract_fake():
    """Exercise Runtime collective wiring; this is not an accelerator run."""
    events = []

    def ppermute(value, axis_name, permutation):
        events.append((float(value), axis_name, permutation))
        return 7.0

    def pmap(function, *, axis_name, devices):
        assert axis_name == "mps_rank"
        assert len(devices) == 2

        def mapped(parameters):
            rows = [function(theta) for theta in parameters]
            return tuple(
                np.asarray(values).view(_ReadyArray)
                for values in zip(*rows, strict=True)
            )

        return mapped

    runtime = SimpleNamespace(
        config=SimpleNamespace(update=lambda *args: None),
        local_devices=lambda: (
            SimpleNamespace(platform="gpu", device_kind="contract_fake"),
        )
        * 2,
        process_count=lambda: 1,
        pmap=pmap,
        lax=SimpleNamespace(ppermute=ppermute),
        tree_util=SimpleNamespace(tree_leaves=lambda leaves: leaves),
        device_get=lambda value: value,
    )
    return runtime, events


def test_backward_collective_consumes_replaced_local_site_response(monkeypatch):
    runtime, events = _collective_contract_fake()
    monkeypatch.setattr(backward, "_require_jax", lambda: (runtime, np))
    calls = []

    def replacement(theta):
        calls.append(float(theta))
        return np.asarray([3.0, 4.0]), 5.0, 13.0

    monkeypatch.setattr(numerics, "_jax_mps_ry_site_response", replacement)
    summary = backward._execute_minimal_mps_sharded_backward(
        [0.2, 0.4], execution_backend="accelerator"
    )
    assert calls == [0.2, 0.4, 0.2, 0.4]
    assert summary["gradient"] == (91.0, 91.0)
    assert events == [(5.0, "mps_rank", ((0, 1), (1, 0)))] * 4


def test_optimizer_collective_consumes_replaced_z_response(monkeypatch):
    runtime, events = _collective_contract_fake()
    summary = backward._execute_minimal_mps_sharded_backward(
        [0.2, 0.4], execution_backend="cpu"
    )
    summary["claim_evidence_type"] = "production_runtime"
    monkeypatch.setattr(
        canonicalization,
        "_execute_minimal_mps_sharded_backward",
        lambda *a, **k: summary,
    )
    monkeypatch.setattr(canonicalization, "_require_jax", lambda: (runtime, np))
    calls = []

    def replacement(theta):
        calls.append(float(theta))
        return 5.0, 13.0

    monkeypatch.setattr(numerics, "_jax_mps_ry_z_response", replacement)
    result = canonicalization._execute_minimal_mps_sharded_optimizer_step(
        [0.2, 0.4], learning_rate=0.05, execution_backend="accelerator"
    )
    assert calls == [0.2, 0.4]
    assert events == [(5.0, "mps_rank", ((0, 1), (1, 0)))] * 2
    assert tuple(
        record["gradient_value"] for record in result["gradient_ownership"]
    ) == (
        91.0,
        91.0,
    )
    np.testing.assert_array_equal(
        result["updated_parameters"], np.asarray([0.2, 0.4]) - 0.05 * 91.0
    )

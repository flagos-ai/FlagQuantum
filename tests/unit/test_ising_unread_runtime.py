"""Native unread observation through existing planning, replay and refusals."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ising import IsingBraidProgram
from flagquantum.errors import CapabilityError
from flagquantum.runtime.execution_plan_contract import (
    ExecutionPlanContractError,
    canonical_hash,
)

pytestmark = pytest.mark.unit


def _source() -> IsingBraidProgram:
    return IsingBraidProgram().exchange(1).dephase_pair().exchange(1, clockwise=True)


def test_unread_native_plan_replays_a_mixed_density_without_circuit_analysis() -> None:
    source = _source()
    selected = fq.plan(
        source, outputs=fq.expectation(source.projector("psi"), name="psi")
    )
    restored = fq.ExecutionPlan.from_json(selected.to_json())
    assert selected.schema_version == "1.2"
    assert selected.to_dict()["program"]["version"] == "1.2"
    assert selected.analysis is None
    summary = selected.summary()
    assert summary["depth"] == summary["n_instructions"] == 3
    assert summary["n_wires"] is None
    assert selected.state_bytes == 64
    assert restored.to_json() == selected.to_json()
    results = (fq.run(selected), fq.run(restored))
    for result in results:
        assert result.state is not None and result.samples is None
        assert result.state.shape == (1, 2, 2)
        assert result.state.dtype == torch.complex128
        assert result.state.device.type == "cpu"
        torch.testing.assert_close(
            result.state[0],
            torch.eye(2, dtype=torch.complex128) / 2,
            atol=1e-12,
            rtol=0,
        )
        torch.testing.assert_close(
            result.expectation("psi"),
            torch.tensor([0.5], dtype=torch.float64),
            atol=1e-12,
            rtol=0,
        )
    assert torch.equal(results[0].state, results[1].state)
    assert fq.plan(IsingBraidProgram()).schema_version == "1.1"
    assert fq.plan(fq.Circuit(1)).schema_version == "1.0"


@pytest.mark.parametrize(
    "plan_version,source_version", [("1.1", "1.2"), ("1.2", "1.1"), ("1.2", "1.2")]
)
def test_rehashed_mismatched_or_relabelled_sources_are_refused(
    plan_version: str, source_version: str
) -> None:
    # The equal-version case deliberately labels a braid-only source as 1.2.
    source = _source() if plan_version == "1.1" else IsingBraidProgram().exchange(1)
    payload = fq.plan(source).to_dict()
    payload["version"] = plan_version
    payload["program"]["version"] = source_version
    payload["fingerprints"]["program"] = canonical_hash(payload["program"])
    payload["identity"] = canonical_hash(
        {k: v for k, v in payload.items() if k != "identity"}
    )
    with pytest.raises(ExecutionPlanContractError) as error:
        fq.ExecutionPlan.from_dict(payload)
    assert error.value.reason_code == "program_fingerprint_mismatch"


@pytest.mark.parametrize(
    "field,value",
    [
        ("shots", 10),
        ("require_gradients", True),
        ("backend", "jax"),
        ("device", "cuda"),
        ("batch_size", 2),
        ("memory_limit_bytes", 63),
        ("allow_approximate", True),
        ("allow_backend_fallback", True),
    ],
)
def test_unread_keeps_existing_cpu_options_and_allocation_refusals(
    field: str, value: object
) -> None:
    constructor: Callable[..., fq.ExecutionOptions] = fq.ExecutionOptions
    with (
        patch("flagquantum.simulation.ising_braid.evolve_ising_braid") as kernel,
        pytest.raises(CapabilityError),
    ):
        fq.run(_source(), options=constructor(**{field: value}))
    kernel.assert_not_called()


def test_unread_planning_and_saved_execution_refuse_a_distributed_environment() -> None:
    selected = fq.plan(_source())
    with (
        patch(
            "flagquantum.runtime.distributed.backend_policy.resolve_distributed_backend_policy",
            return_value=SimpleNamespace(effective_world_size=2),
        ),
        patch("flagquantum.simulation.ising_braid.evolve_ising_braid") as kernel,
    ):
        with pytest.raises(CapabilityError):
            fq.plan(_source())
        with pytest.raises(ExecutionPlanContractError, match="world_size_mismatch"):
            fq.run(selected)
    kernel.assert_not_called()

"""Real sampled plans, environment binding and failure-before-draw boundaries."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest
import torch

import flagquantum as fq
from flagquantum.core.ising import IsingBraidProgram
from flagquantum.errors import CapabilityError, ExecutionError
from flagquantum.runtime.execution_plan_contract import (
    ExecutionPlanContractError,
    canonical_hash,
)

pytestmark = pytest.mark.unit


def _source() -> IsingBraidProgram:
    return IsingBraidProgram().exchange(1).measure_pair().exchange(0)


def _plan() -> fq.ExecutionPlan:
    source = _source()
    return fq.plan(
        source,
        options=fq.ExecutionOptions(shots=1, seed=17),
        outputs=fq.expectation(source.projector("psi"), name="psi"),
    )


def _rehash(payload: dict[str, Any]) -> None:
    for name, field in (
        ("program", "program"),
        ("options", "resolved_options"),
        ("environment", "environment_requirements"),
    ):
        payload["fingerprints"][name] = canonical_hash(payload[field])
    payload["identity"] = canonical_hash(
        {k: v for k, v in payload.items() if k != "identity"}
    )


def test_real_plan_and_saved_replay_expose_conditional_results() -> None:
    plan = _plan()
    payload = plan.to_dict()
    assert plan.schema_version == "1.3" and plan.analysis is None
    assert plan.state_bytes == 64
    assert payload["program"]["version"] == "1.3"
    assert payload["environment_requirements"]["version"] == "1.1"
    assert payload["environment_requirements"]["rng_torch_version"] == str(
        torch.__version__
    )
    first = fq.run(plan)
    repeat = fq.run(fq.ExecutionPlan.from_json(plan.to_json()))
    event = first.measurement(-1)
    assert event.value.shape == (1, 1) and event.value.dtype == torch.int64
    index = int(event.value[0, 0])
    assert event.metadata["pair_anyon_ids"] == ["a", "c"]
    assert event.metadata["source_operation_index"] == 1
    assert abs(float(first.expectation("psi")[0]) - index) < 1e-12
    assert torch.equal(first.state, repeat.state)
    assert torch.equal(event.value, repeat.measurement(-1).value)
    assert first.samples is None and first.plan is plan
    assert (
        fq.plan(IsingBraidProgram()).to_dict()["environment_requirements"]["version"]
        == "1.0"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"version": "1.0"},
        {"rng_torch_version": "other-build"},
        {"rng_torch_version": None},
        {"extra": True},
    ],
)
def test_rehashed_environment_mismatch_is_refused(change: dict[str, object]) -> None:
    payload = _plan().to_dict()
    payload["environment_requirements"].update(change)
    _rehash(payload)
    with pytest.raises(ExecutionPlanContractError) as error:
        fq.ExecutionPlan.from_dict(payload)
    assert error.value.reason_code == "environment_incompatible"


def test_changed_torch_build_refuses_reader_and_preflight_before_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    text = plan.to_json()
    monkeypatch.setattr(torch, "__version__", "unmatched-build")
    with (
        patch("flagquantum.runtime.measurements._draw_probability_indices") as draw,
        patch("flagquantum.simulation.ising_braid._ising_sampled_prefix") as prefix,
    ):
        calls: tuple[Callable[[], object], ...] = (
            lambda: fq.ExecutionPlan.from_json(text),
            lambda: fq.run(plan),
        )
        for call in calls:
            with pytest.raises(ExecutionPlanContractError) as error:
                call()
            assert error.value.reason_code == "environment_incompatible"
        draw.assert_not_called()
        prefix.assert_not_called()
    # Old profiles keep their old environment protocol and are not RNG-bound.
    assert fq.run(IsingBraidProgram()).state is not None


@pytest.mark.parametrize("version", ["1.1", "1.2"])
def test_old_profiles_reject_rng_requirement_even_after_rehash(version: str) -> None:
    source = IsingBraidProgram()
    if version == "1.2":
        source.dephase_pair()
    payload = fq.plan(source).to_dict()
    payload["environment_requirements"].update(
        version="1.1", rng_torch_version=str(torch.__version__)
    )
    _rehash(payload)
    with pytest.raises(ExecutionPlanContractError) as error:
        fq.ExecutionPlan.from_dict(payload)
    assert error.value.reason_code == "environment_incompatible"


@pytest.mark.parametrize(
    "plan_version,source_version",
    [("1.1", "1.3"), ("1.2", "1.3"), ("1.3", "1.1"), ("1.3", "1.2")],
)
def test_source_and_plan_versions_cannot_be_relabelled(
    plan_version: str, source_version: str
) -> None:
    payload = _plan().to_dict()
    payload["version"] = plan_version
    payload["program"]["version"] = source_version
    _rehash(payload)
    with pytest.raises(ExecutionPlanContractError) as error:
        fq.ExecutionPlan.from_dict(payload)
    assert error.value.reason_code == "program_fingerprint_mismatch"


@pytest.mark.parametrize(
    "field,value", [("shots", None), ("shots", 2), ("seed", None), ("seed", 2**64)]
)
@pytest.mark.parametrize("requested_only", [False, True])
def test_rehashed_sampling_controls_remain_explicit(
    field: str, value: object, requested_only: bool
) -> None:
    payload = _plan().to_dict()
    payload["requested_options"][field] = value
    if not requested_only:
        payload["resolved_options"][field] = value
    _rehash(payload)
    with pytest.raises(ExecutionPlanContractError) as error:
        fq.ExecutionPlan.from_dict(payload)
    assert error.value.reason_code == "options_fingerprint_mismatch"


@pytest.mark.parametrize("seed", [0, 17, 2**64 - 1])
def test_explicit_seed_limits_and_resource_preflight(seed: int) -> None:
    source = _source()
    with patch("flagquantum.simulation.ising_braid._ising_sampled_prefix") as prefix:
        with pytest.raises(CapabilityError):
            fq.plan(
                source,
                options=fq.ExecutionOptions(shots=1, seed=seed, memory_limit_bytes=215),
            )
        prefix.assert_not_called()
    plan = fq.plan(
        source, options=fq.ExecutionOptions(shots=1, seed=seed, memory_limit_bytes=216)
    )
    assert fq.run(plan).state is not None
    with pytest.raises(CapabilityError):
        fq.plan(source, options=fq.ExecutionOptions(shots=1, seed=2**64))


@pytest.mark.parametrize("selected", [-1, 2, 1])
def test_invalid_or_zero_weight_draw_never_returns_partial_result(
    selected: int,
) -> None:
    plan = fq.plan(
        IsingBraidProgram().measure_pair(),
        options=fq.ExecutionOptions(shots=1, seed=17),
    )
    with (
        patch(
            "flagquantum.runtime.measurements._draw_probability_indices",
            return_value=torch.tensor([[selected]], dtype=torch.int64),
        ),
        pytest.raises(ExecutionError),
    ):
        fq.run(plan)


def test_plan_override_remains_forbidden() -> None:
    plan = _plan()
    with pytest.raises(TypeError):
        fq.run(plan, options=fq.ExecutionOptions(shots=1, seed=18))


@pytest.mark.parametrize(
    "draw",
    [torch.tensor([[0.5]]), torch.tensor([0]), torch.tensor([[0]], dtype=torch.int32)],
)
def test_malformed_draw_is_not_coerced(draw: torch.Tensor) -> None:
    plan = _plan()
    with (
        patch(
            "flagquantum.runtime.measurements._draw_probability_indices",
            return_value=draw,
        ),
        pytest.raises(ExecutionError),
    ):
        fq.run(plan)


@pytest.mark.parametrize(
    "options", [None, fq.ExecutionOptions(shots=1), fq.ExecutionOptions(seed=17)]
)
def test_sampling_controls_must_be_explicit_not_inherited(
    options: fq.ExecutionOptions | None,
) -> None:
    from flagquantum.runtime.options_resolver import resolve_execution_options

    inherited = resolve_execution_options(
        fq.ExecutionOptions(shots=1, seed=17, precision="complex128")
    )
    with (
        patch(
            "flagquantum.runtime._ising_plan.resolve_execution_options",
            return_value=inherited,
        ),
        pytest.raises(CapabilityError),
    ):
        fq.plan(_source(), options=options)


def test_shared_dense_ceiling_is_checked_before_prefix() -> None:
    with (
        patch("flagquantum.runtime._ising_plan.DEFAULT_DENSE_MATRIX_BYTES", 215),
        patch("flagquantum.simulation.ising_braid._ising_sampled_prefix") as prefix,
    ):
        with pytest.raises(CapabilityError):
            _plan()
        prefix.assert_not_called()
        assert fq.plan(IsingBraidProgram()).state_bytes == 64


def test_sampled_profiles_keep_noise_cost_and_distributed_exits_closed() -> None:
    from types import SimpleNamespace

    plan = _plan()
    with pytest.raises(CapabilityError, match="cost model"):
        plan.calibrated_cost({})
    with pytest.raises(CapabilityError, match="noise"):
        fq.plan(
            _source(),
            options=fq.ExecutionOptions(shots=1, seed=17),
            noise_model=object(),
        )
    with (
        patch(
            "flagquantum.runtime.distributed.backend_policy.resolve_distributed_backend_policy",
            return_value=SimpleNamespace(effective_world_size=2),
        ),
        patch("flagquantum.simulation.ising_braid._ising_sampled_prefix") as prefix,
    ):
        with pytest.raises(CapabilityError):
            _plan()
        with pytest.raises(ExecutionPlanContractError) as error:
            fq.run(plan)
        assert error.value.reason_code == "world_size_mismatch"
        prefix.assert_not_called()

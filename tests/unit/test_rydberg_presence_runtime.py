"""Real sampled artifacts, admission, and preflight boundaries for proposal 092."""

from __future__ import annotations

from typing import Any

import pytest
import torch

import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum.core.rydberg import RydbergProgram
from flagquantum.errors import CapabilityError, ExecutionError, SerializationError
from flagquantum.lindblad._plan import _identity
from flagquantum.runtime.execution_plan_contract import ExecutionPlanContractError
from tests.unit.test_rydberg_presence_contract import _program

pytestmark = pytest.mark.unit


def _plan(**kwargs: Any) -> fql.LindbladPlan:
    source = _program(RydbergProgram, False, parameterized=True)
    source.measure_presence("a", at=0.5, efficiency=0.8)
    settings = dict(
        options=fq.ExecutionOptions(shots=1, seed=17),
        parameter_values=[{"omega": 1.2}] * 32 + [{"omega": 0.4}] * 32,
        return_density_matrices=True,
    )
    settings.update(kwargs)
    return fql.plan(
        source,
        source.basis_state(("g",)),
        torch.linspace(0, 1, 65, dtype=torch.float64),
        **settings,
    )


def test_real_artifact_roundtrip_has_exact_fields_and_canonical_results() -> None:
    plan = _plan()
    payload = plan.to_dict()
    assert set(payload) == {
        "schema",
        "version",
        "request",
        "decision",
        "program",
        "bindings",
        "identity",
        "options",
        "environment_requirements",
    }
    assert (payload["version"], payload["program"]["version"]) == ("1.8", "1.4")
    assert payload["environment_requirements"]["rng_torch_version"] == str(
        torch.__version__
    )
    assert plan.to_json() == fql.LindbladPlan.from_json(plan.to_json()).to_json()
    result = fq.run(plan)
    assert result.plan is plan and result.value is None and result.samples is None
    assert result.state is not None and result.state.shape == (1, 3, 3)
    assert result.measurement("density_matrix").value.shape == (65, 3, 3)
    assert result.to("cpu").plan is plan and result.summary()["has_plan"]
    assert torch.equal(result.state, fq.run(fql.LindbladPlan.from_dict(payload)).state)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", "1.7"),
        ("extra", None),
        ("options", None),
        ("bindings", [{"omega": 5.0}] * 64),
    ],
)
def test_rehashed_outer_tampering_is_refused(field: str, value: Any) -> None:
    payload = _plan().to_dict()
    payload[field] = value
    payload["identity"] = _identity(payload)
    with pytest.raises(SerializationError):
        fql.LindbladPlan.from_dict(payload)


@pytest.mark.parametrize(
    "mutation",
    ["source_version", "missing_event", "matrix", "time", "shot", "gradient"],
)
def test_rehashed_semantic_tampering_is_refused(mutation: str) -> None:
    payload = _plan().to_dict()
    if mutation == "source_version":
        payload["program"]["version"] = "1.0"
    elif mutation == "missing_event":
        # A valid old source is still not a sampled source.
        source = _program(RydbergProgram, False)
        from flagquantum.core.ir.program.physical_codec import native_program_to_dict

        payload["program"] = native_program_to_dict(source.to_ir())
    elif mutation == "matrix":
        payload["request"]["hamiltonians"][0]["values"][0][0][0] += 0.1
    elif mutation == "time":
        payload["request"]["times"][32] += 0.001
    elif mutation == "shot":
        payload["options"]["shots"] = 2
    else:
        payload["decision"]["numerics"]["require_gradients"] = True
    payload["identity"] = _identity(payload)
    with pytest.raises(SerializationError):
        fql.LindbladPlan.from_dict(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rng_torch_version", "other-build"),
        ("world_size", 2),
        ("version", "1.0"),
        ("extra", None),
    ],
)
def test_environment_refusal_precedes_capture_and_draw(
    field: str, value: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _plan().to_dict()
    payload["environment_requirements"][field] = value
    payload["identity"] = _identity(payload)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("must refuse before numeric capture/draw")

    monkeypatch.setattr("flagquantum.lindblad._native_plan.native_request", forbidden)
    monkeypatch.setattr(
        "flagquantum.runtime.measurements._draw_probability_indices", forbidden
    )
    with pytest.raises(ExecutionPlanContractError) as error:
        fql.LindbladPlan.from_dict(payload)
    assert error.value.reason_code == "environment_incompatible"
    with pytest.raises(ExecutionPlanContractError):
        fq.run(fql.LindbladPlan(__import__("json").dumps(payload)))


@pytest.mark.parametrize(
    "options",
    [
        None,
        fq.ExecutionOptions(seed=17),
        fq.ExecutionOptions(shots=1),
        fq.ExecutionOptions(shots=2, seed=17),
        fq.ExecutionOptions(shots=1, seed=17, precision="complex64"),
        fq.ExecutionOptions(shots=1, seed=17, require_gradients=True),
        fq.ExecutionOptions(shots=1, seed=17, memory_limit_bytes=4096),
    ],
)
def test_unsupported_options_refused_before_capture(
    options: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("must refuse before native capture")

    monkeypatch.setattr("flagquantum.lindblad._native_plan.native_request", forbidden)
    with pytest.raises(CapabilityError):
        _plan(options=options)


@pytest.mark.parametrize(
    "keyword",
    [
        "options",
        "outputs",
        "noise_model",
        "shots",
        "target",
        "compiler",
        "name",
        "target_qubits",
    ],
)
def test_sealed_root_plan_refuses_overrides(keyword: str) -> None:
    with pytest.raises(TypeError):
        fq.run(_plan(), **{keyword: object()})


@pytest.mark.parametrize(
    "selected",
    [
        torch.tensor([[2]]),
        torch.tensor([[True]]),
        torch.tensor([0]),
        torch.tensor([[0.0]]),
        None,
    ],
)
def test_bad_draw_returns_no_partial_result(
    selected: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan()
    monkeypatch.setattr(
        "flagquantum.runtime.measurements._draw_probability_indices",
        lambda *args, **kwargs: selected,
    )
    with pytest.raises(ExecutionError):
        fq.run(plan)


def test_resource_ceiling_precedes_native_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flagquantum.runtime._rydberg_presence.DEFAULT_DENSE_MATRIX_BYTES", 455
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("must refuse before allocation")

    monkeypatch.setattr("flagquantum.lindblad._native_plan.native_request", forbidden)
    with pytest.raises(CapabilityError, match="ceiling"):
        _plan(return_density_matrices=False)


def test_public_unconditional_and_adjoint_paths_do_not_ignore_presence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _program(RydbergProgram, False).measure_presence(
        "a", at=0.5, efficiency=0.8
    )
    grid = torch.linspace(0, 1, 65, dtype=torch.float64)
    initial = source.basis_state(("g",))

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("unconditional execution must remain closed")

    monkeypatch.setattr(fql, "_evolve_density_matrix", forbidden)
    with pytest.raises(CapabilityError, match="fq.run"):
        fql.run(source, initial, grid, options=fq.ExecutionOptions(shots=1, seed=17))
    with pytest.raises(CapabilityError, match="conditional trajectory"):
        fql.adjoint_gradient(source, initial, grid, lambda value: value.real.sum())


def test_reader_budget_refusal_precedes_matrix_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _plan().to_dict()

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("reader must refuse before matrix assembly")

    monkeypatch.setattr(
        "flagquantum.runtime._rydberg_presence.DEFAULT_DENSE_MATRIX_BYTES", 455
    )
    monkeypatch.setattr("flagquantum.lindblad._native_plan.native_request", forbidden)
    with pytest.raises(SerializationError, match="ceiling"):
        fql.LindbladPlan.from_dict(payload)


def test_world_size_refused_before_capture_and_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    plan = _plan()
    monkeypatch.setattr(
        "flagquantum.runtime.distributed.backend_policy.resolve_distributed_backend_policy",
        lambda: SimpleNamespace(effective_world_size=2),
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("world-size refusal precedes prefix/draw")

    monkeypatch.setattr(
        "flagquantum.simulation.rydberg_presence._rydberg_presence_prefix", forbidden
    )
    with pytest.raises(CapabilityError, match="world_size"):
        _plan()
    with pytest.raises(ExecutionPlanContractError) as error:
        fq.run(plan)
    assert error.value.reason_code == "environment_incompatible"


def test_native_source_stays_closed_at_circuit_and_deployment_frontdoors() -> None:
    from flagquantum.deployment import create_deployment_package

    source = _program(RydbergProgram, False).measure_presence(
        "a", at=0.5, efficiency=0.8
    )
    with pytest.raises(CapabilityError):
        fq.compile(source)
    with pytest.raises(CapabilityError):
        create_deployment_package(source)

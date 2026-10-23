"""Native plan admission, immutable identities, replay and closed exits."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import flagquantum as fq
from flagquantum._api import plan, run
from flagquantum.core.ir import IRValidationError
from flagquantum.core.ising import IsingBraidProgram
from flagquantum.core.runtime_config import runtime_config
from flagquantum.errors import CapabilityError, ValidationError
from flagquantum.runtime.execution import run as runtime_run
from flagquantum.runtime.execution_plan import ExecutionPlan
from flagquantum.runtime.execution_plan_contract import (
    EXECUTION_PLAN_VERSION,
    ExecutionPlanContractError,
    canonical_hash,
)
from flagquantum.runtime.options import ExecutionOptions
from flagquantum.runtime.planner import plan as runtime_plan
from flagquantum.runtime.result import ExecutionResult

pytestmark = pytest.mark.unit


def _rehash(payload: dict[str, object]) -> None:
    payload["identity"] = canonical_hash(
        {key: value for key, value in payload.items() if key != "identity"}
    )


def test_existing_plan_and_audit_have_native_meaning_without_qubit_analysis() -> None:
    program = IsingBraidProgram().exchange(1).exchange(0, clockwise=True)
    selected = plan(program)
    assert isinstance(selected, ExecutionPlan)
    assert selected.analysis is None and selected.layers == ()
    assert selected.world_size == 1 and selected.shardable_wires == ()
    assert selected.state_bytes == 64
    assert selected.state_mode == selected.recommended_mode == "density_matrix"
    assert selected.noisy_execution_plan is None
    summary = selected.summary()
    assert summary["depth"] == summary["n_instructions"] == 2
    assert (
        summary["n_wires"]
        is summary["two_qubit_gates"]
        is summary["multi_qubit_gates"]
        is None
    )
    assert summary["distribution_semantics"] == "single_device_fast_path"
    assert (
        not summary["scalability_claim_allowed"] and not summary["release_gate_allowed"]
    )
    audit = selected.to_contract()
    assert audit.plan_id == selected.identity
    assert audit.estimated.depth == 2 and audit.estimated.memory_bytes == 64
    with pytest.raises(CapabilityError, match="cost model"):
        selected.calibrated_cost({})
    assert EXECUTION_PLAN_VERSION == "1.0"
    assert selected.schema_version == "1.1"
    assert plan(fq.Circuit(1)).schema_version == "1.0"


def test_direct_planned_and_json_paths_preserve_ordered_outputs_and_exact_state() -> (
    None
):
    source = IsingBraidProgram(initial_channel="psi").exchange(1).exchange(0)
    requests = [
        fq.expectation(source.projector("psi")),
        fq.expectation(source.projector("1"), name="one"),
    ]
    selected = plan(source, outputs=requests)
    encoded = selected.to_json()
    replay = ExecutionPlan.from_json(encoded)
    results = (run(source, outputs=requests), run(selected), runtime_run(replay))
    reference = results[0]
    assert reference.state is not None and reference.value is not None
    for result in results:
        assert result.state is not None and result.value is not None
        assert torch.equal(result.state, reference.state)
        assert torch.equal(result.value, reference.value)
        assert result.state.shape == (1, 2, 2) and result.value.shape == (1, 2)
        assert result.samples is None and result.plan is not None
        assert result.runtime["exact_plan_execution"]
        assert result.measurements[0].metadata["fq_output_name"] is None
        assert result.measurements[1].metadata["fq_output_name"] == "one"
        assert result.expectation("one").shape == (1,)
    assert replay.to_json() == encoded and replay.identity == selected.identity


def test_snapshot_capture_covers_source_matrix_names_and_output_order() -> None:
    program = IsingBraidProgram().exchange(1)
    matrix = program.projector("1")
    first = fq.expectation(matrix, name="one")
    second = fq.expectation(program.projector("psi"), name="psi")
    captured = plan(program, outputs=[first, second])
    encoded = captured.to_json()
    assert plan(program, outputs=[second, first]).identity != captured.identity
    assert (
        plan(program, outputs=fq.expectation(matrix, name="renamed")).identity
        != plan(program, outputs=first).identity
    )
    program.exchange(0)
    matrix.zero_()
    assert captured.to_json() == encoded
    result = run(captured)
    torch.testing.assert_close(
        result.expectation("one"),
        torch.tensor([0.5], dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "statevector"),
        ("mode", "mps"),
        ("mode", "tensor_network"),
        ("mode", "stabilizer"),
        ("backend", "jax"),
        ("backend", "tensorflow"),
        ("device", "cuda"),
        ("device", "meta"),
        ("target", "samples"),
        ("target", "amplitudes"),
        ("batch_size", 2),
        ("shots", 3),
        ("require_gradients", True),
        ("allow_approximate", True),
        ("allow_backend_fallback", True),
        ("memory_limit_bytes", 63),
    ],
)
def test_valid_unsupported_options_fail_before_numerical_execution(
    field: str, value: object
) -> None:
    constructor: Callable[..., ExecutionOptions] = ExecutionOptions
    options = constructor(**{field: value})
    with (
        patch("flagquantum.simulation.ising_braid.evolve_ising_braid") as numerical,
        pytest.raises(CapabilityError),
    ):
        run(IsingBraidProgram(), options=options)
    assert not numerical.called


def test_shared_options_precision_precedence_and_targets_remain_explicit() -> None:
    program = IsingBraidProgram().exchange(1)
    with runtime_config(complex_dtype="complex64"):
        assert plan(program).precision == "complex128"
    with pytest.raises(ValidationError, match="demote"):
        plan(program, options=ExecutionOptions(precision="complex64"))
    with runtime_config(backend="jax"):
        with pytest.raises(CapabilityError):
            plan(program)
        assert (
            plan(program, options=ExecutionOptions(backend="pytorch")).backend
            == "pytorch"
        )
    with pytest.raises(ValidationError, match="output"):
        plan(program, options=ExecutionOptions(target="expectation"))
    selected = plan(
        program,
        options=ExecutionOptions(target="state", seed=7, memory_limit_bytes=64),
        outputs=fq.expectation(program.projector("psi")),
    )
    assert selected.target == "state"
    result = run(selected)
    assert result.value is not None and result.state is not None
    assert run(program).value is None


@pytest.mark.parametrize(
    "matrix",
    [
        torch.eye(2, dtype=torch.complex64),
        torch.eye(2, dtype=torch.float64),
        torch.eye(2, dtype=torch.complex128, requires_grad=True),
    ],
)
def test_unsupported_output_storage_is_not_cast_or_detached(
    matrix: torch.Tensor,
) -> None:
    with pytest.raises(CapabilityError):
        plan(IsingBraidProgram(), outputs=fq.expectation(matrix))


@pytest.mark.parametrize(
    "matrix",
    [
        torch.eye(3, dtype=torch.complex128),
        torch.tensor([[1, 1], [0, 1]], dtype=torch.complex128),
        torch.tensor([[float("nan"), 0], [0, 1]], dtype=torch.complex128),
        torch.tensor([[float("inf"), 0], [0, 1]], dtype=torch.complex128),
    ],
)
def test_invalid_numerical_outputs_use_existing_matrix_validation(
    matrix: torch.Tensor,
) -> None:
    with pytest.raises(ValidationError):
        plan(IsingBraidProgram(), outputs=fq.expectation(matrix))


def test_output_kinds_naming_and_raw_source_authority_are_not_widened() -> None:
    program = IsingBraidProgram()
    for output in (fq.expectation(fq.X(0)), fq.probabilities(), fq.density_matrix()):
        with pytest.raises(CapabilityError):
            plan(program, outputs=output)
    with pytest.raises(TypeError, match="non-empty"):
        plan(program, outputs=[])
    with pytest.raises(ValidationError, match="unique"):
        plan(
            program,
            outputs=[
                fq.expectation(program.projector("1"), name="same"),
                fq.expectation(program.projector("psi"), name="same"),
            ],
        )
    entry: Callable[..., ExecutionPlan] = plan
    with pytest.raises(CapabilityError, match="Ising"):
        entry(program.to_ir())
    assert runtime_plan(program).analysis is None
    assert runtime_run(program).state is not None


@pytest.mark.parametrize(
    "field",
    [
        "options",
        "outputs",
        "noise_model",
        "shots",
        "compiler",
        "target",
        "target_qubits",
        "name",
    ],
)
def test_saved_plan_does_not_accept_overrides_or_remote_routes(field: str) -> None:
    selected = plan(IsingBraidProgram())
    entry: Callable[..., ExecutionResult] = run
    value = ExecutionOptions() if field == "options" else object()
    with (
        patch("flagquantum.simulation.ising_braid.evolve_ising_braid") as numerical,
        pytest.raises((TypeError, CapabilityError)),
    ):
        entry(selected, **{field: value})
    assert not numerical.called


@pytest.mark.parametrize("field", ["program", "options", "environment", "compiler"])
def test_saved_fingerprint_drift_fails_before_kernel_launch(field: str) -> None:
    payload = plan(IsingBraidProgram()).to_dict()
    fingerprints = payload["fingerprints"]
    assert isinstance(fingerprints, dict)
    fingerprints[field] = "0" * 64
    with pytest.raises(ExecutionPlanContractError):
        ExecutionPlan.from_dict(payload)


@pytest.mark.parametrize("shape", [[1000000, 1000000], [2.0, 2], [True, 2], [2, 2, 1]])
def test_output_shape_is_refused_before_the_core_tensor_decoder(
    shape: list[object],
) -> None:
    program = IsingBraidProgram()
    payload = json.loads(
        plan(program, outputs=fq.expectation(program.projector("1"))).to_json()
    )
    payload["outputs"][0]["operator"]["$tensor"]["shape"] = shape
    _rehash(payload)
    with (
        patch("flagquantum.runtime._ising_plan._decode_value") as decode,
        pytest.raises(ExecutionPlanContractError),
    ):
        ExecutionPlan.from_dict(payload)
    assert not decode.called


def test_rehashing_does_not_authorize_unknown_fields_or_contradictory_semantics() -> (
    None
):
    source = IsingBraidProgram()
    original = json.loads(plan(source).to_json())
    mutations = [
        ("top", "extra", True),
        ("decision", "world_size", 2),
        ("decision", "state_bytes", 0),
        ("decision", "target", "samples"),
        ("resolved_options", "require_gradients", True),
        ("requested_options", "precision", "complex64"),
    ]
    for section, key, value in mutations:
        payload = deepcopy(original)
        node = payload if section == "top" else payload[section]
        node[key] = value
        _rehash(payload)
        with pytest.raises(ExecutionPlanContractError):
            ExecutionPlan.from_dict(payload)


def test_malformed_source_and_environment_are_refused_before_numerical_work() -> None:
    program = IsingBraidProgram()
    source = program.to_ir()
    program._record = type(source)(())
    with pytest.raises(IRValidationError):
        plan(program)
    selected = plan(IsingBraidProgram())
    with (
        patch(
            "flagquantum.runtime.distributed.backend_policy.resolve_distributed_backend_policy",
            return_value=SimpleNamespace(effective_world_size=2),
        ),
        patch("flagquantum.simulation.ising_braid.evolve_ising_braid") as numerical,
    ):
        with pytest.raises(CapabilityError):
            plan(IsingBraidProgram())
        with pytest.raises(ExecutionPlanContractError, match="world_size_mismatch"):
            run(selected)
    assert not numerical.called


@pytest.mark.parametrize("profile", ["circuit", "rydberg"])
def test_native_saved_plan_cannot_borrow_another_program_profile(profile: str) -> None:
    payload = json.loads(plan(IsingBraidProgram()).to_json())
    if profile == "circuit":
        program = fq.Circuit(1).to_ir().to_dict()
    else:
        captures = json.loads(
            (
                Path(__file__).parents[1] / "fixtures/ising_braid_legacy_plans.json"
            ).read_text()
        )
        program = captures["plans"]["native_rydberg"]["program"]
    payload["program"] = program
    payload["fingerprints"]["program"] = canonical_hash(program)
    _rehash(payload)
    with pytest.raises(
        ExecutionPlanContractError, match="program_fingerprint_mismatch"
    ):
        ExecutionPlan.from_dict(payload)


def test_dispatch_profile_is_authenticated_by_the_seal_not_the_analysis_projection() -> (
    None
):
    circuit = plan(fq.Circuit(1))
    native = plan(IsingBraidProgram())
    for broken in (
        replace(circuit, analysis=None),
        replace(native, analysis=circuit.analysis),
    ):
        with (
            patch("flagquantum.simulation.ising_braid.evolve_ising_braid") as numerical,
            pytest.raises(ExecutionPlanContractError, match="analysis contradicts"),
        ):
            run(broken)
        assert not numerical.called


@pytest.mark.parametrize("mutation", ["extra", "dtype", "rows", "entry"])
def test_encoded_tensor_layout_is_bounded_before_decode(mutation: str) -> None:
    program = IsingBraidProgram()
    payload = json.loads(
        plan(program, outputs=fq.expectation(program.projector("psi"))).to_json()
    )
    tensor = payload["outputs"][0]["operator"]["$tensor"]
    if mutation == "extra":
        tensor["unreviewed"] = True
    elif mutation == "dtype":
        tensor["dtype"] = "complex64"
    elif mutation == "rows":
        tensor["data"] *= 100
    else:
        tensor["data"][0][0] = {"$parameter": "not_a_matrix_entry"}
    _rehash(payload)
    with (
        patch("flagquantum.runtime._ising_plan._decode_value") as decode,
        pytest.raises(ExecutionPlanContractError),
    ):
        ExecutionPlan.from_dict(payload)
    assert not decode.called


def test_native_reader_keeps_cpu_tensors_under_a_foreign_default_device() -> None:
    program = IsingBraidProgram()
    encoded = plan(program, outputs=fq.expectation(program.projector("1"))).to_json()
    with torch.device("meta"):
        restored = ExecutionPlan.from_json(encoded)
    assert restored.to_json() == encoded


def test_scalar_overflow_in_saved_tensor_is_a_structured_plan_error() -> None:
    program = IsingBraidProgram()
    payload = json.loads(
        plan(program, outputs=fq.expectation(program.projector("1"))).to_json()
    )
    payload["outputs"][0]["operator"]["$tensor"]["data"][0][0]["$complex"][0] = 10**400
    _rehash(payload)
    with pytest.raises(ExecutionPlanContractError):
        ExecutionPlan.from_dict(payload)

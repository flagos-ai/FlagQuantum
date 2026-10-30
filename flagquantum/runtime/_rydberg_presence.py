"""Plan admission and canonical results for one seeded presence trajectory."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch

from ..core.geometry import same_time
from ..core.ir.program.model import ProgramRecord
from ..core.ir.program.physical import RydbergContext, RydbergPresence
from ..core.ir.program.physical_codec import (
    native_program_from_dict,
    native_program_to_dict,
)
from ..errors import CapabilityError, ExecutionError, SerializationError
from ..observables.boson import DEFAULT_DENSE_MATRIX_BYTES
from ..simulation.lindblad_request import _error
from .execution_plan_contract import (
    ExecutionPlanContractError,
    _environment_requirements,
)
from .options import ExecutionOptions
from .result import ExecutionResult, MeasurementResult

if TYPE_CHECKING:
    from ..lindblad import LindbladPlan


def _sampling_options(options: ExecutionOptions | None) -> ExecutionOptions:
    from ..lindblad import _execution_settings

    if options is not None and not isinstance(options, ExecutionOptions):
        raise TypeError("options must be an fq.ExecutionOptions instance")
    if (
        options is None
        or type(options.shots) is not int
        or options.shots != 1
        or options.seed is None
    ):
        raise CapabilityError("presence requires explicit shots=1 and seed")
    if type(options.seed) is not int:
        raise TypeError("seed must be a Python integer")
    if not 0 <= options.seed < 2**64:
        raise ValueError("seed must be in [0, 2**64-1]")
    numeric = replace(options, shots=None, seed=None)
    device, dtype, batch, gradients = _execution_settings(numeric)
    if (
        str(device).split(":", 1)[0] not in {"cpu", "flagos"}
        or dtype != torch.complex128
        or batch not in (None, 1)
        or gradients
    ):
        raise CapabilityError(
            "presence requires one CPU complex128 trajectory without gradients"
        )
    return numeric


def _trainable(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return value.requires_grad
    if isinstance(value, Mapping):
        return any(_trainable(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_trainable(item) for item in value)
    return False


def _environment(device: str = "cpu") -> dict[str, object]:
    return _environment_requirements(
        {
            "backend": "pytorch",
            "device": device,
            "precision": "complex128",
            "world_size": 1,
            "require_gradients": False,
            "allow_approximate": False,
        },
        rng_torch_version=str(torch.__version__),
    )


def validate_presence_options(payload: Mapping[str, Any]) -> None:
    """Check the sealed RNG build before source assembly or numerical work."""
    if payload.get("environment_requirements") != _environment(
        payload.get("options", {}).get("device") or "cpu"
    ):
        raise ExecutionPlanContractError(
            "environment_incompatible",
            "presence requires the captured Torch RNG build and CPU environment",
        )
    try:
        _sampling_options(ExecutionOptions.from_dict(payload["options"]))
    except (TypeError, ValueError, CapabilityError, KeyError) as exc:
        raise SerializationError(f"invalid presence options: {exc}") from exc


def check_presence_budget(record: ProgramRecord, count: int, retain: bool) -> None:
    context = record.physical_context
    if not isinstance(context, RydbergContext) or not 1 <= len(context.atoms) <= 2:
        raise CapabilityError("presence requires one or two Rydberg actors")
    dimension = 3 ** len(context.atoms)
    subtotal = 48 * dimension**2 + 24
    if retain:
        subtotal += 16 * count * dimension**2
    if subtotal > DEFAULT_DENSE_MATRIX_BYTES:
        raise CapabilityError(
            "presence numeric allocations exceed the dense allocation ceiling"
        )


def check_presence_request(record: ProgramRecord, request: Mapping[str, Any]) -> None:
    context = record.physical_context
    if not isinstance(context, RydbergContext) or not 1 <= len(context.atoms) <= 2:
        raise CapabilityError("presence requires one or two Rydberg actors")
    if request.get("require_gradients") or _trainable(request):
        raise CapabilityError(
            "presence inputs, grids, controls and outputs cannot require gradients"
        )
    initial = torch.as_tensor(request["initial_state"])
    if (
        initial.device.type not in {"cpu", "flagos"}
        or initial.ndim not in (1, 2)
        or (initial.ndim == 2 and initial.shape[0] != initial.shape[1])
    ):
        raise CapabilityError("presence requires one unbatched CPU state or density")
    times = torch.as_tensor(request["times"], dtype=torch.float64)
    observations = [
        op.physical_data
        for op in record.functions[0].blocks[0].operations
        if isinstance(op.physical_data, RydbergPresence)
    ]
    if len(observations) != 1:
        raise CapabilityError("presence requires exactly one observation")
    matches = (
        [
            i
            for i, t in enumerate(times.tolist())
            if same_time(t, observations[0].at, precision="float64")
        ]
        if times.ndim == 1
        else []
    )
    if len(matches) != 1 or not 0 < matches[0] < times.numel() - 1:
        raise _error(
            "invalid_native_request",
            "presence needs one distinct internal grid boundary",
            "times",
        )
    check_presence_budget(
        record, times.numel(), bool(request.get("return_density_matrices"))
    )
    reserved = {"mid_circuit_measurement"}
    if request.get("return_density_matrices"):
        reserved.add("density_matrix")
    if reserved.intersection(request.get("observables") or {}):
        raise _error(
            "duplicate_output_name", "reserved presence output name", "outputs"
        )


def build_presence_plan(
    record: ProgramRecord,
    initial_state: Any,
    times: Any,
    *,
    options: ExecutionOptions | None,
    outputs: Any,
    **settings: Any,
) -> LindbladPlan:
    from ..lindblad import LindbladPlan, _observables
    from ..lindblad._native_plan import native_request
    from ..lindblad._plan import _identity, build_lindblad_plan
    from .distributed.backend_policy import resolve_distributed_backend_policy

    numeric_options = _sampling_options(options)
    if resolve_distributed_backend_policy().effective_world_size != 1:
        raise CapabilityError("presence requires world_size=1")
    observables = _observables(outputs)
    check_presence_request(
        record,
        dict(
            initial_state=initial_state,
            times=times,
            observables=observables,
            **settings,
        ),
    )
    numeric, bindings = native_request(
        record,
        times,
        allow_presence=True,
        dtype=torch.complex128,
        device=numeric_options.device or "cpu",
        **{
            key: settings[key]
            for key in (
                "n_qubits",
                "dimensions",
                "basis_labels",
                "collapse_operators",
                "parameter_values",
            )
        },
    )
    plan = build_lindblad_plan(
        **numeric,
        initial_state=initial_state,
        observables=observables,
        device=numeric_options.device or "cpu",
        dtype=torch.complex128,
        batch_size=numeric_options.batch_size,
        require_gradients=False,
        return_density_matrices=settings["return_density_matrices"],
        method=settings["method"],
        solve_tolerance=settings["solve_tolerance"],
    )
    payload = plan.to_dict()
    assert options is not None
    payload.update(
        version="1.8",
        program=native_program_to_dict(record),
        bindings=bindings,
        options=options.to_dict(),
        environment_requirements=_environment(numeric_options.device or "cpu"),
    )
    payload["identity"] = _identity(payload)
    # No caller graph or mutable tensor is retained by the sampled profile.
    return LindbladPlan.from_dict(payload)


def execute_presence_plan(plan: LindbladPlan) -> ExecutionResult:
    from ..lindblad import LindbladPlan
    from ..simulation.rydberg_presence import (
        _continue_rydberg_presence,
        _presence_operators,
        _rydberg_presence_prefix,
    )
    from ._terminal_instruments import _instrument_probabilities
    from .distributed.backend_policy import resolve_distributed_backend_policy
    from .measurements import _draw_probability_indices

    payload = plan.to_dict()
    if payload.get("version") != "1.8":
        raise CapabilityError("fq.run accepts only sampled Lindblad plan 1.8")
    checked = LindbladPlan.from_dict(payload)
    if resolve_distributed_backend_policy().effective_world_size != 1:
        raise ExecutionPlanContractError(
            "environment_incompatible", "presence requires world_size=1"
        )
    record = native_program_from_dict(payload["program"])
    request = checked._execution_request()
    numeric = {
        key: request[key]
        for key in (
            "hamiltonian",
            "n_qubits",
            "times",
            "dimensions",
            "collapse_operators",
            "bound_hamiltonians",
        )
    }
    grid = torch.as_tensor(numeric["times"], dtype=torch.float64)
    numeric["times"] = grid
    source_index, observation = next(
        (i, op.physical_data)
        for i, op in enumerate(record.functions[0].blocks[0].operations)
        if isinstance(op.physical_data, RydbergPresence)
    )
    boundary = next(
        i
        for i, value in enumerate(grid.tolist())
        if same_time(value, observation.at, precision="float64")
    )
    settings = {key: request[key] for key in ("method", "solve_tolerance")}
    prefix = _rydberg_presence_prefix(
        numeric, request["initial_state"].to(request["device"]), boundary, **settings
    )
    branches: list[tuple[str, tuple[torch.Tensor, ...]]] = [
        (label, (operator,))
        for label, operator in zip(
            ("no_flag", "flag"), _presence_operators(record, prefix.device), strict=True
        )
    ]
    weights, probabilities = _instrument_probabilities(
        prefix[-1].unsqueeze(0), branches, 1e-10
    )
    seed = payload["options"]["seed"]
    selected = _draw_probability_indices(
        probabilities,
        1,
        generator=torch.Generator(device=prefix.device).manual_seed(seed),
    )
    if (
        not isinstance(selected, torch.Tensor)
        or selected.shape != (1, 1)
        or selected.dtype != torch.int64
        or selected.device != prefix.device
    ):
        raise ExecutionError("presence draw must be one CPU int64 index")
    index = int(selected[0, 0])
    if index not in (0, 1) or weights[0, index] <= 0:
        raise ExecutionError("presence selected a zero-probability or invalid branch")
    trajectory = _continue_rydberg_presence(record, numeric, prefix, index, **settings)
    records = []
    for i, (name, operator) in enumerate(request["observables"].items()):
        value = torch.einsum(
            "tij,ji->t", trajectory, operator.to(trajectory.device)
        ).real
        records.append(
            MeasurementResult(
                "expectation",
                (),
                value,
                metadata={
                    "name": name,
                    "fq_output_name": name,
                    "fq_output_index": i,
                    "fq_output_kind": "expectation",
                },
            )
        )
    if request["return_density_matrices"]:
        records.append(
            MeasurementResult(
                "density_matrix",
                (),
                trajectory,
                metadata={"axes": ["time", "basis_row", "basis_column"]},
            )
        )
    context = record.physical_context
    assert isinstance(context, RydbergContext)
    records.append(
        MeasurementResult(
            "mid_circuit_measurement",
            (),
            selected,
            shots=1,
            metadata={
                "outcome_labels": ["no_flag", "flag"],
                "probabilities": weights.detach().cpu().tolist(),
                "probability_correction": float((probabilities - weights).abs().max()),
                "atom_id": observation.atom,
                "time_index": boundary,
                "time": observation.at,
                "time_unit": context.time_unit,
                "stage": "intermediate",
                "source_operation_index": source_index,
            },
        )
    )
    return ExecutionResult(
        state=trajectory[-1].unsqueeze(0),
        plan=plan,
        measurements=tuple(records),
        runtime={
            "mode": "density_matrix",
            "backend": "pytorch",
            "device": str(trajectory.device),
            "precision": "complex128",
            "world_size": 1,
            "is_distributed": False,
            "distribution_semantics": "single_device_fast_path",
            "local_world_size": 1,
            "node_count": 1,
            "execution_path": "native_rydberg_presence",
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
            "exact_plan_execution": True,
            "plan_identity": checked.identity,
            "seed": seed,
            "shots": 1,
            "time_grid": grid.tolist(),
            "state_axes": ["sample", "basis_row", "basis_column"],
            "expectation_axes": ["time"],
        },
    )

"""Canonical execution services for FlagQuantum circuits and devices."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Sequence

import torch

from ..compiler import compile as compile_program
from ..compiler import lower_noise_model
from ..core.ir import CircuitIR, MeasurementNode, ensure_circuit_ir
from ..core.numerics import coerce_accuracy_requirement, coerce_precision_plan
from ..core.runtime_config import (
    RuntimeConfig,
    get_runtime_config,
    runtime_config,
)
from ..errors import ExecutionError
from ..runtime.backend_registry import resolve_device, resolve_dtype
from ..runtime.distributed.backend_policy import (
    DistributedBackendPolicy,
    resolve_distributed_backend_policy,
)
from .execution_plan import ExecutionPlan
from .executors.jax import (
    run_jax_sharded_mps,
    run_jax_sharded_tensor_network,
)
from .executors.statevector import (
    execute_torch_distributed_statevector,
    simulate_distributed_statevector_local,
)
from .planner import build_noisy_execution_plan, select_execution_mode
from .planner import plan_advanced as build_plan

if TYPE_CHECKING:
    from ..circuit import Circuit
    from ..noise import NoiseModel
    from .options import ExecutionOptions
    from .result import ExecutionResult


def _resolve_policy_from_options(
    device_options: dict[str, Any],
) -> DistributedBackendPolicy:
    backend_policy = device_options.pop("distributed_backend_policy", None)
    distributed_profile = device_options.pop("distributed_profile", None)
    jax_backend = device_options.pop("jax_backend", None)
    torch_backend = device_options.pop("torch_backend", None)
    if backend_policy is not None:
        policy = backend_policy
    else:
        policy = resolve_distributed_backend_policy(profile=distributed_profile)
    if jax_backend is None and torch_backend is None:
        return policy
    source = dict(policy.source)
    if jax_backend is not None:
        source["runtime_jax_backend"] = str(jax_backend)
    if torch_backend is not None:
        source["runtime_torch_backend"] = str(torch_backend)
    return replace(
        policy,
        jax_backend=str(jax_backend or policy.jax_backend),
        torch_backend=str(torch_backend or policy.torch_backend),
        source=source,
    )


def _peek_policy_from_options(options: dict[str, Any]) -> DistributedBackendPolicy:
    backend_policy = options.get("distributed_backend_policy")
    jax_backend = options.get("jax_backend")
    torch_backend = options.get("torch_backend")
    if backend_policy is not None:
        policy = backend_policy
    else:
        policy = resolve_distributed_backend_policy(
            profile=options.get("distributed_profile")
        )
    if jax_backend is None and torch_backend is None:
        return policy
    return replace(
        policy,
        jax_backend=str(jax_backend or policy.jax_backend),
        torch_backend=str(torch_backend or policy.torch_backend),
    )


def _env_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _resolve_execution_precision(
    program_dtype: str | torch.dtype,
    *,
    requested_dtype: str | torch.dtype | None = None,
    requested_precision: str | torch.dtype | None = None,
) -> torch.dtype:
    if requested_dtype is not None and requested_precision is not None:
        _, dtype_precision = resolve_dtype(requested_dtype)
        _, named_precision = resolve_dtype(requested_precision)
        if dtype_precision != named_precision:
            raise ValueError(
                "dtype and precision request different execution precisions"
            )
    _, program_precision = resolve_dtype(program_dtype)
    _, execution_precision = resolve_dtype(
        requested_dtype or requested_precision or program_precision
    )
    if program_precision == torch.complex128 and execution_precision == torch.complex64:
        raise ValueError(
            "precision=complex64 would demote a complex128 program; construct "
            "the program with complex64 precision instead"
        )
    return execution_precision


def _validate_flagos_statevector(
    ir: CircuitIR,
    options: dict[str, Any],
    *,
    accuracy_requirement: Any = None,
    precision_plan: Any = None,
) -> tuple[Any, Any, Any, Any] | None:
    """Validate the requested FlagOS statevector workload and its numerics."""

    requested_device = options.get("device", "auto")
    device = resolve_device(requested_device)
    if device.type != "flagos":
        if accuracy_requirement is not None or precision_plan is not None:
            raise NotImplementedError(
                "explicit numerical contract enforcement is currently available "
                "only for local statevector execution through flagos"
            )
        return None
    from ..compute import get_platform_runtime
    from .numerical_validation import certify_statevector_local_p0
    from .operator_probes import preflight_statevector_local_p0

    platform = get_platform_runtime("flagos")
    provider = platform.identity().provider
    _, complex_dtype = resolve_dtype(options.get("dtype") or ir.dtype)
    dtype = str(complex_dtype).removeprefix("torch.")
    report = preflight_statevector_local_p0(
        device=device,
        dtype=dtype,
        provider=provider,
    )
    report.require_supported()
    accuracy = coerce_accuracy_requirement(accuracy_requirement, dtype=dtype)
    precision = coerce_precision_plan(precision_plan, dtype=dtype)
    numerical_report = certify_statevector_local_p0(
        device=device,
        dtype=dtype,
        provider=provider,
        accuracy_requirement=accuracy,
        precision_plan=precision,
    )
    numerical_report.require_accepted()
    return report, accuracy, precision, numerical_report


def _distributed_world_size_from_options(
    options: dict[str, Any],
    *,
    backend_policy: DistributedBackendPolicy,
    default: int = 1,
) -> int:
    """Resolve effective distributed world size without changing user code.

    Explicit API options win. Otherwise production follows torchrun's
    WORLD_SIZE, while development follows FQ_LOCAL_WORLD_SIZE so a laptop CPU
    can exercise the same rank ownership and communication signatures.
    """

    if "world_size" in options:
        return max(1, int(options["world_size"]))
    if "world_sz" in options:
        return max(1, int(options["world_sz"]))
    env_world_size = _env_int("WORLD_SIZE", 0)
    if backend_policy.profile == "production" and env_world_size > 0:
        return max(1, env_world_size)
    if backend_policy.profile == "development" and backend_policy.local_world_size > 1:
        return max(1, int(backend_policy.local_world_size))
    return max(1, int(default))


def _distributed_local_world_size(
    *,
    world_size: int,
    backend_policy: DistributedBackendPolicy,
) -> int:
    if backend_policy.local_world_size > 1:
        return max(1, min(int(world_size), int(backend_policy.local_world_size)))
    return max(1, int(world_size))


def run_distributed(
    ir: CircuitIR,
    *,
    device: torch.device | str | None = None,
    measure: bool = False,
    optimize: bool = True,
    return_plan: bool = False,
    coupling_map: Any | None = None,
    **device_options: Any,
) -> Any:
    """Execute a FlagQuantum IR on a native distributed device."""

    if device is not None and not isinstance(device, (str, torch.device)):
        raise TypeError("device must be a torch.device, device string, or None")
    provided_execution_plan = device_options.pop("_execution_plan", None)
    execution_ir = (
        ir
        if provided_execution_plan is not None
        else compile_program(
            ir,
            coupling_map=coupling_map,
            optimize=optimize,
        )
    )
    requested_dtype = device_options.pop("dtype", None)
    requested_precision = device_options.pop("precision", None)
    execution_precision = _resolve_execution_precision(
        execution_ir.dtype,
        requested_dtype=requested_dtype,
        requested_precision=requested_precision,
    )
    if "world_size" in device_options and "world_sz" not in device_options:
        device_options["world_sz"] = device_options.pop("world_size")
    backend_policy = _resolve_policy_from_options(device_options)
    if device is None:
        device = device_options.pop("device", None)
    if device is None or str(device) == "auto":
        device = resolve_device(device)
    world_size = _distributed_world_size_from_options(
        device_options,
        backend_policy=backend_policy,
    )
    device_options["world_sz"] = world_size
    batch_size = int(device_options.get("bsz", 1))
    local_world_size = _distributed_local_world_size(
        world_size=world_size,
        backend_policy=backend_policy,
    )
    execution_plan = provided_execution_plan or build_plan(
        execution_ir,
        bsz=batch_size,
        world_size=world_size,
        complex_bytes=execution_precision.itemsize,
        optimize=False,
    )
    if (
        backend_policy.profile == "development"
        and backend_policy.torch_backend == "local_tensor"
    ):
        from ..simulation.statevector.noisy import expectation_z

        local_result = simulate_distributed_statevector_local(
            execution_ir,
            world_size=world_size,
            bsz=batch_size,
            device=device,
            dtype=execution_precision,
            backend_policy=backend_policy,
        )
        result = (
            expectation_z(local_result.state, execution_ir.n_wires)
            if measure
            else local_result
        )
    elif (
        backend_policy.profile == "production"
        and backend_policy.torch_backend == "torch_distributed"
    ):
        if world_size > 1 and not torch.distributed.is_initialized():
            raise ExecutionError(
                "production distributed statevector execution requires an initialized "
                "torch.distributed process group"
            )
        process_group = device_options.get("process_group")
        actual_world_size = (
            torch.distributed.get_world_size(process_group)
            if torch.distributed.is_initialized()
            else 1
        )
        if actual_world_size != world_size:
            raise ExecutionError(
                "planned distributed world size does not match the active process group: "
                f"{world_size} != {actual_world_size}"
            )
        if measure:
            raise ExecutionError(
                "measure=True is not supported by the rank-local production result; "
                "use a supported distributed observable execution path"
            )
        executor_option_names = {
            "exchange_buffer_bytes",
            "compact_index_threshold",
            "process_group",
            "fuse_cross_shard_gates",
            "pipeline_pair_exchange",
            "wire_layout",
            "preferred_local_wires",
            "persistent_wire_layout",
        }
        executor_options = {
            name: device_options[name]
            for name in executor_option_names
            if name in device_options
        }
        result = execute_torch_distributed_statevector(
            execution_ir,
            device=device,
            dtype=execution_precision,
            local_world_size=local_world_size,
            **executor_options,
        )
    else:
        raise ExecutionError(
            "unsupported distributed statevector backend policy: "
            f"{backend_policy.profile}/{backend_policy.torch_backend}"
        )
    if return_plan:
        return result, execution_plan
    return result


def run_native(
    circuit_or_ir: Any,
    *,
    noise_model: NoiseModel | None = None,
    mode: str = "auto",
    return_plan: bool = False,
    **options: Any,
) -> Any:
    """Run FlagQuantum IR through native local execution paths."""

    config_active = bool(options.pop("_runtime_config_active", False))
    selected_config = options.pop("config", None)
    if not config_active:
        ir = ensure_circuit_ir(circuit_or_ir)
        if selected_config is None:
            metadata = ir.metadata
            manifest = metadata.get("runtime_config")
            selected_config = (
                RuntimeConfig.from_manifest(manifest)
                if manifest is not None
                else get_runtime_config()
            )
        execution_dtype = _resolve_execution_precision(
            ir.dtype,
            requested_dtype=options.get("dtype"),
        )
        selected_config = selected_config.with_overrides(
            complex_dtype=str(execution_dtype).removeprefix("torch.")
        )
        options = {**options, "dtype": execution_dtype}
        with runtime_config(selected_config):
            return run_native(
                circuit_or_ir,
                noise_model=noise_model,
                mode=mode,
                return_plan=return_plan,
                _runtime_config_active=True,
                **options,
            )

    operator_backend_name = options.pop("operator_backend", None)
    operator_backend_include = options.pop("operator_backend_include", None)
    operator_backend_include_experimental = options.pop(
        "operator_backend_include_experimental", None
    )
    operator_backend_strict = options.pop("operator_backend_strict", None)
    operator_backend_active = bool(options.pop("_operator_backend_active", False))
    if not operator_backend_active and (
        operator_backend_name is not None or os.environ.get("FQ_OPERATOR_BACKEND")
    ):
        from ..compute.flaggems import (
            operator_backend as operator_backend_context,
        )

        wrapped_options = dict(options)
        wrapped_options["_operator_backend_active"] = True
        with operator_backend_context(
            operator_backend_name,
            include=operator_backend_include,
            include_experimental=operator_backend_include_experimental,
            strict=operator_backend_strict,
        ):
            return run_native(
                circuit_or_ir,
                noise_model=noise_model,
                mode=mode,
                return_plan=return_plan,
                **wrapped_options,
            )

    provided_execution_plan = options.pop("_execution_plan", None)
    if mode == "distributed":
        mode = "distributed_statevector"
    if mode == "tn":
        mode = "tensor_network"
    if mode == "distributed_tn":
        mode = "distributed_tensor_network"
    if mode == "jax_sharded_tn":
        mode = "jax_sharded_tensor_network"
    output_target = options.pop("output_target", "full_state")
    target_count = int(options.pop("target_count", 1))
    require_gradients = bool(options.pop("require_gradients", False))
    allow_approximate = bool(options.pop("allow_approximate", True))
    noise_performance_calibration = options.pop("noise_performance_calibration", None)
    selector_min_trajectories = int(options.get("min_trajectories", 1))
    selector_target_standard_error = options.get("target_standard_error")
    selector_pilot_variance = options.pop("pilot_variance", None)
    selector_pilot_trajectories = options.pop("pilot_trajectories", None)
    selector_pilot_confidence_level = options.pop("pilot_confidence_level", None)
    selector_pilot_observable_count = int(options.pop("pilot_observable_count", 1))
    ir = ensure_circuit_ir(circuit_or_ir)
    distributed_mode_requested = mode in {
        "auto",
        "distributed_statevector",
        "distributed_mps",
        "jax_sharded_mps",
        "distributed_tensor_network",
        "jax_sharded_tensor_network",
    }
    local_mode_world_size = int(options.get("world_size", options.get("world_sz", 1)))
    if distributed_mode_requested:
        backend_policy_for_planning = _peek_policy_from_options(options)
        effective_world_size = _distributed_world_size_from_options(
            options,
            backend_policy=backend_policy_for_planning,
        )
    else:
        effective_world_size = local_mode_world_size
    plan_options: dict[str, Any] = {
        "bsz": int(options.get("bsz", 1)),
        "world_size": (
            effective_world_size
            if distributed_mode_requested
            else local_mode_world_size
        ),
        "target": output_target,
        "target_count": target_count,
        "require_gradients": require_gradients,
        "allow_approximate": allow_approximate,
        "complex_bytes": torch.empty((), dtype=options["dtype"]).element_size(),
    }
    if "max_bond" in options:
        plan_options["max_bond"] = options["max_bond"]
    if "cutoff" in options:
        plan_options["cutoff"] = options["cutoff"]
    if "memory_limit_bytes" in options:
        plan_options["memory_limit_bytes"] = options["memory_limit_bytes"]
    if "trajectories" in options:
        plan_options["trajectories"] = options["trajectories"]
    if options.get("device") is None or str(options.get("device")) == "auto":
        options = dict(options)
        source_device = getattr(circuit_or_ir, "device", None)
        if source_device is not None and str(source_device) != "auto":
            options["device"] = str(source_device)
        else:
            options["device"] = str(resolve_device(options.get("device")))
    optimize = bool(options.get("optimize", True))
    coupling_map = options.get("coupling_map")
    execution_ir = (
        ir
        if provided_execution_plan is not None
        else compile_program(
            ir,
            coupling_map=coupling_map,
            optimize=optimize if coupling_map is not None else False,
        )
    )
    if mode == "auto":
        mode = select_execution_mode(
            execution_ir,
            noise_model=noise_model,
            trajectory_batch_size=int(options.get("trajectory_batch_size", 32)),
            noise_performance_calibration=noise_performance_calibration,
            min_trajectories=selector_min_trajectories,
            target_standard_error=selector_target_standard_error,
            pilot_variance=selector_pilot_variance,
            pilot_trajectories=selector_pilot_trajectories,
            pilot_confidence_level=selector_pilot_confidence_level,
            pilot_observable_count=selector_pilot_observable_count,
            **plan_options,
        )

    if mode == "distributed_statevector":
        if noise_model is not None:
            raise ValueError("Noise models require density_matrix mode.")
        distributed_options = dict(options)
        distributed_options["precision"] = distributed_options.pop("dtype")
        distributed_options.pop("max_bond", None)
        distributed_options.pop("cutoff", None)
        distributed_options.pop("memory_limit_bytes", None)
        distributed_options.pop("optimize", None)
        return run_distributed(
            execution_ir,
            optimize=False,
            return_plan=return_plan,
            _execution_plan=provided_execution_plan,
            **distributed_options,
        )

    if mode == "statevector":
        if noise_model is not None:
            raise ValueError("Noise models require density_matrix mode.")
        accuracy_requirement_option = options.pop("accuracy_requirement", None)
        precision_plan_option = options.pop("precision_plan", None)
        statevector_contracts = _validate_flagos_statevector(
            execution_ir,
            options,
            accuracy_requirement=accuracy_requirement_option,
            precision_plan=precision_plan_option,
        )
        if (
            provided_execution_plan is None
            and coupling_map is None
            and hasattr(circuit_or_ir, "state")
            and getattr(circuit_or_ir, "dtype", None) == options["dtype"]
        ):
            result = circuit_or_ir.state()
        else:
            from ..simulation.statevector.local import run_local_statevector

            resolved_device = resolve_device(options.get("device"))
            result = run_local_statevector(
                execution_ir,
                batch_size=int(options.get("bsz", 1)),
                device=resolved_device,
                dtype=options["dtype"],
            )
        execution_plan = provided_execution_plan or build_plan(
            execution_ir, state_mode="statevector", **plan_options
        )
        if statevector_contracts is not None:
            (
                operator_preflight,
                accuracy_requirement,
                precision_plan,
                numerical_validation,
            ) = statevector_contracts
            routing_plan = dict(execution_plan.routing_plan or {})
            routing_plan["operator_preflight"] = operator_preflight.to_dict()
            routing_plan["accuracy_requirement"] = accuracy_requirement.to_dict()
            routing_plan["accuracy_requirement_hash"] = (
                accuracy_requirement.content_hash()
            )
            routing_plan["precision_plan"] = precision_plan.to_dict()
            routing_plan["precision_plan_hash"] = precision_plan.content_hash()
            routing_plan["numerical_validation"] = numerical_validation.to_dict()
            if provided_execution_plan is None:
                execution_plan = replace(execution_plan, routing_plan=routing_plan)
    elif mode == "density_matrix":
        from .noise_registry import execute_noisy_plan

        lowered = (
            execution_ir
            if provided_execution_plan is not None
            else lower_noise_model(execution_ir, noise_model)
        )
        execution_plan = provided_execution_plan or build_plan(
            lowered,
            state_mode="density_matrix",
            **plan_options,
        )
        noisy_plan = build_noisy_execution_plan(
            execution_plan,
            representation="density_matrix",
            evolution="exact_channel",
            memory_limit_bytes=options.get("memory_limit_bytes"),
            noise_model_identity=getattr(noise_model, "identity", None),
        )
        if provided_execution_plan is None:
            execution_plan = replace(
                execution_plan,
                noisy_execution_plan=noisy_plan,
            )
        density_options = dict(options)
        density_options.pop("coupling_map", None)
        density_options.pop("optimize", None)
        result = execute_noisy_plan(
            lowered,
            noisy_plan,
            options=density_options,
        )
    elif mode == "mps":
        if noise_model is not None:
            raise ValueError("Noise models require density_matrix mode.")
        from ..simulation.mps.entrypoints import run_mps

        mps_options = dict(options)
        mps_options.pop("memory_limit_bytes", None)
        mps_options.pop("world_size", None)
        mps_options.pop("world_sz", None)
        mps_options.pop("coupling_map", None)
        mps_options.pop("optimize", None)
        result = run_mps(
            execution_ir if coupling_map is not None else circuit_or_ir, **mps_options
        )
        execution_plan = provided_execution_plan or build_plan(
            execution_ir,
            noise_model=noise_model,
            state_mode="mps",
            **plan_options,
        )
    elif mode == "adaptive_mps":
        if noise_model is not None:
            raise ValueError("Noise models require density_matrix mode.")
        from ..simulation.mps.entrypoints import run_mps_adaptive

        mps_options = dict(options)
        mps_options.pop("memory_limit_bytes", None)
        mps_options.pop("world_size", None)
        mps_options.pop("world_sz", None)
        mps_options.pop("coupling_map", None)
        mps_options.pop("optimize", None)
        result = run_mps_adaptive(
            execution_ir if coupling_map is not None else circuit_or_ir, **mps_options
        )
        execution_plan = provided_execution_plan or build_plan(
            execution_ir,
            noise_model=noise_model,
            state_mode="mps",
            **plan_options,
        )
    elif mode == "distributed_mps":
        if noise_model is not None:
            raise ValueError("Noise models require density_matrix mode.")
        from .executors.mps import run_distributed_mps

        mps_options = dict(options)
        world_size = _distributed_world_size_from_options(
            mps_options,
            backend_policy=_peek_policy_from_options(mps_options),
        )
        mps_options.pop("world_size", None)
        mps_options.pop("world_sz", None)
        mps_options.pop("memory_limit_bytes", None)
        mps_options.pop("coupling_map", None)
        mps_options.pop("optimize", None)
        result = run_distributed_mps(
            execution_ir if coupling_map is not None else circuit_or_ir,
            world_size=world_size,
            **mps_options,
        )
        execution_plan = provided_execution_plan or build_plan(
            execution_ir,
            noise_model=noise_model,
            state_mode="mps",
            world_size=world_size,
            **{
                key: value for key, value in plan_options.items() if key != "world_size"
            },
        )
    elif mode == "jax_sharded_mps":
        if noise_model is not None:
            raise ValueError("Noise models require density_matrix mode.")

        mps_options = dict(options)
        world_size = _distributed_world_size_from_options(
            mps_options,
            backend_policy=_peek_policy_from_options(mps_options),
        )
        mps_options.pop("world_size", None)
        mps_options.pop("world_sz", None)
        mps_options.pop("memory_limit_bytes", None)
        mps_options.pop("coupling_map", None)
        mps_options.pop("optimize", None)
        result = run_jax_sharded_mps(
            execution_ir if coupling_map is not None else circuit_or_ir,
            world_size=world_size,
            **mps_options,
        )
        execution_plan = provided_execution_plan or build_plan(
            execution_ir,
            noise_model=noise_model,
            state_mode="mps",
            world_size=world_size,
            **{
                key: value for key, value in plan_options.items() if key != "world_size"
            },
        )
    elif mode == "tensor_network":
        if noise_model is not None:
            raise ValueError("Noise models require density_matrix mode.")
        from ..simulation.tensor_network.entrypoints import run_tensor_network

        tn_options = dict(options)
        tn_options.pop("memory_limit_bytes", None)
        tn_options.pop("world_size", None)
        tn_options.pop("world_sz", None)
        tn_options.pop("coupling_map", None)
        tn_options.pop("optimize", None)
        result = run_tensor_network(
            execution_ir if coupling_map is not None else circuit_or_ir, **tn_options
        )
        execution_plan = provided_execution_plan or build_plan(
            execution_ir,
            noise_model=noise_model,
            state_mode="tensor_network",
            **plan_options,
        )
    elif mode == "distributed_tensor_network":
        if noise_model is not None:
            raise ValueError("Noise models require density_matrix mode.")
        from .executors.tensor_network.execution import run_distributed_tensor_network

        tn_options = dict(options)
        world_size = _distributed_world_size_from_options(
            tn_options,
            backend_policy=_peek_policy_from_options(tn_options),
        )
        tn_options.pop("world_size", None)
        tn_options.pop("world_sz", None)
        tn_options.pop("memory_limit_bytes", None)
        tn_options.pop("coupling_map", None)
        tn_options.pop("optimize", None)
        result = run_distributed_tensor_network(
            execution_ir if coupling_map is not None else circuit_or_ir,
            world_size=world_size,
            **tn_options,
        )
        execution_plan = provided_execution_plan or build_plan(
            execution_ir,
            noise_model=noise_model,
            state_mode="tensor_network",
            world_size=world_size,
            **{
                key: value for key, value in plan_options.items() if key != "world_size"
            },
        )
    elif mode == "jax_sharded_tensor_network":
        if noise_model is not None:
            raise ValueError("Noise models require density_matrix mode.")

        tn_options = dict(options)
        world_size = _distributed_world_size_from_options(
            tn_options,
            backend_policy=_peek_policy_from_options(tn_options),
        )
        tn_options.pop("world_size", None)
        tn_options.pop("world_sz", None)
        tn_options.pop("memory_limit_bytes", None)
        tn_options.pop("coupling_map", None)
        tn_options.pop("optimize", None)
        result = run_jax_sharded_tensor_network(
            execution_ir if coupling_map is not None else circuit_or_ir,
            world_size=world_size,
            **tn_options,
        )
        execution_plan = provided_execution_plan or build_plan(
            execution_ir,
            noise_model=noise_model,
            state_mode="tensor_network",
            world_size=world_size,
            **{
                key: value for key, value in plan_options.items() if key != "world_size"
            },
        )
    elif mode == "noisy_statevector":
        if noise_model is None:
            raise ValueError("noisy_statevector mode requires a noise_model")
        from .executors.statevector import run_noisy_statevector
        from .executors.statevector.noisy import _run_lowered_noisy_statevector

        statevector_options = dict(options)
        statevector_options.pop("memory_limit_bytes", None)
        if "world_sz" in statevector_options:
            statevector_options["world_size"] = int(statevector_options.pop("world_sz"))
        statevector_options.pop("coupling_map", None)
        statevector_options.pop("optimize", None)
        result = (
            _run_lowered_noisy_statevector(
                execution_ir,
                noise_model,
                **statevector_options,
            )
            if provided_execution_plan is not None
            else run_noisy_statevector(
                execution_ir if coupling_map is not None else circuit_or_ir,
                noise_model,
                **statevector_options,
            )
        )
        if provided_execution_plan is not None:
            execution_plan = provided_execution_plan
        else:
            execution_plan = build_plan(
                execution_ir,
                noise_model=noise_model,
                state_mode="statevector",
                **plan_options,
            )
            execution_plan = replace(
                execution_plan,
                noisy_execution_plan=build_noisy_execution_plan(
                    execution_plan,
                    representation="statevector",
                    evolution="quantum_trajectory",
                    trajectories=int(options.get("trajectories", 32)),
                    seed=options.get("seed", 0),
                    min_trajectories=int(options.get("min_trajectories", 1)),
                    target_standard_error=options.get("target_standard_error"),
                    memory_limit_bytes=options.get("memory_limit_bytes"),
                    estimated_memory_bytes=(
                        execution_plan.state_bytes
                        * min(
                            int(options.get("trajectories", 32)),
                            int(options.get("trajectory_batch_size", 32)),
                        )
                        * (
                            2
                            + max(
                                4 if noise_model.device_profile else 1,
                                max(
                                    (
                                        len(rule.channel.kraus)
                                        for rule in noise_model.rules
                                    ),
                                    default=1,
                                ),
                            )
                        )
                    ),
                    noise_model_identity=noise_model.identity,
                ),
            )
    elif mode == "mps_trajectory":
        from .executors.mps.noisy import run_lowered_noisy_mps_trajectory

        mps_options = dict(options)
        mps_options.pop("memory_limit_bytes", None)
        mps_options.pop("world_size", None)
        mps_options.pop("world_sz", None)
        mps_options.pop("coupling_map", None)
        mps_options.pop("optimize", None)
        source = execution_ir if coupling_map is not None else circuit_or_ir
        lowered = (
            execution_ir
            if provided_execution_plan is not None
            else lower_noise_model(execution_ir, noise_model)
        )
        result = run_lowered_noisy_mps_trajectory(
            lowered,
            source=source,
            **mps_options,
        )
        if provided_execution_plan is not None:
            execution_plan = provided_execution_plan
        else:
            execution_plan = build_plan(
                execution_ir,
                noise_model=noise_model,
                state_mode="mps",
                **plan_options,
            )
            execution_plan = replace(
                execution_plan,
                noisy_execution_plan=build_noisy_execution_plan(
                    execution_plan,
                    representation="mps",
                    evolution="quantum_trajectory",
                    trajectories=1,
                    seed=options.get("seed"),
                    cutoff=float(options.get("cutoff", 0.0)),
                    memory_limit_bytes=options.get("memory_limit_bytes"),
                    noise_model_identity=getattr(noise_model, "identity", None),
                ),
            )
    elif mode == "noisy_mps":
        from .executors.mps.noisy import run_lowered_noisy_mps

        mps_options = dict(options)
        mps_options.pop("memory_limit_bytes", None)
        mps_options["world_size"] = int(
            mps_options.pop(
                "world_sz",
                mps_options.get("world_size", 1),
            )
        )
        if mps_options["world_size"] > 1 and "rank" not in mps_options:
            raise NotImplementedError(
                "public noisy_mps execution does not perform distributed statistics "
                "reduction; pass an explicit rank for rank-local execution and "
                "merge_noisy_mps_results, or use noisy_statevector"
            )
        mps_options.pop("coupling_map", None)
        mps_options.pop("optimize", None)
        source = execution_ir if coupling_map is not None else circuit_or_ir
        lowered = (
            execution_ir
            if provided_execution_plan is not None
            else lower_noise_model(execution_ir, noise_model)
        )
        result = run_lowered_noisy_mps(
            lowered,
            source=source,
            noise_model=noise_model,
            **mps_options,
        )
        if provided_execution_plan is not None:
            execution_plan = provided_execution_plan
        else:
            execution_plan = build_plan(
                execution_ir,
                noise_model=noise_model,
                state_mode="mps",
                **plan_options,
            )
            execution_plan = replace(
                execution_plan,
                noisy_execution_plan=build_noisy_execution_plan(
                    execution_plan,
                    representation="mps",
                    evolution="quantum_trajectory",
                    trajectories=int(options.get("trajectories", 32)),
                    seed=options.get("seed"),
                    min_trajectories=int(options.get("min_trajectories", 1)),
                    target_standard_error=options.get("target_standard_error"),
                    cutoff=float(options.get("cutoff", 0.0)),
                    memory_limit_bytes=options.get("memory_limit_bytes"),
                    noise_model_identity=getattr(noise_model, "identity", None),
                ),
            )
    else:
        raise ValueError(
            "mode must be 'auto', 'statevector', 'distributed_statevector', 'density_matrix', 'mps', 'adaptive_mps', 'distributed_mps', 'jax_sharded_mps', 'tensor_network', 'distributed_tensor_network', 'jax_sharded_tensor_network', 'jax_sharded_tn', 'distributed_tn', 'tn', 'noisy_statevector', 'mps_trajectory', or 'noisy_mps'."
        )

    if return_plan:
        return result, execution_plan
    return result


def run(
    program_or_plan: Circuit | CircuitIR | ExecutionPlan,
    *,
    options: ExecutionOptions | None = None,
    measurements: Sequence[MeasurementNode] | None = None,
    noise_model: NoiseModel | None = None,
) -> ExecutionResult:
    """Plan and execute a program, or execute one validated plan exactly."""

    from .plan_execution import execute_plan

    if isinstance(program_or_plan, ExecutionPlan):
        if options is not None:
            raise TypeError("options must be None when executing an ExecutionPlan")
        if measurements is not None:
            raise TypeError("measurements must be None when executing an ExecutionPlan")
        if noise_model is not None:
            raise TypeError("noise_model must be None when executing an ExecutionPlan")
        return execute_plan(program_or_plan)
    from .planner import plan

    return execute_plan(
        plan(
            program_or_plan,
            options=options,
            measurements=measurements,
            noise_model=noise_model,
        )
    )


def run_advanced(
    program: Any,
    *,
    mode: str = "auto",
    noise_model: Any | None = None,
    measurements: Sequence[MeasurementNode] | None = None,
    **options: Any,
) -> ExecutionResult:
    """Experimental normalized execution with backend-specific controls."""

    source_ir = ensure_circuit_ir(program)
    requests = (
        tuple(source_ir.measurements) if measurements is None else tuple(measurements)
    )
    if any(not isinstance(request, MeasurementNode) for request in requests):
        raise TypeError("measurements must contain MeasurementNode instances")
    from .measurements import validate_measurements

    validate_measurements(requests, n_wires=source_ir.n_wires)
    output, execution_plan = run_native(
        program,
        noise_model=noise_model,
        mode=mode,
        return_plan=True,
        **options,
    )
    aliases = {
        "distributed": "distributed_statevector",
        "tn": "tensor_network",
        "distributed_tn": "distributed_tensor_network",
        "jax_sharded_tn": "jax_sharded_tensor_network",
    }
    selected_mode = aliases.get(mode, mode)
    if selected_mode == "auto":
        noisy_plan = execution_plan.noisy_execution_plan
        selected_mode = (
            "noisy_statevector"
            if noisy_plan is not None
            and noisy_plan.representation == "statevector"
            and noisy_plan.evolution == "quantum_trajectory"
            else execution_plan.state_mode
        )
    return _normalize_execution_output(
        output,
        execution_plan,
        source_ir=source_ir,
        requests=requests,
        mode=selected_mode,
        noise_model=noise_model,
    )


def _normalize_execution_output(
    output: Any,
    execution_plan: ExecutionPlan,
    *,
    source_ir: CircuitIR,
    requests: Sequence[MeasurementNode],
    mode: str,
    noise_model: NoiseModel | None = None,
) -> ExecutionResult:
    from .measurements import execute_measurements
    from .result import normalize_execution_result

    result = normalize_execution_result(output, mode=mode, plan=execution_plan)
    if (
        mode == "statevector"
        and execution_plan.world_size == 1
        and isinstance(result.state, torch.Tensor)
    ):
        from ..compute import get_platform_runtime

        actual_device = result.state.device
        platform = get_platform_runtime(actual_device.type)
        requested_device = str(
            (execution_plan.runtime_config or {}).get("device", actual_device)
        )
        if requested_device.split(":", 1)[0] != actual_device.type:
            raise ExecutionError(
                "local statevector execution returned data on "
                f"{actual_device} instead of planned device {requested_device}"
            )
        native_output = result.__dict__.get("_native_output")
        result = replace(
            result,
            runtime={
                **dict(result.runtime),
                "execution_path": "local_statevector",
                "simulation_engine": "pytorch_statevector",
                "platform_provider": platform.identity().provider,
                "device": str(actual_device),
                "distribution_semantics": "single_device_fast_path",
            },
            provenance={
                **dict(result.provenance),
                "requested_device": requested_device,
                "selected_device": str(actual_device),
                "cpu_fallback_used": False,
            },
        )
        if native_output is not None:
            object.__setattr__(result, "_native_output", native_output)
    measurement_results = execute_measurements(
        output,
        requests,
        n_wires=source_ir.n_wires,
        noise_model=noise_model,
    )
    first_samples = next(
        (
            item.value
            for item in measurement_results
            if item.kind == "sample" and isinstance(item.value, torch.Tensor)
        ),
        result.samples,
    )
    normalized = replace(
        result,
        samples=first_samples,
        measurements=measurement_results,
    )
    native = result.__dict__.get("_native_output")
    if native is not None:
        object.__setattr__(normalized, "_native_output", native)
    return normalized


__all__ = [
    "ExecutionPlan",
    "run",
    "run_distributed",
    "run_native",
]

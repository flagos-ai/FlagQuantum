"""Canonical execution services for FlagQuantum circuits and devices."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

import torch

from ..compilation.compiler import compile_for_backend
from ..compilation.planner import ExecutionPlan, select_execution_mode
from ..compilation.planner import plan as build_plan
from ..core.ir import CircuitIR, Instruction, ensure_circuit_ir
from ..core.parameters import value_to_tensor
from ..core.runtime_config import (
    RuntimeConfig,
    get_runtime_config,
    runtime_config,
)
from ..devices import DistributedQuantumDevice
from ..measurement import measure_allZ
from ..ops import functional
from ..runtime.backend_registry import resolve_device
from ..runtime.distributed.backend_policy import (
    DistributedBackendPolicy,
    resolve_distributed_backend_policy,
)
from .backends.jax import (
    plan_jax_distributed_quantum_backend,
    run_jax_sharded_mps,
    run_jax_sharded_tensor_network,
)
from .backends.statevector import (
    plan_distributed_statevector,
    simulate_distributed_statevector_local,
)

_GATE_ALIASES = {
    "cnot": "cx",
    "ccnot": "ccx",
    "toffoli": "ccx",
    "fredkin": "cswap",
    "sd": "sdg",
    "td": "tdg",
    "p": "phase",
}

_PARAM_ALIASES = {
    "rx": ("theta",),
    "ry": ("theta",),
    "rz": ("theta",),
    "phase": ("theta",),
    "p": ("theta",),
    "u1": ("theta",),
    "u2": ("phi", "lbd"),
    "u3": ("theta", "phi", "lbd"),
    "crx": ("theta",),
    "cry": ("theta",),
    "crz": ("theta",),
    "cphase": ("theta",),
    "rxx": ("theta",),
    "ryy": ("theta",),
    "rzz": ("theta",),
}


def _as_parameter_tensor(instruction: Instruction) -> torch.Tensor | None:
    names = _PARAM_ALIASES.get(instruction.name)
    if not names:
        return None
    values = [instruction.params.get(name) for name in names]
    if any(value is None for value in values):
        return None
    tensors = [value_to_tensor(value) for value in values]
    params = torch.stack(
        [tensor.reshape(()) if tensor.ndim == 0 else tensor for tensor in tensors],
        dim=-1,
    )
    return params.to(dtype=torch.float32)


def _matrix_to_torch(matrix: Any, device: torch.device | str) -> torch.Tensor:
    tensor = getattr(matrix, "tensor", matrix)
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    tensor = torch.as_tensor(tensor, dtype=torch.complex64, device=device).reshape(-1)
    width = int(tensor.numel() ** 0.5)
    return tensor.reshape(width, width)


def _statevector_all_z_expectation(state: torch.Tensor, n_wires: int) -> torch.Tensor:
    probs = torch.abs(state) ** 2
    probs = probs.reshape((probs.shape[0],) + (2,) * n_wires)
    values = []
    for wire in range(n_wires):
        axes = tuple(axis for axis in range(1, n_wires + 1) if axis != wire + 1)
        marginal = probs.sum(dim=axes) if axes else probs
        values.append(marginal[:, 0] - marginal[:, 1])
    return torch.stack(values, dim=-1)


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


class DistributedExecutor:
    """Run unified circuit IR on the native distributed statevector device."""

    def __init__(self, device: DistributedQuantumDevice):
        self.device = device

    def apply(self, instruction: Instruction) -> None:
        if instruction.metadata.get("is_channel"):
            raise NotImplementedError(
                "Distributed execution currently supports unitary instructions."
            )
        if instruction.metadata.get("mpo") or instruction.metadata.get("diagonal"):
            raise NotImplementedError(
                "MPO and diagonal tensor-network instructions require tensor-network execution."
            )

        name = _GATE_ALIASES.get(instruction.name, instruction.name)
        params = _as_parameter_tensor(instruction)
        if hasattr(self.device, name):
            getattr(self.device, name)(wires=list(instruction.wires), params=params)
            return

        if instruction.matrix is None:
            raise KeyError(f"No native distributed implementation for gate {name!r}.")
        matrix = _matrix_to_torch(instruction.matrix, self.device.device)
        functional.gate(
            matrix, self.device, wires=list(instruction.wires), params=params
        )

    def run(self, ir: CircuitIR) -> DistributedQuantumDevice:
        for instruction in ir:
            self.apply(instruction)
        return self.device


def run_distributed(
    ir: CircuitIR,
    *,
    device: DistributedQuantumDevice | torch.device | str | None = None,
    measure: bool = False,
    optimize: bool = True,
    return_plan: bool = False,
    coupling_map: Any | None = None,
    **device_options: Any,
) -> Any:
    """Execute a FlagQuantum IR on a native distributed device."""

    execution_ir = compile_for_backend(
        ir,
        coupling_map=coupling_map,
        optimize=optimize,
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
    execution_plan = build_plan(
        execution_ir,
        bsz=batch_size,
        world_size=world_size,
        optimize=False,
    )
    statevector_plan = plan_distributed_statevector(
        execution_ir,
        bsz=batch_size,
        world_size=world_size,
        local_world_size=local_world_size,
    )
    jax_distributed_plan = plan_jax_distributed_quantum_backend(
        execution_ir,
        mode="statevector",
        world_size=world_size,
        local_world_size=local_world_size,
        bsz=batch_size,
        distributed_backend_policy=backend_policy,
    ).summary()

    if (
        world_size > 1
        and backend_policy.profile == "development"
        and backend_policy.torch_backend == "local_tensor"
    ):
        local_result = simulate_distributed_statevector_local(
            execution_ir,
            world_size=world_size,
            bsz=batch_size,
            device=device,
            backend_policy=backend_policy,
            jax_distributed_plan=jax_distributed_plan,
        )
        result = (
            _statevector_all_z_expectation(local_result.state, execution_ir.n_wires)
            if measure
            else local_result
        )
        if return_plan:
            return result, execution_plan
        return result

    if isinstance(device, DistributedQuantumDevice):
        qdev = device
    else:
        if device is not None:
            device_options["device"] = str(device)
        qdev = DistributedQuantumDevice(n_wires=execution_ir.n_wires, **device_options)
    DistributedExecutor(qdev).run(execution_ir)
    qdev.execution_ir = execution_ir
    qdev.execution_plan = execution_plan
    qdev.distributed_statevector_plan = statevector_plan
    qdev.distributed_statevector_summary = statevector_plan.summary()
    qdev.distributed_backend_policy = backend_policy
    qdev.distributed_statevector_summary["distributed_backend_policy"] = (
        backend_policy.summary()
    )
    qdev.distributed_statevector_summary["jax_distributed_plan"] = jax_distributed_plan

    result = measure_allZ(qdev) if measure else qdev
    if return_plan:
        return result, execution_plan
    return result


def run_native(
    circuit_or_ir: Any,
    *,
    noise_model: Any | None = None,
    mode: str = "auto",
    return_plan: bool = False,
    **options: Any,
) -> Any:
    """Run FlagQuantum IR through native local execution paths."""

    config_active = bool(options.pop("_runtime_config_active", False))
    selected_config = options.pop("config", None)
    if not config_active:
        if selected_config is None:
            metadata = getattr(ensure_circuit_ir(circuit_or_ir), "metadata", {})
            manifest = metadata.get("runtime_config")
            selected_config = (
                RuntimeConfig.from_manifest(manifest)
                if manifest is not None
                else get_runtime_config()
            )
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
        from .operator_backends import operator_backend as operator_backend_context

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

    if mode == "distributed":
        mode = "distributed_statevector"
    if mode == "tn":
        mode = "tensor_network"
    if mode == "distributed_tn":
        mode = "distributed_tensor_network"
    if mode == "jax_sharded_tn":
        mode = "jax_sharded_tensor_network"
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
    execution_ir = compile_for_backend(
        ir,
        coupling_map=coupling_map,
        optimize=optimize if coupling_map is not None else False,
    )
    if mode == "auto":
        mode = select_execution_mode(
            execution_ir,
            noise_model=noise_model,
            **plan_options,
        )

    if mode == "distributed_statevector":
        if noise_model is not None:
            raise ValueError("Noise models require density_matrix mode.")
        distributed_options = dict(options)
        distributed_options.pop("dtype", None)
        distributed_options.pop("max_bond", None)
        distributed_options.pop("cutoff", None)
        distributed_options.pop("memory_limit_bytes", None)
        distributed_options.pop("optimize", None)
        return run_distributed(
            execution_ir,
            optimize=False,
            return_plan=return_plan,
            **distributed_options,
        )

    if mode == "statevector":
        if noise_model is not None:
            raise ValueError("Noise models require density_matrix mode.")
        if coupling_map is None and hasattr(circuit_or_ir, "state"):
            result = circuit_or_ir.state()
        else:
            from ..circuit import Circuit

            circuit_options = dict(options)
            circuit_options.pop("coupling_map", None)
            circuit_options.pop("optimize", None)
            circuit_options.pop("memory_limit_bytes", None)
            circuit_options.pop("world_size", None)
            circuit_options.pop("world_sz", None)
            result = Circuit.from_ir(execution_ir, **circuit_options).state()
        execution_plan = build_plan(
            execution_ir, state_mode="statevector", **plan_options
        )
    elif mode == "density_matrix":
        from ..simulation.noise import density_matrix_from_ir, lower_noise_model

        lowered = lower_noise_model(execution_ir, noise_model)
        density_options = dict(options)
        density_options.pop("coupling_map", None)
        density_options.pop("optimize", None)
        result = density_matrix_from_ir(lowered, **density_options)
        execution_plan = build_plan(
            lowered,
            state_mode="density_matrix",
            **plan_options,
        )
    elif mode == "mps":
        if noise_model is not None:
            raise ValueError("Noise models require density_matrix mode.")
        from ..simulation.mps import run_mps

        mps_options = dict(options)
        mps_options.pop("memory_limit_bytes", None)
        mps_options.pop("world_size", None)
        mps_options.pop("world_sz", None)
        mps_options.pop("coupling_map", None)
        mps_options.pop("optimize", None)
        result = run_mps(
            execution_ir if coupling_map is not None else circuit_or_ir, **mps_options
        )
        execution_plan = build_plan(
            execution_ir,
            noise_model=noise_model,
            state_mode="mps",
            **plan_options,
        )
    elif mode == "adaptive_mps":
        if noise_model is not None:
            raise ValueError("Noise models require density_matrix mode.")
        from ..simulation.mps import run_mps_adaptive

        mps_options = dict(options)
        mps_options.pop("memory_limit_bytes", None)
        mps_options.pop("world_size", None)
        mps_options.pop("world_sz", None)
        mps_options.pop("coupling_map", None)
        mps_options.pop("optimize", None)
        result = run_mps_adaptive(
            execution_ir if coupling_map is not None else circuit_or_ir, **mps_options
        )
        execution_plan = build_plan(
            execution_ir,
            noise_model=noise_model,
            state_mode="mps",
            **plan_options,
        )
    elif mode == "distributed_mps":
        if noise_model is not None:
            raise ValueError("Noise models require density_matrix mode.")
        from .distributed import run_distributed_mps

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
        execution_plan = build_plan(
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
        execution_plan = build_plan(
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
        from ..simulation.tensor import run_tensor_network

        tn_options = dict(options)
        tn_options.pop("memory_limit_bytes", None)
        tn_options.pop("world_size", None)
        tn_options.pop("world_sz", None)
        tn_options.pop("coupling_map", None)
        tn_options.pop("optimize", None)
        result = run_tensor_network(
            execution_ir if coupling_map is not None else circuit_or_ir, **tn_options
        )
        execution_plan = build_plan(
            execution_ir,
            noise_model=noise_model,
            state_mode="tensor_network",
            **plan_options,
        )
    elif mode == "distributed_tensor_network":
        if noise_model is not None:
            raise ValueError("Noise models require density_matrix mode.")
        from .distributed import run_distributed_tensor_network

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
        execution_plan = build_plan(
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
        execution_plan = build_plan(
            execution_ir,
            noise_model=noise_model,
            state_mode="tensor_network",
            world_size=world_size,
            **{
                key: value for key, value in plan_options.items() if key != "world_size"
            },
        )
    elif mode == "mps_trajectory":
        from ..simulation.mps import run_noisy_mps_trajectory

        mps_options = dict(options)
        mps_options.pop("memory_limit_bytes", None)
        mps_options.pop("world_size", None)
        mps_options.pop("world_sz", None)
        mps_options.pop("coupling_map", None)
        mps_options.pop("optimize", None)
        result = run_noisy_mps_trajectory(
            execution_ir if coupling_map is not None else circuit_or_ir,
            noise_model,
            **mps_options,
        )
        execution_plan = build_plan(
            execution_ir,
            noise_model=noise_model,
            state_mode="mps",
            **plan_options,
        )
    elif mode == "noisy_mps":
        from ..simulation.mps import run_noisy_mps

        mps_options = dict(options)
        mps_options.pop("memory_limit_bytes", None)
        mps_options.pop("world_size", None)
        mps_options.pop("world_sz", None)
        mps_options.pop("coupling_map", None)
        mps_options.pop("optimize", None)
        result = run_noisy_mps(
            execution_ir if coupling_map is not None else circuit_or_ir,
            noise_model,
            **mps_options,
        )
        execution_plan = build_plan(
            execution_ir,
            noise_model=noise_model,
            state_mode="mps",
            **plan_options,
        )
    else:
        raise ValueError(
            "mode must be 'auto', 'statevector', 'distributed_statevector', 'density_matrix', 'mps', 'adaptive_mps', 'distributed_mps', 'jax_sharded_mps', 'tensor_network', 'distributed_tensor_network', 'jax_sharded_tensor_network', 'jax_sharded_tn', 'distributed_tn', 'tn', 'mps_trajectory', or 'noisy_mps'."
        )

    if return_plan:
        return result, execution_plan
    return result


def run(
    circuit_or_ir: Any,
    *,
    mode: str = "auto",
    noise_model: Any | None = None,
    **options: Any,
) -> Any:
    """Execute a circuit and always return the stable ``ExecutionResult``.

    This is the recommended public execution entry point. Backend-specific
    runners remain available for advanced use cases that need their native
    result objects.
    """

    if "return_plan" in options:
        raise TypeError(
            "fq.run() always includes the execution plan in result.plan; "
            "remove the return_plan option"
        )
    if "result" in options:
        raise TypeError(
            "fq.run() always returns ExecutionResult; remove the result option"
        )

    output, execution_plan = run_native(
        circuit_or_ir,
        noise_model=noise_model,
        mode=mode,
        return_plan=True,
        **options,
    )
    from .result import normalize_execution_result

    aliases = {
        "distributed": "distributed_statevector",
        "tn": "tensor_network",
        "distributed_tn": "distributed_tensor_network",
        "jax_sharded_tn": "jax_sharded_tensor_network",
    }
    selected_mode = aliases.get(mode, mode)
    if selected_mode == "auto":
        selected_mode = execution_plan.state_mode
    return normalize_execution_result(
        output,
        mode=selected_mode,
        plan=execution_plan,
    )


__all__ = [
    "DistributedExecutor",
    "ExecutionPlan",
    "run",
    "run_distributed",
    "run_native",
]

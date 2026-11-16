"""Canonical versioned training checkpoint, precision, seed, and debug contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import torch

from ..compute import get_platform_runtime
from ..core.ir import IR_VERSION, ensure_circuit_ir
from ..errors import ExecutionError

_LEGACY_TRAINING_STATE_VERSION = "flagquantum.training_state.v1"
_CAPTURED_TRAINING_STATE_VERSION = "flagquantum.training_state.v2"
TRAINING_STATE_VERSION = "flagquantum.training_state.v3"
TRAINING_RESTORE_SCHEMA = "flagquantum.training_checkpoint_restore"
TRAINING_RESTORE_VERSION = "1.0"


class TrainingStateError(ExecutionError):
    """A checkpoint or numerical correctness contract was violated."""


class TopologyMismatchError(TrainingStateError):
    """Checkpoint sharding topology differs from the active topology."""


class PrecisionPolicyError(TrainingStateError):
    """A precision request would silently lose trainable information."""


class NonFiniteTrainingError(TrainingStateError):
    """NaN or Inf was detected at a named training boundary."""


@dataclass(frozen=True)
class PrecisionPolicy:
    complex_dtype: str = "complex64"
    parameter_dtype: str = "float32"
    allow_parameter_downcast: bool = False
    atol: float = 1e-5
    rtol: float = 1e-5

    def __post_init__(self) -> None:
        if self.complex_dtype not in {"complex64", "complex128"}:
            raise PrecisionPolicyError("complex dtype must be complex64 or complex128")
        if self.parameter_dtype not in {"float32", "float64"}:
            raise PrecisionPolicyError("parameter dtype must be float32 or float64")
        expected_real = "float64" if self.complex_dtype == "complex128" else "float32"
        if self.parameter_dtype != expected_real:
            raise PrecisionPolicyError(
                f"{self.complex_dtype} requires parameter_dtype={expected_real!r}"
            )
        if any(
            not math.isfinite(value) or value <= 0 for value in (self.atol, self.rtol)
        ):
            raise PrecisionPolicyError(
                "precision tolerances must be positive and finite"
            )

    @property
    def torch_parameter_dtype(self) -> torch.dtype:
        return torch.float64 if self.parameter_dtype == "float64" else torch.float32

    def apply(self, module: torch.nn.Module) -> torch.nn.Module:
        target = self.torch_parameter_dtype
        for name, parameter in module.named_parameters():
            if parameter.is_complex():
                raise PrecisionPolicyError(f"trainable parameter {name} must be real")
            if (
                parameter.element_size() > torch.empty((), dtype=target).element_size()
                and not self.allow_parameter_downcast
            ):
                raise PrecisionPolicyError(
                    f"precision policy would downcast trainable parameter {name}; "
                    "set allow_parameter_downcast=True explicitly"
                )
        return module.to(dtype=target)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class SeedContract:
    seed: int
    sampling_seed: int
    trajectory_seed: int
    jax_key_words: tuple[int, int]

    def torch_generator(self, device: str = "cpu") -> torch.Generator:
        generator = torch.Generator(device=device)
        generator.manual_seed(self.sampling_seed)
        return generator


class TrainingCheckpointRestore(TypedDict):
    """Versioned metadata returned after a successful checkpoint restore."""

    schema: str
    version: str
    step: int
    seed: SeedContract
    precision: PrecisionPolicy
    runtime_plan: object | None
    topology: Mapping[str, object]
    ir_hash: str


def seed_everything(
    seed: int, *, deterministic_algorithms: bool = True
) -> SeedContract:
    """Seed Python/Torch and describe explicit JAX/sampling/trajectory streams."""

    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    torch.manual_seed(seed)
    cuda_platform = get_platform_runtime("cuda")
    if cuda_platform.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return SeedContract(
        seed=seed,
        sampling_seed=(seed ^ 0x5A17) & 0x7FFF_FFFF,
        trajectory_seed=(seed ^ 0x71EC70) & 0x7FFF_FFFF,
        jax_key_words=(0, seed & 0xFFFF_FFFF),
    )


def _rng_state(
    seed: SeedContract, module: torch.nn.Module | None = None
) -> dict[str, Any]:
    cuda_platform = get_platform_runtime("cuda")
    cuda_device = torch.cuda.current_device() if cuda_platform.is_available() else None
    # A provider may expose a CUDA compatibility API while keeping tensors on
    # its own logical device. Preserve the native CUDA slot only for CUDA.
    if cuda_device is not None and torch.device("cuda", cuda_device).type != "cuda":
        cuda_device = None
    result: dict[str, Any] = {
        "contract": seed.__dict__,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            cuda_platform.rng_state(torch.device("cuda", cuda_device))
            if cuda_device is not None
            else None
        ),
        "cuda_device": cuda_device,
        "python": random.getstate(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    if module is not None:
        devices = {
            tensor.device
            for tensor in (*module.parameters(), *module.buffers())
            if tensor.device.type not in {"cpu", "cuda"}
        }
        if devices:
            # Optional provider state lives inside the opaque RNG payload; the
            # checkpoint metadata and legacy CPU/CUDA payload stay unchanged.
            result["platform_rng"] = {
                str(device): get_platform_runtime(device.type).rng_state(device)
                for device in sorted(devices, key=str)
            }
    return result


def _restore_rng(payload: Mapping[str, Any]) -> SeedContract:
    contract = SeedContract(**dict(payload["contract"]))
    random.setstate(tuple(payload["python"]))
    torch.set_rng_state(payload["torch_cpu"])
    cuda_state = payload.get("torch_cuda")
    cuda_platform = get_platform_runtime("cuda")
    if cuda_state is not None and cuda_platform.is_available():
        cuda_platform.restore_rng_state(
            torch.device("cuda", torch.cuda.current_device()), cuda_state
        )
    for name, state in payload.get("platform_rng", {}).items():
        device_type = str(name).split(":", 1)[0]
        platform = get_platform_runtime(device_type)
        if not platform.is_available():
            raise TrainingStateError(
                f"checkpoint RNG platform {device_type!r} is unavailable"
            )
        platform.restore_rng_state(torch.device(name), state)
    torch.use_deterministic_algorithms(bool(payload["deterministic_algorithms"]))
    return contract


def _topology(module: torch.nn.Module) -> dict[str, Any]:
    import torch.distributed as dist

    owner = _quantum_owner(module)
    plan = getattr(owner, "_parallel_plan", None)
    initialized = dist.is_initialized()
    actual_world_size = dist.get_world_size() if initialized else 1
    actual_rank = dist.get_rank() if initialized else 0
    backend = str(dist.get_backend()) if initialized else "single_process"
    state_group = getattr(owner, "_state_process_group", None)
    distributed_state = (
        getattr(getattr(owner, "policy", None), "mode", None) == "statevector"
        and actual_world_size > 1
    )
    if initialized and (state_group is not None or distributed_state):
        state_world_size = dist.get_world_size(state_group)
        state_rank = dist.get_rank(state_group)
    else:
        state_world_size = 1
        state_rank = 0
    if plan is None:
        return {
            "world_size": actual_world_size,
            "rank": actual_rank,
            "backend": backend,
            "data_parallel_size": 1,
            "state_parallel_size": state_world_size,
            "state_rank": state_rank,
        }
    if int(plan.world_size) != actual_world_size:
        raise TopologyMismatchError(
            "parallel plan world_size does not match the active process group"
        )
    if int(plan.state_parallel_size) != state_world_size:
        raise TopologyMismatchError(
            "parallel plan state_parallel_size does not match the state process group"
        )
    return {
        "world_size": plan.world_size,
        "rank": actual_rank,
        "backend": backend,
        "data_parallel_size": plan.data_parallel_size,
        "state_parallel_size": plan.state_parallel_size,
        "state_rank": state_rank,
        "model_parallel_size": plan.model_parallel_size,
        "data_groups": plan.data_groups,
        "state_groups": plan.state_groups,
    }


def _workload_signature(ir: Any) -> str:
    payload = ir.to_dict()
    for source, encoded_instruction in zip(
        ir.instructions, payload["instructions"], strict=True
    ):
        for name, value in source.params.items():
            if not isinstance(value, torch.Tensor) or not value.requires_grad:
                continue
            encoded = encoded_instruction["params"][name]["$tensor"]
            encoded_instruction["params"][name] = {
                "$trainable_tensor": {
                    "dtype": encoded["dtype"],
                    "shape": encoded["shape"],
                }
            }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _current_workload_signature(owner: torch.nn.Module) -> str:
    ir = getattr(owner, "_last_ir", None)
    if ir is None:
        builder = getattr(owner, "_build", None)
        owned_parameters = getattr(owner, "_owned_parameters", None)
        parameters = (
            owned_parameters()
            if callable(owned_parameters)
            else getattr(owner, "parameters_tensor", None)
        )
        if not callable(builder) or parameters is None:
            raise TrainingStateError(
                "checkpoint restore requires an executable fq.Module workload"
            )
        try:
            ir = ensure_circuit_ir(builder(None, parameters))
        except Exception as error:
            raise TrainingStateError(
                "execute the target module with representative inputs before restore"
            ) from error
    return _workload_signature(ir)


def _quantum_owner(module: torch.nn.Module) -> torch.nn.Module:
    if hasattr(module, "_last_ir"):
        return module
    owners = [item for item in module.modules() if hasattr(item, "_last_ir")]
    if len(owners) != 1:
        raise TrainingStateError(
            "checkpoint requires exactly one fq.Module quantum owner"
        )
    return owners[0]


def _optimizer_contract(
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, str] | None:
    if optimizer is None:
        return None
    cls = type(optimizer)
    return {"module": cls.__module__, "qualname": cls.__qualname__}


def _captured_workload_signature(
    owner: torch.nn.Module, *, prepare: bool = False, parameter_training: bool = False
) -> str | None:
    signature = getattr(owner, "_checkpoint_signature", None)
    if not callable(signature):
        return None
    try:
        result = (
            signature(prepare=prepare, parameter_training=True)
            if parameter_training
            else signature(prepare=prepare)
        )
    except Exception as error:
        message = (
            "checkpoint program is stale or uncertifiable; execute with representative "
            "inputs after rebuilding the source program"
            if parameter_training
            else "execute the target module with representative inputs before restore"
        )
        raise TrainingStateError(message) from error
    return result if isinstance(result, str) else None


def _parameter_training(
    module: torch.nn.Module, optimizer: torch.optim.Optimizer | None
) -> dict[str, Any]:
    parameters = dict(module.named_parameters())
    names = {id(value): name for name, value in parameters.items()}
    if len(list(module.named_parameters(remove_duplicate=False))) != len(names):
        raise TrainingStateError("checkpoint refuses registered parameter aliases")
    groups = None
    if optimizer is not None:
        groups = []
        seen: set[int] = set()
        for group in optimizer.param_groups:
            current = []
            for parameter in group["params"]:
                identity = id(parameter)
                if identity not in names or identity in seen:
                    raise TrainingStateError(
                        "checkpoint optimizer parameters must be unique and registered"
                    )
                current.append(names[identity])
                seen.add(identity)
            groups.append(current)
    return {
        "requires_grad": {
            name: value.requires_grad for name, value in parameters.items()
        },
        "optimizer_parameters": groups,
    }


def _has_custom_checkpoint_methods(
    module: torch.nn.Module, names: tuple[str, ...]
) -> bool:
    from .module import Module

    expected_type = Module if isinstance(module, Module) else torch.nn.Module
    return any(
        getattr(type(module), name) is not getattr(expected_type, name)
        or name in module.__dict__
        for name in names
    )


def _validate_checkpoint_target(
    module: torch.nn.Module, optimizer: torch.optim.Optimizer | None
) -> None:
    # Arbitrary hooks, custom storage and custom loading methods cannot promise
    # rollback. Refuse them before state_dict or capture can execute user code.
    for child in module.modules():
        for name in (
            "_state_dict_pre_hooks",
            "_state_dict_hooks",
            "_load_state_dict_pre_hooks",
            "_load_state_dict_post_hooks",
        ):
            if getattr(child, name):
                raise TrainingStateError(
                    "checkpoint restore refuses custom state hooks"
                )
        if _has_custom_checkpoint_methods(
            child,
            (
                "state_dict",
                "load_state_dict",
                "_save_to_state_dict",
                "_load_from_state_dict",
                "_apply",
                "get_extra_state",
                "set_extra_state",
            ),
        ):
            raise TrainingStateError(
                "checkpoint restore refuses custom state loading methods"
            )
    storage_owners: set[int] = set()
    tensors = (*module.parameters(), *module.buffers())
    for tensor in tensors:
        if (
            type(tensor) not in (torch.Tensor, torch.nn.Parameter)
            or tensor.layout != torch.strided
            or tensor.device.type == "meta"
            or tensor.is_inference()
            or torch._debug_has_internal_overlap(tensor) != 0
        ):
            raise TrainingStateError(
                "checkpoint restore requires ordinary strided tensor storage"
            )
        storage = tensor.untyped_storage()._cdata
        if storage in storage_owners:
            raise TrainingStateError("checkpoint restore refuses shared tensor storage")
        storage_owners.add(storage)
    if optimizer is not None:
        if not type(optimizer).__module__.startswith("torch.optim."):
            raise TrainingStateError(
                "checkpoint restore refuses custom optimizer implementations"
            )
        if (
            optimizer._optimizer_load_state_dict_pre_hooks
            or optimizer._optimizer_load_state_dict_post_hooks
            or optimizer._optimizer_state_dict_pre_hooks
            or optimizer._optimizer_state_dict_post_hooks
        ):
            raise TrainingStateError("checkpoint restore refuses optimizer load hooks")


def _validate_parameter_training(
    payload: Mapping[str, Any],
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> None:
    training = payload.get("parameter_training")
    expected = _parameter_training(module, optimizer)
    if not isinstance(training, dict) or set(training) != set(expected):
        raise TrainingStateError("checkpoint parameter training fields mismatch")
    flags = training["requires_grad"]
    if (
        not isinstance(flags, dict)
        or set(flags) != set(expected["requires_grad"])
        or any(type(flag) is not bool for flag in flags.values())
    ):
        raise TrainingStateError(
            "checkpoint flags must exactly cover registered parameters"
        )
    groups = training["optimizer_parameters"]
    if groups is not None and (
        not isinstance(groups, list)
        or any(
            not isinstance(group, list) or any(type(name) is not str for name in group)
            for group in groups
        )
    ):
        raise TrainingStateError(
            "checkpoint optimizer bindings must be ordered name lists"
        )
    if groups != expected["optimizer_parameters"]:
        raise TrainingStateError("checkpoint optimizer binding names or order mismatch")

    if optimizer is not None:
        if not isinstance(groups, list):
            raise TrainingStateError("checkpoint optimizer bindings are missing")
        state = payload.get("optimizer_state")
        if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
            raise TrainingStateError("checkpoint optimizer state is malformed")
        saved_groups = state["param_groups"]
        if not isinstance(saved_groups, list) or len(saved_groups) != len(groups):
            raise TrainingStateError("checkpoint optimizer group count mismatch")
        offset = 0
        for saved_group, names in zip(saved_groups, groups, strict=True):
            if (
                not isinstance(saved_group, dict)
                or not isinstance(saved_group.get("params"), list)
                or any(type(index) is not int for index in saved_group["params"])
                or saved_group["params"] != list(range(offset, offset + len(names)))
            ):
                raise TrainingStateError("checkpoint optimizer state binding mismatch")
            offset += len(names)
        if not isinstance(state["state"], dict) or any(
            type(key) is not int or not 0 <= key < offset for key in state["state"]
        ):
            raise TrainingStateError("checkpoint optimizer state parameter mismatch")
    state = payload.get("module_state")
    current = module.state_dict()
    if not isinstance(state, Mapping) or set(state) != set(current):
        raise TrainingStateError("checkpoint module state names mismatch")
    for name, value in current.items():
        saved = state[name]
        if isinstance(value, torch.Tensor):
            if (
                type(saved) is not torch.Tensor
                or saved.shape != value.shape
                or saved.dtype != value.dtype
                or saved.layout != torch.strided
            ):
                raise TrainingStateError(
                    "checkpoint state shape, dtype or layout mismatch"
                )
        elif name.endswith("_extra_state"):
            if not isinstance(saved, Mapping) or set(saved) != {"policy"}:
                raise TrainingStateError("checkpoint module policy state is malformed")
            try:
                from .policy import RuntimePolicy

                RuntimePolicy.from_dict(saved["policy"])
            except (TypeError, ValueError, KeyError) as error:
                raise TrainingStateError(
                    "checkpoint module policy state is malformed"
                ) from error
    rng = payload.get("seed_state")
    try:
        if (
            not isinstance(rng, Mapping)
            or type(rng["deterministic_algorithms"]) is not bool
        ):
            raise ValueError("invalid RNG fields")
        contract = SeedContract(**dict(rng["contract"]))
        if (
            any(
                type(value) is not int or value < 0
                for value in (
                    contract.seed,
                    contract.sampling_seed,
                    contract.trajectory_seed,
                    *contract.jax_key_words,
                )
            )
            or len(contract.jax_key_words) != 2
        ):
            raise ValueError("invalid seed contract")
        random.Random().setstate(rng["python"])
        torch.Generator().set_state(rng["torch_cpu"])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise TrainingStateError("checkpoint RNG state is malformed") from error
    if type(payload.get("step")) is not int or payload["step"] < 0:
        raise TrainingStateError("checkpoint step must be a non-negative integer")


def save_training_checkpoint(
    path: str | Path,
    *,
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    seed: SeedContract,
    precision: PrecisionPolicy,
    runtime_plan: object | None = None,
    step: int = 0,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    owner = _quantum_owner(module)
    ir = getattr(owner, "_last_ir", None)
    if ir is None:
        raise TrainingStateError("execute the module once before checkpointing")
    plan_payload = (
        runtime_plan.to_dict() if hasattr(runtime_plan, "to_dict") else runtime_plan
    )
    workload_signature = _captured_workload_signature(owner, parameter_training=True)
    training = None
    custom_extra_state = any(
        _has_custom_checkpoint_methods(child, ("get_extra_state", "set_extra_state"))
        for child in module.modules()
    )
    # Historical composite models use extra state for deployment metadata. They
    # retain their versioned reader/writer; v3 cannot guarantee callback rollback.
    if workload_signature is not None and not custom_extra_state:
        version = TRAINING_STATE_VERSION
        training = _parameter_training(module, optimizer)
        _validate_checkpoint_target(module, optimizer)
    else:
        if any(not parameter.requires_grad for parameter in module.parameters()):
            raise TrainingStateError(
                "frozen parameters require a qualified v3 checkpoint workload"
            )
        workload_signature = _captured_workload_signature(owner)
        version = (
            _CAPTURED_TRAINING_STATE_VERSION
            if workload_signature is not None
            else _LEGACY_TRAINING_STATE_VERSION
        )
    if training is not None and (type(step) is not int or step < 0):
        raise TrainingStateError("checkpoint step must be a non-negative integer")
    payload = {
        "version": version,
        "step": int(step),
        "module_state": module.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "optimizer_contract": _optimizer_contract(optimizer),
        "runtime_policy": getattr(owner, "policy").to_dict(),
        "runtime_plan": plan_payload,
        "ir_version": getattr(ir, "version", IR_VERSION),
        "ir_hash": getattr(ir, "content_hash", ""),
        "workload_signature": workload_signature or _workload_signature(ir),
        "seed_state": _rng_state(seed, module),
        "precision": precision.to_dict(),
        "topology": _topology(module),
    }
    if training is not None:
        payload["parameter_training"] = training
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _load_training_checkpoint_payload(path: str | Path) -> Mapping[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TrainingStateError("training checkpoint must contain a mapping")
    return payload


def _validate_training_checkpoint(
    payload: Mapping[str, Any],
    *,
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    precision: PrecisionPolicy | None,
) -> tuple[torch.nn.Module, PrecisionPolicy]:
    required = {
        "version",
        "step",
        "module_state",
        "optimizer_state",
        "runtime_plan",
        "ir_version",
        "ir_hash",
        "workload_signature",
        "seed_state",
        "precision",
        "topology",
    }
    if payload.get("version") == TRAINING_STATE_VERSION:
        required.update({"parameter_training", "runtime_policy", "optimizer_contract"})
    missing = required.difference(payload)
    if missing:
        names = ", ".join(sorted(missing))
        raise TrainingStateError(f"training checkpoint is missing fields: {names}")
    version = payload.get("version")
    if version not in (
        _LEGACY_TRAINING_STATE_VERSION,
        _CAPTURED_TRAINING_STATE_VERSION,
        TRAINING_STATE_VERSION,
    ):
        raise TrainingStateError(
            f"unsupported training state {payload.get('version')!r}"
        )
    if payload.get("ir_version") != IR_VERSION:
        raise TrainingStateError("checkpoint IR version requires explicit migration")
    owner = _quantum_owner(module)
    workload_signature = (
        _captured_workload_signature(
            owner, prepare=True, parameter_training=version == TRAINING_STATE_VERSION
        )
        if version in (_CAPTURED_TRAINING_STATE_VERSION, TRAINING_STATE_VERSION)
        else _current_workload_signature(owner)
    )
    if (
        workload_signature is None
        or payload["workload_signature"] != workload_signature
    ):
        raise TrainingStateError("checkpoint workload does not match the target module")
    if dict(payload["topology"]) != _topology(module):
        raise TopologyMismatchError(
            "checkpoint topology does not match active topology"
        )
    saved_precision = PrecisionPolicy(**dict(payload["precision"]))
    if precision is not None and precision != saved_precision:
        raise PrecisionPolicyError(
            "checkpoint precision differs from requested precision"
        )
    checkpoint_has_optimizer = payload["optimizer_state"] is not None
    if checkpoint_has_optimizer != (optimizer is not None):
        raise TrainingStateError(
            "checkpoint optimizer presence does not match the restore request"
        )
    if payload.get("optimizer_contract") != _optimizer_contract(optimizer):
        raise TrainingStateError(
            "checkpoint optimizer type does not match the restore request"
        )
    if version == TRAINING_STATE_VERSION:
        _validate_parameter_training(payload, module, optimizer)
    return owner, saved_precision


def _restore_training_checkpoint(
    payload: Mapping[str, Any],
    *,
    module: torch.nn.Module,
    owner: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    precision: PrecisionPolicy | None,
) -> tuple[SeedContract, PrecisionPolicy]:
    is_v3 = payload.get("version") == TRAINING_STATE_VERSION
    if is_v3:
        _validate_checkpoint_target(module, optimizer)
    original_module_state = copy.deepcopy(module.state_dict())
    original_tensors = (
        [
            (tensor, tensor.data, original_module_state[name])
            for name, tensor in (*module.named_parameters(), *module.named_buffers())
        ]
        if is_v3
        else []
    )
    original_training = (
        [
            (
                parameter,
                parameter.requires_grad,
                parameter.grad,
                None if parameter.grad is None else parameter.grad.clone(),
            )
            for parameter in module.parameters()
        ]
        if is_v3
        else []
    )
    original_optimizer_state = (
        copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
    )
    original_precision = getattr(owner, "precision", None)
    rollback_seed = SeedContract(0, 0, 0, (0, 0))
    original_rng = _rng_state(rollback_seed, module)
    mutation_started = False
    try:
        if payload["version"] in (
            _CAPTURED_TRAINING_STATE_VERSION,
            TRAINING_STATE_VERSION,
        ):
            try:
                _, precision = _validate_training_checkpoint(
                    payload, module=module, optimizer=optimizer, precision=precision
                )
            except (KeyError, TypeError, ValueError, AttributeError) as error:
                if not is_v3:
                    raise
                raise TrainingStateError("malformed v3 checkpoint metadata") from error
        assert precision is not None
        if is_v3:
            if any(
                parameter.dtype != precision.torch_parameter_dtype
                for parameter in module.parameters()
            ):
                raise PrecisionPolicyError(
                    "checkpoint parameter dtype differs from its precision policy"
                )
        else:
            precision.apply(module)
        # V3 has already checked exact tensor dtypes. A whole-module .to would
        # unnecessarily cast buffers and can replace their registered objects.
        mutation_started = True
        setattr(owner, "precision", precision)
        module.load_state_dict(payload["module_state"])
        if optimizer is not None and payload["optimizer_state"] is not None:
            optimizer.load_state_dict(payload["optimizer_state"])
        if is_v3:
            for name, parameter in module.named_parameters():
                parameter.requires_grad_(
                    payload["parameter_training"]["requires_grad"][name]
                )
                parameter.grad = None
        seed = _restore_rng(payload["seed_state"])
    except Exception:
        if is_v3 and mutation_started:
            with torch.no_grad():
                for tensor, storage, value in original_tensors:
                    tensor.data = storage
                    tensor.copy_(value)
                for parameter, flag, gradient, previous_gradient in original_training:
                    parameter.requires_grad_(flag)
                    parameter.grad = gradient
                    if gradient is not None and previous_gradient is not None:
                        gradient.copy_(previous_gradient)
            for name, child in module.named_modules():
                if hasattr(child, "policy") and hasattr(child, "set_extra_state"):
                    key = f"{name}._extra_state" if name else "_extra_state"
                    child.set_extra_state(original_module_state[key])
            setattr(owner, "precision", original_precision)
            if optimizer is not None and original_optimizer_state is not None:
                torch.optim.Optimizer.load_state_dict(
                    optimizer, original_optimizer_state
                )
        elif not is_v3:
            if original_precision is not None:
                original_precision.apply(module)
                setattr(owner, "precision", original_precision)
            module.load_state_dict(original_module_state)
            if optimizer is not None and original_optimizer_state is not None:
                optimizer.load_state_dict(original_optimizer_state)
        _restore_rng(original_rng)
        raise
    if is_v3:
        from .builder_compilation import _reset_builder_capture
        from .module import Module

        # Match historical restore invalidation without applying a tensor cast.
        if isinstance(owner, Module):
            _reset_builder_capture(owner)
    return seed, precision


def load_training_checkpoint(
    path: str | Path,
    *,
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    precision: PrecisionPolicy | None = None,
) -> TrainingCheckpointRestore:
    payload = _load_training_checkpoint_payload(path)
    if payload.get("version") in (
        _CAPTURED_TRAINING_STATE_VERSION,
        TRAINING_STATE_VERSION,
    ):
        owner = _quantum_owner(module)
    else:
        owner, precision = _validate_training_checkpoint(
            payload, module=module, optimizer=optimizer, precision=precision
        )
    seed, saved_precision = _restore_training_checkpoint(
        payload,
        module=module,
        owner=owner,
        optimizer=optimizer,
        precision=precision,
    )
    return {
        "schema": TRAINING_RESTORE_SCHEMA,
        "version": TRAINING_RESTORE_VERSION,
        "step": int(payload["step"]),
        "seed": seed,
        "precision": saved_precision,
        "runtime_plan": payload["runtime_plan"],
        "topology": payload["topology"],
        "ir_hash": payload["ir_hash"],
    }


def assert_finite_training(
    module: torch.nn.Module,
    *,
    value: torch.Tensor | None = None,
    include_gradients: bool = True,
) -> None:
    failures: list[str] = []
    if value is not None and not torch.isfinite(value).all():
        failures.append("result.value")
    for name, parameter in module.named_parameters():
        if not torch.isfinite(parameter).all():
            failures.append(f"parameter:{name}")
        if (
            include_gradients
            and parameter.grad is not None
            and not torch.isfinite(parameter.grad).all()
        ):
            failures.append(f"gradient:{name}")
    if failures:
        raise NonFiniteTrainingError(
            "non-finite training values: " + ", ".join(failures)
        )


__all__ = [
    "NonFiniteTrainingError",
    "PrecisionPolicy",
    "PrecisionPolicyError",
    "SeedContract",
    "TopologyMismatchError",
    "TrainingStateError",
    "TrainingCheckpointRestore",
    "assert_finite_training",
    "load_training_checkpoint",
    "save_training_checkpoint",
    "seed_everything",
]

"""Canonical versioned training checkpoint, precision, seed, and debug contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from ..core.ir import IR_VERSION, ensure_circuit_ir

TRAINING_STATE_VERSION = "flagquantum.training_state.v1"


class TrainingStateError(RuntimeError):
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
    accumulator_dtype: str = "float32"
    mode: str = "full"
    allow_parameter_downcast: bool = False
    atol: float = 1e-5
    rtol: float = 1e-5

    def __post_init__(self) -> None:
        if self.complex_dtype not in {"complex64", "complex128"}:
            raise PrecisionPolicyError("complex dtype must be complex64 or complex128")
        if self.parameter_dtype not in {"float32", "float64"}:
            raise PrecisionPolicyError("parameter dtype must be float32 or float64")
        if self.accumulator_dtype not in {"float32", "float64"}:
            raise PrecisionPolicyError("accumulator dtype must be float32 or float64")
        if self.mode not in {"full", "mixed"}:
            raise PrecisionPolicyError("precision mode must be full or mixed")
        if self.complex_dtype == "complex128" and self.parameter_dtype != "float64":
            raise PrecisionPolicyError(
                "complex128 requires float64 trainable parameters"
            )
        if self.atol <= 0 or self.rtol <= 0:
            raise PrecisionPolicyError("precision tolerances must be positive")

    @property
    def torch_parameter_dtype(self) -> torch.dtype:
        return getattr(torch, self.parameter_dtype)

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


def seed_everything(
    seed: int, *, deterministic_algorithms: bool = True
) -> SeedContract:
    """Seed Python/Torch and describe explicit JAX/sampling/trajectory streams."""

    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return SeedContract(
        seed=seed,
        sampling_seed=(seed ^ 0x5A17) & 0x7FFF_FFFF,
        trajectory_seed=(seed ^ 0x71EC70) & 0x7FFF_FFFF,
        jax_key_words=(0, seed & 0xFFFF_FFFF),
    )


def _rng_state(seed: SeedContract) -> dict[str, Any]:
    cuda_device = torch.cuda.current_device() if torch.cuda.is_available() else None
    return {
        "contract": seed.__dict__,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state(cuda_device) if cuda_device is not None else None
        ),
        "cuda_device": cuda_device,
        "python": random.getstate(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def _restore_rng(payload: Mapping[str, Any]) -> SeedContract:
    contract = SeedContract(**dict(payload["contract"]))
    random.setstate(tuple(payload["python"]))
    torch.set_rng_state(payload["torch_cpu"])
    cuda_state = payload.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_state, torch.cuda.current_device())
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
        getattr(getattr(owner, "policy", None), "mode", None)
        == "distributed_statevector"
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
    for source, encoded_instruction in zip(ir.instructions, payload["instructions"]):
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


def save_training_checkpoint(
    path: str | Path,
    *,
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    seed: SeedContract,
    precision: PrecisionPolicy,
    runtime_plan: Any | None = None,
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
    payload = {
        "version": TRAINING_STATE_VERSION,
        "step": int(step),
        "module_state": module.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "optimizer_contract": _optimizer_contract(optimizer),
        "runtime_policy": getattr(getattr(owner, "policy", None), "__dict__", {}),
        "runtime_plan": plan_payload,
        "ir_version": getattr(ir, "version", IR_VERSION),
        "ir_hash": getattr(ir, "content_hash", ""),
        "workload_signature": _workload_signature(ir),
        "seed_state": _rng_state(seed),
        "precision": precision.to_dict(),
        "topology": _topology(module),
    }
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_training_checkpoint(
    path: str | Path,
    *,
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    precision: PrecisionPolicy | None = None,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("version") != TRAINING_STATE_VERSION:
        raise TrainingStateError(
            f"unsupported training state {payload.get('version')!r}"
        )
    if payload.get("ir_version") != IR_VERSION:
        raise TrainingStateError("checkpoint IR version requires explicit migration")
    owner = _quantum_owner(module)
    if payload.get("workload_signature") != _current_workload_signature(owner):
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
    original_module_state = copy.deepcopy(module.state_dict())
    original_optimizer_state = (
        copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
    )
    original_precision = getattr(owner, "precision", None)
    rollback_seed = SeedContract(0, 0, 0, (0, 0))
    original_rng = _rng_state(rollback_seed)
    try:
        saved_precision.apply(module)
        setattr(owner, "precision", saved_precision)
        module.load_state_dict(payload["module_state"])
        if optimizer is not None and payload["optimizer_state"] is not None:
            optimizer.load_state_dict(payload["optimizer_state"])
        seed = _restore_rng(payload["seed_state"])
    except Exception:
        if original_precision is not None:
            original_precision.apply(module)
            setattr(owner, "precision", original_precision)
        module.load_state_dict(original_module_state)
        if optimizer is not None and original_optimizer_state is not None:
            optimizer.load_state_dict(original_optimizer_state)
        _restore_rng(original_rng)
        raise
    return {
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
    "assert_finite_training",
    "load_training_checkpoint",
    "save_training_checkpoint",
    "seed_everything",
]

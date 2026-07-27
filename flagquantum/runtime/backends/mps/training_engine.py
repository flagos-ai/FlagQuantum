"""Multi-step owner-sharded training over the native distributed MPS runtime."""

from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import torch
import torch.distributed as dist
from torch.profiler import record_function

from ....core.ir import ensure_circuit_ir
from .communication import warmup_mps_neighbor_communicators
from .records import MPSReverseCheckpointPolicy
from .reverse import execute_torch_distributed_mps_reverse
from .reverse_planning import build_mps_parameter_layout
from .state import (
    initial_mps_ownership,
    validate_mps_ownership,
)
from .training import (
    MPSOptimizerOwnership,
    MPSStepMetrics,
    MPSTrainingError,
    ShardedMPSTrainingResult,
)

_initial_ownership = initial_mps_ownership
_parameter_layout = build_mps_parameter_layout


def _nvtx_phase(name: str, device: torch.device):
    return torch.cuda.nvtx.range(name) if device.type == "cuda" else nullcontext()


def _checkpoint_path(root: Path, rank: int) -> Path:
    return root / f"rank-{rank}.pt"


def _initial_state_contract(
    ir: Any,
    initial_mps_tensors: Mapping[int, torch.Tensor] | None,
    initial_bond_dimension: int,
    site_ownership: tuple[tuple[int, ...], ...],
) -> tuple[str, dict[int, tuple[int, ...]]]:
    def tensor_digest(tensor: torch.Tensor) -> str:
        digest = hashlib.sha256()
        value = tensor.detach().contiguous()
        # PyTorch 2.13 rejects a direct complex-to-byte view for some otherwise
        # contiguous MPS layouts because the complex element stride is not a
        # byte stride.  Expose real/imag channels first so checkpoint identity
        # remains a byte-exact digest for both real and complex tensors.
        if value.is_complex():
            value = torch.view_as_real(value)
        raw = value.contiguous().view(torch.uint8).reshape(-1)
        chunk_bytes = 8 * 1024 * 1024
        for start in range(0, raw.numel(), chunk_bytes):
            digest.update(raw[start : start + chunk_bytes].cpu().numpy().tobytes())
        return digest.hexdigest()

    if initial_mps_tensors is None:
        local_descriptors = {
            wire: {
                "shape": (
                    int(ir.metadata.get("batch_size", 1)),
                    1 if wire == 0 else initial_bond_dimension,
                    2,
                    1 if wire == ir.n_wires - 1 else initial_bond_dimension,
                ),
                "initializer": "flagquantum_rank_owned_mps_v1",
            }
            for wire in range(ir.n_wires)
            if wire in site_ownership[dist.get_rank()]
        }
    else:
        local_descriptors = {
            int(wire): {
                "shape": tuple(int(value) for value in tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": tensor_digest(tensor),
            }
            for wire, tensor in initial_mps_tensors.items()
        }
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_descriptors)
    global_descriptors = {
        int(wire): descriptor
        for payload in gathered
        for wire, descriptor in payload.items()
    }
    content = json.dumps(global_descriptors, sort_keys=True, separators=(",", ":"))
    local_shapes = {
        wire: tuple(descriptor["shape"])
        for wire, descriptor in local_descriptors.items()
    }
    return hashlib.sha256(content.encode()).hexdigest(), local_shapes


def _contract(
    ir: Any,
    optimizer: str,
    lr: float,
    lr_decay: float,
    max_bond: int | None,
    cutoff: float,
    *,
    world_size: int,
    initial_state_fingerprint: str,
    gradient_policy: str,
    gradient_tolerance: float,
    initial_mps_left_canonical: bool,
    reverse_checkpoint_policy: MPSReverseCheckpointPolicy,
    gradient_bucket_bytes: int,
    compile_site_kernels: bool,
    prefetch_layer_halos: bool,
    compile_observables: bool,
    fuse_local_reverse: bool,
    canonicalization_policy: str,
    site_ownership: tuple[tuple[int, ...], ...],
) -> dict[str, Any]:
    return {
        "schema": "sharded_mps_training_checkpoint_v1",
        "ir_hash": ir.content_hash,
        "optimizer": optimizer,
        "learning_rate": float(lr),
        "learning_rate_decay": float(lr_decay),
        "max_bond": max_bond,
        "cutoff": float(cutoff),
        "world_size": world_size,
        "initial_state_fingerprint": initial_state_fingerprint,
        "gradient_policy": gradient_policy,
        "gradient_tolerance": float(gradient_tolerance),
        "initial_mps_left_canonical": bool(initial_mps_left_canonical),
        "reverse_checkpoint_policy": asdict(reverse_checkpoint_policy),
        "gradient_bucket_bytes": int(gradient_bucket_bytes),
        "gradient_route": "bounded_dtype_owner_reduce_v1",
        "compile_site_kernels": bool(compile_site_kernels),
        "prefetch_layer_halos": bool(prefetch_layer_halos),
        "compile_observables": bool(compile_observables),
        "fuse_local_reverse": bool(fuse_local_reverse),
        "canonicalization_policy": canonicalization_policy,
        "site_ownership": site_ownership,
    }


def _save_checkpoint(
    root: Path,
    *,
    rank: int,
    world_size: int,
    step: int,
    owned_indices: tuple[int, ...],
    parameters: tuple[torch.Tensor, ...],
    optimizer: torch.optim.Optimizer | None,
    contract: Mapping[str, Any],
    bond_layout: Mapping[int, tuple[int, ...]],
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(root, rank)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {
            "contract": dict(contract),
            "world_size": world_size,
            "completed_steps": step,
            "owned_indices": owned_indices,
            "parameters": {i: parameters[i].detach().cpu() for i in owned_indices},
            "optimizer_state": (
                optimizer.state_dict() if optimizer is not None else None
            ),
            "bond_layout": dict(bond_layout),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state(parameters[0].device)
                if parameters[0].device.type == "cuda"
                else None
            ),
        },
        temporary,
    )
    temporary.replace(path)
    return path


def _load_checkpoint(
    root: Path,
    *,
    rank: int,
    world_size: int,
    owned_indices: tuple[int, ...],
    parameters: tuple[torch.Tensor, ...],
    optimizer: torch.optim.Optimizer | None,
    contract: Mapping[str, Any],
    bond_layout: Mapping[int, tuple[int, ...]],
) -> int:
    path = _checkpoint_path(root, rank)
    if not path.exists():
        raise MPSTrainingError(f"checkpoint missing for rank {rank}: {path}")
    payload = torch.load(path, map_location=parameters[0].device, weights_only=False)
    if (
        payload.get("contract") != dict(contract)
        or payload.get("world_size") != world_size
    ):
        raise MPSTrainingError("checkpoint contract or topology mismatch")
    if tuple(payload.get("owned_indices", ())) != owned_indices:
        raise MPSTrainingError("checkpoint optimizer ownership mismatch")
    if payload.get("bond_layout") != dict(bond_layout):
        raise MPSTrainingError("checkpoint MPS bond layout mismatch")
    for index, value in payload["parameters"].items():
        parameters[int(index)].data.copy_(value.to(parameters[int(index)].device))
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if (
        parameters[0].device.type == "cuda"
        and payload.get("cuda_rng_state") is not None
    ):
        torch.cuda.set_rng_state(payload["cuda_rng_state"].cpu(), parameters[0].device)
    return int(payload["completed_steps"])


def _broadcast_parameters(
    parameters: tuple[torch.Tensor, ...], owners: tuple[int, ...]
) -> int:
    for parameter, owner in zip(parameters, owners):
        dist.broadcast(parameter.data, src=owner)
    return len(parameters)


def _distributed_dot(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    value = torch.dot(left, right)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def _owner_flatten(tensors: Sequence[torch.Tensor], *, gradients: bool) -> torch.Tensor:
    values = []
    for tensor in tensors:
        value = tensor.grad if gradients else tensor.detach()
        if value is None:
            value = torch.zeros_like(tensor)
        values.append(value.reshape(-1))
    if values:
        return torch.cat(values)
    return torch.empty(0, device=tensors[0].device if tensors else "cpu")


def _owner_add_(parameters: Sequence[torch.Tensor], update: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            count = parameter.numel()
            parameter.add_(update[offset : offset + count].reshape_as(parameter))
            offset += count


def _sharded_lbfgs_direction(
    gradient: torch.Tensor,
    history: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Apply distributed L-BFGS two-loop recursion to owner-local shards."""

    q = gradient.clone()
    alphas: list[torch.Tensor] = []
    for step_delta, gradient_delta, inverse_curvature in reversed(history):
        alpha = inverse_curvature * _distributed_dot(step_delta, q)
        alphas.append(alpha)
        q = q - alpha * gradient_delta
    if history:
        step_delta, gradient_delta, _ = history[-1]
        yy = _distributed_dot(gradient_delta, gradient_delta).clamp_min(1e-30)
        scale = _distributed_dot(step_delta, gradient_delta) / yy
        result = scale * q
    else:
        result = q
    for (step_delta, gradient_delta, inverse_curvature), alpha in zip(
        history, reversed(alphas)
    ):
        beta = inverse_curvature * _distributed_dot(gradient_delta, result)
        result = result + step_delta * (alpha - beta)
    return -result


def train_distributed_mps(
    circuit_or_ir: Any | Callable[[], Any],
    *,
    steps: int,
    observable: Mapping[int, str] | None = None,
    observable_terms: (
        Sequence[tuple[Mapping[int, str], torch.Tensor | float]] | None
    ) = None,
    hamiltonian_terms: (
        Sequence[tuple[Mapping[int, str], torch.Tensor | float]] | None
    ) = None,
    optimizer: Literal["sgd", "adam", "adam_lbfgs"] = "sgd",
    lr: float = 0.05,
    lr_decay: float = 1.0,
    lbfgs_start_step: int | None = None,
    lbfgs_lr: float = 0.8,
    lbfgs_history_size: int = 10,
    device: torch.device | str | None = None,
    max_bond: int | None = None,
    cutoff: float = 0.0,
    gradient_policy: Literal["exact", "approximate"] = "exact",
    gradient_tolerance: float = 0.0,
    checkpoint_dir: str | Path | None = None,
    checkpoint_interval: int = 1,
    resume: bool = False,
    memory_leak_tolerance_bytes: int = 0,
    initial_bond_dimension: int = 1,
    initial_mps_tensors: Mapping[int, torch.Tensor] | None = None,
    initial_mps_left_canonical: bool = False,
    site_ownership: Sequence[Sequence[int]] | None = None,
    reverse_checkpoint_policy: MPSReverseCheckpointPolicy | None = None,
    memory_warmup_steps: int = 1,
    gradient_bucket_bytes: int = 25 * 1024 * 1024,
    compile_site_kernels: bool = False,
    prefetch_layer_halos: bool = True,
    compile_observables: bool = False,
    fuse_local_reverse: bool = True,
    canonicalization_policy: Literal["dirty", "full"] = "dirty",
    svd_driver: str | None = "gesvd",
    record_parameter_gradients: bool = False,
) -> ShardedMPSTrainingResult:
    """Train one rank-sharded MPS with native PyTorch SGD or Adam.

    ``hamiltonian_terms`` is a linear energy objective.  Its optimized path
    currently accepts on-site Z fields and adjacent XX, YY, or ZZ couplings and
    contracts them as one five-channel Heisenberg MPO.  ``observable_terms``
    retains its existing multi-target MSE semantics.

    The caller owns process-group initialization and cleanup. This keeps the
    function composable under torchrun and makes failures visible to the
    launcher rather than silently replacing distributed execution.
    """

    if not dist.is_initialized():
        raise MPSTrainingError("train_distributed_mps requires torch.distributed")
    if steps <= 0 or lr <= 0 or checkpoint_interval <= 0:
        raise ValueError("steps, lr and checkpoint_interval must be positive")
    if not math.isfinite(lr_decay) or not 0.0 < lr_decay <= 1.0:
        raise ValueError("lr_decay must be finite and in (0, 1]")
    if optimizer not in {"sgd", "adam", "adam_lbfgs"}:
        raise ValueError("optimizer must be 'sgd', 'adam', or 'adam_lbfgs'")
    if optimizer == "adam_lbfgs" and (
        lbfgs_start_step is None or not 0 < lbfgs_start_step < steps
    ):
        raise ValueError("adam_lbfgs requires 0 < lbfgs_start_step < steps")
    if lbfgs_lr <= 0 or lbfgs_history_size <= 0:
        raise ValueError("lbfgs_lr and lbfgs_history_size must be positive")
    if optimizer == "adam_lbfgs" and (checkpoint_dir is not None or resume):
        raise MPSTrainingError(
            "adam_lbfgs checkpoint/resume requires versioned L-BFGS history support"
        )
    if memory_leak_tolerance_bytes < 0:
        raise ValueError("memory_leak_tolerance_bytes must be non-negative")
    if memory_warmup_steps < 0:
        raise ValueError("memory_warmup_steps must be non-negative")
    if gradient_bucket_bytes <= 0:
        raise ValueError("gradient_bucket_bytes must be positive")
    if canonicalization_policy not in {"dirty", "full"}:
        raise ValueError("canonicalization_policy must be dirty or full")
    circuit_factory = circuit_or_ir if callable(circuit_or_ir) else None
    ir = ensure_circuit_ir(
        circuit_factory() if circuit_factory is not None else circuit_or_ir
    )
    if circuit_factory is not None and (checkpoint_dir is not None or resume):
        raise MPSTrainingError(
            "circuit_factory training does not yet support checkpoint/resume"
        )
    rank, world_size = dist.get_rank(), dist.get_world_size()
    resolved_site_ownership = (
        _initial_ownership(ir.n_wires, world_size)
        if site_ownership is None
        else validate_mps_ownership(site_ownership, ir.n_wires, world_size)
    )
    local_world_size = int(__import__("os").environ.get("LOCAL_WORLD_SIZE", world_size))
    backend = str(dist.get_backend())
    resolved_device = torch.device(
        device
        or (f"cuda:{torch.cuda.current_device()}" if backend == "nccl" else "cpu")
    )
    if backend == "nccl" and resolved_device.type != "cuda":
        raise MPSTrainingError("NCCL MPS training requires a CUDA device")

    parameters, _ = _parameter_layout(ir)
    owners = tuple(index % world_size for index in range(len(parameters)))
    owned_indices = tuple(index for index, owner in enumerate(owners) if owner == rank)
    owned = [parameters[index] for index in owned_indices]
    optimizer_obj: torch.optim.Optimizer | None = None
    if owned:
        optimizer_obj = (
            torch.optim.SGD(owned, lr=lr)
            if optimizer == "sgd"
            else torch.optim.Adam(owned, lr=lr)
        )
    initial_state_fingerprint, local_bond_layout = _initial_state_contract(
        ir, initial_mps_tensors, initial_bond_dimension, resolved_site_ownership
    )
    selected_reverse_policy = reverse_checkpoint_policy or MPSReverseCheckpointPolicy()
    contract = _contract(
        ir,
        optimizer,
        lr,
        lr_decay,
        max_bond,
        cutoff,
        world_size=world_size,
        initial_state_fingerprint=initial_state_fingerprint,
        gradient_policy=gradient_policy,
        gradient_tolerance=gradient_tolerance,
        initial_mps_left_canonical=initial_mps_left_canonical,
        reverse_checkpoint_policy=selected_reverse_policy,
        gradient_bucket_bytes=gradient_bucket_bytes,
        compile_site_kernels=compile_site_kernels,
        prefetch_layer_halos=prefetch_layer_halos,
        compile_observables=compile_observables,
        fuse_local_reverse=fuse_local_reverse,
        canonicalization_policy=canonicalization_policy,
        site_ownership=resolved_site_ownership,
    )
    contract_fingerprint = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    communication_setup_seconds = warmup_mps_neighbor_communicators(resolved_device)
    root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    start_step = 0
    if resume:
        if root is None:
            raise ValueError("resume requires checkpoint_dir")
        start_step = _load_checkpoint(
            root,
            rank=rank,
            world_size=world_size,
            owned_indices=owned_indices,
            parameters=parameters,
            optimizer=optimizer_obj,
            contract=contract,
            bond_layout=local_bond_layout,
        )
        if start_step > steps:
            raise MPSTrainingError(
                f"checkpoint completed step {start_step} exceeds target {steps}"
            )
        completed = torch.tensor(start_step, device=resolved_device)
        gathered = [torch.zeros_like(completed) for _ in range(world_size)]
        dist.all_gather(gathered, completed)
        if len({int(value.item()) for value in gathered}) != 1:
            raise MPSTrainingError("checkpoint generations differ across ranks")
        _broadcast_parameters(parameters, owners)

    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    losses: list[float] = []
    metrics: list[MPSStepMetrics] = []
    checkpoints: list[str] = []
    memories: list[int] = []
    lbfgs_history: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    previous_lbfgs_parameters: torch.Tensor | None = None
    previous_lbfgs_gradient: torch.Tensor | None = None
    for step in range(start_step, steps):
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)
            torch.cuda.reset_peak_memory_stats(resolved_device)
        step_started = time.perf_counter()
        forward_started = step_started
        step_ir = (
            ensure_circuit_ir(circuit_factory()) if circuit_factory is not None else ir
        )
        if circuit_factory is not None:
            step_parameters, _ = _parameter_layout(step_ir)
            if tuple(map(id, step_parameters)) != tuple(map(id, parameters)):
                raise MPSTrainingError(
                    "circuit_factory must reuse the same trainable leaf tensors"
                )
        with (
            record_function("flagquantum::mps::forward"),
            _nvtx_phase("flagquantum::mps::forward", resolved_device),
        ):
            reverse = execute_torch_distributed_mps_reverse(
                step_ir,
                observable=observable,
                observable_terms=observable_terms,
                hamiltonian_terms=hamiltonian_terms,
                device=resolved_device,
                max_bond=max_bond,
                cutoff=cutoff,
                gradient_policy=gradient_policy,
                gradient_tolerance=gradient_tolerance,
                checkpoint_policy=selected_reverse_policy,
                initial_bond_dimension=initial_bond_dimension,
                initial_mps_tensors=initial_mps_tensors,
                initial_mps_left_canonical=initial_mps_left_canonical,
                site_ownership=resolved_site_ownership,
                gradient_owner_ranks=owners,
                gradient_bucket_bytes=gradient_bucket_bytes,
                compile_site_kernels=compile_site_kernels,
                prefetch_layer_halos=prefetch_layer_halos,
                compile_observables=compile_observables,
                fuse_local_reverse=fuse_local_reverse,
                canonicalization_policy=canonicalization_policy,
                svd_driver=svd_driver,
            )
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)
            forward_peak_memory = int(torch.cuda.max_memory_allocated(resolved_device))
            torch.cuda.reset_peak_memory_stats(resolved_device)
        else:
            forward_peak_memory = 0
        forward_seconds = time.perf_counter() - forward_started
        for parameter in parameters:
            parameter.grad = None
        reverse_started = time.perf_counter()
        with (
            record_function("flagquantum::mps::reverse_vjp"),
            _nvtx_phase("flagquantum::mps::reverse_vjp", resolved_device),
        ):
            reverse.backward()
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)
            reverse_peak_memory = int(torch.cuda.max_memory_allocated(resolved_device))
        else:
            reverse_peak_memory = 0
        reverse_seconds = time.perf_counter() - reverse_started
        diagnostic_parameter_values = (
            tuple(
                (index, float(parameters[index].detach().cpu()))
                for index in owned_indices
            )
            if record_parameter_gradients
            else ()
        )
        optimizer_started = time.perf_counter()
        optimizer_stage = (
            "lbfgs"
            if optimizer == "adam_lbfgs" and step >= int(lbfgs_start_step)
            else ("adam" if optimizer == "adam_lbfgs" else optimizer)
        )
        step_learning_rate = (
            lbfgs_lr
            if optimizer_stage == "lbfgs"
            else (
                float(optimizer_obj.param_groups[0]["lr"])
                if optimizer_obj is not None
                else float(lr * lr_decay**step)
            )
        )
        with (
            record_function("flagquantum::mps::optimizer"),
            _nvtx_phase("flagquantum::mps::optimizer", resolved_device),
        ):
            optimizer_collective_count = 0
            if optimizer_stage == "lbfgs":
                current_parameters = _owner_flatten(owned, gradients=False)
                current_gradient = _owner_flatten(owned, gradients=True)
                if previous_lbfgs_parameters is not None:
                    step_delta = current_parameters - previous_lbfgs_parameters
                    gradient_delta = current_gradient - previous_lbfgs_gradient
                    curvature = _distributed_dot(step_delta, gradient_delta)
                    optimizer_collective_count += 1
                    if float(curvature.detach().cpu()) > 1e-14:
                        lbfgs_history.append(
                            (step_delta, gradient_delta, curvature.reciprocal())
                        )
                        del lbfgs_history[:-lbfgs_history_size]
                direction = _sharded_lbfgs_direction(current_gradient, lbfgs_history)
                optimizer_collective_count += 2 * len(lbfgs_history) + (
                    2 if lbfgs_history else 0
                )
                previous_lbfgs_parameters = current_parameters.clone()
                previous_lbfgs_gradient = current_gradient.clone()
                _owner_add_(owned, lbfgs_lr * direction)
            elif optimizer_obj is not None:
                optimizer_obj.step()
                for parameter_group in optimizer_obj.param_groups:
                    parameter_group["lr"] *= lr_decay
            _broadcast_parameters(parameters, owners)
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)
        optimizer_seconds = time.perf_counter() - optimizer_started
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)
            memory = max(forward_peak_memory, reverse_peak_memory)
            allocated_memory = int(torch.cuda.memory_allocated(resolved_device))
            reserved_memory = int(torch.cuda.memory_reserved(resolved_device))
        else:
            memory = sum(
                parameter.numel() * parameter.element_size() for parameter in owned
            )
            allocated_memory = memory
            reserved_memory = memory
        memories.append(allocated_memory)
        owned_records = [
            record for record in reverse.tape.records if record.compute_owner == rank
        ]
        owned_site = sum(record.kind == "one_site" for record in owned_records)
        owned_bond = sum(record.kind != "one_site" for record in owned_records)
        boundaries = sum(
            record.communication_peer is not None and rank in record.owner_ranks
            for record in reverse.tape.records
        )
        boundary_records = tuple(
            record
            for record in reverse.tape.records
            if record.communication_peer is not None and rank in record.owner_ranks
        )
        element_size = torch.empty((), dtype=getattr(torch, ir.dtype)).element_size()
        bond_updates = tuple(
            {
                "operation_id": record.operation_id,
                "bond": min(record.wires),
                "owner_ranks": record.owner_ranks,
                "compute_owner": record.compute_owner,
                "original_rank": record.original_rank,
                "kept_rank": record.kept_rank,
                "discarded_weight": record.discarded_weight,
                "communication_peer": record.communication_peer,
                "communication_sequence": record.communication_sequence,
                "input_shapes": record.input_shapes,
                "output_shapes": record.output_shapes,
                "payload_bytes": sum(
                    math.prod(shape) * element_size
                    for shape in record.input_shapes + record.output_shapes
                ),
                "forward_transport": "batched_isend_irecv",
                "reverse_transport": "batched_isend_irecv",
            }
            for record in reverse.tape.records
            if record.kind == "two_site"
            and record.communication_peer is not None
            and rank in record.owner_ranks
        )
        boundary_bytes = sum(
            sum(math.prod(shape) * element_size for shape in record.input_shapes)
            + sum(math.prod(shape) * element_size for shape in record.output_shapes)
            for record in boundary_records
        )
        optimizer_memory = (
            0
            if optimizer_obj is None
            else sum(
                value.numel() * value.element_size()
                for state in optimizer_obj.state.values()
                for value in state.values()
                if isinstance(value, torch.Tensor)
            )
        )
        optimizer_memory += sum(
            value.numel() * value.element_size()
            for item in lbfgs_history
            for value in item
        )
        optimizer_memory += sum(
            value.numel() * value.element_size()
            for value in (previous_lbfgs_parameters, previous_lbfgs_gradient)
            if value is not None
        )
        loss = float(reverse.value.detach().cpu())
        losses.append(loss)
        metrics.append(
            MPSStepMetrics(
                step=step,
                loss=loss,
                forward_seconds=forward_seconds,
                reverse_seconds=reverse_seconds,
                optimizer_seconds=optimizer_seconds,
                end_to_end_seconds=time.perf_counter() - step_started,
                owned_site_work=owned_site,
                owned_bond_work=owned_bond,
                boundary_exchanges=boundaries,
                svd_activity=sum(record.kind == "two_site" for record in owned_records),
                qr_activity=sum(
                    record.kind.startswith("canonicalize") for record in owned_records
                ),
                peak_memory_bytes=memory,
                useful_work_completed=(owned_site + owned_bond) > 0,
                two_site_splits=sum(
                    record.kind == "two_site" for record in owned_records
                ),
                truncated_splits=sum(
                    record.kind == "two_site"
                    and record.kept_rank is not None
                    and record.original_rank is not None
                    and record.kept_rank < record.original_rank
                    for record in owned_records
                ),
                discarded_weight=sum(
                    record.discarded_weight for record in owned_records
                ),
                boundary_forward_exchanges=len(boundary_records),
                boundary_reverse_exchanges=len(boundary_records),
                boundary_bytes=boundary_bytes,
                bond_updates=bond_updates,
                allocated_memory_bytes=allocated_memory,
                reserved_memory_bytes=reserved_memory,
                tape_memory_bytes=reverse.tape.saved_tensor_bytes,
                optimizer_memory_bytes=optimizer_memory,
                communication_buffer_bytes=boundary_bytes,
                gradient_collective_count=reverse.gradient_collective_count,
                gradient_collective_bytes=reverse.gradient_collective_bytes,
                objective_scan_pairs=reverse.objective_scan_pairs,
                objective_scan_forward_messages=reverse.objective_scan_forward_messages,
                objective_scan_reverse_messages=reverse.objective_scan_reverse_messages,
                objective_execution=reverse.objective_execution,
                learning_rate=step_learning_rate,
                largest_realized_bond=max(
                    (
                        int(record.kept_rank)
                        for record in owned_records
                        if record.kind == "two_site" and record.kept_rank is not None
                    ),
                    default=1,
                ),
                forward_peak_memory_bytes=forward_peak_memory,
                reverse_peak_memory_bytes=reverse_peak_memory,
                optimizer_stage=optimizer_stage,
                objective_evaluations=1,
                parameter_gradients=(
                    tuple(
                        (index, float(parameters[index].grad.detach().cpu()))
                        for index in owned_indices
                    )
                    if record_parameter_gradients
                    else ()
                ),
                parameter_values=diagnostic_parameter_values,
                optimizer_collective_count=optimizer_collective_count,
                optimizer_collective_bytes=(
                    optimizer_collective_count
                    * torch.empty((), dtype=parameters[0].dtype).element_size()
                ),
                lbfgs_history_length=len(lbfgs_history),
            )
        )
        if root is not None and (step + 1) % checkpoint_interval == 0:
            path = _save_checkpoint(
                root,
                rank=rank,
                world_size=world_size,
                step=step + 1,
                owned_indices=owned_indices,
                parameters=parameters,
                optimizer=optimizer_obj,
                contract=contract,
                bond_layout=local_bond_layout,
            )
            checkpoints.append(str(path))
            dist.barrier()

    stable_memories = memories[min(memory_warmup_steps, len(memories) - 1) :]
    growth = (
        max(0, stable_memories[-1] - stable_memories[0])
        if len(stable_memories) > 1
        else 0
    )
    ownership = tuple(
        MPSOptimizerOwnership(
            parameter_index=index,
            owner_rank=owner,
            optimizer_state_local=owner == rank,
            gradient_route="bounded_dtype_owner_reduce",
            update_route="owner_step_then_broadcast",
        )
        for index, owner in enumerate(owners)
    )
    return ShardedMPSTrainingResult(
        losses=tuple(losses),
        completed_steps=steps,
        start_step=start_step,
        rank=rank,
        world_size=world_size,
        local_world_size=local_world_size,
        optimizer=optimizer,
        ownership=ownership,
        steps=tuple(metrics),
        checkpoint_files=tuple(checkpoints),
        memory_growth_bytes=growth,
        suspected_memory_leak=growth > memory_leak_tolerance_bytes,
        checkpoint_contract_fingerprint=contract_fingerprint,
        memory_warmup_steps=memory_warmup_steps,
        communication_setup_seconds=communication_setup_seconds,
    )


__all__ = (
    "MPSOptimizerOwnership",
    "MPSStepMetrics",
    "MPSTrainingError",
    "ShardedMPSTrainingResult",
    "train_distributed_mps",
)

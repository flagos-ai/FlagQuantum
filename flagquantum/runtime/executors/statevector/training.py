"""Reliable owner-sharded training over native statevector reverse mode."""

from __future__ import annotations

import hashlib
import os
import resource
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Literal

import torch
import torch.distributed as dist

from ....compute import get_platform_runtime, resolve_platform_device
from ....core.ir import ensure_circuit_ir
from ....testing.watchdog import PhaseAwareWatchdog, ProgressSnapshot
from .reverse import (
    StatevectorCheckpointPolicy,
    execute_torch_distributed_statevector_reverse,
)


class DistributedTrainingError(RuntimeError):
    """Structured, rank-aware training lifecycle failure."""

    def __init__(self, cause: str, *, rank: int, phase: str, operation: str) -> None:
        self.cause = cause
        self.rank = rank
        self.phase = phase
        self.operation = operation
        super().__init__(f"{cause}: rank={rank} phase={phase} operation={operation}")

    def to_dict(self) -> dict[str, Any]:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        return {
            "cause": self.cause,
            "rank": self.rank,
            "phase": self.phase,
            "last_operation": self.operation,
            "device": (
                f"cuda:{torch.cuda.current_device()}"
                if torch.cuda.is_available()
                else "cpu"
            ),
            "topology": {"world_size": world_size, "rank": self.rank},
            "cleanup_required": True,
        }


class _OperationTimeoutError(TimeoutError):
    pass


@contextmanager
def _operation_deadline(timeout_seconds: float):
    """Interrupt a blocking main-thread runtime operation at its deadline."""

    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def expired(signum: int, frame: Any) -> None:
        raise _OperationTimeoutError(f"operation exceeded {timeout_seconds} seconds")

    signal.signal(signal.SIGALRM, expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


@dataclass(frozen=True)
class TrainingProgress:
    rank: int
    step: int
    phase: str
    operation: str
    completed_units: int
    collective: str
    memory_bytes: int
    useful_work_launched: bool
    device: str
    device_activity: str
    topology: str
    timestamp: float


@dataclass(frozen=True)
class OptimizerOwnership:
    parameter: str
    owner_rank: int
    gradient_reduction: str
    optimizer_state_local: bool
    update_route: str


@dataclass(frozen=True)
class ShardedTrainingResult:
    losses: tuple[float, ...]
    completed_steps: int
    start_step: int
    rank: int
    world_size: int
    local_world_size: int
    node_count: int
    optimizer: str
    ownership: tuple[OptimizerOwnership, ...]
    progress: tuple[TrainingProgress, ...]
    peak_memory_bytes: int
    communication_events: int
    communication_bytes: int
    checkpoint_files: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        ownership_semantics = (
            "sharded_across_ranks" if self.world_size > 1 else "single_device_fast_path"
        )
        return {
            "executor": "pytorch_native_sharded_statevector_training_v1",
            "rank": self.rank,
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "node_count": self.node_count,
            "completed_steps": self.completed_steps,
            "start_step": self.start_step,
            "losses": self.losses,
            "optimizer": self.optimizer,
            "parameter_ownership_semantics": ownership_semantics,
            "gradient_ownership_semantics": (
                "reduced_across_ranks" if self.world_size > 1 else "local"
            ),
            "optimizer_state_ownership_semantics": ownership_semantics,
            "optimizer_update_ownership_semantics": ownership_semantics,
            "optimizer_update_semantics": ownership_semantics,
            "optimizer_ownership": tuple(asdict(item) for item in self.ownership),
            "progress_event_count": len(self.progress),
            "communication_events": self.communication_events,
            "communication_bytes": self.communication_bytes,
            "peak_memory_bytes": self.peak_memory_bytes,
            "checkpoint_files": self.checkpoint_files,
            "distribution_semantics": (
                "sharded_across_ranks"
                if self.world_size > 1
                else "single_device_fast_path"
            ),
            "scalability_claim_allowed": False,
            "release_gate_allowed": False,
            "blockers": ("release_capacity_and_multinode_acceptance_pending",),
        }


def _memory_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        memory = get_platform_runtime(device.type).memory_snapshot(device)
        return int(memory.allocated_bytes or 0)
    # Linux reports ru_maxrss in KiB. This is process-wide CPU resident memory,
    # which is the measurable allocator envelope available without sampling
    # platform-specific malloc internals.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _broadcast_parameters(
    parameters: tuple[torch.Tensor, ...], owners: tuple[int, ...]
) -> int:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return 0
    for parameter, owner in zip(parameters, owners):
        dist.broadcast(parameter.data, src=owner)
    return len(parameters)


def _failure_detection_barrier(timeout_seconds: float) -> None:
    """Enter a backend-appropriate barrier bounded by the process-group timeout."""

    if str(dist.get_backend()) == "gloo":
        dist.monitored_barrier(timeout=timedelta(seconds=timeout_seconds))
    else:
        work = dist.barrier(async_op=True)
        if not work.wait(timeout=timedelta(seconds=timeout_seconds)):
            raise _OperationTimeoutError("distributed barrier deadline expired")


def _raise_coordinated_injected_failure(
    *, fault_rank: int, rank: int, world_size: int, device: torch.device, step: int
) -> None:
    """Propagate a synthetic rank fault without breaking the NCCL process group.

    A rank disappearing while peers are inside an NCCL collective is handled by
    the launcher/control plane, not by Python exception recovery.  Controlled
    fault injection is different: every rank votes before raising so tests can
    exercise structured application teardown without triggering NCCL's aborting
    watchdog path.
    """

    if not 0 <= fault_rank < world_size:
        raise ValueError(
            f"fault_rank must be in [0, {world_size}), received {fault_rank}"
        )
    if world_size > 1:
        fault_vote = torch.tensor(int(rank == fault_rank), device=device)
        dist.all_reduce(fault_vote, op=dist.ReduceOp.SUM)
        if int(fault_vote.item()) != 1:
            raise DistributedTrainingError(
                "fault_injection_protocol_error",
                rank=rank,
                phase="collective",
                operation="fault_vote",
            )
    raise DistributedTrainingError(
        "injected_rank_failure" if rank == fault_rank else "peer_rank_failure",
        rank=rank,
        phase="forward" if rank == fault_rank else "collective",
        operation=f"step:{step}" if rank == fault_rank else "fault_vote",
    )


def _raise_control_plane_timeout(
    *, rank: int, world_size: int, timeout_seconds: float
) -> None:
    """Exercise a bounded collective timeout without poisoning the data group."""

    if world_size == 1:
        raise DistributedTrainingError(
            "collective_timeout",
            rank=rank,
            phase="collective",
            operation="timeout_requires_multiple_ranks",
        )
    control_group = dist.new_group(
        backend="gloo", timeout=timedelta(seconds=timeout_seconds)
    )
    try:
        if rank == 0:
            time.sleep(timeout_seconds * 1.5)
            operation = "rank_did_not_enter_control_barrier"
        else:
            try:
                dist.monitored_barrier(
                    group=control_group,
                    timeout=timedelta(seconds=timeout_seconds),
                    wait_all_ranks=True,
                )
            except RuntimeError:
                operation = "control_plane_monitored_barrier"
            else:
                operation = "control_barrier_unexpectedly_completed"
        raise DistributedTrainingError(
            "collective_timeout",
            rank=rank,
            phase="collective",
            operation=operation,
        )
    finally:
        dist.destroy_process_group(control_group)


def _validate_checkpoint_generations(steps: tuple[int, ...], *, rank: int) -> None:
    if len(set(steps)) != 1:
        raise DistributedTrainingError(
            "checkpoint_generation_mismatch",
            rank=rank,
            phase="resume",
            operation="validate_completed_steps",
        )


def _checkpoint_path(root: Path, rank: int) -> Path:
    return root / f"rank-{rank}.pt"


def _checkpoint_checksum_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_contract(
    ir: Any,
    *,
    observable_wire: int,
    optimizer_name: str,
    lr: float,
    parameters: tuple[torch.Tensor, ...],
) -> dict[str, Any]:
    return {
        "ir_content_hash": ir.content_hash,
        "ir_version": ir.version,
        "n_wires": ir.n_wires,
        "observable_wire": int(observable_wire),
        "optimizer": optimizer_name,
        "learning_rate": float(lr),
        "parameter_schema": tuple(
            (tuple(parameter.shape), str(parameter.dtype)) for parameter in parameters
        ),
    }


def _write_checkpoint(
    root: Path,
    *,
    rank: int,
    world_size: int,
    step: int,
    optimizer_name: str,
    owned_indices: tuple[int, ...],
    parameters: tuple[torch.Tensor, ...],
    optimizer: torch.optim.Optimizer | None,
    contract: dict[str, Any],
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(root, rank)
    payload = {
        "schema_version": "sharded_statevector_training_checkpoint_v1",
        "rank": rank,
        "world_size": world_size,
        "completed_steps": step,
        "optimizer": optimizer_name,
        "owned_indices": owned_indices,
        "owned_parameters": {
            index: parameters[index].detach().cpu() for index in owned_indices
        },
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "contract": contract,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    checksum_path = _checkpoint_checksum_path(path)
    checksum_temporary = checksum_path.with_suffix(checksum_path.suffix + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    checksum_temporary.write_text(_file_sha256(temporary) + "\n", encoding="ascii")
    with checksum_temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)
    checksum_temporary.replace(checksum_path)
    directory_fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return path


def _load_checkpoint(
    root: Path,
    *,
    rank: int,
    world_size: int,
    optimizer_name: str,
    owned_indices: tuple[int, ...],
    parameters: tuple[torch.Tensor, ...],
    optimizer: torch.optim.Optimizer | None,
    contract: dict[str, Any],
) -> int:
    path = _checkpoint_path(root, rank)
    if not path.exists():
        raise DistributedTrainingError(
            "checkpoint_missing", rank=rank, phase="resume", operation=str(path)
        )
    checksum_path = _checkpoint_checksum_path(path)
    if not checksum_path.exists():
        raise DistributedTrainingError(
            "checkpoint_integrity_metadata_missing",
            rank=rank,
            phase="resume",
            operation=str(checksum_path),
        )
    try:
        expected_checksum = checksum_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise DistributedTrainingError(
            "checkpoint_integrity_metadata_invalid",
            rank=rank,
            phase="resume",
            operation=str(checksum_path),
        ) from error
    if len(expected_checksum) != 64 or _file_sha256(path) != expected_checksum:
        raise DistributedTrainingError(
            "checkpoint_integrity_mismatch",
            rank=rank,
            phase="resume",
            operation=str(path),
        )
    try:
        payload = torch.load(
            path, map_location=parameters[0].device, weights_only=False
        )
    except (OSError, RuntimeError, ValueError, EOFError) as error:
        raise DistributedTrainingError(
            "checkpoint_deserialization_failed",
            rank=rank,
            phase="resume",
            operation=str(path),
        ) from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "sharded_statevector_training_checkpoint_v1"
        or payload.get("world_size") != world_size
        or payload.get("optimizer") != optimizer_name
        or payload.get("contract") != contract
    ):
        if isinstance(payload, dict) and "unsafe_custom_object" in payload:
            raise DistributedTrainingError(
                "checkpoint_deserialization_failed",
                rank=rank,
                phase="resume",
                operation="restricted_load",
            )
        raise DistributedTrainingError(
            "checkpoint_contract_mismatch",
            rank=rank,
            phase="resume",
            operation="validate_checkpoint",
        )
    if tuple(payload.get("owned_indices", ())) != owned_indices:
        raise DistributedTrainingError(
            "checkpoint_ownership_mismatch",
            rank=rank,
            phase="resume",
            operation="validate_ownership",
        )
    owned_parameters = payload.get("owned_parameters")
    if not isinstance(owned_parameters, dict) or {
        int(index) for index in owned_parameters
    } != set(owned_indices):
        raise DistributedTrainingError(
            "checkpoint_parameter_payload_mismatch",
            rank=rank,
            phase="resume",
            operation="validate_owned_parameters",
        )
    try:
        for index, value in owned_parameters.items():
            target = parameters[int(index)]
            if not isinstance(value, torch.Tensor) or (
                value.shape != target.shape or value.dtype != target.dtype
            ):
                raise ValueError("owned parameter tensor metadata differs")
            target.data.copy_(value.to(target.device))
        optimizer_state = payload.get("optimizer_state")
        if optimizer is not None and optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        completed_steps = int(payload["completed_steps"])
        if completed_steps < 0:
            raise ValueError("completed_steps must be nonnegative")
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise DistributedTrainingError(
            "checkpoint_payload_invalid",
            rank=rank,
            phase="resume",
            operation="restore_checkpoint_payload",
        ) from error
    return completed_steps


def train_distributed_statevector(
    circuit_or_ir: Any,
    *,
    steps: int,
    observable_wire: int = 0,
    optimizer: Literal["sgd", "adam"] = "sgd",
    lr: float = 0.05,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
    checkpoint_interval: int = 1,
    rematerialization_interval: int = 0,
    memory_budget_bytes: int | None = None,
    timeout_seconds: float = 120.0,
    cancelled: Callable[[], bool] | None = None,
    fault_rank: int | None = None,
    inject_collective_timeout: bool = False,
) -> ShardedTrainingResult:
    """Run ordinary PyTorch backward/optimizer steps with owner-sharded state."""

    if steps <= 0 or lr <= 0 or checkpoint_interval <= 0:
        raise ValueError("steps, lr and checkpoint_interval must be positive")
    if optimizer not in {"sgd", "adam"}:
        raise ValueError("optimizer must be 'sgd' or 'adam'")
    ir = ensure_circuit_ir(circuit_or_ir)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    backend = dist.get_backend() if dist.is_initialized() else "single_process"
    device = (
        resolve_platform_device("cuda")
        if backend == "nccl"
        else next(
            (
                value.device
                for item in ir.instructions
                for value in item.params.values()
                if isinstance(value, torch.Tensor) and value.requires_grad
            ),
            torch.device("cpu"),
        )
    )
    platform = get_platform_runtime(device.type)
    first_reverse = execute_torch_distributed_statevector_reverse(
        ir, observable_wire=observable_wire, device=device
    )
    parameters = first_reverse.parameters
    # Reverse-mode gradients are replicated after all-reduce. Optimizer owners
    # are a separate training concern and are assigned explicitly here.
    owners = tuple(index % world_size for index in range(len(parameters)))
    owned_indices = tuple(index for index, owner in enumerate(owners) if owner == rank)
    owned_parameters = [parameters[index] for index in owned_indices]
    checkpoint_contract = _checkpoint_contract(
        ir,
        observable_wire=observable_wire,
        optimizer_name=optimizer,
        lr=lr,
        parameters=parameters,
    )
    optimizer_obj: torch.optim.Optimizer | None
    if not owned_parameters:
        optimizer_obj = None
    elif optimizer == "sgd":
        optimizer_obj = torch.optim.SGD(owned_parameters, lr=lr)
    else:
        optimizer_obj = torch.optim.Adam(owned_parameters, lr=lr)

    batch_size = int(ir.shape[0]) if ir.shape else 1
    complex_bytes = 16 if any(item.dtype == torch.float64 for item in parameters) else 8
    local_state_bytes = (2**ir.n_wires // world_size) * batch_size * complex_bytes
    # Forward state, reverse adjoint/rematerialization and communication/output
    # buffers can coexist. Keep this preflight deliberately conservative.
    estimated_local_state = 4 * local_state_bytes
    parameter_bytes = sum(
        item.numel() * item.element_size() for item in owned_parameters
    )
    estimated_optimizer = parameter_bytes * (3 if optimizer == "adam" else 1)
    if (
        memory_budget_bytes is not None
        and estimated_local_state + estimated_optimizer > memory_budget_bytes
    ):
        raise DistributedTrainingError(
            "oom_preflight",
            rank=rank,
            phase="preflight",
            operation="estimate_state_and_optimizer_memory",
        )

    start_step = 0
    root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if resume:
        if root is None:
            raise ValueError("resume requires checkpoint_dir")
        checkpoint_error: DistributedTrainingError | None = None
        try:
            start_step = _load_checkpoint(
                root,
                rank=rank,
                world_size=world_size,
                optimizer_name=optimizer,
                owned_indices=owned_indices,
                parameters=parameters,
                optimizer=optimizer_obj,
                contract=checkpoint_contract,
            )
        except DistributedTrainingError as error:
            checkpoint_error = error
        if world_size > 1:
            failure_vote = torch.tensor(
                int(checkpoint_error is not None), dtype=torch.int32, device=device
            )
            dist.all_reduce(failure_vote, op=dist.ReduceOp.MAX)
            if int(failure_vote.item()):
                if checkpoint_error is not None:
                    raise checkpoint_error
                raise DistributedTrainingError(
                    "checkpoint_peer_validation_failed",
                    rank=rank,
                    phase="resume",
                    operation="distributed_checkpoint_preflight",
                )
        elif checkpoint_error is not None:
            raise checkpoint_error
        if start_step > steps:
            raise DistributedTrainingError(
                "resume_step_regression",
                rank=rank,
                phase="resume",
                operation=f"checkpoint:{start_step}>target:{steps}",
            )
        if world_size > 1:
            step_tensor = torch.tensor(start_step, dtype=torch.long, device=device)
            gathered_steps = [torch.zeros_like(step_tensor) for _ in range(world_size)]
            dist.all_gather(gathered_steps, step_tensor)
            _validate_checkpoint_generations(
                tuple(int(item.item()) for item in gathered_steps), rank=rank
            )
        _broadcast_parameters(parameters, owners)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    progress: list[TrainingProgress] = []
    losses: list[float] = []
    checkpoint_files: list[str] = []
    communication_events = 0
    communication_bytes = 0
    watchdog = PhaseAwareWatchdog(
        stall_seconds=timeout_seconds,
        bounded_phase_seconds={"checkpoint": timeout_seconds * 2},
    )
    started = time.monotonic()

    def emit(
        step: int, phase: str, operation: str, completed: int, collective: str = "none"
    ) -> None:
        if device.type == "cuda":
            platform.synchronize(device)
        now = time.monotonic()
        snapshot = ProgressSnapshot(
            timestamp=now,
            phase=phase,
            operation=operation,
            completed_units=completed,
            memory_bytes=_memory_bytes(device),
            collective=collective,
            rank=rank,
            useful_work_launched=completed > 0 or phase in {"forward", "backward"},
        )
        diagnosis = watchdog.observe(snapshot)
        progress.append(
            TrainingProgress(
                rank=rank,
                step=step,
                phase=phase,
                operation=operation,
                completed_units=completed,
                collective=collective,
                memory_bytes=snapshot.memory_bytes,
                useful_work_launched=snapshot.useful_work_launched,
                device=str(device),
                device_activity=(
                    "cuda_phase_synchronized"
                    if device.type == "cuda"
                    else "cpu_operation_completed"
                ),
                topology=f"rank:{rank}/world:{world_size}",
                timestamp=now,
            )
        )
        if diagnosis is not None:
            raise DistributedTrainingError(
                diagnosis.cause, rank=rank, phase=phase, operation=operation
            )

    for step in range(start_step, steps):
        if time.monotonic() - started > timeout_seconds:
            raise DistributedTrainingError(
                "training_timeout", rank=rank, phase="step", operation=str(step)
            )
        local_cancel = bool(cancelled and cancelled())
        cancel_tensor = torch.tensor(int(local_cancel), device=device)
        if world_size > 1:
            dist.all_reduce(cancel_tensor, op=dist.ReduceOp.MAX)
            communication_events += 1
        if int(cancel_tensor.item()):
            raise DistributedTrainingError(
                "cancelled", rank=rank, phase="step", operation=str(step)
            )
        if fault_rank is not None and step == start_step:
            _raise_coordinated_injected_failure(
                fault_rank=fault_rank,
                rank=rank,
                world_size=world_size,
                device=device,
                step=step,
            )
        if inject_collective_timeout and step == start_step:
            _raise_control_plane_timeout(
                rank=rank,
                world_size=world_size,
                timeout_seconds=timeout_seconds,
            )
        emit(step, "forward", "statevector_expectation", step)
        emit(step, "execution_segment", "validated_ir_segment", step + 1)
        emit(step, "gate_block", f"gates:{len(ir.instructions)}", step + 1)
        policy = (
            StatevectorCheckpointPolicy(
                strategy="interval", interval=rematerialization_interval
            )
            if rematerialization_interval > 0
            else StatevectorCheckpointPolicy()
        )
        reverse = execute_torch_distributed_statevector_reverse(
            ir,
            observable_wire=observable_wire,
            checkpoint_policy=policy,
            device=device,
        )
        for parameter in parameters:
            parameter.grad = None
        emit(step, "backward", "explicit_shard_adjoint", step)
        try:
            with _operation_deadline(timeout_seconds):
                reverse.backward()
        except _OperationTimeoutError as error:
            raise DistributedTrainingError(
                "backward_timeout",
                rank=rank,
                phase="backward",
                operation="explicit_shard_adjoint",
            ) from error
        emit(step, "backward_segment", "parameter_vjp_complete", step + 1)
        reverse_summary = reverse.summary()
        communication_events += int(reverse_summary["backward_communication_count"])
        communication_bytes += int(reverse_summary["backward_communication_bytes"])
        emit(step, "optimizer", f"torch.optim.{optimizer}", step)
        emit(step, "optimizer_step", "owner_local_step", step + 1)
        if optimizer_obj is not None:
            optimizer_obj.step()
        try:
            with _operation_deadline(timeout_seconds):
                broadcast_count = _broadcast_parameters(parameters, owners)
        except _OperationTimeoutError as error:
            raise DistributedTrainingError(
                "collective_timeout",
                rank=rank,
                phase="collective",
                operation="parameter_broadcast",
            ) from error
        communication_events += broadcast_count
        if world_size > 1:
            communication_bytes += sum(
                parameter.numel() * parameter.element_size() for parameter in parameters
            )
        losses.append(float(reverse.value.detach().cpu()))
        emit(step, "optimizer", "owner_broadcast_complete", step + 1, "broadcast")
        emit(step, "collective", "parameter_broadcast", step + 1, "broadcast")
        if root is not None and (step + 1) % checkpoint_interval == 0:
            emit(step, "checkpoint", "rank_local_torch_save", step + 1)
            path = _write_checkpoint(
                root,
                rank=rank,
                world_size=world_size,
                step=step + 1,
                optimizer_name=optimizer,
                owned_indices=owned_indices,
                parameters=parameters,
                optimizer=optimizer_obj,
                contract=checkpoint_contract,
            )
            checkpoint_files.append(str(path))
            if world_size > 1:
                try:
                    with _operation_deadline(timeout_seconds):
                        _failure_detection_barrier(timeout_seconds)
                except (RuntimeError, _OperationTimeoutError) as error:
                    raise DistributedTrainingError(
                        "checkpoint_barrier_timeout",
                        rank=rank,
                        phase="checkpoint",
                        operation="checkpoint_generation_barrier",
                    ) from error
                communication_events += 1
                communication_bytes += torch.empty((), dtype=torch.int64).element_size()
    emit(steps, "teardown", "training_complete", steps)
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else max((item.memory_bytes for item in progress), default=0)
    )
    ownership = tuple(
        OptimizerOwnership(
            parameter=f"parameter:{index}",
            owner_rank=owner,
            gradient_reduction="all_reduce_sum" if world_size > 1 else "local",
            optimizer_state_local=owner == rank,
            update_route=(
                "owner_step_then_broadcast" if world_size > 1 else "local_step"
            ),
        )
        for index, owner in enumerate(owners)
    )
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE") or world_size)
    local_world_size = min(world_size, max(1, local_world_size))
    node_count = (world_size + local_world_size - 1) // local_world_size
    return ShardedTrainingResult(
        losses=tuple(losses),
        completed_steps=steps,
        start_step=start_step,
        rank=rank,
        world_size=world_size,
        local_world_size=local_world_size,
        node_count=node_count,
        optimizer=optimizer,
        ownership=ownership,
        progress=tuple(progress),
        peak_memory_bytes=peak_memory,
        communication_events=communication_events,
        communication_bytes=communication_bytes,
        checkpoint_files=tuple(checkpoint_files),
    )


__all__ = (
    "DistributedTrainingError",
    "OptimizerOwnership",
    "ShardedTrainingResult",
    "TrainingProgress",
    "train_distributed_statevector",
)

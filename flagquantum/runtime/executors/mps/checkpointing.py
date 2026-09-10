"""Checkpoint durability, retention, and restore support for sharded MPS training."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import socket
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist

from ....compute import get_platform_runtime
from .errors import MPSTrainingError
from .metadata_transport import all_gather_json


def _checkpoint_path(root: Path, rank: int, step: int | None = None) -> Path:
    suffix = "" if step is None else f"-step-{step}"
    return root / f"rank-{rank}{suffix}.pt"


def _checkpoint_manifest_path(root: Path) -> Path:
    return root / "COMMITTED.json"


def _checkpoint_checksum_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commit_checkpoint_generation(
    root: Path,
    *,
    rank: int,
    world_size: int,
    step: int,
    path: Path,
    contract_fingerprint: str,
) -> Path:
    """Publish one immutable, complete rank-shard checkpoint generation."""
    records = all_gather_json(
        {
            "rank": rank,
            "completed_steps": step,
            "file": path.name,
            "sha256": _file_sha256(path),
        }
    )
    if len(records) != world_size or {int(record["rank"]) for record in records} != set(
        range(world_size)
    ):
        raise MPSTrainingError("checkpoint generation has incomplete rank membership")
    if {int(record["completed_steps"]) for record in records} != {step}:
        raise MPSTrainingError("checkpoint generations differ across ranks")
    manifest = _checkpoint_manifest_path(root)
    if rank == 0:
        temporary = manifest.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema": "sharded_mps_checkpoint_manifest_v1",
                    "world_size": world_size,
                    "completed_steps": step,
                    "contract_fingerprint": contract_fingerprint,
                    "shards": sorted(records, key=lambda record: int(record["rank"])),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(manifest)
        directory_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    dist.barrier()
    return manifest


def _validate_checkpoint_manifest(
    root: Path,
    *,
    rank: int,
    world_size: int,
    contract_fingerprint: str,
) -> tuple[int, Path]:
    manifest = json.loads(_checkpoint_manifest_path(root).read_text("utf-8"))
    if manifest.get("schema") != "sharded_mps_checkpoint_manifest_v1":
        raise ValueError("checkpoint manifest schema mismatch")
    if int(manifest.get("world_size", -1)) != world_size:
        raise ValueError("checkpoint manifest topology mismatch")
    if manifest.get("contract_fingerprint") != contract_fingerprint:
        raise ValueError("checkpoint manifest contract fingerprint mismatch")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or len(shards) != world_size:
        raise ValueError("checkpoint manifest rank membership incomplete")
    shard = next(
        (record for record in shards if int(record.get("rank", -1)) == rank),
        None,
    )
    if shard is None:
        raise ValueError(f"checkpoint manifest missing rank {rank}")
    step = int(manifest.get("completed_steps", -1))
    if int(shard.get("completed_steps", -2)) != step or step < 0:
        raise ValueError("checkpoint manifest generation mismatch")
    path = root / str(shard.get("file", ""))
    if path.parent != root or not path.is_file():
        raise ValueError(f"checkpoint shard missing for rank {rank}")
    expected = str(shard.get("sha256", ""))
    if len(expected) != 64 or _file_sha256(path) != expected:
        raise ValueError(f"checkpoint shard integrity mismatch for rank {rank}")
    return step, path


def _preflight_checkpoint_generation(
    root: Path,
    *,
    rank: int,
    world_size: int,
    contract_fingerprint: str,
) -> tuple[int, Path]:
    """Collectively reject an uncommitted, incomplete, or corrupt generation."""
    issue = ""
    step = -1
    path = _checkpoint_path(root, rank)
    try:
        step, path = _validate_checkpoint_manifest(
            root,
            rank=rank,
            world_size=world_size,
            contract_fingerprint=contract_fingerprint,
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        issue = str(error)
    reports = all_gather_json({"rank": rank, "issue": issue})
    failures = tuple(
        f"rank {report['rank']}: {report['issue']}"
        for report in reports
        if report.get("issue")
    )
    if failures:
        raise MPSTrainingError(
            "checkpoint generation preflight failed: " + "; ".join(failures)
        )
    return step, path


def _checkpoint_start_policy_error(
    *, committed_manifest_exists: bool, resume: bool, allow_overwrite: bool
) -> str | None:
    if committed_manifest_exists and not resume and not allow_overwrite:
        return (
            "checkpoint directory already contains a committed generation; "
            "set resume=True or explicitly allow checkpoint overwrite"
        )
    if resume and not committed_manifest_exists:
        return "resume requires a committed checkpoint manifest"
    return None


def _validate_checkpoint_start_policy_collective(
    root: Path,
    *,
    rank: int,
    resume: bool,
    allow_overwrite: bool,
) -> None:
    local_exists = _checkpoint_manifest_path(root).is_file()
    reports = all_gather_json({"rank": rank, "manifest_exists": local_exists})
    visibility = {bool(report["manifest_exists"]) for report in reports}
    if len(visibility) != 1:
        raise MPSTrainingError("checkpoint manifest visibility differs across ranks")
    error = _checkpoint_start_policy_error(
        committed_manifest_exists=local_exists,
        resume=resume,
        allow_overwrite=allow_overwrite,
    )
    if error is not None:
        raise MPSTrainingError(error)


def _checkpoint_writer_lease_path(root: Path) -> Path:
    return root / "ACTIVE_WRITER.json"


def _break_stale_checkpoint_writer_lease(lease: Path, stale_seconds: float) -> str:
    issue = ""
    try:
        existing = json.loads(lease.read_text(encoding="utf-8"))
        heartbeat = float(
            existing.get(
                "heartbeat_unix_seconds",
                existing.get("created_unix_seconds", 0.0),
            )
        )
        age = max(0.0, time.time() - heartbeat)
        if age < stale_seconds:
            issue = (
                "checkpoint writer lease is not stale: "
                f"age_seconds={age:.3f}, stale_seconds={stale_seconds:.3f}"
            )
        elif existing.get("hostname") == socket.gethostname():
            pid = int(existing.get("pid", -1))
            if pid <= 0:
                issue = "checkpoint writer lease contains an invalid PID"
            else:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    issue = "checkpoint writer lease PID is still active"
                else:
                    issue = "checkpoint writer lease PID is still active"
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        age = max(0.0, time.time() - lease.stat().st_mtime)
        if age < stale_seconds:
            issue = (
                "invalid checkpoint writer lease is not old enough to break: "
                f"age_seconds={age:.3f}, stale_seconds={stale_seconds:.3f}"
            )
    if not issue:
        lease.unlink(missing_ok=True)
    return issue


def _write_checkpoint_writer_lease(
    lease: Path,
    *,
    world_size: int,
    contract_fingerprint: str,
) -> str:
    try:
        descriptor = os.open(
            lease,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        now = time.time()
        payload = (
            json.dumps(
                {
                    "schema": "sharded_mps_checkpoint_writer_lease_v1",
                    "contract_fingerprint": contract_fingerprint,
                    "world_size": world_size,
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "created_unix_seconds": now,
                    "heartbeat_unix_seconds": now,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except FileExistsError:
        return (
            "checkpoint directory has an active writer lease; inspect "
            "ACTIVE_WRITER.json before explicitly breaking a stale lease"
        )
    except OSError as error:
        return f"checkpoint writer lease acquisition failed: {error}"
    return ""


def _acquire_checkpoint_writer_lease(
    root: Path,
    *,
    rank: int,
    world_size: int,
    contract_fingerprint: str,
    allow_break: bool,
    stale_seconds: float,
) -> None:
    issue = ""
    if rank == 0:
        lease = _checkpoint_writer_lease_path(root)
        if allow_break and lease.exists():
            issue = _break_stale_checkpoint_writer_lease(lease, stale_seconds)
        if not issue:
            issue = _write_checkpoint_writer_lease(
                lease,
                world_size=world_size,
                contract_fingerprint=contract_fingerprint,
            )
    reports = all_gather_json({"rank": rank, "issue": issue})
    failures = tuple(str(report["issue"]) for report in reports if report.get("issue"))
    if failures:
        raise MPSTrainingError("; ".join(failures))
    dist.barrier()


def _refresh_checkpoint_writer_lease(
    root: Path,
    *,
    rank: int,
    contract_fingerprint: str,
) -> None:
    issue = ""
    if rank == 0:
        lease = _checkpoint_writer_lease_path(root)
        try:
            payload = json.loads(lease.read_text(encoding="utf-8"))
            if payload.get("schema") != "sharded_mps_checkpoint_writer_lease_v1":
                raise ValueError("checkpoint writer lease schema mismatch")
            if payload.get("contract_fingerprint") != contract_fingerprint:
                raise ValueError("checkpoint writer lease contract mismatch")
            payload["heartbeat_unix_seconds"] = time.time()
            temporary = lease.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            temporary.replace(lease)
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as error:
            issue = f"checkpoint writer lease heartbeat failed: {error}"
    reports = all_gather_json({"rank": rank, "issue": issue})
    failures = tuple(str(report["issue"]) for report in reports if report.get("issue"))
    if failures:
        raise MPSTrainingError("; ".join(failures))


def _release_checkpoint_writer_lease(root: Path, *, rank: int) -> None:
    if rank == 0:
        _checkpoint_writer_lease_path(root).unlink(missing_ok=True)
    dist.barrier()


def _start_checkpoint_session(
    checkpoint_dir: str | Path | None,
    *,
    rank: int,
    world_size: int,
    contract_fingerprint: str,
    resume: bool,
    allow_overwrite: bool,
    allow_lease_break: bool,
    lease_stale_seconds: float,
) -> tuple[Path | None, float]:
    if checkpoint_dir is None:
        return None, 0.0
    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    storage_probe_started = time.perf_counter()
    _validate_shared_checkpoint_root(
        root,
        rank=rank,
        world_size=world_size,
        contract_fingerprint=contract_fingerprint,
    )
    storage_probe_seconds = time.perf_counter() - storage_probe_started
    _validate_checkpoint_start_policy_collective(
        root,
        rank=rank,
        resume=resume,
        allow_overwrite=allow_overwrite,
    )
    _acquire_checkpoint_writer_lease(
        root,
        rank=rank,
        world_size=world_size,
        contract_fingerprint=contract_fingerprint,
        allow_break=allow_lease_break,
        stale_seconds=lease_stale_seconds,
    )
    return root, storage_probe_seconds


def _prune_checkpoint_generations(
    root: Path,
    *,
    rank: int,
    committed_step: int,
    keep_generations: int | None,
) -> tuple[str, ...]:
    """Bound rank-local checkpoint storage without touching the committed shard."""
    if keep_generations is None:
        return ()
    candidates: list[tuple[int, Path]] = []
    prefix = f"rank-{rank}-step-"
    for path in root.glob(f"{prefix}*.pt"):
        try:
            step = int(path.stem.removeprefix(prefix))
        except ValueError:
            continue
        candidates.append((step, path))
    retained_steps = {
        step
        for step, _ in sorted(
            (candidate for candidate in candidates if candidate[0] <= committed_step),
            reverse=True,
        )[:keep_generations]
    }
    retained_steps.add(committed_step)
    deleted: list[str] = []
    for step, path in candidates:
        if step in retained_steps:
            continue
        checksum = _checkpoint_checksum_path(path)
        path.unlink(missing_ok=True)
        checksum.unlink(missing_ok=True)
        deleted.extend((str(path), str(checksum)))
    return tuple(deleted)


def _prune_checkpoint_generations_collective(
    root: Path,
    *,
    rank: int,
    committed_step: int,
    keep_generations: int | None,
) -> tuple[str, ...]:
    deleted: tuple[str, ...] = ()
    issue = ""
    try:
        deleted = _prune_checkpoint_generations(
            root,
            rank=rank,
            committed_step=committed_step,
            keep_generations=keep_generations,
        )
    except OSError as error:
        issue = str(error)
    reports = all_gather_json({"rank": rank, "issue": issue})
    failures = tuple(
        f"rank {report['rank']}: {report['issue']}"
        for report in reports
        if report.get("issue")
    )
    if failures:
        raise MPSTrainingError(
            "checkpoint retention cleanup failed: " + "; ".join(failures)
        )
    return deleted


def _tensor_bytes_in(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(_tensor_bytes_in(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes_in(item) for item in value)
    return 0


def _checkpoint_capacity_error(
    *,
    minimum_free_bytes: int,
    estimated_generation_bytes: int,
    reserve_bytes: int,
) -> str | None:
    required = estimated_generation_bytes + reserve_bytes
    if minimum_free_bytes >= required:
        return None
    return (
        "checkpoint storage capacity preflight failed: "
        f"minimum_free_bytes={minimum_free_bytes}, "
        f"estimated_generation_bytes={estimated_generation_bytes}, "
        f"reserve_bytes={reserve_bytes}, required_bytes={required}"
    )


def _checkpoint_storage_preflight(
    root: Path,
    *,
    rank: int,
    owned_indices: tuple[int, ...],
    parameters: tuple[torch.Tensor, ...],
    optimizer: torch.optim.Optimizer | None,
    reserve_bytes: int,
) -> tuple[int, int]:
    local_estimate = 1 << 20
    local_estimate += sum(
        parameters[index].numel() * parameters[index].element_size()
        for index in owned_indices
    )
    if optimizer is not None:
        local_estimate += _tensor_bytes_in(optimizer.state_dict())
    local_free = int(shutil.disk_usage(root).free)
    reports = all_gather_json(
        {"rank": rank, "free_bytes": local_free, "estimated_bytes": local_estimate}
    )
    minimum_free = min(int(report["free_bytes"]) for report in reports)
    generation_estimate = sum(int(report["estimated_bytes"]) for report in reports)
    error = _checkpoint_capacity_error(
        minimum_free_bytes=minimum_free,
        estimated_generation_bytes=generation_estimate,
        reserve_bytes=reserve_bytes,
    )
    if error is not None:
        raise MPSTrainingError(error)
    return minimum_free, generation_estimate


def _validate_shared_checkpoint_root(
    root: Path,
    *,
    rank: int,
    world_size: int,
    contract_fingerprint: str,
) -> None:
    """Prove every rank observes the same checkpoint namespace before I/O."""
    if world_size == 1:
        return
    probe = root / ".fq-mps-shared-root-probe"
    expected = f"{contract_fingerprint}:{world_size}\n"
    if rank == 0:
        temporary = probe.with_suffix(".tmp")
        temporary.write_text(expected, encoding="ascii")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(probe)
        directory_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    dist.barrier()
    issue = ""
    try:
        observed = probe.read_text(encoding="ascii")
        if observed != expected:
            issue = "checkpoint root probe content mismatch"
    except (OSError, UnicodeError) as error:
        issue = f"checkpoint root probe unavailable: {error}"
    reports = all_gather_json({"rank": rank, "issue": issue})
    if rank == 0:
        probe.unlink(missing_ok=True)
    dist.barrier()
    failures = tuple(
        f"rank {report['rank']}: {report['issue']}"
        for report in reports
        if report.get("issue")
    )
    if failures:
        raise MPSTrainingError(
            "checkpoint shared storage validation failed: " + "; ".join(failures)
        )


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
    path = _checkpoint_path(root, rank, step)
    temporary = path.with_suffix(".pt.tmp")
    checksum_path = _checkpoint_checksum_path(path)
    checksum_temporary = checksum_path.with_suffix(checksum_path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": "sharded_mps_training_checkpoint_v2",
            "rank": rank,
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
                get_platform_runtime("cuda").rng_state(parameters[0].device)
                if parameters[0].device.type == "cuda"
                else None
            ),
        },
        temporary,
    )
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


def _load_verified_checkpoint(
    path: Path,
    *,
    rank: int,
    device: torch.device,
) -> Mapping[str, Any]:
    if not path.exists():
        raise MPSTrainingError(f"checkpoint missing for rank {rank}: {path}")
    checksum_path = _checkpoint_checksum_path(path)
    if not checksum_path.exists():
        raise MPSTrainingError(
            f"checkpoint integrity metadata missing for rank {rank}: {checksum_path}"
        )
    try:
        expected_checksum = checksum_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise MPSTrainingError(
            f"checkpoint integrity metadata invalid for rank {rank}: {checksum_path}"
        ) from error
    if len(expected_checksum) != 64 or _file_sha256(path) != expected_checksum:
        raise MPSTrainingError(f"checkpoint integrity mismatch for rank {rank}: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except (
        OSError,
        RuntimeError,
        ValueError,
        EOFError,
        pickle.UnpicklingError,
    ) as error:
        raise MPSTrainingError(
            f"checkpoint deserialization failed for rank {rank}: {path}"
        ) from error
    if not isinstance(payload, Mapping):
        raise MPSTrainingError("checkpoint schema or rank identity mismatch")
    return payload


def _validate_checkpoint_identity(
    payload: Mapping[str, Any],
    *,
    rank: int,
    world_size: int,
    owned_indices: tuple[int, ...],
    contract: Mapping[str, Any],
    bond_layout: Mapping[int, tuple[int, ...]],
) -> None:
    if (
        payload.get("schema_version") != "sharded_mps_training_checkpoint_v2"
        or payload.get("rank") != rank
    ):
        raise MPSTrainingError("checkpoint schema or rank identity mismatch")
    if (
        payload.get("contract") != dict(contract)
        or payload.get("world_size") != world_size
    ):
        raise MPSTrainingError("checkpoint contract or topology mismatch")
    if tuple(payload.get("owned_indices", ())) != owned_indices:
        raise MPSTrainingError("checkpoint optimizer ownership mismatch")
    if payload.get("bond_layout") != dict(bond_layout):
        raise MPSTrainingError("checkpoint MPS bond layout mismatch")


def _validated_checkpoint_parameters(
    payload: Mapping[str, Any],
    *,
    owned_indices: tuple[int, ...],
    parameters: tuple[torch.Tensor, ...],
) -> list[tuple[int, torch.Tensor]]:
    saved_parameters = payload.get("parameters")
    if not isinstance(saved_parameters, Mapping):
        raise MPSTrainingError("checkpoint parameter payload must be a mapping")
    try:
        saved_indices = {int(index) for index in saved_parameters}
    except (TypeError, ValueError, OverflowError) as error:
        raise MPSTrainingError("checkpoint parameter indices are invalid") from error
    if len(saved_indices) != len(saved_parameters):
        raise MPSTrainingError("checkpoint parameter indices are duplicated")
    if saved_indices != set(owned_indices):
        raise MPSTrainingError("checkpoint parameter shard membership mismatch")
    validated_parameters: list[tuple[int, torch.Tensor]] = []
    for raw_index, value in saved_parameters.items():
        index = int(raw_index)
        target = parameters[index]
        if not isinstance(value, torch.Tensor):
            raise MPSTrainingError("checkpoint parameter value must be a tensor")
        if tuple(value.shape) != tuple(target.shape) or value.dtype != target.dtype:
            raise MPSTrainingError(
                "checkpoint parameter shape or dtype mismatch: "
                f"index={index}, saved_shape={tuple(value.shape)}, "
                f"target_shape={tuple(target.shape)}, saved_dtype={value.dtype}, "
                f"target_dtype={target.dtype}"
            )
        validated_parameters.append((index, value))
    return validated_parameters


def _validated_checkpoint_state(
    payload: Mapping[str, Any],
    *,
    optimizer: torch.optim.Optimizer | None,
    expected_completed_steps: int | None,
) -> tuple[Mapping[str, Any] | None, torch.Tensor, torch.Tensor | None, int]:
    optimizer_state = payload.get("optimizer_state")
    if optimizer is None and optimizer_state is not None:
        raise MPSTrainingError("checkpoint contains unexpected optimizer state")
    if optimizer is not None and not isinstance(optimizer_state, Mapping):
        raise MPSTrainingError("checkpoint optimizer state must be a mapping")
    torch_rng_state = payload.get("torch_rng_state")
    if (
        not isinstance(torch_rng_state, torch.Tensor)
        or torch_rng_state.dtype != torch.uint8
    ):
        raise MPSTrainingError("checkpoint CPU RNG state is invalid")
    cuda_rng_state = payload.get("cuda_rng_state")
    if cuda_rng_state is not None and (
        not isinstance(cuda_rng_state, torch.Tensor)
        or cuda_rng_state.dtype != torch.uint8
    ):
        raise MPSTrainingError("checkpoint CUDA RNG state is invalid")
    completed_steps = payload.get("completed_steps")
    if type(completed_steps) is not int or completed_steps < 0:
        raise MPSTrainingError("checkpoint completed step is invalid")
    if (
        expected_completed_steps is not None
        and completed_steps != expected_completed_steps
    ):
        raise MPSTrainingError("checkpoint payload and manifest generation mismatch")
    return optimizer_state, torch_rng_state, cuda_rng_state, completed_steps


def _restore_checkpoint_state(
    *,
    parameters: tuple[torch.Tensor, ...],
    validated_parameters: list[tuple[int, torch.Tensor]],
    optimizer: torch.optim.Optimizer | None,
    optimizer_state: Mapping[str, Any] | None,
    torch_rng_state: torch.Tensor,
    cuda_rng_state: torch.Tensor | None,
) -> None:

    try:
        if optimizer is not None and optimizer_state is not None:
            optimizer.load_state_dict(dict(optimizer_state))
        for index, value in validated_parameters:
            parameters[index].data.copy_(value.to(parameters[index].device))
        torch.set_rng_state(torch_rng_state.cpu())
    except (RuntimeError, ValueError, KeyError, TypeError) as error:
        raise MPSTrainingError("checkpoint state restoration failed") from error
    if parameters[0].device.type == "cuda" and cuda_rng_state is not None:
        get_platform_runtime("cuda").restore_rng_state(
            parameters[0].device, cuda_rng_state.cpu()
        )


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
    checkpoint_path: Path | None = None,
    expected_completed_steps: int | None = None,
) -> int:
    path = checkpoint_path or _checkpoint_path(root, rank)
    payload = _load_verified_checkpoint(path, rank=rank, device=parameters[0].device)
    _validate_checkpoint_identity(
        payload,
        rank=rank,
        world_size=world_size,
        owned_indices=owned_indices,
        contract=contract,
        bond_layout=bond_layout,
    )
    validated_parameters = _validated_checkpoint_parameters(
        payload,
        owned_indices=owned_indices,
        parameters=parameters,
    )
    optimizer_state, torch_rng_state, cuda_rng_state, completed_steps = (
        _validated_checkpoint_state(
            payload,
            optimizer=optimizer,
            expected_completed_steps=expected_completed_steps,
        )
    )
    _restore_checkpoint_state(
        parameters=parameters,
        validated_parameters=validated_parameters,
        optimizer=optimizer,
        optimizer_state=optimizer_state,
        torch_rng_state=torch_rng_state,
        cuda_rng_state=cuda_rng_state,
    )
    return completed_steps


def _restore_training_checkpoint(
    root: Path,
    *,
    rank: int,
    world_size: int,
    steps: int,
    device: torch.device,
    contract_fingerprint: str,
    contract: Mapping[str, Any],
    bond_layout: Mapping[int, tuple[int, ...]],
    owned_indices: tuple[int, ...],
    parameters: tuple[torch.Tensor, ...],
    optimizer: torch.optim.Optimizer | None,
) -> int:
    committed_step, committed_path = _preflight_checkpoint_generation(
        root,
        rank=rank,
        world_size=world_size,
        contract_fingerprint=contract_fingerprint,
    )
    start_step = _load_checkpoint(
        root,
        rank=rank,
        world_size=world_size,
        owned_indices=owned_indices,
        parameters=parameters,
        optimizer=optimizer,
        contract=contract,
        bond_layout=bond_layout,
        checkpoint_path=committed_path,
        expected_completed_steps=committed_step,
    )
    if start_step > steps:
        raise MPSTrainingError(
            f"checkpoint completed step {start_step} exceeds target {steps}"
        )
    completed = torch.tensor(start_step, device=device)
    gathered = [torch.zeros_like(completed) for _ in range(world_size)]
    dist.all_gather(gathered, completed)
    if len({int(value.item()) for value in gathered}) != 1:
        raise MPSTrainingError("checkpoint generations differ across ranks")
    return int(start_step)

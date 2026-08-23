"""Durable rank-local checkpoints for dynamic tensor-network reverse execution."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from .dynamic_reverse import (
    DistributedTNDynamicReverseSegment,
    DistributedTNDynamicReverseSegmentResult,
)

TN_DYNAMIC_CHECKPOINT_VERSION = "flagquantum.distributed_tn_checkpoint.v2"


@dataclass(frozen=True)
class DistributedTNCheckpointAudit:
    """Read-only durability and writer-lock state for operator decisions."""

    directory: str
    manifest_exists: bool
    manifest_identity_valid: bool
    committed_exists: bool
    commit_identity_matches: bool
    expected_rank_file_count: int
    present_rank_file_count: int
    rank_file_integrity_valid: bool
    writer_lock_exists: bool
    writer_lock_identity: str | None
    writer_pid: int | None
    writer_segment_identity: str | None
    writer_next_record_index: int | None

    @property
    def loadable(self) -> bool:
        return (
            self.manifest_exists
            and self.manifest_identity_valid
            and self.committed_exists
            and self.commit_identity_matches
            and self.expected_rank_file_count == self.present_rank_file_count
            and self.rank_file_integrity_valid
        )


def inspect_dynamic_tn_checkpoint(
    directory: str | os.PathLike[str],
) -> DistributedTNCheckpointAudit:
    """Inspect checkpoint state without mutating files or requiring a group."""

    root = Path(directory)
    manifest_exists = (root / "manifest.json").is_file()
    manifest: dict[str, Any] = {}
    manifest_identity_valid = False
    if manifest_exists:
        try:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            identity = str(manifest.get("checkpoint_identity", ""))
            body = {
                key: value
                for key, value in manifest.items()
                if key != "checkpoint_identity"
            }
            manifest_identity_valid = bool(identity) and (
                identity == _json_identity(body)
            )
        except (OSError, json.JSONDecodeError):
            manifest = {}
    committed_exists = (root / "COMMITTED").is_file()
    committed_identity = ""
    if committed_exists:
        try:
            committed_identity = (
                (root / "COMMITTED").read_text(encoding="utf-8").strip()
            )
        except OSError:
            committed_exists = False
    rank_entries = manifest.get("rank_files", ())
    expected_files = len(rank_entries)
    present_files = 0
    rank_file_integrity_valid = manifest_identity_valid
    for rank, entry in enumerate(rank_entries):
        candidate = root / f"rank-{rank:05d}.pt"
        present_files += int(candidate.is_file())
        try:
            rank_file_integrity_valid = rank_file_integrity_valid and (
                entry.get("rank") == rank
                and entry.get("filename") == candidate.name
                and candidate.stat().st_size == int(entry["bytes"])
                and _file_sha256(candidate) == str(entry["sha256"])
            )
        except (KeyError, OSError, TypeError, ValueError):
            rank_file_integrity_valid = False
    lock_path = root / ".checkpoint.lock"
    lock_exists = lock_path.is_file()
    lock_identity = None
    lock_payload: dict[str, Any] = {}
    if lock_exists:
        try:
            lock_bytes = lock_path.read_bytes()
            lock_identity = hashlib.sha256(lock_bytes).hexdigest()
            lock_payload = json.loads(lock_bytes)
        except (OSError, json.JSONDecodeError):
            lock_payload = {}
    return DistributedTNCheckpointAudit(
        directory=str(root),
        manifest_exists=manifest_exists,
        manifest_identity_valid=manifest_identity_valid,
        committed_exists=committed_exists,
        commit_identity_matches=(
            manifest_identity_valid
            and committed_identity == manifest.get("checkpoint_identity")
        ),
        expected_rank_file_count=expected_files,
        present_rank_file_count=present_files,
        rank_file_integrity_valid=rank_file_integrity_valid,
        writer_lock_exists=lock_exists,
        writer_lock_identity=lock_identity,
        writer_pid=_optional_int(lock_payload.get("pid")),
        writer_segment_identity=lock_payload.get("segment_identity"),
        writer_next_record_index=_optional_int(lock_payload.get("next_record_index")),
    )


def clear_dynamic_tn_checkpoint_writer_lock(
    directory: str | os.PathLike[str],
    *,
    expected_lock_identity: str,
) -> None:
    """Clear exactly the audited lock, refusing replacement races."""

    root = Path(directory)
    lock_path = root / ".checkpoint.lock"
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags)
    try:
        opened_stat = os.fstat(descriptor)
        payload = b""
        while True:
            chunk = os.read(descriptor, 4096)
            if not chunk:
                break
            payload += chunk
        if hashlib.sha256(payload).hexdigest() != expected_lock_identity:
            raise RuntimeError("dynamic TN checkpoint writer lock identity changed")
        current_stat = os.stat(lock_path, follow_symlinks=False)
        if (
            current_stat.st_dev != opened_stat.st_dev
            or current_stat.st_ino != opened_stat.st_ino
        ):
            raise RuntimeError("dynamic TN checkpoint writer lock was replaced")
        lock_path.unlink()
    finally:
        os.close(descriptor)
    directory_fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def save_dynamic_tn_reverse_checkpoint(
    result: DistributedTNDynamicReverseSegmentResult,
    segment: DistributedTNDynamicReverseSegment,
    directory: str | os.PathLike[str],
) -> Path:
    """Atomically save one file per rank and a hash-verified shared manifest."""

    if not dist.is_initialized():
        raise RuntimeError("dynamic TN checkpoint save requires torch.distributed")
    if result.segment_identity != segment.identity or result.completed:
        raise ValueError("only an incomplete matching dynamic segment can be saved")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = next(iter(result.cotangents.values())).device
    lock_acquired = False
    if rank == 0:
        lock_acquired = _acquire_checkpoint_lock(root, segment, result)
    lock_status = torch.tensor(int(lock_acquired), dtype=torch.uint8, device=device)
    dist.broadcast(lock_status, src=0)
    if not bool(lock_status.item()):
        raise RuntimeError("dynamic TN checkpoint writer lock is already held")
    if rank == 0:
        committed = root / "COMMITTED"
        if committed.exists():
            committed.unlink()
    dist.barrier()
    rank_path = root / f"rank-{rank:05d}.pt"
    temporary = root / f".rank-{rank:05d}.pt.tmp-{os.getpid()}"
    torch.save(_checkpoint_payload(result), temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, rank_path)
    dist.barrier()
    if rank == 0:
        rank_files = []
        for candidate_rank in range(world_size):
            candidate = root / f"rank-{candidate_rank:05d}.pt"
            rank_files.append(
                {
                    "rank": candidate_rank,
                    "filename": candidate.name,
                    "sha256": _file_sha256(candidate),
                    "bytes": candidate.stat().st_size,
                }
            )
        manifest_body = {
            "version": TN_DYNAMIC_CHECKPOINT_VERSION,
            "segment_identity": segment.identity,
            "world_size": world_size,
            "next_record_index": result.next_record_index,
            "rank_files": rank_files,
        }
        checkpoint_identity = _json_identity(manifest_body)
        manifest = manifest_body | {
            "checkpoint_identity": checkpoint_identity,
        }
        _atomic_write_text(
            root / "manifest.json",
            json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        )
        _atomic_write_text(
            root / "COMMITTED",
            checkpoint_identity + "\n",
        )
    dist.barrier()
    if rank == 0:
        _release_checkpoint_lock(root)
    dist.barrier()
    return root / "manifest.json"


def load_dynamic_tn_reverse_checkpoint(
    segment: DistributedTNDynamicReverseSegment,
    directory: str | os.PathLike[str],
    *,
    device: torch.device,
) -> DistributedTNDynamicReverseSegmentResult:
    """Load and verify the current rank's durable reverse checkpoint."""

    if not dist.is_initialized():
        raise RuntimeError("dynamic TN checkpoint load requires torch.distributed")
    root = Path(directory)
    manifest_invalid = False
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        committed_identity = (root / "COMMITTED").read_text(encoding="utf-8").strip()
    except (OSError, json.JSONDecodeError):
        manifest = {}
        committed_identity = ""
        manifest_invalid = True
    if _collective_invalid(manifest_invalid, device):
        raise RuntimeError("dynamic TN checkpoint is not durably committed")
    checkpoint_identity = str(manifest.get("checkpoint_identity", ""))
    manifest_body = {
        key: value for key, value in manifest.items() if key != "checkpoint_identity"
    }
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if (
        manifest.get("version") != TN_DYNAMIC_CHECKPOINT_VERSION
        or manifest.get("segment_identity") != segment.identity
        or manifest.get("world_size") != world_size
        or checkpoint_identity != _json_identity(manifest_body)
        or committed_identity != checkpoint_identity
    ):
        raise RuntimeError("dynamic TN checkpoint manifest is incompatible")
    entries = {int(entry["rank"]): entry for entry in manifest.get("rank_files", ())}
    if tuple(sorted(entries)) != tuple(range(world_size)) or any(
        entry.get("filename") != f"rank-{entry_rank:05d}.pt"
        for entry_rank, entry in entries.items()
    ):
        raise RuntimeError("dynamic TN checkpoint rank manifest is invalid")
    entry = entries.get(rank)
    invalid = entry is None
    rank_path = root / (
        f"rank-{rank:05d}.pt" if entry is None else str(entry["filename"])
    )
    if not invalid:
        try:
            invalid = rank_path.stat().st_size != int(entry["bytes"]) or _file_sha256(
                rank_path
            ) != str(entry["sha256"])
        except OSError:
            invalid = True
    if _collective_invalid(invalid, device):
        raise RuntimeError("dynamic TN checkpoint rank file failed integrity check")
    payload = torch.load(rank_path, map_location=device, weights_only=True)
    result = _result_from_payload(payload)
    incompatible = (
        result.segment_identity != segment.identity
        or result.completed
        or result.next_record_index != int(manifest["next_record_index"])
    )
    if _collective_invalid(incompatible, device):
        raise RuntimeError("dynamic TN checkpoint payload is incompatible")
    return result


def _checkpoint_payload(
    result: DistributedTNDynamicReverseSegmentResult,
) -> dict[str, Any]:
    return {
        "version": TN_DYNAMIC_CHECKPOINT_VERSION,
        "cotangents": dict(result.cotangents),
        "segment_identity": result.segment_identity,
        "executed_reverse_ids": result.executed_reverse_ids,
        "subgroup_collective_count": result.subgroup_collective_count,
        "subgroup_collective_bytes": result.subgroup_collective_bytes,
        "redistribution_count": result.redistribution_count,
        "redistribution_sent_bytes": result.redistribution_sent_bytes,
        "redistribution_received_bytes": result.redistribution_received_bytes,
        "accumulated_cotangent_count": result.accumulated_cotangent_count,
        "released_forward_value_count": result.released_forward_value_count,
        "peak_cached_forward_bytes": result.peak_cached_forward_bytes,
        "ready_cotangent_ids": result.ready_cotangent_ids,
        "pending_cotangent_contributions": dict(result.pending_cotangent_contributions),
        "rank_consensus_validated": result.rank_consensus_validated,
        "rank_tensor_preflight_validated": (result.rank_tensor_preflight_validated),
        "completed": result.completed,
        "next_record_index": result.next_record_index,
    }


def _result_from_payload(
    payload: dict[str, Any],
) -> DistributedTNDynamicReverseSegmentResult:
    if payload.pop("version", None) != TN_DYNAMIC_CHECKPOINT_VERSION:
        raise RuntimeError("unsupported dynamic TN checkpoint payload version")
    return DistributedTNDynamicReverseSegmentResult(**payload)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collective_invalid(invalid: bool, device: torch.device) -> bool:
    status = torch.tensor(int(invalid), dtype=torch.uint8, device=device)
    dist.all_reduce(status, op=dist.ReduceOp.MAX)
    return bool(status.item())


def _json_identity(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _acquire_checkpoint_lock(
    root: Path,
    segment: DistributedTNDynamicReverseSegment,
    result: DistributedTNDynamicReverseSegmentResult,
) -> bool:
    lock_path = root / ".checkpoint.lock"
    try:
        descriptor = os.open(
            lock_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        return False
    payload = json.dumps(
        {
            "pid": os.getpid(),
            "segment_identity": segment.identity,
            "next_record_index": result.next_record_index,
        },
        sort_keys=True,
    ).encode()
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return True


def _release_checkpoint_lock(root: Path) -> None:
    (root / ".checkpoint.lock").unlink()
    directory_fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


__all__ = (
    "DistributedTNCheckpointAudit",
    "clear_dynamic_tn_checkpoint_writer_lock",
    "inspect_dynamic_tn_checkpoint",
    "load_dynamic_tn_reverse_checkpoint",
    "save_dynamic_tn_reverse_checkpoint",
)

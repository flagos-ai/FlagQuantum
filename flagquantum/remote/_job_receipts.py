"""Local request exclusion and receipt binding; never retry remote submissions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from ..core.ir import CircuitIR, ensure_circuit_ir
from ..observables import OutputRequest, lower_outputs

if TYPE_CHECKING:
    from ..circuit import Circuit
    from .jobs import RemoteJob

_SCHEMA = "flagquantum.remote-request.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class _Journal(TypedDict):
    schema: str
    request_sha256: str
    receipt_sha256: str | None


class _SubmissionSettings(TypedDict):
    target: str
    shots: int | None
    devices: int | None
    compiler: str | None
    target_qubits: Sequence[int] | None
    name: str | None
    image: str | None
    workspace: str | None
    project: str | None
    queue: str | None


def request_digest(
    program: Circuit | CircuitIR | Sequence[Circuit | CircuitIR],
    *,
    outputs: OutputRequest | Sequence[OutputRequest] | None,
    shots: int | None,
    settings: _SubmissionSettings,
) -> str:
    if isinstance(program, Sequence) and not isinstance(program, (str, bytes)):
        collection = True
        programs = tuple(program)
    else:
        collection = False
        programs = (program,)
    if not programs:
        raise ValueError("program collection must not be empty")
    hashes = []
    for item in programs:
        ir = ensure_circuit_ir(item)
        measurements = lower_outputs(outputs, n_qubits=ir.n_wires, shots=shots)
        if measurements is not None:
            ir = replace(ir, measurements=measurements)
        hashes.append(ir.content_hash)
    encoded = json.dumps(
        [collection, hashes, settings],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _receipt_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_journal(path: Path) -> _Journal:
    with path.open(encoding="utf-8") as stream:
        text = stream.read(4097)
    if len(text) > 4096:
        raise ValueError("Remote request journal exceeds its size limit")
    raw = json.loads(text)
    if (
        not isinstance(raw, dict)
        or set(raw) != {"schema", "request_sha256", "receipt_sha256"}
        or raw["schema"] != _SCHEMA
        or not isinstance(raw["request_sha256"], str)
        or _SHA256.fullmatch(raw["request_sha256"]) is None
        or (
            raw["receipt_sha256"] is not None
            and (
                not isinstance(raw["receipt_sha256"], str)
                or _SHA256.fullmatch(raw["receipt_sha256"]) is None
            )
        )
    ):
        raise ValueError("Invalid remote request journal; no resubmission")
    return _Journal(
        schema=raw["schema"],
        request_sha256=raw["request_sha256"],
        receipt_sha256=raw["receipt_sha256"],
    )


def _reserve(path: Path, journal: _Journal) -> None:
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            "Another attempt owns this receipt; no resubmission"
        ) from exc
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(journal, stream)
        stream.flush()
        os.fsync(stream.fileno())


def _complete(path: Path, journal: _Journal) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(journal, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def resume_or_submit(
    path: str | Path,
    *,
    identity: str,
    target: str,
    submitter: Callable[[], RemoteJob],
    restore: Callable[[Path], RemoteJob],
) -> RemoteJob:
    if not isinstance(path, (str, Path)):
        raise TypeError("receipt must be a string or Path")
    if isinstance(path, str) and not path.strip():
        raise ValueError("receipt must name a file")
    receipt = Path(path).absolute()
    journal_path = Path(str(receipt) + ".request.json")
    if receipt.is_symlink() or journal_path.is_symlink():
        raise ValueError("Receipt and request journal must not be symbolic links")
    if journal_path.exists():
        saved = _read_journal(journal_path)
        if saved["request_sha256"] != identity:
            raise ValueError(
                "Submission configuration changed; use a new receipt for a new experiment"
            )
        if not receipt.exists() or saved["receipt_sha256"] is None:
            raise RuntimeError(
                "Submission may already exist; reconcile with the provider before retrying"
            )
        if _receipt_digest(receipt) != saved["receipt_sha256"]:
            raise ValueError("Receipt content changed; no resubmission")
        job = restore(receipt)
        if job.target != target:
            raise ValueError("Restored receipt target differs from the request")
        return job
    if receipt.exists():
        raise ValueError(
            "Receipt has no request journal; use fq.restore_job explicitly"
        )
    journal = _Journal(schema=_SCHEMA, request_sha256=identity, receipt_sha256=None)
    _reserve(journal_path, journal)
    job = submitter()
    try:
        job.save(receipt)
        journal["receipt_sha256"] = _receipt_digest(receipt)
        _complete(journal_path, journal)
    except OSError as exc:
        raise OSError(
            f"Job {job.id} was submitted but its recovery record could not be completed; do not resubmit: {exc}"
        ) from exc
    return job

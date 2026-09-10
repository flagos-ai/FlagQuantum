"""Workspace-managed artifacts for Jiuding program jobs."""

from __future__ import annotations

import hashlib
import io
import json
import shlex
import subprocess
import tarfile
import tempfile
import uuid
from pathlib import Path
from typing import Any

from ._program_job import write_program_bundle

_MANAGED_ROOT = "/share/project/.flagquantum"
_REMOTE_PYTHON = "/opt/conda/bin/python"
_MAX_SOURCE_BYTES = 64 * 1024 * 1024
_MAX_RESULT_BYTES = 64 * 1024 * 1024
_IGNORED_PARTS = {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}


def _source_files() -> tuple[tuple[Path, str], ...]:
    package = Path(__file__).resolve().parents[2]
    files = []
    for path in package.rglob("*"):
        if (
            path.is_file()
            and not path.is_symlink()
            and not any(part in _IGNORED_PARTS for part in path.parts)
            and path.suffix != ".pyc"
        ):
            files.append((path, str(Path("flagquantum") / path.relative_to(package))))
    return tuple(sorted(files, key=lambda item: item[1]))


def _source_archive() -> tuple[str, bytes]:
    digest = hashlib.sha256()
    stream = io.BytesIO()
    with tarfile.open(
        fileobj=stream, mode="w:gz", format=tarfile.PAX_FORMAT
    ) as archive:
        for path, name in _source_files():
            data = path.read_bytes()
            digest.update(name.encode("utf-8") + b"\0" + data)
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
    payload = stream.getvalue()
    if len(payload) > _MAX_SOURCE_BYTES:
        raise RuntimeError("FlagQuantum source snapshot exceeds the 64 MiB limit")
    return digest.hexdigest()[:20], payload


def _run_ssh(
    client: Any, arguments: list[str], **options: Any
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [*client._ssh_base(), shlex.join(arguments)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=options.pop("timeout", 30),
        check=False,
        **options,
    )
    if result.returncode:
        message = result.stderr.decode("utf-8", "replace").strip()[:1000]
        raise RuntimeError(f"Jiuding workspace artifact transfer failed: {message}")
    return result


def _upload_archive(client: Any, destination: str, payload: bytes) -> None:
    _run_ssh(client, ["mkdir", destination])
    _run_ssh(
        client,
        ["tar", "-xzf", "-", "-C", destination],
        input=payload,
        timeout=60,
    )


def _ensure_source(client: Any) -> str:
    fingerprint, payload = _source_archive()
    parent = f"{_MANAGED_ROOT}/source"
    destination = f"{parent}/{fingerprint}"
    _run_ssh(client, ["mkdir", "-p", parent])
    probe = subprocess.run(
        [*client._ssh_base(), shlex.join(["test", "-f", f"{destination}/.complete"])],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    if probe.returncode == 0:
        return destination
    if probe.returncode != 1:
        raise RuntimeError("Could not inspect the Jiuding source snapshot")
    _upload_archive(client, destination, payload)
    _run_ssh(client, ["touch", f"{destination}/.complete"])
    return destination


def submit_managed_program(
    client: Any,
    program: Any,
    *,
    target: str,
    selected_target: str,
    operation: str,
    image: str,
    image_region: str,
    cpus: int,
    memory_gib: int,
) -> dict[str, Any]:
    """Stage one program through the selected workspace and submit it."""

    source = _ensure_source(client)
    job_directory = f"{_MANAGED_ROOT}/jobs/{uuid.uuid4().hex}"
    receipt = f"{job_directory}/receipt.json"
    with tempfile.TemporaryDirectory(prefix="fq-jiuding-") as temporary:
        local_receipt = Path(temporary) / "receipt.json"
        request, runner = write_program_bundle(
            program,
            target=selected_target,
            operation=operation,
            receipt=local_receipt,
        )
        runner.write_text(
            runner.read_text(encoding="utf-8").replace(
                str(request), f"{job_directory}/{request.name}"
            ),
            encoding="utf-8",
        )
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            archive.add(request, arcname=request.name)
            archive.add(runner, arcname=runner.name)
        _run_ssh(client, ["mkdir", "-p", f"{_MANAGED_ROOT}/jobs"])
        _upload_archive(client, job_directory, stream.getvalue())

    payload = {
        "workspace": client.workspace()["name"],
        "endpoint": client.endpoint,
        "script": f"{job_directory}/{runner.name}",
        "target": target,
        "image": image,
        "receipt": receipt,
        "image_region": image_region,
        "pythonpath": source,
        "cpus": cpus,
        "memory_gib": memory_gib,
    }
    code = (
        "import json,sys; "
        "from flagquantum.remote.compute.jiuding import JiudingClient; "
        "p=json.loads(sys.argv[1]); "
        "c=JiudingClient(workspace=p.pop('workspace'),endpoint=p.pop('endpoint')); "
        "print(json.dumps(c.submit(**p),separators=(',',':')))"
    )
    command = [
        "env",
        f"PYTHONPATH={source}",
        _REMOTE_PYTHON,
        "-c",
        code,
        json.dumps(payload, separators=(",", ":")),
    ]
    try:
        result = _run_ssh(client, command, timeout=90)
    except RuntimeError as error:
        raise RuntimeError(
            "Jiuding managed submission outcome is uncertain; inspect "
            f"{receipt} before retrying"
        ) from error
    try:
        record = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Jiuding managed submission returned invalid data"
        ) from error
    if not isinstance(record, dict) or not record.get("jobId"):
        raise RuntimeError("Jiuding managed submission returned no Job ID")
    return {
        **record,
        "submission_kind": "flagquantum_program",
        "artifact_transport": "workspace_ssh",
        "source_snapshot": source,
        "program_request": f"{job_directory}/{request.name}",
        "requested_target": target,
        "selected_target": selected_target,
        "operation": operation,
    }


def read_managed_result(client: Any, receipt: dict[str, Any]) -> object:
    """Read one bounded result through the workspace SSH connection."""

    path = receipt.get("result_path")
    if not isinstance(path, str) or not path.startswith(_MANAGED_ROOT + "/jobs/"):
        raise RuntimeError("Jiuding managed receipt has an invalid result path")
    code = "\n".join(
        (
            "import pathlib, sys",
            "path = pathlib.Path(sys.argv[1])",
            "if path.stat().st_size > int(sys.argv[2]):",
            "    raise RuntimeError('result exceeds limit')",
            "sys.stdout.buffer.write(path.read_bytes())",
        )
    )
    result = _run_ssh(
        client,
        [_REMOTE_PYTHON, "-c", code, path, str(_MAX_RESULT_BYTES)],
        timeout=60,
    )
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Jiuding managed result is not valid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError("Jiuding managed result must be a JSON object")
    if value.get("run_id") != receipt.get("run_id"):
        raise RuntimeError("Result does not belong to this submission")
    if "value" not in value:
        raise RuntimeError("Jiuding job result is missing its value")
    return value["value"]


__all__ = ("read_managed_result", "submit_managed_program")

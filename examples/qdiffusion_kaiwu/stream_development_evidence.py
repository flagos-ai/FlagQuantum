"""Validate inputs and privately capture a streamed A800 development record."""

from __future__ import annotations

import argparse
import hashlib
import re
import stat
import sys
from pathlib import Path
from typing import Any

from examples.qdiffusion_kaiwu.private_io import (
    read_private_bytes,
    write_private_json_exclusive,
)
from examples.qdiffusion_kaiwu.source_preflight import (
    load_source_preflight,
    validate_transfer_manifest_record,
)
from examples.qdiffusion_kaiwu.strict_json import loads_json_strict
from examples.qdiffusion_kaiwu.verify_transfer_bundle import verify_transfer_bundle

HOSTNAMES = {
    "jp-a800-171": "bm-baai-dx-zone1-lc-a800-80g-15-171",
    "jp-a800-172": "bm-baai-dx-zone1-lc-a800-80g-15-172",
}
FULL_REVISION = re.compile(r"[0-9a-f]{40}")
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
MAX_RECORD_BYTES = 4 * 1024 * 1024


def validate_stream_inputs(
    *,
    execution_host: str,
    expected_hostname: str,
    transfer_dir: Path,
    source_preflight: Path,
    output: Path,
    source_revision: str,
    plugin_revision: str,
) -> dict[str, str]:
    """Fail before SSH unless the local stream is the reviewed source set."""

    if HOSTNAMES.get(execution_host) != expected_hostname:
        raise ValueError("execution host and observed hostname do not match the review")
    if FULL_REVISION.fullmatch(source_revision) is None:
        raise ValueError("source revision must be a full lowercase Git revision")
    if FULL_REVISION.fullmatch(plugin_revision) is None:
        raise ValueError("plugin revision must be a full lowercase Git revision")
    if not output.is_absolute():
        raise ValueError("output path must be absolute")
    output_parent = output.parent
    parent_metadata = output_parent.lstat()
    if (
        output_parent.is_symlink()
        or not stat.S_ISDIR(parent_metadata.st_mode)
        or parent_metadata.st_mode & 0o077
    ):
        raise ValueError("output parent must be an existing private directory")
    if output.exists() or output.is_symlink():
        raise ValueError("refusing to overwrite existing development evidence")
    if not transfer_dir.is_absolute():
        raise ValueError("transfer directory must be absolute")
    metadata = transfer_dir.lstat()
    if (
        transfer_dir.is_symlink()
        or not transfer_dir.is_dir()
        or metadata.st_mode & 0o077
    ):
        raise ValueError("transfer directory must be an existing private directory")

    manifest = (
        transfer_dir
        / f"flagquantum-qboson-a800-bundle-{source_revision[:10]}.manifest.json"
    )
    transfer_record = verify_transfer_bundle(manifest, target_host=execution_host)
    preflight_record, preflight_sha256 = load_source_preflight(
        source_preflight,
        execution_host=execution_host,
        source_revision=source_revision,
        plugin_revision=plugin_revision,
    )
    if preflight_record["manifest_sha256"] != transfer_record["manifest_sha256"]:
        raise ValueError("source preflight and transfer manifest digests differ")
    return {
        "manifest": str(manifest),
        "manifest_sha256": transfer_record["manifest_sha256"],
        "source_preflight_sha256": preflight_sha256,
    }


def validate_development_record(
    record: Any,
    *,
    execution_host: str,
    expected_hostname: str,
    source_revision: str,
    plugin_revision: str,
    validation_image_id: str,
    source_preflight_sha256: str,
    transfer_manifest_sha256: str,
) -> dict[str, Any]:
    """Reject remote output that could overstate or lose its source identity."""

    if not isinstance(record, dict):
        raise ValueError("streamed development evidence must be a JSON object")
    if HOSTNAMES.get(execution_host) != expected_hostname:
        raise ValueError("execution host and observed hostname do not match the review")
    if (
        FULL_REVISION.fullmatch(source_revision) is None
        or FULL_REVISION.fullmatch(plugin_revision) is None
    ):
        raise ValueError("streamed development evidence revisions are invalid")
    if IMAGE_ID.fullmatch(validation_image_id) is None:
        raise ValueError("streamed development evidence image identity is invalid")
    if re.fullmatch(r"[0-9a-f]{64}", transfer_manifest_sha256) is None:
        raise ValueError("streamed development evidence manifest digest is invalid")
    expected = {
        "schema": "flagquantum.qboson_qdiffusion_system_development",
        "version": "1.0",
        "evidence_class": "development_fake_transport",
        "system_acceptance": False,
        "source_revision": source_revision,
        "kaiwu_pytorch_plugin_revision": plugin_revision,
        "source_preflight_sha256": source_preflight_sha256,
        "transfer_manifest_sha256": transfer_manifest_sha256,
        "validation_image_id": validation_image_id,
        "execution_host": execution_host,
        "observed_hostname": expected_hostname,
        "requested_cuda_device": "cuda:0",
        "observed_tensor_device": "cuda:0",
        "observed_gpu_model": "NVIDIA A800-SXM4-80GB",
        "transport": "in_memory_fake",
        "qboson_hardware_used": False,
        "real_provider_evidence": False,
        "fallback_occurred": False,
        "token_constraints_passed": True,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(f"streamed development evidence has invalid {field}")
    for field in ("proposal_device", "energy_device", "generated_device"):
        if record.get(field) != "cuda:0":
            raise ValueError(f"streamed development evidence has invalid {field}")
    call_count = record.get("remote_call_count")
    call_budget = record.get("remote_call_budget")
    task_count = record.get("task_count")
    if (
        not isinstance(call_count, int)
        or isinstance(call_count, bool)
        or not isinstance(call_budget, int)
        or isinstance(call_budget, bool)
        or call_count <= 0
        or call_count > call_budget
        or task_count != call_count
    ):
        raise ValueError("streamed development evidence has invalid call accounting")
    return record


def validate_retained_development_record(
    *,
    record_path: Path,
    execution_host: str,
    expected_hostname: str,
    source_revision: str,
    plugin_revision: str,
    validation_image_id: str,
    source_preflight: Path,
    transfer_manifest: Path,
) -> dict[str, str]:
    """Revalidate one retained local record without contacting a host."""

    preflight_record, preflight_sha256 = load_source_preflight(
        source_preflight,
        execution_host=execution_host,
        source_revision=source_revision,
        plugin_revision=plugin_revision,
    )
    manifest_bytes = read_private_bytes(
        transfer_manifest,
        label="transfer manifest",
        max_bytes=MAX_RECORD_BYTES,
    )
    manifest = loads_json_strict(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("transfer manifest must be a JSON object")
    validate_transfer_manifest_record(
        manifest,
        source_revision=source_revision,
        plugin_revision=plugin_revision,
    )
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if preflight_record.get("manifest_sha256") != manifest_sha256:
        raise ValueError("source preflight and retained transfer manifest differ")
    record_bytes = read_private_bytes(
        record_path,
        label="development record",
        max_bytes=MAX_RECORD_BYTES,
    )
    validate_development_record(
        loads_json_strict(record_bytes),
        execution_host=execution_host,
        expected_hostname=expected_hostname,
        source_revision=source_revision,
        plugin_revision=plugin_revision,
        validation_image_id=validation_image_id,
        source_preflight_sha256=preflight_sha256,
        transfer_manifest_sha256=manifest_sha256,
    )
    return {
        "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
        "source_preflight_sha256": preflight_sha256,
        "transfer_manifest_sha256": manifest_sha256,
    }


def _capture(arguments: argparse.Namespace) -> None:
    if IMAGE_ID.fullmatch(arguments.validation_image_id) is None:
        raise ValueError("validation image ID must be a full SHA-256 identity")
    if not arguments.output.is_absolute():
        raise ValueError("output path must be absolute")
    encoded = sys.stdin.buffer.read(MAX_RECORD_BYTES + 1)
    if len(encoded) > MAX_RECORD_BYTES:
        raise ValueError("streamed development evidence exceeds the size limit")
    _, source_preflight_sha256 = load_source_preflight(
        arguments.source_preflight,
        execution_host=arguments.execution_host,
        source_revision=arguments.source_revision,
        plugin_revision=arguments.plugin_revision,
    )
    record = validate_development_record(
        loads_json_strict(encoded),
        execution_host=arguments.execution_host,
        expected_hostname=arguments.expected_hostname,
        source_revision=arguments.source_revision,
        plugin_revision=arguments.plugin_revision,
        validation_image_id=arguments.validation_image_id,
        source_preflight_sha256=source_preflight_sha256,
        transfer_manifest_sha256=arguments.transfer_manifest_sha256,
    )
    write_private_json_exclusive(arguments.output, record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-inputs")
    capture = subparsers.add_parser("capture")
    retained = subparsers.add_parser("validate-record")
    for command in (validate, capture, retained):
        command.add_argument(
            "--execution-host", choices=sorted(HOSTNAMES), required=True
        )
        command.add_argument("--expected-hostname", required=True)
        command.add_argument("--source-revision", required=True)
        command.add_argument("--plugin-revision", required=True)
        command.add_argument("--source-preflight", required=True, type=Path)
    validate.add_argument("--transfer-dir", required=True, type=Path)
    validate.add_argument("--output", required=True, type=Path)
    capture.add_argument("--validation-image-id", required=True)
    capture.add_argument("--transfer-manifest-sha256", required=True)
    capture.add_argument("--output", required=True, type=Path)
    retained.add_argument("--validation-image-id", required=True)
    retained.add_argument("--transfer-manifest", required=True, type=Path)
    retained.add_argument("--record", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        if arguments.command == "validate-inputs":
            result = validate_stream_inputs(
                execution_host=arguments.execution_host,
                expected_hostname=arguments.expected_hostname,
                transfer_dir=arguments.transfer_dir,
                source_preflight=arguments.source_preflight,
                output=arguments.output,
                source_revision=arguments.source_revision,
                plugin_revision=arguments.plugin_revision,
            )
            print(result["manifest_sha256"])
        elif arguments.command == "capture":
            _capture(arguments)
        else:
            result = validate_retained_development_record(
                record_path=arguments.record,
                execution_host=arguments.execution_host,
                expected_hostname=arguments.expected_hostname,
                source_revision=arguments.source_revision,
                plugin_revision=arguments.plugin_revision,
                validation_image_id=arguments.validation_image_id,
                source_preflight=arguments.source_preflight,
                transfer_manifest=arguments.transfer_manifest,
            )
            print(result["record_sha256"])
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

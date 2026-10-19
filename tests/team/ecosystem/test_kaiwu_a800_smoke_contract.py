from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from examples.qdiffusion_kaiwu import private_io as private_io_module
from examples.qdiffusion_kaiwu.a800_sampler_smoke import _write_private_json
from examples.qdiffusion_kaiwu.qdiffusion_system_development_probe import (
    _write_private_json as _write_development_json,
)
from examples.qdiffusion_kaiwu.stream_development_evidence import (
    validate_development_record,
    validate_retained_development_record,
    validate_stream_inputs,
)

pytestmark = pytest.mark.unit


def test_a800_smoke_source_cannot_claim_real_provider_acceptance() -> None:
    source = (
        Path(__file__).parents[3]
        / "examples"
        / "qdiffusion_kaiwu"
        / "a800_sampler_smoke.py"
    ).read_text(encoding="utf-8")

    assert '"evidence_class": "development_fake_transport"' in source
    assert '"system_acceptance": False' in source
    assert '"qboson_hardware_used": False' in source
    assert '"real_provider_evidence": False' in source
    assert '"transport": "in_memory_fake"' in source


def test_a800_smoke_record_is_exclusive_and_mode_0600(tmp_path: Path) -> None:
    path = tmp_path / "probe.json"

    _write_private_json(path, {"evidence_class": "development_fake_transport"})

    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        _write_private_json(path, {"evidence_class": "development_fake_transport"})


@pytest.mark.parametrize("writer", (_write_private_json, _write_development_json))
@pytest.mark.parametrize("unsafe_kind", ("public", "symlink"))
def test_development_records_require_private_real_parent(
    tmp_path: Path, unsafe_kind: str, writer
) -> None:
    private_parent = tmp_path / "private"
    private_parent.mkdir(mode=0o700)
    if unsafe_kind == "public":
        private_parent.chmod(0o755)
        output = private_parent / "probe.json"
    else:
        linked_parent = tmp_path / "linked"
        linked_parent.symlink_to(private_parent, target_is_directory=True)
        output = linked_parent / "probe.json"

    with pytest.raises(ValueError, match="existing private, non-symlink"):
        writer(output, {"evidence_class": "development_fake_transport"})

    assert not output.exists()


def test_a800_container_runner_keeps_execution_bounded() -> None:
    source = (
        Path(__file__).parents[3]
        / "examples"
        / "qdiffusion_kaiwu"
        / "run_a800_development_probe.sh"
    ).read_text(encoding="utf-8")

    assert "--gpus device=0" in source
    assert "--network none" in source
    assert "--read-only" in source
    assert "--log-driver none" in source
    assert "--hostname $expected_hostname" in source
    assert "--tmpfs /workspace:" in source
    assert "--volume" not in source
    assert "COPYFILE_DISABLE=1 tar --no-xattrs -cf -" in source
    assert source.count("tar --no-same-owner -") == 4
    assert '| ssh "$execution_host"' in source
    assert "stream_development_evidence capture" in source
    assert "--source-preflight /workspace/input/" in source
    assert "--plugin-root /workspace/kaiwu-pytorch-plugin-" in source
    assert "VALIDATION_IMAGE_ID" in source
    assert '"$validation_image_id"' in source
    assert "PYTHONNOUSERSITE=1" in source
    assert "-m examples.qdiffusion_kaiwu.qdiffusion_system_development_probe" in source


def test_stream_inputs_reject_foreign_owned_output_parent_before_bundle_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transfer = tmp_path / "transfer"
    transfer.mkdir(mode=0o700)
    effective_uid = os.geteuid()
    monkeypatch.setattr(private_io_module.os, "geteuid", lambda: effective_uid + 1)

    with pytest.raises(ValueError, match="output parent must"):
        validate_stream_inputs(
            execution_host="jp-a800-171",
            expected_hostname="bm-baai-dx-zone1-lc-a800-80g-15-171",
            transfer_dir=transfer,
            source_preflight=tmp_path / "missing-preflight.json",
            output=tmp_path / "development.json",
            source_revision="a" * 40,
            plugin_revision="b" * 40,
        )


def _streamed_record() -> dict[str, object]:
    return {
        "schema": "flagquantum.qboson_qdiffusion_system_development",
        "version": "1.0",
        "evidence_class": "development_fake_transport",
        "system_acceptance": False,
        "source_revision": "a" * 40,
        "kaiwu_pytorch_plugin_revision": "b" * 40,
        "source_preflight_sha256": "c" * 64,
        "transfer_manifest_sha256": "d" * 64,
        "validation_image_id": f"sha256:{'e' * 64}",
        "execution_host": "jp-a800-171",
        "observed_hostname": "bm-baai-dx-zone1-lc-a800-80g-15-171",
        "requested_cuda_device": "cuda:0",
        "observed_tensor_device": "cuda:0",
        "observed_gpu_model": "NVIDIA A800-SXM4-80GB",
        "proposal_device": "cuda:0",
        "energy_device": "cuda:0",
        "generated_device": "cuda:0",
        "transport": "in_memory_fake",
        "qboson_hardware_used": False,
        "real_provider_evidence": False,
        "fallback_occurred": False,
        "token_constraints_passed": True,
        "remote_call_count": 10,
        "remote_call_budget": 64,
        "task_count": 10,
    }


def _validate_streamed(record: dict[str, object]) -> dict[str, object]:
    return validate_development_record(
        record,
        execution_host="jp-a800-171",
        expected_hostname="bm-baai-dx-zone1-lc-a800-80g-15-171",
        source_revision="a" * 40,
        plugin_revision="b" * 40,
        validation_image_id=f"sha256:{'e' * 64}",
        source_preflight_sha256="c" * 64,
        transfer_manifest_sha256="d" * 64,
    )


def test_streamed_development_record_preserves_non_acceptance_identity() -> None:
    record = _streamed_record()

    assert _validate_streamed(record) is record


def test_retained_development_record_revalidates_private_hash_chain(
    tmp_path: Path,
) -> None:
    source_revision = "a" * 40
    plugin_revision = "b" * 40
    community_revision = "b648b531c034bd6ae9b7a34fed994c717967cc72"
    roles = (
        ("flagquantum-qboson-", "FlagQuantum-", source_revision),
        ("kaiwu-plugin-", "kaiwu-pytorch-plugin-", plugin_revision),
        ("kaiwu-community-", "kaiwu-community-", community_revision),
    )
    manifest = {
        "schema": "flagquantum.qboson_a800_transfer_bundle",
        "version": "1.0",
        "created_for_hosts": ["jp-a800-171", "jp-a800-172"],
        "classification": "local_preparation_only_not_execution_evidence",
        "artifacts": [
            {
                "filename": f"{prefix}{revision[:10]}.tar.gz",
                "revision": revision,
                "sha256": "f" * 64,
            }
            for prefix, _, revision in roles
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_path.chmod(0o600)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    preflight = {
        "schema": "flagquantum.qboson_a800_extracted_bundle_verification",
        "version": "1.0",
        "evidence_class": "extraction_preflight_only",
        "verification_hostname": "review-host",
        "verified_for_target_host": "jp-a800-171",
        "manifest_sha256": manifest_sha256,
        "extracted_content_verified": True,
        "artifacts": [
            {
                "filename": f"{prefix}{revision[:10]}.tar.gz",
                "revision": revision,
                "extracted_root": f"{root}{revision[:10]}",
                "file_count": 1,
                "content_set_sha256": "e" * 64,
            }
            for prefix, root, revision in roles
        ],
        "qboson_hardware_used": False,
        "a800_execution_verified": False,
        "acceptance_evidence": False,
    }
    preflight_path = tmp_path / "source-preflight.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    preflight_path.chmod(0o600)
    record = _streamed_record()
    record["source_preflight_sha256"] = hashlib.sha256(
        preflight_path.read_bytes()
    ).hexdigest()
    record["transfer_manifest_sha256"] = manifest_sha256
    record_path = tmp_path / "development.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    record_path.chmod(0o600)

    result = validate_retained_development_record(
        record_path=record_path,
        execution_host="jp-a800-171",
        expected_hostname="bm-baai-dx-zone1-lc-a800-80g-15-171",
        source_revision=source_revision,
        plugin_revision=plugin_revision,
        validation_image_id=f"sha256:{'e' * 64}",
        source_preflight=preflight_path,
        transfer_manifest=manifest_path,
    )

    assert result == {
        "record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
        "source_preflight_sha256": record["source_preflight_sha256"],
        "transfer_manifest_sha256": manifest_sha256,
    }

    record["source_preflight_sha256"] = "0" * 64
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="source_preflight_sha256"):
        validate_retained_development_record(
            record_path=record_path,
            execution_host="jp-a800-171",
            expected_hostname="bm-baai-dx-zone1-lc-a800-80g-15-171",
            source_revision=source_revision,
            plugin_revision=plugin_revision,
            validation_image_id=f"sha256:{'e' * 64}",
            source_preflight=preflight_path,
            transfer_manifest=manifest_path,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("system_acceptance", True),
        ("qboson_hardware_used", True),
        ("transport", "kaiwu_cim"),
        ("observed_tensor_device", "cpu"),
        ("source_preflight_sha256", "f" * 64),
        ("remote_call_count", 65),
    ),
)
def test_streamed_development_record_rejects_overclaim_or_identity_drift(
    field: str, value: object
) -> None:
    record = _streamed_record()
    record[field] = value

    with pytest.raises(ValueError):
        _validate_streamed(record)


def test_qdiffusion_development_source_cannot_claim_acceptance() -> None:
    source = (
        Path(__file__).parents[3]
        / "examples"
        / "qdiffusion_kaiwu"
        / "qdiffusion_system_development_probe.py"
    ).read_text(encoding="utf-8")

    assert '"evidence_class": "development_fake_transport"' in source
    assert '"system_acceptance": False' in source
    assert '"qboson_hardware_used": False' in source
    assert '"real_provider_evidence": False' in source
    assert '"transport": "in_memory_fake"' in source
    assert '"validation_image_id": validation_image_id' in source
    assert '"source_preflight_sha256": source_preflight_sha256' in source
    assert '"transfer_manifest_sha256": transfer_manifest_sha256' in source


def test_qdiffusion_live_source_requires_cost_and_provider_identity() -> None:
    source = (
        Path(__file__).parents[3]
        / "examples"
        / "qdiffusion_kaiwu"
        / "qdiffusion_system_live.py"
    ).read_text(encoding="utf-8")

    assert "I_ACKNOWLEDGE_QBOSON_QUOTA_USAGE" in source
    assert "provider_identity_complete" in source
    assert "retrieval_resubmitted is False" in source
    assert '"returned_samples"' in source
    assert '"fallback_occurred": False' in source
    assert "resolve_kaiwu_credentials" in source
    assert "_write_private_redacted_json" in source


def test_local_golden_path_is_pinned_and_credential_free() -> None:
    path = (
        Path(__file__).parents[3]
        / "examples"
        / "qdiffusion_kaiwu"
        / "run_local_conformance.sh"
    )
    source = path.read_text(encoding="utf-8")

    assert path.stat().st_mode & 0o111
    assert "b648b531c034bd6ae9b7a34fed994c717967cc72" in source
    assert "f047bce7b1077449967bbe9e9fab5741542b48d4" in source
    assert "status --porcelain --untracked-files=all" in source
    assert "unset QBOSON_USER_ID QBOSON_SDK_CODE QBOSON_PROJECT_NO" in source
    assert "PYTHONNOUSERSITE=1" in source
    assert "FLAGQUANTUM_NETWORK_DISABLED=1" in source
    assert 'PYTHONPATH="$OFFLINE_GUARD_DIR:' in source
    assert (
        'PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/flagquantum-kaiwu-pycache-$$"' in source
    )
    assert '"$PYTHON_BIN" -B -m pytest' in source
    assert "test_kaiwu_community_conformance.py" in source
    assert "test_kaiwu_pytorch_plugin_conformance.py" in source
    assert "test_qdiffusion_environment_lock.py" in source
    assert "test_qdiffusion_sdk_approval.py" in source
    assert "test_kaiwu_vertical_slice.py" in source
    assert "qboson_live_smoke.py" not in source


def test_local_golden_path_offline_guard_denies_dns_and_ip_connections() -> None:
    repository = Path(__file__).parents[3]
    guard = repository / "examples" / "qdiffusion_kaiwu" / "offline_guard"
    program = """
import errno
import socket

operations = (
    lambda: socket.getaddrinfo("example.com", 443),
    lambda: socket.create_connection(("127.0.0.1", 9)),
    lambda: socket.socket().connect(("127.0.0.1", 9)),
)
for operation in operations:
    try:
        operation()
    except OSError as exc:
        assert exc.errno == errno.ENETUNREACH
        assert "local-conformance guard" in str(exc)
    else:
        raise AssertionError("network operation escaped the offline guard")
"""
    environment = {
        **os.environ,
        "PYTHONPATH": str(guard),
        "PYTHONNOUSERSITE": "1",
        "FLAGQUANTUM_NETWORK_DISABLED": "1",
    }

    completed = subprocess.run(
        [sys.executable, "-s", "-c", program],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ENTRYPOINTS = (
    "a800_sampler_smoke",
    "assemble_acceptance",
    "build_environment_lock",
    "build_transfer_bundle",
    "plan_quota",
    "preflight_protein_artifacts",
    "qboson_live_smoke",
    "qdiffusion_portability_replay_live",
    "qdiffusion_protein_evaluate",
    "qdiffusion_protein_training_live",
    "qdiffusion_system_development_probe",
    "qdiffusion_system_live",
    "stream_development_evidence",
    "validate_acceptance",
    "verify_extracted_bundle",
    "verify_environment_lock",
    "verify_transfer_bundle",
)

PINNED_SDK_ENTRYPOINTS = (
    "qboson_live_smoke",
    "qdiffusion_portability_replay_live",
    "qdiffusion_protein_evaluate",
    "qdiffusion_protein_training_live",
    "qdiffusion_system_live",
)

LIVE_HOST_ENTRYPOINTS = (
    "qdiffusion_portability_replay_live",
    "qdiffusion_protein_evaluate",
    "qdiffusion_protein_training_live",
    "qdiffusion_system_live",
)

RUNBOOK_ACCEPTANCE_ENTRYPOINTS = (
    "build_transfer_bundle",
    "verify_transfer_bundle",
    "verify_extracted_bundle",
    "build_environment_lock",
    "verify_environment_lock",
    "qboson_live_smoke",
    "qdiffusion_system_live",
    "preflight_protein_artifacts",
    "qdiffusion_protein_training_live",
    "qdiffusion_protein_evaluate",
    "qdiffusion_portability_replay_live",
    "assemble_acceptance",
    "validate_acceptance",
)


def _runbook_command(entrypoint: str) -> str:
    runbook = (
        Path(__file__).parents[3]
        / "docs"
        / "guides"
        / "QBOSON_QDIFFUSION_RUNBOOK.md"
    ).read_text(encoding="utf-8")
    marker = f"-m examples.qdiffusion_kaiwu.{entrypoint}"
    start = runbook.index(marker)
    end = runbook.index("```", start)
    return runbook[start:end]


def _argument_flags(entrypoint: str) -> tuple[set[str], set[str]]:
    source = (
        Path(__file__).parents[3]
        / "examples"
        / "qdiffusion_kaiwu"
        / f"{entrypoint}.py"
    ).read_text(encoding="utf-8")
    all_flags: set[str] = set()
    required_flags: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Attribute)
            or node.func.attr != "add_argument"
            or not node.args
            or not isinstance(node.args[0], ast.Constant)
            or not isinstance(node.args[0].value, str)
            or not node.args[0].value.startswith("--")
        ):
            continue
        flag = node.args[0].value
        all_flags.add(flag)
        if any(
            keyword.arg == "required"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        ):
            required_flags.add(flag)
    return all_flags, required_flags


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
def test_documented_module_entrypoint_binds_current_checkout(
    tmp_path: Path, entrypoint: str
) -> None:
    repository = Path(__file__).parents[3]
    environment = {
        **os.environ,
        "PYTHONPATH": str(repository),
        "PYTHONNOUSERSITE": "1",
    }

    completed = subprocess.run(
        [
            sys.executable,
            "-s",
            "-m",
            f"examples.qdiffusion_kaiwu.{entrypoint}",
            "--help",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()


@pytest.mark.parametrize("entrypoint", PINNED_SDK_ENTRYPOINTS)
def test_live_entrypoint_rejects_unpinned_sdk_version_before_other_inputs(
    tmp_path: Path, entrypoint: str
) -> None:
    repository = Path(__file__).parents[3]
    environment = {
        **os.environ,
        "PYTHONPATH": str(repository),
        "PYTHONNOUSERSITE": "1",
    }

    completed = subprocess.run(
        [
            sys.executable,
            "-s",
            "-m",
            f"examples.qdiffusion_kaiwu.{entrypoint}",
            "--expected-sdk-version",
            "1.4.1",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "invalid choice: '1.4.1'" in completed.stderr


@pytest.mark.parametrize("entrypoint", PINNED_SDK_ENTRYPOINTS)
def test_runbook_pins_sdk_version_for_every_sdk_entrypoint(entrypoint: str) -> None:
    command = _runbook_command(entrypoint)

    assert "--expected-sdk-version 1.3.1" in command


@pytest.mark.parametrize(
    "entrypoint",
    (
        "qboson_live_smoke",
        "qdiffusion_system_live",
        "qdiffusion_protein_training_live",
        "qdiffusion_portability_replay_live",
    ),
)
def test_runbook_never_relies_on_default_provider_sample_count(
    entrypoint: str,
) -> None:
    command = _runbook_command(entrypoint)

    assert '--requested-samples "$FROZEN_REQUESTED_SAMPLES"' in command


@pytest.mark.parametrize("entrypoint", RUNBOOK_ACCEPTANCE_ENTRYPOINTS)
def test_runbook_commands_match_required_cli_contract(entrypoint: str) -> None:
    command = _runbook_command(entrypoint)
    all_flags, required_flags = _argument_flags(entrypoint)
    documented_flags = set(re.findall(r"--[a-z][a-z0-9-]*", command))

    assert required_flags <= documented_flags
    assert documented_flags <= all_flags


@pytest.mark.parametrize("entrypoint", LIVE_HOST_ENTRYPOINTS)
def test_live_entrypoint_binds_expected_hostname_to_frozen_host_identity(
    entrypoint: str,
) -> None:
    source = (
        Path(__file__).parents[3]
        / "examples"
        / "qdiffusion_kaiwu"
        / f"{entrypoint}.py"
    ).read_text(encoding="utf-8")

    assert "expected_hostname !=" in source
    assert "config[\"host_identities\"]" in source
    assert (
        'parser.error("--expected-hostname differs from the frozen host identity")'
        in source
    )
    assert source.index("expected_hostname !=") < source.index("socket.gethostname()")

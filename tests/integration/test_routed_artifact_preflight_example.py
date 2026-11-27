"""Exercise the documented JSON handoff through the example's CLI entry."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

from examples.circuits.compiler.routed_artifact_preflight import main
from flagquantum.core._artifacts import (
    ProgramArtifactV2,
    ProgramArtifactV3,
    read_program_artifact_json,
)
from flagquantum.core._compilation_evidence import (
    read_compilation_evidence_bundle_json,
)
from flagquantum.core._compilation_evidence_v3 import CompilationEvidenceBundleV3

pytestmark = pytest.mark.integration


def test_documented_example_saves_and_verifies_an_offline_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def reject_connection(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("The offline example attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", reject_connection)
    monkeypatch.setattr(socket.socket, "connect", reject_connection)
    output_dir = tmp_path / "routed-artifact"
    monkeypatch.setattr(
        sys, "argv", ["routed_artifact_preflight", "--output-dir", str(output_dir)]
    )
    main()

    source = read_program_artifact_json(
        (output_dir / "source.json").read_text(encoding="utf-8")
    )
    artifact = read_program_artifact_json(
        (output_dir / "program.json").read_text(encoding="utf-8")
    )
    evidence = read_compilation_evidence_bundle_json(
        (output_dir / "evidence.json").read_text(encoding="utf-8")
    )
    assert isinstance(source, ProgramArtifactV2)
    assert isinstance(artifact, ProgramArtifactV3)
    assert isinstance(evidence, CompilationEvidenceBundleV3)
    assert evidence.source["source_artifact_identity"] == source.artifact_identity
    assert evidence.output["executable_artifact_identity"] == artifact.artifact_identity
    assert artifact.result_schema["physical_result_slots"] == (2, 0)
    assert evidence.physical_plan.initial_logical_to_physical == (2, 0)
    assert evidence.physical_plan.final_logical_to_physical == (2, 0)

    stdout = capsys.readouterr().out
    assert "Inserted SWAPs 2" in stdout
    assert "Logical result slots (2, 0)" in stdout
    assert "Offline preflight and CPU state verification passed" in stdout

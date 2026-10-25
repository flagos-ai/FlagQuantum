"""Coordinator software fakes; no container, CUDA, or research data execution."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.qec import compare_ising as comparison
from examples.qec import continued_assessment as inputs
from examples.qec import continued_parity as parity
from examples.qec import continued_primary as primary
from examples.qec import continued_selection as selection
from examples.qec import diagnose_predecoder as archive
from examples.qec import run_continued_assessment as run

pytestmark = pytest.mark.integration


@pytest.fixture
def options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> run.Options:
    inspect = tmp_path / "inspect.json"
    data = [
        {
            "Id": "a" * 64,
            "Image": parity.IMAGE,
            "Config": {
                "Hostname": "a" * 12,
                "Labels": {"flagquantum.host": "a800-172"},
            },
            "HostConfig": {
                "NetworkMode": "none",
                "Memory": 16 * 1024**3,
                "NanoCpus": 8_000_000_000,
                "DeviceRequests": [{"DeviceIDs": ["fixture-gpu"]}],
            },
            "Mounts": [
                {"Destination": str(tmp_path), "RW": False},
                {"Destination": str(archive.ROOT), "RW": False},
            ],
        }
    ]
    inspect.write_text(json.dumps(data))
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "node", lambda: "a" * 12)
    return run.Options(
        tmp_path / "original",
        tmp_path / "continued",
        tmp_path / "ising",
        tmp_path / "fast.safetensors",
        inspect,
        archive.digest(inspect),
        "a800-172",
        tmp_path / "study",
        archive.ROOT
        / "docs/development/evidence/qec_ai_ising_a800_20261025/requirements.lock",
    )


@pytest.mark.parametrize(
    "failure",
    ["image", "network", "memory", "cpus", "hostname", "host", "readonly", "hash"],
)
def test_bad_controller_binding_refuses(options: run.Options, failure: str) -> None:
    raw = json.loads(options.inspect.read_text())
    if failure == "hash":
        raw[0]["Id"] = "b" * 64
    elif failure == "image":
        raw[0]["Image"] = "sha256:" + "0" * 64
    elif failure == "network":
        raw[0]["HostConfig"]["NetworkMode"] = "bridge"
    elif failure == "memory":
        raw[0]["HostConfig"]["Memory"] = 32 * 1024**3
    elif failure == "cpus":
        raw[0]["HostConfig"]["NanoCpus"] = 16_000_000_000
    elif failure == "hostname":
        raw[0]["Config"]["Hostname"] = "another-container"
    elif failure == "host":
        raw[0]["Config"]["Labels"]["flagquantum.host"] = "a800-171"
    elif failure == "readonly":
        raw[0]["Mounts"][0]["RW"] = True
    options.inspect.write_text(json.dumps(raw))
    current = (
        options
        if failure == "hash"
        else replace(options, inspect_sha256=archive.digest(options.inspect))
    )
    with pytest.raises(ValueError):
        run._container(current)


def test_input_whitelist_refuses_old_arrays_before_load_or_device(
    options: run.Options,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options.original.mkdir()
    (options.original / "old-test-shots.pt").write_bytes(b"never mounted for real")

    def forbidden(*a: object, **k: object) -> None:
        raise AssertionError("unexpected input reached model/device")

    monkeypatch.setattr(inputs, "load_six", forbidden)
    monkeypatch.setattr(comparison, "cuda_metadata", forbidden)
    with pytest.raises(ValueError, match="whitelist"):
        run.preflight(options)
    assert not options.study.exists()


@pytest.fixture
def stages(options: run.Options, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    class Circuit:
        def __str__(self) -> str:
            return "software coordinator circuit"

    circuit = Circuit()
    monkeypatch.setattr(
        run,
        "preflight",
        lambda opts: ([], SimpleNamespace(circuit=circuit), {"fixture": "input"}),
    )

    def output(phase: str, path: Path, filename: str) -> None:
        calls.append(phase)
        path.mkdir(parents=True, exist_ok=False)
        (path / filename).write_text("{}")

    monkeypatch.setattr(
        selection,
        "calibrate",
        lambda models, path: output("calibrate", path, "manifest.json"),
    )
    monkeypatch.setattr(
        inputs,
        "capture_validation",
        lambda c, path: output("validation", path, "manifest.json"),
    )
    monkeypatch.setattr(
        selection,
        "freeze_choices",
        lambda m, c, v, path, **kw: output("choices", path, "choices.json"),
    )
    monkeypatch.setattr(comparison, "global_decoder", lambda c: None)
    monkeypatch.setattr(
        parity,
        "capture_probes",
        lambda c, ch, m, path, **kw: output("probes", path, "manifest.json"),
    )
    monkeypatch.setattr(
        parity,
        "record_device_receipt",
        lambda m, ch, p, path, **kw: output(kw["host"], path, "receipt.json"),
    )
    monkeypatch.setattr(
        primary,
        "run_primary",
        lambda m, ch, p, r, path, **kw: output("primary", path, "manifest.json"),
    )

    def summarize(path: Path, output_path: Path, **kw: object) -> None:
        calls.append("summary")
        output_path.write_text("{}")

    monkeypatch.setattr(primary, "summarize_primary", summarize)
    return calls


def test_prepare_peer_finish_order_and_nonrepeatable_locks(
    options: run.Options,
    stages: list[str],
    tmp_path: Path,
) -> None:
    handoff = run.prepare(options)
    assert stages == ["calibrate", "validation", "choices", "probes", "a800-172"]
    plan = json.loads((options.study / "plan.json").read_text())
    assert plan["deadline"] - plan["started"] == 86400
    assert (options.study / "launch.lock").exists()
    with pytest.raises(FileExistsError):
        run.prepare(options)
    assert len(stages) == 5
    peer_options = replace(options, host="a800-171")
    peer_sha = run.peer(peer_options, handoff, options.study / "receipt-171")
    assert stages[-1] == "a800-171"
    run.finish(options, handoff, peer_sha)
    assert stages[-2:] == ["primary", "summary"]
    assert (options.study / "complete.json").exists()
    with pytest.raises(FileExistsError):
        run.finish(options, handoff, peer_sha)
    assert stages.count("primary") == 1


def test_failed_prepare_keeps_lock_and_blocks_data_retry(
    options: run.Options,
    stages: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed(*a: object, **k: object) -> None:
        raise RuntimeError("synthetic validation failure")

    monkeypatch.setattr(inputs, "capture_validation", failed)
    with pytest.raises(RuntimeError, match="validation failure"):
        run.prepare(options)
    assert stages == ["calibrate"] and (options.study / "launch.lock").exists()
    assert not (options.study / "ready.json").exists()
    with pytest.raises(FileExistsError):
        run.prepare(options)
    assert stages == ["calibrate"]


@pytest.mark.parametrize("failure", ["handoff", "source", "input", "deadline"])
def test_handoff_refusal_precedes_primary(
    options: run.Options,
    stages: list[str],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    handoff = run.prepare(options)
    if failure == "source":
        monkeypatch.setattr(run, "_fingerprints", lambda: {"changed": "source"})
    elif failure == "input":
        monkeypatch.setattr(
            run, "preflight", lambda opts: ([], None, {"different": "input"})
        )
    elif failure == "deadline":
        plan_path = options.study / "plan.json"
        plan = json.loads(plan_path.read_text())
        plan["started"] = time.time() - 90000
        plan["deadline"] = plan["started"] + 86400
        plan_path.write_text(json.dumps(plan))
        ready_path = options.study / "ready.json"
        ready = json.loads(ready_path.read_text())
        ready["plan_sha256"] = archive.digest(plan_path)
        ready_path.write_text(json.dumps(ready))
        handoff = archive.digest(ready_path)
    else:
        handoff = "0" * 64
    with pytest.raises((ValueError, TimeoutError)):
        run.finish(options, handoff, "0" * 64)
    assert "primary" not in stages and not (options.study / "complete.json").exists()


def test_deadline_interrupts_only_the_running_process() -> None:
    with (
        pytest.raises(TimeoutError, match="deadline"),
        run.deadline(time.time() + 0.04),
    ):
        time.sleep(0.15)


def test_run_consumes_external_peer_hash_after_prepare(
    options: run.Options,
    stages: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incoming = tmp_path / "peer.sha256"
    prepare = run.prepare

    def deliver(options: run.Options) -> str:
        result = prepare(options)
        incoming.write_text("e" * 64 + "\n")
        return result

    monkeypatch.setattr(run, "prepare", deliver)
    run.run(options, incoming)
    assert stages == [
        "calibrate",
        "validation",
        "choices",
        "probes",
        "a800-172",
        "primary",
        "summary",
    ]
    assert (options.study / "complete.json").exists()


def test_stale_peer_delivery_refuses_before_prepare(
    options: run.Options,
    stages: list[str],
    tmp_path: Path,
) -> None:
    incoming = tmp_path / "stale.sha256"
    incoming.write_text("e" * 64)
    with pytest.raises(FileExistsError, match="start empty"):
        run.run(options, incoming)
    assert not stages and not options.study.exists()


def test_source_package_refuses_old_score_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "source"
    monkeypatch.setattr(archive, "ROOT", root)
    evidence = root / "docs/development/evidence"
    for name in ("manifest.json", "results.json", "choices.json", "audit.json"):
        path = evidence / "qec_ai_ising_a800_20261025/a800_172" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    for name in (
        "qec_ai_ising_a800_20261025/requirements.lock",
        "qec_ai_training_response_20261025/audit.json",
    ):
        path = evidence / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    run._source_evidence()
    (evidence / "qec_ai_training_response_20261025/old-heldout.json").write_text(
        "[0.2]"
    )
    with pytest.raises(ValueError, match="only required metadata"):
        run._source_evidence()

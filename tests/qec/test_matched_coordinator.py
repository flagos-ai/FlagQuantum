"""Explicit container/device/layout software fakes; no actual preflight evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from examples.qec import run_matched_assessment as run
from flagquantum.qec.decoders._ai.predecoder import Predecoder
from flagquantum.qec.decoders._ai.training import Trainer

pytestmark = pytest.mark.integration


@pytest.fixture
def context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[run.Options, dict[str, Any], list[str]]:
    root = tmp_path / "source"
    evidence = root / "docs/development/evidence/qec_ai_ising_a800_20261025"
    anchor = evidence / "a800_172"
    anchor.mkdir(parents=True)
    (anchor / "manifest.json").write_text(
        json.dumps({"python": "fixture-python", "torch": str(torch.__version__)})
    )
    (anchor / "results.json").write_text("{}")
    (evidence / "requirements.lock").write_text("pinned dependency fixture")
    lock = tmp_path / "requirements.lock"
    lock.write_text((evidence / "requirements.lock").read_text())
    weights = tmp_path / "weights"
    weights.write_bytes(b"explicitly fake weight file, never loaded")
    source = tmp_path / "vendor"
    source.mkdir()
    inspect = tmp_path / "inspect.json"
    raw = {
        "Id": "a" * 64,
        "Image": run.device.IMAGE,
        "Config": {"Hostname": "a" * 12, "Labels": {"flagquantum.host": "a800-172"}},
        "HostConfig": {
            "NetworkMode": "none",
            "NanoCpus": 8000000000,
            "Memory": 16 * 1024**3,
            "AutoRemove": True,
            "DeviceRequests": [
                {"DeviceIDs": ["GPU-11111111-1111-1111-1111-111111111111"]}
            ],
        },
        "Mounts": [{"Destination": str(tmp_path), "RW": False}],
    }
    inspect.write_text(json.dumps([raw]))
    options = run.Options(
        source,
        weights,
        inspect,
        run.archive.digest(inspect),
        "a800-172",
        tmp_path / "study",
        lock,
    )
    monkeypatch.setattr(run.archive, "ROOT", root)
    monkeypatch.setattr(run.archive, "ARCHIVE", anchor)
    monkeypatch.setattr(run.platform, "system", lambda: "Linux")
    monkeypatch.setattr(run.platform, "node", lambda: "a" * 12)
    monkeypatch.setattr(run.platform, "python_version", lambda: "fixture-python")
    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(run.comparison, "FAST_SHA256", run.archive.digest(weights))
    monkeypatch.setattr(
        run.comparison, "source_digest", lambda p: run.comparison.ISING_SOURCE_SHA256
    )
    monkeypatch.setattr(run, "_fingerprints", lambda: {"fixture-source": "a" * 64})
    protocol = tmp_path / "protocol.md"
    correction = tmp_path / "correction.md"
    protocol.write_text("prospective software fixture")
    correction.write_text("layout correction fixture")
    monkeypatch.setattr(run.checkpoints, "PROTOCOLS", (protocol, correction))
    monkeypatch.setattr(run.parity, "_precision", lambda: None)
    calls: list[str] = []
    dem = "synthetic DEM identity"
    template = SimpleNamespace(dem_text=dem)
    hardware = dict(
        device="cuda:0",
        name="A800",
        image_id=run.device.IMAGE,
        tf32=False,
        autocast=False,
        deterministic_algorithms=True,
        nvidia_smi_uuid_name_driver="GPU-11111111-1111-1111-1111-111111111111,A800,fixture",
    )
    pipeline = torch.nn.Linear(1, 1).eval()

    def build(p: Path) -> tuple[Any, dict[str, object]]:
        calls.append("layout")
        torch.rand(3)
        return template, dict(
            circuit_sha256="a" * 64,
            dem_sha256=hashlib.sha256(dem.encode()).hexdigest(),
            layout_sha256="b" * 64,
            new_samples=0,
            new_optimizer_steps=0,
        )

    def encoding(model: Any) -> None:
        assert model is template
        calls.append("encoding")

    def reference(*a: Any, **k: Any) -> tuple[dict[str, Any], Any]:
        calls.append("hardware_reference")
        return hardware, SimpleNamespace(pipeline=pipeline)

    def forbidden(*a: Any, **k: Any) -> None:
        raise AssertionError(
            "zero-sample preflight reached sampling, training or forward"
        )

    monkeypatch.setattr(run.layout, "build_model", build)
    monkeypatch.setattr(run.layout, "verify_encoding", encoding)
    monkeypatch.setattr(run.device, "_device_reference", reference)
    monkeypatch.setattr(Predecoder, "sample", forbidden)
    monkeypatch.setattr(Trainer, "train", forbidden)
    monkeypatch.setattr(pipeline, "forward", forbidden)
    return (
        options,
        dict(
            hardware=hardware,
            pipeline=pipeline,
            raw=raw,
            protocol=protocol,
            template=template,
        ),
        calls,
    )


def test_zero_sample_preflight_uses_shared_container_guards_and_preserves_rng(
    context: tuple[run.Options, dict[str, Any], list[str]],
) -> None:
    options, parts, calls = context
    rng = torch.get_rng_state().clone()
    report = run.preflight(options)
    assert torch.equal(rng, torch.get_rng_state())
    assert calls == ["layout", "encoding", "hardware_reference"]
    assert report["new_samples"] == report["new_optimizer_steps"] == 0
    assert report["formal_launch_authorized"] is False
    assert report["protocols"] == {
        p.name: run.archive.digest(p) for p in run.checkpoints.PROTOCOLS
    }
    assert not options.study.exists()


@pytest.mark.parametrize(
    "damage",
    [
        "hash",
        "image",
        "network",
        "memory",
        "cpus",
        "host",
        "hostname",
        "readonly",
        "autoremove",
        "lock",
        "arrays",
        "weights",
        "source",
        "runtime",
        "autocast",
    ],
)
def test_invalid_scope_rejects_before_layout_or_device(
    context: tuple[run.Options, dict[str, Any], list[str]],
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    options, parts, calls = context
    raw = parts["raw"]
    if damage == "hash":
        raw["Id"] = "b" * 64
    elif damage == "image":
        raw["Image"] = "sha256:" + "0" * 64
    elif damage == "network":
        raw["HostConfig"]["NetworkMode"] = "bridge"
    elif damage == "memory":
        raw["HostConfig"]["Memory"] = 32 * 1024**3
    elif damage == "cpus":
        raw["HostConfig"]["NanoCpus"] = 16000000000
    elif damage == "host":
        raw["Config"]["Labels"]["flagquantum.host"] = "a800-171"
    elif damage == "hostname":
        raw["Config"]["Hostname"] = "another"
    elif damage == "readonly":
        raw["Mounts"][0]["RW"] = True
    elif damage == "autoremove":
        raw["HostConfig"]["AutoRemove"] = False
    elif damage == "lock":
        options.dependency_lock.write_text("changed")
    elif damage == "arrays":
        (run.archive.ARCHIVE / "old-test-array.pt").write_bytes(b"not admitted")
    elif damage == "weights":
        options.ising_weights.write_bytes(b"changed")
    elif damage == "source":
        monkeypatch.setattr(run.comparison, "source_digest", lambda p: "b" * 64)
    elif damage == "runtime":
        monkeypatch.setattr(run.platform, "python_version", lambda: "wrong")
    elif damage == "autocast":
        monkeypatch.setattr(torch, "is_autocast_enabled", lambda *a: True)
    options.inspect.write_text(json.dumps([raw]))
    if damage != "hash":
        options = replace(options, inspect_sha256=run.archive.digest(options.inspect))
    with pytest.raises(ValueError):
        run.preflight(options)
    assert calls == [] and not options.study.exists()


@pytest.mark.parametrize(
    "damage",
    [
        "uuid",
        "multiple_devices",
        "requests",
        "fp64",
        "nonfinite",
        "dem",
        "source",
        "protocol",
        "inspect",
        "metadata",
        "precision",
    ],
)
def test_reference_or_concurrent_mutation_cannot_return_preflight(
    context: tuple[run.Options, dict[str, Any], list[str]],
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    options, parts, calls = context
    if damage == "requests":
        parts["raw"]["HostConfig"]["DeviceRequests"].append({"DeviceIDs": ["extra"]})
        options.inspect.write_text(json.dumps([parts["raw"]]))
        options = replace(options, inspect_sha256=run.archive.digest(options.inspect))
    original = run.device._device_reference

    def reference(*a: Any, **k: Any) -> tuple[dict[str, Any], Any]:
        result = original(*a, **k)
        if damage == "uuid":
            parts["hardware"][
                "nvidia_smi_uuid_name_driver"
            ] = "GPU-22222222-2222-2222-2222-222222222222,A800,fixture"
        elif damage == "multiple_devices":
            monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        elif damage == "fp64":
            parts["pipeline"].double()
        elif damage == "nonfinite":
            with torch.no_grad():
                parts["pipeline"].weight.fill_(float("nan"))
        elif damage == "dem":
            parts["template"].dem_text = "changed"
        elif damage == "source":
            monkeypatch.setattr(run, "_fingerprints", lambda: {"changed": "b" * 64})
        elif damage == "protocol":
            parts["protocol"].write_text("changed")
        elif damage == "inspect":
            options.inspect.write_text("changed")
        elif damage == "metadata":
            (run.archive.ARCHIVE / "results.json").write_text("changed")
        elif damage == "precision":

            def refuse() -> None:
                raise ValueError("precision changed")

            monkeypatch.setattr(run.parity, "_precision", refuse)
        return result

    monkeypatch.setattr(run.device, "_device_reference", reference)
    with pytest.raises(ValueError):
        run.preflight(options)
    assert not options.study.exists()


@pytest.fixture
def preparation(
    context: tuple[run.Options, dict[str, Any], list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[run.Options, list[str], dict[str, Any]]:
    options, parts, _ = context
    report = run.preflight(options)

    class Circuit:
        def __str__(self) -> str:
            return "synthetic preparation circuit"

    circuit = Circuit()
    report["inputs"]["circuit_sha256"] = hashlib.sha256(
        str(circuit).encode()
    ).hexdigest()
    monkeypatch.setattr(run, "preflight", lambda opts: json.loads(json.dumps(report)))
    calls: list[str] = []

    def output(name: str, path: Path, filename: str) -> None:
        assert (options.study / "launch.lock").exists()
        calls.append(name)
        path.mkdir(parents=True, exist_ok=False)
        (path / filename).write_text(json.dumps({"software_stage": name}))

    def initialize(source: Path, path: Path) -> dict[str, Any]:
        output("initialize", path, "initialization.json")
        return {}

    def train(
        source: Path, initial: Path, path: Path, *, initialization_sha256: str
    ) -> None:
        assert initialization_sha256 == run.archive.digest(
            initial / "initialization.json"
        )
        output("train", path, "manifest.json")

    def calibrate(
        source: Path, initial: Path, trained: Path, path: Path, *, training_sha256: str
    ) -> dict[str, Any]:
        assert training_sha256 == run.archive.digest(trained / "manifest.json")
        output("calibrate", path, "manifest.json")
        return {}

    def reference(*a: Any, **k: Any) -> Any:
        assert k == {"load_model": False}
        calls.append("reference")
        return SimpleNamespace(circuit=circuit)

    def validate(
        source: Path, initial: Path, trained: Path, cal: Path, path: Path, **k: Any
    ) -> dict[str, Any]:
        assert k["training_sha256"] == run.archive.digest(trained / "manifest.json")
        assert k["calibration_sha256"] == run.archive.digest(cal / "manifest.json")
        assert k["device"] == torch.device("cuda:0") and k["circuit"] is circuit
        output("validation", path, "choices.json")
        return {}

    def probe(
        source: Path,
        initial: Path,
        trained: Path,
        cal: Path,
        choices: Path,
        path: Path,
        **k: Any,
    ) -> Any:
        assert k["expected_choices_sha256"] == run.archive.digest(
            choices / "choices.json"
        )
        assert k["circuit"] is circuit
        output("probes", path, "manifest.json")

    def receipt(
        source: Path,
        initial: Path,
        trained: Path,
        cal: Path,
        choices: Path,
        probes: Path,
        path: Path,
        **k: Any,
    ) -> dict[str, Any]:
        assert k["host"] in ("a800-172", "a800-171")
        assert k["choices_sha256"] == run.archive.digest(choices / "choices.json")
        assert k["probes_sha256"] == run.archive.digest(probes / "manifest.json")
        assert k["ising_weights"] == options.ising_weights
        output("receipt" if k["host"] == "a800-172" else "peer", path, "receipt.json")
        return {}

    monkeypatch.setattr(run.initialization, "initialize", initialize)
    monkeypatch.setattr(run.training, "run_training", train)
    monkeypatch.setattr(run.selection, "calibrate", calibrate)
    monkeypatch.setattr(run.comparison, "reference", reference)
    monkeypatch.setattr(run.comparison, "global_decoder", lambda c: None)
    monkeypatch.setattr(run.validation, "validate_and_freeze", validate)
    monkeypatch.setattr(run.parity, "capture_probes", probe)
    monkeypatch.setattr(run.parity, "record_device_receipt", receipt)
    return options, calls, parts


def test_preparation_order_frozen_handoff_and_nonrepeatable_lock(
    preparation: tuple[run.Options, list[str], dict[str, Any]],
) -> None:
    options, calls, _ = preparation
    digest = run.prepare(options)
    assert calls == [
        "initialize",
        "train",
        "calibrate",
        "reference",
        "validation",
        "probes",
        "receipt",
    ]
    plan = json.loads((options.study / "plan.json").read_text())
    ready = json.loads((options.study / "ready.json").read_text())
    assert plan["deadline"] - plan["started"] == 86400
    assert digest == run.archive.digest(options.study / "ready.json")
    assert ready["plan_sha256"] == run.archive.digest(options.study / "plan.json")
    assert set(ready["files"]) == {
        "initial",
        "trained",
        "calibration",
        "choices",
        "probes",
        "receipt-172",
    }
    for name, files in ready["files"].items():
        assert files == run.validation._files(options.study / name)
    with pytest.raises(FileExistsError):
        run.prepare(options)
    assert len(calls) == 7


@pytest.mark.parametrize(
    "stage",
    [
        "initialization",
        "training",
        "calibration",
        "reference",
        "validation",
        "probes",
        "receipt",
    ],
)
def test_stage_failure_keeps_partial_and_cannot_retry(
    preparation: tuple[run.Options, list[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    options, calls, _ = preparation
    owners = {
        "initialization": (run.initialization, "initialize"),
        "training": (run.training, "run_training"),
        "calibration": (run.selection, "calibrate"),
        "reference": (run.comparison, "reference"),
        "validation": (run.validation, "validate_and_freeze"),
        "probes": (run.parity, "capture_probes"),
        "receipt": (run.parity, "record_device_receipt"),
    }
    owner, attr = owners[stage]

    def fail(*a: Any, **k: Any) -> None:
        raise RuntimeError("synthetic failed " + stage)

    monkeypatch.setattr(owner, attr, fail)
    with pytest.raises(RuntimeError, match="synthetic failed"):
        run.prepare(options)
    before = list(calls)
    assert (options.study / "launch.lock").exists() and not (
        options.study / "ready.json"
    ).exists()
    with pytest.raises(FileExistsError):
        run.prepare(options)
    assert calls == before


@pytest.mark.parametrize(
    "damage",
    [
        "source",
        "protocol",
        "metadata",
        "weights",
        "lock",
        "inspect",
        "plan",
        "control",
        "completed_stage",
        "extra",
        "deadline",
    ],
)
def test_preparation_seals_prior_outputs_inputs_and_common_deadline(
    preparation: tuple[run.Options, list[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    options, calls, parts = preparation
    original = run.training.run_training

    def mutate(*a: Any, **k: Any) -> None:
        original(*a, **k)
        if damage == "source":
            monkeypatch.setattr(run, "_fingerprints", lambda: {"changed": "b" * 64})
        elif damage == "protocol":
            parts["protocol"].write_text("changed")
        elif damage == "metadata":
            (run.archive.ARCHIVE / "results.json").write_text("changed")
        elif damage == "weights":
            options.ising_weights.write_bytes(b"changed")
        elif damage == "lock":
            options.dependency_lock.write_text("changed")
        elif damage == "inspect":
            options.inspect.write_text("changed")
        elif damage == "plan":
            (options.study / "plan.json").write_text("changed")
        elif damage == "control":
            (options.study / "launch.lock").write_text("changed")
        elif damage == "completed_stage":
            (options.study / "initial/initialization.json").write_text("changed")
        elif damage == "extra":
            (options.study / "extra").write_text("extra")
        else:
            raise TimeoutError("synthetic common deadline expired")

    monkeypatch.setattr(run.training, "run_training", mutate)
    with pytest.raises((ValueError, TimeoutError)):
        run.prepare(options)
    assert calls == ["initialize", "train"]
    assert not (options.study / "ready.json").exists()


def test_peer_host_refuses_before_preflight(
    preparation: tuple[run.Options, list[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options, calls, _ = preparation

    def forbidden(opts: run.Options) -> None:
        raise AssertionError("wrong host entered preflight")

    monkeypatch.setattr(run, "preflight", forbidden)
    with pytest.raises(ValueError, match="primary host"):
        run.prepare(replace(options, host="a800-171"))
    assert not calls and not options.study.exists()


@pytest.fixture
def phases(
    preparation: tuple[run.Options, list[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[run.Options, list[str], dict[str, Any]]:
    """Mock only stage/hardware boundaries; exercise real locks, hashes and deadline."""
    options, calls, parts = preparation

    def scope(*a: Any, **k: Any) -> dict[str, Any]:
        for path in k["readonly_paths"]:
            run.coordinator._read_only(parts["raw"], path)
        return parts["raw"]

    monkeypatch.setattr(run.coordinator, "_container_scope", scope)

    def primary(*a: Any, **k: Any) -> None:
        assert k["host"] == "a800-172"
        for path, sha in a[6].values():
            assert run.archive.digest(path) == sha
        assert k["choices_sha256"] == run.archive.digest(a[4] / "choices.json")
        assert k["probes_sha256"] == run.archive.digest(a[5] / "manifest.json")
        calls.append("primary")
        a[7].mkdir()
        (a[7] / "manifest.json").write_text('{"software_primary":true}')

    def summary(path: Path, output: Path, *, manifest_sha256: str) -> None:
        assert run.archive.digest(path / "manifest.json") == manifest_sha256
        calls.append("summary")
        output.write_text('{"software_summary":true}')

    monkeypatch.setattr(run.primary, "run_primary", primary)
    monkeypatch.setattr(run.primary, "summarize_primary", summary)
    return options, calls, parts


def test_internal_phases_share_frozen_deadline_without_repeating_stages(
    phases: tuple[run.Options, list[str], dict[str, Any]],
) -> None:
    options, calls, _ = phases
    sha = run.prepare(options)
    before = list(calls)
    peer = options.study.parent / "peer-output"
    peer_sha = run.peer(replace(options, host="a800-171"), sha, peer)
    assert calls == before + ["peer"]
    run.finish(options, sha, peer / "device/receipt.json", peer_sha)
    assert calls == before + ["peer", "primary", "summary"]
    complete = json.loads((options.study / "complete.json").read_text())
    assert (
        complete["handoff_sha256"] == sha
        and complete["peer_receipt_sha256"] == peer_sha
    )
    assert complete["summary_sha256"] == run.archive.digest(
        options.study / "summary.json"
    )
    with pytest.raises(FileExistsError):
        run.finish(options, sha, peer / "device/receipt.json", peer_sha)
    with pytest.raises(FileExistsError):
        run.peer(replace(options, host="a800-171"), sha, peer)


@pytest.mark.parametrize(
    "damage",
    [
        "hash",
        "plan",
        "stage",
        "extra",
        "expired",
        "interval",
        "source",
        "control",
        "inventory",
        "purpose",
    ],
)
def test_peer_refuses_damaged_handoff_before_receipt(
    phases: tuple[run.Options, list[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    options, calls, _ = phases
    sha = run.prepare(options)
    before = list(calls)
    ready_path = options.study / "ready.json"
    ready = json.loads(ready_path.read_text())
    if damage == "hash":
        sha = "0" * 64
    elif damage == "plan":
        (options.study / "plan.json").write_text("changed")
    elif damage == "stage":
        (options.study / "trained/manifest.json").write_text("changed")
    elif damage == "extra":
        (options.study / "extra").touch()
    elif damage == "source":
        monkeypatch.setattr(run, "_fingerprints", lambda: {"changed": "a" * 64})
    elif damage in ("expired", "interval"):
        plan_path = options.study / "plan.json"
        plan = json.loads(plan_path.read_text())
        if damage == "expired":
            plan["started"] -= 90000
            plan["deadline"] -= 90000
        else:
            plan["deadline"] += 1
        plan_path.write_text(json.dumps(plan))
        ready["control"]["plan.json"] = ready["plan_sha256"] = run.archive.digest(
            plan_path
        )
    elif damage == "control":
        ready["control"].pop("launch.lock")
    elif damage == "inventory":
        ready["files"].pop("initial")
    else:
        ready["purpose"] = "wrong"
    if damage in ("expired", "interval", "control", "inventory", "purpose"):
        ready_path.write_text(json.dumps(ready))
        sha = run.archive.digest(ready_path)
    with pytest.raises((ValueError, TimeoutError)):
        run.peer(replace(options, host="a800-171"), sha, options.study.parent / "peer")
    assert calls == before


@pytest.mark.parametrize("phase", ["peer", "primary", "summary"])
def test_phase_failure_retains_lock_and_rejects_retry(
    phases: tuple[run.Options, list[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    options, calls, _ = phases
    sha = run.prepare(options)
    peer = options.study.parent / "peer"

    def fail(*a: Any, **k: Any) -> None:
        raise RuntimeError("synthetic phase failure")

    if phase == "peer":
        monkeypatch.setattr(run.parity, "record_device_receipt", fail)
        with pytest.raises(RuntimeError):
            run.peer(replace(options, host="a800-171"), sha, peer)
        with pytest.raises(FileExistsError):
            run.peer(replace(options, host="a800-171"), sha, peer)
        assert (peer / "peer.lock").exists()
    else:
        peer_sha = run.peer(replace(options, host="a800-171"), sha, peer)
        monkeypatch.setattr(
            run.primary,
            "run_primary" if phase == "primary" else "summarize_primary",
            fail,
        )
        with pytest.raises(RuntimeError):
            run.finish(options, sha, peer / "device/receipt.json", peer_sha)
        with pytest.raises(FileExistsError):
            run.finish(options, sha, peer / "device/receipt.json", peer_sha)
        assert (options.study / "finish.lock").exists()
    assert not (options.study / "complete.json").exists()


@pytest.mark.parametrize(
    "damage", ["receipt", "source", "stage", "primary", "control", "extra"]
)
def test_finish_rechecks_evidence_after_summary(
    phases: tuple[run.Options, list[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    options, calls, _ = phases
    sha = run.prepare(options)
    peer = options.study.parent / "peer"
    peer_sha = run.peer(replace(options, host="a800-171"), sha, peer)
    original = run.primary.summarize_primary

    def mutate(*a: Any, **k: Any) -> None:
        original(*a, **k)
        if damage == "receipt":
            (peer / "device/receipt.json").write_text("changed")
        elif damage == "source":
            monkeypatch.setattr(run, "_fingerprints", lambda: {"changed": "b" * 64})
        elif damage == "stage":
            (options.study / "initial/initialization.json").write_text("changed")
        elif damage == "primary":
            (options.study / "primary/manifest.json").write_text("changed")
        elif damage == "control":
            (options.study / "finish.lock").write_text("changed")
        else:
            (options.study / "extra").touch()

    monkeypatch.setattr(run.primary, "summarize_primary", mutate)
    with pytest.raises(ValueError):
        run.finish(options, sha, peer / "device/receipt.json", peer_sha)
    assert not (options.study / "complete.json").exists()


def test_run_waits_for_externally_delivered_receipt_without_new_prepare(
    phases: tuple[run.Options, list[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options, calls, _ = phases
    receipt = options.study.parent / "delivered.json"
    incoming = options.study.parent / "delivered.sha256"
    original = run.prepare

    def prepare(opts: run.Options) -> str:
        sha = original(opts)
        receipt.write_text('{"software_delivery":true}')
        incoming.write_text(run.archive.digest(receipt))
        return sha

    monkeypatch.setattr(run, "prepare", prepare)
    run.run(options, incoming, receipt)
    assert calls.count("initialize") == 1 and calls[-2:] == ["primary", "summary"]
    assert (options.study / "complete.json").exists()


@pytest.mark.parametrize(
    "damage", ["stale_hash", "stale_receipt", "bad_hash", "wrong_hash"]
)
def test_run_rejects_stale_or_invalid_external_delivery(
    phases: tuple[run.Options, list[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    options, calls, _ = phases
    receipt = options.study.parent / "delivered.json"
    incoming = options.study.parent / "delivered.sha256"
    if damage.startswith("stale"):
        (incoming if damage == "stale_hash" else receipt).touch()
    else:
        original = run.prepare

        def prepare(opts: run.Options) -> str:
            sha = original(opts)
            receipt.write_text("software only")
            incoming.write_text("invalid" if damage == "bad_hash" else "0" * 64)
            return sha

        monkeypatch.setattr(run, "prepare", prepare)
    with pytest.raises((ValueError, FileExistsError)):
        run.run(options, incoming, receipt)
    assert "primary" not in calls
    assert not (options.study / "complete.json").exists()


@pytest.mark.parametrize("phase", ["peer", "finish"])
def test_phase_host_guard_precedes_any_preflight(
    phases: tuple[run.Options, list[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    options, calls, _ = phases

    def forbidden(*a: Any, **k: Any) -> None:
        raise AssertionError("wrong host reached preflight")

    monkeypatch.setattr(run, "preflight", forbidden)
    with pytest.raises(ValueError):
        if phase == "peer":
            run.peer(options, "0" * 64, options.study.parent / "peer")
        else:
            run.finish(
                replace(options, host="a800-171"), "0" * 64, Path("unused"), "0" * 64
            )
    assert not calls


@pytest.mark.parametrize("damage", ["input", "control", "extra"])
def test_peer_rechecks_handoff_and_own_output_after_device_stage(
    phases: tuple[run.Options, list[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    options, calls, _ = phases
    sha = run.prepare(options)
    output = options.study.parent / "peer"
    original = run.parity.record_device_receipt

    def mutate(*a: Any, **k: Any) -> dict[str, Any]:
        result = original(*a, **k)
        if damage == "input":
            (options.study / "probes/manifest.json").write_text("changed")
        elif damage == "control":
            (output / "peer.lock").write_text("changed")
        else:
            (output / "extra").touch()
        return result

    monkeypatch.setattr(run.parity, "record_device_receipt", mutate)
    with pytest.raises(ValueError):
        run.peer(replace(options, host="a800-171"), sha, output)
    with pytest.raises(FileExistsError):
        run.peer(replace(options, host="a800-171"), sha, output)


def test_waiting_deadline_failure_never_starts_primary(
    phases: tuple[run.Options, list[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options, calls, _ = phases

    def expired(seconds: float) -> None:
        raise TimeoutError("synthetic waiting deadline")

    monkeypatch.setattr(run.time, "sleep", expired)
    with pytest.raises(TimeoutError):
        run.run(
            options,
            options.study.parent / "incoming.sha256",
            options.study.parent / "incoming.json",
        )
    assert "primary" not in calls and (options.study / "ready.json").exists()
    with pytest.raises(FileExistsError):
        run.run(
            options,
            options.study.parent / "incoming.sha256",
            options.study.parent / "incoming.json",
        )

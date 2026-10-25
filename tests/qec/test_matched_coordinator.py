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

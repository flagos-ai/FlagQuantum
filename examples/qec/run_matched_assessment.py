"""Container preflight and internal primary-host matched-study preparation.

No launch CLI or trusted SSH controller is supplied. The trusted SSH controller must supply
and freeze the real inspect document; parsing a host label is not attestation.
Complete workflow acceptance remains required before any samples or updates.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from examples.qec import compare_ising as comparison
from examples.qec import continued_parity as device
from examples.qec import diagnose_predecoder as archive
from examples.qec import ising_layout as layout
from examples.qec import matched_checkpoints as checkpoints
from examples.qec import matched_initialization as initialization
from examples.qec import matched_parity as parity
from examples.qec import matched_primary as primary
from examples.qec import matched_selection as selection
from examples.qec import matched_training as training
from examples.qec import matched_validation as validation
from examples.qec import run_continued_assessment as coordinator


@dataclass(frozen=True)
class Options:
    ising_source: Path
    ising_weights: Path
    inspect: Path
    inspect_sha256: str
    host: str
    study: Path
    dependency_lock: Path


def _fingerprints() -> dict[str, str]:
    result = primary._sources()
    for path in (Path(__file__), Path(coordinator.__file__)):
        result[str(path.relative_to(archive.ROOT))] = archive.digest(path)
    return result


def preflight(options: Options) -> dict[str, Any]:
    """Verify the inspected container, pinned layout and reference without draws.

    The returned metadata is a software/environment preflight, never a device
    prediction receipt, formal checkpoint bundle or study launch authorization.
    """
    checkpoints._sha256(options.inspect_sha256)
    if options.inspect.is_symlink():
        raise ValueError("controller inspect must not be a symlink")
    raw = coordinator._container_scope(
        options.inspect,
        expected_sha256=options.inspect_sha256,
        host=options.host,
        readonly_paths=(
            archive.ROOT,
            options.ising_source,
            options.ising_weights,
            options.inspect,
            options.dependency_lock,
        ),
    )
    if raw["HostConfig"].get("AutoRemove") is not True:
        raise ValueError("matched containers must be automatically removed")
    # Old checkpoints/arrays are neither required nor admitted for fresh models.
    stem = "qec_ai_ising_a800_20261025/"
    coordinator._source_evidence(
        {
            stem + "a800_172/manifest.json",
            stem + "a800_172/results.json",
            stem + "requirements.lock",
        }
    )
    lock_sha = archive.digest(options.dependency_lock)
    if lock_sha != archive.digest(
        archive.ROOT / "docs/development/evidence" / stem / "requirements.lock"
    ):
        raise ValueError("producing dependency lock mismatch")
    anchor = json.loads((archive.ARCHIVE / "manifest.json").read_text())
    if (
        platform.python_version() != anchor["python"]
        or str(torch.__version__) != anchor["torch"]
    ):
        raise ValueError("preflight runtime differs from the pinned environment")
    if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
        raise ValueError("preflight requires autocast disabled")
    if (
        archive.digest(options.ising_weights) != comparison.FAST_SHA256
        or comparison.source_digest(options.ising_source)
        != comparison.ISING_SOURCE_SHA256
    ):
        raise ValueError("preflight requires pinned vendor source and Fast weights")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    sources = _fingerprints()
    protocols = {p.name: archive.digest(p) for p in checkpoints.PROTOCOLS}
    metadata = {
        p: archive.digest(p)
        for p in (archive.ARCHIVE / "manifest.json", archive.ARCHIVE / "results.json")
    }
    with torch.random.fork_rng(devices=[]):
        template, inputs = layout.build_model(options.ising_source)
        layout.verify_encoding(template)
        hardware, reference = device._device_reference(
            options.ising_source,
            options.ising_weights,
            circuit_sha256=str(inputs["circuit_sha256"]),
            dem_sha256=str(inputs["dem_sha256"]),
        )
    uuid = device._hardware_uuid(hardware)
    requests = raw["HostConfig"]["DeviceRequests"]
    if (
        len(requests) != 1
        or len(requests[0]["DeviceIDs"]) != 1
        or str(requests[0]["DeviceIDs"][0]).strip().lower() != uuid
        or torch.cuda.device_count() != 1
    ):
        raise ValueError("controller must bind the sole visible A800 UUID")
    parity._precision()
    if any(p.dtype != torch.float32 for p in reference.pipeline.parameters()) or any(
        not bool(torch.isfinite(v).all())
        for v in reference.pipeline.state_dict().values()
    ):
        raise ValueError("preflight reference must have finite FP32 parameters")
    if hashlib.sha256(template.dem_text.encode()).hexdigest() != inputs["dem_sha256"]:
        raise ValueError("preflight native layout changed its DEM identity")
    if (
        torch.get_num_threads() != 1
        or torch.get_num_interop_threads() != 1
        or _fingerprints() != sources
        or {p.name: archive.digest(p) for p in checkpoints.PROTOCOLS} != protocols
        or any(archive.digest(p) != sha for p, sha in metadata.items())
        or archive.digest(options.inspect) != options.inspect_sha256
        or archive.digest(options.dependency_lock) != lock_sha
        or archive.digest(options.ising_weights) != comparison.FAST_SHA256
        or comparison.source_digest(options.ising_source)
        != comparison.ISING_SOURCE_SHA256
    ):
        raise ValueError("preflight inputs, metadata, source or protocols changed")
    return {
        "purpose": "matched_container_preflight",
        "host": options.host,
        "container_id": raw["Id"],
        "controller_inspect_sha256": options.inspect_sha256,
        "protocols": protocols,
        "source_files": sources,
        "dependency_lock_sha256": lock_sha,
        "fast_weights_sha256": comparison.FAST_SHA256,
        "ising_source_sha256": comparison.ISING_SOURCE_SHA256,
        "metadata_sha256": {p.name: sha for p, sha in metadata.items()},
        "inputs": inputs,
        "hardware": hardware,
        "pipeline_state_sha256": archive.state_digest(reference.pipeline.state_dict()),
        "new_samples": 0,
        "new_optimizer_steps": 0,
        "formal_launch_authorized": False,
    }


def _verify_preflight(options: Options, report: dict[str, Any]) -> None:
    if (
        _fingerprints() != report["source_files"]
        or {p.name: archive.digest(p) for p in checkpoints.PROTOCOLS}
        != report["protocols"]
        or archive.digest(options.inspect) != options.inspect_sha256
        or archive.digest(options.dependency_lock) != report["dependency_lock_sha256"]
        or archive.digest(options.ising_weights) != report["fast_weights_sha256"]
        or comparison.source_digest(options.ising_source)
        != report["ising_source_sha256"]
        or any(
            archive.digest(archive.ARCHIVE / p) != sha
            for p, sha in report["metadata_sha256"].items()
        )
    ):
        raise ValueError("matched preflight source or inputs changed")


def prepare(options: Options) -> str:
    """Internal primary-host preparation; no formal invocation before full acceptance.

    The existing stage implementations own every draw, update, checkpoint and
    decision. This function owns only the fresh study lock, common deadline,
    ordered composition and frozen handoff. A failed study cannot be resumed.
    """
    if options.host != "a800-172":
        raise ValueError("prepare runs only on the primary host")
    if options.study.exists():
        raise FileExistsError(options.study)
    report = preflight(options)
    protocols = {p.name: archive.digest(p) for p in checkpoints.PROTOCOLS}
    if report["source_files"] != _fingerprints() or report["protocols"] != protocols:
        raise ValueError("preflight source or protocol changed before prepare")
    options.study.mkdir(parents=True, exist_ok=False)
    started = time.time()
    end = started + coordinator.LIMIT
    coordinator._write(
        options.study / "launch.lock", {"pid": os.getpid(), "started": started}
    )
    coordinator._write(
        options.study / "plan.json",
        {
            "purpose": "matched_prepare_plan",
            "preflight": report,
            "started": started,
            "deadline": end,
        },
    )
    control = {
        name: archive.digest(options.study / name)
        for name in ("launch.lock", "plan.json")
    }
    files: dict[str, dict[str, str]] = {}

    def seal(name: str | None = None) -> None:
        _verify_preflight(options, report)
        expected = {"launch.lock", "plan.json", *files}
        if name is not None:
            expected.add(name)
        if (
            set(p.name for p in options.study.iterdir()) != expected
            or any(
                archive.digest(options.study / p) != sha for p, sha in control.items()
            )
            or any(
                validation._files(options.study / p) != inventory
                for p, inventory in files.items()
            )
        ):
            raise ValueError("prepare source, inputs, plan or completed stage changed")
        if name is not None:
            files[name] = validation._files(options.study / name)

    def phase(name: str) -> None:
        seal()
        print(json.dumps({"phase": name, "deadline": end}), flush=True)

    initial = options.study / "initial"
    trained = options.study / "trained"
    calibration = options.study / "calibration"
    choices = options.study / "choices"
    probes = options.study / "probes"
    receipt = options.study / "receipt-172"
    with coordinator.deadline(end):
        phase("initialization")
        initialization.initialize(options.ising_source, initial)
        initial_sha = archive.digest(initial / "initialization.json")
        seal("initial")
        phase("training")
        training.run_training(
            options.ising_source, initial, trained, initialization_sha256=initial_sha
        )
        training_sha = archive.digest(trained / "manifest.json")
        seal("trained")
        phase("calibration")
        selection.calibrate(
            options.ising_source,
            initial,
            trained,
            calibration,
            training_sha256=training_sha,
        )
        calibration_sha = archive.digest(calibration / "manifest.json")
        seal("calibration")
        phase("validation")
        ref = comparison.reference(
            options.ising_source,
            None,
            comparison.Config(True, distance=9, rounds=9),
            load_model=False,
        )
        if (
            hashlib.sha256(str(ref.circuit).encode()).hexdigest()
            != report["inputs"]["circuit_sha256"]
        ):
            raise ValueError("prepare validation circuit differs from preflight")
        validation.validate_and_freeze(
            options.ising_source,
            initial,
            trained,
            calibration,
            choices,
            training_sha256=training_sha,
            calibration_sha256=calibration_sha,
            circuit=ref.circuit,
            global_decode=comparison.global_decoder(ref.circuit),
            device=torch.device("cuda:0"),
        )
        choices_sha = archive.digest(choices / "choices.json")
        seal("choices")
        phase("physical_probe")
        parity.capture_probes(
            options.ising_source,
            initial,
            trained,
            calibration,
            choices,
            probes,
            expected_choices_sha256=choices_sha,
            circuit=ref.circuit,
        )
        probes_sha = archive.digest(probes / "manifest.json")
        seal("probes")
        phase("device_parity")
        parity.record_device_receipt(
            options.ising_source,
            initial,
            trained,
            calibration,
            choices,
            probes,
            receipt,
            host=options.host,
            choices_sha256=choices_sha,
            probes_sha256=probes_sha,
            ising_weights=options.ising_weights,
        )
        receipt_sha = archive.digest(receipt / "receipt.json")
        seal("receipt-172")
        seal()
        coordinator._write(
            options.study / "ready.json",
            {
                "purpose": "matched_prepare_handoff",
                "plan_sha256": control["plan.json"],
                "control": control,
                "initialization_sha256": initial_sha,
                "training_sha256": training_sha,
                "calibration_sha256": calibration_sha,
                "choices_sha256": choices_sha,
                "probes_sha256": probes_sha,
                "receipt172_sha256": receipt_sha,
                "files": files,
            },
        )
    return archive.digest(options.study / "ready.json")


def _handoff(
    options: Options,
    expected: str,
    report: dict[str, Any],
    *,
    extras: tuple[str, ...] = (),
) -> dict[str, Any]:
    checkpoints._sha256(expected)
    ready_path = options.study / "ready.json"
    if archive.digest(ready_path) != expected:
        raise ValueError("frozen matched handoff hash mismatch")
    ready: dict[str, Any] = json.loads(ready_path.read_text())
    controls = ready["control"]
    if (
        set(controls) != {"launch.lock", "plan.json"}
        or ready["plan_sha256"] != controls["plan.json"]
    ):
        raise ValueError("matched handoff control inventory mismatch")
    if any(archive.digest(options.study / p) != sha for p, sha in controls.items()):
        raise ValueError("matched plan or launch lock changed")
    plan = json.loads((options.study / "plan.json").read_text())
    if plan["purpose"] != "matched_prepare_plan":
        raise ValueError("wrong matched preparation purpose")
    stable = (
        "protocols",
        "source_files",
        "metadata_sha256",
        "dependency_lock_sha256",
        "fast_weights_sha256",
        "ising_source_sha256",
        "inputs",
        "pipeline_state_sha256",
    )
    if any(plan["preflight"][key] != report[key] for key in stable):
        raise ValueError("matched handoff source or inputs differ across stages")
    started, end = plan["started"], plan["deadline"]
    if (
        any(type(x) not in (int, float) or not math.isfinite(x) for x in (started, end))
        or end - started != coordinator.LIMIT
    ):
        raise ValueError("invalid fixed matched deadline")
    if not started <= time.time() < end:
        raise TimeoutError("formal 24-hour matched deadline expired or invalid")
    stages = {
        "initial": ("initialization.json", "initialization_sha256"),
        "trained": ("manifest.json", "training_sha256"),
        "calibration": ("manifest.json", "calibration_sha256"),
        "choices": ("choices.json", "choices_sha256"),
        "probes": ("manifest.json", "probes_sha256"),
        "receipt-172": ("receipt.json", "receipt172_sha256"),
    }
    if ready["purpose"] != "matched_prepare_handoff" or set(ready["files"]) != set(
        stages
    ):
        raise ValueError("matched handoff stage inventory mismatch")
    if set(p.name for p in options.study.iterdir()) != {
        "ready.json",
        *controls,
        *stages,
        *extras,
    }:
        raise ValueError("matched handoff output inventory mismatch")
    for name, (filename, key) in stages.items():
        if (
            validation._files(options.study / name) != ready["files"][name]
            or archive.digest(options.study / name / filename) != ready[key]
        ):
            raise ValueError("completed matched stage changed before handoff")
    _verify_preflight(options, report)
    return {**ready, "deadline": end}


def peer(options: Options, handoff_sha256: str, output: Path) -> str:
    """Read the frozen preparation on A800-171; write only one new receipt."""
    if options.host != "a800-171":
        raise ValueError("peer parity runs only on the peer host")
    if output.exists():
        raise FileExistsError(output)
    if output.resolve().is_relative_to(options.study.resolve()):
        raise ValueError("peer output must be outside the read-only handoff")
    report = preflight(options)
    coordinator._container_scope(
        options.inspect,
        expected_sha256=options.inspect_sha256,
        host=options.host,
        readonly_paths=(options.study,),
    )
    ready = _handoff(options, handoff_sha256, report)
    with coordinator.deadline(ready["deadline"]):
        output.mkdir(parents=True, exist_ok=False)
        coordinator._write(output / "peer.lock", {"handoff_sha256": handoff_sha256})
        lock_sha = archive.digest(output / "peer.lock")
        parity.record_device_receipt(
            options.ising_source,
            options.study / "initial",
            options.study / "trained",
            options.study / "calibration",
            options.study / "choices",
            options.study / "probes",
            output / "device",
            host=options.host,
            choices_sha256=ready["choices_sha256"],
            probes_sha256=ready["probes_sha256"],
            ising_weights=options.ising_weights,
        )
        _handoff(options, handoff_sha256, report)
        if (
            set(p.name for p in output.iterdir()) != {"peer.lock", "device"}
            or archive.digest(output / "peer.lock") != lock_sha
        ):
            raise ValueError("peer control or output inventory changed")
        return archive.digest(output / "device/receipt.json")


def finish(
    options: Options, handoff_sha256: str, peer_receipt: Path, peer_receipt_sha256: str
) -> None:
    """One primary and summary, using the preparation's remaining fixed deadline."""
    if options.host != "a800-172":
        raise ValueError("finish runs only on the primary host")
    if (options.study / "finish.lock").exists():
        raise FileExistsError(options.study / "finish.lock")
    checkpoints._sha256(peer_receipt_sha256)
    if archive.digest(peer_receipt) != peer_receipt_sha256:
        raise ValueError("externally delivered peer receipt hash mismatch")
    report = preflight(options)
    coordinator._container_scope(
        options.inspect,
        expected_sha256=options.inspect_sha256,
        host=options.host,
        readonly_paths=(peer_receipt,),
    )
    ready = _handoff(options, handoff_sha256, report)
    with coordinator.deadline(ready["deadline"]):
        coordinator._write(
            options.study / "finish.lock",
            {
                "handoff_sha256": handoff_sha256,
                "peer_receipt_sha256": peer_receipt_sha256,
            },
        )
        lock_sha = archive.digest(options.study / "finish.lock")
        receipts = {
            "a800-172": (
                options.study / "receipt-172/receipt.json",
                ready["receipt172_sha256"],
            ),
            "a800-171": (peer_receipt, peer_receipt_sha256),
        }
        primary.run_primary(
            options.ising_source,
            options.study / "initial",
            options.study / "trained",
            options.study / "calibration",
            options.study / "choices",
            options.study / "probes",
            receipts,
            options.study / "primary",
            host=options.host,
            choices_sha256=ready["choices_sha256"],
            probes_sha256=ready["probes_sha256"],
            ising_weights=options.ising_weights,
        )
        primary_sha = archive.digest(options.study / "primary/manifest.json")
        primary_files = validation._files(options.study / "primary")
        primary.summarize_primary(
            options.study / "primary",
            options.study / "summary.json",
            manifest_sha256=primary_sha,
        )
        summary_sha = archive.digest(options.study / "summary.json")
        _handoff(
            options,
            handoff_sha256,
            report,
            extras=("finish.lock", "primary", "summary.json"),
        )
        if (
            archive.digest(peer_receipt) != peer_receipt_sha256
            or archive.digest(options.study / "finish.lock") != lock_sha
            or validation._files(options.study / "primary") != primary_files
        ):
            raise ValueError("finish receipt, control or primary output changed")
        coordinator._write(
            options.study / "complete.json",
            {
                "handoff_sha256": handoff_sha256,
                "peer_receipt_sha256": peer_receipt_sha256,
                "primary_manifest_sha256": primary_sha,
                "summary_sha256": summary_sha,
                "completed": time.time(),
            },
        )


def run(options: Options, incoming_sha256_file: Path, peer_receipt: Path) -> None:
    """Await trusted external peer delivery without restarting the fixed study."""
    coordinator._container_scope(
        options.inspect,
        expected_sha256=options.inspect_sha256,
        host=options.host,
        readonly_paths=(incoming_sha256_file, peer_receipt),
    )
    if incoming_sha256_file.exists() or peer_receipt.exists():
        raise FileExistsError("peer delivery must start empty")
    handoff_sha = prepare(options)
    end = json.loads((options.study / "plan.json").read_text())["deadline"]
    with coordinator.deadline(end):
        last_notice = 0.0
        while not incoming_sha256_file.exists():
            if time.monotonic() - last_notice >= 30:
                print(
                    json.dumps(
                        {
                            "phase": "await_peer_receipt",
                            "handoff_sha256": handoff_sha,
                            "deadline": end,
                        }
                    ),
                    flush=True,
                )
                last_notice = time.monotonic()
            time.sleep(1)
        peer_sha = incoming_sha256_file.read_text().strip()
        checkpoints._sha256(peer_sha)
        finish(options, handoff_sha, peer_receipt, peer_sha)

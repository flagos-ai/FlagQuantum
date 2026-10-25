"""Run frozen continued-assessment stages inside controller-bound A800 containers.

The authorized SSH controller supplies hashed Docker inspect data, read-only
source/inputs, isolated devices and a hard 24-hour container timeout. The peer
stage consumes the primary host's exact frozen probe; it never resamples it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import signal
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from examples.qec import compare_ising as comparison
from examples.qec import continued_assessment as inputs
from examples.qec import continued_parity as parity
from examples.qec import continued_primary as primary
from examples.qec import continued_selection as selection
from examples.qec import diagnose_predecoder as archive

LIMIT = 24 * 3600


@dataclass(frozen=True)
class Options:
    original: Path
    continued: Path
    ising_source: Path
    ising_weights: Path
    inspect: Path
    inspect_sha256: str
    host: str
    study: Path
    dependency_lock: Path


def _fingerprints() -> dict[str, str]:
    return primary._sources() | {
        str(Path(__file__).relative_to(archive.ROOT)): archive.digest(Path(__file__))
    }


def _inventory(directory: Path, expected: set[str]) -> None:
    if {p.name for p in directory.iterdir()} != expected or any(
        not p.is_file() or p.is_symlink() for p in directory.iterdir()
    ):
        raise ValueError("input whitelist mismatch; do not mount old score/test arrays")


def _read_only(raw: dict[str, Any], path: Path) -> None:
    resolved = path.resolve()
    mounts = [
        m for m in raw["Mounts"] if resolved.is_relative_to(Path(m["Destination"]))
    ]
    if (
        not mounts
        or max(mounts, key=lambda m: len(m["Destination"]))["RW"] is not False
    ):
        raise ValueError("source and all input mounts must be read-only")


def _container(options: Options) -> dict[str, Any]:
    if archive.digest(options.inspect) != options.inspect_sha256:
        raise ValueError("controller inspect hash mismatch")
    records = json.loads(options.inspect.read_text())
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError("requires one controller-inspected container")
    raw: dict[str, Any] = records[0]
    host = raw["HostConfig"]
    if (
        platform.system() != "Linux"
        or options.host not in ("a800-172", "a800-171")
        or raw["Config"]["Hostname"] != platform.node()
        or raw["Config"]["Hostname"] != raw["Id"][:12]
        or raw["Config"]["Labels"].get("flagquantum.host") != options.host
        or raw["Image"] != parity.IMAGE
        or host["NetworkMode"] != "none"
        or host["NanoCpus"] != 8_000_000_000
        or host["Memory"] != 16 * 1024**3
    ):
        raise ValueError("controller container identity/resource binding mismatch")
    for path in (
        archive.ROOT,
        options.original,
        options.continued,
        options.ising_source,
        options.ising_weights,
        options.inspect,
        options.dependency_lock,
    ):
        _read_only(raw, path)
    return raw


def _source_evidence() -> None:
    root = archive.ROOT / "docs/development/evidence"
    allowed = {
        "qec_ai_ising_a800_20261025/a800_172/" + n
        for n in ("manifest.json", "results.json", "choices.json", "audit.json")
    }
    allowed |= {
        "qec_ai_ising_a800_20261025/requirements.lock",
        "qec_ai_training_response_20261025/audit.json",
    }
    paths = [p for p in root.rglob("*") if p.is_file()]
    if {str(p.relative_to(root)) for p in paths} != allowed or any(
        p.is_symlink() for p in paths
    ):
        raise ValueError(
            "source evidence must contain only required metadata, not old score/test arrays"
        )


def preflight(
    options: Options,
) -> tuple[list[inputs.VerifiedCheckpoint], comparison.Reference, dict[str, str]]:
    """Validate inputs/environment/reference/device, without any sampler call."""
    raw = _container(options)
    _inventory(
        options.original,
        {
            "manifest.json",
            "results.json",
            "choices.json",
            "audit.json",
            "training.dem",
            *(f"native-{s}.pt" for s in archive.SEEDS),
        },
    )
    _inventory(
        options.continued,
        {
            "manifest.json",
            *(f"native-{s}-step-{h}.pt" for s in archive.SEEDS for h in (200, 2000)),
        },
    )
    _source_evidence()
    lock_sha = archive.digest(options.dependency_lock)
    if lock_sha != archive.digest(
        archive.ROOT
        / "docs/development/evidence/qec_ai_ising_a800_20261025/requirements.lock"
    ):
        raise ValueError("producing dependency lock mismatch")
    if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
        raise ValueError("autocast must be disabled")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    models, hashes = inputs.load_six(options.original, options.continued)
    hardware = comparison.cuda_metadata(parity.IMAGE)
    uuid = hardware["nvidia_smi_uuid_name_driver"].split(",")[0].strip()
    devices = raw["HostConfig"]["DeviceRequests"]
    if len(devices) != 1 or devices[0]["DeviceIDs"] != [uuid]:
        raise ValueError("inspected GPU does not match the sole visible device")
    ref = comparison.reference(
        options.ising_source,
        options.ising_weights,
        comparison.Config(False, distance=9, rounds=9, a800=True),
    )
    anchor = json.loads((archive.ARCHIVE / "results.json").read_text())
    if (
        hashlib.sha256(str(ref.circuit).encode()).hexdigest()
        != anchor["circuit_sha256"]
        or hashlib.sha256(
            comparison.training_dem(ref.circuit).to_stim_text().encode()
        ).hexdigest()
        != anchor["training_dem_sha256"]
        or any(
            hashlib.sha256(m.model.dem_text.encode()).hexdigest()
            != anchor["training_dem_sha256"]
            for m in models
        )
    ):
        raise ValueError("reference/native circuit or DEM differs from archive")
    hashes["dependency_lock"] = lock_sha
    hashes["ising_weights"] = archive.digest(options.ising_weights)
    hashes["ising_source"] = comparison.source_digest(options.ising_source)
    return models, ref, hashes


@contextmanager
def deadline(end: float) -> Iterator[None]:
    remaining = end - time.time()
    if not math.isfinite(remaining) or not 0 < remaining <= LIMIT:
        raise TimeoutError("formal 24-hour deadline expired or invalid")

    def expired(signum: int, frame: Any) -> None:
        raise TimeoutError("formal 24-hour deadline expired")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _write(path: Path, value: object) -> None:
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _handoff(options: Options, expected: str, hashes: dict[str, str]) -> dict[str, Any]:
    path = options.study / "ready.json"
    if archive.digest(path) != expected:
        raise ValueError("frozen handoff hash mismatch")
    ready: dict[str, Any] = json.loads(path.read_text())
    plan = options.study / "plan.json"
    if archive.digest(plan) != ready["plan_sha256"]:
        raise ValueError("formal plan hash mismatch")
    raw = json.loads(plan.read_text())
    if (
        raw["protocol_sha256"] != archive.digest(inputs.PROTOCOL)
        or raw["source_sha256"] != _fingerprints()
        or raw["input_sha256"] != hashes
        or raw["deadline"] - raw["started"] != LIMIT
    ):
        raise ValueError("handoff source/input/protocol budget mismatch")
    ready["deadline"] = raw["deadline"]
    return ready


def prepare(options: Options) -> str:
    if options.host != "a800-172":
        raise ValueError("prepare runs only on the primary host")
    if options.study.exists():
        raise FileExistsError(options.study)
    models, ref, hashes = preflight(options)
    options.study.mkdir(parents=True, exist_ok=False)
    started = time.time()
    _write(options.study / "launch.lock", {"pid": os.getpid(), "started": started})
    _write(
        options.study / "plan.json",
        {
            "protocol_sha256": archive.digest(inputs.PROTOCOL),
            "source_sha256": _fingerprints(),
            "input_sha256": hashes,
            "started": started,
            "deadline": started + LIMIT,
            "controller_inspect_sha256": options.inspect_sha256,
        },
    )
    with deadline(started + LIMIT):
        print(json.dumps({"phase": "calibration"}), flush=True)
        selection.calibrate(models, options.study / "calibration")
        print(json.dumps({"phase": "validation_capture"}), flush=True)
        inputs.capture_validation(ref.circuit, options.study / "validation")
        circuit_sha = hashlib.sha256(str(ref.circuit).encode()).hexdigest()
        print(json.dumps({"phase": "validation_selection"}), flush=True)
        selection.freeze_choices(
            models,
            options.study / "calibration",
            options.study / "validation",
            options.study / "choices",
            circuit_sha256=circuit_sha,
            calibration_sha256=archive.digest(
                options.study / "calibration/manifest.json"
            ),
            global_decode=comparison.global_decoder(ref.circuit),
            device=torch.device("cuda:0"),
        )
        choice_sha = archive.digest(options.study / "choices/choices.json")
        print(json.dumps({"phase": "physical_probe"}), flush=True)
        parity.capture_probes(
            ref.circuit,
            options.study / "choices",
            models,
            options.study / "probes",
            expected_choices_sha256=choice_sha,
            circuit_sha256=circuit_sha,
        )
        probe_sha = archive.digest(options.study / "probes/manifest.json")
        print(json.dumps({"phase": "device_parity"}), flush=True)
        parity.record_device_receipt(
            models,
            options.study / "choices",
            options.study / "probes",
            options.study / "receipt-172",
            host=options.host,
            choices_sha256=choice_sha,
            probes_sha256=probe_sha,
            ising_source=options.ising_source,
            ising_weights=options.ising_weights,
        )
        _write(
            options.study / "ready.json",
            {
                "plan_sha256": archive.digest(options.study / "plan.json"),
                "choices_sha256": choice_sha,
                "probes_sha256": probe_sha,
                "receipt172_sha256": archive.digest(
                    options.study / "receipt-172/receipt.json"
                ),
            },
        )
    return archive.digest(options.study / "ready.json")


def peer(options: Options, handoff_sha256: str, output: Path) -> str:
    if options.host != "a800-171":
        raise ValueError("peer receipt runs only on the peer host")
    models, _, hashes = preflight(options)
    ready = _handoff(options, handoff_sha256, hashes)
    with deadline(ready["deadline"]):
        print(json.dumps({"phase": "device_parity"}), flush=True)
        parity.record_device_receipt(
            models,
            options.study / "choices",
            options.study / "probes",
            output,
            host=options.host,
            choices_sha256=ready["choices_sha256"],
            probes_sha256=ready["probes_sha256"],
            ising_source=options.ising_source,
            ising_weights=options.ising_weights,
        )
    return archive.digest(output / "receipt.json")


def finish(options: Options, handoff_sha256: str, peer_receipt_sha256: str) -> None:
    if options.host != "a800-172":
        raise ValueError("primary finish runs only on the primary host")
    models, _, hashes = preflight(options)
    ready = _handoff(options, handoff_sha256, hashes)
    with deadline(ready["deadline"]):
        _write(options.study / "finish.lock", {"pid": os.getpid()})
        receipts = {
            "a800-172": (
                options.study / "receipt-172/receipt.json",
                ready["receipt172_sha256"],
            ),
            "a800-171": (
                options.study / "receipt-171/receipt.json",
                peer_receipt_sha256,
            ),
        }
        print(json.dumps({"phase": "primary"}), flush=True)
        primary.run_primary(
            models,
            options.study / "choices",
            options.study / "probes",
            receipts,
            options.study / "primary",
            choices_sha256=ready["choices_sha256"],
            probes_sha256=ready["probes_sha256"],
            ising_source=options.ising_source,
            ising_weights=options.ising_weights,
        )
        print(json.dumps({"phase": "summary"}), flush=True)
        primary.summarize_primary(
            options.study / "primary",
            options.study / "summary.json",
            manifest_sha256=archive.digest(options.study / "primary/manifest.json"),
        )
        _write(
            options.study / "complete.json",
            {
                "summary_sha256": archive.digest(options.study / "summary.json"),
                "completed": time.time(),
            },
        )


def run(options: Options, incoming_sha256_file: Path) -> None:
    """Keep the primary CUDA process alive across the externally delivered peer stage."""
    _read_only(_container(options), incoming_sha256_file)
    if incoming_sha256_file.exists():
        raise FileExistsError("peer delivery must start empty")
    handoff_sha = prepare(options)
    plan = json.loads((options.study / "plan.json").read_text())
    with deadline(plan["deadline"]):
        last_notice = 0.0
        while not incoming_sha256_file.exists():
            if time.monotonic() - last_notice >= 30:
                print(
                    json.dumps(
                        {
                            "phase": "await_peer_receipt",
                            "handoff_sha256": handoff_sha,
                            "deadline": plan["deadline"],
                        }
                    ),
                    flush=True,
                )
                last_notice = time.monotonic()
            time.sleep(1)
        peer_sha = incoming_sha256_file.read_text().strip()
        if len(peer_sha) != 64 or any(c not in "0123456789abcdef" for c in peer_sha):
            raise ValueError("invalid externally delivered peer receipt hash")
        finish(options, handoff_sha, peer_sha)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("preflight", "prepare", "peer", "finish", "run")
    )
    for name in (
        "original",
        "continued",
        "ising-source",
        "ising-weights",
        "inspect",
        "study",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--inspect-sha256", required=True)
    parser.add_argument("--dependency-lock", type=Path, required=True)
    parser.add_argument("--host", choices=("a800-172", "a800-171"), required=True)
    parser.add_argument("--handoff-sha256")
    parser.add_argument("--peer-receipt-sha256")
    parser.add_argument("--receipt-directory", type=Path)
    parser.add_argument("--incoming-sha256-file", type=Path)
    args = parser.parse_args()
    options = Options(
        args.original,
        args.continued,
        args.ising_source,
        args.ising_weights,
        args.inspect,
        args.inspect_sha256,
        args.host,
        args.study,
        args.dependency_lock,
    )
    if args.stage in ("peer", "finish") and args.handoff_sha256 is None:
        parser.error("peer/finish require --handoff-sha256")
    if args.stage == "peer" and args.receipt_directory is None:
        parser.error("peer requires --receipt-directory")
    if args.stage == "finish" and args.peer_receipt_sha256 is None:
        parser.error("finish requires --peer-receipt-sha256")
    if args.stage == "run" and args.incoming_sha256_file is None:
        parser.error("run requires --incoming-sha256-file")
    if args.stage == "run":
        run(options, args.incoming_sha256_file)
    elif args.stage == "preflight":
        models, _, _ = preflight(options)
        print(json.dumps({"verified_checkpoints": len(models), "new_samples": 0}))
    elif args.stage == "prepare":
        print(json.dumps({"handoff_sha256": prepare(options)}))
    elif args.stage == "peer":
        print(
            json.dumps(
                {
                    "receipt_sha256": peer(
                        options, args.handoff_sha256, args.receipt_directory
                    )
                }
            )
        )
    else:
        finish(options, args.handoff_sha256, args.peer_receipt_sha256)


if __name__ == "__main__":
    main()

"""Guarded continued-study primary shards and registered offline summaries.

No CLI or calibration/peer-host execution coordinator is exposed here. Complete
matching-environment preflight remains required before any real collection.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypedDict

import torch

from examples.qec import compare_ising as comparison
from examples.qec import continued_assessment as inputs
from examples.qec import continued_parity as parity
from examples.qec import continued_selection as selection
from examples.qec import diagnose_predecoder as archive
from flagquantum.qec.decoders._ai import exact_statistics
from flagquantum.qec.decoders._ai.assessment import (
    PairedCounts,
    paired_counts,
    summarize,
)
from flagquantum.qec.decoders._ai.comparison import Correction, GlobalDecode, decode

SEED = 16000011
SHOTS = 4194304
CHUNK = 4096
ALPHA = 0.05 / 26
COMPARISONS = tuple(
    (name, ref) for name in parity.PATHS[3:] for ref in ("global", "fast")
) + (("fast", "global"), ("mask-only", "global"))


@dataclass(frozen=True)
class Binding:
    protocol_sha256: str
    circuit_sha256: str
    choices_sha256: str
    probes_sha256: str
    receipt_sha256: dict[str, str]
    source_sha256: dict[str, str]
    hardware: dict[str, object]


@dataclass(frozen=True)
class PredictionShard:
    index: int
    filename: str
    file_sha256: str
    detectors_sha256: str
    truth_sha256: str
    prediction_sha256: dict[str, str]


@dataclass(frozen=True)
class PrimaryManifest:
    purpose: str
    seed: int
    shots: int
    chunk_size: int
    detectors: int
    observables: int
    binding: Binding
    shards: list[PredictionShard]


def _sources() -> dict[str, str]:
    paths = [
        Path(__file__),
        Path(parity.__file__),
        Path(selection.__file__),
        Path(inputs.__file__),
        Path(exact_statistics.__file__),
        Path(comparison.__file__),
        Path(archive.__file__),
        *sorted(Path(exact_statistics.__file__).parent.glob("*.py")),
    ]
    return {str(p.relative_to(archive.ROOT)): archive.digest(p) for p in paths}


@torch.inference_mode()
def _capture(
    circuit: Any,
    corrections: dict[str, Correction],
    global_decode: GlobalDecode,
    output: Path,
    binding: Binding,
) -> PrimaryManifest:
    """Internal capture after all guards; exact software fixtures replace budgets."""
    if tuple(corrections) != parity.PATHS:
        raise ValueError("requires all fifteen ordered prediction paths")
    widths = circuit.num_detectors, circuit.num_observables
    if any(type(n) is not int or n < 1 for n in widths):
        raise ValueError("requires positive primary dimensions")
    if hashlib.sha256(str(circuit).encode()).hexdigest() != binding.circuit_sha256:
        raise ValueError("primary circuit hash mismatch")
    output.mkdir(parents=True, exist_ok=False)
    plan = PrimaryManifest(
        "continued_primary_prediction_capture", SEED, SHOTS, CHUNK, *widths, binding, []
    )
    with (output / "plan.json").open("x") as stream:
        json.dump(asdict(plan), stream, indent=2, allow_nan=False)
    sampler = circuit.compile_detector_sampler(seed=SEED)
    records = []
    for index in range(SHOTS // CHUNK):
        bits, truth = sampler.sample(CHUNK, separate_observables=True)
        bits = inputs._bits(torch.as_tensor(bits), shots=CHUNK, width=widths[0])
        truth = inputs._bits(torch.as_tensor(truth), shots=CHUNK, width=widths[1])
        predictions = {
            name: torch.cat(
                [
                    decode(bits[start : start + 32], correct, global_decode)
                    for start in range(0, CHUNK, 32)
                ]
            )
            for name, correct in corrections.items()
        }
        for value in predictions.values():
            inputs._bits(value, shots=CHUNK, width=widths[1])
        path = output / f"chunk-{index:04d}.pt"
        with path.open("xb") as stream:
            torch.save(
                {"detectors": bits, "truth": truth, "predictions": predictions}, stream
            )
        records.append(
            PredictionShard(
                index,
                path.name,
                archive.digest(path),
                inputs._bit_hash(bits),
                inputs._bit_hash(truth),
                {name: inputs._bit_hash(value) for name, value in predictions.items()},
            )
        )
        print(
            json.dumps({"phase": "primary", "completed_shots": (index + 1) * CHUNK}),
            flush=True,
        )
    if binding.source_sha256 != _sources() or binding.protocol_sha256 != archive.digest(
        inputs.PROTOCOL
    ):
        raise ValueError("primary source or protocol changed during capture")
    result = PrimaryManifest(
        plan.purpose, SEED, SHOTS, CHUNK, *widths, binding, records
    )
    with (output / "manifest.json").open("x") as stream:
        stream.write(json.dumps(asdict(result), indent=2, allow_nan=False) + "\n")
    return result


def run_primary(
    models: list[inputs.VerifiedCheckpoint],
    choices: Path,
    probes: Path,
    receipts: dict[str, tuple[Path, str]],
    output: Path,
    *,
    choices_sha256: str,
    probes_sha256: str,
    ising_source: Path,
    ising_weights: Path,
) -> PrimaryManifest:
    """A800-172-only primary stage; every guard precedes sampler construction.

    A trusted launcher must bind the SSH host, formal lock and producing image,
    dependencies, CPU/thread/memory limits, source mounts and total time budget.
    This stage never substitutes fixture receipts or falls back to CPU inference.
    """
    anchor = json.loads((archive.ARCHIVE / "results.json").read_text())
    circuit_sha = anchor["circuit_sha256"]
    selection.verify_choices(
        choices,
        expected_sha256=choices_sha256,
        models=models,
        circuit_sha256=circuit_sha,
    )
    parity.verify_two_hosts(
        receipts,
        choices_sha256=choices_sha256,
        probes_sha256=probes_sha256,
        circuit_sha256=circuit_sha,
        probe_directory=probes,
    )
    if output.exists():
        raise FileExistsError(output)
    if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
        raise ValueError("primary requires autocast disabled")
    receipt = json.loads(receipts["a800-172"][0].read_text())
    if str(torch.__version__) != receipt["torch_version"]:
        raise ValueError("primary Torch version differs from frozen device evidence")
    hardware = comparison.cuda_metadata(parity.IMAGE)
    if (
        hardware["nvidia_smi_uuid_name_driver"].split(",")[0]
        != receipt["hardware"]["nvidia_smi_uuid_name_driver"].split(",")[0]
    ):
        raise ValueError("primary must use the verified A800-172 device")
    reference = comparison.reference(
        ising_source,
        ising_weights,
        comparison.Config(False, distance=9, rounds=9, a800=True),
    )
    if (
        hashlib.sha256(str(reference.circuit).encode()).hexdigest() != circuit_sha
        or hashlib.sha256(
            comparison.training_dem(reference.circuit).to_stim_text().encode()
        ).hexdigest()
        != anchor["training_dem_sha256"]
        or archive.state_digest(reference.pipeline.state_dict())
        != receipt["pipeline_state_sha256"]
    ):
        raise ValueError("primary reference differs from frozen parity inputs")
    pairs = parity.path_pairs(
        models, choices, reference.pipeline, device=torch.device("cuda:0")
    )
    binding = Binding(
        archive.digest(inputs.PROTOCOL),
        circuit_sha,
        choices_sha256,
        probes_sha256,
        {host: sha for host, (_, sha) in receipts.items()},
        _sources(),
        hardware,
    )
    # Recheck immediately before the only primary sampler can be constructed.
    selection.verify_choices(
        choices,
        expected_sha256=choices_sha256,
        models=models,
        circuit_sha256=circuit_sha,
    )
    parity.verify_two_hosts(
        receipts,
        choices_sha256=choices_sha256,
        probes_sha256=probes_sha256,
        circuit_sha256=circuit_sha,
        probe_directory=probes,
    )
    return _capture(
        reference.circuit,
        {name: pair[1] for name, pair in pairs.items()},
        comparison.global_decoder(reference.circuit),
        output,
        binding,
    )


def replay_primary(
    directory: Path,
    *,
    expected_manifest_sha256: str,
) -> Iterator[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
    """Validate the complete inventory before exposing any prediction shard."""
    manifest = directory / "manifest.json"
    if archive.digest(manifest) != expected_manifest_sha256:
        raise ValueError("primary manifest hash mismatch")
    raw = json.loads(manifest.read_text())
    if (
        raw["purpose"] != "continued_primary_prediction_capture"
        or raw["seed"] != SEED
        or raw["shots"] != SHOTS
        or raw["chunk_size"] != CHUNK
        or raw["binding"]["protocol_sha256"] != archive.digest(inputs.PROTOCOL)
        or raw["binding"]["source_sha256"] != _sources()
    ):
        raise ValueError("primary manifest differs from the frozen stage")
    widths = raw["detectors"], raw["observables"]
    if any(type(n) is not int or n < 1 for n in widths):
        raise ValueError("invalid primary dimensions")
    names = [f"chunk-{i:04d}.pt" for i in range(SHOTS // CHUNK)]
    rows = raw["shards"]
    if (
        len(rows) != len(names)
        or sorted(p.name for p in directory.glob("chunk-*.pt")) != names
    ):
        raise ValueError("primary inventory mismatch")
    for index, (name, row) in enumerate(zip(names, rows, strict=True)):
        if (
            row["index"] != index
            or row["filename"] != name
            or archive.digest(directory / name) != row["file_sha256"]
        ):
            raise ValueError("primary shard order or hash mismatch")
    for name, row in zip(names, rows, strict=True):
        path = directory / name
        if archive.digest(path) != row["file_sha256"]:
            raise ValueError("primary shard changed during replay")
        payload = torch.load(path, weights_only=True, map_location="cpu")
        if (
            not isinstance(payload, dict)
            or set(payload) != {"detectors", "truth", "predictions"}
            or not isinstance(payload["detectors"], torch.Tensor)
            or not isinstance(payload["truth"], torch.Tensor)
            or not isinstance(payload["predictions"], dict)
            or tuple(payload["predictions"]) != parity.PATHS
        ):
            raise ValueError("primary shard payload mismatch")
        bits = inputs._bits(payload["detectors"], shots=CHUNK, width=widths[0])
        truth = inputs._bits(payload["truth"], shots=CHUNK, width=widths[1])
        if (
            inputs._bit_hash(bits) != row["detectors_sha256"]
            or inputs._bit_hash(truth) != row["truth_sha256"]
        ):
            raise ValueError("primary input bit hash mismatch")
        predictions = {}
        if set(row["prediction_sha256"]) != set(parity.PATHS):
            raise ValueError("primary prediction fingerprint inventory mismatch")
        for key, value in payload["predictions"].items():
            if not isinstance(value, torch.Tensor):
                raise ValueError("prediction requires tensor bits")
            value = inputs._bits(value, shots=CHUNK, width=widths[1])
            if inputs._bit_hash(value) != row["prediction_sha256"][key]:
                raise ValueError("primary prediction bit hash mismatch")
            predictions[key] = value
        yield truth, predictions


class ComparisonReport(TypedDict):
    candidate: str
    reference: str
    counts: dict[str, int]
    exact: dict[str, object]
    legacy_hoeffding: dict[str, object]


class RegisteredSummary(TypedDict):
    purpose: str
    manifest_sha256: str
    shots: int
    comparison_alpha: float
    comparisons: list[ComparisonReport]
    native_quality_gate: str
    competitive_with_fast: bool
    horizon_counts_descriptive_only: list[dict[str, object]]
    independent_physical_shots: int
    speed_claim_allowed: bool


def summarize_primary(directory: Path, output: Path, *, manifest_sha256: str) -> None:
    """Only complete frozen shards supply the registered 26 comparisons."""
    totals = {pair: [0] * 5 for pair in COMPARISONS}
    horizons = {
        (f"native-{s}-step-2000-{m}", f"native-{s}-step-200-{m}"): [0] * 5
        for s in archive.SEEDS
        for m in ("raw", "calibrated")
    }
    for truth, predictions in replay_primary(
        directory, expected_manifest_sha256=manifest_sha256
    ):
        for (candidate, reference), values in (totals | horizons).items():
            count = paired_counts(truth, predictions[reference], predictions[candidate])
            for index, value in enumerate(asdict(count).values()):
                values[index] += value
    if any(v[0] != SHOTS for v in totals.values()):
        raise ValueError("incomplete primary comparison counts")
    records: list[ComparisonReport] = []
    bounds = {}
    for (candidate, reference), values in totals.items():
        count = PairedCounts(*values)
        exact = exact_statistics.summarize_exact(count, alpha=ALPHA)
        records.append(
            {
                "candidate": candidate,
                "reference": reference,
                "counts": asdict(count),
                "exact": asdict(exact),
                "legacy_hoeffding": summarize(count, alpha=ALPHA),
            }
        )
        bounds[(candidate, reference)] = exact.ratio_upper
    primary = [f"native-{s}-step-2000-calibrated" for s in archive.SEEDS]

    def passes(reference: str, limit: float) -> bool:
        values = [bounds[(name, reference)] for name in primary]
        return all(v is not None and v <= limit for v in values)

    result: RegisteredSummary = {
        "purpose": "continued_primary_registered_summary",
        "manifest_sha256": manifest_sha256,
        "shots": SHOTS,
        "comparison_alpha": ALPHA,
        "comparisons": records,
        "native_quality_gate": (
            "supported_10_percent_reduction"
            if passes("global", 0.9)
            else "inconclusive"
        ),
        "competitive_with_fast": passes("fast", 1.05),
        "horizon_counts_descriptive_only": [
            {"candidate": c, "reference": r, "counts": asdict(PairedCounts(*v))}
            for (c, r), v in horizons.items()
        ],
        "independent_physical_shots": SHOTS,
        "speed_claim_allowed": False,
    }
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")

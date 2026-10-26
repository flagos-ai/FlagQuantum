"""Guarded matched primary capture, replay and registered count summaries.

No CLI or trusted execution controller is supplied. Final-source CI and complete
matching preflight remain prerequisites to any formal sampling. Host labels and
parsed receipts alone do not authenticate the SSH host or container.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch

from examples.qec import compare_ising as comparison
from examples.qec import continued_parity as device
from examples.qec import continued_primary as primary
from examples.qec import diagnose_predecoder as archive
from examples.qec import matched_checkpoints as checkpoints
from examples.qec import matched_parity as parity
from examples.qec import matched_training as training
from examples.qec import matched_validation as validation

SEED = 22000011
SHOTS = 4194304
CHUNK = 4096
ALPHA = 0.05 / 32
COMPARISONS = (
    tuple((name, ref) for name in parity.PATHS[3:] for ref in ("global", "fast"))
    + tuple(
        (f"cnn-{seed}-step-2000-{mode}", f"mlp-{seed}-step-2000-{mode}")
        for seed in (11, 22, 33)
        for mode in ("raw", "calibrated")
    )
    + (("fast", "global"), ("mask-only", "global"))
)


class CountReduction(TypedDict):
    shots: int
    comparison_alpha: float
    comparisons: list[primary.ComparisonReport]
    native_quality_gate: str
    matched_recipe_incremental_benefit: str
    competitive_with_fast: bool


def _reduce_predictions(
    chunks: Iterable[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> CountReduction:
    """Reduce all 32 fixed comparisons; propagate any exact interval failure."""
    totals = {pair: [0] * 5 for pair in COMPARISONS}
    primary._accumulate_counts(chunks, totals, paths=parity.PATHS, shots=SHOTS)
    records, bounds = primary._comparison_reports(totals, alpha=ALPHA)

    def passes(reference: str, limit: float) -> bool:
        for seed in (11, 22, 33):
            candidate = f"cnn-{seed}-step-2000-calibrated"
            ref = (
                f"mlp-{seed}-step-2000-calibrated" if reference == "mlp" else reference
            )
            upper = bounds[candidate, ref]
            if upper is None or upper > limit:
                return False
        return True

    quality = passes("global", 0.9)
    return {
        "shots": SHOTS,
        "comparison_alpha": ALPHA,
        "comparisons": records,
        "native_quality_gate": (
            "supported_10_percent_reduction" if quality else "inconclusive"
        ),
        "matched_recipe_incremental_benefit": (
            "supported_10_percent_reduction"
            if quality and passes("mlp", 0.9)
            else "inconclusive"
        ),
        "competitive_with_fast": passes("fast", 1.05),
    }


@dataclass(frozen=True)
class Binding(primary.Binding):
    protocols: dict[str, str]
    input_files: dict[str, dict[str, str]]
    pipeline_state_sha256: str
    fast_weights_sha256: str
    torch_version: str


def _sources() -> dict[str, str]:
    result = parity._sources()
    result[str(Path(__file__).relative_to(archive.ROOT))] = archive.digest(
        Path(__file__)
    )
    return result


def run_primary(
    source: Path,
    initial: Path,
    trained: Path,
    calibration: Path,
    choices: Path,
    probes: Path,
    receipts: dict[str, tuple[Path, str]],
    output: Path,
    *,
    host: str,
    choices_sha256: str,
    probes_sha256: str,
    ising_weights: Path,
) -> primary.PrimaryManifest:
    """One guarded A800-172 stage; a trusted controller must authenticate execution."""
    if host != "a800-172":
        raise ValueError("primary collection requires the authenticated A800-172 host")
    if output.exists():
        raise FileExistsError(output)
    models, frozen = validation.restore_choices(
        source, initial, trained, calibration, choices, expected_sha256=choices_sha256
    )
    parity.verify_two_hosts(
        source,
        initial,
        trained,
        calibration,
        choices,
        probes,
        receipts,
        choices_sha256=choices_sha256,
        probes_sha256=probes_sha256,
    )
    receipt = json.loads(receipts[host][0].read_text())
    if (
        str(torch.__version__) != receipt["torch_version"]
        or torch.get_num_threads() != 1
    ):
        raise ValueError("primary environment differs from frozen receipts")
    if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
        raise ValueError("primary requires autocast disabled")
    weight_hash = archive.digest(ising_weights)
    if weight_hash != comparison.FAST_SHA256:
        raise ValueError("primary requires pinned official Fast weights")
    if comparison.source_digest(source) != comparison.ISING_SOURCE_SHA256:
        raise ValueError("primary requires pinned vendor source")
    directories = dict(
        initial=initial,
        trained=trained,
        calibration=calibration,
        choices=choices,
        probes=probes,
    )
    files = {name: validation._files(p) for name, p in directories.items()}
    states = [training._state(t) for _, _, t in models]
    protocols = {p.name: archive.digest(p) for p in checkpoints.PROTOCOLS}
    sources = _sources()
    circuit_sha = frozen["inputs"]["circuit_sha256"]
    dem_sha = frozen["inputs"]["dem_sha256"]
    if any(
        hashlib.sha256(t.model.dem_text.encode()).hexdigest() != dem_sha
        for _, _, t in models
    ):
        raise ValueError("primary models differ from canonical DEM")
    hardware, reference = device._device_reference(
        source, ising_weights, circuit_sha256=circuit_sha, dem_sha256=dem_sha
    )
    if device._hardware_uuid(hardware) != device._hardware_uuid(receipt["hardware"]):
        raise ValueError("primary device differs from verified A800-172 UUID")
    pipeline_hash = archive.state_digest(reference.pipeline.state_dict())
    if pipeline_hash != receipt["pipeline_state_sha256"]:
        raise ValueError("primary pipeline differs from verified receipt")
    if any(p.dtype != torch.float32 for p in reference.pipeline.parameters()) or any(
        not bool(torch.isfinite(v).all())
        for v in reference.pipeline.state_dict().values()
    ):
        raise ValueError("primary requires finite FP32 pipeline parameters")
    parity._precision()
    pairs = parity.path_pairs(
        models, frozen, reference.pipeline, device=torch.device("cuda:0")
    )
    global_decode = comparison.global_decoder(reference.circuit)
    binding = Binding(
        protocols[checkpoints.PROTOCOLS[0].name],
        circuit_sha,
        choices_sha256,
        probes_sha256,
        {host: sha for host, (_, sha) in receipts.items()},
        sources,
        hardware,
        protocols,
        files,
        pipeline_hash,
        weight_hash,
        str(torch.__version__),
    )

    def verify() -> None:
        parity._precision()
        if (
            torch.get_num_threads() != 1
            or str(torch.__version__) != receipt["torch_version"]
            or comparison.source_digest(source) != comparison.ISING_SOURCE_SHA256
            or _sources() != sources
            or {p.name: archive.digest(p) for p in checkpoints.PROTOCOLS} != protocols
            or any(
                validation._files(p) != files[name] for name, p in directories.items()
            )
            or any(archive.digest(p) != sha for p, sha in receipts.values())
            or archive.digest(ising_weights) != weight_hash
            or archive.state_digest(reference.pipeline.state_dict()) != pipeline_hash
            or any(
                training._state(t) != state or t.model.training
                for (_, _, t), state in zip(models, states, strict=True)
            )
        ):
            raise ValueError("primary source, inputs, receipts or model state changed")

    return primary._capture_predictions(
        reference.circuit,
        {name: pair[1] for name, pair in pairs.items()},
        global_decode,
        output,
        binding,
        seed=SEED,
        shots=SHOTS,
        chunk=CHUNK,
        paths=parity.PATHS,
        purpose="matched_primary_prediction_capture",
        verify=verify,
    )


def replay_primary(
    directory: Path,
    *,
    expected_manifest_sha256: str,
) -> Iterator[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
    """Replay externally frozen complete capture files without hardware or sampling."""
    checkpoints._sha256(expected_manifest_sha256)
    manifest = directory / "manifest.json"
    if archive.digest(manifest) != expected_manifest_sha256:
        raise ValueError("primary manifest hash mismatch")
    raw = json.loads(manifest.read_text())
    protocols = {p.name: archive.digest(p) for p in checkpoints.PROTOCOLS}
    binding = raw["binding"]
    if (
        set(raw) != set(primary.PrimaryManifest.__dataclass_fields__)
        or set(binding) != set(Binding.__dataclass_fields__)
        or raw["purpose"] != "matched_primary_prediction_capture"
        or any(type(raw[k]) is not int for k in ("seed", "shots", "chunk_size"))
        or (raw["seed"], raw["shots"], raw["chunk_size"]) != (SEED, SHOTS, CHUNK)
        or binding["protocols"] != protocols
        or binding["protocol_sha256"] != protocols[checkpoints.PROTOCOLS[0].name]
        or binding["source_sha256"] != _sources()
        or binding["fast_weights_sha256"] != comparison.FAST_SHA256
        or set(binding["receipt_sha256"]) != {"a800-172", "a800-171"}
        or set(binding["input_files"])
        != {"initial", "trained", "calibration", "choices", "probes"}
        or binding["torch_version"]
        != json.loads((archive.ARCHIVE / "manifest.json").read_text())["torch"]
        or binding["circuit_sha256"]
        != json.loads((archive.ARCHIVE / "results.json").read_text())["circuit_sha256"]
    ):
        raise ValueError("primary capture differs from frozen matched stage")
    device._hardware_uuid(binding["hardware"])
    for value in [
        binding["pipeline_state_sha256"],
        binding["choices_sha256"],
        binding["probes_sha256"],
        *binding["receipt_sha256"].values(),
    ]:
        checkpoints._sha256(value)
    for name, files in binding["input_files"].items():
        if not files or any(not isinstance(k, str) for k in files):
            raise ValueError("primary frozen input inventory is empty or invalid")
        for value in files.values():
            checkpoints._sha256(value)
    if (
        binding["input_files"]["choices"].get("choices.json")
        != binding["choices_sha256"]
        or binding["input_files"]["probes"].get("manifest.json")
        != binding["probes_sha256"]
    ):
        raise ValueError("primary choices/probe inventories differ from binding")
    plan = json.loads((directory / "plan.json").read_text())
    if plan != {**raw, "shards": []}:
        raise ValueError("primary plan differs from final capture")
    names = {
        "plan.json",
        "manifest.json",
        *(f"chunk-{i:04d}.pt" for i in range(SHOTS // CHUNK)),
    }
    if set(p.name for p in directory.iterdir()) != names:
        raise ValueError("primary output inventory mismatch")
    files = validation._files(directory)
    yield from primary._replay_predictions(
        directory, raw, paths=parity.PATHS, shots=SHOTS, chunk=CHUNK
    )
    if (
        validation._files(directory) != files
        or archive.digest(manifest) != expected_manifest_sha256
    ):
        raise ValueError("primary capture changed during replay")


def summarize_primary(directory: Path, output: Path, *, manifest_sha256: str) -> None:
    """Write only after complete frozen replay and all registered exact intervals."""
    if output.exists():
        raise FileExistsError(output)
    files = validation._files(directory)
    sources = _sources()
    protocols = {p.name: archive.digest(p) for p in checkpoints.PROTOCOLS}
    result = _reduce_predictions(
        replay_primary(directory, expected_manifest_sha256=manifest_sha256)
    )
    if (
        archive.digest(directory / "manifest.json") != manifest_sha256
        or validation._files(directory) != files
        or _sources() != sources
        or {p.name: archive.digest(p) for p in checkpoints.PROTOCOLS} != protocols
    ):
        raise ValueError("primary manifest changed during reduction")
    report = {
        "purpose": "matched_primary_registered_summary",
        "manifest_sha256": manifest_sha256,
        "independent_physical_shots": SHOTS,
        "speed_claim_allowed": False,
        **result,
    }
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")

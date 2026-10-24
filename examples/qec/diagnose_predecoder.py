"""Run the fixed archived-checkpoint CPU label diagnostic, without retraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TypedDict, cast

import torch

from flagquantum.qec.decoders._ai.diagnostics import (
    EquivalenceDiagnostics,
    FeatureAmbiguity,
    FeatureTable,
    LossDiagnostics,
    diagnose,
    diagnose_equivalence,
)
from flagquantum.qec.decoders._ai.predecoder import Predecoder
from flagquantum.qec.decoders._ai.training import Trainer
from flagquantum.qec.dem.model import DetectorErrorModel

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "docs/development/evidence/qec_ai_ising_a800_20261025/a800_172"
PROTOCOL = ROOT / "docs/development/QEC_AI_LABEL_IDENTIFIABILITY.md"
SEEDS = (11, 22, 33)


class PriorResults(TypedDict):
    manifest_sha256: str
    choices_sha256: str
    training_dem_sha256: str
    checkpoint_sha256: dict[str, str]


class PriorAudit(TypedDict):
    native_model_state_sha256: dict[str, str]


class PriorManifest(TypedDict):
    torch: str
    python: str
    dependencies: dict[str, str]


class Choice(TypedDict):
    temperature: float
    threshold: float


class RunManifest(TypedDict):
    purpose: str
    protocol_sha256: str
    source_sha256: dict[str, str]
    input_sha256: dict[str, str]
    diagnostic_shots_per_seed: int
    seeds: list[int]
    torch: str
    python: str
    platform: str
    threads: int
    new_training_steps: int
    new_primary_test_shots: int
    scientific_benefit_claim_allowed: bool


class GroupReport(TypedDict):
    features: tuple[float, float, float]
    observations: int
    positive_labels: int
    negative_labels: int
    positive_fraction: float
    mean_logit: float
    mean_probability: float
    minimum_logit: float
    maximum_logit: float
    empirical_entropy_nats: float


@dataclass(frozen=True)
class ArchivedModel:
    seed: int
    model: Predecoder
    checkpoint_sha256: str
    model_state_sha256: str
    choice: Choice


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_bytes(value: torch.Tensor) -> bytes:
    """Return one bounded CPU tensor block's native bytes without NumPy."""
    if value.device.type != "cpu":
        raise ValueError("archived tensor hashing requires CPU tensors")
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8)
    return bytes(cast(list[int], raw.tolist()))


def state_digest(state: dict[str, torch.Tensor]) -> str:
    """Match the producing audit's name/dtype/shape/native-byte fingerprint."""
    hashed = hashlib.sha256()
    for name, value in sorted(state.items()):
        hashed.update(name.encode() + b"\0")
        hashed.update(str(value.dtype).encode() + b"\0")
        hashed.update(str(tuple(value.shape)).encode() + b"\0")
        flat = value.detach().contiguous().reshape(-1)
        for start in range(0, flat.numel(), 65536):
            hashed.update(tensor_bytes(flat[start : start + 65536]))
    return hashed.hexdigest()


def load_inputs(directory: Path) -> tuple[list[ArchivedModel], dict[str, str]]:
    """Verify every archived input and all seeds before any new draw is possible."""
    hashes: dict[str, str] = {}
    for name in ("manifest.json", "results.json", "choices.json", "audit.json"):
        actual = digest(directory / name)
        if actual != digest(ARCHIVE / name):
            raise ValueError(f"archived input hash mismatch: {name}")
        hashes[name] = actual
    results = cast(PriorResults, json.loads((directory / "results.json").read_text()))
    audit = cast(PriorAudit, json.loads((directory / "audit.json").read_text()))
    manifest = cast(
        PriorManifest, json.loads((directory / "manifest.json").read_text())
    )
    if (
        hashes["manifest.json"] != results["manifest_sha256"]
        or hashes["choices.json"] != results["choices_sha256"]
    ):
        raise ValueError("archived input cross-reference mismatch")
    if (
        torch.__version__ != manifest["torch"]
        or platform.python_version() != manifest["python"]
    ):
        raise ValueError("diagnostics require the producing Python/PyTorch environment")
    for name, expected in manifest["dependencies"].items():
        if version(name) != expected:
            raise ValueError(f"producing dependency version mismatch: {name}")
    hashes["training.dem"] = digest(directory / "training.dem")
    if hashes["training.dem"] != results["training_dem_sha256"]:
        raise ValueError("archived training DEM hash mismatch")
    dem = DetectorErrorModel.from_stim_text((directory / "training.dem").read_text())
    choices = cast(
        dict[str, dict[str, Choice]],
        json.loads((directory / "choices.json").read_text()),
    )
    loaded: list[ArchivedModel] = []
    for seed in SEEDS:
        path = directory / f"native-{seed}.pt"
        hashes[path.name] = digest(path)
        if hashes[path.name] != results["checkpoint_sha256"][str(seed)]:
            raise ValueError(f"archived checkpoint hash mismatch: seed {seed}")
        raw: object = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(raw, dict)
            or raw.get("version") != 1
            or type(raw.get("step")) is not int
            or raw["step"] != 200
        ):
            raise ValueError(
                f"requires the archived version-1 step-200 checkpoint: seed {seed}"
            )
        if raw.get("dem") != dem.to_stim_text():
            raise ValueError(f"checkpoint DEM mismatch: seed {seed}")
        state = raw.get("model")
        if not isinstance(state, dict) or not all(
            isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
        ):
            raise ValueError(f"checkpoint model state is malformed: seed {seed}")
        state = cast(dict[str, torch.Tensor], state)
        fingerprint = state_digest(state)
        if fingerprint != audit["native_model_state_sha256"][str(seed)]:
            raise ValueError(f"archived tensor-state hash mismatch: seed {seed}")
        model = Predecoder(dem, allow_hyperedges=True)
        Trainer(model).restore(path)
        model.eval()
        choice = choices[str(seed)]["flagquantum_calibrated"]
        if not (
            0 < choice["threshold"] <= 1
            and math.isfinite(choice["temperature"])
            and choice["temperature"] > 0
        ):
            raise ValueError(f"invalid frozen inference choice: seed {seed}")
        loaded.append(
            ArchivedModel(seed, model, hashes[path.name], fingerprint, choice)
        )
    return loaded, hashes


def add_equivalence(
    a: EquivalenceDiagnostics, b: EquivalenceDiagnostics
) -> EquivalenceDiagnostics:
    return EquivalenceDiagnostics(
        a.shots + b.shots,
        a.latent_equal + b.latent_equal,
        a.detector_equal + b.detector_equal,
        a.detector_and_logical_equal + b.detector_and_logical_equal,
        a.detector_equal_logical_different + b.detector_equal_logical_different,
        a.latent_mismatch_bits + b.latent_mismatch_bits,
    )


def add_loss(a: LossDiagnostics | None, b: LossDiagnostics) -> LossDiagnostics:
    if a is None:
        return b
    n = a.total_labels + b.total_labels
    return LossDiagnostics(
        n,
        a.positive_labels + b.positive_labels,
        a.negative_labels + b.negative_labels,
        a.positive_bce_sum + b.positive_bce_sum,
        a.negative_bce_sum + b.negative_bce_sum,
        a.predicted_positive_labels + b.predicted_positive_labels,
        (a.mean_probability * a.total_labels + b.mean_probability * b.total_labels) / n,
        (a.brier_score * a.total_labels + b.brier_score * b.total_labels) / n,
    )


def group_reports(summary: FeatureAmbiguity) -> list[GroupReport]:
    return [
        GroupReport(
            features=g.features,
            observations=g.observations,
            positive_labels=g.positive_labels,
            negative_labels=g.negative_labels,
            positive_fraction=g.positive_fraction,
            mean_logit=g.mean_logit,
            mean_probability=g.mean_probability,
            minimum_logit=g.minimum_logit,
            maximum_logit=g.maximum_logit,
            empirical_entropy_nats=g.empirical_entropy_nats,
        )
        for g in summary.groups
    ]


@dataclass(frozen=True)
class SeedReport:
    seed: int
    sampler_seed: int
    diagnostic_shots: int
    checkpoint_sha256: str
    model_state_sha256: str
    latent_bytes_sha256: str
    frozen_choice: Choice
    controls: dict[str, EquivalenceDiagnostics]
    raw_loss: LossDiagnostics
    calibrated_loss: LossDiagnostics
    feature_summary: FeatureAmbiguity


@torch.no_grad()
def diagnose_model(archived: ArchivedModel, shots: int) -> SeedReport:
    """One predetermined stream, no optimizer, fitting or logical-test access."""
    if shots not in (32, 4096):
        raise ValueError("diagnostic budget must be smoke 32 or full 4096")
    model = archived.model
    generator = torch.Generator().manual_seed(archived.seed + 9100000)
    table = FeatureTable()
    controls = {
        name: EquivalenceDiagnostics(0, 0, 0, 0, 0, 0)
        for name in ("zero", "raw_0.5", "raw_0.9", "frozen_calibrated")
    }
    raw_loss: LossDiagnostics | None = None
    calibrated_loss: LossDiagnostics | None = None
    latent_hash = hashlib.sha256()
    h, logical = model.h.double(), model.logical.double()
    for _ in range(shots // 32):
        batch = model.sample(32, generator)
        labels, detectors = batch.mechanisms, batch.sample.detectors
        logits = model(detectors)
        scaled = logits / archived.choice["temperature"]
        table.update(model._features(detectors), logits, labels)
        raw_loss = add_loss(raw_loss, diagnose(logits, labels))
        calibrated_loss = add_loss(calibrated_loss, diagnose(scaled, labels))
        latent_hash.update(tensor_bytes(labels.to(torch.int8)))
        candidates = {
            "zero": torch.zeros_like(labels),
            "raw_0.5": logits.sigmoid() >= 0.5,
            "raw_0.9": logits.sigmoid() >= 0.9,
            "frozen_calibrated": scaled.sigmoid() >= archived.choice["threshold"],
        }
        for name, candidate in candidates.items():
            controls[name] = add_equivalence(
                controls[name], diagnose_equivalence(h, logical, labels, candidate)
            )
    assert raw_loss is not None and calibrated_loss is not None
    summary = table.summarize()
    return SeedReport(
        archived.seed,
        archived.seed + 9100000,
        shots,
        archived.checkpoint_sha256,
        archived.model_state_sha256,
        latent_hash.hexdigest(),
        archived.choice,
        controls,
        raw_loss,
        calibrated_loss,
        summary,
    )


def fixture_smoke() -> EquivalenceDiagnostics:
    """Exact two-vector cycle probe; no noise sampling or training."""
    dem = DetectorErrorModel.from_stim_text(
        "error(0.1) D0 D1\nerror(0.1) D1 D2\nerror(0.1) D0 D2\n"
        "detector D0\ndetector D1\ndetector D2\n"
    )
    return diagnose_equivalence(
        dem.detector_error_matrix(),
        dem.observables_flips_matrix(),
        torch.zeros(1, 3),
        torch.ones(1, 3),
    )


def run(directory: Path, output: Path, *, smoke: bool = False) -> None:
    models, inputs = load_inputs(directory)
    output.mkdir(parents=True, exist_ok=False)
    source = [
        Path(__file__),
        ROOT / "flagquantum/qec/decoders/_ai/diagnostics.py",
        ROOT / "flagquantum/qec/decoders/_ai/predecoder.py",
        ROOT / "flagquantum/qec/decoders/_ai/training.py",
    ]
    manifest: RunManifest = {
        "purpose": (
            "smoke" if smoke else "fixed_budget_training_distribution_diagnostics"
        ),
        "protocol_sha256": digest(PROTOCOL),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source},
        "input_sha256": inputs,
        "diagnostic_shots_per_seed": 32 if smoke else 4096,
        "seeds": [11] if smoke else list(SEEDS),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "threads": torch.get_num_threads(),
        "new_training_steps": 0,
        "new_primary_test_shots": 0,
        "scientific_benefit_claim_allowed": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for archived in models[:1] if smoke else models:
        print(json.dumps({"phase": "diagnostic", "seed": archived.seed}), flush=True)
        result = diagnose_model(archived, 32 if smoke else 4096)
        report = asdict(result)
        cast(dict[str, object], report["feature_summary"])["groups"] = group_reports(
            result.feature_summary
        )
        (output / f"seed-{archived.seed}.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    print(json.dumps({"status": "completed", "output_directory": str(output)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fixture-smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.fixture_smoke:
        print(json.dumps({"purpose": "exact_fixture_smoke", **asdict(fixture_smoke())}))
        return
    if args.input_directory is None or args.output_directory is None:
        parser.error(
            "archived-input runs require --input-directory and --output-directory"
        )
    run(args.input_directory, args.output_directory, smoke=args.smoke)


if __name__ == "__main__":
    main()

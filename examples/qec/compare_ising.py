"""Run the pinned Ising CPU pipeline and native controls on shared circuit shots.

The external checkout is an optional research dependency, never a library import.
Smoke uses an explicitly untrained upstream model and cannot establish a ranking.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from flagquantum.qec.decoders._ai.assessment import paired_counts, summarize
from flagquantum.qec.decoders._ai.calibration import fit_temperature
from flagquantum.qec.decoders._ai.comparison import (
    Correction,
    decode,
    measure,
    on_device,
    require_parity,
)
from flagquantum.qec.decoders._ai.predecoder import Predecoder
from flagquantum.qec.decoders._ai.training import Trainer
from flagquantum.qec.dem.model import DemError, DetectorErrorModel

ISING_COMMIT = "ea7c1bb3b823fc6a38472d0d95974b6d01932288"
ISING_SOURCE_SHA256 = "ed9e6c56f263e0cd9fcc04a494e9cf6544c9f78cbb54c788fcce71b2e961b70e"
PATHS = ("pymatching", "flagquantum_raw", "flagquantum_calibrated", "ising")
FAST_SHA256 = "22db7ad25b584d7588493d6ef815381c8d1263cfb29f212391a591b285ad0a05"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def peak_rss_bytes() -> int:
    """Read the Linux address-space peak, excluding inherited pre-exec history."""
    if sys.platform.startswith("linux"):
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if fields and fields[0] == "VmHWM:":
                if len(fields) != 3 or fields[2] != "kB" or int(fields[1]) <= 0:
                    raise ValueError("invalid Linux VmHWM memory counter")
                return int(fields[1]) * 1024
        raise ValueError("Linux process memory measurement requires VmHWM")
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def source_digest(root: Path) -> str:
    hashed = hashlib.sha256()
    for path in sorted([*root.glob("code/**/*.py"), *root.glob("conf/*.yaml")]):
        hashed.update(path.relative_to(root).as_posix().encode() + b"\0")
        hashed.update(path.read_bytes() + b"\0")
    return hashed.hexdigest()


@dataclass(frozen=True)
class Config:
    smoke: bool
    distance: int = 3
    rounds: int = 3
    probability: float = 0.003
    basis: str = "X"
    rotation: str = "XV"
    threads: int = 1
    a800: bool = False

    @property
    def device(self) -> torch.device:
        return torch.device("cuda:0" if self.a800 else "cpu")

    @property
    def stream_offset(self) -> int:
        return 100000 if self.a800 else 0

    @property
    def seeds(self) -> tuple[int, ...]:
        return (11,) if self.smoke else (11, 22, 33)

    @property
    def shots(self) -> int:
        return 32 if self.smoke else 65536


@dataclass(frozen=True)
class Reference:
    # Stim/config/model objects stop at this optional external-code boundary.
    circuit: Any
    pipeline: nn.Module
    noise: dict[str, float]
    weights: dict[str, str]


def check_logits(
    module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
) -> None:
    if not bool(torch.isfinite(output).all()):
        raise ValueError("nonfinite Ising logits before thresholding")


def reference(
    root: Path, checkpoint: Path | None, config: Config, *, load_model: bool = True
) -> Reference:
    if source_digest(root) != ISING_SOURCE_SHA256:
        raise ValueError(f"Ising source does not match pinned commit {ISING_COMMIT}")
    if (not config.smoke or config.a800) and (
        checkpoint is None or not checkpoint.is_file()
    ):
        raise ValueError(
            "non-smoke comparison requires an authorized .safetensors file"
        )
    if config.a800 and (checkpoint is None or digest(checkpoint) != FAST_SHA256):
        raise ValueError("A800 protocol requires the pinned official Fast checkpoint")
    sys.path.insert(0, str(root / "code"))
    memory = importlib.import_module("qec.surface_code.memory_circuit")
    noise_module = importlib.import_module("qec.noise_model")
    noise = noise_module.NoiseModel.from_single_p(config.probability)
    circuit = memory.MemoryCircuit(
        distance=config.distance,
        n_rounds=config.rounds,
        idle_error=config.probability,
        sqgate_error=config.probability,
        tqgate_error=config.probability,
        spam_error=2 * config.probability / 3,
        basis=config.basis,
        code_rotation=config.rotation,
        noise_model=noise,
        add_boundary_detectors=True,
    )
    circuit.set_error_rates()
    if circuit.stim_circuit.num_detectors != config.rounds * (config.distance**2 - 1):
        raise ValueError(
            "upstream detector layout does not match the evaluation volume"
        )
    if not load_model:
        return Reference(
            circuit.stim_circuit, nn.Identity(), noise.canonical_parameters(), {}
        )
    exports = importlib.import_module("export.safetensors_utils")
    evaluation = importlib.import_module("evaluation.logical_error_rate")
    if config.smoke and not config.a800:
        torch.manual_seed(7)
        cfg = exports._build_minimal_cfg(1)
        factory = importlib.import_module("model.factory")
        model = factory.ModelFactory.create_model(cfg)
        weights = {"status": "untrained_smoke_only", "model_id": "1"}
    else:
        if checkpoint is None:
            raise ValueError("requires an authorized checkpoint")
        model, metadata = exports.load_safetensors(str(checkpoint), device="cpu")
        if metadata.get("model_id") not in ("1", "4"):
            raise ValueError("requires the surface Fast (1) or Accurate (4) checkpoint")
        cfg = exports._build_minimal_cfg(metadata["model_id"])
        weights = {
            "status": "external_checkpoint_training_history_unverified",
            "sha256": digest(checkpoint),
            **metadata,
        }
    cfg.distance, cfg.n_rounds = config.distance, config.rounds
    cfg.data.code_rotation = config.rotation
    cfg.test.meas_basis_test, cfg.test.n_rounds = config.basis, config.rounds
    cfg.test.th_data = cfg.test.th_syn = 0.0
    cfg.test.sampling_mode, cfg.enable_fp16 = "threshold", False
    model.register_forward_hook(check_logits)
    pipeline = evaluation.PreDecoderMemoryEvalModule(
        model.float().eval(),
        cfg,
        evaluation._build_stab_maps(config.distance, config.rotation),
        torch.device("cpu"),
    ).eval()
    return Reference(
        circuit.stim_circuit, pipeline, noise.canonical_parameters(), weights
    )


def training_dem(circuit: Any) -> DetectorErrorModel:
    """Keep complete hyperedges; explicitly approximate disjoint Pauli alternatives.

    Only training priors use Stim's independent-DEM approximation. Validation and
    test shots always come from the original categorical-noise circuit.
    """
    dem = circuit.detector_error_model(
        decompose_errors=False, approximate_disjoint_errors=True
    ).flattened()
    errors: list[DemError] = []
    for instruction in dem:
        if instruction.type == "error":
            targets = instruction.targets_copy()
            errors.append(
                DemError(
                    instruction.args_copy()[0],
                    tuple(t.val for t in targets if t.is_relative_detector_id()),
                    tuple(t.val for t in targets if t.is_logical_observable_id()),
                )
            )
    # Flattening may leave repeated complete signatures (observed at d=9).
    # Exact independent-parity merging preserves the declared DEM distribution.
    return DetectorErrorModel(
        dem.num_detectors, dem.num_observables, tuple(errors)
    ).merge_duplicate_mechanisms()


def sample(circuit: Any, shots: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    detectors, observables = circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True
    )
    return torch.from_numpy(detectors).to(torch.int8), torch.from_numpy(observables).to(
        torch.int8
    )


def boundary_only(bits: torch.Tensor, config: Config) -> torch.Tensor:
    """Independent control for the pinned upstream's terminal complement mask."""
    half = (config.distance**2 - 1) // 2
    end = bits.shape[1] - (half if config.basis == "X" else 2 * half)
    masked = bits.clone()
    masked[:, end - half : end] = 0
    return masked


def global_decoder(circuit: Any) -> Callable[[torch.Tensor], torch.Tensor]:
    pymatching = importlib.import_module("pymatching")
    matcher = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(
            decompose_errors=True, approximate_disjoint_errors=True
        )
    )

    def run(bits: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(
            matcher.decode_batch(
                bits.numpy().astype("uint8"), enable_correlations=False
            )
        ).to(torch.int8)

    return run


def correction(
    name: str,
    native: Predecoder | None,
    upstream: nn.Module,
    threshold: float = 1.0,
    temperature: float = 1.0,
) -> Correction:
    def run(bits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if name == "pymatching":
            return bits, torch.zeros((bits.shape[0], 1), dtype=torch.int8)
        if name == "ising":
            output = upstream(bits)
            return output[:, 1:], output[:, :1]
        if native is None:
            raise ValueError("native comparison requires its trained model")
        return native.correct(bits, threshold=threshold, temperature=temperature)

    if name not in PATHS:
        raise ValueError("unknown comparison path")
    return run


@torch.no_grad()
def choices(
    model: Predecoder,
    ref: Reference,
    decode_global: Callable[[torch.Tensor], torch.Tensor],
    seed: int,
    smoke: bool,
    stream_offset: int = 0,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed + 4000000 + stream_offset)
    # Bound activation memory at d=9; sampling order stays explicit and deterministic.
    if smoke or not stream_offset:
        batches = [model.sample(32 if smoke else 4096, generator)]
    else:
        batches = [model.sample(128, generator) for _ in range(32)]
    temperature = fit_temperature(
        torch.cat([model(batch.sample.detectors) for batch in batches]),
        torch.cat([batch.mechanisms for batch in batches]),
    ).temperature
    detectors, truth = sample(
        ref.circuit, 32 if smoke else 4096, seed + 5000000 + stream_offset
    )
    selected: dict[str, Any] = {}
    for name, temp in (
        ("flagquantum_raw", 1.0),
        ("flagquantum_calibrated", temperature),
    ):
        trials = []
        for threshold in (0.5, 0.7, 0.9, 0.99, 1.0):
            correct = correction(name, model, ref.pipeline, threshold, temp)
            if stream_offset:
                predicted = torch.cat(
                    [
                        decode(detectors[start : start + 32], correct, decode_global)
                        for start in range(0, detectors.shape[0], 32)
                    ]
                )
            else:
                predicted = decode(detectors, correct, decode_global)
            trials.append((int((predicted != truth).any(1).sum()), -threshold))
        selected[name] = {
            "threshold": -min(trials)[1],
            "temperature": temp,
            "trials": [
                {"threshold": -threshold, "logical_failures": failures}
                for failures, threshold in trials
            ],
        }
    return selected


def cuda_metadata(image_id: str | None) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise ValueError("A800 comparison requires CUDA; CPU fallback is forbidden")
    if image_id is None or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        raise ValueError("A800 comparison requires the exact --image-id SHA-256")
    properties = torch.cuda.get_device_properties(0)
    if "A800" not in properties.name:
        raise ValueError("A800 protocol requires an A800 device")
    queried = (
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        .stdout.strip()
        .splitlines()
    )
    if len(queried) != 1:
        raise ValueError("expose exactly one physical GPU to the isolated container")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return {
        "device": "cuda:0",
        "name": properties.name,
        "nvidia_smi_uuid_name_driver": queried[0],
        "total_memory_bytes": properties.total_memory,
        "free_memory_bytes_at_preflight": torch.cuda.mem_get_info(0)[0],
        "cuda_runtime": torch.version.cuda,
        "image_id": image_id,
        "hostname": platform.node(),
        "tf32": False,
        "autocast": False,
        "deterministic_algorithms": True,
    }


def device_parity(
    config: Config,
    ref: Reference,
    dem: DetectorErrorModel,
    output: Path,
    selected: dict[str, Any],
) -> None:
    """Validate independent probe inputs before creating a primary test sampler."""
    bits, _ = sample(ref.circuit, 64, 8100011)
    width = bits.shape[1]
    probes = torch.cat(
        (
            bits,
            torch.eye(width, dtype=torch.int8),
            torch.zeros(1, width, dtype=torch.int8),
            torch.ones(1, width, dtype=torch.int8),
        )
    )
    upstream = copy.deepcopy(ref.pipeline).to(config.device)
    decode_global = global_decoder(ref.circuit)
    for seed in config.seeds:
        native = Predecoder(dem, allow_hyperedges=True)
        Trainer(native).restore(output / f"native-{seed}.pt")
        native.eval()
        accelerated = copy.deepcopy(native).to(config.device)
        for name in PATHS[1:]:
            choice = selected[str(seed)].get(name, {})
            threshold, temperature = choice.get("threshold", 1.0), choice.get(
                "temperature", 1.0
            )
            first = correction(name, native, ref.pipeline, threshold, temperature)
            second = on_device(
                correction(name, accelerated, upstream, threshold, temperature),
                config.device,
            )
            for start in range(0, probes.shape[0], 32):
                require_parity(
                    probes[start : start + 32],
                    first,
                    second,
                    decode_global if start < 64 else None,
                )
    (output / "device-parity.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "probe_shots": probes.shape[0],
                "probe_seed": 8100011,
                "seeds": config.seeds,
                "scope": "exact_residual_frame_bits_all_probes_and_final_prediction_bits_64_physical_shots",
            },
            indent=2,
        )
        + "\n"
    )


def timing_worker(args: argparse.Namespace, config: Config, ref: Reference) -> None:
    manifest = json.loads((args.output_directory / "manifest.json").read_text())
    selected = json.loads((args.output_directory / "choices.json").read_text())["11"]
    name = args.timing_worker
    native = None
    if not config.a800 or name.startswith("flagquantum"):
        native = Predecoder(training_dem(ref.circuit), allow_hyperedges=True)
        Trainer(native).restore(args.output_directory / "native-11.pt")
        native.eval()
        if config.a800:
            native.to(config.device)
    if config.a800 and name == "ising":
        ref.pipeline.to(config.device)
    threshold = selected.get(name, {}).get("threshold", 1.0)
    temperature = selected.get(name, {}).get("temperature", 1.0)
    correct = correction(name, native, ref.pipeline, threshold, temperature)
    if config.a800 and name != "pymatching":
        correct = on_device(correct, config.device)
    decode_global = global_decoder(ref.circuit)
    detectors, _ = sample(
        ref.circuit, 128 if config.a800 else 32, 7000011 + config.stream_offset
    )
    if config.a800:
        torch.cuda.reset_peak_memory_stats(config.device)
    timings = [
        asdict(
            measure(
                detectors[:size],
                correct,
                decode_global,
                warmup=3 if config.smoke or not config.a800 else 10,
                repetitions=4 if config.smoke else (200 if config.a800 else 32),
                synchronize=(
                    (lambda: torch.cuda.synchronize(config.device))
                    if config.a800 and name != "pymatching"
                    else None
                ),
            )
        )
        for size in ((1, 32, 128) if config.a800 else (1, 32))
    ]
    result = {
        "path": name,
        "timings": timings,
        "isolated_process_peak_rss_bytes": peak_rss_bytes(),
        "memory_scope": "whole_worker_including_imports_active_model_and_graph_not_training",
        "rss_counter": (
            "proc_self_status_VmHWM"
            if sys.platform.startswith("linux")
            else "ru_maxrss"
        ),
        "manifest_sha256": digest(args.output_directory / "manifest.json"),
        "smoke": manifest["config"]["smoke"],
    }
    if config.a800:
        result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(
            config.device
        )
        result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(
            config.device
        )
        result["cuda_memory_scope"] = (
            "active_model_and_dem_plus_inference_buffers_not_training_or_total_device_usage"
        )
    (args.output_directory / f"timing-{name}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


def run(args: argparse.Namespace, config: Config, ref: Reference) -> None:
    output: Path = args.output_directory
    output.mkdir(parents=True, exist_ok=False)
    protocol = Path(__file__).resolve().parents[2] / (
        "docs/development/QEC_AI_ISING_A800.md"
        if config.a800
        else "docs/development/QEC_AI_ISING_COMPARISON.md"
    )
    sources = [
        Path(__file__),
        *Path(Predecoder.__module__.replace(".", "/")).parent.glob("*.py"),
    ]
    manifest = {
        "config": asdict(config),
        "ising_source": {"repository": "NVIDIA/Ising-Decoding", "commit": ISING_COMMIT},
        "ising_source_sha256": source_digest(args.ising_source),
        "weights": ref.weights,
        "protocol_sha256": digest(protocol),
        "source_sha256": {str(p): digest(p) for p in sources},
        "noise_parameters": ref.noise,
        "training_dem_approximate_disjoint_errors": True,
        "training_dem_duplicate_rule": "independent_parity_complete_detector_logical_signature",
        "test_sampling": "original_stim_circuit_no_dem_sampling",
        "matching": "shared_pymatching_uncorrelated_batch_decode",
        "ising_terminal_complement_input": "upstream_masks_it_including_without_local_correction",
        "dtype": "float32_both_models_external_fp16_weights_promoted_if_applicable",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "stim", "pymatching", "omegaconf", "safetensors")
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "distribution_semantics": "single_device_fast_path",
        "training_budget_comparable": False,
        "scientific_ranking_allowed": False,
        "purpose": (
            "smoke" if config.smoke else "fixed_budget_exploratory_cpu_comparison"
        ),
    }
    if config.a800:
        manifest["purpose"] = (
            "smoke" if config.smoke else "fixed_budget_a800_checkpoint_comparison"
        )
        manifest["hardware"] = args.hardware
        manifest["native_training_device"] = "cpu"
        manifest["primary_test_seed"] = 6001011
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "circuit.stim").write_text(str(ref.circuit))
    dem = training_dem(ref.circuit)
    (output / "training.dem").write_text(dem.to_stim_text())
    decode_global = global_decoder(ref.circuit)
    selected = {}
    training_cost = {}
    for seed in config.seeds:
        print(json.dumps({"phase": "training", "seed": seed}), flush=True)
        torch.manual_seed(seed)
        native = Predecoder(dem, allow_hyperedges=True)
        trainer = Trainer(native, seed=seed + 104729)
        start = time.perf_counter()
        trainer.train(
            2 if config.smoke else 200, batch_size=16 if config.smoke else 128
        )
        trainer.save(output / f"native-{seed}.pt")
        training_cost[str(seed)] = {
            "steps": trainer.step,
            "model_shots": trainer.step * (16 if config.smoke else 128),
            "training_and_save_seconds": time.perf_counter() - start,
            "parameters": sum(p.numel() for p in native.parameters()),
        }
    for seed in config.seeds:
        print(json.dumps({"phase": "validation", "seed": seed}), flush=True)
        native = Predecoder(dem, allow_hyperedges=True)
        Trainer(native).restore(output / f"native-{seed}.pt")
        native.eval()
        selected[str(seed)] = choices(
            native, ref, decode_global, seed, config.smoke, config.stream_offset
        )
    (output / "choices.json").write_text(json.dumps(selected, indent=2) + "\n")
    if config.a800:
        print(json.dumps({"phase": "device_parity"}), flush=True)
        device_parity(config, ref, dem, output, selected)
        ref.pipeline.to(config.device)
    # No test sampler is constructed before every training and validation choice is frozen.
    print(json.dumps({"phase": "testing", "shots": config.shots}), flush=True)
    detectors, truth = sample(
        ref.circuit, config.shots, 6001011 if config.a800 else 6000011
    )
    torch.save({"detectors": detectors, "observables": truth}, output / "test-shots.pt")
    baseline = decode_global(detectors)
    boundary_control = paired_counts(
        truth, baseline, decode_global(boundary_only(detectors, config))
    )
    records = []
    for seed in config.seeds:
        native = Predecoder(dem, allow_hyperedges=True)
        Trainer(native).restore(output / f"native-{seed}.pt")
        native.eval()
        if config.a800:
            native.to(config.device)
        predictions = {}
        for name in PATHS[1:]:
            threshold = selected[str(seed)].get(name, {}).get("threshold", 1.0)
            temperature = selected[str(seed)].get(name, {}).get("temperature", 1.0)
            correct = correction(name, native, ref.pipeline, threshold, temperature)
            if config.a800:
                correct = on_device(correct, config.device)
            predictions[name] = torch.cat(
                [
                    decode(detectors[start : start + 32], correct, decode_global)
                    for start in range(0, detectors.shape[0], 32)
                ]
            )
            residual = torch.cat(
                [
                    correct(detectors[start : start + 32])[0]
                    for start in range(0, detectors.shape[0], 32)
                ]
            )
            counts = paired_counts(truth, baseline, predictions[name])
            records.append(
                {
                    "seed": seed,
                    "path": name,
                    "probability_threshold": None if name == "ising" else threshold,
                    "ising_logit_threshold_data_and_syndrome": (
                        0.0 if name == "ising" else None
                    ),
                    "temperature": temperature,
                    "input_detector_density": float(detectors.float().mean()),
                    "residual_detector_density": float(residual.float().mean()),
                    "changed_detector_bits": int((residual != detectors).sum()),
                    "counts": asdict(counts),
                    "summary": summarize(counts, alpha=0.05 / 12),
                }
            )
        counts = paired_counts(
            truth, predictions["ising"], predictions["flagquantum_calibrated"]
        )
        records.append(
            {
                "seed": seed,
                "path": "flagquantum_calibrated_vs_ising",
                "counts": asdict(counts),
                "summary": summarize(counts, alpha=0.05 / 12),
            }
        )
        torch.save(predictions, output / f"predictions-{seed}.pt")
    report = {
        "manifest_sha256": digest(output / "manifest.json"),
        "circuit_sha256": digest(output / "circuit.stim"),
        "training_dem_sha256": digest(output / "training.dem"),
        "choices_sha256": digest(output / "choices.json"),
        "checkpoint_sha256": {
            str(seed): digest(output / f"native-{seed}.pt") for seed in config.seeds
        },
        "training_cost": training_cost,
        "ising_boundary_only_control": asdict(boundary_control),
        "predictions_sha256": {
            str(seed): digest(output / f"predictions-{seed}.pt")
            for seed in config.seeds
        },
        "test_shots_sha256": digest(output / "test-shots.pt"),
        "distinct_test_shots": config.shots,
        "detectors_sha256": hashlib.sha256(detectors.numpy().tobytes()).hexdigest(),
        "truth_sha256": hashlib.sha256(truth.numpy().tobytes()).hexdigest(),
        "records": records,
    }
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    if config.a800:
        # Timing workers own their active model; the idle parent retains no tensors on CUDA.
        native.cpu()
        ref.pipeline.cpu()
        torch.cuda.empty_cache()
    for name in PATHS:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--timing-worker",
                name,
            ],
            check=True,
            timeout=180,
        )
    print(
        json.dumps(
            {
                "purpose": manifest["purpose"],
                "paths": list(PATHS),
                "distinct_test_shots": config.shots,
                "scientific_ranking_allowed": False,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ising-source", type=Path, required=True)
    parser.add_argument("--ising-checkpoint", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--a800", action="store_true", help="run the frozen d=9 CUDA protocol"
    )
    parser.add_argument(
        "--image-id", help="exact isolated container image SHA-256 for --a800"
    )
    parser.add_argument("--timing-worker", choices=PATHS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = Config(
        args.smoke,
        distance=9 if args.a800 else 3,
        rounds=9 if args.a800 else 3,
        a800=args.a800,
    )
    if config.a800:
        if os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8") != ":4096:8":
            raise ValueError("A800 protocol requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    args.hardware = cuda_metadata(args.image_id) if config.a800 else None
    with torch.inference_mode(False):
        ref = reference(
            args.ising_source.resolve(),
            args.ising_checkpoint,
            config,
            load_model=args.timing_worker in (None, "ising"),
        )
        if args.timing_worker:
            timing_worker(args, config, ref)
        else:
            run(args, config, ref)


if __name__ == "__main__":
    main()

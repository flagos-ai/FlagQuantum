"""Freeze all six initial native models without authorizing formal training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from examples.qec import diagnose_predecoder as archive
from examples.qec import ising_layout
from examples.qec import matched_checkpoints as checkpoints
from flagquantum.qec.decoders._ai.predecoder import Predecoder
from flagquantum.qec.decoders._ai.spacetime import SpacetimePredecoder
from flagquantum.qec.decoders._ai.training import Trainer
from flagquantum.qec.dem.model import DetectorErrorModel


def initialize(source: Path, output: Path) -> dict[str, Any]:
    """Verify the pinned cell and freeze every seed/architecture before updates.

    A failed write leaves partial output that cannot be retried. This preflight
    is not a study launch, a device receipt or a checkpoint-selection mechanism.
    """
    if output.exists():
        raise FileExistsError(output)
    if torch.get_num_threads() != 1:
        raise ValueError("initialization requires one Torch thread")
    protocols = {p.name: archive.digest(p) for p in checkpoints.PROTOCOLS}
    with torch.random.fork_rng(devices=[]):
        template, inputs = ising_layout.build_model(source)
    layout_sha = inputs["layout_sha256"]
    if not isinstance(layout_sha, str):
        raise ValueError("verified layout digest is missing")
    dem = DetectorErrorModel.from_stim_text(template.dem_text)
    sites = tuple(int(site) for site in template.sites.tolist())
    kinds = tuple(1 if bool(template.masks[1].flatten()[site]) else 2 for site in sites)
    expected = archive.state_digest(
        {key: getattr(template, key) for key in ("h", "logical", "prior")}
    )
    initial: list[tuple[int, str, Trainer]] = []
    rng: dict[str, torch.Tensor] = {}
    # Do not let initialization consume the caller's stream, and never let one
    # architecture's parameter count advance another architecture's seed.
    with torch.random.fork_rng(devices=[]):
        for seed in archive.SEEDS:
            for architecture in ("cnn", "mlp"):
                name = f"{architecture}-{seed}"
                torch.manual_seed(seed)
                rng[name + "-before"] = torch.get_rng_state().clone()
                model: Predecoder
                if architecture == "cnn":
                    model = SpacetimePredecoder.from_dem(
                        dem, grid_shape=template.grid_shape, sites=sites, kinds=kinds
                    )
                else:
                    model = Predecoder(dem, allow_hyperedges=True)
                rng[name + "-after"] = torch.get_rng_state().clone()
                common = archive.state_digest(
                    {key: getattr(model, key) for key in ("h", "logical", "prior")}
                )
                if model.dem_text != template.dem_text or common != expected:
                    raise ValueError(
                        "initial models do not share the exact canonical DEM buffers"
                    )
                initial.append(
                    (seed, architecture, Trainer(model, seed=seed + 18100000))
                )
    if [(seed, arch) for seed, arch, _ in initial] != [
        (seed, arch) for seed in (11, 22, 33) for arch in ("cnn", "mlp")
    ]:
        raise ValueError("requires all six registered initial models")
    # All six exist before the first checkpoint is written; no update operation
    # belongs in this phase. The future runner must verify the external inventory.
    output.mkdir(parents=True, exist_ok=False)
    rng_path = output / "initialization-rng.pt"
    torch.save(rng, rng_path)
    rows: list[dict[str, object]] = []
    for seed, architecture, trainer in initial:
        name = f"{architecture}-{seed}"
        directory = output / name
        digest = checkpoints.save_bound(
            trainer, directory, seed=seed, layout_sha256=layout_sha
        )
        restored = checkpoints.restore_bound(
            trainer.model,
            directory,
            seed=seed,
            layout_sha256=layout_sha,
            binding_sha256=digest,
        )
        state = archive.state_digest(trainer.model.state_dict())
        if (
            restored.step != 0
            or archive.state_digest(restored.model.state_dict()) != state
        ):
            raise ValueError("initial checkpoint failed exact restoration")
        rows.append(
            {
                "seed": seed,
                "architecture": architecture,
                "step": 0,
                "binding_sha256": digest,
                "model_state_sha256": state,
                "initialization_rng_before_sha256": archive.state_digest(
                    {"rng": rng[name + "-before"]}
                ),
                "initialization_rng_after_sha256": archive.state_digest(
                    {"rng": rng[name + "-after"]}
                ),
                "training_generator_sha256": archive.state_digest(
                    {"rng": trainer.generator.get_state()}
                ),
            }
        )
    if {p.name: archive.digest(p) for p in checkpoints.PROTOCOLS} != protocols:
        raise ValueError("prospective documents changed during initialization")
    manifest = {
        "purpose": "matched_initialization_preflight_only",
        "inputs": inputs,
        "protocol_sha256": protocols,
        "initialization_rng_file_sha256": archive.digest(rng_path),
        "initial_checkpoints": rows,
        "threads": torch.get_num_threads(),
        "torch": str(torch.__version__),
        "new_samples": 0,
        "new_optimizer_steps": 0,
        "formal_launch_authorized": False,
    }
    with (output / "initialization.json").open("x") as stream:
        stream.write(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ising-source", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report = initialize(args.ising_source, args.output_directory)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "purpose",
                    "new_samples",
                    "new_optimizer_steps",
                    "formal_launch_authorized",
                )
            }
            | {"verified_initial_checkpoints": len(report["initial_checkpoints"])},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

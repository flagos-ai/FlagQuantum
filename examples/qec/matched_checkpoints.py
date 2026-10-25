"""Bind unchanged Trainer checkpoints to the matched study's external identity.

These helpers do not authorize a study launch or authenticate physical devices.
The caller must supply the separately verified layout and frozen artifact hash.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import torch

from examples.qec import diagnose_predecoder as archive
from flagquantum.qec.decoders._ai.predecoder import Predecoder
from flagquantum.qec.decoders._ai.spacetime import SpacetimePredecoder
from flagquantum.qec.decoders._ai.training import Trainer

PROTOCOLS = (
    archive.ROOT / "docs/development/QEC_AI_ISING_CNN_COMPARISON.md",
    archive.ROOT / "docs/development/QEC_AI_ISING_CNN_LAYOUT_VERIFICATION.md",
)
HORIZONS = (0, 200, 2000)


def _sha256(value: str) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("requires an external lowercase SHA-256 digest")


def _identity(model: Predecoder, seed: int, layout_sha256: str) -> dict[str, object]:
    _sha256(layout_sha256)
    if type(seed) is not int or seed not in (11, 22, 33):
        raise ValueError("requires a registered model seed")
    if type(model) not in (Predecoder, SpacetimePredecoder):
        raise ValueError("requires the fixed native MLP or CNN class")
    cnn = isinstance(model, SpacetimePredecoder)
    expected = 2817 if cnn else 81
    if sum(p.numel() for p in model.parameters()) != expected:
        raise ValueError("model differs from the fixed parameter budget")
    if any(
        p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()
    ):
        raise ValueError("checkpoint binding requires CPU FP32 model parameters")
    if any(v.device.type != "cpu" for v in model.state_dict().values()):
        raise ValueError("checkpoint buffers must remain on CPU")
    if not all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()):
        raise ValueError("model contains nonfinite values")
    return {
        "architecture": "cnn" if cnn else "mlp",
        "seed": seed,
        "layout_sha256": layout_sha256,
        "grid_shape": (
            list(model.grid_shape) if isinstance(model, SpacetimePredecoder) else None
        ),
        "dem_sha256": hashlib.sha256(model.dem_text.encode()).hexdigest(),
        "buffer_sha256": archive.state_digest(dict(model.named_buffers())),
        "protocol_sha256": {p.name: archive.digest(p) for p in PROTOCOLS},
    }


def _validate(trainer: Trainer, seed: int) -> None:
    if type(trainer.step) is not int or trainer.step not in HORIZONS:
        raise ValueError("checkpoint is not at a registered horizon")
    groups = trainer.optimizer.param_groups
    expected = {
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0,
        "amsgrad": False,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
        "foreach": None,
        "fused": None,
    }
    if (
        type(trainer.optimizer) is not torch.optim.Adam
        or len(groups) != 1
        or any(groups[0].get(k) != v for k, v in expected.items())
    ):
        raise ValueError("requires the fixed Adam recipe")
    parameters = list(trainer.model.parameters())
    if trainer.step == 0:
        if not torch.equal(
            trainer.generator.get_state(),
            torch.Generator().manual_seed(seed + 18100000).get_state(),
        ):
            raise ValueError("initial training stream differs from registered seed")
        if trainer.optimizer.state:
            raise ValueError("initial checkpoint must have no optimizer moments")
    else:
        if len(trainer.optimizer.state) != len(parameters):
            raise ValueError("incomplete optimizer state")
        for parameter in parameters:
            state = trainer.optimizer.state[parameter]
            if (
                set(state) != {"step", "exp_avg", "exp_avg_sq"}
                or float(state["step"]) != trainer.step
            ):
                raise ValueError("optimizer state counter differs from horizon")
            for key in ("exp_avg", "exp_avg_sq"):
                value = state[key]
                if value.shape != parameter.shape or not bool(
                    torch.isfinite(value).all()
                ):
                    raise ValueError("malformed optimizer moments")
            if bool((state["exp_avg_sq"] < 0).any()):
                raise ValueError("negative optimizer second moment")


def save_bound(
    trainer: Trainer, directory: Path, *, seed: int, layout_sha256: str
) -> str:
    """Write a fresh pair; preserve partial output on failure and forbid retries.

    Return the binding-file digest for the caller's external frozen inventory.
    The binary checkpoint retains exactly the existing version-1 payload.
    """
    identity = _identity(trainer.model, seed, layout_sha256)
    _validate(trainer, seed)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "checkpoint.pt"
    trainer.save(path)
    record = {
        "identity": identity,
        "step": trainer.step,
        "checkpoint_sha256": archive.digest(path),
        "model_state_sha256": archive.state_digest(trainer.model.state_dict()),
        "generator_sha256": archive.state_digest(
            {"generator": trainer.generator.get_state()}
        ),
    }
    binding = directory / "binding.json"
    with binding.open("x") as stream:
        stream.write(
            json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
    return archive.digest(binding)


def restore_bound(
    model: Predecoder,
    directory: Path,
    *,
    seed: int,
    layout_sha256: str,
    binding_sha256: str,
) -> Trainer:
    """Verify external identity before restore and buffers/state afterward.

    Return a restored copy. A refusal never mutates the caller's model and never
    samples or updates an optimizer. No self-reported hash is a trusted receipt.
    """
    _sha256(binding_sha256)
    identity = _identity(model, seed, layout_sha256)
    binding, path = directory / "binding.json", directory / "checkpoint.pt"
    if archive.digest(binding) != binding_sha256:
        raise ValueError("external binding hash mismatch")
    record = json.loads(binding.read_text())
    if (
        set(record)
        != {
            "identity",
            "step",
            "checkpoint_sha256",
            "model_state_sha256",
            "generator_sha256",
        }
        or record["identity"] != identity
    ):
        raise ValueError("checkpoint identity differs from current verified inputs")
    if type(record["step"]) is not int or record["step"] not in HORIZONS:
        raise ValueError("binding is not at a registered horizon")
    if archive.digest(path) != record["checkpoint_sha256"]:
        raise ValueError("checkpoint file hash mismatch")
    candidate = Trainer(copy.deepcopy(model), seed=seed + 18100000)
    candidate.restore(path)
    _validate(candidate, seed)
    if _identity(candidate.model, seed, layout_sha256) != identity:
        raise ValueError("restored checkpoint changed bound buffers or geometry")
    if (
        candidate.step != record["step"]
        or archive.state_digest(candidate.model.state_dict())
        != record["model_state_sha256"]
        or archive.state_digest({"generator": candidate.generator.get_state()})
        != record["generator_sha256"]
        or archive.digest(path) != record["checkpoint_sha256"]
        or archive.digest(binding) != binding_sha256
    ):
        raise ValueError("restored checkpoint state or frozen files changed")
    return candidate

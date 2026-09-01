#!/usr/bin/env python3
"""Validate P5 Double-Single SGD on native CUDA or Torch-FL flagos."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _git_output(*args: str) -> str:
    try:
        return subprocess.run(
            ("git", *args),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unavailable"


def _source_provenance() -> dict[str, Any]:
    revision = _git_output("rev-parse", "HEAD")
    if revision == "unavailable":
        revision = os.environ.get("FLAGQUANTUM_SOURCE_REVISION", "unavailable")
    status = _git_output("status", "--porcelain")
    if status == "unavailable":
        dirty_value = os.environ.get("FLAGQUANTUM_SOURCE_TREE_DIRTY")
        tree_dirty = None if dirty_value is None else dirty_value == "1"
    else:
        tree_dirty = bool(status)
    return {"revision": revision, "tree_dirty": tree_dirty}


def validate(*, device_name: str, checkpoints: tuple[int, ...]) -> dict[str, Any]:
    if device_name.startswith("flagos"):
        try:
            importlib.import_module("torch_fl")
        except ImportError as exc:
            raise RuntimeError("Torch-FL is required for flagos validation") from exc
    torch = importlib.import_module("torch")
    platforms = importlib.import_module("flagquantum.runtime.platforms")
    optimizer = importlib.import_module(
        "flagquantum.runtime.backends.statevector." "split_real_imag_autograd_optimizer"
    )
    conformance = importlib.import_module(
        "flagquantum.runtime.backends.statevector."
        "split_real_imag_optimizer_conformance"
    )
    if device_name.startswith("flagos") and (
        not hasattr(torch, "flagos") or not torch.flagos.is_available()
    ):
        raise RuntimeError("Torch-FL did not expose an available flagos device")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("native CUDA is unavailable")
    device = platforms.resolve_platform_device(device_name)
    report = conformance.run_split_real_imag_optimizer_conformance(
        device, checkpoints=checkpoints
    )
    report.require_accepted()
    probe_state = optimizer.initialize_split_real_imag_double_single_sgd(
        {"theta": torch.tensor(0.23, dtype=torch.float32, device=device)}
    )
    if probe_state.parameters.high.device != device:
        raise AssertionError("P5 high word escaped the requested logical device")
    if probe_state.parameters.low.device != device:
        raise AssertionError("P5 low word escaped the requested logical device")
    payload = {
        "schema": "flagquantum_split_real_imag_p5_optimizer_accelerator_route_v1",
        "status": "passed",
        "validation_scope": "single_device_accelerator_portability_reference",
        "device": str(device),
        "provider": report.provider,
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "torch_fl_package": (
            importlib.metadata.version("torch-fl")
            if device_name.startswith("flagos")
            else None
        ),
        "validated_steps": list(checkpoints),
        "conformance": report.to_dict(),
        "logical_device_residency": True,
        "parameter_representation": "double_single_high_low",
        "parameter_word_dtype": "float32",
        "parameter_word_count": 2,
        "gradient_representation": "double_single_high_low",
        "tensor_grad_used": False,
        "accelerator_float64_tensor_materialized": False,
        "provider_internal_route_audited": False,
        "flagcx_collectives_validated": False,
        "convergence_certification": False,
        "hardware_certification": False,
        "domestic_accelerator_certification": False,
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "performance_claim_allowed": False,
        "production_claim_allowed": False,
        "source": _source_provenance(),
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--checkpoints", default="1,16,64")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checkpoints = tuple(
        int(item) for item in args.checkpoints.split(",") if item.strip()
    )
    payload = validate(device_name=args.device, checkpoints=checkpoints)
    rendered = json.dumps(payload, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

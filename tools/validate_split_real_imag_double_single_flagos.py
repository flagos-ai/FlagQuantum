#!/usr/bin/env python3
"""Validate full Double-Single P3 through Torch-FL's logical flagos device."""

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


def validate(
    *, device_name: str, depths: tuple[int, ...], seeds: tuple[int, ...]
) -> dict[str, Any]:
    try:
        importlib.import_module("torch_fl")
    except ImportError as exc:
        raise RuntimeError("Torch-FL is required for flagos validation") from exc
    torch = importlib.import_module("torch")
    platforms = importlib.import_module("flagquantum.providers.platform")
    conformance = importlib.import_module(
        "flagquantum.runtime.backends.statevector."
        "split_real_imag_double_single_conformance"
    )
    if not hasattr(torch, "flagos") or not torch.flagos.is_available():
        raise RuntimeError("Torch-FL did not expose an available flagos device")
    device = platforms.resolve_platform_device(device_name)
    report = conformance.run_split_real_imag_double_single_conformance(
        device, depths=depths, seeds=seeds
    )
    report.require_accepted()
    if report.device != str(device):
        raise AssertionError("P3 conformance escaped the logical flagos device")
    return {
        "schema": "flagquantum_split_real_imag_p3_flagos_reference_v1",
        "status": "passed",
        "validation_scope": "flagos_cuda_reference",
        "device": str(device),
        "provider": report.provider,
        "torch_version": str(torch.__version__),
        "torch_fl_package": importlib.metadata.version("torch-fl"),
        "representation": "double_single_fp32_complex",
        "state_word_dtype": "float32",
        "state_word_count_per_amplitude": 4,
        "gradient_method": "parameter_shift",
        "validated_depths": list(depths),
        "validated_seeds": list(seeds),
        "conformance": report.to_dict(),
        "operator_profile": report.operator_profile,
        "logical_device_residency": True,
        "host_gate_encoding": True,
        "state_host_fallback": False,
        "provider_internal_route_audited": False,
        "device_only_double_single_trigonometry": False,
        "convergence_certification": False,
        "hardware_certification": False,
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "source": _source_provenance(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="flagos:0")
    parser.add_argument("--depths", default="8,32,128")
    parser.add_argument("--seeds", default="0,7")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    depths = tuple(int(item) for item in args.depths.split(",") if item.strip())
    seeds = tuple(int(item) for item in args.seeds.split(",") if item.strip())
    payload = validate(device_name=args.device, depths=depths, seeds=seeds)
    rendered = json.dumps(payload, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

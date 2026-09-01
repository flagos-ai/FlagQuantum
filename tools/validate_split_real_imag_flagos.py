#!/usr/bin/env python3
"""Validate split real/imag P0 through Torch-FL's CUDA-backed flagos device."""

from __future__ import annotations

import argparse
import importlib
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


def validate(*, device_name: str, depths: tuple[int, ...]) -> dict[str, Any]:
    try:
        torch_fl = importlib.import_module("torch_fl")
    except ImportError as exc:
        raise RuntimeError("Torch-FL is required for flagos validation") from exc

    # Torch-FL's CUDA distribution requires this import order.
    torch = importlib.import_module("torch")
    fq = importlib.import_module("flagquantum")
    platforms = importlib.import_module("flagquantum.runtime.platforms")
    split = importlib.import_module(
        "flagquantum.runtime.backends.statevector.split_real_imag"
    )

    if not hasattr(torch, "flagos") or not torch.flagos.is_available():
        raise RuntimeError("Torch-FL did not expose an available flagos device")
    device = platforms.resolve_platform_device(device_name)
    platform = platforms.get_platform_runtime(device.type)
    report = split.run_split_real_imag_conformance(device, depths=depths)
    report.require_accepted()

    circuit = fq.Circuit(3).h(0).ry(1, theta=0.27).cx(0, 2).rzz(1, 2, theta=-0.19)
    result = split.execute_split_real_imag_statevector(circuit.to_ir(), device=device)
    if result.real.device.type != "flagos" or result.imag.device.type != "flagos":
        raise AssertionError("split state escaped the logical flagos device")
    summary = result.summary()
    identity = platform.identity()
    platform.synchronize(device)
    return {
        "schema": "flagquantum_split_real_imag_flagos_reference_v1",
        "status": "passed",
        "validation_scope": "flagos_cuda_reference",
        "device": str(device),
        "provider": identity.provider,
        "torch_version": str(torch.__version__),
        "torch_fl_version": str(getattr(torch_fl, "__version__", "unknown")),
        "representation": "split_real_imag",
        "storage_dtype": "float32",
        "validated_depths": list(depths),
        "conformance": report.to_dict(),
        "operator_profile": result.operator_profile,
        "operator_evidence_count": len(result.operator_evidence_ids),
        "logical_device_residency": True,
        "flagquantum_host_fallback": summary["flagquantum_host_fallback"],
        "provider_internal_route_audited": False,
        "hardware_certification": False,
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "source": _source_provenance(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="flagos:0")
    parser.add_argument("--depths", default="8,32,128")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    depths = tuple(int(item) for item in args.depths.split(",") if item.strip())
    payload = validate(device_name=args.device, depths=depths)
    rendered = json.dumps(payload, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

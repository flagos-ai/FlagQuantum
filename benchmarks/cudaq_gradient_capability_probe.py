#!/usr/bin/env python3
"""Version-bound CUDA-Q gradient capability probe for the SC27 baseline table."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.sc27_metadata import (  # noqa: E402
    driver_version,
    gpu_identity,
    source_identity,
    topology_snapshot,
)


def gradient_capabilities(cudaq: Any) -> dict[str, Any]:
    gradients = getattr(cudaq, "gradients", None)
    exported = [] if gradients is None else sorted(
        name for name in dir(gradients) if not name.startswith("_")
    )
    lower = {name.lower() for name in exported}
    matched_names = sorted(
        name
        for name in exported
        if any(token in name.lower() for token in ("adjoint", "reverse", "backprop"))
    )
    return {
        "exported_gradient_symbols": exported,
        "parameter_shift_available": any("parametershift" in name for name in lower),
        "central_difference_available": any("centraldifference" in name for name in lower),
        "forward_difference_available": any("forwarddifference" in name for name in lower),
        "matched_reverse_mode_symbols": matched_names,
        "matched_value_and_gradient_available": bool(matched_names),
        "inspection_method": "runtime_public_python_symbol_inspection",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--independent-run-index", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--container-digest", required=True)
    parser.add_argument("--raw-log-sha256")
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()
    import cudaq

    workload = {
        "name": "full_width_linear_hea",
        "n_wires": 31,
        "layers": 8,
        "parameter_count": 248,
        "observable": "Z(15)",
        "dtype": "complex64",
        "requested_operation": "value_and_full_reverse_gradient",
    }
    payload = {
        "schema": "flagquantum.external.cudaq_gradient_capability.v1",
        "benchmark": "cudaq_gradient_capability_probe",
        "comparison_class": "explicitly_unsupported_or_requires_matched_run",
        "world_size": 1,
        "rank_placement": [
            {
                "rank": 0, "local_rank": 0, "hostname": platform.node(),
                **gpu_identity(0), "topology": topology_snapshot(),
            }
        ],
        "source_identity": source_identity(
            repo_root=REPO_ROOT,
            workload=workload,
            container_digest=args.container_digest,
            raw_log_sha256=args.raw_log_sha256,
        ),
        "environment": {
            "python": platform.python_version(),
            "cudaq": importlib.metadata.version("cudaq"),
            "driver_version": driver_version(),
        },
        "workload": workload,
        "protocol": {"independent_run_index": args.independent_run_index},
        "capability": gradient_capabilities(cudaq),
        "official_documentation": {
            "url": (
                "https://nvidia.github.io/cuda-quantum/latest/examples/python/"
                "optimizers_gradients.html"
            ),
            "documented_gradient_strategies": [
                "CentralDifference",
                "ForwardDifference",
                "ParameterShift",
            ],
            "accessed": "2026-08-11",
        },
        "fallback_events": [],
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

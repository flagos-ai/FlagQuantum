#!/usr/bin/env python3
"""Emit non-release capability evidence for MPS solver workspace control."""

from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flagquantum.runtime.backends.mps.solver_workspace import (  # noqa: E402
    probe_solver_workspace_capabilities,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capabilities = probe_solver_workspace_capabilities()
    payload = {
        "schema": "flagquantum.mps_solver_workspace_probe.v1",
        "artifact_class": "derived_development_report",
        "benchmark_evidence_class": "non_release_smoke",
        "status": "passed",
        "environment": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
        },
        "capabilities": capabilities.as_dict(),
        "decision": {
            "direct_python_ctypes_execution_allowed": False,
            "private_aten_abi_allowed": False,
            "production_integration": capabilities.recommended_path,
            "reason": (
                "torch.linalg has no public caller-owned workspace contract; "
                "native cuSOLVER integration requires a compiled, stream-aware "
                "extension with explicit lifetime and numerical audits"
            ),
        },
        "blockers": [
            "native_extension_not_implemented",
            "stream_and_allocator_contract_not_audited",
            "xgesvd_numerical_parity_not_audited",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

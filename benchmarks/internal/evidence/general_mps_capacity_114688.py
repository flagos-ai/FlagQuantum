"""Independently attributable 114,688-site χ768 capacity workload."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.internal.evidence import general_mps_capacity_16 as base  # noqa: E402

N_SITES = 114_688


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    base.N_SITES = N_SITES
    base.main()
    if int(os.environ.get("RANK", "0")) != 0:
        return
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    payload = json.loads(output.read_text())
    if "source_artifacts" not in payload:
        return
    launcher = Path(__file__).resolve()
    base_source = Path(base.__file__).resolve()
    sources = payload["source_artifacts"]
    workload = next(item for item in sources if item["kind"] == "workload")
    workload.update(
        path=str(launcher.relative_to(REPO_ROOT)), sha256=_sha256(launcher)
    )
    sources.append(
        {
            "kind": "base_workload",
            "path": str(base_source.relative_to(REPO_ROOT)),
            "sha256": _sha256(base_source),
        }
    )
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

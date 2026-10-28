"""Hold the program fixed and move it across ranks: does the answer move?

`test_representation_agreement_sweep.py` compares numerical paths on one device.
This compares *placements* of the same path, which is the question #899 adds: a
user changing how many ranks own the MPS needs to know that the objective and the
derivative still describe the same computation, separately from what the
truncation policy already does to them.

Both arms run the two-site split at a cap that discards nothing and at a cap that
does, so a disagreement can be attributed to placement or to truncation instead
of to their sum.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.distributed_cpu

#: Placement must not add anything measurable. The unsharded reference and the
#: sharded run evaluate the same program, so only accumulation order separates
#: them; complex128 over a few dozen amplitudes leaves that at machine epsilon.
PLACEMENT_RELATIVE_BOUND = 1e-12


def _run(world_size: int, output: Path) -> dict:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={world_size}",
            str(ROOT / "benchmarks/mps_placement_conformance.py"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        timeout=300,
    )
    return json.loads(output.with_suffix(".rank0.json").read_text())


@pytest.mark.parametrize("world_size", (2, 3))
def test_moving_the_same_mps_across_ranks_preserves_value_and_gradient(
    world_size: int, tmp_path: Path
) -> None:
    payload = _run(world_size, tmp_path / "placement.json")
    arms = {arm["max_bond"]: arm for arm in payload["arms"]}

    assert payload["world_size"] == world_size

    # The control: a cap that discards nothing isolates placement.
    exact = arms[64]
    assert exact["truncated"] is False
    assert exact["discarded_weight"] == 0.0
    assert exact["distribution_semantics"] == "sharded_across_ranks"
    assert exact["value_relative_error"] is not None
    assert exact["value_relative_error"] < PLACEMENT_RELATIVE_BOUND
    assert exact["gradient"]["relative_l2_error"] < PLACEMENT_RELATIVE_BOUND
    assert exact["gradient"]["cosine_similarity"] > 1 - PLACEMENT_RELATIVE_BOUND

    # The truncating arm separates the two contributions rather than blending
    # them: placement still adds nothing, and the discarded weight is reported
    # beside it so a reader can see what the approximation is.
    truncated = arms[2]
    assert truncated["truncated"] is True
    assert truncated["discarded_weight"] > 0
    assert truncated["gradient_policy"] == "approximate"
    assert truncated["value_relative_error"] < PLACEMENT_RELATIVE_BOUND
    assert truncated["gradient"]["relative_l2_error"] < PLACEMENT_RELATIVE_BOUND


def test_a_sharded_result_is_not_reported_as_scalability_evidence(
    tmp_path: Path,
) -> None:
    """A CPU semantic check must not be quotable as capacity or speed evidence."""

    payload = _run(2, tmp_path / "placement.json")

    assert "not capacity" in payload["claim_boundary"]
    for arm in payload["arms"]:
        assert arm["distribution_semantics"] == "sharded_across_ranks"
        assert arm["blockers"] is not None

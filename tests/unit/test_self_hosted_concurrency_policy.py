"""Self-hosted capacity is bounded by available runners, not workflow budgets."""

from itertools import product
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
WORKFLOWS = Path(__file__).resolve().parents[2] / ".github/workflows"


def test_self_hosted_jobs_have_no_matrix_parallelism_limit() -> None:
    found = []
    for path in sorted(WORKFLOWS.glob("*y*ml")):
        for name, job in yaml.safe_load(path.read_text())["jobs"].items():
            route = str(job.get("runs-on", ""))
            if "self-hosted" not in route and not any(
                label in route
                for label in ("flagquantum-a100-cpu", "flagquantum-cpu-general")
            ):
                continue
            found.append((path.name, name))
            assert "max-parallel" not in job.get("strategy", {}), (path.name, name)
    assert ("ci.yml", "dependency-bounds") in found
    assert ("ci.yml", "coverage-shard") in found
    assert ("scheduled-hardware.yml", "local-scale-scheduled") in found


def test_self_hosted_cpu_capacity_groups_do_not_serialize_ready_work() -> None:
    groups: set[str] = set()
    found = []
    for workflow in ("ci.yml", "pre-commit.yaml"):
        for name, job in yaml.safe_load((WORKFLOWS / workflow).read_text())[
            "jobs"
        ].items():
            route = str(job.get("runs-on", ""))
            if "fromJSON(" in route:
                guard = (
                    "github.event_name == 'push' || (github.event_name == 'pull_request' "
                    "&& github.event.pull_request.head.repo.full_name == github.repository)"
                )
                prefix = "self-hosted"
            elif any(
                label in route
                for label in ("flagquantum-a100-cpu", "flagquantum-cpu-general")
            ):
                guard = (
                    "github.event_name != 'pull_request' || "
                    "github.event.pull_request.head.repo.full_name == github.repository"
                )
                prefix = "a100-cpu"
            else:
                if (
                    "self-hosted" in route
                    and "gpu" not in route
                    and "flagquantum-local" not in route
                ):
                    assert "concurrency" not in job, (workflow, name)
                continue
            found.append(name)
            selector = (
                "${{ ("
                + guard
                + ") && format('"
                + prefix
                + "-{0}-"
                + name
                + "-{1}', github.run_id, strategy.job-index || 0) || 'gh' }}"
            )
            template = job["concurrency"]["group"]
            assert selector in template, (workflow, name)
            matrix = job.get("strategy", {}).get("matrix", {})
            rows = matrix.get("include") or [
                dict(zip(matrix, values, strict=True))
                for values in product(*matrix.values())
            ]
            for run in (1, 2):
                for index, _row in enumerate(rows):
                    group = template.replace(selector, f"{prefix}-{run}-{name}-{index}")
                    assert (
                        group not in groups
                    ), f"Ready jobs share a capacity lock: {group}"
                    groups.add(group)
    assert {"dependency-bounds", "coverage-shard", "pre-commit"} <= set(found)

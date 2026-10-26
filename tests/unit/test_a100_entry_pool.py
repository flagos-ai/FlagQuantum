"""Reserved entry slots cannot accept routine builds or change host resources."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from tools import configure_a100_entry_pool as deployment
from tools.configure_a100_entry_pool import ENTRY, GENERAL, ORDINARY, Runner, partition

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def inventory() -> list[Runner]:
    return [
        {
            "id": index,
            "name": f"fq-a100-cpu{index:02d}",
            "status": "online",
            "busy": False,
            "labels": [
                {
                    "name": label,
                    "type": (
                        "read-only"
                        if label in {"self-hosted", "Linux", "X64"}
                        else "custom"
                    ),
                }
                for label in [
                    "self-hosted",
                    "Linux",
                    "X64",
                    "a100-cpu",
                    f"cpu{index:02d}",
                    ORDINARY,
                    GENERAL,
                ]
            ],
        }
        for index in range(1, 17)
    ]


def test_partition_excludes_long_jobs_and_preserves_identity_labels() -> None:
    runners = inventory()
    original = deepcopy(runners)
    changes = partition(runners)
    assert runners == original
    assert [change["id"] for change in changes] == [15, 16]
    for change in changes:
        assert set(change["labels"]) == {"a100-cpu", f"cpu{change['id']:02d}", ENTRY}
        runner = runners[change["id"] - 1]
        runner["labels"] = [
            label for label in runner["labels"] if label["type"] == "read-only"
        ] + [{"name": name, "type": "custom"} for name in change["labels"]]
    assert partition(runners) == changes
    for label, count in [(ENTRY, 2), (ORDINARY, 14), (GENERAL, 14), ("a100-cpu", 16)]:
        assert (
            sum(
                any(item["name"] == label for item in runner["labels"])
                for runner in runners
            )
            == count
        )


@pytest.mark.parametrize(
    "invalid",
    ["busy", "offline", "missing", "duplicate", "identity", "ordinary", "extra-entry"],
)
def test_invalid_partition_is_refused_before_any_mutation(invalid: str) -> None:
    runners = inventory()
    if invalid == "busy":
        runners[14]["busy"] = True
    elif invalid == "offline":
        runners[15]["status"] = "offline"
    elif invalid == "missing":
        runners.pop()
    elif invalid == "duplicate":
        runners.append(deepcopy(runners[-1]))
    elif invalid == "identity":
        runners[15]["labels"] = [
            item for item in runners[15]["labels"] if item["name"] != "cpu16"
        ]
    elif invalid == "ordinary":
        runners[0]["labels"] = [
            item for item in runners[0]["labels"] if item["name"] != ORDINARY
        ]
    else:
        runners[0]["labels"].append({"name": ENTRY, "type": "custom"})
    before = deepcopy(runners)
    with pytest.raises(ValueError):
        partition(runners)
    assert runners == before


def test_only_source_producers_and_hygiene_can_target_entry_slots() -> None:
    expected = {
        ("ci.yml", "prepare-source"),
        ("ci.yml", "pre-commit"),
        ("local-gpu.yml", "prepare-source"),
        ("pre-commit.yaml", "pre-commit"),
    }
    found = set()
    for path in (ROOT / ".github/workflows").glob("*y*ml"):
        for name, job in yaml.safe_load(path.read_text())["jobs"].items():
            if ENTRY not in str(job.get("runs-on", "")):
                continue
            found.add((path.name, name))
            assert (
                "github.event.pull_request.head.repo.full_name == github.repository"
                in job["runs-on"]
            )
            assert "'ubuntu-latest'" in job["runs-on"]
            assert job.get("timeout-minutes", 15) <= 15
            assert not job.get("continue-on-error", False)
            commands = "\n".join(step.get("run", "") for step in job["steps"])
            assert "pytest" not in commands and "pip install -e" not in commands
    assert found == expected


def test_manual_pool_validation_keeps_all_sixteen_slots_and_regressions() -> None:
    job = yaml.safe_load((ROOT / ".github/workflows/local-gpu.yml").read_text())[
        "jobs"
    ]["validate-cpu-pool"]
    assert job["if"] == "${{ inputs.validate_pool }}"
    assert job["runs-on"] == [
        "self-hosted",
        "linux",
        "x64",
        "a100-cpu",
        "${{ matrix.slot }}",
    ]
    assert job["strategy"]["matrix"]["slot"] == [
        f"cpu{index:02d}" for index in range(1, 17)
    ]
    commands = "\n".join(step.get("run", "") for step in job["steps"])
    assert "tests/test_native_circuit.py tests/test_backends.py" in commands
    assert "len(os.sched_getaffinity(0)) == 8" in commands
    assert "torch.cuda.device_count() == 0" in commands


@pytest.mark.parametrize("became_busy", [False, True])
def test_apply_preserves_other_registrations_and_stops_on_a_new_active_job(
    monkeypatch: pytest.MonkeyPatch, became_busy: bool
) -> None:
    runners = inventory()
    runners[14]["labels"].append({"name": "existing-private-label", "type": "custom"})
    untouched = deepcopy(runners[:14])
    mutations: list[int] = []

    def fake_api(path: str, labels: list[str] | None = None) -> object:
        if labels is None:
            return {"total_count": 16, "runners": deepcopy(runners)}
        identifier = int(path.split("/")[2])
        runner = next(item for item in runners if item["id"] == identifier)
        runner["labels"] = [
            label for label in runner["labels"] if label["type"] == "read-only"
        ] + [{"name": name, "type": "custom"} for name in labels]
        mutations.append(identifier)
        if became_busy:
            runners[15]["busy"] = True
        return {"labels": deepcopy(runner["labels"])}

    monkeypatch.setattr(deployment, "api", fake_api)
    monkeypatch.setattr(sys, "argv", ["configure_a100_entry_pool.py", "--apply"])
    if became_busy:
        with pytest.raises(ValueError, match="online and idle"):
            deployment.main()
        assert mutations == [15]
    else:
        deployment.main()
        assert mutations == [15, 16]
    assert runners[:14] == untouched
    assert any(
        label["name"] == "existing-private-label" for label in runners[14]["labels"]
    )

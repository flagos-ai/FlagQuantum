"""Exercise planned Runtime collection, original outcomes and CPU admission."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from tools.ci import ci_tier

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
COMMAND = (sys.executable, "-m", "pytest", "-m", "integration and not slow", "-q")


def environment(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (root / "tests").mkdir()
    (root / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    integration\n    slow\n    unit\n"
    )
    report = root / "report"
    report.mkdir()
    (report / "resources.json").write_text(
        json.dumps(
            {
                "effective_cpus": 4,
                "shared_registrations_budgeted": 1,
                "workers": 4,
                "threads_per_worker": 1,
            }
        )
    )
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    monkeypatch.setenv(
        "PYTEST_ADDOPTS",
        f"-n 2 --dist=worksteal --fq-shard-count 7 --fq-shard-index 6 --junitxml={report / 'pytest.xml'}",
    )
    monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", raising=False)
    monkeypatch.setenv("GITHUB_SHA", "synthetic-selection-fixture-not-a-CI-commit")
    monkeypatch.setenv("FQ_CI_REPORT_DIR", str(report))
    monkeypatch.setenv("FQ_PYTEST_WORKERS", "4")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    return report


def test_original_selection_executes_once_without_recollecting_unselected_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = environment(tmp_path, monkeypatch)
    executed = tmp_path / "executed.txt"
    collected = tmp_path / "unselected-collection.txt"
    (tmp_path / "tests/test_selected.py").write_text(
        "import pytest\nfrom pathlib import Path\n"
        "@pytest.mark.integration\n@pytest.mark.parametrize('case', range(6))\n"
        "def test_case(case, request):\n"
        f"    with Path({str(executed)!r}).open('a') as out:\n"
        "        out.write(request.node.nodeid + '\\n')\n"
        "    if case == 2: pytest.skip('original skip')\n"
        "    assert case != 4, 'original deliberate failure'\n"
        "@pytest.mark.integration\n@pytest.mark.slow\n"
        "def test_slow(): raise AssertionError('slow body must stay excluded')\n"
    )
    (tmp_path / "tests/test_unselected.py").write_text(
        "import pytest\nfrom pathlib import Path\n"
        f"with Path({str(collected)!r}).open('a') as out: out.write('collected\\n')\n"
        "@pytest.mark.unit\n"
        "def test_unit(): raise AssertionError('unit body must stay excluded')\n"
    )
    (tmp_path / "tests/test_optional.py").write_text(
        "import pytest\npytest.skip('original optional collection skip', allow_module_level=True)\n"
    )
    outcomes = []
    for index in range(3):
        outcomes.append(
            ci_tier.run_planned_partition(
                COMMAND, cwd=tmp_path, count=3, index=index, dry_run=False
            )
        )
    assert sorted(outcomes) == [0, 0, 1]
    expected = {f"tests/test_selected.py::test_case[{case}]" for case in range(6)}
    actual = executed.read_text().splitlines()
    assert len(actual) == len(set(actual)) == 6
    assert set(actual) == expected
    assert collected.read_text().splitlines() == ["collected"] * 3
    plans = list(report.glob("collection-*/selection/selection.json"))
    assert len(plans) == 3
    selected = []
    for manifest in plans:
        node_file, exclusive = ci_tier._verified_selection(manifest.parent)
        metadata = json.loads(manifest.read_text())
        assert (
            "original optional collection skip"
            in (manifest.parent.parent / "collection.xml").read_text()
        )
        assert metadata["collected_tests"] == 6
        assert (
            metadata["collected_sha256"]
            == hashlib.sha256(("\n".join(sorted(expected)) + "\n").encode()).hexdigest()
        )
        assert metadata["shard_count"] == 3
        assert 0 <= metadata["shard_index"] < 3
        assert not exclusive
        selected.extend(node_file.read_text().splitlines())
    assert len(selected) == 6 and set(selected) == expected


def test_collection_errors_propagate_without_executing_partial_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment(tmp_path, monkeypatch)
    executed = tmp_path / "executed"
    (tmp_path / "tests/test_valid.py").write_text(
        "import pytest\nfrom pathlib import Path\n"
        "@pytest.mark.integration\ndef test_valid():\n"
        f"    Path({str(executed)!r}).touch()\n"
    )
    (tmp_path / "tests/test_broken.py").write_text("def invalid syntax\n")
    code = ci_tier.run_planned_partition(
        COMMAND, cwd=tmp_path, count=1, index=0, dry_run=False
    )
    assert code == 2
    assert not executed.exists()


@pytest.mark.parametrize(
    "mutation",
    ["source", "content", "duplicate", "schema", "count", "exclusive", "nonobject"],
)
def test_selection_refuses_unbound_or_changed_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    monkeypatch.setenv("GITHUB_SHA", "synthetic-source")
    raw = b"tests/test_case.py::test_one\n"
    payload: dict[str, Any] = {
        "schema": "flagquantum_pytest_selection_v1",
        "source_sha": "synthetic-source",
        "tests": 1,
        "collected_tests": 1,
        "collected_sha256": hashlib.sha256(raw).hexdigest(),
        "shard_count": 1,
        "shard_index": 0,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "exclusive": False,
    }
    if mutation == "source":
        payload["source_sha"] = "other-source"
    elif mutation == "content":
        raw += b"tests/test_case.py::test_two\n"
    elif mutation == "duplicate":
        raw *= 2
        payload.update(tests=2, sha256=hashlib.sha256(raw).hexdigest())
    elif mutation == "schema":
        payload["schema"] = "unknown"
    elif mutation == "count":
        payload["tests"] = True
    elif mutation == "exclusive":
        payload["exclusive"] = 1
    (tmp_path / "nodes.txt").write_bytes(raw)
    (tmp_path / "selection.json").write_text(
        json.dumps([] if mutation == "nonobject" else payload)
    )
    with pytest.raises(ValueError, match="manifest"):
        ci_tier._verified_selection(tmp_path)


@pytest.mark.parametrize("budget_exit", [0, 23])
def test_sole_heavy_case_uses_the_existing_admission_owner_and_retains_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget_exit: int
) -> None:
    old_report = environment(tmp_path, monkeypatch)
    original_resource = {
        "effective_cpus": 4,
        "shared_registrations_budgeted": 1,
        "workers": 4,
        "threads_per_worker": 1,
    }
    (old_report / "resources.json").write_text(json.dumps(original_resource))
    github_env = tmp_path / "github.env"
    github_env.touch()
    monkeypatch.setenv("GITHUB_ENV", str(github_env))
    (tmp_path / "conftest.py").write_text(
        "from tools.ci import pytest_shard\n"
        "pytest_shard.load_duration_hints = lambda: (0.05, {'tests/test_heavy.py::test_case': 150.0})\n"
    )
    executed = tmp_path / "executed"
    (tmp_path / "tests/test_heavy.py").write_text(
        "import os, pytest\nfrom pathlib import Path\n"
        "@pytest.mark.integration\ndef test_case():\n"
        "    assert os.environ['FQ_PYTEST_WORKERS'] == '1'\n"
        "    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:\n"
        "        assert os.environ[name] == '2'\n"
        f"    Path({str(executed)!r}).touch()\n"
    )
    new_report = tmp_path / "exclusive-report"
    real_run = subprocess.run
    seen: list[tuple[str, ...]] = []

    def run(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        seen.append(tuple(command))
        if command[-1] != "--exclusive-test":
            return real_run(command, **kwargs)
        assert tuple(command) == (
            sys.executable,
            "-I",
            str(tmp_path / "tools/ci/ci_cpu_budget.py"),
            "--exclusive-test",
        )
        if budget_exit:
            return subprocess.CompletedProcess(command, budget_exit)
        new_report.mkdir()
        (new_report / "resources.json").write_text(
            json.dumps(
                {
                    "effective_cpus": 2,
                    "shared_registrations_budgeted": 1,
                    "workers": 1,
                    "threads_per_worker": 2,
                }
            )
        )
        emitted = {
            "FQ_PYTEST_WORKERS": "1",
            "FQ_DISTRIBUTED_WORKERS": "1",
            "FQ_CI_REPORT_DIR": str(new_report),
            "FQ_CI_HOST_BUDGET": str(old_report / "resources.json"),
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
        }
        Path(kwargs["env"]["GITHUB_ENV"]).write_text(
            "".join(f"{key}={value}\n" for key, value in emitted.items())
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(ci_tier.subprocess, "run", run)
    code = ci_tier.run_planned_partition(
        COMMAND, cwd=tmp_path, count=1, index=0, dry_run=False
    )
    assert code == budget_exit
    assert executed.exists() is (budget_exit == 0)
    assert (old_report / "resources.json").read_text() == json.dumps(original_resource)
    if not budget_exit:
        assert seen[-1][-2:] == ("-n", "1")
        assert (
            json.loads((new_report / "collection/resources.json").read_text())
            == original_resource
        )
        assert (new_report / "collection/selection/selection.json").is_file()
        assert (new_report / "pytest.xml").is_file()
        assert not (old_report / "pytest.xml").exists()
        assert "FQ_CI_REPORT_DIR=" + str(new_report) in github_env.read_text()


def test_local_planned_selection_does_not_invent_a_ci_resource_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment(tmp_path, monkeypatch)
    monkeypatch.delenv("FQ_CI_REPORT_DIR")
    (tmp_path / "tests/test_local.py").write_text(
        "import pytest\n@pytest.mark.integration\ndef test_local(): assert True\n"
    )
    assert (
        ci_tier.run_planned_partition(
            COMMAND, cwd=tmp_path, count=1, index=0, dry_run=False
        )
        == 0
    )


def test_planned_cli_keeps_the_original_runtime_selector_and_exit_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def planned(command: ci_tier.Command, **kwargs: Any) -> int:
        calls.append((command, kwargs))
        return 17

    monkeypatch.setattr(ci_tier, "run_planned_partition", planned)
    assert (
        ci_tier.main(
            [
                "pr-runtime",
                "--shard-count",
                "128",
                "--shard-index",
                "19",
                "--plan-shard",
            ]
        )
        == 17
    )
    assert calls == [
        (COMMAND, {"cwd": ROOT, "count": 128, "index": 19, "dry_run": False})
    ]


def test_planned_cli_refuses_to_drop_other_tier_commands() -> None:
    assert ci_tier.main(["pr-distributed", "--plan-shard"]) == 2

"""The scheduled CPU entry preserves source roles and executes the existing tier."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

from tools.ci_tier import CI_TIERS

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/cpu-nightly.yml"


def _workflow() -> dict[str, Any]:
    # BaseLoader keeps the external workflow vocabulary's `on` key literal.
    document = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    assert isinstance(document, dict)
    return document


def test_nightly_native_scheduling_and_permissions() -> None:
    document = _workflow()
    assert set(document["on"]) == {"schedule", "workflow_dispatch"}
    assert document["on"]["schedule"] == [{"cron": "0 11 * * 1-5"}]
    assert document["permissions"] == {"contents": "read"}
    assert document["concurrency"]["cancel-in-progress"] == "false"
    assert document["concurrency"]["queue"] == "max"
    assert not CI_TIERS["nightly"].blocks_default_pr


def test_nightly_environment_is_cpu_only_and_prepared_before_execution() -> None:
    document = _workflow()
    assert set(document["jobs"]) == {"nightly-cpu"}
    job = document["jobs"]["nightly-cpu"]
    assert job["runs-on"] == "ubuntu-24.04"
    assert job["timeout-minutes"] == "120"
    assert job["concurrency"] == {"group": "flagquantum-gh-4", "queue": "max"}
    assert job["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert job["env"]["JAX_PLATFORMS"] == "cpu"
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert job["env"][variable] == "1"
    steps = job["steps"]
    actions = [step.get("uses") for step in steps]
    assert actions[0] == "actions/checkout@v5"
    assert steps[0]["with"]["persist-credentials"] == "false"
    assert actions.index("./.github/actions/isolated-python") < actions.index(
        "./.github/actions/install-cpu-torch"
    )
    graphviz = next(step for step in steps if "setup-graphviz@" in step.get("uses", ""))
    assert graphviz["uses"].endswith("c001ccfb5aff62e28bda6a6c39b59a7e061be5b9")
    install = next(
        step for step in steps if "--print-test-environment" in step.get("run", "")
    )
    assert "ftqc-reference" in install["run"] and "pip check" in install["run"]
    prerequisites = next(step for step in steps if "dot -V" in step.get("run", ""))
    assert 'hasattr(kickmix, "Simulator")' in prerequisites["run"]
    execution = next(
        step for step in steps if step.get("run") == "python tools/ci_tier.py nightly"
    )
    assert steps.index(install) < steps.index(prerequisites) < steps.index(execution)
    assert execution["timeout-minutes"] == "90"
    assert not any(step.get("continue-on-error") == "true" for step in steps)


@pytest.mark.parametrize("failing", (False, True))
def test_real_nightly_dispatch_selection_and_exit_status(
    tmp_path: Path, failing: bool
) -> None:
    """Execute the actual workflow command on a small isolated pytest suite."""
    job = _workflow()["jobs"]["nightly-cpu"]
    command = next(
        step["run"]
        for step in job["steps"]
        if step.get("run") == "python tools/ci_tier.py nightly"
    )
    arguments = shlex.split(command)
    config = tmp_path / "pytest.ini"
    config.write_bytes((ROOT / "pytest.ini").read_bytes())
    fixture = tmp_path / "test_nightly_selection.py"
    fixture.write_text(textwrap.dedent("""
        import os
        from pathlib import Path
        import pytest

        def record(name):
            with Path(os.environ["FQ_NIGHTLY_TEST_RECORD"]).open("a") as stream:
                stream.write(name + "\\n")

        def test_cpu():
            record("cpu")
            assert os.environ["FQ_NIGHTLY_TEST_FAIL"] == "0"

        @pytest.mark.slow
        def test_slow_cpu():
            record("slow_cpu")

        @pytest.mark.distributed_cpu
        def test_distributed_cpu():
            record("distributed_cpu")

        @pytest.mark.release_gate
        def test_release_gate():
            record("release_gate")
    """))
    with fixture.open("a") as stream:
        for marker in (
            "benchmark_contract",
            "scalability",
            "distributed_multinode",
            "distributed_accel",
            "gpu",
        ):
            stream.write(
                f"\n@pytest.mark.{marker}\ndef test_{marker}():\n    raise AssertionError('Excluded hardware or benchmark ran')\n"
            )
    record = tmp_path / "selected.txt"
    env = dict(os.environ, **job["env"])
    env.update(
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_ADDOPTS=job["env"]["PYTEST_ADDOPTS"]
        + " "
        + shlex.join(["-p", "xdist.plugin", "-c", str(config), str(fixture)]),
        FQ_NIGHTLY_TEST_RECORD=str(record),
        FQ_NIGHTLY_TEST_FAIL="1" if failing else "0",
    )
    result = subprocess.run(
        [sys.executable, *arguments[1:]],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == (1 if failing else 0), result.stdout + result.stderr
    assert set(record.read_text().splitlines()) == {
        "cpu",
        "slow_cpu",
        "distributed_cpu",
        "release_gate",
    }
    assert "5 deselected" in result.stdout

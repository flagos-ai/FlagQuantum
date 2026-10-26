"""Admission evidence and deterministic routing keep unqualified sets out of CI."""

import copy
from pathlib import Path

import pytest

from tools.ci.r300_pool import load_pool, plan, qualification_targets

pytestmark = pytest.mark.unit


def test_every_route_is_qualified_and_jobs_use_distinct_targets() -> None:
    pool = load_pool()
    qualified = {m["label"] for m in pool["members"] if m["state"] == "qualified"}
    observed: set[str] = set()
    for seed in range(100):
        routes = plan(pool, seed)
        assert routes == plan(pool, seed)
        assert set(routes) == set(pool["jobs"])
        assert len(set(routes.values())) == len(routes)
        assert set(routes.values()) <= qualified
        observed.update(routes.values())
    assert observed == qualified


def test_candidates_and_quarantined_sets_never_receive_production_jobs() -> None:
    pool = load_pool()
    blocked = {m["label"] for m in pool["members"] if m["state"] != "qualified"}
    assert blocked
    for seed in range(len(pool["members"])):
        assert not blocked.intersection(plan(pool, seed).values())
    candidates = qualification_targets(pool, 4)
    assert len(candidates) == min(
        4,
        sum(
            m["state"] == "candidate" and m["label"] in pool["qualification_batch"]
            for m in pool["members"]
        ),
    )
    assert all(
        next(m for m in pool["members"] if m["label"] == label)["state"] == "candidate"
        for label in candidates
    )


def test_quarantine_immediately_removes_a_member_from_new_plans() -> None:
    pool = copy.deepcopy(load_pool())
    qualified = [m for m in pool["members"] if m["state"] == "qualified"]
    removed = qualified[0]
    removed["state"] = "quarantined"
    pool["jobs"] = pool["jobs"][: len(qualified) - 1]
    for seed in range(20):
        assert removed["label"] not in plan(pool, seed).values()


def test_insufficient_qualified_members_fail_without_routing_to_candidates() -> None:
    pool = copy.deepcopy(load_pool())
    pool["jobs"] += [f"extra-job-{i}" for i in range(len(pool["members"]))]
    with pytest.raises(ValueError, match="distinct qualified"):
        plan(pool, 1)


@pytest.mark.parametrize("limit", [0, 5, -1])
def test_qualification_fanout_is_bounded(limit: int) -> None:
    with pytest.raises(ValueError, match="at most four"):
        qualification_targets(load_pool(), limit)


def test_registry_rejects_duplicate_targets_and_missing_evidence(
    tmp_path: Path,
) -> None:
    import json

    pool = load_pool()
    pool["members"].append(pool["members"][0])
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(pool))
    with pytest.raises(ValueError, match="duplicate"):
        load_pool(path)
    pool = load_pool()
    next(m for m in pool["members"] if m["state"] == "qualified").pop("evidence")
    path.write_text(json.dumps(pool))
    with pytest.raises(ValueError, match="Missing qualification"):
        load_pool(path)


def test_pool_qualification_stays_manual_while_pr_routing_is_paused() -> None:
    import yaml

    workflows = Path(__file__).resolve().parents[2] / ".github/workflows"
    jobs = yaml.safe_load((workflows / "ci.yml").read_text())["jobs"]
    assert "r300-routes" not in jobs["prepare-source"]["outputs"]
    assert all(
        "outputs.r300-routes" not in str(job.get("runs-on", ""))
        and "flagcicd-r300-" not in str(job.get("runs-on", ""))
        for job in jobs.values()
    )
    validation = yaml.safe_load((workflows / "r300-validation.yml").read_text())
    events = validation.get("on") or validation[True]
    assert set(events) == {"workflow_dispatch"}
    qualification = validation["jobs"]["pool-qualification"]
    assert "github.event_name == 'workflow_dispatch'" in qualification["if"]
    assert "inputs.qualify_pool" in qualification["if"]
    assert qualification["strategy"]["max-parallel"] == 4
    assert qualification["strategy"]["matrix"]["scale_set"] == (
        "${{ fromJSON(needs.prepare-source.outputs.qualification-targets) }}"
    )


def test_cli_outputs_valid_workflow_json(tmp_path: Path) -> None:
    import json
    import subprocess
    import sys

    output = tmp_path / "outputs"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.ci.r300_pool",
            "--seed",
            "42",
            "--github-output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert json.loads(values["r300-routes"]) == plan(load_pool(), 42)
    assert json.loads(values["qualification-targets"]) == qualification_targets(
        load_pool(), 4
    )


def test_completed_batch_does_not_start_an_unselected_batch() -> None:
    pool = copy.deepcopy(load_pool())
    for member in pool["members"]:
        if member["label"] in pool["qualification_batch"]:
            member["state"] = "qualified"
    assert any(member["state"] == "candidate" for member in pool["members"])
    assert qualification_targets(pool, 4) == []


def test_manual_network_retest_targets_previous_failures_without_admission(
    tmp_path: Path,
) -> None:
    import json
    import os
    import subprocess

    import yaml

    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load(
        (root / ".github/workflows/r300-validation.yml").read_text()
    )
    prepare = workflow["jobs"]["prepare-source"]
    selector = next(step for step in prepare["steps"] if step.get("id") == "pool")
    output = tmp_path / "outputs"
    registry = (root / "tools/ci/r300_pool.json").read_bytes()
    subprocess.run(
        ["bash", "-e", "-c", selector["run"]],
        cwd=root,
        env={**os.environ, "DIAGNOSE_NETWORK": "true", "GITHUB_OUTPUT": str(output)},
        check=True,
        timeout=5,
    )
    key, value = output.read_text().strip().split("=", 1)
    assert key == "qualification-targets"
    assert json.loads(value) == [
        f"flagcicd-r300-cpu{n}" for n in ("03", "04", "05", "16")
    ]
    assert (root / "tools/ci/r300_pool.json").read_bytes() == registry
    assert "!inputs.diagnose_network" in workflow["jobs"]["resources"]["if"]
    assert "!inputs.diagnose_network" in workflow["jobs"]["cpu-workload"]["if"]
    qualification = workflow["jobs"]["pool-qualification"]
    assert "inputs.diagnose_network" in qualification["if"]
    probe = qualification["steps"][0]
    assert probe["if"] == "inputs.diagnose_network"
    assert probe["timeout-minutes"] == 8
    assert qualification["steps"][1]["uses"].startswith("actions/download-artifact@")


def test_network_retest_cannot_claim_a_cached_torch_as_a_download() -> None:
    import yaml

    root = Path(__file__).resolve().parents[2]
    action = yaml.safe_load(
        (root / ".github/workflows/r300-validation.yml").read_text()
    )
    steps = action["jobs"]["pool-qualification"]["steps"]
    installer = next(
        i
        for i, step in enumerate(steps)
        if step.get("uses") == "./.github/actions/install-cpu-torch"
    )
    cold = steps[installer - 1]
    assert cold["if"] == "inputs.diagnose_network"
    assert 'test -n "$VIRTUAL_ENV"' in cold["run"]
    assert "python -m pip uninstall -y torch" in cold["run"]
    assert steps[installer]["env"]["PIP_NO_CACHE_DIR"] == "1"
    assert steps[installer]["env"]["FQ_CPU_WHEEL_CACHE"].startswith(
        "${{ runner.temp }}"
    )


@pytest.mark.parametrize("bootstrap_kind", ["prepare", "probe"])
@pytest.mark.parametrize("existing", ["", "localhost,.example.test"])
def test_domestic_direct_routing_preserves_other_proxy_configuration(
    tmp_path: Path, bootstrap_kind: str, existing: str
) -> None:
    import os
    import subprocess

    import yaml

    root = Path(__file__).resolve().parents[2]
    action = yaml.safe_load(
        (root / ".github/actions/prepare-r300-cpu/action.yml").read_text()
    )
    if bootstrap_kind == "prepare":
        script = action["runs"]["steps"][0]["run"].split(". /etc/os-release", 1)[0]
    else:
        workflow = yaml.safe_load(
            (root / ".github/workflows/r300-validation.yml").read_text()
        )
        script = workflow["jobs"]["pool-qualification"]["steps"][0]["run"].split(
            "printf 'UTC:", 1
        )[0]
    output = tmp_path / "env"
    result = subprocess.run(
        [
            "bash",
            "-eu",
            "-c",
            script
            + script
            + 'printf "RESULT=%s|%s|%s\\n" "$NO_PROXY" "$no_proxy" "$HTTPS_PROXY"',
        ],
        env={
            **os.environ,
            "GITHUB_ENV": str(output),
            "NO_PROXY": existing,
            "no_proxy": existing,
            "HTTPS_PROXY": "http://proxy.example.test:3128",
        },
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    upper, lower, proxy = result.stdout.split("RESULT=", 1)[1].strip().split("|")
    expected = ([existing] if existing else []) + [
        "mirrors.tuna.tsinghua.edu.cn",
        "pypi.tuna.tsinghua.edu.cn",
    ]
    assert upper == lower == ",".join(expected)
    assert proxy == "http://proxy.example.test:3128"
    assert {line.split("=", 1)[0] for line in output.read_text().splitlines()} == {
        "NO_PROXY",
        "no_proxy",
    }
    assert (
        "Acquire::https::Proxy::mirrors.tuna.tsinghua.edu.cn=DIRECT"
        in action["runs"]["steps"][0]["run"]
    )

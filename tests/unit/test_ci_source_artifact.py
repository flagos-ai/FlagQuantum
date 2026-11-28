"""Round-trip the exact source scripts the A800 jobs execute."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    os.name == "nt", reason="A800 source restoration runs in Linux bash"
)
@pytest.mark.parametrize(
    "workflow,job",
    [
        ("ci.yml", "cpu-runtime"),
        ("ci.yml", "jax-optional"),
        ("ci.yml", "azure-optional"),
        ("ci.yml", "triton-optional"),
        pytest.param("ci.yml", "qiskit-compat", id="ci.yml-qiskit-optional"),
        ("ci.yml", "cirq-optional"),
        ("ci.yml", "braket-optional"),
        ("ci.yml", "cudaq-optional"),
        ("ci.yml", "pennylane-optional"),
        ("ci.yml", "dependency-bounds"),
        pytest.param("ci.yml", "distributed-cpu-shard", id="ci.yml-distributed-cpu"),
        ("ci.yml", "coverage-plan"),
        ("ci.yml", "coverage-shard"),
        ("ci.yml", "coverage"),
        ("ci.yml", "supply-chain"),
        ("local-gpu.yml", "one-gpu-local"),
    ],
)
def test_source_artifact_restores_the_exact_tree_and_rejects_another_commit(
    tmp_path: Path, workflow: str, job: str
) -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())["jobs"]
    pack = next(
        step["run"]
        for step in jobs["prepare-source"]["steps"]
        if step.get("id") == "pack-source"
    )
    restore = next(
        step["run"]
        for step in jobs[job]["steps"]
        if step.get("name") == "Restore and verify this run's exact source commit"
    )
    probe = next(
        (
            step["run"]
            for step in jobs[job]["steps"]
            if step.get("id") == "source-cache"
        ),
        None,
    )
    source = tmp_path / "source"
    target = tmp_path / "target"
    temporary = tmp_path / "temporary"
    for directory in (source, target, temporary):
        directory.mkdir()
    environment = {
        **os.environ,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "RUNNER_TEMP": str(temporary),
    }

    def git(*arguments: str, cwd: Path = source) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()

    def shell(
        script: str, cwd: Path, *, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "-eo", "pipefail", "-c", script],
            cwd=cwd,
            env=environment,
            check=check,
            capture_output=True,
            text=True,
            timeout=20,
        )

    git("init")
    git("config", "user.name", "FlagQuantum")
    git("config", "user.email", "liuwei.chem.phys@gmail.com")
    (source / "tracked.txt").write_text("parent\n")
    (source / "unchanged.txt").write_text("cached baseline\n")
    (source / "tools").mkdir()
    helper = ROOT / "tools/ci_source_archive.py"
    (source / "tools/ci_source_archive.py").write_bytes(helper.read_bytes())
    git("add", ".")
    git("commit", "-m", "parent")
    parent = git("rev-parse", "HEAD")
    (source / "tracked.txt").write_text("current\n")
    (source / "binary.dat").write_bytes(bytes(range(256)))
    (source / "link").symlink_to("tracked.txt")
    git("add", ".")
    git("commit", "-m", "current")
    sha = git("rev-parse", "HEAD")
    environment["GITHUB_SHA"] = sha
    output = temporary / "outputs"
    environment.update(
        GITHUB_RUN_ID="123", GITHUB_RUN_ATTEMPT="1", GITHUB_OUTPUT=str(output)
    )
    # A previous consumer or producer may have left these shared paths behind.
    for name in ("ci-source", "ci-source-incremental"):
        (temporary / name).mkdir()
        (temporary / name / "retained.txt").write_text("previous job\n")
    shell(pack, source)
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert outputs["artifact-key"] == "123-1"
    producer = Path(outputs["staging-root"])
    full = Path(outputs["full-path"])
    incremental = Path(outputs["incremental-path"])
    retained_pack = (full / "source.pack").read_bytes()
    shell(pack, source)
    repeated = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert repeated["staging-root"] != str(producer)
    assert (full / "source.pack").read_bytes() == retained_pack
    assert (Path(repeated["full-path"]) / "source.pack").read_bytes() == retained_pack
    for step in jobs["prepare-source"]["steps"]:
        artifact = step.get("with", {}).get("name", "")
        if artifact.startswith("ci-source"):
            key = "incremental-path" if "incremental" in artifact else "full-path"
            assert step["with"]["path"] == f"${{{{ steps.pack-source.outputs.{key} }}}}"
    # Downloaded artifact contents retain their consumer-facing paths.
    shutil.copytree(full, temporary / "ci-source", dirs_exist_ok=True)
    shutil.copytree(
        incremental, temporary / "ci-source-incremental", dirs_exist_ok=True
    )
    # A failed consumer can rerun without rerunning the successful producer.
    environment["GITHUB_RUN_ATTEMPT"] = "2"
    artifact_key = jobs["prepare-source"]["outputs"]["artifact-key"]
    assert artifact_key == "${{ steps.pack-source.outputs.artifact-key }}"
    for step in jobs[job]["steps"]:
        if step.get("uses", "").startswith("actions/download-artifact@") and step.get(
            "with", {}
        ).get("name", "").startswith("ci-source"):
            assert (
                "${{ needs.prepare-source.outputs.artifact-key }}"
                in step["with"]["name"]
            )
    full_pack = (temporary / "ci-source/source.pack").read_bytes()
    shutil.copyfile(
        temporary / "ci-source-incremental/source.pack",
        temporary / "ci-source/source.pack",
    )
    if probe:
        shell(probe, target)
        assert output.read_text().splitlines()[-1] == "complete=false"
        assert (
            git("show", f"{sha}:tools/ci_source_archive.py", cwd=target)
            == helper.read_text().strip()
        )
    (temporary / "ci-source/source.pack").write_bytes(full_pack)

    for attempt in range(2):
        if attempt:
            git("sparse-checkout", "set", "--no-cone", "tracked.txt", cwd=target)
            (target / "tracked.txt").write_text("dirty\n")
            (target / "leftover.txt").write_text("stale\n")
        shell(restore, target)
        shutil.copyfile(
            temporary / "ci-source-incremental/source.pack",
            temporary / "ci-source/source.pack",
        )
        if probe:
            shell(probe, target)
            assert output.read_text().splitlines()[-1] == "complete=true"
        assert git("rev-parse", "HEAD", cwd=target) == sha
        assert git("status", "--porcelain", cwd=target) == ""
        assert (target / "tracked.txt").read_text() == "current\n"
        assert (target / "unchanged.txt").read_text() == "cached baseline\n"
        assert (target / "binary.dat").read_bytes() == bytes(range(256))
        assert (target / "link").is_symlink()
        assert not (target / "leftover.txt").exists()

    (temporary / "ci-source/source.sha").write_text(parent + "\n")
    assert shell(restore, target, check=False).returncode != 0
    assert git("rev-parse", "HEAD", cwd=target) == sha

    cleanup = next(
        step
        for step in jobs["prepare-source"]["steps"]
        if step.get("name") == "Remove this producer's temporary source packs"
    )
    assert "always()" in cleanup["if"]
    # A failed producer still publishes its private root for final cleanup.
    environment["GITHUB_SHA"] = parent
    assert shell(pack, source, check=False).returncode != 0
    failed = dict(line.split("=", 1) for line in output.read_text().splitlines())
    for root in (str(producer), repeated["staging-root"], failed["staging-root"]):
        environment["FQ_SOURCE_STAGING"] = root
        assert Path(root).is_dir()
        shell(cleanup["run"], source)
        assert not Path(root).exists()
    for name in ("ci-source", "ci-source-incremental"):
        assert (temporary / name / "retained.txt").read_text() == "previous job\n"


@pytest.mark.parametrize(
    "job",
    [
        pytest.param("distributed-cpu-shard", id="distributed-cpu"),
        "coverage-shard",
        "coverage",
        "dependency-bounds",
    ],
)
def test_checkout_jobs_use_the_exact_workflow_revision(job: str) -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    steps = jobs[job]["steps"]
    assert not any(
        step.get("uses", "").startswith("actions/checkout@") for step in steps
    )
    restore = next(
        step["run"]
        for step in steps
        if step.get("name") == "Restore and verify this run's exact source commit"
    )
    assert 'test "$(git rev-parse HEAD)" = "$GITHUB_SHA"' in restore
    download = next(
        step
        for step in steps
        if step.get("uses", "").startswith("actions/download-artifact@")
    )
    assert download["with"]["digest-mismatch"] == "error"


def test_partial_source_caches_get_bounded_verified_api_repair_before_full_transfer() -> (
    None
):
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    consumers = []
    for name, job in jobs.items():
        steps = job["steps"]
        if not any(step.get("id") == "source-cache" for step in steps):
            continue
        consumers.append(name)
        repair = next(step for step in steps if step.get("id") == "source-repair")
        assert (
            repair["if"]
            == "runner.environment == 'self-hosted' && steps.source-cache.outputs.complete != 'true'"
        )
        assert 'git show "$sha:tools/ci_source_archive.py"' in repair["run"]
        assert 'test "$sha" = "$GITHUB_SHA"' in repair["run"]
        assert "timeout --kill-after=15s 120s python3" in repair["run"]
        assert 'echo "complete=true"' in repair["run"]
        assert 'echo "complete=false"' in repair["run"]
        assert not repair.get("continue-on-error", False)
        full = next(
            step
            for step in steps
            if step.get("with", {}).get("name", "").startswith("ci-source-${{")
        )
        assert (
            full["if"]
            == "steps.source-cache.outputs.complete != 'true' && steps.source-repair.outputs.complete != 'true'"
        )
        assert full["with"]["digest-mismatch"] == "error"
    assert {
        "coverage",
        "coverage-shard",
        "distributed-cpu-shard",
        "qiskit-compat",
        "dependency-bounds",
    } <= set(consumers)


@pytest.mark.skipif(os.name == "nt", reason="Source repair runs in Linux bash")
@pytest.mark.parametrize(
    "transport,api_success,expected",
    [
        ("success", False, True),
        ("failure", True, True),
        ("failure", False, False),
        ("wrong-head", False, False),
        ("missing-object", False, False),
    ],
)
def test_dependency_source_repair_verifies_git_transport_before_api_fallback(
    tmp_path: Path, transport: str, api_success: bool, expected: bool
) -> None:
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    repair = next(
        step["run"]
        for step in jobs["dependency-bounds"]["steps"]
        if step.get("id") == "source-repair"
    )
    environment = {
        **os.environ,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_REPOSITORY": "fixture/source",
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "TRANSPORT": transport,
    }
    source = tmp_path / "origin"
    source.mkdir()

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=source,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()

    git("init")
    git("config", "user.name", "FlagQuantum")
    git("config", "user.email", "liuwei.chem.phys@gmail.com")
    (source / "tools").mkdir()
    (source / "tools/ci_source_archive.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(tmp_path / 'api-called')!r}).touch()\n"
        f"raise SystemExit({0 if api_success else 1})\n"
    )
    git("add", ".")
    git("commit", "-m", "Fixture source")
    sha = git("rev-parse", "HEAD")
    environment.update(GITHUB_SHA=sha, FIXTURE_ORIGIN=str(source))
    (tmp_path / "ci-source").mkdir()
    (tmp_path / "ci-source/source.sha").write_text(sha + "\n")
    target = tmp_path / "target"
    subprocess.run(
        ["git", "clone", "--no-checkout", str(source), str(target)],
        env=environment,
        check=True,
        capture_output=True,
        timeout=20,
    )
    # Only the network boundary is replaced; object and commit checks use Git.
    boundary = """
    timeout() { shift 2; "$@"; }
    git() {
      if test "$1" = -c; then
        test "$2" = http.version=HTTP/1.1 || return 90
        test "$3" = fetch || return 91
        test "$4" = --no-tags || return 92
        test "$5" = --depth=1 || return 93
        test "$6" = https://github.com/fixture/source.git || return 94
        test "$7" = "$GITHUB_SHA" || return 95
        test "$TRANSPORT" != failure || return 1
        command git fetch --no-tags --depth=1 "$FIXTURE_ORIGIN" "$7"
      elif test "$1 $2" = 'rev-parse FETCH_HEAD' &&
           test "$TRANSPORT" = wrong-head; then
        printf '%040d\\n' 0
      elif test "$1" = rev-list && test "$TRANSPORT" = missing-object; then
        printf '?missing-object\\n'
      else
        command git "$@"
      fi
    }
    """
    subprocess.run(
        ["bash", "-eo", "pipefail", "-c", boundary + repair],
        cwd=target,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (tmp_path / "output").read_text() == f"complete={str(expected).lower()}\n"
    assert (tmp_path / "api-called").exists() == (transport != "success")


def test_source_cache_stops_previous_runner_writers_before_checkout() -> None:
    for workflow in ("ci.yml", "local-gpu.yml"):
        jobs = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())[
            "jobs"
        ]
        checked = 0
        for job in jobs.values():
            probe = next(
                (
                    s.get("run", "")
                    for s in job["steps"]
                    if s.get("id") == "source-cache"
                ),
                None,
            )
            if probe is None:
                continue
            checked += 1
            assert 'git show "$sha:tools/ci_source_archive.py"' in probe
            assert "--cleanup-containers" in probe
            assert probe.index("--cleanup-containers") < probe.index("complete=false")
        assert checked


@pytest.mark.skipif(os.name == "nt", reason="Source packing runs in Linux bash")
@pytest.mark.parametrize(
    "mode", ["pull_request", "override", "push", "linear", "invalid"]
)
def test_source_pack_uses_the_previous_pr_tree_with_verified_full_fallback(
    tmp_path: Path, mode: str
) -> None:
    """Exclude cached PR objects, retaining cold-cache and non-PR behavior."""
    jobs = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())["jobs"]
    pack = next(
        step["run"]
        for step in jobs["prepare-source"]["steps"]
        if step.get("id") == "pack-source"
    )
    source = tmp_path / "source"
    source.mkdir()
    environment = {
        **os.environ,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_EVENT_NAME": "push" if mode == "push" else "pull_request",
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_OUTPUT": str(tmp_path / "outputs"),
    }
    environment.pop("FQ_CI_SOURCE_BASE", None)

    def git(*arguments: str, cwd: Path = source, data: bytes | None = None) -> str:
        return (
            subprocess.run(
                ["git", *arguments],
                cwd=cwd,
                env=environment,
                input=data,
                check=True,
                capture_output=True,
                timeout=20,
            )
            .stdout.decode()
            .strip()
        )

    git("init", "-b", "baseline")
    git("config", "user.name", "FlagQuantum")
    git("config", "user.email", "liuwei.chem.phys@gmail.com")
    (source / "tools").mkdir()
    helper = (ROOT / "tools/ci_source_archive.py").read_bytes()
    (source / "tools/ci_source_archive.py").write_bytes(helper)
    git("add", ".")
    git("commit", "-m", "baseline")
    baseline = git("rev-parse", "HEAD")
    git("checkout", "-b", "topic")
    (source / "cached.txt").write_text("previous PR tree\n")
    git("add", ".")
    git("commit", "-m", "previous PR head")
    cached_blob = git("rev-parse", "HEAD:cached.txt")
    (source / "current.txt").write_text("current PR tree\n")
    git("add", ".")
    git("commit", "-m", "current PR head")
    if mode != "linear":
        git("checkout", "baseline")
        (source / "main.txt").write_text("new main tree\n")
        git("add", ".")
        git("commit", "-m", "main advanced")
        git("merge", "--no-ff", "topic", "-m", "synthetic PR merge")
    if mode == "override":
        environment["FQ_CI_SOURCE_BASE"] = baseline
    elif mode == "invalid":
        environment["FQ_CI_SOURCE_BASE"] = "missing-explicit-base"
    sha = git("rev-parse", "HEAD")
    environment["GITHUB_SHA"] = sha
    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", pack],
        cwd=source,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if mode == "invalid":
        assert result.returncode != 0
        return
    assert result.returncode == 0, result.stderr
    outputs = dict(
        line.split("=", 1)
        for line in Path(environment["GITHUB_OUTPUT"]).read_text().splitlines()
    )
    for kind in ("incremental", "full"):
        target = tmp_path / kind
        target.mkdir()
        git("init", cwd=target)
        artifact = Path(outputs[f"{kind}-path"])
        assert (artifact / "source.sha").read_text().strip() == sha
        git(
            "index-pack",
            "--stdin",
            cwd=target,
            data=(artifact / "source.pack").read_bytes(),
        )
        assert git("rev-parse", sha, cwd=target) == sha
        assert (
            git("show", f"{sha}:tools/ci_source_archive.py", cwd=target)
            == helper.decode().strip()
        )
        assert git("show", f"{sha}:current.txt", cwd=target) == "current PR tree"
        present = (
            subprocess.run(
                ["git", "cat-file", "-e", cached_blob],
                cwd=target,
                env=environment,
                capture_output=True,
                timeout=20,
            ).returncode
            == 0
        )
        assert present == (kind == "full" or mode != "pull_request")
        if kind == "full":
            # A consumer with no prior objects still obtains the complete tree.
            git("checkout", "--detach", sha, cwd=target)
            assert (target / "cached.txt").read_text() == "previous PR tree\n"
            assert git("status", "--porcelain", cwd=target) == ""

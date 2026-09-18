"""What the two-node lane must hold true, checked without the two nodes.

The lane only exists on hosts that can reach each other, and nearly every way
it can go wrong is one thing done on one node and not the other: an interface
one rank takes and its peer does not, a file one rank writes where the other
cannot read it, a launcher that outlives the client that started it. Those are
the properties below.

Every check that can be driven by a fake is driven by one, so the rules hold on
a machine with no peer and no device. The checks that cannot be faked -- a real
launch on the real pair -- are not pretended here.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import tools.multinode_launch_plan as plan

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "scheduled-hardware.yml"

PYTHON = "/opt/conda/envs/flagquantum/bin/python"
STAGING = "/nfs/fq-multinode-unit"
OUTPUT = "/nfs/fq-multinode-unit-out"


def _plan(**overrides):
    kwargs = {
        "peer_host": "peer-node",
        "launch_address": "10.0.0.1",
        "staging": STAGING,
        "output_directory": OUTPUT,
        "python": PYTHON,
        "source_revision": "deadbeef",
    }
    kwargs.update(overrides)
    return plan.build_plan(**kwargs)


def _by_name(payload) -> dict[str, dict]:
    return {job["name"]: job for job in payload["jobs"]}


def _environment(command: list[str]) -> dict[str, str]:
    """The `env A=1` prefix, as a mapping, on either side of the ssh wrapper."""
    assert command[0] in {"env", "ssh"}, command[:1]
    stop = command.index(PYTHON)
    start = command.index("env") + 1
    return dict(token.split("=", 1) for token in command[start:stop])


def _flag(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


# --- the plan the watchdog consumes -----------------------------------------


def test_the_plan_is_two_ranks_one_per_node() -> None:
    payload = _plan()

    assert payload["node_count"] == plan.NODE_COUNT == 2
    assert payload["local_world_size"] == 1
    names = [job["name"] for job in payload["jobs"]]
    assert len(names) == len(set(names)), names
    assert sorted(_flag(job["command"], "--node-rank") for job in payload["jobs"]) == [
        "0",
        "1",
    ]
    for job in payload["jobs"]:
        # The watchdog requires two jobs with unique non-empty names and a
        # command each, and it silently ignores any other key: a `timeout` or
        # `retries` added here would read as configured and do nothing.
        assert set(job) <= {"name", "command", "cwd", "env", "cleanup_command"}
        assert job["name"] and job["command"] and job["cwd"] == STAGING
        # Both nodes have to be killable. The watchdog signals the process
        # group it started, which on the peer holds only the ssh client, so a
        # plan without this leaves a launcher running on the other node.
        assert job["cleanup_command"]
    assert json.loads(json.dumps(payload)) == payload


def test_both_ranks_are_given_the_same_network_environment() -> None:
    jobs = _by_name(_plan())
    ranks = {
        rank: _environment(
            jobs[f"rank-{rank}-{'launch-host' if rank == 0 else 'peer'}"]["command"]
        )
        for rank in (0, 1)
    }

    differing = {
        key
        for key in set(ranks[0]) | set(ranks[1])
        if ranks[0].get(key) != ranks[1].get(key)
    }
    # An interface one rank takes and its peer does not is not an error: NCCL
    # falls back, and the debug log then disagrees with the network evidence
    # the probe is asked to verify.
    assert differing == {"NCCL_DEBUG_FILE"}, ranks
    assert ranks[0]["NCCL_SOCKET_IFNAME"] == ranks[1]["NCCL_SOCKET_IFNAME"]
    assert ranks[0]["NCCL_IB_DISABLE"] == ranks[1]["NCCL_IB_DISABLE"] == "1"
    assert ranks[0]["PYTHONPATH"] == ranks[1]["PYTHONPATH"] == STAGING


def test_each_rank_writes_the_debug_log_the_probe_reads() -> None:
    for job in _plan()["jobs"]:
        environment = _environment(job["command"])
        assert environment["NCCL_DEBUG_FILE"] == _flag(job["command"], "--network-log")
    # Two ranks writing one file would interleave, and the probe would verify
    # the route from a log that also holds its peer's view.
    debug_logs = {
        _environment(job["command"])["NCCL_DEBUG_FILE"] for job in _plan()["jobs"]
    }
    assert len(debug_logs) == 2


def test_only_the_peer_rank_crosses_ssh() -> None:
    jobs = _by_name(_plan())

    launch = jobs["rank-0-launch-host"]["command"]
    peer = jobs["rank-1-peer"]["command"]
    assert launch[0] == "env"
    assert peer[0] == "ssh"
    assert "peer-node" in peer
    assert "-o" in peer and "BatchMode=yes" in peer
    assert "StrictHostKeyChecking=yes" in peer
    assert jobs["rank-1-peer"]["cleanup_command"][0] == "ssh"
    assert "pkill" not in " ".join(jobs["rank-1-peer"]["cleanup_command"])


def test_the_two_ranks_run_the_same_work_against_the_same_rendezvous() -> None:
    jobs = _by_name(_plan())
    launch = jobs["rank-0-launch-host"]["command"]
    peer = jobs["rank-1-peer"]["command"]
    peer = peer[peer.index("env") :]

    for flag in ("--nnodes", "--nproc-per-node", "--master-addr", "--master-port"):
        assert _flag(launch, flag) == _flag(peer, flag), flag
    # One workload, one probe, one interpreter: only the rank, and the files
    # that belong to that rank, differ, so the peer cannot drift into running
    # something else.
    probe = str(Path(STAGING) / plan.PROBE)
    assert probe in launch and probe in peer
    assert launch.count(PYTHON) == peer.count(PYTHON) == 1
    assert _flag(launch, "--node-rank") == "0"
    assert _flag(peer, "--node-rank") == "1"


def test_the_staged_tool_is_the_one_the_cleanup_commands_name() -> None:
    # Cleanup runs after the run, on both nodes. The launch host's checkout is
    # not a path the peer has; only the staged tree is on both.
    for job in _plan()["jobs"]:
        cleanup = job["cleanup_command"]
        assert f"{STAGING}/{plan.TOOL}" in cleanup, cleanup
        assert f"{PYTHON}" in cleanup
        assert plan.LAUNCHER_PATTERN not in cleanup


# --- what may cross to the peer ---------------------------------------------


@pytest.mark.parametrize(
    "token",
    ["/nfs/a b/x", "/nfs/x;y", "/nfs/x$y", "/nfs/x`y`", "/nfs/x\ty", "", "/nfs/x&y"],
)
def test_a_remote_token_a_login_shell_would_split_is_refused(token: str) -> None:
    with pytest.raises(plan.RemoteTokenError):
        plan.remote_safe(token)


@pytest.mark.parametrize(
    "token",
    [
        "/nfs/fq-multinode-1234",
        "/opt/conda/envs/flagquantum/bin/python",
        "peer-node",
        "0",
    ],
)
def test_an_ordinary_remote_token_is_accepted(token: str) -> None:
    # A guard that rejects everything passes the rule above while making the
    # lane impossible, so the accepted side is asserted too.
    assert plan.remote_safe(token) == token


def test_a_staging_path_a_login_shell_would_split_stops_the_plan() -> None:
    with pytest.raises(plan.RemoteTokenError):
        _plan(staging="/nfs/fq-multinode-with a space")


# --- what may be emptied, and where evidence may be written ------------------


@pytest.mark.parametrize(
    "staging",
    [
        "/nfs/fq-multinode",
        "/nfs/fq-multinode-99",
        "/nfs/parent/fq-multinode-99",
        "fq-multinode-x",
    ],
)
def test_a_staging_directory_this_lane_owns_is_accepted(staging: str) -> None:
    # The name that has to match is the leaf, because that is the directory
    # `rsync --delete` empties.
    assert plan.checked_staging(staging) == Path(staging)


@pytest.mark.parametrize("staging", ["/", "/nfs", "/nfs/somebody-elses-tree", "/tmp"])
def test_a_staging_directory_this_lane_does_not_own_is_refused(staging: str) -> None:
    # Staging copies with `--delete`. The name is what keeps a mistyped path
    # from being emptied.
    with pytest.raises(plan.StagingError):
        plan.checked_staging(staging)


@pytest.mark.parametrize("output", [STAGING, f"{STAGING}/out", f"{STAGING}/out/deeper"])
def test_an_output_directory_inside_the_staging_tree_is_refused(output: str) -> None:
    # Staging removes it, and nothing reports that: NCCL does not complain
    # about a debug log it cannot open, so the run fails two steps later at the
    # probe reading the missing log.
    with pytest.raises(plan.StagingError):
        plan.checked_output(STAGING, output)


@pytest.mark.parametrize("output", [OUTPUT, "/nfs/elsewhere", f"{STAGING}-x"])
def test_an_output_directory_beside_the_staging_tree_is_accepted(output: str) -> None:
    assert plan.checked_output(STAGING, output) == Path(output)


# --- the rendezvous port ----------------------------------------------------


def test_an_explicit_master_port_is_used_as_given() -> None:
    assert plan.resolve_master_port("127.0.0.1", 29500) == 29500


def test_an_automatic_master_port_is_free_when_it_is_handed_over() -> None:
    # The hosts are shared and 29500 is what every other rank-2 job on them
    # reaches for; a preflight against the fixed port refused a real launch
    # because a neighbour held it.
    chosen = plan.resolve_master_port("127.0.0.1", plan.AUTO_MASTER_PORT)

    assert 0 < chosen < 65536
    with plan.socket.socket(plan.socket.AF_INET, plan.socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", chosen))


# --- the checks that answer a question a timeout would otherwise answer ------


def _answers(mapping: dict[str, tuple[int, str]]):
    """A fake runner keyed by a fragment of the command it is given."""

    def run(argv, timeout):
        text = " ".join(str(item) for item in argv)
        for fragment, (code, out) in mapping.items():
            if fragment in text:
                return subprocess.CompletedProcess(list(argv), code, out, "")
        return subprocess.CompletedProcess(list(argv), 0, "", "")

    return run


def _interface_root(tmp_path: Path, interface: str = "ens22f0") -> Path:
    root = tmp_path / "net"
    (root / interface).mkdir(parents=True)
    return root


def test_preflight_asks_every_question_it_is_meant_to_ask(tmp_path: Path) -> None:
    checks = plan.preflight(
        peer_host="peer-node",
        peer_address="127.0.0.1",
        launch_address="127.0.0.1",
        interface="ens22f0",
        master_port=0,
        python=PYTHON,
        interface_root=_interface_root(tmp_path),
        run=_answers(
            {
                "hostname": (0, "peer-node\n"),
                "nvidia-smi": (0, "0, 4, 81920, 0\n"),
            }
        ),
    )

    # Pinned as a set: a check dropped from the tuple stops being asked, and
    # nothing else would report that.
    assert {check.name for check in checks} == {
        "peer_is_reachable",
        "this_host_is_the_launch_host",
        "launch_address_is_local",
        "launch_host_has_a_device",
        "peer_has_a_device",
        "launch_host_has_the_interface",
        "peer_has_the_interface",
        "peer_has_the_interpreter",
        "rendezvous_port_is_free",
    }
    assert all(check.ok for check in checks), [c for c in checks if not c.ok]


def test_preflight_refuses_when_the_peer_cannot_be_reached(tmp_path: Path) -> None:
    checks = plan.preflight(
        peer_host="peer-node",
        peer_address="127.0.0.1",
        launch_address="127.0.0.1",
        interface="ens22f0",
        master_port=0,
        python=PYTHON,
        interface_root=_interface_root(tmp_path),
        run=_answers({"ssh": (255, "")}),
    )
    failed = {check.name for check in checks if not check.ok}

    assert "peer_is_reachable" in failed
    # Without the peer, this host cannot be shown to be the launch host either,
    # and the run would otherwise start and hang at the rendezvous.
    assert "this_host_is_the_launch_host" in failed


def test_preflight_refuses_when_the_interface_is_missing_on_the_launch_host(
    tmp_path: Path,
) -> None:
    checks = plan.preflight(
        peer_host="peer-node",
        peer_address="127.0.0.1",
        launch_address="127.0.0.1",
        interface="ens22f0",
        master_port=0,
        python=PYTHON,
        interface_root=_interface_root(tmp_path, interface="ens99f9"),
        run=_answers(
            {"hostname": (0, "peer-node\n"), "nvidia-smi": (0, "0, 1, 2, 3\n")}
        ),
    )
    failed = {check.name for check in checks if not check.ok}

    assert failed == {"launch_host_has_the_interface"}


def _staged(tmp_path: Path) -> str:
    """A staging directory with the package the peer is asked about."""
    staging = tmp_path / f"{plan.STAGING_PREFIX}-unit"
    (staging / "flagquantum").mkdir(parents=True)
    (staging / "flagquantum" / "__init__.py").write_text("", encoding="utf-8")
    return str(staging)


def test_the_staging_check_retries_and_reports_how_long_the_peer_took(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    slept: list[float] = []

    def run(argv, timeout):
        calls.append(" ".join(str(item) for item in argv))
        return subprocess.CompletedProcess(
            list(argv), 0 if len(calls) >= 3 else 1, "", ""
        )

    check = plan.staging_check(
        peer_host="peer-node",
        staging=_staged(tmp_path),
        attempts=5,
        interval=2.0,
        run=run,
        sleep=slept.append,
    )
    assert check.ok
    assert "attempt 3" in check.detail
    assert len(calls) == 3
    assert slept == [2.0, 2.0]


def test_the_staging_check_fails_closed_without_raising(tmp_path: Path) -> None:
    def run(argv, timeout):
        return subprocess.CompletedProcess(list(argv), 1, "", "")

    check = plan.staging_check(
        peer_host="peer-node",
        staging=_staged(tmp_path),
        attempts=3,
        interval=0.0,
        run=run,
        sleep=lambda _seconds: None,
    )

    assert not check.ok
    assert "3 attempts" in check.detail


def test_the_staging_check_does_not_ask_a_peer_about_a_path_never_staged(
    tmp_path: Path,
) -> None:
    def run(argv, timeout):  # pragma: no cover - it must not be reached
        raise AssertionError("the peer was asked about a path that is not here")

    check = plan.staging_check(
        peer_host="peer-node",
        staging=str(tmp_path / f"{plan.STAGING_PREFIX}-absent"),
        run=run,
        sleep=lambda _s: None,
    )

    assert not check.ok
    assert "was not staged" in check.detail


def test_the_output_check_refuses_a_directory_it_cannot_write(tmp_path: Path) -> None:
    blocker = tmp_path / "a-file"
    blocker.write_text("not a directory", encoding="utf-8")

    check = plan.output_check(output_directory=blocker / "out")
    assert not check.ok
    assert check.name == "output_directory_is_writable"


def test_the_output_check_accepts_a_directory_it_creates(tmp_path: Path) -> None:
    check = plan.output_check(output_directory=tmp_path / "fresh" / "out")

    assert check.ok
    # Leaves nothing behind: the ranks write here, and a stray file would end
    # up in the artifact.
    assert list((tmp_path / "fresh" / "out").iterdir()) == []


def test_a_refused_preflight_exits_nonzero_and_names_the_checks(capsys) -> None:
    checks = (
        plan.Check("peer_is_reachable", True, "ok"),
        plan.Check("rendezvous_port_is_free", False, "Address already in use"),
    )

    assert plan._fail(checks, "preflight") == 1
    assert "rendezvous_port_is_free" in capsys.readouterr().err


# --- what the lane is allowed to depend on ----------------------------------


@pytest.mark.parametrize(
    "module", ["tools/multinode_launch_plan.py", "tools/run_multinode_watchdog.py"]
)
def test_the_lane_runs_before_anything_is_installed(module: str) -> None:
    """The lane's own tools import nothing outside the standard library.

    This job installs nothing: both ranks import the staged tree through
    `PYTHONPATH`, and these tools run before that. An import of pytest, yaml or
    torch here would work in every lane that installs first and fail in this
    one.
    """
    tree = ast.parse((ROOT / module).read_text(encoding="utf-8"), filename=module)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])

    assert imported, f"{module} parsed to no imports at all"
    assert imported <= sys.stdlib_module_names, sorted(
        imported - sys.stdlib_module_names
    )


# --- the workflow that runs it ----------------------------------------------


def _multinode_job() -> dict:
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document["jobs"]["multinode"]


def test_the_workflow_runs_the_lane_this_module_drives() -> None:
    job = _multinode_job()
    runs = [
        str(step["run"])
        for step in job["steps"]
        if isinstance(step, dict) and "run" in step
    ]
    joined = "\n".join(runs)

    assert f"tools/{Path(plan.__file__).name} --run" in joined
    assert "--staging" in joined and "--report-directory" in joined
    assert any("rm -rf" in run and "STAGING" in run for run in runs), runs


def test_the_multinode_job_is_manual_only() -> None:
    job = _multinode_job()

    # The pair holds a device on each of two shared hosts. A schedule would
    # take them whether or not anyone is watching.
    assert "workflow_dispatch" in str(job["if"])
    assert "schedule" not in str(job["if"])


def test_the_multinode_job_is_pinned_to_the_host_that_can_reach_the_peer() -> None:
    job = _multinode_job()
    labels = job["runs-on"]

    # SSH is provisioned one way. `multinode` is on the launch host only, and
    # the label is the only thing keeping the job off the peer.
    assert "multinode" in labels
    assert "self-hosted" in labels


def test_the_multinode_job_stages_where_both_nodes_can_read_it() -> None:
    staging = _multinode_job()["env"]["STAGING"]

    # A tree on the launch host's own disk passes every other check and then
    # fails at import on one rank, which reads as a code problem.
    assert staging.startswith("/nfs/")
    assert Path(staging.split("${{")[0]).name.startswith(plan.STAGING_PREFIX)

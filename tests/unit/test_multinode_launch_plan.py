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
import shlex
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
CHECKPOINTS = "/nfs/fq-multinode-unit-checkpoints"


def _plan(**overrides):
    kwargs = {
        "peer_host": "peer-node",
        "launch_address": "10.0.0.1",
        "staging": STAGING,
        "output_directory": OUTPUT,
        "checkpoint_directory": CHECKPOINTS,
        "python": PYTHON,
        "source_revision": "deadbeef",
    }
    kwargs.update(overrides)
    return plan.build_plan(**kwargs)


def _by_name(payload) -> dict[str, dict]:
    return {job["name"]: job for job in payload["jobs"]}


def _peer_argv(command: list[str]) -> list[str]:
    """The argument list the peer receives from an `ssh`-wrapped job.

    Reading the wrapped command back with the shell's own rules is the check
    as well as the convenience: it is what proves the peer sees the arguments
    the launch host meant, rather than whatever the login shell made of them.
    """
    return shlex.split(command[-1]) if command[0] == "ssh" else command


def _environment(command: list[str]) -> dict[str, str]:
    """The `env A=1` prefix, as a mapping, on either side of the ssh wrapper."""
    remote = _peer_argv(command)
    assert remote[0] == "env", remote[:1]
    stop = remote.index(PYTHON)
    return dict(token.split("=", 1) for token in remote[1:stop])


def _flag(command: list[str], name: str) -> str:
    remote = _peer_argv(command)
    return remote[remote.index(name) + 1]


# --- the plan the watchdog consumes -----------------------------------------


def test_the_plan_is_two_nodes_and_the_shape_it_was_asked_for() -> None:
    payload = _plan()

    assert payload["node_count"] == plan.NODE_COUNT == 2
    # One rank per node is the shape the recorded evidence was taken on, and a
    # default that changed would silently make every recorded artifact describe
    # a lane that no longer exists.
    assert payload["local_world_size"] == 1
    assert payload["world_size"] == 2
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


def test_a_wider_per_node_width_is_the_same_lane_with_more_ranks_each() -> None:
    """2xN is the same two nodes; only the per-node rank count changes."""

    narrow = _plan()
    wide = _plan(local_world_size=4)

    assert wide["node_count"] == narrow["node_count"] == 2
    assert wide["local_world_size"] == 4
    assert wide["world_size"] == 8
    # Still one command per node, so still two jobs: a wider node is a wider
    # `torchrun`, not more of them.
    assert len(wide["jobs"]) == len(narrow["jobs"])
    for job in wide["jobs"]:
        assert _flag(job["command"], "--nproc-per-node") == "4"
        assert _flag(job["command"], "--nnodes") == "2"
    assert [_flag(job["command"], "--node-rank") for job in wide["jobs"]] == [
        _flag(job["command"], "--node-rank") for job in narrow["jobs"]
    ]


@pytest.mark.parametrize("probe", sorted(plan.PROBES))
def test_no_probe_is_offered_a_width_of_zero_or_less(probe: str) -> None:
    # A node runs at least one rank, whatever the probe; `torchrun` would
    # otherwise start nothing and the lane would report a shape it never ran.
    for width in (0, -1):
        with pytest.raises(SystemExit) as raised:
            plan.checked_local_world_size(width, probe=probe)
        assert "at least one" in str(raised.value)


@pytest.mark.parametrize("width", [3, 5, 6])
def test_a_per_node_width_the_statevector_executor_cannot_shard_is_refused(
    width: int,
) -> None:
    """Three ranks on a node would shard the amplitudes unevenly.

    The planner shards by rank-address bits, so anything but a power of two has
    to be refused before a launch rather than after a run that reported a shape
    the executor had already declined to execute.
    """

    with pytest.raises(SystemExit) as raised:
        plan.checked_local_world_size(width, probe="statevector")

    assert "power of two" in str(raised.value)


@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_a_per_node_width_the_statevector_executor_can_shard_is_accepted(
    width: int,
) -> None:
    assert plan.checked_local_world_size(width, probe="statevector") == width


def test_a_probe_that_can_serve_a_width_the_statevector_cannot_is_allowed_it() -> None:
    """The width rule belongs to the probe, not to the lane.

    A six-wire rank-owned MPS run is at its widest at six ranks, which is three
    per node: a power-of-two rule applied to the lane refused a shape the MPS
    executor serves exactly.
    """

    assert plan.checked_local_world_size(3, probe="mps") == 3
    # Every rank owns at least one site, so a seventh rank has nothing to own.
    with pytest.raises(SystemExit) as raised:
        plan.checked_local_world_size(4, probe="mps")
    assert "every rank owns at least one site" in str(raised.value)


def test_a_tensor_network_width_has_to_divide_the_slice_count() -> None:
    """The slice count is fixed, so only its divisors describe a whole partition."""

    assert plan.checked_local_world_size(1, probe="tn") == 1
    assert plan.checked_local_world_size(2, probe="tn") == 2
    with pytest.raises(SystemExit) as raised:
        plan.checked_local_world_size(3, probe="tn")
    assert "divide it" in str(raised.value)


def _literal(path: str, name: str) -> int:
    """One integer constant, read out of a module without importing it.

    The probes import torch and the launcher may not, so the launcher carries
    the two widths its refusals are written in terms of. This is what keeps the
    two copies from drifting: the numbers are read from the probe sources.
    """
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if name in targets and isinstance(node.value, ast.Constant):
                return int(node.value.value)
    raise AssertionError(f"{name} is not a module-level constant of {path}")


def test_the_widths_the_lane_refuses_on_are_the_probes_own() -> None:
    assert _literal("tools/probe_cuda_multinode_mps.py", "N_WIRES") == plan.MPS_WIRES
    assert (
        _literal("tools/probe_cuda_multinode_tn.py", "EXPECTED_SLICE_COUNT")
        == plan.TN_SLICE_COUNT
    )


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


def test_each_node_writes_the_debug_log_the_probe_reads() -> None:
    for job in _plan()["jobs"]:
        environment = _environment(job["command"])
        assert environment["NCCL_DEBUG_FILE"] == _flag(job["command"], "--network-log")
    # Two nodes writing one file would interleave, and the probe would verify
    # the route from a log that also holds its peer's view. The file is scoped
    # to the node rather than to the rank because one `torchrun` starts all of a
    # node's ranks from one environment; at one rank per node the two are the
    # same file, and the probe reports which of the two it read.
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
    peer = _peer_argv(jobs["rank-1-peer"]["command"])

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
        cleanup = _peer_argv(job["cleanup_command"])
        assert f"{STAGING}/{plan.TOOL}" in cleanup, cleanup
        assert PYTHON in cleanup
        assert plan.LAUNCHER_PATTERN not in cleanup


def test_both_ranks_checkpoint_where_both_ranks_can_read_the_resume_leg() -> None:
    payload = _plan()

    # The probe's second leg resumes from what its first leg wrote, so both
    # ranks have to be handed one directory on a filesystem both nodes mount.
    directories = {
        _flag(job["command"], "--checkpoint-directory") for job in payload["jobs"]
    }
    assert directories == {CHECKPOINTS}
    assert payload["checkpoint_directory"] == CHECKPOINTS
    # Staging copies with `--delete`, so a checkpoint directory inside it would
    # be removed by the next run's staging step rather than restored from.
    assert not Path(CHECKPOINTS).is_relative_to(Path(STAGING))


@pytest.mark.parametrize(
    "directory",
    [
        STAGING,
        f"{STAGING}/checkpoints",
        "/nfs/checkpoints",
        "/tmp/fq-not-a-lane-directory",
    ],
)
def test_a_directory_this_lane_may_not_empty_is_refused(directory: str) -> None:
    # The lane empties both of these before a run, so a mistyped path must not
    # be removable. Inside the staging tree a directory would also be deleted by
    # staging, which is the failure the location check exists to prevent.
    for purpose in ("output", "checkpoint"):
        with pytest.raises(plan.StagingError):
            plan.checked_lane_directory(STAGING, directory, purpose=purpose)


def test_this_lane_owns_the_directories_it_empties() -> None:
    for purpose, directory in (("output", OUTPUT), ("checkpoint", CHECKPOINTS)):
        assert plan.checked_lane_directory(STAGING, directory, purpose=purpose) == Path(
            directory
        )
        # Beside the staged tree, not inside it, and named so that the removal
        # is a decision this lane has already made.
        assert Path(directory).name.startswith(plan.STAGING_PREFIX)


def test_a_run_starts_from_directories_holding_nothing_from_the_last_one(
    tmp_path: Path,
) -> None:
    output = tmp_path / "fq-multinode-unit-out"
    checkpoints = tmp_path / "fq-multinode-unit-checkpoints"
    for directory, name in ((output, "node-0.json"), (checkpoints, "rank-0.pt")):
        directory.mkdir(parents=True)
        (directory / name).write_text("stale", encoding="utf-8")

    for directory in (output, checkpoints):
        plan.clear_lane_directory(directory)

    # A stale rank record is the one that would be published if this run failed
    # before writing its own, and a stale checkpoint is the one the resume leg
    # would restore instead of the checkpoint it wrote.
    for directory in (output, checkpoints):
        assert not directory.exists()


def test_clearing_a_directory_that_was_never_written_needs_no_decision(
    tmp_path: Path,
) -> None:
    # The first run on a host has no directory to empty, which is not an error.
    plan.clear_lane_directory(tmp_path / "fq-multinode-absent")

    assert not (tmp_path / "fq-multinode-absent").exists()


# --- what may cross to the peer ---------------------------------------------


@pytest.mark.parametrize(
    "token",
    [
        "/nfs/a b/x",
        "/nfs/x;y",
        "/nfs/x$y",
        "/nfs/x`y`",
        "/nfs/x\ty",
        "",
        "/nfs/x&y",
        "a'b",
    ],
)
def test_an_argument_a_login_shell_would_split_reaches_the_peer_whole(
    token: str,
) -> None:
    # `ssh host a b c` is joined into one string and re-split by the peer's
    # login shell, so an argument is usable only if it survives that trip as
    # one word. The round-trip is the property, rather than a hand-built list
    # of allowed characters: the staging gate passes a `-c` program, which
    # contains spaces and a semicolon, and a wrapper that refused those made
    # this lane unable to reach its own import check.
    command = plan.ssh_argv("peer-node", ["/usr/bin/python", "-c", token])

    assert shlex.split(command[-1]) == ["/usr/bin/python", "-c", token]
    # ssh, its options, the host, then exactly one forwarded argument, so the
    # peer's shell has one thing to split and cannot see the launch host's own
    # word boundaries.
    assert shlex.split(command[-1])[0] == command[-1].split()[0]
    assert len(command) == 2 + len(plan.SSH_OPTIONS) + 1


def test_an_argument_no_quoting_can_carry_is_still_refused() -> None:
    # A NUL byte cannot be an argv element at all, so there is nothing to quote.
    with pytest.raises(plan.RemoteTokenError):
        plan.remote_safe("a\x00b")


@pytest.mark.parametrize(
    "token",
    [
        "/nfs/fq-multinode-1234",
        "/opt/conda/envs/flagquantum/bin/python",
        "peer-node",
        "0",
    ],
)
def test_an_ordinary_remote_token_is_carried_unchanged(token: str) -> None:
    # An ordinary word in, the same word out: quoting that changed these would
    # rewrite the paths the plan is made of.
    assert shlex.split(plan.remote_safe(token)) == [token]


def test_a_staging_path_a_login_shell_would_split_still_reaches_the_peer() -> None:
    # A path this lane empties with `--delete` and then tells both ranks to run
    # from. If it could not be named to the peer the plan would be unusable,
    # which is a worse outcome than carrying it correctly.
    staging = "/nfs/fq-multinode-with a space"
    peer = _by_name(_plan(staging=staging))["rank-1-peer"]

    assert f"PYTHONPATH={staging}" in _peer_argv(peer["command"])
    assert staging in _peer_argv(peer["cleanup_command"])[1]


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


# --- the tree that is staged -------------------------------------------------


def _recorded(monkeypatch, tmp_path: Path, codes: list[int] | None = None) -> list[str]:
    """A fake runner that records the argument lists the lane hands it."""
    seen: list[str] = []
    remaining = list(codes or [])

    def run(argv, timeout, cwd=None):
        seen.append(" ".join(str(item) for item in argv))
        code = remaining.pop(0) if remaining else 0
        return subprocess.CompletedProcess(list(argv), code, "", "")

    monkeypatch.setattr(plan, "_run", run)
    return seen


def test_the_staged_tree_is_compiled_before_any_rank_starts(
    monkeypatch, tmp_path: Path
) -> None:
    """The copies arrive without `__pycache__`, and every rank starts at once.

    Two ranks of two nodes then compile the same modules and write the same
    `.pyc` entries through atomic renames on the shared filesystem, where the
    loser waits on an inode the winner holds. The lane met that as a
    `job_timeout` with no rank past process-group creation, so the compile is
    part of staging rather than of the run.
    """
    staging = str(tmp_path / f"{plan.STAGING_PREFIX}-stage")
    seen = _recorded(monkeypatch, tmp_path)

    report = plan.stage(source=tmp_path, staging=staging, python=PYTHON)

    assert seen[0].startswith("rsync ")
    assert seen[1] == f"{PYTHON} -m compileall -q {staging}"
    assert report["bytecode_files"] == 0


def test_a_staged_tree_that_cannot_be_compiled_stops_the_lane(
    monkeypatch, tmp_path: Path
) -> None:
    staging = str(tmp_path / f"{plan.STAGING_PREFIX}-stage")
    _recorded(monkeypatch, tmp_path, codes=[0, 1])

    with pytest.raises(RuntimeError) as raised:
        plan.stage(source=tmp_path, staging=staging, python=PYTHON)

    assert "precompiling the staged tree" in str(raised.value)


def test_the_ranks_are_told_not_to_write_bytecode_beside_each_other() -> None:
    # A stale entry is worth rewriting to the rank that found it; it is not
    # worth four ranks doing it at once on a shared mount.
    for job in _plan()["jobs"]:
        assert _environment(job["command"])["PYTHONDONTWRITEBYTECODE"] == "1"


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


def test_the_import_check_accepts_a_tree_inside_the_staging_directory() -> None:
    def run(argv, timeout, cwd=None):
        return subprocess.CompletedProcess(
            list(argv), 0, f"{STAGING}/flagquantum/__init__.py", ""
        )

    check = plan.import_check(node="peer", staging=STAGING, python=PYTHON, run=run)

    assert check.ok
    assert check.name == "peer_imports_the_staged_tree"
    # The environment is what makes it true, so the check has to set it on
    # whichever side of ssh it runs.
    assert check.detail.startswith(STAGING)


def test_the_import_check_runs_a_local_rank_where_the_rank_will_run() -> None:
    """`''` precedes `PYTHONPATH` on `sys.path`, so the directory is the answer.

    A check that inherited the operator's own directory answered about the
    checkout the operator happened to be standing in: run this tool from the
    repository root and the launch-host rank was reported as importing a tree
    the lane never staged, while the rank it was asking about runs from the
    staging directory and reads the staged tree.
    """
    seen: list[object] = []

    def run(argv, timeout, cwd=None):
        seen.append(cwd)
        return subprocess.CompletedProcess(
            list(argv), 0, f"{STAGING}/flagquantum/__init__.py", ""
        )

    plan.import_check(node="launch_host", staging=STAGING, python=PYTHON, run=run)
    plan.import_check(
        node="peer", staging=STAGING, python=PYTHON, peer_host="peer-node", run=run
    )

    # This node's rank is started by the watchdog, which is given a directory.
    assert seen[0] == plan.rank_working_directory(STAGING) == STAGING
    # The peer's is started by the peer's login shell, which `ssh` cannot be
    # given one for, so the check leaves that side to the peer.
    assert seen[1] is None


def test_the_import_check_refuses_the_runners_own_checkout() -> None:
    """The failure this exists for: an editable install shadowing the staging.

    The runner's environment has `flagquantum` installed editable against its
    own workspace, so an interpreter can import a revision this lane never
    staged -- and the artifact would then record a revision neither rank ran.
    """

    def run(argv, timeout, cwd=None):
        return subprocess.CompletedProcess(
            list(argv), 0, "/runner/_work/FlagQuantum/flagquantum/__init__.py", ""
        )

    check = plan.import_check(
        node="launch_host", staging=STAGING, python=PYTHON, run=run
    )

    assert not check.ok
    assert "resolved outside" in check.detail


def test_the_import_check_fails_closed_when_the_interpreter_says_nothing() -> None:
    def run(argv, timeout, cwd=None):
        return subprocess.CompletedProcess(list(argv), 1, "", "ModuleNotFoundError")

    check = plan.import_check(node="peer", staging=STAGING, python=PYTHON, run=run)

    assert not check.ok


def test_the_import_check_program_reaches_the_peer_as_one_argument() -> None:
    """The staging gate's own `-c` program, through the real ssh wrapper.

    The check exists to catch an editable install shadowing the staged tree on
    either node. It carries a program with spaces and a semicolon in it, and a
    wrapper that only accepted shell-ordinary words raised before the check
    ever ran -- so the gate that exists to refuse a wrong revision was the one
    thing that stopped the lane.
    """
    seen: list[list[str]] = []

    def run(argv, timeout, cwd=None):
        seen.append([str(item) for item in argv])
        return subprocess.CompletedProcess(
            list(argv), 0, f"{STAGING}/flagquantum/__init__.py", ""
        )

    check = plan.import_check(
        node="peer",
        staging=STAGING,
        python=PYTHON,
        peer_host="peer-node",
        run=run,
    )

    assert check.ok
    forwarded = _peer_argv(seen[0])
    assert forwarded == ["env", f"PYTHONPATH={STAGING}", PYTHON, *plan.IMPORT_PROBE]
    assert "import flagquantum" in forwarded[-1]


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

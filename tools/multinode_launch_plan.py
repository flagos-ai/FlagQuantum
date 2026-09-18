#!/usr/bin/env python
"""Drive the supervised two-node lane: preflight, stage, plan, run.

The two nodes are not interchangeable. SSH is provisioned one way only: the
launch host reaches the peer, and the peer cannot reach the launch host, so a
job that lands on the peer cannot start the pair at all. What pins the launch
host is a runner label -- a convention rather than a capability -- and the
preflight checks here are what notice when the convention stops holding.

Both ranks run the same staged tree, read from a filesystem both nodes mount,
so the two ranks cannot disagree about which revision they are evidence about.
`tools/run_multinode_watchdog.py` runs them as one unit, so a launcher that
dies on one node does not leave the other waiting for a rendezvous that will
never arrive.

It cannot import a sibling tool, because a script run as `python tools/x.py`
has `tools/` on `sys.path` rather than the repository root.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

TOOL = "tools/multinode_launch_plan.py"
PROBE = "tools/probe_cuda_multinode_statevector.py"
WATCHDOG = "tools/run_multinode_watchdog.py"

# Two ranks, one per node. The probe refuses anything else.
NODE_COUNT = 2
LOCAL_WORLD_SIZE = 1
# `--master-port 0` picks a port that is free at plan time. The hosts are
# shared, and 29500 is the default every other rank-2 job on them reaches for:
# a preflight against the fixed port refused a launch because a neighbour
# already held it. An explicit port is still accepted, because a recorded run
# is easier to reproduce with one.
AUTO_MASTER_PORT = 0
DEFAULT_MASTER_PORT = AUTO_MASTER_PORT

# `--staging` is copied into with `--delete`, so it has to be a name this tool
# is allowed to empty. Nothing else stops a mistyped path from being wiped.
STAGING_PREFIX = "fq-multinode"

# The route the recorded two-node artifact was taken on
# (`artifacts/cuda_multinode_statevector_a800_jp171_jp172_20260907.json`): the
# management interface with InfiniBand disabled, so the NCCL debug log can be
# checked for both the socket transport and this interface name. The hosts do
# carry RoCE devices; using them is a separate change with its own evidence.
DEFAULT_INTERFACE = "ens22f0"

# Passed to `pkill -f`. Long enough to name the launcher and nothing else.
LAUNCHER_PATTERN = "probe_cuda_multinode_statevector"

SSH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=15",
    "-o",
    "StrictHostKeyChecking=yes",
)

# What the staged copy does not need, and what must never be copied over a
# staging directory that a previous run left behind.
STAGING_EXCLUDES = (
    ".git",
    "*.egg-info",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "hardware-run",
    ".venv",
)

# Without `nounits` the answer reads `4 MiB` where a parser expects `4`, which
# parses as no devices at all -- the same answer an idle host gives.
DEVICE_QUERY = "index,memory.used,memory.total,utilization.gpu"
UNITLESS_FORMAT = "--format=csv,noheader,nounits"

# `ssh host a b c` is joined into one string and re-split by the peer's login
# shell, so a token carrying whitespace arrives as several arguments and a
# token carrying a metacharacter is interpreted rather than passed.
_UNSAFE_REMOTE_TOKEN = re.compile(r"[\s'\"`$&|;<>()*?\\\[\]{}~#!\x00-\x1f]")


class RemoteTokenError(ValueError):
    """A token that cannot survive the trip to the peer as one argument."""


class StagingError(ValueError):
    """A staging directory this tool refuses to copy into."""


def remote_safe(token: str) -> str:
    if not token or _UNSAFE_REMOTE_TOKEN.search(token):
        raise RemoteTokenError(
            f"{token!r} would not reach the peer as a single argument; the "
            "peer's login shell re-splits the command"
        )
    return token


def checked_staging(staging: str | Path) -> Path:
    """Refuse a staging directory that is not ours to empty."""
    path = Path(staging)
    if not path.name.startswith(STAGING_PREFIX):
        raise StagingError(
            f"{path} is copied into with `--delete`, and only a directory named "
            f"{STAGING_PREFIX}* is treated as this lane's own"
        )
    return path


def checked_output(staging: str | Path, output_directory: str | Path) -> Path:
    """Refuse an output directory inside the staging tree.

    Staging copies with `--delete`, so an output directory created before it
    is removed by it. Nothing then reports the removal: NCCL does not complain
    about a `NCCL_DEBUG_FILE` it cannot open, and the run fails later at the
    probe reading the log it was asked to verify.
    """
    output = Path(output_directory)
    staging_path = Path(staging)
    if output == staging_path or staging_path in output.parents:
        raise StagingError(
            f"{output} is inside {staging_path}, which staging copies with "
            "`--delete`; the evidence would be removed before it was written"
        )
    return output


@dataclass(frozen=True)
class Check:
    """One preflight finding. `detail` is the evidence either way."""

    name: str
    ok: bool
    detail: str


def ssh_argv(peer_host: str, remote: Sequence[str]) -> list[str]:
    return ["ssh", *SSH_OPTIONS, peer_host, *(remote_safe(item) for item in remote)]


def shared_environment(*, staging: str, interface: str) -> dict[str, str]:
    """The variables both ranks need, and both ranks are given.

    Neither rank may take a different interface than its peer: NCCL does not
    fail when it does, it falls back to something slower and the debug log then
    disagrees with the network evidence the probe is asked to verify. So the
    plan spells the same values into both commands rather than relying on one
    rank inheriting an environment the other one cannot.
    """
    return {
        "PYTHONPATH": staging,
        "NCCL_SOCKET_IFNAME": interface,
        "NCCL_IB_DISABLE": "1",
        "NCCL_DEBUG": "INFO",
    }


def rank_environment(
    *,
    node_rank: int,
    staging: str,
    interface: str,
    output_directory: str,
    source_revision: str | None = None,
) -> dict[str, str]:
    """One rank's environment. Only the debug log differs between ranks.

    `NCCL_DEBUG_FILE` has to be per rank -- two ranks writing one file would
    interleave -- and that is why it is here rather than in a workflow `env:`
    block: a value that depends on the node is not something one block can
    express, and a rank that quietly lacked it would fail after NCCL init
    rather than before.
    """
    environment = {
        **shared_environment(staging=staging, interface=interface),
        "NCCL_DEBUG_FILE": str(Path(output_directory) / f"nccl-rank-{node_rank}.log"),
    }
    if source_revision is not None:
        # The probe records the revision it is evidence about. The staged copy
        # carries no `.git`, so it cannot read one for itself.
        environment["FLAGQUANTUM_SOURCE_REVISION"] = source_revision
    return environment


def _env_prefix(environment: dict[str, str]) -> list[str]:
    """`env A=1 B=2` -- an exec, not a shell, so quoting never enters into it.

    `env` without `-i` adds to the environment rather than replacing it, so
    this keeps the runner's PATH while pinning everything this lane decides.
    """
    return ["env", *(f"{key}={value}" for key, value in environment.items())]


def rank_command(
    *,
    node_rank: int,
    python: str,
    staging: str,
    output_directory: str,
    master_address: str,
    master_port: int,
    interface: str,
    source_revision: str | None = None,
) -> list[str]:
    return [
        *_env_prefix(
            rank_environment(
                node_rank=node_rank,
                staging=staging,
                interface=interface,
                output_directory=output_directory,
                source_revision=source_revision,
            )
        ),
        python,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        str(NODE_COUNT),
        "--nproc-per-node",
        str(LOCAL_WORLD_SIZE),
        "--node-rank",
        str(node_rank),
        "--master-addr",
        master_address,
        "--master-port",
        str(master_port),
        # Absolute, so the peer does not depend on where ssh leaves it.
        str(Path(staging) / PROBE),
        "--output",
        str(Path(output_directory) / f"rank-{node_rank}.json"),
        # The probe writes its artifact on rank 0 only, because the artifact
        # carries both ranks' records. Rank 1 keeps the flag so that the file
        # NCCL is told to write and the file the probe would read stay the same
        # path on both nodes.
        "--network-log",
        str(Path(output_directory) / f"nccl-rank-{node_rank}.log"),
    ]


def cleanup_command(
    *, node_rank: int, python: str, staging: str, peer_host: str | None = None
) -> list[str]:
    """End a launcher that the watchdog's process group could not reach.

    On the launch host the watchdog already signalled the process group. On the
    peer it did not: the group it signalled holds the `ssh` client, and the
    remote process outlives its client. Both commands therefore exist, and both
    report rather than assert, so that a node with nothing left to kill does not
    read as a cleanup failure.

    The tool is named at its staged path, which both nodes see, rather than at
    the launch host's checkout, which only the launch host has.
    """
    command = [
        python,
        str(Path(staging) / TOOL),
        "--cleanup",
        "--node-rank",
        str(node_rank),
    ]
    if peer_host is None:
        return command
    return ssh_argv(peer_host, command)


def build_plan(
    *,
    peer_host: str,
    launch_address: str,
    staging: str,
    output_directory: str,
    python: str,
    master_port: int = DEFAULT_MASTER_PORT,
    interface: str = DEFAULT_INTERFACE,
    source_revision: str | None = None,
) -> dict[str, Any]:
    """Write the watchdog's config: two ranks, one node each."""
    ranks = {
        node_rank: rank_command(
            node_rank=node_rank,
            python=python,
            staging=staging,
            output_directory=output_directory,
            master_address=launch_address,
            master_port=master_port,
            interface=interface,
            source_revision=source_revision,
        )
        for node_rank in range(NODE_COUNT)
    }
    return {
        "schema": "flagquantum.multinode_launch_plan.v1",
        "node_count": NODE_COUNT,
        "local_world_size": LOCAL_WORLD_SIZE,
        "launch_address": launch_address,
        "peer_host": peer_host,
        "master_port": master_port,
        "interface": interface,
        "staging_directory": staging,
        "output_directory": output_directory,
        "jobs": [
            {
                "name": "rank-0-launch-host",
                "command": ranks[0],
                "cwd": staging,
                "cleanup_command": cleanup_command(
                    node_rank=0, python=python, staging=staging
                ),
            },
            {
                "name": "rank-1-peer",
                # ssh does not forward the launch host's environment, so the
                # peer's rank carries its own copy of the same values.
                "command": ssh_argv(peer_host, ranks[1]),
                "cwd": staging,
                "cleanup_command": cleanup_command(
                    node_rank=1, python=python, staging=staging, peer_host=peer_host
                ),
            },
        ],
    }


def _run(argv: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(item) for item in argv],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _first_line(text: str) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[0].strip() if lines else ""


def local_address_towards(peer_address: str, port: int = 22) -> str:
    """The local address the peer is reachable from, without sending a packet."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.settimeout(5.0)
        probe.connect((peer_address, port))
        return str(probe.getsockname()[0])


def free_port(address: str) -> int:
    """A port that was free a moment ago, for a rendezvous that starts now.

    The window between this and the ranks binding it is small and unavoidable.
    It is still better than a fixed port on a shared host, where a neighbour
    holding it is a matter of when and not whether; and the preflight below
    checks this port again before anything is launched.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((address, 0))
        return int(probe.getsockname()[1])


def resolve_master_port(address: str, requested: int) -> int:
    return requested if requested > 0 else free_port(address)


def _device_answer(
    argv: Sequence[str],
    timeout: float,
    run: Callable[[Sequence[str], float], subprocess.CompletedProcess[str]] = _run,
) -> tuple[bool, str]:
    try:
        completed = run(argv, timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, f"{type(error).__name__}: {error}"
    if completed.returncode != 0:
        answer = _first_line(completed.stderr) or "no output"
        return False, f"exit {completed.returncode}: {answer}"
    rows = [row for row in completed.stdout.strip().splitlines() if row.strip()]
    return bool(rows), f"{len(rows)} device(s): {completed.stdout.strip()!r}"


def preflight(
    *,
    peer_host: str,
    peer_address: str,
    interface: str,
    master_port: int,
    python: str,
    launch_address: str | None = None,
    timeout: float = 30.0,
    interface_root: Path = Path("/sys/class/net"),
    run: Callable[[Sequence[str], float], subprocess.CompletedProcess[str]] = _run,
) -> tuple[Check, ...]:
    """Establish what the plan needs to be true, before anything is launched.

    Every check answers a question that would otherwise be answered by a
    timeout: a rendezvous that never completes looks the same whether the peer
    is unreachable, has no device, or is listening on another interface.
    """
    checks: list[Check] = []
    device_query = ("nvidia-smi", f"--query-gpu={DEVICE_QUERY}", UNITLESS_FORMAT)

    def remote(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return run(ssh_argv(peer_host, argv), timeout)

    def safely(argv: Sequence[str]) -> tuple[bool, str]:
        try:
            completed = remote(argv)
        except (OSError, subprocess.TimeoutExpired) as error:
            return False, f"{type(error).__name__}: {error}"
        return completed.returncode == 0, (
            _first_line(completed.stderr) or f"exit {completed.returncode}"
        )

    # The launch host is the one that can reach the peer. If this host can ssh
    # to the peer it is not the peer, and if it cannot, nothing below matters.
    peer_name = ""
    try:
        completed = remote(["hostname"])
        if completed.returncode == 0:
            peer_name = _first_line(completed.stdout)
        reached = completed.returncode == 0 and bool(peer_name)
        detail = (
            f"ssh {peer_host} -> {peer_name}"
            if reached
            else f"exit {completed.returncode}: {_first_line(completed.stderr)}"
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        reached, detail = False, f"{type(error).__name__}: {error}"
    checks.append(Check("peer_is_reachable", reached, detail))

    local_name = platform.node()
    checks.append(
        Check(
            "this_host_is_the_launch_host",
            reached and local_name != peer_name,
            f"local={local_name!r} peer={peer_name!r}",
        )
    )

    # The address the peer will rendezvous against has to be one this host
    # answers on, and it has to be the one facing the peer.
    if launch_address is None:
        try:
            launch_address = local_address_towards(peer_address)
            local, detail = (
                True,
                f"route to {peer_address} leaves from {launch_address}",
            )
        except OSError as error:
            launch_address = ""
            local, detail = False, f"{type(error).__name__}: {error}"
    else:
        try:
            routed = local_address_towards(peer_address)
            local = routed == launch_address
            detail = f"route to {peer_address} leaves from {routed}"
        except OSError as error:
            local, detail = False, f"{type(error).__name__}: {error}"
    checks.append(Check("launch_address_is_local", local, detail))

    ok, detail = _device_answer(device_query, timeout, run)
    checks.append(Check("launch_host_has_a_device", ok, detail))

    ok, detail = _device_answer(ssh_argv(peer_host, device_query), timeout, run)
    checks.append(Check("peer_has_a_device", ok, detail))

    # An interface name that exists on one node only is not an error at launch
    # time: NCCL falls back, and the debug log then disagrees with the network
    # evidence the probe is asked to verify.
    checks.append(
        Check(
            "launch_host_has_the_interface",
            (interface_root / interface).is_dir(),
            f"{interface_root / interface}",
        )
    )
    ok, detail = safely(["test", "-d", f"/sys/class/net/{interface}"])
    checks.append(Check("peer_has_the_interface", ok, f"on {peer_host}: {detail}"))

    # The peer's interpreter is a path, not a lookup: ssh starts a shell whose
    # PATH is not the runner's, so the plan names the interpreter and this is
    # where that name is checked.
    ok, detail = safely(["test", "-x", python])
    checks.append(
        Check("peer_has_the_interpreter", ok, f"{python} on {peer_host}: {detail}")
    )

    # A rendezvous port already in use fails at init, after both ranks are up.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind((launch_address, master_port))
        free, detail = True, f"{launch_address}:{master_port} binds"
    except OSError as error:
        free, detail = False, f"{type(error).__name__}: {error}"
    checks.append(Check("rendezvous_port_is_free", free, detail))

    return tuple(checks)


def staging_check(
    *,
    peer_host: str,
    staging: str,
    timeout: float = 30.0,
    attempts: int = 15,
    interval: float = 2.0,
    run: Callable[[Sequence[str], float], subprocess.CompletedProcess[str]] = _run,
    sleep: Callable[[float], None] = time.sleep,
) -> Check:
    """The peer has to read the staged tree, not a path that only exists here.

    A job that stages onto local disk passes every other check and then fails
    at import on one rank, which reads as a code problem rather than a
    filesystem one.

    It retries, because a shared NFS client caches what it was told: the peer
    having recently looked up this path and been told it did not exist is
    enough to answer "no" for a while after rsync has finished. The first run
    of this lane failed this check and the next run passed it unchanged, which
    is a flaky gate rather than a check. It still fails closed, and it reports
    how long the peer took.
    """
    staged_package = Path(staging) / "flagquantum" / "__init__.py"
    remote = ["test", "-f", str(staged_package)]
    if not staged_package.is_file():
        return Check(
            "peer_reads_the_staged_tree",
            False,
            f"{staged_package} was not staged",
        )
    started = time.monotonic()
    detail = "not attempted"
    for attempt in range(1, max(1, attempts) + 1):
        try:
            completed = run(ssh_argv(peer_host, remote), timeout)
            ok = completed.returncode == 0
            detail = f"exit {completed.returncode}"
        except (OSError, subprocess.TimeoutExpired) as error:
            ok, detail = False, f"{type(error).__name__}: {error}"
        if ok:
            return Check(
                "peer_reads_the_staged_tree",
                True,
                f"{peer_host} read {staged_package} after "
                f"{time.monotonic() - started:.1f}s (attempt {attempt})",
            )
        if attempt < attempts:
            sleep(interval)
    return Check(
        "peer_reads_the_staged_tree",
        False,
        f"{peer_host} did not see {staged_package} within "
        f"{time.monotonic() - started:.1f}s ({attempts} attempts, last {detail})",
    )


def stage(*, source: str | Path, staging: str | Path, python: str) -> dict[str, Any]:
    """Copy the tree both ranks will run onto the filesystem they share."""
    destination = checked_staging(staging)
    destination.mkdir(parents=True, exist_ok=True)
    command = ["rsync", "-a", "--delete"]
    for pattern in STAGING_EXCLUDES:
        command.extend(["--exclude", pattern])
    command.extend([f"{Path(source).resolve()}/", f"{destination}/"])
    completed = _run(command, 900.0)
    if completed.returncode != 0:
        raise RuntimeError(
            f"staging failed with exit {completed.returncode}: "
            f"{_first_line(completed.stderr)}"
        )
    return {
        "schema": "flagquantum.multinode_staging.v1",
        "source": str(Path(source).resolve()),
        "destination": str(destination),
        "excluded": list(STAGING_EXCLUDES),
        "python": python,
        "files": sum(1 for _ in destination.rglob("*.py")),
    }


def output_check(*, output_directory: str | Path) -> Check:
    """The ranks have to be able to write where they are told to write.

    NCCL does not report a `NCCL_DEBUG_FILE` it cannot open. It proceeds, and
    the failure surfaces later as the probe unable to read the network evidence
    it was asked to verify -- two steps away from the directory that was
    missing.
    """
    path = Path(output_directory)
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".multinode-writable"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as error:
        return Check("output_directory_is_writable", False, f"{path}: {error}")
    return Check("output_directory_is_writable", True, str(path))


def cleanup(*, node_rank: int, pattern: str = LAUNCHER_PATTERN) -> dict[str, Any]:
    """End anything still named `pattern`. Reports; does not assert."""
    completed = _run(["pkill", "-f", pattern], 30.0)
    return {
        "schema": "flagquantum.multinode_cleanup.v1",
        "node_rank": node_rank,
        "pattern": pattern,
        # 0 killed something, 1 found nothing. Neither is a failure: the
        # watchdog signals the launch host's own process group first.
        "matched": completed.returncode == 0,
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peer-host", default="jp-a800-171")
    parser.add_argument("--launch-address", default=None)
    parser.add_argument("--staging", default=None)
    parser.add_argument("--source", default=str(ROOT))
    parser.add_argument(
        "--output-directory",
        default=None,
        help=(
            "where the ranks write, on a filesystem both nodes mount; "
            "defaults to <staging>-out, beside the staged tree rather than "
            "inside it, because staging copies with `--delete`"
        ),
    )
    parser.add_argument("--report-directory", default="hardware-run")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--master-port", type=int, default=DEFAULT_MASTER_PORT)
    parser.add_argument("--interface", default=DEFAULT_INTERFACE)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--preflight-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--run-timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="check what the plan needs to be true and exit without writing one",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="preflight, stage, plan and supervise the two ranks as one unit",
    )
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--node-rank", type=int, default=0)
    return parser


def _peer_address(peer_host: str, timeout: float) -> str:
    """Ask the peer for its own address rather than keeping a second copy.

    A copy of the peer's address that drifted from the ssh alias would be
    wrong in the one place that matters: the address every check is run
    against.
    """
    try:
        completed = _run(ssh_argv(peer_host, ["hostname", "-I"]), timeout)
    except (OSError, subprocess.TimeoutExpired):
        return peer_host
    addresses = _first_line(completed.stdout).split()
    return addresses[0] if addresses else peer_host


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")


def _report_checks(checks: Sequence[Check]) -> None:
    for check in checks:
        print(f"{'ok  ' if check.ok else 'FAIL'} {check.name}: {check.detail}")


def _fail(checks: Sequence[Check], stage_name: str) -> int:
    failed = [check.name for check in checks if not check.ok]
    print(f"{stage_name} refused the launch: " + ", ".join(failed), file=sys.stderr)
    return 1


def _governed_run(args: argparse.Namespace) -> int:
    """The whole lane, in the order the checks require."""
    report = Path(args.report_directory)
    report.mkdir(parents=True, exist_ok=True)
    staging = checked_staging(args.staging)
    output_directory = checked_output(
        staging, args.output_directory or f"{staging}-out"
    )
    peer_address = _peer_address(args.peer_host, args.preflight_timeout_seconds)
    # The address the checks pass against is the one the plan must use, so it
    # is settled before the checks rather than after them.
    launch_address = args.launch_address or local_address_towards(peer_address)
    master_port = resolve_master_port(launch_address, args.master_port)
    print(f"rendezvous: {launch_address}:{master_port}")

    checks = preflight(
        peer_host=args.peer_host,
        peer_address=peer_address,
        interface=args.interface,
        master_port=master_port,
        python=args.python,
        launch_address=launch_address,
        timeout=args.preflight_timeout_seconds,
    )
    _report_checks(checks)
    if not all(check.ok for check in checks):
        _write(report / "preflight.json", [check.__dict__ for check in checks])
        return _fail(checks, "preflight")

    print(json.dumps(stage(source=args.source, staging=staging, python=args.python)))

    # Both of these happen after staging, which copies with `--delete` and
    # would remove a directory created before it.
    after_staging = [
        output_check(output_directory=output_directory),
        staging_check(peer_host=args.peer_host, staging=str(staging)),
    ]
    _report_checks(after_staging)
    if not all(check.ok for check in after_staging):
        return _fail(after_staging, "staging")

    plan = build_plan(
        peer_host=args.peer_host,
        launch_address=launch_address,
        staging=str(staging),
        output_directory=str(output_directory),
        python=args.python,
        master_port=master_port,
        interface=args.interface,
        source_revision=args.source_revision,
    )
    _write(report / "plan.json", plan)
    _write(report / "preflight.json", [check.__dict__ for check in checks])
    print(f"plan: {report / 'plan.json'}")

    watchdog_logs = report / "watchdog-logs"
    completed = _run(
        [
            args.python,
            str(ROOT / WATCHDOG),
            "--config",
            str(report / "plan.json"),
            "--output-dir",
            str(watchdog_logs),
            "--timeout-seconds",
            str(args.run_timeout_seconds),
        ],
        args.run_timeout_seconds + 120.0,
    )
    # The watchdog writes this file itself. Reading the file rather than the
    # exit code means a run that failed to report is distinguishable from a run
    # that reported a failure.
    summary_path = watchdog_logs / "watchdog.json"
    if not summary_path.is_file():
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        print(f"the watchdog wrote no summary to {summary_path}", file=sys.stderr)
        return 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _write(report / "watchdog.json", summary)
    print(json.dumps(summary, sort_keys=True))

    # The ranks write over the shared filesystem; the artifact is assembled
    # here so that one upload carries the evidence and not only the verdict.
    for produced in sorted(output_directory.glob("*")):
        if produced.is_file():
            shutil.copy2(produced, report / produced.name)
    return 0 if summary.get("status") == "passed" else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.cleanup:
        print(json.dumps(cleanup(node_rank=args.node_rank), sort_keys=True))
        return 0

    if args.run:
        if args.staging is None:
            _parser().error("--run needs --staging")
        return _governed_run(args)

    if args.preflight:
        peer_address = _peer_address(args.peer_host, args.preflight_timeout_seconds)
        launch_address = args.launch_address or local_address_towards(peer_address)
        master_port = resolve_master_port(launch_address, args.master_port)
        print(f"rendezvous: {launch_address}:{master_port}")
        checks = preflight(
            peer_host=args.peer_host,
            peer_address=peer_address,
            interface=args.interface,
            master_port=master_port,
            python=args.python,
            launch_address=launch_address,
            timeout=args.preflight_timeout_seconds,
        )
        _report_checks(checks)
        if not all(check.ok for check in checks):
            return _fail(checks, "preflight")
        print(json.dumps({"status": "passed"}, sort_keys=True))
        return 0

    missing = [
        name
        for name in ("staging", "output_directory", "launch_address")
        if getattr(args, name) is None
    ]
    if missing:
        _parser().error(
            "planning needs "
            + ", ".join("--" + name.replace("_", "-") for name in missing)
        )
    plan = build_plan(
        peer_host=args.peer_host,
        launch_address=args.launch_address,
        staging=args.staging,
        output_directory=args.output_directory,
        python=args.python,
        master_port=args.master_port,
        interface=args.interface,
        source_revision=args.source_revision,
    )
    encoded = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

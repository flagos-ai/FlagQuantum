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
import shlex
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
WATCHDOG = "tools/run_multinode_watchdog.py"

# One two-node workload per probe. Each owns its own circuit, its own evidence
# schema and its own capability entry; they share the plan, the preflight and
# the watchdog, because those are properties of the pair rather than of what is
# run on it.
PROBES = {
    "statevector": "tools/probe_cuda_multinode_statevector.py",
    "mps": "tools/probe_cuda_multinode_mps.py",
    "tn": "tools/probe_cuda_multinode_tn.py",
}
DEFAULT_PROBE = "statevector"
# The lane the runbook and the launch-plan tests describe.
PROBE = PROBES[DEFAULT_PROBE]

# Two nodes. `LOCAL_WORLD_SIZE` is the default number of ranks each node runs;
# `--local-world-size` overrides it, so the same lane covers a 2x1 pair and a
# 2xN one. One rank per node is what the recorded evidence was taken on, and a
# default that changes the shape of an existing lane would invalidate it, so it
# stays the default. The probes accept any power-of-two value.
NODE_COUNT = 2
LOCAL_WORLD_SIZE = 1

# The two widths the probe workloads are built on, as the launcher has to know
# them to refuse a shape before a launch: the MPS probe's circuit has six wires
# and the tensor-network probe's has a fixed four-way slice count. Each probe
# carries the same number and enforces it again before it builds a process
# group; `tests/unit/test_multinode_launch_plan.py` reads both and fails if they
# drift apart.
MPS_WIRES = 6
TN_SLICE_COUNT = 4
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

# The route the recorded two-node artifacts were taken on: the management
# interface with InfiniBand disabled, so the NCCL debug log can be checked for
# both the socket transport and this interface name. It stays the default
# because a lane's default decides what a run means, and the recorded evidence
# was taken on this one.
#
# The hosts also carry RoCE devices, and `--transport rdma` is the lane that
# uses them: it enables InfiniBand, keeps this interface as NCCL's out-of-band
# bootstrap, and makes the preflight refuse a node whose fabric NCCL cannot
# reach. Neither transport is inferred -- the plan spells it into both ranks'
# environment, and the probe records the route the debug log shows, so a run
# reported as one transport and observed on the other is refused rather than
# recorded.
DEFAULT_INTERFACE = "ens22f0"

# The two transports a lane may be asked for. `socket` means InfiniBand is
# disabled rather than absent: a host with a fabric that was not asked for is
# not the same as a host without one, and the probe records which it saw.
TRANSPORT_SOCKET = "socket"
TRANSPORT_RDMA = "rdma"
DEFAULT_TRANSPORT = TRANSPORT_SOCKET

# Where a node's fabric devices are listed. Read instead of shelling out to
# `ibstat`, which is not installed on every host that has the devices.
INFINIBAND_DEVICES = Path("/sys/class/infiniband")

# Passed to `pkill -f`. Long enough to name a two-node probe launcher and
# nothing else: it is the stem both probes share, and `cleanup` runs on a node
# this lane gives one probe at a time, so it cannot reach an unrelated process.
LAUNCHER_PATTERN = "probe_cuda_multinode"

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

# The variable each rank is given to pin the devices it may use. It is the only
# device-selection mechanism the CUDA runtime reads, and it is per-process, so
# it has to be written into the command rather than inferred from the host: two
# ranks on two nodes share one host-level environment only by convention, and
# ssh forwards none of it.
DEVICE_ENVIRONMENT_VARIABLE = "CUDA_VISIBLE_DEVICES"

# `nvidia-smi -i` addresses physical devices by index and ignores the masking
# variable, which is what makes it usable as a witness for what a consumer of
# that index will really get. An index the host does not have answers with a
# non-zero exit, so absence and idleness are distinguishable answers.
DEVICE_APPLICATION_QUERY = "--query-compute-apps=pid"


# `ssh host a b c` is joined into one string and re-split by the peer's login
# shell, so a token carrying whitespace would arrive as several arguments and a
# token carrying a metacharacter would be interpreted rather than passed. Every
# token is therefore quoted into exactly one shell word, which is what lets a
# `-c` program or a path with a space in it reach the peer intact. A NUL byte is
# the one thing quoting cannot carry, because it cannot be an argument at all.
class RemoteTokenError(ValueError):
    """An argument that could not reach the peer even after quoting."""


class StagingError(ValueError):
    """A staging directory this tool refuses to copy into."""


def remote_safe(token: str) -> str:
    """One shell word the peer's login shell reads back as exactly `token`."""
    if "\x00" in token:
        raise RemoteTokenError(
            f"{token!r} contains a NUL byte, which cannot be passed to the "
            "peer as an argument at all"
        )
    return shlex.quote(token)


def peer_command(remote: Sequence[str]) -> str:
    """The single argument `ssh` forwards, which the peer re-splits exactly once.

    Reading it back with `shlex.split` reproduces `remote` exactly; that
    round-trip is what the tests assert, because it is the property the peer
    depends on and the property a hand-built command string loses.
    """
    return " ".join(remote_safe(item) for item in remote)


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


def clear_lane_directory(directory: Path) -> None:
    """Empty a directory this lane owns, so the run starts from nothing.

    The lane publishes whatever the output directory holds and the probe
    resumes from the first checkpoint it finds, so a run that failed before
    writing would otherwise be reported as the previous run's evidence and a
    resumed leg could restore a checkpoint it did not write. It is created
    rather than left absent because the ranks are told to write into it.
    """

    if directory.exists():
        shutil.rmtree(directory)


def checked_lane_directory(
    staging: str | Path, directory: str | Path, *, purpose: str
) -> Path:
    """Refuse a directory this lane is not allowed to empty.

    Staging is copied into with `--delete` and the output and checkpoint
    directories are emptied before the runs that write them, so all three have
    to be names this lane owns. The name is what makes the removal safe, and
    nothing about the contents can substitute for it.

    The output case is not merely tidiness: the lane copies whatever is in the
    output directory into the report, so a `node-0.json` left by an earlier run
    would be published as this run's evidence if this one failed before writing
    its own. The checkpoint case is the same hazard one step further in: the
    probe resumes from the checkpoints its own first leg wrote, and a file left
    behind would be restored instead.
    """
    path = checked_output(staging, directory)
    if not path.name.startswith(STAGING_PREFIX):
        raise StagingError(
            f"{path} is this lane's {purpose} directory, which it empties, and "
            f"only a directory named {STAGING_PREFIX}* is treated as this "
            "lane's own"
        )
    return path


@dataclass(frozen=True)
class Check:
    """One preflight finding. `detail` is the evidence either way."""

    name: str
    ok: bool
    detail: str


def ssh_argv(peer_host: str, remote: Sequence[str]) -> list[str]:
    return ["ssh", *SSH_OPTIONS, peer_host, peer_command(remote)]


def shared_environment(
    *,
    staging: str,
    interface: str,
    transport: str = DEFAULT_TRANSPORT,
    infiniband_hcas: str | None = None,
    visible_devices: str | None = None,
) -> dict[str, str]:
    """The variables both ranks need, and both ranks are given.

    Neither rank may take a different interface than its peer: NCCL does not
    fail when it does, it falls back to something slower and the debug log then
    disagrees with the network evidence the probe is asked to verify. So the
    plan spells the same values into both commands rather than relying on one
    rank inheriting an environment the other one cannot.

    `NCCL_IB_DISABLE` is always written, never left to the host's default. An
    unset value means the host decides, and a lane whose transport depends on
    which host it landed on is not a lane whose evidence can be compared with
    another run's.
    """

    environment = {
        "PYTHONPATH": staging,
        "NCCL_SOCKET_IFNAME": interface,
        "NCCL_IB_DISABLE": "1" if transport == TRANSPORT_SOCKET else "0",
        "NCCL_DEBUG": "INFO",
        # Every rank reads the one staged tree over the shared filesystem, and
        # they all start together. A rank that compiled a module would write its
        # `__pycache__` entry next to everyone else's, so the ranks are told not
        # to: `stage` compiles the tree once, before any of them starts.
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if infiniband_hcas is not None:
        # Pinned only when asked for: NCCL's own discovery skips a port that is
        # down, and a hard-coded rail list would be a fact about these two hosts
        # written into a tool that runs on others.
        environment["NCCL_IB_HCA"] = infiniband_hcas
    if visible_devices is not None:
        # Named rather than left to `local_rank`, which resolves to the host's
        # index of that number: on a node whose index 0 belongs to something
        # else, a lane that named nothing would contend for it. Both ranks are
        # given the same list, so a rank resolves its device by local rank into
        # a list that is identical on both nodes.
        environment[DEVICE_ENVIRONMENT_VARIABLE] = visible_devices
    return environment


def rank_environment(
    *,
    node_rank: int,
    staging: str,
    interface: str,
    output_directory: str,
    source_revision: str | None = None,
    transport: str = DEFAULT_TRANSPORT,
    infiniband_hcas: str | None = None,
    visible_devices: str | None = None,
) -> dict[str, str]:
    """One node's environment for every rank that node runs.

    `NCCL_DEBUG_FILE` is scoped to the node, not to the rank. One `torchrun`
    per node starts all of its ranks from one environment, so a per-rank value
    is not something this plan can express: at `--local-world-size 1` the node's
    log is exactly one rank's log, and above that the node's ranks share it.
    The probe reports which of the two it read, so the difference is recorded
    rather than assumed.

    It lives here rather than in a workflow `env:` block because a value that
    depends on the node is not something one block can express, and a node that
    quietly lacked it would fail after NCCL init rather than before.
    """
    environment = {
        **shared_environment(
            staging=staging,
            interface=interface,
            transport=transport,
            infiniband_hcas=infiniband_hcas,
            visible_devices=visible_devices,
        ),
        "NCCL_DEBUG_FILE": str(Path(output_directory) / f"nccl-node-{node_rank}.log"),
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
    checkpoint_directory: str,
    master_address: str,
    master_port: int,
    interface: str,
    probe: str = PROBE,
    source_revision: str | None = None,
    local_world_size: int = LOCAL_WORLD_SIZE,
    transport: str = DEFAULT_TRANSPORT,
    infiniband_hcas: str | None = None,
    visible_devices: str | None = None,
    measure: bool = False,
) -> list[str]:
    return [
        *_env_prefix(
            rank_environment(
                node_rank=node_rank,
                staging=staging,
                interface=interface,
                output_directory=output_directory,
                source_revision=source_revision,
                transport=transport,
                infiniband_hcas=infiniband_hcas,
                visible_devices=visible_devices,
            )
        ),
        python,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        str(NODE_COUNT),
        "--nproc-per-node",
        str(local_world_size),
        "--node-rank",
        str(node_rank),
        "--master-addr",
        master_address,
        "--master-port",
        str(master_port),
        # Absolute, so the peer does not depend on where ssh leaves it.
        str(Path(staging) / probe),
        "--output",
        # One slot per node, and only the node holding world rank 0 fills it:
        # the artifact carries every rank's record, so a file per rank would be
        # the same evidence written several times over, once per process.
        str(Path(output_directory) / f"node-{node_rank}.json"),
        # Both ranks write their owner-sharded checkpoint here and both read it
        # back on the resume leg, so it has to be a filesystem both nodes mount
        # -- the same one the staged tree came from.
        "--checkpoint-directory",
        str(Path(checkpoint_directory)),
        # The probe writes its artifact on the world's rank 0 only, because the
        # artifact carries every rank's record. Rank 0 of the peer node keeps
        # the flag so that the file NCCL is told to write and the file the probe
        # would read stay the same path on both nodes.
        "--network-log",
        str(Path(output_directory) / f"nccl-node-{node_rank}.log"),
        # Both nodes carry the flag, not only the node that writes the
        # artifact: a lane whose two ranks disagreed about how many times to
        # run the workload would have them enter different numbers of
        # collectives, which reports as a hang rather than as the mismatch.
        *(["--measure"] if measure else []),
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
    checkpoint_directory: str,
    python: str,
    master_port: int = DEFAULT_MASTER_PORT,
    interface: str = DEFAULT_INTERFACE,
    probe: str = PROBE,
    source_revision: str | None = None,
    local_world_size: int = LOCAL_WORLD_SIZE,
    transport: str = DEFAULT_TRANSPORT,
    infiniband_hcas: str | None = None,
    visible_devices: str | None = None,
    measure: bool = False,
) -> dict[str, Any]:
    """Write the watchdog's config: two nodes, `local_world_size` ranks each."""
    visible_devices = checked_visible_devices(
        visible_devices, local_world_size=local_world_size
    )
    ranks = {
        node_rank: rank_command(
            node_rank=node_rank,
            python=python,
            staging=staging,
            output_directory=output_directory,
            checkpoint_directory=checkpoint_directory,
            master_address=launch_address,
            master_port=master_port,
            interface=interface,
            probe=probe,
            source_revision=source_revision,
            local_world_size=local_world_size,
            transport=transport,
            infiniband_hcas=infiniband_hcas,
            visible_devices=visible_devices,
            measure=measure,
        )
        for node_rank in range(NODE_COUNT)
    }
    return {
        "schema": "flagquantum.multinode_launch_plan.v1",
        "node_count": NODE_COUNT,
        "local_world_size": local_world_size,
        "world_size": NODE_COUNT * local_world_size,
        "launch_address": launch_address,
        "peer_host": peer_host,
        "master_port": master_port,
        "interface": interface,
        # Recorded rather than left to be read back out of the commands: the
        # transport is the one property of a plan that decides what a run's
        # route evidence is allowed to say, so a report states it directly.
        "transport": transport,
        "infiniband_hcas": infiniband_hcas,
        # Recorded because it decides which device each rank's measured memory
        # and timings belong to, so a report can state it rather than derive it
        # from a command line that the peer's `env` prefix repeats.
        "visible_devices": visible_devices,
        "measure": measure,
        "probe": probe,
        "staging_directory": staging,
        "output_directory": output_directory,
        "checkpoint_directory": checkpoint_directory,
        "jobs": [
            {
                "name": "rank-0-launch-host",
                "command": ranks[0],
                "cwd": rank_working_directory(staging),
                "cleanup_command": cleanup_command(
                    node_rank=0, python=python, staging=staging
                ),
            },
            {
                "name": "rank-1-peer",
                # ssh does not forward the launch host's environment, so the
                # peer's rank carries its own copy of the same values.
                "command": ssh_argv(peer_host, ranks[1]),
                # The peer's own directory is not something `ssh` can be given,
                # so this `cwd` applies to the local ssh client only. The peer's
                # rank resolves the staged tree through `PYTHONPATH`, which is
                # what `import_check` measures on the far side.
                "cwd": rank_working_directory(staging),
                "cleanup_command": cleanup_command(
                    node_rank=1, python=python, staging=staging, peer_host=peer_host
                ),
            },
        ],
    }


def _run(
    argv: Sequence[str], timeout: float, *, cwd: str | Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(item) for item in argv],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
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


def requested_fabric_hcas(value: str | None) -> tuple[str, ...]:
    """The pinned device names, in the order they were given.

    NCCL takes a comma-separated list and rejects an empty element, so an empty
    or blank-only value is refused here rather than by a rank that has already
    started.
    """
    if value is None:
        return ()
    names = tuple(name.strip() for name in value.split(","))
    if not names or any(not name for name in names):
        raise SystemExit(
            f"--infiniband-hcas {value!r} is not a comma-separated list of "
            "device names"
        )
    return names


def checked_visible_devices(value: str | None, *, local_world_size: int) -> str | None:
    """The device index each rank may use, refused unless it is one per rank.

    A rank that is not told which device to use takes device `local_rank`, so on
    a host whose index 0 belongs to another tenant every lane would contend for
    a device the lane does not own. This is the value that prevents that, and it
    is refused unless it names exactly one device per rank: one entry per rank,
    in local-rank order, is the mapping a rank can resolve without a second
    rule. A shorter list would leave a rank unmapped and a longer one would
    imply a selection this tool does not perform.

    Names are checked here because the value reaches both nodes as an
    `env` assignment and a rank that was handed a non-numeric device would fail
    after the rendezvous rather than before it.
    """

    if value is None:
        return None
    names = [name.strip() for name in value.split(",")]
    if any(not name for name in names):
        raise SystemExit(
            f"--visible-devices {value!r} is not a comma-separated list of "
            "device indices"
        )
    if len(names) != int(local_world_size):
        raise SystemExit(
            f"--visible-devices {value!r} names {len(names)} device(s) for "
            f"{local_world_size} rank(s) per node; each rank is given one"
        )
    if len(set(names)) != len(names):
        raise SystemExit(
            f"--visible-devices {value!r} names a device twice; two ranks "
            "cannot share one"
        )
    for name in names:
        if not name.isdigit():
            raise SystemExit(
                f"--visible-devices {value!r} is not a list of numeric device "
                "indices"
            )
    return ",".join(names)


def _fabric_checks(
    *,
    peer_host: str,
    infiniband_hcas: str | None,
    timeout: float,
    fabric_root: Path,
    remote: Callable[[Sequence[str]], subprocess.CompletedProcess[str]],
) -> list[Check]:
    """The fabric devices a lane asked for are present on both nodes.

    Read from `/sys/class/infiniband` rather than from `ibstat`, which is not
    installed on every host that has the devices, and from the same path on both
    nodes so the two answers are comparable.
    """

    checks: list[Check] = []
    local = _fabric_devices(fabric_root)
    checks.append(
        Check(
            "launch_host_has_the_fabric",
            bool(local),
            f"{fabric_root}: {', '.join(local) if local else 'no devices'}",
        )
    )
    try:
        completed = remote(["ls", "-1", str(fabric_root)])
        peer = [
            line.strip()
            for line in completed.stdout.splitlines()
            if line.strip() and completed.returncode == 0
        ]
        detail = (
            ", ".join(peer)
            if peer
            else f"exit {completed.returncode}: {_first_line(completed.stderr)}"
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        peer, detail = [], f"{type(error).__name__}: {error}"
    checks.append(Check("peer_has_the_fabric", bool(peer), f"on {peer_host}: {detail}"))

    for name in requested_fabric_hcas(infiniband_hcas):
        checks.append(
            Check(
                f"launch_host_has_fabric_device_{name}",
                (fabric_root / name).is_dir(),
                f"{fabric_root / name}",
            )
        )
        try:
            completed = remote(["test", "-d", f"{fabric_root / name}"])
            present = completed.returncode == 0
            detail = (
                "present"
                if present
                else _first_line(completed.stderr) or f"exit {completed.returncode}"
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            present, detail = False, f"{type(error).__name__}: {error}"
        checks.append(
            Check(
                f"peer_has_fabric_device_{name}",
                present,
                f"on {peer_host}: {detail}",
            )
        )
    return checks


def _fabric_devices(root: Path = INFINIBAND_DEVICES) -> list[str]:
    """The fabric device names this node publishes, sorted and without a shell."""

    if not root.is_dir():
        return []
    return sorted(entry.name for entry in root.iterdir())


def _pinned_device_answer(
    visible_devices: Sequence[str],
    timeout: float,
    run: Callable[[Sequence[str], float], subprocess.CompletedProcess[str]],
) -> tuple[bool, str]:
    """Whether every pinned device exists and is idle, with the reading either way.

    Both halves matter. A device this lane pinned and another process is using
    is not a launch failure: it is a contaminated measurement, because the probe
    reports peak memory and a neighbour inside that memory turns the number into
    the sum of two workloads. A device that is absent fails at device
    resolution, which is the one case that reads as a hardware fault rather than
    as contention. The answer is reported per device so the finding names the
    index rather than the host.
    """

    findings: list[str] = []
    ok = True
    for index in visible_devices:
        try:
            completed = run(
                [
                    "nvidia-smi",
                    "-i",
                    str(index),
                    DEVICE_APPLICATION_QUERY,
                    UNITLESS_FORMAT,
                ],
                timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            ok = False
            findings.append(f"{index}: {type(error).__name__}: {error}")
            continue
        if completed.returncode != 0:
            ok = False
            findings.append(
                f"{index}: absent ({_first_line(completed.stderr) or completed.returncode})"
            )
            continue
        users = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if users:
            ok = False
            findings.append(f"{index}: {len(users)} process(es) {', '.join(users)}")
        else:
            findings.append(f"{index}: idle")
    return ok, "; ".join(findings)


def _device_pinning_checks(
    *,
    peer_host: str,
    visible_devices: str | None,
    timeout: float,
    local: Callable[[Sequence[str], float], subprocess.CompletedProcess[str]] = _run,
    remote: Callable[[Sequence[str]], subprocess.CompletedProcess[str]],
) -> list[Check]:
    """The devices a lane pinned are the ones it gets, and nothing else is on them.

    Only asked when a lane pinned devices: a lane that named none takes
    `local_rank`, which is the host's own business, and inventing a check for a
    decision this tool did not make would report a verdict about something else.
    """

    if visible_devices is None:
        return []
    indices = tuple(visible_devices.split(","))

    def on_peer(argv: Sequence[str], _timeout: float):
        return remote(argv)

    local_ok, local_detail = _pinned_device_answer(indices, timeout, local)
    peer_ok, peer_detail = _pinned_device_answer(indices, timeout, on_peer)
    return [
        Check("launch_host_has_every_pinned_device", local_ok, local_detail),
        Check(
            "peer_has_every_pinned_device",
            peer_ok,
            f"on {peer_host}: {peer_detail}",
        ),
    ]


def preflight(
    *,
    peer_host: str,
    peer_address: str,
    interface: str,
    master_port: int,
    python: str,
    launch_address: str | None = None,
    transport: str = DEFAULT_TRANSPORT,
    infiniband_hcas: str | None = None,
    visible_devices: str | None = None,
    timeout: float = 30.0,
    interface_root: Path = Path("/sys/class/net"),
    fabric_root: Path = INFINIBAND_DEVICES,
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

    # A lane that pinned devices is checked against the devices it pinned, not
    # against the host's device count: the two are the same question only when
    # index 0 is free, which is exactly the assumption a shared host breaks.
    checks.extend(
        _device_pinning_checks(
            peer_host=peer_host,
            visible_devices=visible_devices,
            timeout=timeout,
            local=run,
            remote=remote,
        )
    )

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

    # A lane asked for the fabric has to have one on both nodes. NCCL does not
    # fail when it cannot find a device it was told to use: it falls back to
    # whatever else is reachable, and the run then reports the transport it was
    # configured for while the debug log shows another one.
    if transport == TRANSPORT_RDMA:
        checks.extend(
            _fabric_checks(
                peer_host=peer_host,
                infiniband_hcas=infiniband_hcas,
                timeout=timeout,
                fabric_root=fabric_root,
                remote=remote,
            )
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
    """Copy the tree both ranks will run onto the filesystem they share.

    The copy arrives without `__pycache__`, and every rank of both nodes starts
    at once, so an uncompiled tree has all of them compiling the same modules
    and writing the same `.pyc` files. Those writes are atomic renames, and on
    the shared filesystem the loser waits on an inode the winner holds while it
    is still paging its own libraries in. This lane met exactly that as a
    `job_timeout` with no rank past process-group creation and no NCCL debug log
    on the launch host. Compiling here, once and before any rank starts, is what
    keeps the lane's startup a read. `PYTHONDONTWRITEBYTECODE` then keeps a rank
    that finds a stale entry from writing one back.
    """
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
    compiled = _run([python, "-m", "compileall", "-q", str(destination)], 900.0)
    if compiled.returncode != 0:
        raise RuntimeError(
            f"precompiling the staged tree failed with exit "
            f"{compiled.returncode}: {_first_line(compiled.stderr)}"
        )
    return {
        "schema": "flagquantum.multinode_staging.v1",
        "source": str(Path(source).resolve()),
        "destination": str(destination),
        "excluded": list(STAGING_EXCLUDES),
        "python": python,
        "files": sum(1 for _ in destination.rglob("*.py")),
        "bytecode_files": sum(1 for _ in destination.rglob("*.pyc")),
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


IMPORT_PROBE = ("-c", "import flagquantum, sys; sys.stdout.write(flagquantum.__file__)")


def rank_working_directory(staging: str | Path) -> str:
    """The directory a rank runs from, on the node that starts it.

    Named once because two places need the same answer: the plan gives both
    ranks this `cwd`, and the import check has to run the probe from it. `''`
    -- the working directory -- precedes `PYTHONPATH` on `sys.path`, so a check
    that inherited the operator's own directory answers a different question
    from the one the rank will answer, and answers it about a checkout the rank
    never runs.
    """
    return str(staging)


def import_check(
    *,
    node: str,
    staging: str,
    python: str,
    peer_host: str | None = None,
    timeout: float = 120.0,
    run: Callable[..., subprocess.CompletedProcess[str]] = _run,
) -> Check:
    """Each rank has to import the staged tree, and not some other copy.

    Staging one tree for both nodes is only worth anything if both nodes import
    it. The runner's environment has `flagquantum` installed editable against
    the runner's own workspace, so an interpreter can resolve the package to a
    checkout this lane never staged -- and the artifact would then record a
    revision that neither rank ran.

    Measured rather than assumed: `PYTHONPATH` wins over that editable install
    here, because the finder it installs is appended to `sys.meta_path` behind
    the default path finder. That is a property of how setuptools wrote the
    install, not a promise, so the lane asks both nodes instead of trusting it.

    Each side is asked from the directory its rank will run from: the launch
    host's rank is started by the watchdog with this `cwd`, and the peer's is
    started by the peer's login shell, which `ssh` does not give a directory
    either.
    """
    remote: list[str] = [
        python,
        *IMPORT_PROBE,
    ]
    environment = _env_prefix({"PYTHONPATH": staging})
    if peer_host is None:
        command = [*environment, *remote]
        working_directory: str | Path | None = rank_working_directory(staging)
    else:
        command = ssh_argv(peer_host, [*environment, *remote])
        # Not this tool's own directory, and not a path only this node has: the
        # peer's shell decides this one, so the check reproduces what the ranks
        # get rather than what the operator happens to have.
        working_directory = None
    try:
        completed = run(command, timeout, cwd=working_directory)
    except (OSError, subprocess.TimeoutExpired) as error:
        return Check(
            f"{node}_imports_the_staged_tree", False, f"{type(error).__name__}: {error}"
        )
    resolved = _first_line(completed.stdout)
    inside = bool(resolved) and Path(resolved).is_relative_to(Path(staging))
    detail = (
        f"{resolved or 'no output'}"
        if inside
        else f"resolved outside {staging}: {resolved or completed.stderr.strip()!r}"
    )
    return Check(f"{node}_imports_the_staged_tree", inside, detail)


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
    parser.add_argument(
        "--checkpoint-directory",
        default=None,
        help=(
            "shared, empty directory the training legs checkpoint into and "
            "resume from; defaults to <staging>-checkpoints, and is emptied "
            "before the run because the probe resumes from its own first leg"
        ),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--probe",
        choices=sorted(PROBES),
        default=DEFAULT_PROBE,
        help="which two-node workload to run on the pair",
    )
    parser.add_argument("--master-port", type=int, default=DEFAULT_MASTER_PORT)
    parser.add_argument("--interface", default=DEFAULT_INTERFACE)
    parser.add_argument(
        "--transport",
        choices=(TRANSPORT_SOCKET, TRANSPORT_RDMA),
        default=DEFAULT_TRANSPORT,
        help=(
            "the route both ranks are configured to use. `socket` disables "
            "InfiniBand over the interface above, which is the route the "
            "recorded evidence was taken on. `rdma` enables it and keeps the "
            "interface as NCCL's out-of-band bootstrap. The probe records the "
            "route the debug log shows, so a run configured for one transport "
            "and observed on another is refused rather than recorded"
        ),
    )
    parser.add_argument(
        "--infiniband-hcas",
        default=None,
        help=(
            "comma-separated fabric device names to pin for --transport rdma, "
            "for example mlx5_101,mlx5_102. Left unset, NCCL discovers the "
            "devices itself and skips a port that is down. The names are "
            "checked on both nodes before anything is launched"
        ),
    )
    parser.add_argument(
        "--visible-devices",
        default=None,
        help=(
            "comma-separated physical device indices to pin, one per rank per "
            "node, for example 1 or 1,2,3,4. Both nodes are given the same "
            "list and each rank takes the entry for its local rank. Left "
            "unset, a rank takes device local_rank, which is the host's index "
            "of that number. The pins are checked on both nodes before "
            "anything is launched: a named device that is absent fails, and "
            "one another process is using fails as a contaminated measurement "
            "rather than as a launch failure"
        ),
    )
    parser.add_argument("--source-revision", default=None)
    parser.add_argument(
        "--local-world-size",
        type=int,
        default=LOCAL_WORLD_SIZE,
        help=(
            "ranks each of the two nodes runs. One is the shape the recorded "
            "evidence was taken on, and it stays the default. What a wider "
            "value may be is the probe's business: a power of two for the "
            "statevector workload, at most three for the MPS one, and one or "
            "two for the tensor-network one. A width the probe would refuse is "
            "refused here, before a rank is started"
        ),
    )
    parser.add_argument(
        "--measure",
        action="store_true",
        help=(
            "ask the probe to time the sharded workload with warmup and "
            "repeats and record the samples, instead of the single unsynced "
            "reading the training legs already report. Both ranks are given "
            "the flag; a lane whose ranks disagreed would hang"
        ),
    )
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
        help="preflight, stage, plan and supervise the two nodes as one unit",
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


def checked_local_world_size(value: int, *, probe: str) -> int:
    """How many ranks one node runs, refused unless that probe can serve it.

    Every probe shards over the whole world size and reports the placement it
    resolved, so a node count and a per-node width are enough to describe the
    shape. What each probe can serve differs, and the launcher refuses a shape
    its probe would have refused anyway -- before a rank is started, which is
    the earliest point the answer is knowable.

    The two widths the probes are built on are mirrored here rather than
    imported: the probes import torch and this tool installs nothing, which
    `test_the_lane_runs_before_anything_is_installed` pins. They are pinned
    against the probes' own constants instead.
    """

    size = int(value)
    if size < 1:
        raise SystemExit(
            f"--local-world-size must be at least one, received {value}; a node "
            "runs at least one rank"
        )
    world_size = size * NODE_COUNT
    if probe == "statevector" and world_size & (world_size - 1):
        raise SystemExit(
            f"--local-world-size {value} gives a world size of {world_size}, "
            "which the statevector executor refuses: it shards by rank address "
            "bits, so the world size has to be a power of two"
        )
    if probe == "mps" and world_size > MPS_WIRES:
        raise SystemExit(
            f"--local-world-size {value} gives a world size of {world_size}, "
            "which the rank-owned MPS executor refuses: every rank owns at "
            f"least one site and the workload has {MPS_WIRES}"
        )
    if probe == "tn" and TN_SLICE_COUNT % world_size:
        raise SystemExit(
            f"--local-world-size {value} gives a world size of {world_size}, "
            "which the tensor-network executor refuses: the workload has a "
            f"fixed {TN_SLICE_COUNT}-way slice count, so the world size has to "
            "divide it"
        )
    return size


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
    output_directory = checked_lane_directory(
        staging,
        args.output_directory or f"{staging}-out",
        purpose="output",
    )
    checkpoint_directory = checked_lane_directory(
        staging,
        args.checkpoint_directory or f"{staging}-checkpoints",
        purpose="checkpoint",
    )
    peer_address = _peer_address(args.peer_host, args.preflight_timeout_seconds)
    # The address the checks pass against is the one the plan must use, so it
    # is settled before the checks rather than after them.
    launch_address = args.launch_address or local_address_towards(peer_address)
    master_port = resolve_master_port(launch_address, args.master_port)
    print(f"rendezvous: {launch_address}:{master_port}")
    # Resolved before the checks rather than after them: a device list this tool
    # cannot honor is not a finding to report, it is a plan it must not build.
    visible_devices = checked_visible_devices(
        args.visible_devices,
        local_world_size=checked_local_world_size(
            args.local_world_size, probe=args.probe
        ),
    )

    checks = preflight(
        peer_host=args.peer_host,
        peer_address=peer_address,
        interface=args.interface,
        transport=args.transport,
        infiniband_hcas=args.infiniband_hcas,
        visible_devices=visible_devices,
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
    for directory in (output_directory, checkpoint_directory):
        clear_lane_directory(directory)

    after_staging = [
        output_check(output_directory=output_directory),
        output_check(output_directory=checkpoint_directory),
        staging_check(peer_host=args.peer_host, staging=str(staging)),
        import_check(node="launch_host", staging=str(staging), python=args.python),
        import_check(
            node="peer",
            staging=str(staging),
            python=args.python,
            peer_host=args.peer_host,
        ),
    ]
    _report_checks(after_staging)
    if not all(check.ok for check in after_staging):
        return _fail(after_staging, "staging")

    plan = build_plan(
        peer_host=args.peer_host,
        launch_address=launch_address,
        staging=str(staging),
        output_directory=str(output_directory),
        checkpoint_directory=str(checkpoint_directory),
        python=args.python,
        master_port=master_port,
        interface=args.interface,
        transport=args.transport,
        infiniband_hcas=args.infiniband_hcas,
        visible_devices=visible_devices,
        measure=args.measure,
        probe=PROBES[args.probe],
        source_revision=args.source_revision,
        local_world_size=checked_local_world_size(
            args.local_world_size, probe=args.probe
        ),
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
            transport=args.transport,
            infiniband_hcas=args.infiniband_hcas,
            visible_devices=checked_visible_devices(
                args.visible_devices,
                local_world_size=checked_local_world_size(
                    args.local_world_size, probe=args.probe
                ),
            ),
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
        checkpoint_directory=(
            args.checkpoint_directory or f"{args.staging}-checkpoints"
        ),
        python=args.python,
        master_port=args.master_port,
        interface=args.interface,
        transport=args.transport,
        infiniband_hcas=args.infiniband_hcas,
        visible_devices=args.visible_devices,
        measure=args.measure,
        probe=PROBES[args.probe],
        source_revision=args.source_revision,
        local_world_size=checked_local_world_size(
            args.local_world_size, probe=args.probe
        ),
    )
    encoded = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Measure the frozen 1/2/4/8/16-rank MPS workload under one GPU reservation.

Run through the manual Scheduled hardware lane, which reserves the shared A800
pair. Each worker owns one process group; cleanup checks its PID start time and
never selects another campaign by a process-name pattern.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.multinode_launch_plan import import_check, ssh_argv  # noqa: E402
from tools.run_multinode_watchdog import run_jobs  # noqa: E402


def _run(command: list[str], *, cwd: Path | None = None, timeout: int = 60) -> str:
    return subprocess.check_output(command, cwd=cwd, timeout=timeout, text=True).strip()


def _start_time(pid: int) -> str:
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def stop_worker(pid_file: Path) -> None:
    """Signal only a live process group with the recorded Linux start time."""
    if not pid_file.exists():
        return
    record = json.loads(pid_file.read_text())
    pid = int(record["pid"])
    try:
        if _start_time(pid) != record["start_time"] or os.getpgid(pid) != pid:
            raise RuntimeError("worker identity changed; refusing to signal it")
        os.killpg(pid, signal.SIGTERM)
        for _ in range(50):
            if not Path(f"/proc/{pid}").exists():
                break
            time.sleep(0.1)
        else:
            if _start_time(pid) == record["start_time"]:
                os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except FileNotFoundError:
        pass
    pid_file.unlink(missing_ok=True)


def point_command(
    world: int,
    node: int,
    staging: Path,
    output: Path,
    address: str,
    port: int,
    *,
    transport_check: bool = False,
) -> list[str]:
    """Keep circuit geometry fixed while varying only deployment topology."""
    if world not in (1, 2, 4, 8, 16) or node not in range(2 if world == 16 else 1):
        raise ValueError("unsupported series topology")
    local = min(world, 8)
    launch = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        str(2 if world == 16 else 1),
        "--nproc-per-node",
        str(local),
        "--node-rank",
        str(node),
        "--master-addr",
        address,
        "--master-port",
        str(port),
    ]
    if transport_check:
        return [
            *launch,
            str(staging / "tests/distributed/mps_training_runtime.py"),
            "--backend",
            "nccl",
        ]
    return [
        *launch,
        str(staging / "benchmarks/mps_release_evidence.py"),
        "--role",
        "capacity-series-point",
        "--release-manifest",
        str(staging / "benchmarks/manifests/mps_release_v1.json"),
        "--local-world-size",
        str(local),
        "--node-count",
        str(2 if world == 16 else 1),
        "--premise",
        str(
            staging
            / "benchmarks/results/local/mps_capacity_131072q_chi768_16xa800_repeat_complete_20260806.json"
        ),
        "--measurements",
        str(output / f"world-{world}.json"),
    ]


def worker(args: argparse.Namespace) -> int:
    command = point_command(
        args.world,
        args.node,
        args.staging,
        args.output,
        args.address,
        args.port,
        transport_check=args.transport_check,
    )
    env = dict(
        os.environ,
        PYTHONPATH=str(args.staging),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES=",".join(str(i) for i in range(min(args.world, 8))),
        NCCL_SOCKET_IFNAME=args.interface,
        GLOO_SOCKET_IFNAME=args.interface,
        NCCL_DEBUG="INFO",
        NCCL_IB_DISABLE="0",
        NCCL_IB_HCA=args.infiniband_hcas,
        FQ_TEST_CHECKPOINT=str(args.output / "transport-checkpoint"),
        TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    process = subprocess.Popen(
        command, cwd=args.staging, env=env, start_new_session=True
    )
    args.pid_file.write_text(
        json.dumps({"pid": process.pid, "start_time": _start_time(process.pid)})
    )
    try:
        return process.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return 124
    finally:
        stop_worker(args.pid_file)
        process.wait(timeout=10)


def _inventory(peer: str | None = None) -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=uuid,name,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    if peer:
        command = ssh_argv(peer, command)
    rows = _run(command, timeout=20).splitlines()
    return [
        dict(
            zip(
                ("uuid", "name", "memory_mib", "utilization"),
                (v.strip() for v in row.split(",")),
                strict=True,
            )
        )
        for row in rows
    ]


def campaign(args: argparse.Namespace) -> int:
    if not args.staging.name.startswith("fq-mps-capacity-") or args.staging.exists():
        raise SystemExit(
            "use a fresh fq-mps-capacity-* directory on the shared filesystem"
        )
    if _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT):
        raise SystemExit("capacity measurements require committed, clean source")
    revision = _run(["git", "rev-parse", "HEAD"], cwd=ROOT)
    inventory = [_inventory(), _inventory(args.peer)]
    if any(len(node) != 8 for node in inventory) or any(
        int(gpu["memory_mib"]) > 256 or int(gpu["utilization"])
        for node in inventory
        for gpu in node
    ):
        raise SystemExit("the reserved pair must contain sixteen idle GPUs")
    args.staging.mkdir(parents=True)
    args.output.mkdir(parents=True, exist_ok=False)
    _run(["git", "init", str(args.staging)])
    objects = subprocess.check_output(
        ["git", "rev-list", "--objects", "--no-walk", "HEAD"], cwd=ROOT, timeout=30
    )
    pack = subprocess.check_output(
        ["git", "pack-objects", "--stdout"], input=objects, cwd=ROOT, timeout=90
    )
    subprocess.run(
        ["git", "index-pack", "--stdin"],
        input=pack,
        cwd=args.staging,
        check=True,
        timeout=60,
    )
    (args.staging / ".git/shallow").write_text(revision + "\n")
    _run(["git", "checkout", "--detach", revision], cwd=args.staging)
    _run([sys.executable, "-m", "compileall", "-q", str(args.staging)], timeout=180)
    for peer in (None, args.peer):
        check = import_check(
            node=peer or "launch",
            staging=str(args.staging),
            python=sys.executable,
            peer_host=peer,
        )
        if not check.ok:
            raise SystemExit(str(check))
    (args.output / "inventory.json").write_text(
        json.dumps(
            {
                "commit": revision,
                "hosts": inventory,
                "launch_address": args.address,
                "peer": args.peer,
                "cuda_allocator_policy": "expandable_segments:True",
                "infiniband_hcas": args.infiniband_hcas,
                "topology": [
                    _run(["nvidia-smi", "topo", "-m"]),
                    _run(ssh_argv(args.peer, ["nvidia-smi", "topo", "-m"])),
                ],
            },
            indent=2,
        )
        + "\n"
    )
    for world, transport_check in (
        (16, True),
        (1, False),
        (2, False),
        (4, False),
        (8, False),
        (16, False),
    ):
        point = "transport-check" if transport_check else f"world-{world}"
        stop = threading.Event()
        log = args.output / f"{point}-telemetry.jsonl"

        def monitor(
            stop: threading.Event = stop,
            world: int = world,
            log: Path = log,
            point: str = point,
        ) -> None:
            while not stop.is_set():
                try:
                    sample = {
                        "time": time.time(),
                        "phase": point,
                        "world_size": world,
                        "hosts": [_inventory(), _inventory(args.peer)],
                    }
                    with log.open("a") as stream:
                        stream.write(json.dumps(sample) + "\n")
                    print(json.dumps(sample), flush=True)
                except (subprocess.SubprocessError, OSError) as exc:
                    print(f"telemetry failed: {exc}", flush=True)
                stop.wait(30)

        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        jobs: list[dict[str, Any]] = []
        for node in range(2 if world == 16 else 1):
            pid_file = args.output / f"{point}-node-{node}.pid"
            base = [
                sys.executable,
                str(args.staging / "tools/run_mps_capacity_series.py"),
            ]
            command = [
                *base,
                "--worker",
                "--staging",
                str(args.staging),
                "--output",
                str(args.output),
                "--world",
                str(world),
                "--node",
                str(node),
                "--pid-file",
                str(pid_file),
                "--address",
                args.address,
                "--port",
                str(args.port + world),
                "--interface",
                args.interface,
                "--timeout",
                str(args.timeout),
                "--infiniband-hcas",
                args.infiniband_hcas,
                *(["--transport-check"] if transport_check else []),
            ]
            cleanup = [*base, "--stop-worker", "--pid-file", str(pid_file)]
            if node:
                command, cleanup = ssh_argv(args.peer, command), ssh_argv(
                    args.peer, cleanup
                )
            jobs.append(
                {
                    "name": f"node-{node}",
                    "command": command,
                    "cwd": str(args.staging),
                    "cleanup_command": cleanup,
                }
            )
        (args.output / f"{point}-launch.json").write_text(
            json.dumps(
                {"commit": revision, "world_size": world, "jobs": jobs}, indent=2
            )
            + "\n"
        )
        try:
            if len(jobs) == 1:
                with (args.output / f"{point}.log").open("w") as stream:
                    result = subprocess.run(
                        jobs[0]["command"],
                        cwd=args.staging,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        timeout=args.timeout + 30,
                        check=False,
                    )
                passed = result.returncode == 0
            else:
                report = run_jobs(
                    jobs,
                    output_dir=args.output / f"{point}-logs",
                    timeout_seconds=args.timeout + 30,
                )
                (args.output / f"{point}-watchdog.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
                passed = report["reason"] == "completed"
        finally:
            stop.set()
            for job in jobs:
                subprocess.run(job["cleanup_command"], timeout=30, check=True)
            watcher.join(timeout=45)
        if not passed:
            raise SystemExit(
                f"{point} failed; retain all logs and diagnose before retry"
            )
    digests = {
        str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.output.rglob("*")
        if p.is_file()
    }
    (args.output / "raw-digests.json").write_text(
        json.dumps({"commit": revision, "sha256": digests}, indent=2) + "\n"
    )
    return 0


def main() -> int:
    def interrupted(signum: int, frame: Any) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--peer", default="jp-a800-171")
    parser.add_argument("--address", default="10.1.15.172")
    parser.add_argument("--port", type=int, default=29640)
    parser.add_argument("--interface", default="ens22f0")
    parser.add_argument(
        "--infiniband-hcas",
        default="mlx5_101,mlx5_102,mlx5_103,mlx5_104,mlx5_105,mlx5_106,mlx5_107,mlx5_108",
    )
    parser.add_argument("--transport-check", action="store_true")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--stop-worker", action="store_true")
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--world", type=int)
    parser.add_argument("--node", type=int)
    args = parser.parse_args()
    if (args.worker or args.stop_worker) and args.pid_file is None:
        parser.error("workers require --pid-file")
    if not args.stop_worker and (args.staging is None or args.output is None):
        parser.error("--staging and --output are required")
    if args.worker and (args.world is None or args.node is None):
        parser.error("workers require --world and --node")
    if args.stop_worker:
        stop_worker(args.pid_file)
        return 0
    if args.worker:
        return worker(args)
    return campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())

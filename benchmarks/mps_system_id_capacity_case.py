"""Run and audit the three-way MPS system-identification capacity case."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN = REPO_ROOT / "examples/mps_hamiltonian_identification/train.py"


def _workload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "n_wires": args.n_wires,
        "n_initial_states": args.n_initial_states,
        "time_steps": tuple(args.time_steps),
        "observation_stride": args.observation_stride,
        "probe_batch_size": args.probe_batch_size,
        "max_bond": args.max_bond,
        "cutoff": args.cutoff,
        "steps": args.steps,
        "learning_rate": args.lr,
        "seed": args.seed,
        "dtype": "complex64",
    }


def _fingerprint(workload: dict[str, Any]) -> str:
    encoded = json.dumps(workload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def classify_failure(returncode: int, output: str) -> str:
    lowered = output.lower()
    if returncode == 0:
        return "passed"
    if "out of memory" in lowered or "cuda_oom" in lowered:
        return "cuda_oom"
    if returncode == 124 or "timed out" in lowered:
        return "timeout"
    return "runtime_error"


def capacity_gate(
    single: dict[str, Any], replicated: dict[str, Any], sharded: dict[str, Any]
) -> tuple[bool, tuple[str, ...]]:
    blockers = []
    fingerprints = {item.get("workload_sha256") for item in (single, replicated, sharded)}
    if len(fingerprints) != 1 or None in fingerprints:
        blockers.append("workload_identity_mismatch")
    if single.get("status") != "cuda_oom":
        blockers.append("single_gpu_capacity_failure_not_measured")
    if replicated.get("status") != "cuda_oom":
        blockers.append("replicated_capacity_failure_not_measured")
    payload = sharded.get("result") or {}
    if sharded.get("status") != "passed":
        blockers.append("site_sharded_training_not_completed")
    else:
        if payload.get("distribution_semantics") != "sharded_across_ranks":
            blockers.append("site_sharded_semantics_missing")
        if payload.get("teacher_full_mps_materialization") is not False:
            blockers.append("teacher_full_mps_materialization_not_rejected")
        if payload.get("optimizer_steps", 0) < 1:
            blockers.append("adam_update_not_completed")
        if not payload.get("best_checkpoints"):
            blockers.append("best_checkpoint_missing")
        evidence = payload.get("gpu_evidence") or {}
        if not evidence.get("accelerator") or not evidence.get("peak_memory_bytes_by_rank"):
            blockers.append("per_rank_gpu_evidence_missing")
        communication = payload.get("communication") or {}
        if communication.get("backend") != "nccl" or communication.get(
            "measured_logical_boundary_bytes", 0
        ) <= 0:
            blockers.append("nccl_communication_evidence_missing")
    return not blockers, tuple(blockers)


def _command(
    args: argparse.Namespace, mode: str, output: Path
) -> tuple[list[str], dict[str, str]]:
    world = 1 if mode == "single_gpu" else args.world_size
    command = []
    if world > 1:
        command.extend(
            [args.torchrun, "--standalone", f"--nproc-per-node={world}"]
        )
    else:
        command.append(args.python)
    command.extend(
        [
            str(TRAIN),
            "--n-wires", str(args.n_wires),
            "--n-initial-states", str(args.n_initial_states),
            "--time-steps", ",".join(map(str, args.time_steps)),
            "--observation-stride", str(args.observation_stride),
            "--probe-batch-size", str(args.probe_batch_size),
            "--max-bond", str(args.max_bond),
            "--reverse-max-saved-gib", str(args.reverse_max_saved_gib),
            "--cutoff", str(args.cutoff),
            "--steps", str(args.steps),
            "--lr", str(args.lr),
            "--log-every", str(args.log_every),
            "--early-stopping-patience", str(args.early_stopping_patience),
            "--early-stopping-min-delta", str(args.early_stopping_min_delta),
            "--validation-initial-states", str(args.validation_initial_states),
            "--validation-time-steps", ",".join(map(str, args.validation_time_steps)),
            "--seed", str(args.seed),
            "--device", "auto",
            "--quiet-rank-heartbeats",
            "--output", str(output),
        ]
    )
    if args.early_stopping_loss is not None:
        command.extend(["--early-stopping-loss", str(args.early_stopping_loss)])
    if mode == "site_sharded":
        command.append("--site-sharded")
        if args.compile_site_kernels:
            command.append("--compile-site-kernels")
        if args.compile_observables:
            command.append("--compile-observables")
    elif mode == "replicated":
        command.append("--replicated-capacity-baseline")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in range(world))
    return command, env


def _gpu_sample() -> list[dict[str, int]]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    samples = []
    for line in completed.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 4:
            samples.append({"gpu": int(fields[0]), "memory_used_mib": int(float(fields[1])),
                "sm_utilization_percent": int(float(fields[2])), "power_watts": int(float(fields[3]))})
    return samples


def _run(args: argparse.Namespace, mode: str, workload_sha256: str) -> dict[str, Any]:
    result_path = args.output_dir / f"{mode}.result.json"
    command, env = _command(args, mode, result_path)
    started = time.perf_counter()
    log_path = args.output_dir / f"{mode}.log"
    samples = []
    with log_path.open("w") as stream:
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, text=True,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + args.timeout_seconds
        while process.poll() is None and time.monotonic() < deadline:
            samples.append({"elapsed_seconds": time.perf_counter() - started, "gpus": _gpu_sample()})
            time.sleep(args.sample_interval_seconds)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            returncode = 124
            stream.write("\ntimed out\n")
        else:
            returncode = int(process.returncode)
    log = log_path.read_text()
    status = classify_failure(returncode, log)
    result = json.loads(result_path.read_text()) if status == "passed" and result_path.exists() else None
    per_gpu = {}
    for gpu in range(1 if mode == "single_gpu" else args.world_size):
        rows = [item for sample in samples for item in sample["gpus"] if item["gpu"] == gpu]
        utilization = [item["sm_utilization_percent"] for item in rows]
        per_gpu[str(gpu)] = {
            "sample_count": len(rows),
            "mean_sm_utilization_percent": statistics.fmean(utilization) if utilization else None,
            "median_sm_utilization_percent": statistics.median(utilization) if utilization else None,
            "p95_sm_utilization_percent": sorted(utilization)[min(len(utilization) - 1, int(0.95 * len(utilization)))] if utilization else None,
            "max_memory_used_mib": max((item["memory_used_mib"] for item in rows), default=None),
            "mean_power_watts": statistics.fmean(item["power_watts"] for item in rows) if rows else None,
        }
    return {
        "mode": mode,
        "status": status,
        "returncode": returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "workload_sha256": workload_sha256,
        "command": command,
        "log": str(log_path),
        "result_path": str(result_path) if result is not None else None,
        "result": result,
        "gpu_samples": samples,
        "gpu_summary": per_gpu,
    }


def _parse_times(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-wires", type=int, default=1024)
    parser.add_argument("--n-initial-states", type=int, default=1)
    parser.add_argument("--time-steps", type=_parse_times, default=(3,))
    parser.add_argument("--observation-stride", type=int, default=16)
    parser.add_argument("--probe-batch-size", type=int, default=1)
    parser.add_argument("--max-bond", type=int, default=128)
    parser.add_argument("--reverse-max-saved-gib", type=float, default=36.0)
    parser.add_argument("--cutoff", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-8)
    parser.add_argument("--early-stopping-loss", type=float, default=1e-7)
    parser.add_argument("--validation-initial-states", type=int, default=2)
    parser.add_argument("--validation-time-steps", type=_parse_times, default=(4,))
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.5)
    parser.add_argument("--modes", default="site_sharded",
        help="Comma-separated subset of single_gpu,replicated,site_sharded")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--torchrun", default=str(Path(sys.executable).with_name("torchrun")))
    parser.add_argument("--compile-site-kernels", action="store_true")
    parser.add_argument("--compile-observables", action="store_true")
    args = parser.parse_args()
    if args.world_size < 2 or args.probe_batch_size != 1:
        parser.error("the capacity case requires world-size >=2 and probe-batch-size=1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workload = _workload(args)
    workload_sha256 = _fingerprint(workload)
    modes = tuple(item for item in args.modes.split(",") if item)
    if not modes or any(item not in {"single_gpu", "replicated", "site_sharded"} for item in modes):
        parser.error("modes must select single_gpu, replicated, or site_sharded")
    records = {
        mode: _run(args, mode, workload_sha256)
        for mode in modes
    }
    complete = all(mode in records for mode in ("single_gpu", "replicated", "site_sharded"))
    passed, blockers = (
        capacity_gate(records["single_gpu"], records["replicated"], records["site_sharded"])
        if complete else (False, ("single_gpu_and_replicated_baselines_pending",))
    )
    payload = {
        "schema": "flagquantum.mps_hamiltonian_identification.capacity.v1",
        "claim_evidence_type": "production_training_benchmark",
        "workload": workload,
        "workload_sha256": workload_sha256,
        "runs": records,
        "single_gpu_expected_oom": records.get("single_gpu", {}).get("status") == "cuda_oom",
        "replicated_expected_oom": records.get("replicated", {}).get("status") == "cuda_oom",
        "distribution_semantics": "sharded_across_ranks" if passed else "unverified",
        "sharded_optimizer_update_evidence": passed,
        "capacity_gate_passed": passed,
        "scalability_claim_allowed": passed,
        "release_gate_allowed": False,
        "scalability_blockers": blockers + (("independent_release_review_pending",) if passed else ()),
    }
    output = args.output_dir / "capacity_comparison.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "passed": passed, "blockers": blockers}))


if __name__ == "__main__":
    main()

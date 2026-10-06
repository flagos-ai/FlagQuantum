"""Measure where a sharded MPS matched-speed step spends its time.

The frozen matched-speed contract publishes a ratio between one device and a
sharded leg.  A ratio says how much faster the sharded leg ran; it does not say
which phase owned the time, so a leg that fails the efficiency threshold cannot
be repaired from the published number alone -- the number names a shortfall
without naming a term.

This probe runs the acceptance rung of the frozen ladder through the same
frozen body the release producer uses and records, per rank and per iteration,
the phase breakdown the executor already keeps in ``MPSStepMetrics``: forward,
reverse, optimizer and diagnostics seconds beside the end-to-end step, the
layer-halo wait, the collective counts and the ownership counters.  The point is
to name the term that owns the missing speedup rather than to publish a number,
so the probe changes nothing: same body, same rung, same optimizer, same
gradient policy, same reverse-tape rule, same launcher.

Nothing here is evidence.  This is a diagnostic, it writes no artifact the
release lane reads, and its output is deliberately not a sealed envelope.

Its output is kept anyway, because it is what the frozen shardability calibration
is fitted from.  ``mps_matched_speed_sweep.sh`` varies the rung at each world size
and the records land in
``benchmarks/results/local/mps_matched_speed_shardability``, from which
``benchmarks/build_mps_shardability_calibration.py`` recomputes the serial
fraction without asking anything of the machine that measured it.  The protocol
is the frozen one -- the same warmup and measured step counts the release producer
is required to time -- because a rung timed at some other protocol is not the
ladder the contract froze.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(
    os.environ.get("FQ_REPOSITORY_ROOT", str(Path(__file__).resolve().parents[1]))
)
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        raise SystemExit(f"{name} is not an integer") from None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--rung", default="")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-seconds", type=float, default=0.0)
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "profile one extra unmeasured iteration and record the per-range "
            "CPU and CUDA totals, so a phase that does not shard can be told "
            "from a phase that does"
        ),
    )
    arguments = parser.parse_args()

    import torch
    import torch.distributed as dist

    # The frozen manifest names its workload, ladder and premise files by
    # repository-relative path, and the release roles are launched from the
    # repository root so those paths resolve.  Resolve them the same way here
    # rather than relying on the launcher's working directory.
    os.chdir(REPOSITORY_ROOT)

    import benchmarks.mps_release_evidence as producer
    import flagquantum.experimental.distributed as fqxd
    import flagquantum.runtime.executors.mps.records as fqxm

    contract = producer._capacity_contract(arguments.release_manifest)
    manifest = producer._load_frozen_manifest(arguments.release_manifest)
    speed = manifest["speed_workload"]
    ladder = producer._speed_ladder(speed)
    steps = producer._ladder_steps(speed)
    acceptance = str(speed["acceptance_configuration"])
    rung = next(
        entry
        for entry in ladder
        if str(entry["name"]) == (arguments.rung or acceptance)
    )
    frozen_protocol = (int(speed["warmup_steps"]), int(speed["measured_steps"]))
    if (arguments.warmup, arguments.iterations) != frozen_protocol:
        raise SystemExit(
            "the frozen matched-speed protocol requires --warmup "
            f"{frozen_protocol[0]} --iterations {frozen_protocol[1]}; a probe at "
            "another protocol timed a step nobody froze"
        )

    world = _env_int("WORLD_SIZE", 1)
    device = torch.device("cuda", _env_int("LOCAL_RANK", 0))
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    rank = _env_int("RANK", 0)

    workload = producer._frozen_workload_at(
        contract, int(rung["n_sites"]), int(rung["trained_max_bond"])
    )
    logical_bytes = int(workload.logical_mps_bytes(workload.N_SITES, workload.MAX_BOND))
    budget_bytes = int(workload.reverse_checkpoint_capacity_bytes(logical_bytes, world))
    initial = workload.rank_owned_initial_mps(
        workload.N_SITES, workload.MAX_BOND, device
    )
    parameters = workload.frozen_parameters(device)
    circuit = workload.workload(parameters)
    # The objective term the frozen capacity producer names, so the probe times
    # the observable the release record times rather than a cheaper one.
    observable = {workload.N_SITES // 2: "z"}

    def run_once() -> tuple[float, dict[str, object]]:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = fqxd.train_distributed_mps(
            circuit,
            steps=steps,
            observable=observable,
            optimizer=str(contract["optimizer"]),
            lr=float(contract["learning_rate"]),
            device=device,
            max_bond=int(workload.MAX_BOND),
            gradient_policy=str(contract["gradient_policy"]),
            gradient_tolerance=float(contract["truncation_error_budget"]),
            initial_mps_tensors=initial,
            initial_mps_left_canonical=bool(contract["initial_mps_left_canonical"]),
            canonicalization_policy=str(contract["canonicalization_policy"]),
            svd_driver=str(contract["svd_driver"]),
            reverse_checkpoint_policy=fqxm.MPSReverseCheckpointPolicy(
                max_saved_bytes=budget_bytes
            ),
        )
        torch.cuda.synchronize(device)
        wall = time.perf_counter() - started
        summary = dict(result.summary())
        peak = int(torch.cuda.max_memory_allocated(device))
        del result
        return wall, {
            "wall_seconds": wall,
            "peak_memory_bytes": peak,
            "steps": summary["step_metrics"],
        }

    iterations: list[dict[str, object]] = []
    for iteration in range(arguments.warmup + arguments.iterations):
        wall, record = run_once()
        if iteration < arguments.warmup:
            continue
        record["iteration"] = iteration - arguments.warmup
        iterations.append(record)

    profile_ranges: dict[str, object] | None = None
    if arguments.profile:
        # The profiler adds its own overhead, so this iteration is excluded from
        # the timed series and read only for where the time went. CUPTI is left
        # out on purpose: instrumenting the CUDA side makes every collective
        # report spin time, which reads as communication that is not there. A
        # host-only profile reports each annotated range's wall span, which is
        # the split this probe is asking for.
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU]) as prof:
            run_once()
        rows = {}
        top = {}
        for entry in prof.key_averages():
            key = str(entry.key)
            payload = {
                "count": int(entry.count),
                "cpu_seconds_total": float(entry.cpu_time_total) / 1e6,
                "cpu_seconds_self": float(entry.self_cpu_time_total) / 1e6,
            }
            if key.startswith("flagquantum::mps::"):
                rows[key] = payload
            if float(entry.cpu_time_total) / 1e6 >= 0.005 or (
                float(entry.self_cpu_time_total) / 1e6 >= 0.005
            ):
                top[key] = payload
        profile_ranges = {
            "phase_ranges": rows,
            "entries_over_5ms": dict(
                sorted(
                    top.items(),
                    key=lambda item: -float(item[1]["cpu_seconds_self"]),
                )
            ),
        }

    if arguments.device_seconds:
        time.sleep(arguments.device_seconds)

    record = {
        "probe": "mps_matched_speed_phase_breakdown",
        "rank": rank,
        "world_size": world,
        "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", 1)),
        "hostname": socket.gethostname(),
        "device_name": torch.cuda.get_device_properties(device).name,
        "rung": str(rung["name"]),
        "n_sites": int(workload.N_SITES),
        "trained_max_bond": int(workload.MAX_BOND),
        "checkpoint_budget_bytes": budget_bytes,
        "logical_mps_bytes": logical_bytes,
        "steps_per_iteration": steps,
        "iterations": iterations,
        "profile": profile_ranges,
    }
    gathered: list[object] = [None] * world
    dist.all_gather_object(gathered, record)
    if rank == 0:
        arguments.output.write_text(
            json.dumps(
                {
                    # The record lands beside the release results, and the audit
                    # reads every JSON under `benchmarks/results`. This one is a
                    # diagnostic input rather than a claim, so it says so in the
                    # field the audit classifies by. Declaring it here rather than
                    # leaving the classification to a reader is what keeps the
                    # record from being audited as a claim it never made.
                    "artifact_class": "auxiliary_report",
                    "non_release_evidence": True,
                    "release_gate_allowed": False,
                    "scalability_claim_allowed": False,
                    "claim_evidence_type": "development_smoke",
                    "world_size": world,
                    "acceptance_configuration": acceptance,
                    "rung": str(rung["name"]),
                    "ranks": gathered,
                },
                indent=2,
            )
        )
        print(f"PHASE {arguments.output}", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""torchrun correctness harness for distributed statevector planning.

Run manually, for example:

    torchrun --standalone --nproc_per_node=2 tests/distributed/statevector_correctness.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flagquantum as fq  # noqa: E402


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def build_reference_circuit(n_wires: int) -> fq.Circuit:
    circuit = fq.Circuit(n_wires)
    circuit.h(0)
    if n_wires > 1:
        circuit.rx(1, theta=0.2)
    if n_wires > 2:
        circuit.x(n_wires - 1).rx(n_wires - 1, theta=0.4).cx(0, n_wires - 1)
    if n_wires > 3:
        circuit.rz(n_wires - 1, theta=-0.3).cx(1, n_wires - 2)
    if n_wires > 4:
        circuit.x(n_wires - 2)
    return circuit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=_env_int("WORLD_SIZE", 1))
    parser.add_argument("--n-wires", type=int, default=4)
    parser.add_argument("--distribution", default=None)
    parser.add_argument("--topology", default=None)
    parser.add_argument("--backend", default=os.environ.get("FQ_DIST_BACKEND", "gloo"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _gather_summaries(summary: dict, world_size: int) -> list[dict]:
    if (
        world_size <= 1
        or not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
    ):
        return [summary]
    gathered: list[dict | None] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(gathered, summary)
    return [item for item in gathered if item is not None]


def _log(rank: int, message: str, *, enabled: bool) -> None:
    if enabled:
        print(f"[rank {rank}] {message}", flush=True)


def _barrier(label: str, rank: int, *, enabled: bool) -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        _log(rank, f"barrier enter: {label}", enabled=enabled)
        if torch.cuda.is_available() and torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        else:
            torch.distributed.barrier()
        _log(rank, f"barrier exit: {label}", enabled=enabled)


def run_statevector_correctness(
    *,
    world_size: int,
    n_wires: int,
    distribution: str | None = None,
    topology: str | None = None,
    backend: str = "gloo",
    device: str = "cpu",
    verbose: bool = False,
) -> dict:
    env_world_size = _env_int("WORLD_SIZE", world_size)
    rank = _env_int("RANK", 0)
    world_size = max(1, int(world_size or env_world_size))
    if env_world_size != world_size:
        world_size = env_world_size

    _log(rank, "initializing process group", enabled=verbose)
    context = fq.init_torch_distributed(
        backend=backend,
        world_size=world_size,
        device=device,
    )
    rank = context.rank
    world_size = context.world_size
    _log(
        rank,
        f"initialized backend={context.backend} device={context.device}",
        enabled=verbose,
    )

    _barrier("after_init", rank, enabled=verbose)
    circuit = build_reference_circuit(n_wires)
    plan = fq.plan_distributed_statevector(circuit, world_size=world_size)
    validation = plan.validate()
    report = plan.execute_dry_run()
    _log(
        rank,
        f"plan ready distribution={plan.distribution} topology={plan.topology.layout} "
        f"segments={len(plan.execution_segments)}",
        enabled=verbose,
    )
    _barrier("before_transport", rank, enabled=verbose)
    transport = fq.execute_distributed_statevector_transport(plan, device=device)
    _log(rank, f"transport summary={transport.summary()}", enabled=verbose)
    _barrier("after_transport", rank, enabled=verbose)
    expectation = circuit.expectation_z().detach().cpu()

    if not validation.valid:
        raise AssertionError(
            f"invalid distributed statevector plan: {validation.errors}"
        )
    if not report.valid:
        raise AssertionError(f"invalid dry-run executor report: {report.errors}")
    if distribution and plan.distribution != distribution:
        raise AssertionError(
            f"distribution mismatch: {plan.distribution} != {distribution}"
        )
    if topology and plan.topology.layout != topology:
        raise AssertionError(f"topology mismatch: {plan.topology.layout} != {topology}")
    if report.summary()["trace_event_count"] != len(validation.events):
        raise AssertionError("dry-run report does not cover every trace event")
    if not transport.valid:
        raise AssertionError(f"distributed transport failed: {transport.errors}")

    summary = {
        "rank": rank,
        "world_size": world_size,
        "plan": plan.summary(),
        "trace": validation.summary(),
        "executor": report.summary(),
        "transport": transport.summary(),
        "expectation_z": tuple(float(value) for value in expectation.reshape(-1)),
    }
    _log(rank, "gathering summaries", enabled=verbose)
    summaries = _gather_summaries(summary, world_size)
    if rank == 0:
        reference = summaries[0]["plan"]
        for item in summaries:
            if item["plan"] != reference:
                raise AssertionError("plan summaries differ across ranks")
    return {
        "rank": rank,
        "world_size": world_size,
        "plan": plan,
        "validation": validation,
        "executor": report,
        "transport": transport,
        "summaries": summaries,
    }


def main() -> None:
    args = parse_args()
    result = run_statevector_correctness(
        world_size=args.world_size,
        n_wires=args.n_wires,
        distribution=args.distribution,
        topology=args.topology,
        backend=args.backend,
        device=args.device,
        verbose=args.verbose,
    )
    if result["rank"] == 0:
        plan = result["plan"]
        validation = result["validation"]
        report = result["executor"]
        summaries = result["summaries"]
        print(
            "statevector_correctness "
            f"world_size={result['world_size']} "
            f"distribution={plan.distribution} "
            f"topology={plan.topology.layout} "
            f"trace_events={validation.summary()['event_count']} "
            f"transport_events={sum(item['transport']['event_count'] for item in summaries)} "
            f"peak_buffer_bytes={report.peak_buffer_bytes}"
        )

    fq.destroy_torch_distributed()


if __name__ == "__main__":
    main()

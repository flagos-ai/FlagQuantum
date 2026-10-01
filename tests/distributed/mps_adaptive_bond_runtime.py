"""Rank-sharded MPS adaptive bond planning regression runtime.

Run under a real launcher; every rank checks the same invariants after the ranks
exchange their measurements, so a broken invariant is reported instead of
leaving the other ranks blocked in a collective:

    torchrun --standalone --nproc-per-node=2 tests/distributed/mps_adaptive_bond_runtime.py

Two coupled defects are pinned here.

1. ``case_site_local_records`` hits a two-wire gate whose two sites are both
   inside one rank's shard. That path called ``ShardedMPSState.apply_two_local``,
   which returned only the discarded weight, so the split metadata was discarded
   and no ``MPSTruncationRecord`` was appended: ``MPSState.truncation_errors``
   received the measured weight while ``MPSState.truncation_records`` stayed
   empty. A cross-shard gate never had this problem -- it is applied through
   ``MPSState`` on the owning rank, which records -- so the defect only shows up
   in the shard layouts where a pair is site-local.

2. ``case_merged_plan`` uses a circuit whose only truncating gate crosses a shard
   boundary. Only the owning rank measures it, so on the other ranks the rank
   local plan is empty. ``_allreduce_adaptive_plan`` reduced just the observed
   error and copied ``hot_bonds``/``suggested_max_bond`` from the rank-local
   plan, so the ranks agreed that the budget was violated but disagreed about
   what to do: the owner reran with a grown bond, the others reran unchanged.
"""

from __future__ import annotations

import json
import os

import torch

from flagquantum.circuit import Circuit
from flagquantum.runtime.executors.mps.execution import (
    destroy_torch_distributed,
    run_distributed_mps,
)
from flagquantum.runtime.executors.mps.metadata_transport import all_gather_json
from flagquantum.simulation.mps.entrypoints import run_mps, run_mps_adaptive

BUDGET = 0.0


def fidelity(vector: torch.Tensor, reference: torch.Tensor) -> float:
    vector = vector.reshape(-1).to(reference.dtype)
    reference = reference.reshape(-1)
    return float(
        (torch.abs(torch.vdot(reference, vector)) ** 2)
        / (reference.norm() ** 2 * vector.norm() ** 2)
    )


def fixture(n_wires: int, gates: tuple[tuple[int, int], ...]) -> Circuit:
    circuit = Circuit(n_wires)
    circuit.h(0)
    for left, right in gates:
        circuit.cx(left, right)
    return circuit


def exact_statevector(circuit: Circuit) -> torch.Tensor:
    return run_mps(circuit, max_bond=None, cutoff=0.0).to_statevector()


def case_site_local_records(rank: int, world: int) -> dict[str, object]:
    """A truncating two-wire gate inside one shard must leave a record."""
    circuit = fixture(4, ((0, 1),))
    result = run_distributed_mps(
        circuit,
        world_size=world,
        distributed_executor="torch",
        rank=rank,
        max_bond=1,
        cutoff=0.0,
    )
    state = result.local_state
    plan = state.adaptive_bond_plan(global_error_budget=BUDGET)
    by_bond = state.truncation_error_by_bond()
    return {
        "rank": rank,
        "measured_total": float(sum(v for v in state.truncation_errors if v > 0.0)),
        "recorded_total": float(sum(by_bond.values())),
        "record_count": len(state.truncation_records),
        "hot_bonds": [int(bond) for bond in plan.hot_bonds],
        "suggested_max_bond": int(plan.suggested_max_bond),
        "current_max_bond": int(plan.current_max_bond),
    }


def case_merged_plan(rank: int, world: int) -> dict[str, object]:
    """Ranks that measured nothing must still follow the global plan."""
    circuit = fixture(3, ((0, 1),))
    reference = exact_statevector(circuit)
    result = run_distributed_mps(
        circuit,
        world_size=world,
        distributed_executor="torch",
        rank=rank,
        adaptive=True,
        initial_max_bond=1,
        global_error_budget=BUDGET,
        cutoff=0.0,
    )
    # `to_mps` gathers the authoritative owner tensor of every wire, so every
    # rank computes the same global state rather than its own partial replica.
    gathered = result.sharded_state.to_mps()
    summary = result.summary()
    initial = summary["adaptive_initial_plan"]
    final = summary["adaptive_final_plan"]
    return {
        "rank": rank,
        "adaptive_rerun": bool(summary["adaptive_rerun"]),
        "initial_hot_bonds": [int(bond) for bond in initial["hot_bonds"]],
        "initial_suggested_max_bond": int(initial["suggested_max_bond"]),
        "initial_current_max_bond": int(initial["current_max_bond"]),
        "final_budget_satisfied": bool(final["budget_satisfied"]),
        "bond_dims": [int(dim) for dim in gathered.bond_dims],
        "returned_max_bond": int(result.local_state.config.max_bond),
        "fidelity": fidelity(gathered.to_statevector(), reference),
    }


def local_adaptive_control() -> dict[str, object]:
    """The single-process adaptive path, as the sharded path must reproduce it."""
    circuit = fixture(3, ((0, 1),))
    reference = exact_statevector(circuit)
    adaptive = run_mps_adaptive(
        circuit,
        initial_max_bond=1,
        global_error_budget=BUDGET,
    )
    return {
        "hot_bonds": [int(bond) for bond in adaptive.initial_plan.hot_bonds],
        "suggested_max_bond": int(adaptive.initial_plan.suggested_max_bond),
        "rerun": bool(adaptive.rerun),
        "final_budget_satisfied": bool(adaptive.final_plan.budget_satisfied),
        "bond_dims": [int(dim) for dim in adaptive.state.bond_dims],
        "fidelity": fidelity(adaptive.state.to_statevector(), reference),
    }


def check_site_local_records(records: tuple[dict[str, object], ...]) -> list[str]:
    problems: list[str] = []
    measured = sum(float(item["measured_total"]) for item in records)
    if not measured > 0.0:
        return ["the fixture must truncate on the rank that owns the gate"]
    for item in records:
        if item["record_count"] == 0 and float(item["measured_total"]) > 0.0:
            problems.append(
                f"rank {item['rank']} measured discarded weight "
                f"{item['measured_total']} but recorded no truncation record"
            )
        if float(item["recorded_total"]) < float(item["measured_total"]) - 1e-12:
            problems.append(
                f"rank {item['rank']} recorded {item['recorded_total']} of "
                f"{item['measured_total']} measured discarded weight"
            )
        if float(item["measured_total"]) <= 0.0:
            continue
        if not item["hot_bonds"]:
            problems.append(
                f"rank {item['rank']} truncated a bond but named no hot bond"
            )
        if int(item["suggested_max_bond"]) <= int(item["current_max_bond"]):
            problems.append(
                f"rank {item['rank']} truncated a bond but asked to keep "
                f"max_bond={item['suggested_max_bond']}"
            )
    return problems


def check_merged_plan(
    plans: tuple[dict[str, object], ...], control: dict[str, object]
) -> list[str]:
    problems: list[str] = []
    identities = {
        (tuple(item["initial_hot_bonds"]), item["initial_suggested_max_bond"])
        for item in plans
    }
    if len(identities) != 1:
        problems.append(f"ranks act on different adaptive plans: {sorted(identities)}")
    for item in plans:
        if not item["adaptive_rerun"]:
            problems.append(f"rank {item['rank']} never triggered the adaptive rerun")
        if item["initial_hot_bonds"] != [0]:
            problems.append(
                f"rank {item['rank']} named hot bonds {item['initial_hot_bonds']} "
                "instead of the truncated bond [0]"
            )
        if int(item["initial_suggested_max_bond"]) <= int(
            item["initial_current_max_bond"]
        ):
            problems.append(
                f"rank {item['rank']} asked to keep the truncating bond limit"
            )
        if not item["final_budget_satisfied"]:
            problems.append(f"rank {item['rank']} ended outside the global budget")
        if int(item["returned_max_bond"]) != int(item["initial_suggested_max_bond"]):
            problems.append(
                f"rank {item['rank']} reran at max_bond={item['returned_max_bond']} "
                f"instead of the planned {item['initial_suggested_max_bond']}"
            )
        if item["bond_dims"] != control["bond_dims"]:
            problems.append(
                f"rank {item['rank']} returned bond dimensions {item['bond_dims']}, "
                f"but the single-process adaptive run returned {control['bond_dims']}"
            )
        if not float(item["fidelity"]) >= 1.0 - 1e-9:
            problems.append(
                f"rank {item['rank']} reached fidelity {item['fidelity']} against "
                "the exact statevector after the adaptive rerun"
            )
    if not control["rerun"] or not control["final_budget_satisfied"]:
        problems.append(
            "the single-process control must trigger one adaptive rerun that "
            "satisfies the budget; the sharded path is compared against it"
        )
    return problems


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])

    records = all_gather_json(case_site_local_records(rank, world))
    plans = all_gather_json(case_merged_plan(rank, world))
    control = local_adaptive_control()

    problems = check_site_local_records(records) + check_merged_plan(plans, control)
    assert not problems, "rank-sharded MPS adaptive bond planning: " + "; ".join(
        problems
    )

    if rank == 0:
        print(
            json.dumps(
                {
                    "world_size": world,
                    "status": "passed",
                    "site_local_records": list(records),
                    "merged_plan": list(plans),
                    "single_process_control": control,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    destroy_torch_distributed()


if __name__ == "__main__":
    main()

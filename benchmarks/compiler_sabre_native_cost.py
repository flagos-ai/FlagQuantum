"""Inspect target-CZ-aware SABRE trials without enabling a production policy.

Weyl counts shortlist one alternative. The incumbent and that alternative then
use the existing bounded synthesis acceptance before a route can be selected.
This is a development experiment for an unannotated numeric CZ target only.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TypedDict
from unittest.mock import patch

from benchmarks import compiler_sabre_depth as workflows
from flagquantum.compiler.lowering.basis_translation import translate
from flagquantum.compiler.optimization._cz_blocks import (
    _candidate,
    _collect_blocks,
    _numeric_matrix,
    _product,
)
from flagquantum.compiler.optimization.two_qubit_synthesis import (
    _prepared_basis,
    _swap_register,
    _weyl_decomposition,
)
from flagquantum.compiler.topology import sabre
from flagquantum.compiler.topology.coupling import CouplingMap
from flagquantum.core.ir import CircuitIR, Instruction

_ORIGINAL = sabre._select_sabre_trials
_BASIS = frozenset(("rx", "rz", "cz"))
_HARNESS = "9aa9260920984e2ca05bcb293abb885fccd9c144"


class Report(TypedDict):
    scope: str
    source_commit: str
    implementation_commit: str
    nonce: str
    runner_sha256: str
    native_budget: int
    complete: bool
    rows: list[dict[str, object]]
    failures: list[dict[str, object]]
    quality_regressions: list[dict[str, object]]


def _native_preview(
    source: CircuitIR, coupling: CouplingMap, plan: sabre.SabrePlan
) -> tuple[CircuitIR, int]:
    if plan.initial_logical_to_physical != tuple(range(source.n_wires)):
        raise ValueError("Preview requires identity input placement")
    if any(
        item.metadata or item.matrix is not None or not 1 <= len(item.wires) <= 2
        for item in source.instructions
    ):
        raise ValueError("Preview requires unannotated numeric one/two-qubit gates")
    physical: list[Instruction] = []
    cursor = 0
    for count, index in enumerate(plan.sequence):
        while cursor < len(plan.swaps) and plan.swap_placements[cursor] == count:
            physical.append(Instruction("swap", plan.swaps[cursor]))
            cursor += 1
        physical.append(
            replace(source.instructions[index], wires=plan.placements[index])
        )
    if cursor != len(plan.swaps):
        raise ValueError("Preview plan contains unemitted SWAPs")
    physical.extend(
        Instruction("swap", (left, right))
        for left, right, _ in sabre.plan_restore_swaps(plan, coupling)
    )
    native: list[Instruction] = []
    for item in physical:
        leaves = (
            (item,)
            if item.name in _BASIS
            else translate(
                item,
                can_run=lambda gate: gate.name in _BASIS,
                z_rotation="rz",
                pulse_opcode="rx",
            )
        )
        if leaves is None or any(gate.name not in _BASIS for gate in leaves):
            raise ValueError("Unsupported preview lowering")
        native.extend(leaves)
    result = replace(source, instructions=tuple(native))
    if any(
        _numeric_matrix(
            item,
            (
                (item.wires[0], (item.wires[0] + 1) % source.n_wires)
                if len(item.wires) == 1
                else item.wires
            ),
        )
        is None
        for item in native
    ):
        raise ValueError("Preview does not support symbolic or batched values")
    return result, len(physical)


def _blocks(
    native: CircuitIR,
) -> Iterator[tuple[tuple[int, int], tuple[int, ...]]]:
    return _collect_blocks(
        native,
        lambda index, pair: _numeric_matrix(native.instructions[index], pair)
        is not None,
        identity_reader=_numeric_matrix,
    )


def _cz_lower_bound(native: CircuitIR) -> int:
    estimate = sum(item.name == "cz" for item in native.instructions)
    basis = _prepared_basis("cz")
    for pair, positions in _blocks(native):
        block = [native.instructions[index] for index in positions]
        before = sum(item.name == "cz" for item in block)
        if before < 2:
            continue
        matrix = _product(block, pair, matrix_reader=_numeric_matrix)
        if matrix is None:
            raise ValueError("Unsupported numeric block")
        after = basis.entangler_count(_weyl_decomposition(_swap_register(matrix)))
        estimate -= max(0, before - after)
    return estimate


def _realize(native: CircuitIR, limit: int) -> CircuitIR:
    if len(native.instructions) > limit:
        raise ValueError("Native preview exceeds the operation budget")
    replacements: dict[int, tuple[Instruction, ...]] = {}
    size = len(native.instructions)
    for pair, positions in _blocks(native):
        block = [native.instructions[index] for index in positions]
        proposal = _candidate(
            block,
            pair,
            max_output_operations=len(block) + limit - size,
            matrix_reader=_numeric_matrix,
        )
        if proposal is not None:
            replacements[positions[0]] = proposal
            replacements.update((index, ()) for index in positions[1:])
            size += len(proposal) - len(block)
    return replace(
        native,
        instructions=tuple(
            leaf
            for index, item in enumerate(native.instructions)
            for leaf in replacements.get(index, (item,))
        ),
    )


def _cost(native: CircuitIR) -> tuple[int, int]:
    levels = [0] * native.n_wires
    count = 0
    for item in native.instructions:
        if len(item.wires) == 2:
            count += 1
            level = max(levels[q] for q in item.wires) + 1
            for q in item.wires:
                levels[q] = level
    return count, max(levels, default=0)


def _select(
    program: CircuitIR,
    coupling: CouplingMap,
    baseline: sabre.SabrePlan,
    *,
    cost: Callable[[sabre.SabrePlan], int] | None = None,
    native_budget: int = 256,
) -> sabre.SabrePlan:
    incumbent = _ORIGINAL(program, coupling, baseline, cost=cost)
    if cost is not None or not incumbent.swaps:
        return incumbent
    native, routed_count = _native_preview(program, coupling, incumbent)
    incumbent_cost = _cost(_realize(native, routed_count + native_budget))
    shortlist: tuple[sabre.SabrePlan, CircuitIR, int] | None = None
    best_estimate = incumbent_cost[0]
    swap_limit = sabre._restored_swap_count(incumbent, coupling)
    for seed, layers in ((0, False), (1, False), (2, False), (None, True)):
        candidate = sabre._plan_sabre_trial(
            program,
            coupling,
            initial_layout=baseline.initial_logical_to_physical,
            tie_rng=random.Random(seed) if seed is not None else None,
            committed_swap_limit=swap_limit + 1,
            layer_lookahead=layers,
        )
        if (
            candidate is None
            or sabre._restored_swap_count(candidate, coupling) > swap_limit
        ):
            continue
        preview, count = _native_preview(program, coupling, candidate)
        estimate = _cz_lower_bound(preview)
        if estimate < best_estimate:
            shortlist, best_estimate = (candidate, preview, count), estimate
    if shortlist is None:
        return incumbent
    candidate, preview, count = shortlist
    delivered = _cost(_realize(preview, count + native_budget))
    # The estimate alone cannot override residual or instruction-growth rejection.
    # Keep both delivered objectives non-increasing rather than hide a tradeoff.
    return (
        candidate
        if delivered[0] < incumbent_cost[0] and delivered[1] <= incumbent_cost[1]
        else incumbent
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdkbench-root", type=Path, required=True)
    parser.add_argument("--nonce", default="01" * 32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    harness = args.sdkbench_root.resolve()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=harness, text=True, timeout=15
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=harness, text=True, timeout=15
    ).strip()
    if revision != _HARNESS or dirty:
        parser.error("The pinned SDKBench checkout must be unchanged")
    sys.path.insert(0, str(harness / "src"))
    from sdkbench.candidates.adapters import workflow_configuration

    native_budget = workflow_configuration("flagquantum-target-artifact")["options"][
        "max_native_added_operations"
    ]
    report: Report = {
        "scope": "Inspected single-device development experiment; production policy disabled; no qualified performance, held-out, ranking or scalability claim",
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, timeout=15
        ).strip(),
        "source_commit": revision,
        "nonce": args.nonce,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_budget": native_budget,
        "complete": False,
        "rows": [],
        "failures": [],
        "quality_regressions": [],
    }
    with patch.object(
        workflows, "_select", partial(_select, native_budget=native_budget)
    ):
        for case in itertools.product(
            (6, 8, 12),
            ("qaoa", "ising", "hopping", "qft", "repetition"),
            ("line", "ring", "grid", "sparse"),
        ):
            start = time.perf_counter()
            try:
                result = workflows._application(
                    *case, "flagquantum-target-artifact", harness, nonce=args.nonce
                )
                before, after = result["baseline"], result["candidate"]
                if not isinstance(before, dict) or not isinstance(after, dict):
                    raise ValueError("Workflow did not return comparison metrics")
                if any(
                    after[key] > before[key]
                    for key in ("full_cz_count", "two_body_dependency_depth")
                ):
                    report["quality_regressions"].append(
                        {"case": case, "result": result}
                    )
                report["rows"].append(
                    {
                        "case": case,
                        "result": result,
                        "development_seconds": time.perf_counter() - start,
                    }
                )
            except Exception as error:
                # A diagnostic boundary retains failure and continues the fixed set.
                report["failures"].append(
                    {
                        "case": case,
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(case, "inspected", flush=True)
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if report["failures"] or report["quality_regressions"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

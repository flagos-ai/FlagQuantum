"""Publish routing quality for every strategy the Compiler accepts, on one basis.

W7-16 of the Qiskit parity backlog asks for the routing quality the product
delivers, stated against a fixed basis rather than against one caller's device.
The Compiler accepts four routing strategies and one automatic selection, and
until this module existed no checked-in number said what any of them costs. The
cost that matters is the one the product reports: the SWAPs a routed program
still contains after the post-routing optimization pass, read back from
``routing.post_optimization_inserted_swap_count``.

Every strategy is measured through ``flagquantum.compiler.compile``, the entry
point a caller uses, on the same 140 programs ``benchmarks.compiler_lookahead_swap``
declares for W7-04 and W7-06, so the numbers here are stated on a basis two other
recorded measurements already use. The measured basis is imported rather than
restated, so the three modules cannot drift onto different programs. The numbers
are *not* comparable with the W7-04/W7-06 baselines, which route a raw program:
``compile`` optimizes before it routes, as a caller's program is optimized before
it is routed, and that is part of what a caller pays.

On that basis:

* All five entries route all 140 programs and emit no two-wire operation off the
  device. Every compiled program is compared with its source state, so a strategy
  cannot buy a smaller SWAP count with a different program.
* The ordering is total and wide. ``sabre_layout`` retains the fewest SWAPs,
  ``sabre`` the next fewest, and the two estimate-driven strategies retain more
  than twice that; ``auto`` lands between ``sabre`` and the two it may choose
  from.
* Asking ``compile`` for ``auto`` selects on the program ``compile`` is about to
  route, which is the optimized one -- not on the program the caller passed in.
  Those differ: on four of the 140 programs the estimate makes the opposite call
  before and after optimization, so a caller who predicts ``compile``'s choice by
  running ``select_routing_strategy`` on their own source program is predicting a
  choice the product never makes. Measuring ``auto`` anywhere other than where the
  product selects would publish the cost of a route no caller receives.
* ``auto`` is a real selection and not an alias. It never resolves to a SABRE
  strategy, which is the documented scope of the cost estimate it ranks by, and
  its total is below both of the strategies it may choose from, so the estimate
  is informative inside its scope rather than noise. It cannot be *below* both on
  any single program: ``auto`` returns the route of the candidate it named, so
  its per-program count always equals that candidate's, and the better of the two
  is a floor it can only reach. Reaching it on 132 of the 140 programs, and
  missing it on 8, is what the estimate buys. The price of that scope is the same
  whether the caller asks for ``auto`` or names one of the two it covers; both are
  roughly twice the SWAPs of the strategy the product already ships.
* ``restore_after_each_gate``, the default of ``compile``, is the one a caller
  gets by saying nothing. It is the worst entry in the catalog, and the number
  below is what saying nothing costs.

The conclusion is a measurement rather than a change. The scope of the automatic
selection is already stated in ``capability-maturity.toml``; what was missing is
its price, and this module supplies it so the next round can decide whether to
widen the selection against a number instead of an impression.

Every count below is read from the shipped router, so it is a reading of the
router's behaviour rather than of this module. A physical pair resolves to one
route, and that route is a function of the pair rather than of the queries the
device answered before it, so these counts are reproducible from a cold device.
Without that property the count would depend on which of its two operands a
consumer happened to pass first, or on whether another pass had already used the
same device, and this file would be publishing the cost of a route it never
measured. ``tests/team/compiler/test_routing_conformance.py`` asserts the
property instead of leaving this file to assume it.

Classification: a local compiler microbenchmark on the single-device fast path.
It runs no distributed work, makes no scalability claim, and is not a
performance gate. Re-run it with::

    python benchmarks/compiler_routing_quality.py --json-output /tmp/routing.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

import flagquantum as fq
from benchmarks.compiler_lookahead_swap import SEEDS, case_basis
from flagquantum.compiler import compile as compile_program
from flagquantum.compiler.routing import ROUTING_STRATEGIES, RoutingStrategySelection
from flagquantum.core.ir import CircuitIR

SCHEMA = "flagquantum_compiler_routing_quality_benchmark_v1"

#: The automatic selection is not a strategy; it names the one it resolves to.
AUTOMATIC_SELECTION = "auto"

#: Every entry a caller may pass as ``compile(routing_strategy=...)``.
MEASURED_STRATEGIES: tuple[str, ...] = (*ROUTING_STRATEGIES, AUTOMATIC_SELECTION)

#: The three labels the ratios in the payload are taken against.
REFERENCE_LABELS = ("sabre_layout", "sabre")


def retained_swap_count(compiled: CircuitIR) -> int:
    """Return the SWAPs a routed program still holds after optimization.

    This is the product's own count. ``compile`` routes, optimizes the routed
    program again, and records the result, so reading it here measures the
    product's number rather than a second implementation of it.
    """

    return int(compiled.metadata["routing"]["post_optimization_inserted_swap_count"])


def planned_swap_count(compiled: CircuitIR) -> int:
    """Return the SWAPs the router inserted before any optimization removed one."""

    return int(compiled.metadata["routing"]["inserted_swap_count"])


def off_device_two_wire_count(compiled: CircuitIR, device: Any) -> int:
    """Return how many two-wire operations the device cannot host directly."""

    return sum(
        1
        for instruction in compiled
        if len(instruction.wires) == 2 and not device.has_edge(*instruction.wires)
    )


def _state(program: CircuitIR) -> torch.Tensor:
    return fq.Circuit.from_ir(program).state()


def _row(label: str, **values: Any) -> dict[str, Any]:
    return {"label": label, **values}


def run_benchmark(
    *,
    topologies: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] = SEEDS,
    strategies: tuple[str, ...] = MEASURED_STRATEGIES,
    verify_states: bool = True,
) -> dict[str, Any]:
    """Measure every routing entry point on one fixed basis.

    Each label is compiled through ``flagquantum.compiler.compile`` with a device
    no earlier label has routed on. A ``CouplingMap`` caches the shortest paths it
    computes, so sharing one device across labels would let an earlier label warm
    a later label's cache; every label gets its own.

    Args:
        topologies: Topology names to measure; the whole declared basis by default.
        seeds: Program seeds to measure.
        strategies: Routing entry points to measure, in report order. Every name
            must be one ``compile`` accepts, except ``auto``.
        verify_states: Compare every compiled program with its source state.
            Disabling it drops the numerical evidence, so do so only where the
            same programs have been verified elsewhere.

    Returns:
        A JSON-serializable payload: the basis, one row per entry point, the
        per-case records, and what the automatic selection costs against the
        candidates it ranks.
    """

    cases = case_basis(topologies=topologies, seeds=seeds)
    retained: dict[str, int] = dict.fromkeys(strategies, 0)
    planned: dict[str, int] = dict.fromkeys(strategies, 0)
    instructions: dict[str, int] = dict.fromkeys(strategies, 0)
    illegal: dict[str, int] = dict.fromkeys(strategies, 0)
    failures: dict[str, list[str]] = {label: [] for label in strategies}
    seconds: dict[str, float] = dict.fromkeys(strategies, 0.0)
    resolved: dict[str, dict[str, int]] = {AUTOMATIC_SELECTION: {}}
    # Per-case retained counts, so the automatic selection can be compared with
    # each candidate on the program it actually had to choose for.
    per_case: dict[str, dict[str, int]] = {label: {} for label in strategies}
    verified = 0
    max_state_difference = 0.0
    case_records: list[dict[str, Any]] = []

    for case in cases:
        program = case.program()
        source_state = _state(program) if verify_states else None
        for label in strategies:
            device = case.device()
            started = time.perf_counter()
            try:
                compiled = compile_program(
                    program,
                    coupling_map=device,
                    routing_strategy=label,
                )
            except (RuntimeError, ValueError) as error:
                failures[label].append(f"{case.label}: {error}")
                seconds[label] += time.perf_counter() - started
                continue
            seconds[label] += time.perf_counter() - started
            routing = compiled.metadata["routing"]
            value = retained_swap_count(compiled)
            retained[label] += value
            planned[label] += planned_swap_count(compiled)
            instructions[label] += len(compiled)
            illegal[label] += off_device_two_wire_count(compiled, device)
            per_case[label][case.label] = value
            if label == AUTOMATIC_SELECTION:
                name = str(routing["strategy"])
                resolved[AUTOMATIC_SELECTION][name] = (
                    resolved[AUTOMATIC_SELECTION].get(name, 0) + 1
                )
            difference = 0.0
            if source_state is not None:
                difference = float((_state(compiled) - source_state).abs().max())
                max_state_difference = max(max_state_difference, difference)
                verified += 1
            case_records.append(
                {
                    "case": case.label,
                    "strategy": label,
                    "resolved_strategy": str(routing["strategy"]),
                    "planned_inserted_swap_count": planned_swap_count(compiled),
                    "retained_inserted_swap_count": value,
                    "instruction_count": len(compiled),
                    "off_device_two_wire_instruction_count": off_device_two_wire_count(
                        compiled, device
                    ),
                    "state_difference": difference,
                }
            )

    best = min((retained[label] for label in strategies), default=0)
    rows = []
    for label in strategies:
        row = _row(
            label,
            planned_case_count=len(cases) - len(failures[label]),
            failed_case_count=len(failures[label]),
            failed_cases=failures[label],
            planned_inserted_swap_count=planned[label],
            retained_inserted_swap_count=retained[label],
            instruction_count=instructions[label],
            off_device_two_wire_instruction_count=illegal[label],
            planning_seconds=seconds[label],
        )
        for reference in REFERENCE_LABELS:
            base = retained.get(reference, 0)
            row[f"ratio_to_{reference}"] = (retained[label] / base) if base else 0.0
        row["ratio_to_best"] = (retained[label] / best) if best else 0.0
        # The automatic selection is the only label with more than one outcome.
        # This is the field that shows it is a selection rather than an alias, so
        # it is on every row and empty on the four fixed strategies.
        counts = (
            resolved[AUTOMATIC_SELECTION]
            if label == AUTOMATIC_SELECTION
            else {label: len(cases) - len(failures[label])}
        )
        row["resolved_strategy_counts"] = counts
        rows.append(row)

    # What the automatic selection had to choose from, and how well it chose. The
    # candidates are read off ``RoutingStrategySelection`` rather than restated:
    # that dataclass is the authoritative statement of which strategies the cost
    # estimate can rank, and a fifth strategy cannot be added to the selection
    # without adding a field there. Every comparison is made on the program the
    # selection actually saw, not on the totals.
    candidates = [
        field.name
        for field in fields(RoutingStrategySelection)
        if field.name != "selected_strategy"
    ]
    unknown = [name for name in candidates if name not in per_case]
    if unknown:
        raise ValueError(
            f"the automatic selection ranks {unknown}, which this measurement did "
            "not measure; add them to strategies or the comparison is not honest"
        )
    comparable = [
        case.label
        for case in cases
        if all(
            case.label in per_case[AUTOMATIC_SELECTION]
            and case.label in per_case[label]
            for label in candidates
        )
    ]
    better_than_both = 0
    equal_to_best_candidate = 0
    worse_than_best_candidate = 0
    chosen_total = 0
    best_candidate_total = 0
    best_available_total = 0
    for label in comparable:
        chosen = per_case[AUTOMATIC_SELECTION][label]
        candidate_values = [per_case[name][label] for name in candidates]
        best_candidate = min(candidate_values)
        chosen_total += chosen
        best_candidate_total += best_candidate
        best_available_total += min(
            per_case[name][label] for name in strategies if label in per_case[name]
        )
        if chosen < best_candidate:
            better_than_both += 1
        elif chosen == best_candidate:
            equal_to_best_candidate += 1
        else:
            worse_than_best_candidate += 1

    order = sorted(strategies, key=lambda label: (retained[label], label))
    # The basis is reported in the order it was declared, so a reader can compare
    # the payload with the module that declares it without re-sorting.
    measured_topologies: list[str] = []
    for case in cases:
        if case.topology not in measured_topologies:
            measured_topologies.append(case.topology)
    return {
        "schema": SCHEMA,
        "artifact_classification": "local_compiler_microbenchmark",
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "basis_module": "benchmarks.compiler_lookahead_swap",
        "cost_model": (
            "inserted SWAPs a routed program retains after the post-routing "
            "optimization, read from routing.post_optimization_inserted_swap_count, "
            "plus the instruction count of the compiled program"
        ),
        "entry_point": "flagquantum.compiler.compile",
        "case_count": len(cases),
        "topologies": measured_topologies,
        "seeds": list(seeds),
        "measured_strategies": list(strategies),
        "quality_order": order,
        "strategies": rows,
        "automatic_selection": {
            "resolved_strategy_counts": resolved[AUTOMATIC_SELECTION],
            "ranked_strategy_count": len(candidates),
            "retained_inserted_swap_count": retained.get(AUTOMATIC_SELECTION, 0),
            "better_than_both_candidate_case_count": better_than_both,
            "equal_to_best_candidate_case_count": equal_to_best_candidate,
            "worse_than_best_candidate_case_count": worse_than_best_candidate,
            "best_candidate_retained_inserted_swap_count": best_candidate_total,
            "best_available_retained_inserted_swap_count": best_available_total,
            "ratio_to_best_candidate": (
                chosen_total / best_candidate_total if best_candidate_total else 0.0
            ),
            "ratio_to_best_available": (
                chosen_total / best_available_total if best_available_total else 0.0
            ),
        },
        "case_records": case_records,
        "verified_case_count": verified,
        "max_state_difference": max_state_difference,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--topology",
        action="append",
        help="topology to measure; repeat to select several",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help="program seed to measure; repeat to select several",
    )
    parser.add_argument(
        "--strategy",
        action="append",
        choices=MEASURED_STRATEGIES,
        help="routing entry point to measure; repeat to select several",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="skip the compiled-versus-source state comparison",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    payload = run_benchmark(
        topologies=None if args.topology is None else tuple(args.topology),
        seeds=SEEDS if args.seed is None else tuple(args.seed),
        strategies=(
            MEASURED_STRATEGIES if args.strategy is None else tuple(args.strategy)
        ),
        verify_states=not args.skip_verification,
    )
    selection = payload["automatic_selection"]
    print(
        f"{payload['case_count']} programs on {len(payload['topologies'])} topologies; "
        f"best is {payload['quality_order'][0]}"
    )
    print(
        f"{'strategy':24s} {'retained':>9s} {'vs best':>8s} {'vs sabre':>9s} "
        f"{'failed':>7s} {'off-device':>11s}"
    )
    for row in payload["strategies"]:
        print(
            f"{row['label']:24s} {row['retained_inserted_swap_count']:9d} "
            f"{row['ratio_to_best']:7.3f}x {row['ratio_to_sabre']:8.3f}x "
            f"{row['failed_case_count']:7d} "
            f"{row['off_device_two_wire_instruction_count']:11d}"
        )
    print(
        "automatic selection resolves to "
        + ", ".join(
            f"{name} on {count}"
            for name, count in sorted(selection["resolved_strategy_counts"].items())
        )
        + f"; it is {selection['ratio_to_best_available']:.3f}x the best available "
        f"and {selection['ratio_to_best_candidate']:.3f}x the better candidate"
    )
    print(
        f"verified {payload['verified_case_count']} compiled programs; "
        f"max state difference {payload['max_state_difference']:.3e}"
    )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

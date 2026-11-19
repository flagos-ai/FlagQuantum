"""Counterbalanced development probe of SABRE scoring, not SDK comparison evidence."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import types
from dataclasses import asdict
from pathlib import Path

from flagquantum.compiler import CouplingMap
from flagquantum.compiler.topology import sabre
from flagquantum.core.ir import CircuitIR, Instruction

BASELINE = "1e2fbd3f45687afc2ba0e057f2f0f01c67a9e53e"  # Integrated main before the relative-score adaptation.


def main() -> None:
    source = subprocess.run(
        ["git", "show", f"{BASELINE}:flagquantum/compiler/topology/sabre.py"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout
    baseline = types.ModuleType("flagquantum.compiler.topology._scoring_baseline")
    baseline.__package__ = "flagquantum.compiler.topology"
    sys.modules[baseline.__name__] = baseline
    exec(compile(source, "<retained-sabre-baseline>", "exec"), baseline.__dict__)
    rows = []
    for n in (6, 12, 16, 24, 32, 64):
        for seed in range(3):
            rng = random.Random(seed)
            instructions: list[Instruction] = []
            for _ in range(4):
                order = list(range(n))
                rng.shuffle(order)
                instructions.extend(
                    Instruction("cx", (order[i], order[i + 1]))
                    for i in range(0, n - 1, 2)
                )
            program = CircuitIR(n, tuple(instructions))
            coupling = CouplingMap.line(n)
            times: dict[str, list[float]] = {"baseline": [], "relative": []}
            plans = {}
            for block in range(6):
                labels = (
                    ("baseline", "relative")
                    if block % 2 == 0
                    else ("relative", "baseline")
                )
                for label in labels:
                    impl = baseline if label == "baseline" else sabre
                    start = time.perf_counter()
                    plan = impl.plan_sabre_swaps(program, coupling)
                    elapsed = time.perf_counter() - start
                    if block:
                        times[label].append(elapsed)
                    plans[label] = plan
                    assert sorted(plan.sequence) == list(range(len(instructions)))
                    assert all(
                        coupling.has_edge(*plan.placements[i]) for i in plan.sequence
                    )
            # Dataclass types differ between loaded modules; compare their fields.
            same = asdict(plans["baseline"]) == asdict(plans["relative"])
            rows.append(
                {
                    "n_wires": n,
                    "seed": seed,
                    "source_operations": len(instructions),
                    "identical_plans": same,
                    "measurements": {
                        label: {
                            "seconds": values,
                            "median_seconds": statistics.median(values),
                            "forward_swaps": len(plans[label].swaps),
                        }
                        for label, values in times.items()
                    },
                }
            )
    print(
        json.dumps(
            {
                "evidence_class": "development",
                "distribution_semantics": "single_device_fast_path",
                "scalability_claim_allowed": False,
                "python": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "cpu_count": os.cpu_count(),
                "baseline_revision": BASELINE,
                "baseline_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "candidate_source_sha256": hashlib.sha256(
                    Path(sabre.__file__).read_bytes()
                ).hexdigest(),
                "method": "line topology; four shuffled matchings; one warmup and five counterbalanced samples; planning only, no native lowering",
                "rows": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Run the internal #1131 estimate, or replay its JSON without external services."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import flagquantum as fq
from flagquantum.algorithms._fault_tolerant_execution import (
    _estimate_execution,
    _MachineAssumptions,
)
from flagquantum.algorithms.logical_resources import SURFACE_CODE_MODEL
from flagquantum.core.ir import CircuitIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay", type=Path, help="Replay a previously emitted JSON report"
    )
    args = parser.parse_args()
    if args.replay:
        payload = json.loads(args.replay.read_text(encoding="utf-8"))
        assumptions = payload["assumptions"]
        assumptions["operation_durations"] = tuple(
            tuple(pair) for pair in assumptions["operation_durations"]
        )
        machine = _MachineAssumptions(**assumptions)
        program = CircuitIR.from_dict(payload["program"])
    else:
        # Illustrative assumptions chosen for arithmetic, not device calibration.
        machine = _MachineAssumptions(
            name="illustrative_two_patch_machine",
            time_unit="illustrative_tick",
            operation_durations=(("h", 1), ("t", 2), ("tdg", 2), ("cx", 3)),
            code_model=SURFACE_CODE_MODEL,
            code_distance=3,
            logical_data_capacity=2,
            factory_count=1,
            factory_physical_qubits=100,
            factory_startup=0,
            factory_period=4,
            inventory_capacity=1,
            initial_inventory=0,
            storage_physical_qubits_per_state=17,
            physical_qubit_budget=151,
        )
        program = fq.Circuit(2).h(0).t(0).t(1).cx(0, 1).tdg(1).to_ir()
    report = _estimate_execution(program, machine)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

"""Audit matched FlagQuantum and TensorCircuit-NG ADAPT-VQE result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("flagquantum", type=Path)
    parser.add_argument("tensorcircuit_ng", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    fq = json.loads(args.flagquantum.read_text(encoding="utf-8"))
    tcng = json.loads(args.tensorcircuit_ng.read_text(encoding="utf-8"))
    contract_fields = (
        "workload",
        "rows",
        "cols",
        "qubits",
        "cycles",
        "extra_entanglers",
        "initial_state",
        "dtype",
        "parameter_dtype",
        "operator_pool",
        "operator_pool_size",
        "hamiltonian_terms",
        "adapt_iterations_requested",
        "optimization_steps_per_iteration",
        "learning_rate",
    )
    mismatches = [field for field in contract_fields if fq[field] != tcng[field]]
    if mismatches:
        raise ValueError(f"workload contract mismatch: {mismatches}")
    if fq["selected_pool_indices"] != tcng["selected_pool_indices"]:
        raise ValueError("selected operator trajectories do not match")
    if len(fq["iterations"]) != len(tcng["iterations"]):
        raise ValueError("completed ADAPT iteration counts do not match")
    if fq.get("planner") != "cotengra" or tcng.get("planner") != "cotengra":
        raise ValueError("fair comparison requires Cotengra on both implementations")
    planner_fields = ("path_repeats", "path_minimize")
    planner_mismatches = [
        field for field in planner_fields if fq.get(field) != tcng.get(field)
    ]
    if planner_mismatches:
        raise ValueError(f"Cotengra contract mismatch: {planner_mismatches}")
    fq_target_elements = (
        None
        if fq.get("max_intermediate_bytes") is None
        else fq["max_intermediate_bytes"] // 16
    )
    if fq_target_elements != tcng.get("target_peak_elements"):
        raise ValueError("Cotengra target intermediate size does not match")

    gradient_errors = []
    history_errors = []
    for fq_iteration, tcng_iteration in zip(fq["iterations"], tcng["iterations"]):
        if len(fq_iteration["pool_gradients"]) != len(tcng_iteration["pool_gradients"]):
            raise ValueError("pool-gradient vector lengths do not match")
        if len(fq_iteration["optimization_history"]) != len(
            tcng_iteration["optimization_history"]
        ):
            raise ValueError("optimizer-history lengths do not match")
        gradient_errors.extend(
            abs(left - right)
            for left, right in zip(
                fq_iteration["pool_gradients"], tcng_iteration["pool_gradients"]
            )
        )
        history_errors.extend(
            abs(left - right)
            for left, right in zip(
                fq_iteration["optimization_history"],
                tcng_iteration["optimization_history"],
            )
        )
    audit = {
        "schema_version": 1,
        "status": "pass",
        "flagquantum_result": str(args.flagquantum),
        "tensorcircuit_ng_result": str(args.tensorcircuit_ng),
        "qubits": fq["qubits"],
        "screening_method": fq.get("screening_method", "append"),
        "selected_pool_indices": fq["selected_pool_indices"],
        "initial_energy_absolute_error": abs(
            fq["initial_energy"] - tcng["initial_energy"]
        ),
        "final_energy_absolute_error": abs(fq["final_energy"] - tcng["final_energy"]),
        "max_pool_gradient_absolute_error": max(gradient_errors, default=0.0),
        "max_optimization_history_absolute_error": max(history_errors, default=0.0),
        "flagquantum_seconds": fq["execution_seconds"],
        "tensorcircuit_ng_seconds": tcng["execution_seconds"],
        "flagquantum_speedup": tcng["execution_seconds"] / fq["execution_seconds"],
        "performance_comparison_authorized": False,
        "performance_comparison_blocker": (
            "matching Cotengra policy does not prove an identical contraction tree"
        ),
        "tolerance": args.tolerance,
    }
    numerical_fields = (
        "initial_energy_absolute_error",
        "final_energy_absolute_error",
        "max_pool_gradient_absolute_error",
        "max_optimization_history_absolute_error",
    )
    if "initial_product_state_energy" in fq or "initial_product_state_energy" in tcng:
        if "initial_product_state_energy" not in fq or "initial_product_state_energy" not in tcng:
            raise ValueError("initial product-state reference is missing on one side")
        audit["initial_product_state_energy_absolute_error"] = abs(
            fq["initial_product_state_energy"] - tcng["initial_product_state_energy"]
        )
        numerical_fields += ("initial_product_state_energy_absolute_error",)
    exceeded = tuple(
        field for field in numerical_fields if audit[field] > args.tolerance
    )
    if exceeded:
        audit["status"] = "fail"
        audit["exceeded_fields"] = exceeded
    rendered = json.dumps(audit, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if exceeded:
        raise ValueError(f"numerical audit exceeded tolerance: {exceeded}")


if __name__ == "__main__":
    main()

"""Merge a separately measured JAX leg into a VQE comparison JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--jax", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.base.read_text())
    jax_leg = json.loads(args.jax.read_text())
    for key in (
        "n_wires",
        "local_rx_rz_depth",
        "parameters",
        "iterations",
        "learning_rate",
    ):
        if base[key] != jax_leg[key]:
            raise ValueError(f"incompatible {key}: {base[key]} != {jax_leg[key]}")

    triton = base["triton"]
    jax_result = jax_leg["jax"]
    base["jax"] = jax_result
    base["jax_runtime"] = jax_leg["jax_runtime"]
    base["jax_measurement"] = {
        "device": jax_leg["device"],
        "physical_gpu": jax_leg["physical_gpu"],
        "separate_process": True,
        "xla_python_client_preallocate": False,
    }
    base.setdefault(
        "timing_boundary",
        {
            "training_curves": "post_warmup_steady_state",
            "compile_and_first_execution_excluded": True,
            "startup_seconds": None,
            "startup_measurement_note": (
                "Not recorded in this run; no compile cost is inferred after the fact."
            ),
        },
    )
    base["runtime_ratio_jax_over_triton"] = (
        jax_result["cumulative_seconds"][-1]
        / triton["cumulative_seconds"][-1]
    )
    base["cumulative_jax_over_triton_factor"] = [
        jax_time / triton_time
        for jax_time, triton_time in zip(
            jax_result["cumulative_seconds"], triton["cumulative_seconds"]
        )
    ]
    base["final_energy_abs_delta_jax_triton"] = abs(
        jax_result["energies"][-1] - triton["energies"][-1]
    )
    args.output.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

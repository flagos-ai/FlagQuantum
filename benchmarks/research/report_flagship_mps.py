"""Render a Markdown report from a FlagQuantum flagship MPS benchmark JSON.

Example
-------
    python benchmarks/research/report_flagship_mps.py \
        --input benchmarks/results/flagship_mps_1000q_dimer_cpu_jax.json \
        --output benchmarks/results/flagship_mps_1000q_dimer_cpu_jax.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _row(label: str, value: Any) -> str:
    return f"| {label} | {_fmt(value)} |"


def render_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# FlagQuantum Flagship MPS Benchmark")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---:|")
    lines.append(_row("Benchmark", payload.get("benchmark")))
    lines.append(_row("Device", payload.get("device")))
    lines.append(_row("Distribution semantics", payload.get("distribution_semantics")))
    lines.append(_row("Scalability claim allowed", payload.get("scalability_claim_allowed")))
    config = payload.get("configuration", {})
    lines.append(_row("Cases", config.get("cases")))
    lines.append(_row("JIT", config.get("jit")))
    lines.append(_row("Steps", config.get("steps")))
    lines.append(_row("Warmup", config.get("warmup")))
    lines.append(_row("Iters", config.get("iters")))
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append(
        "| Case | Kind | Qubits | Params | First loss+grad s | Steady loss+grad s | "
        "Loss | Grad norm | Claim boundary |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for item in payload.get("results", []):
        first = item.get("first_loss_grad", {})
        steady = item.get("steady_loss_grad", {})
        loss = steady.get("loss", first.get("loss"))
        grad_norm = steady.get("grad_norm", first.get("grad_norm"))
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(item.get("case")),
                    _fmt(item.get("kind")),
                    _fmt(item.get("n_wires")),
                    _fmt(item.get("parameters")),
                    _fmt(first.get("seconds")),
                    _fmt(steady.get("avg_seconds")),
                    _fmt(loss),
                    _fmt(grad_norm),
                    _fmt(item.get("claim_boundary")),
                ]
            )
            + " |"
        )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(payload.get("recommended_claim", "n/a"))
    lines.append("")
    lines.append("## Do Not Claim")
    lines.append("")
    lines.append(payload.get("do_not_claim", "n/a"))
    lines.append("")
    lines.append("## Audit")
    lines.append("")
    if payload.get("distribution_semantics") == "single_device_fast_path" and not payload.get(
        "scalability_claim_allowed", True
    ):
        lines.append("This benchmark is a single-device fast-path result and is not a distributed scalability claim.")
    else:
        lines.append("Review required: distribution metadata does not match the expected single-device fast-path pattern.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_report(payload), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Summarize retained CPU cases without changing their raw evidence."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def render_report(directory: Path) -> str:
    """Require a consistent producer/protocol before reporting completed cases."""
    freeze = json.loads((directory / "freeze.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if freeze["schema"] == "flagquantum.qng_hva_ablation.v1":
        from benchmarks.qng_hva_ablation import render_report as render_qng_report

        return render_qng_report(directory)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    failures = []
    for case in summary["cases"]:
        if case["status"] != "completed":
            failures.append(f"- {case['case']}: {case['status']}")
            continue
        result = json.loads((directory / case["result"]).read_text())
        if result["source"] != freeze["source"]:
            raise ValueError(f"producer identity changed: {case['case']}")
        if result["protocol"] != freeze["protocol"]:
            raise ValueError(f"protocol changed: {case['case']}")
        if not summary["smoke"] and result["data_sha256"] != freeze["data_sha256"]:
            raise ValueError(f"dataset changed: {case['case']}")
        if result["status"] != "completed" or not result["parity"]["passed"]:
            raise ValueError(f"completed case has no valid parity: {case['case']}")
        key = result["task"], result["engine"]["engine"]
        groups.setdefault(key, []).append(result)
    lines = [
        "# CPU training comparison",
        "",
        f"Producer: `{freeze['source']['code_version']}`.",
        f"Protocol SHA-256: `{freeze['source']['protocol_sha256']}`.",
        f"Smoke: `{summary['smoke']}`. This is CPU comparison evidence, not release or scalability evidence.",
        "",
        "Training time includes optimizer updates and per-update validation; imports, oracle checks and replay are outside that timer.",
        "Peak RSS includes imports, oracle checks, warmup and replay. These small, serial-example workloads are not a ranking of the fastest available configurations.",
        "",
        "| Task | Engine | Completed | Target reached | Training seconds, median | Final task score, median [min, max] | Peak process RSS MiB, median |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for (task, engine), results in sorted(groups.items()):
        scores = [
            r["assessment"][
                (
                    "test_accuracy"
                    if task == "classifier"
                    else "final_relative_energy_error"
                )
            ]
            for r in results
        ]
        timing = statistics.median(r["training_seconds"] for r in results)
        memory = statistics.median(r["process_peak_rss_bytes"] for r in results) / 2**20
        reached = sum(r["time_to_target_seconds"] is not None for r in results)
        lines.append(
            f"| {task} | {engine} | {len(results)} | {reached} | {timing:.4f} | {statistics.median(scores):.6g} [{min(scores):.6g}, {max(scores):.6g}] | {memory:.1f} |"
        )
    lines += [
        "",
        "Classifier score is held-out test accuracy after fixed-step training. Its time-to-target uses validation accuracy >= 0.8. Heisenberg score is final relative energy error; its target is <= 0.01.",
        "The six-parameter quadratic classical control is well matched to the XOR label rule. Its zero initialization is identical across paired runs, so those runs do not establish seed-dependent model variability.",
        "",
        "## Per-seed outcomes",
        "",
        "| Task | Engine | Seed | Final score | Training seconds | Warmup seconds | First value/gradient seconds | Time to target seconds | Replay max error |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for (task, engine), results in sorted(groups.items()):
        for result in sorted(results, key=lambda r: r["seed"]):
            score = result["assessment"][
                (
                    "test_accuracy"
                    if task == "classifier"
                    else "final_relative_energy_error"
                )
            ]
            target = result["time_to_target_seconds"]
            replay = max(
                result[k]
                for k in (
                    "checkpoint_replay_max_parameter_error",
                    "checkpoint_replay_max_gradient_error",
                    "checkpoint_replay_max_optimizer_error",
                )
            )
            target_text = "not reached" if target is None else f"{target:.4f}"
            lines.append(
                f"| {task} | {engine} | {result['seed']} | {score:.8g} | {result['training_seconds']:.4f} | {result['warmup_seconds']:.4f} | {result['parity']['first_value_grad_seconds']:.4f} | {target_text} | {replay:.3g} |"
            )
    lines += [
        "",
        "## Total cost including first call",
        "",
        "Measured total includes setup, first call, warmup, initial validation (when recorded), training and final assessment. Training already includes checkpoint writing. Oracle and recovery are excluded from measured total. Process wall includes interpreter startup, imports, oracle and recovery; it is an application diagnostic, not a pure simulator ranking.",
        "",
        "| Task | Engine | Seed | Per-step update s, median | Total measured s | Total time to target s | Fresh-process wall s |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    process_wall = {
        case["case"]: case.get("process_wall_seconds") for case in summary["cases"]
    }
    for (task, engine), results in sorted(groups.items()):
        for result in sorted(results, key=lambda r: r["seed"]):
            target = result.get("time_to_target_total_seconds")
            target_text = (
                "not recorded by historical producer"
                if "time_to_target_total_seconds" not in result
                else "not reached" if target is None else f"{target:.6f}"
            )
            wall = process_wall.get(f"{task}-{engine}-{result['seed']}")
            wall_text = "not recorded" if wall is None else f"{wall:.6f}"
            step = statistics.median(
                point["update_seconds"] for point in result["trace"]
            )
            lines.append(
                f"| {task} | {engine} | {result['seed']} | {step:.6f} | {result['total_measured_seconds']:.6f} | {target_text} | {wall_text} |"
            )
    lines += ["", "## Retained failures and limits", ""]
    lines += failures or ["No case failed or timed out in this directory."]
    lines += [
        "",
        "QNG algorithm/fusion ablation, batched/vmapped throughput, larger-task quality, repeated timing campaigns, GPU runs and AI-assisted error correction are outside this artifact. No quantum advantage or generalization advantage is established.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    (args.directory / "REPORT.md").write_text(render_report(args.directory))


if __name__ == "__main__":
    main()

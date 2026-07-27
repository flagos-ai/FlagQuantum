"""Plot three-seed, 100-step end-to-end QNG scaling and result parity."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"reference_qng": "#d24b40", "triton_qng": "#238b45"}
LABELS = {"reference_qng": "Reference block-QNG", "triton_qng": "Triton block-QNG"}


def experiment(payload, backend):
    return next(item for item in payload["experiments"] if item["backend"] == backend)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    grouped = defaultdict(list)
    for path in args.inputs:
        payload = json.loads(path.read_text())
        assert payload["steps"] == 100 and payload["depth"] == 2
        grouped[payload["n_wires"]].append(payload)
    n_values = sorted(grouped)
    assert all(len(grouped[n]) == 3 for n in n_values)

    times, errors = {}, {}
    for backend in ("reference_qng", "triton_qng"):
        times[backend] = [[experiment(p, backend)["trace"][-1]["wall_time_seconds"] for p in grouped[n]] for n in n_values]
        errors[backend] = [[experiment(p, backend)["final_relative_error"] for p in grouped[n]] for n in n_values]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    width = 0.34
    medians = {b: np.median(times[b], axis=1) for b in times}
    for offset, backend in ((-width / 2, "reference_qng"), (width / 2, "triton_qng")):
        lo = medians[backend] - np.min(times[backend], axis=1)
        hi = np.max(times[backend], axis=1) - medians[backend]
        axes[0].bar(np.array(n_values) + offset, medians[backend], width, yerr=[lo, hi], capsize=3,
                    label=LABELS[backend], color=COLORS[backend])
    speedups = medians["reference_qng"] / medians["triton_qng"]
    for n, ref_time, speedup in zip(n_values, medians["reference_qng"], speedups):
        axes[0].annotate(f"{speedup:.2f}x", (n - width / 2, ref_time), xytext=(0, 5),
                         textcoords="offset points", ha="center", fontsize=9)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Qubits N (depth=2)")
    axes[0].set_ylabel("Wall time for 100 complete steps (s)")
    axes[0].set_title("Long-run end-to-end time")
    axes[0].grid(alpha=.25, axis="y", which="both")
    axes[0].legend(fontsize=9)

    floor = 1e-12
    for backend in ("reference_qng", "triton_qng"):
        median_error = np.median(errors[backend], axis=1)
        axes[1].plot(n_values, np.maximum(median_error, floor), marker="o", linewidth=2,
                     label=LABELS[backend], color=COLORS[backend])
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Qubits N (depth=2)")
    axes[1].set_ylabel("Median final relative error")
    axes[1].set_title("Equivalent result after 100 steps")
    axes[1].grid(alpha=.25, which="both")
    axes[1].legend(fontsize=9)
    fig.suptitle("Figure 2 · 100-step training benefit (A800, 3 seeds)")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=180)
    fig.savefig(args.output.with_suffix(".svg"))
    plt.close(fig)

    summary = {str(n): {"reference_median_seconds": float(medians["reference_qng"][i]),
                        "triton_median_seconds": float(medians["triton_qng"][i]),
                        "speedup": float(speedups[i]),
                        "reference_median_final_error": float(np.median(errors["reference_qng"][i])),
                        "triton_median_final_error": float(np.median(errors["triton_qng"][i]))}
               for i, n in enumerate(n_values)}
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

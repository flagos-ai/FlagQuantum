"""Plot standalone RX/RZ tangent time and memory on A800."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.results]
    labels = [f'{p["pairs"]} pairs\ndepth {p["depth"]}' for p in payloads]
    x = range(len(payloads))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    for axis, key, ylabel, scale in (
        (axes[0], "median_seconds", "Steady tangent time (ms)", 1e3),
        (axes[1], "peak_memory_bytes", "Peak allocated memory (MiB)", 1 / 2**20),
    ):
        reference = [p["vectorized_reverse"][key] * scale for p in payloads]
        triton = [p["triton"][key] * scale for p in payloads]
        axis.bar([i - 0.18 for i in x], reference, 0.36, label="Vectorized reverse")
        axis.bar([i + 0.18 for i in x], triton, 0.36, label="Triton tangent")
        axis.set_xticks(list(x), labels)
        axis.set_ylabel(ylabel)
        axis.set_yscale("log")
        axis.grid(alpha=0.2, axis="y", which="both")
    axes[0].legend()
    fig.suptitle("A800 standalone persistent RX/RZ tangent building block")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".svg"))
    fig.savefig(args.output.with_suffix(".png"), dpi=180)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render the flagship FlagQuantum exact-statevector scaling figure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flagquantum-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strong", type=Path, required=True)
    parser.add_argument("--weak", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    strong, weak = _load(args.strong), _load(args.weak)
    if strong.get("schema_version") != "flagquantum.statevector.scaling_report.v1":
        raise ValueError("invalid strong-scaling report")
    if weak.get("schema_version") != "flagquantum.statevector.weak_scaling_report.v1":
        raise ValueError("invalid weak-scaling report")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfe",
            "axes.edgecolor": "#405166",
            "grid.color": "#c9d2dc",
            "grid.alpha": 0.55,
            "lines.linewidth": 2.2,
            "lines.markersize": 7,
        }
    )
    blue, orange, green, gray = "#1769aa", "#e07a1f", "#16856b", "#66788a"
    sp, wp = strong["points"], weak["points"]
    sx, wx = [p["world_size"] for p in sp], [p["world_size"] for p in wp]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2))

    ax = axes[0, 0]
    st = [p["median_seconds"] for p in sp]
    ax.plot(sx, st, "o-", color=blue, label="Measured")
    ax.plot(sx, [st[0] / x for x in sx], "--", color=gray, label="Ideal 1/N")
    ax.set_title("a  Strong scaling · fixed 28 qubits, 64 gates", loc="left")
    ax.set_ylabel("Forward time (s), median of 10")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    speed = [p["speedup"]["speedup"] for p in sp]
    low = [p["speedup"]["confidence_interval"][0] for p in sp]
    high = [p["speedup"]["confidence_interval"][1] for p in sp]
    ax.errorbar(
        sx,
        speed,
        yerr=[[v - lo for v, lo in zip(speed, low)], [hi - v for v, hi in zip(speed, high)]],
        fmt="o-",
        capsize=3,
        color=orange,
        label="Measured (95% bootstrap CI)",
    )
    ax.plot(sx, sx, "--", color=gray, label="Ideal linear")
    ax.set_title("b  Strong-scaling speedup", loc="left")
    ax.set_ylabel("Speedup vs 1 GPU")
    ax.legend(frameon=False)
    ax.annotate(
        f"{speed[-1]:.2f}× on 16 GPUs\n{sp[-1]['scaling_efficiency']:.1%} efficiency",
        (sx[-1], speed[-1]),
        xytext=(-92, 10),
        textcoords="offset points",
        color=orange,
        weight="bold",
    )

    ax = axes[1, 0]
    wt = [p["median_seconds"] for p in wp]
    eff = [p["weak_scaling_efficiency"]["speedup"] for p in wp]
    ax.plot(wx, wt, "o-", color=green, label="Time")
    ax.axhline(wt[0], color=gray, linestyle="--", label="Ideal constant time")
    ax.set_title("c  Weak scaling · constant 2²⁸ amplitudes/GPU", loc="left")
    ax.set_ylabel("Forward time (s), median of 10")
    other = ax.twinx()
    other.plot(wx, eff, "s:", color=orange, label="Efficiency")
    other.set_ylabel("Weak-scaling efficiency", color=orange)
    other.yaxis.set_major_formatter(PercentFormatter(1.0))
    other.tick_params(axis="y", labelcolor=orange)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = other.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="upper left")
    labels_q = [f"{p['world_size']} GPU\n{p['n_wires']}q" for p in wp]
    ax.set_xticks(wx, labels_q)

    ax = axes[1, 1]
    strong_mem = [p["rank_peak_memory_bytes_max"] / 2**30 for p in sp]
    weak_mem = [p["rank_peak_memory_bytes_max"] / 2**30 for p in wp]
    ax.plot(sx, strong_mem, "o-", color=blue, label="Strong: fixed 28q")
    ax.plot(wx, weak_mem, "s-", color=green, label="Weak: fixed local state")
    ax.set_title("d  Peak allocated memory per rank", loc="left")
    ax.set_ylabel("Peak allocated (GiB/rank)")
    ax.legend(frameon=False)
    ax.annotate(
        "2 nodes",
        (16, weak_mem[-1]),
        xytext=(-8, 13),
        textcoords="offset points",
        ha="right",
        color=green,
        weight="bold",
    )

    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xticks(sx, [str(x) for x in sx])
        ax.set_xlabel("A800 GPUs")
        ax.grid(True, which="major")
        ax.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_xticks(wx, labels_q)
    fig.suptitle(
        "FlagQuantum distributed exact-statevector scaling · up to 16× NVIDIA A800",
        fontsize=16,
        weight="bold",
        x=0.06,
        ha="left",
    )
    fig.text(
        0.06,
        0.015,
        "complex64 · depth 8 · deterministic 64-gate circuit · 3 warmups + 10 measured runs · "
        "CUDA-event synchronized · NCCL · 16-GPU point spans two nodes",
        fontsize=9,
        color="#4c5c6d",
    )
    fig.tight_layout(rect=(0.03, 0.045, 0.99, 0.94), h_pad=2.2, w_pad=2.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix, options in (
        (".png", {"dpi": 240}),
        (".svg", {}),
        (".pdf", {}),
    ):
        fig.savefig(args.output.with_suffix(suffix), bbox_inches="tight", **options)
    plt.close(fig)


if __name__ == "__main__":
    main()

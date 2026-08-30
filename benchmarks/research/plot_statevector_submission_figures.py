#!/usr/bin/env python3
"""Generate compact paper-ready distributed-statevector submission figures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flagquantum-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
RED = "#D55E00"
GREY = "#6C757D"
LIGHT = "#E9ECEF"


def load(root: Path, name: str) -> dict[str, Any]:
    return json.loads((root / name).read_text(encoding="utf-8"))


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.32,
            "lines.linewidth": 1.7,
            "lines.markersize": 5,
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel(ax: Any, label: str, title: str) -> None:
    ax.set_title(rf"$\bf{{{label}}}$  {title}", loc="left", pad=8)
    ax.grid(True, axis="y")
    ax.spines[["top", "right"]].set_visible(False)


def save(fig: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".svg"):
        fig.savefig(output.with_suffix(suffix), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure1_scaling(root: Path) -> Any:
    files = {
        1: "statevector_training_28q_1xa800_triton_vjp_v3.json",
        2: "statevector_training_28q_2xa800_alignment_aware_v4.json",
        4: "statevector_training_28q_4xa800_alignment_aware_v4.json",
        8: "statevector_training_28q_8xa800_alignment_aware_v4.json",
    }
    data = {world: load(root, name) for world, name in files.items()}
    worlds = list(data)
    e2e = [data[w]["end_to_end"]["median_seconds"] for w in worlds]
    backward = [data[w]["backward"]["median_seconds"] for w in worlds]
    e2e_speed = [e2e[0] / value for value in e2e]
    backward_speed = [backward[0] / value for value in backward]
    memory = [data[w]["end_to_end"]["peak_memory_bytes_max"] / 2**30 for w in worlds]
    fig, axes = plt.subplots(1, 3, figsize=(7.12, 2.22))
    axes[0].plot(worlds, e2e, "o-", color=GREEN, label="End-to-end")
    axes[0].plot(worlds, backward, "s-", color=ORANGE, label="Backward")
    axes[0].set(xlabel="A800 GPUs", ylabel="Median time (s)", xticks=worlds)
    axes[0].legend(frameon=False, loc="upper right")
    panel(axes[0], "a", "Fixed-28q training")
    axes[1].plot(worlds, e2e_speed, "o-", color=GREEN, label="End-to-end")
    axes[1].plot(worlds, backward_speed, "s-", color=ORANGE, label="Backward")
    axes[1].plot(worlds, worlds, "--", color=GREY, label="Ideal")
    axes[1].set(xlabel="A800 GPUs", ylabel="Speedup vs. 1 GPU", xticks=worlds)
    axes[1].legend(frameon=False, loc="upper left")
    panel(axes[1], "b", "Strong scaling")
    width = 0.34
    axes[2].bar(
        [w - width / 2 for w in worlds],
        [speed / world for speed, world in zip(e2e_speed, worlds)],
        width=width,
        color=GREEN,
        label="Efficiency",
    )
    memory_normalized = [value / memory[0] for value in memory]
    axes[2].bar(
        [w + width / 2 for w in worlds],
        memory_normalized,
        width=width,
        color=PURPLE,
        label="Memory / 1-GPU",
    )
    axes[2].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[2].set(
        xlabel="A800 GPUs\n(green: efficiency; purple: memory)",
        ylabel="Normalized",
        xticks=worlds,
        ylim=(0, 1.08),
    )
    panel(axes[2], "c", "Efficiency and memory")
    fig.tight_layout(w_pad=1.4)
    return fig


def figure2_mechanisms(root: Path) -> Any:
    fallback = load(root, "statevector_training_28q_2xa800_alignment_aware_fallback_v4.json")
    fused = load(root, "statevector_training_28q_2xa800_alignment_aware_v4.json")
    ring_off = load(root, "statevector_training_ring_28q_2xa800_default_v7.json")
    ring_on = load(root, "statevector_training_ring_28q_2xa800_cross_shard_cx_v7.json")
    topo_off = load(root, "statevector_training_linear_28q_2node16_topology_off_v11.json")
    topo_on = load(root, "statevector_training_linear_28q_2node16_topology_on_v11.json")
    sync_old = load(root, "statevector_training_speed_24q_d8_8xa800_legacy_sync_smoke_v15.json")
    sync_new = load(root, "statevector_training_speed_24q_d8_8xa800_packed_sync_smoke_v15.json")
    fig, axes = plt.subplots(1, 4, figsize=(7.12, 1.92))
    axes[0].bar(
        ["Fallback", "Fused"],
        [fallback["backward"]["median_seconds"], fused["backward"]["median_seconds"]],
        color=[GREY, ORANGE],
    )
    axes[0].set_ylabel("Backward (s)")
    axes[0].set_ylim(
        0,
        max(
            fallback["backward"]["median_seconds"],
            fused["backward"]["median_seconds"],
        )
        * 1.22,
    )
    axes[0].annotate(
        f"{fallback['backward']['median_seconds'] / fused['backward']['median_seconds']:.2f}×",
        (0.5, fallback["backward"]["median_seconds"] * 1.08),
        ha="center",
        weight="bold",
    )
    panel(axes[0], "a", "Fused VJP")
    axes[1].bar(
        ["Default", "Packed"],
        [
            ring_off["backward_communication_bytes_per_rank_max"] / 2**30,
            ring_on["backward_communication_bytes_per_rank_max"] / 2**30,
        ],
        color=[GREY, BLUE],
    )
    axes[1].set_ylabel("Comm. (GiB/rank)")
    ring_peak = ring_off["backward_communication_bytes_per_rank_max"] / 2**30
    axes[1].set_ylim(0, ring_peak * 1.22)
    axes[1].annotate(
        "−42.9%", (0.5, ring_peak * 1.08), ha="center", weight="bold"
    )
    panel(axes[1], "b", "Cross-shard CX")
    axes[2].bar(
        ["Off", "On"],
        [
            topo_off["backward_inter_node_communication_bytes_per_rank_max"] / 2**30,
            topo_on["backward_inter_node_communication_bytes_per_rank_max"] / 2**30,
        ],
        color=[GREY, PURPLE],
    )
    axes[2].set_ylabel("Inter-node (GiB/rank)")
    topology_peak = (
        topo_off["backward_inter_node_communication_bytes_per_rank_max"] / 2**30
    )
    axes[2].set_ylim(0, topology_peak * 1.22)
    axes[2].annotate(
        "−22.2%", (0.5, topology_peak * 1.08), ha="center", weight="bold"
    )
    panel(axes[2], "c", "Topology-aware")
    axes[3].bar(
        ["Broadcast", "Packed"],
        [
            sync_old["optimizer"]["median_seconds"] * 1e3,
            sync_new["optimizer"]["median_seconds"] * 1e3,
        ],
        color=[GREY, GREEN],
    )
    axes[3].set_ylabel("Adam + sync. (ms)")
    sync_peak = sync_old["optimizer"]["median_seconds"] * 1e3
    axes[3].set_ylim(0, sync_peak * 1.22)
    axes[3].annotate(
        "192 → 1 collectives",
        (0.5, sync_peak * 1.08),
        ha="center",
        weight="bold",
        fontsize=6.5,
    )
    panel(axes[3], "d", "Parameter sync.")
    fig.tight_layout(w_pad=1.25)
    return fig


def figure3_training_capacity(root: Path) -> Any:
    local = load(root, "statevector_training_science_30q_1xa800_v12.json")
    distributed = load(root, "statevector_training_science_30q_2node16xa800_v12.json")
    oom = load(root, "statevector_training_science_35q_1xa800_oom_v12.json")
    capacity = load(root, "statevector_training_science_35q_2node16xa800_v12.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.12, 2.18))
    width = 0.34
    x = [0, 1]
    axes[0].bar(
        [v - width / 2 for v in x],
        [local["backward"]["median_seconds"], distributed["backward"]["median_seconds"]],
        width,
        color=ORANGE,
        label="Backward",
    )
    axes[0].bar(
        [v + width / 2 for v in x],
        [local["end_to_end"]["median_seconds"], distributed["end_to_end"]["median_seconds"]],
        width,
        color=GREEN,
        label="End-to-end",
    )
    axes[0].set_xticks(x, ["1 GPU", "16 GPUs\n2 nodes"])
    axes[0].set_ylabel("Median time (s)")
    axes[0].legend(frameon=False, loc="upper right")
    panel(axes[0], "a", "30q science training")
    axes[1].bar(
        ["E2E", "Backward"],
        [
            local["end_to_end"]["median_seconds"]
            / distributed["end_to_end"]["median_seconds"],
            local["backward"]["median_seconds"]
            / distributed["backward"]["median_seconds"],
        ],
        color=[GREEN, ORANGE],
    )
    axes[1].axhline(1.0, linestyle="--", color=GREY)
    axes[1].set_ylabel("Speedup vs. 1 GPU")
    panel(axes[1], "b", "Multi-node speedup")
    device_gib = oom["device"]["total_memory_bytes"] / 2**30
    request_gib = 256.0 if "256.00 GiB" in (oom.get("oom_message") or "") else device_gib
    per_rank_gib = capacity["end_to_end"]["peak_memory_bytes_max"] / 2**30
    axes[2].bar(
        ["A800\ncapacity", "Failed\nrequest", "16-GPU\npeak/rank"],
        [device_gib, request_gib, per_rank_gib],
        color=[GREY, RED, PURPLE],
    )
    axes[2].set_ylabel("Memory (GiB)")
    axes[2].set_ylim(0, request_gib * 1.16)
    axes[2].annotate(
        "OOM",
        (1, request_gib * 1.045),
        ha="center",
        va="bottom",
        color=RED,
        weight="bold",
    )
    panel(axes[2], "c", "35q capacity expansion")
    fig.tight_layout(w_pad=1.35)
    return fig


def figure4_statistics(root: Path) -> Any:
    report = load(root, "statevector_training_speed_24q_d8_1v8_report_v14.json")
    baseline = load(root, "statevector_training_speed_24q_d8_1xa800_local_v14.json")
    candidate = load(root, "statevector_training_speed_24q_d8_8xa800_local_v14.json")
    external = load(root, "custatevec_statevector_20q_l2_a800_current_v13.json")
    phases = ("differentiable_forward", "backward", "end_to_end")
    labels = ("Forward", "Backward", "End-to-end")
    colors = (PURPLE, ORANGE, GREEN)
    values = [report["phases"][phase]["speedup"] for phase in phases]
    low = [
        value - report["phases"][phase]["confidence_interval"][0]
        for phase, value in zip(phases, values)
    ]
    high = [
        report["phases"][phase]["confidence_interval"][1] - value
        for phase, value in zip(phases, values)
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.12, 2.15))
    axes[0].bar(labels, values, color=colors)
    axes[0].errorbar(
        range(3), values, yerr=[low, high], fmt="none", color="black", capsize=2
    )
    axes[0].axhline(1.0, linestyle="--", color=GREY)
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].set_ylabel("8-GPU speedup")
    panel(axes[0], "a", "Bootstrap 95% CI")
    boxes = axes[1].boxplot(
        [
            baseline["end_to_end"]["samples_seconds"],
            candidate["end_to_end"]["samples_seconds"],
        ],
        tick_labels=["1 GPU", "8 GPUs"],
        patch_artist=True,
        boxprops={"facecolor": GREEN, "alpha": 0.65},
        medianprops={"color": "black", "linewidth": 1.5},
    )
    del boxes
    axes[1].set_ylabel("End-to-end time (s)")
    panel(axes[1], "b", "30 synchronized samples")
    axes[2].bar(
        ["FlagQuantum", "cuStateVec"],
        [
            external["flagquantum"]["median_seconds"] * 1e3,
            external["custatevec"]["median_seconds"] * 1e3,
        ],
        color=[BLUE, GREY],
    )
    axes[2].set_ylabel("Forward time (ms)")
    external_peak = external["flagquantum"]["median_seconds"] * 1e3
    axes[2].set_ylim(0, external_peak * 1.25)
    axes[2].annotate(
        f"gap {external['speedup_custatevec_over_flagquantum']:.2f}×; error 1.33e−7",
        (0.5, external_peak * 1.11),
        ha="center",
        va="bottom",
        weight="bold",
        fontsize=6.5,
    )
    panel(axes[2], "c", "External baseline")
    fig.tight_layout(w_pad=1.35)
    return fig


def figure5_external_differentiable(root: Path) -> Any:
    forward = load(root, "custatevec_statevector_20q_l2_a800_current_v13.json")
    differentiable = json.loads(
        (root.parent / "pennylane_lightning_gpu_24q_depth8_single_a800.json").read_text(
            encoding="utf-8"
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.12, 2.15))
    forward_values = [
        forward["flagquantum"]["median_seconds"] * 1e3,
        forward["custatevec"]["median_seconds"] * 1e3,
    ]
    axes[0].bar(["FlagQuantum", "cuStateVec"], forward_values, color=[BLUE, GREY])
    axes[0].set_ylabel("Forward time (ms)")
    axes[0].set_ylim(0, max(forward_values) * 1.27)
    axes[0].annotate(
        f"{forward['speedup_custatevec_over_flagquantum']:.2f}× gap",
        (0.5, max(forward_values) * 1.12),
        ha="center",
        weight="bold",
    )
    panel(axes[0], "a", "Raw forward primitive")

    fq = differentiable["flagquantum"]
    pl = differentiable["pennylane_lightning_gpu"]
    differentiable_values = [fq["median_seconds"], pl["median_seconds"]]
    axes[1].bar(
        ["FlagQuantum", "Lightning-GPU"],
        differentiable_values,
        color=[BLUE, GREY],
    )
    axes[1].set_ylabel("Value + full gradient (s)")
    axes[1].set_ylim(0, max(differentiable_values) * 1.27)
    axes[1].annotate(
        f"{differentiable['speedup_pennylane_over_flagquantum']:.2f}× gap",
        (0.5, max(differentiable_values) * 1.12),
        ha="center",
        weight="bold",
    )
    panel(axes[1], "b", "24q, depth 8, 192 parameters")
    fig.tight_layout(w_pad=2.2)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path("benchmarks/results/comparison")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/legacy/statevector_submission_figures"),
    )
    args = parser.parse_args()
    style()
    figures = (
        ("fig1_scaling", figure1_scaling),
        ("fig2_optimization_mechanisms", figure2_mechanisms),
        ("fig3_training_and_capacity", figure3_training_capacity),
        ("fig4_statistics_and_external_baseline", figure4_statistics),
        ("fig5_differentiable_external_baseline", figure5_external_differentiable),
    )
    for name, builder in figures:
        save(builder(args.results), args.output_dir / name)
    manifest = {
        "schema": "flagquantum.statevector.submission_figures.v1",
        "artifact_class": "derived_development_visualization",
        "figures": [name for name, _ in figures],
        "formats": ["pdf", "svg", "png"],
        "width_inches": 7.12,
        "claim_scope": "NVIDIA A800, complex64, measured workloads shown in captions",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

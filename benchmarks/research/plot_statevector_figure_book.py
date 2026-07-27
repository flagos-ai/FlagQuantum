#!/usr/bin/env python3
"""Build a systematic figure book from measured statevector evidence."""

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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter

BLUE = "#1769aa"
PURPLE = "#6f58a8"
ORANGE = "#e07a1f"
GREEN = "#16856b"
RED = "#c64b4b"
GREY = "#66788a"


def _load(root: Path, name: str) -> dict[str, Any]:
    return json.loads((root / name).read_text(encoding="utf-8"))


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.facecolor": "#fbfcfe",
            "axes.edgecolor": "#405166",
            "grid.color": "#c9d2dc",
            "grid.alpha": 0.55,
            "lines.linewidth": 2.2,
            "lines.markersize": 7,
        }
    )


def _finish(fig: Any, output: Path, book: PdfPages) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    book.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _decorate(axes: Any) -> None:
    for ax in axes:
        ax.grid(True, axis="y")
        ax.spines[["top", "right"]].set_visible(False)


def strong_scaling(root: Path) -> Any:
    names = {
        1: "statevector_training_28q_1xa800_triton_vjp_v3.json",
        2: "statevector_training_28q_2xa800_alignment_aware_v4.json",
        4: "statevector_training_28q_4xa800_alignment_aware_v4.json",
        8: "statevector_training_28q_8xa800_alignment_aware_v4.json",
    }
    data = {world: _load(root, name) for world, name in names.items()}
    worlds = list(data)
    e2e = [data[w]["end_to_end"]["median_seconds"] for w in worlds]
    backward = [data[w]["backward"]["median_seconds"] for w in worlds]
    e2e_speed = [e2e[0] / value for value in e2e]
    backward_speed = [backward[0] / value for value in backward]
    memory = [data[w]["end_to_end"]["peak_memory_bytes_max"] / 2**30 for w in worlds]
    errors = [
        data[w]["correctness"].get("reference_gradient_absolute_error_max")
        for w in worlds
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.5))
    axes[0, 0].plot(worlds, e2e, "o-", color=GREEN, label="End-to-end")
    axes[0, 0].plot(worlds, backward, "o-", color=ORANGE, label="Backward")
    axes[0, 0].set_title("a  Fixed-28q training time", loc="left")
    axes[0, 0].set_ylabel("Median seconds")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(worlds, e2e_speed, "o-", color=GREEN, label="End-to-end")
    axes[0, 1].plot(worlds, backward_speed, "o-", color=ORANGE, label="Backward")
    axes[0, 1].plot(worlds, worlds, "--", color=GREY, label="Ideal")
    axes[0, 1].set_title("b  Strong-scaling speedup", loc="left")
    axes[0, 1].set_ylabel("Speedup vs 1 GPU")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].plot(
        worlds, [s / w for s, w in zip(e2e_speed, worlds)], "o-", color=GREEN
    )
    axes[1, 0].plot(
        worlds,
        [s / w for s, w in zip(backward_speed, worlds)],
        "o-",
        color=ORANGE,
    )
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_title("c  Parallel efficiency", loc="left")
    axes[1, 0].set_ylabel("Speedup / GPU count")
    axes[1, 1].bar(worlds, memory, color=PURPLE, width=0.7)
    for x, value, error in zip(worlds, memory, errors):
        error_text = "1-GPU reference" if error is None else f"grad err {error:.1e}"
        axes[1, 1].annotate(
            f"{value:.1f} GiB\n{error_text}",
            (x, value * 0.82),
            ha="center",
            va="top",
            fontsize=9,
            color="white",
            weight="bold",
        )
    axes[1, 1].set_title("d  Memory and gradient correctness", loc="left")
    axes[1, 1].set_ylabel("Peak allocated GiB/rank")
    for ax in axes.flat:
        ax.set_xticks(worlds)
        ax.set_xlabel("A800 GPUs")
    _decorate(axes.flat)
    fig.suptitle(
        "FlagQuantum distributed differentiable statevector · same-host scaling",
        fontsize=16,
        weight="bold",
        x=0.06,
        ha="left",
    )
    fig.text(
        0.06,
        0.015,
        "complex64 · 28 qubits · full-width linear HEA · RY + directed CX · "
        "sharded forward, adjoint backward and owner-sharded Adam",
        color="#4c5c6d",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.035, 0.05, 0.99, 0.94), h_pad=2.3, w_pad=2.0)
    return fig


def ablations(root: Path) -> Any:
    fallback = _load(root, "statevector_training_28q_2xa800_alignment_aware_fallback_v4.json")
    fused = _load(root, "statevector_training_28q_2xa800_alignment_aware_v4.json")
    ring_off = _load(root, "statevector_training_ring_28q_2xa800_default_v7.json")
    ring_on = _load(root, "statevector_training_ring_28q_2xa800_cross_shard_cx_v7.json")
    topo_off = _load(root, "statevector_training_linear_28q_2node16_topology_off_v11.json")
    topo_on = _load(root, "statevector_training_linear_28q_2node16_topology_on_v11.json")
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6))
    labels = ["Fallback", "Fused VJP"]
    x = range(2)
    axes[0].bar(
        [v - 0.18 for v in x],
        [fallback["backward"]["median_seconds"], fused["backward"]["median_seconds"]],
        width=0.36,
        color=ORANGE,
        label="Backward",
    )
    axes[0].bar(
        [v + 0.18 for v in x],
        [fallback["end_to_end"]["median_seconds"], fused["end_to_end"]["median_seconds"]],
        width=0.36,
        color=GREEN,
        label="End-to-end",
    )
    axes[0].set_xticks(list(x), labels)
    axes[0].set_ylabel("Median seconds")
    axes[0].set_title("a  Operator fusion · 2 GPUs", loc="left")
    axes[0].legend(frameon=False)
    labels = ["Default", "CX pack"]
    axes[1].bar(
        labels,
        [
            ring_off["backward_communication_bytes_per_rank_max"] / 2**30,
            ring_on["backward_communication_bytes_per_rank_max"] / 2**30,
        ],
        color=[GREY, BLUE],
    )
    axes[1].set_ylabel("Backward communication GiB/rank")
    axes[1].set_title("b  Cross-shard CX packing · Ring", loc="left")
    labels = ["Topology off", "Topology on"]
    axes[2].bar(
        labels,
        [
            topo_off["backward_inter_node_communication_bytes_per_rank_max"] / 2**30,
            topo_on["backward_inter_node_communication_bytes_per_rank_max"] / 2**30,
        ],
        color=[GREY, PURPLE],
    )
    axes[2].set_ylabel("Inter-node backward GiB/rank")
    axes[2].set_title("c  Rank-bit topology policy · 16 GPUs", loc="left")
    for ax in axes:
        for patch in ax.patches:
            ax.annotate(
                f"{patch.get_height():.2f}",
                (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
    _decorate(axes)
    fig.suptitle(
        "Where the speedup comes from · compute, communication and topology",
        fontsize=16,
        weight="bold",
        x=0.04,
        ha="left",
    )
    fig.tight_layout(rect=(0.025, 0.02, 0.99, 0.9), w_pad=2.4)
    return fig


def generality(root: Path) -> Any:
    pairs = {
        "Linear": (
            "statevector_training_28q_2xa800_alignment_aware_v4.json",
            "statevector_training_28q_2xa800_cross_shard_cx_triton_v6.json",
        ),
        "Ring": (
            "statevector_training_ring_28q_2xa800_default_v7.json",
            "statevector_training_ring_28q_2xa800_cross_shard_cx_v7.json",
        ),
        "Brickwork": (
            "statevector_training_brickwork_28q_2xa800_default_v7.json",
            "statevector_training_brickwork_28q_2xa800_cross_shard_cx_v7.json",
        ),
    }
    labels, speedups, byte_reductions, errors = [], [], [], []
    for label, (off_name, on_name) in pairs.items():
        off, on = _load(root, off_name), _load(root, on_name)
        labels.append(label)
        speedups.append(
            off["end_to_end"]["median_seconds"] / on["end_to_end"]["median_seconds"]
        )
        byte_reductions.append(
            1
            - on["backward_communication_bytes_per_rank_max"]
            / off["backward_communication_bytes_per_rank_max"]
        )
        errors.append(on["correctness"]["reference_gradient_absolute_error_max"])
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6))
    axes[0].bar(labels, speedups, color=[BLUE, PURPLE, GREEN])
    axes[0].axhline(1.0, color=GREY, linestyle="--")
    axes[0].set_ylabel("End-to-end speedup")
    axes[0].set_title("a  Improvement across circuit families", loc="left")
    axes[1].bar(labels, byte_reductions, color=[BLUE, PURPLE, GREEN])
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_ylim(0, max(byte_reductions) * 1.3)
    axes[1].set_ylabel("Backward bytes reduction")
    axes[1].set_title("b  Communication reduction", loc="left")
    axes[2].bar(labels, errors, color=[BLUE, PURPLE, GREEN])
    axes[2].axhline(3e-5, color=RED, linestyle="--", label="Tolerance 3e-5")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Max raw-gradient absolute error")
    axes[2].set_title("c  Gradient preservation", loc="left")
    axes[2].legend(frameon=False)
    _decorate(axes)
    fig.suptitle(
        "Optimization generality · Full-width Linear, Ring and Brickwork",
        fontsize=16,
        weight="bold",
        x=0.04,
        ha="left",
    )
    fig.tight_layout(rect=(0.025, 0.02, 0.99, 0.9), w_pad=2.4)
    return fig


def science_and_capacity(root: Path) -> Any:
    local = _load(root, "statevector_training_science_30q_1xa800_v12.json")
    distributed = _load(root, "statevector_training_science_30q_2node16xa800_v12.json")
    oom = _load(root, "statevector_training_science_35q_1xa800_oom_v12.json")
    capacity = _load(root, "statevector_training_science_35q_2node16xa800_v12.json")
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6))
    x = [0, 1]
    axes[0].bar(
        [v - 0.18 for v in x],
        [local["backward"]["median_seconds"], distributed["backward"]["median_seconds"]],
        width=0.36,
        color=ORANGE,
        label="Backward",
    )
    axes[0].bar(
        [v + 0.18 for v in x],
        [local["end_to_end"]["median_seconds"], distributed["end_to_end"]["median_seconds"]],
        width=0.36,
        color=GREEN,
        label="End-to-end",
    )
    axes[0].set_xticks(x, ["1 GPU", "16 GPUs\n2 nodes"])
    axes[0].set_ylabel("Median seconds")
    axes[0].set_title("a  30q science training", loc="left")
    axes[0].legend(frameon=False)
    axes[1].bar(
        ["End-to-end", "Backward"],
        [
            local["end_to_end"]["median_seconds"]
            / distributed["end_to_end"]["median_seconds"],
            local["backward"]["median_seconds"]
            / distributed["backward"]["median_seconds"],
        ],
        color=[GREEN, ORANGE],
    )
    axes[1].axhline(1.0, color=GREY, linestyle="--")
    axes[1].set_ylabel("Speedup vs 1 GPU")
    axes[1].set_title("b  Multi-node training acceleration", loc="left")
    requested = oom["device"]["total_memory_bytes"] / 2**30
    if oom.get("oom_message") and "256.00 GiB" in oom["oom_message"]:
        requested = 256.0
    axes[2].bar(
        ["1 GPU\ncapacity", "35q request", "16-GPU\npeak/rank"],
        [
            oom["device"]["total_memory_bytes"] / 2**30,
            requested,
            capacity["end_to_end"]["peak_memory_bytes_max"] / 2**30,
        ],
        color=[GREY, RED, PURPLE],
    )
    axes[2].set_ylabel("GiB")
    axes[2].set_title("c  35q capacity expansion", loc="left")
    axes[2].annotate(
        "OOM",
        (1, requested),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        color=RED,
        weight="bold",
    )
    _decorate(axes)
    fig.suptitle(
        "End-to-end result · faster training and a workload that only multi-GPU can run",
        fontsize=15,
        weight="bold",
        x=0.04,
        ha="left",
    )
    fig.tight_layout(rect=(0.025, 0.02, 0.99, 0.9), w_pad=2.4)
    return fig


def external_baseline(root: Path) -> Any:
    external = _load(root, "custatevec_statevector_20q_l2_a800_current_v13.json")
    gate2 = _load(root, "statevector_gate2_report_v13.json")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axes[0].bar(
        ["FlagQuantum", "cuStateVec"],
        [
            external["flagquantum"]["median_seconds"] * 1e3,
            external["custatevec"]["median_seconds"] * 1e3,
        ],
        color=[BLUE, GREY],
    )
    axes[0].set_ylabel("Forward median (ms)")
    axes[0].set_title("a  Same A800, same exact circuit", loc="left")
    axes[0].text(
        0.5,
        0.95,
        f"Gap: {external['speedup_custatevec_over_flagquantum']:.2f}×",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        weight="bold",
    )
    categories = ["State error", "FQ CV", "cuStateVec CV"]
    values = [
        external["correctness"]["max_abs_error"],
        external["flagquantum"]["coefficient_of_variation"],
        external["custatevec"]["coefficient_of_variation"],
    ]
    axes[1].bar(categories, values, color=[GREEN, BLUE, GREY])
    axes[1].set_yscale("log")
    axes[1].set_title("b  Correctness and measurement stability", loc="left")
    axes[1].set_ylabel("Absolute error / coefficient of variation")
    axes[1].text(
        0.5,
        0.95,
        f"Gate 2: {'PASS' if gate2['passed'] else 'FAIL'}",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        color=GREEN if gate2["passed"] else RED,
        weight="bold",
    )
    _decorate(axes)
    fig.suptitle(
        "External black-box baseline · benchmark-only, no cuQuantum dependency",
        fontsize=15,
        weight="bold",
        x=0.05,
        ha="left",
    )
    fig.tight_layout(rect=(0.035, 0.02, 0.99, 0.9), w_pad=3.0)
    return fig


def statistical_training_speed(root: Path) -> Any:
    report = _load(root, "statevector_training_speed_24q_d8_1v8_report_v14.json")
    baseline = _load(root, "statevector_training_speed_24q_d8_1xa800_local_v14.json")
    candidate = _load(root, "statevector_training_speed_24q_d8_8xa800_local_v14.json")
    phases = ("differentiable_forward", "backward", "optimizer", "end_to_end")
    labels = ("Forward", "Backward", "Adam + broadcast", "End-to-end")
    colors = (PURPLE, ORANGE, GREY, GREEN)
    speedups = [report["phases"][phase]["speedup"] for phase in phases]
    lower = [
        speedup - report["phases"][phase]["confidence_interval"][0]
        for phase, speedup in zip(phases, speedups)
    ]
    upper = [
        report["phases"][phase]["confidence_interval"][1] - speedup
        for phase, speedup in zip(phases, speedups)
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6))
    axes[0].bar(labels, speedups, color=colors)
    axes[0].errorbar(
        range(len(phases)),
        speedups,
        yerr=[lower, upper],
        fmt="none",
        ecolor="#222222",
        capsize=4,
    )
    axes[0].axhline(1.0, color=RED, linestyle="--", label="No improvement")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].set_ylabel("8-GPU speedup vs 1 GPU")
    axes[0].set_title("a  Bootstrap 95% confidence intervals", loc="left")
    axes[0].legend(frameon=False)
    axes[1].boxplot(
        [
            baseline["end_to_end"]["samples_seconds"],
            candidate["end_to_end"]["samples_seconds"],
        ],
        tick_labels=["1 GPU", "8 GPUs"],
        patch_artist=True,
        boxprops={"facecolor": GREEN, "alpha": 0.65},
        medianprops={"color": "#222222", "linewidth": 2},
    )
    axes[1].set_ylabel("End-to-end seconds")
    axes[1].set_title("b  All 30 synchronized samples", loc="left")
    axes[2].bar(
        ["1 GPU", "8 GPUs\nper rank"],
        [
            report["peak_memory_bytes"]["baseline"] / 2**30,
            report["peak_memory_bytes"]["candidate_per_rank"] / 2**30,
        ],
        color=[GREY, PURPLE],
    )
    axes[2].set_ylabel("Peak allocated GiB")
    axes[2].set_title("c  Per-rank memory", loc="left")
    axes[2].text(
        0.5,
        0.95,
        "value err 1.12e-8\ngrad err 2.24e-8",
        transform=axes[2].transAxes,
        ha="center",
        va="top",
        weight="bold",
    )
    _decorate(axes)
    fig.suptitle(
        "Frozen speed-sweep point · 24q, depth 8, 376 gates, 192 parameters",
        fontsize=15,
        weight="bold",
        x=0.04,
        ha="left",
    )
    fig.tight_layout(rect=(0.025, 0.02, 0.99, 0.9), w_pad=2.4)
    return fig


def parameter_synchronization(root: Path) -> Any:
    legacy = _load(
        root, "statevector_training_speed_24q_d8_8xa800_legacy_sync_smoke_v15.json"
    )
    packed = _load(
        root, "statevector_training_speed_24q_d8_8xa800_packed_sync_smoke_v15.json"
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6))
    axes[0].bar(
        ["Per-parameter\nbroadcast", "Packed\nall-reduce"],
        [
            legacy["benchmark_protocol"][
                "parameter_synchronization_collectives_per_step"
            ],
            packed["benchmark_protocol"][
                "parameter_synchronization_collectives_per_step"
            ],
        ],
        color=[GREY, BLUE],
    )
    axes[0].set_ylabel("Collectives per optimizer step")
    axes[0].set_title("a  Collective count", loc="left")
    axes[1].bar(
        ["Legacy", "Packed"],
        [
            legacy["optimizer"]["median_seconds"] * 1e3,
            packed["optimizer"]["median_seconds"] * 1e3,
        ],
        color=[GREY, BLUE],
    )
    axes[1].set_ylabel("Adam + synchronization (ms)")
    axes[1].set_title("b  Optimizer phase", loc="left")
    axes[1].text(
        0.5,
        0.95,
        (
            f"{legacy['optimizer']['median_seconds'] / packed['optimizer']['median_seconds']:.2f}×"
            " faster"
        ),
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        weight="bold",
    )
    axes[2].bar(
        ["Legacy", "Packed"],
        [
            legacy["end_to_end"]["median_seconds"],
            packed["end_to_end"]["median_seconds"],
        ],
        color=[GREY, GREEN],
    )
    axes[2].set_ylim(1.40, 1.48)
    axes[2].set_ylabel("End-to-end seconds")
    axes[2].set_title("c  Full training step", loc="left")
    parameter_difference = max(
        abs(a - b)
        for a, b in zip(
            legacy["correctness"]["final_parameters"],
            packed["correctness"]["final_parameters"],
        )
    )
    axes[2].text(
        0.5,
        0.95,
        f"final-parameter max diff {parameter_difference:.1e}",
        transform=axes[2].transAxes,
        ha="center",
        va="top",
        weight="bold",
    )
    _decorate(axes)
    fig.suptitle(
        "Owner-sharded optimizer synchronization · 192 broadcasts collapsed to one collective",
        fontsize=15,
        weight="bold",
        x=0.04,
        ha="left",
    )
    fig.tight_layout(rect=(0.025, 0.02, 0.99, 0.9), w_pad=2.4)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("benchmarks/results/comparison"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/statevector_figure_book"),
    )
    args = parser.parse_args()
    _style()
    builders = (
        ("fig01_same_host_strong_scaling", strong_scaling),
        ("fig02_compute_communication_topology_ablations", ablations),
        ("fig03_circuit_family_generality", generality),
        ("fig04_science_training_and_capacity", science_and_capacity),
        ("fig05_external_custatevec_baseline", external_baseline),
        ("fig06_statistical_training_speed", statistical_training_speed),
        ("fig07_parameter_synchronization", parameter_synchronization),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    book_path = args.output_dir / "FlagQuantum_Distributed_Statevector_Figure_Book.pdf"
    with PdfPages(book_path) as book:
        for name, builder in builders:
            _finish(builder(args.results), args.output_dir / name, book)
    manifest = {
        "schema": "flagquantum.distributed_statevector.figure_book.v1",
        "artifact_class": "derived_development_visualization",
        "figure_book": str(book_path),
        "figures": [name for name, _ in builders],
        "claim_scope": (
            "Measured development evidence on NVIDIA A800; external cuStateVec "
            "is isolated single-device forward-only evidence."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a non-destructive Figure 0 variant with current TQD evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from plot_statevector_mlsys_current import (
    BLUE,
    GREEN,
    GREY,
    ORANGE,
    PURPLE,
    RED,
    fig0_final_scaling,
    save,
    style,
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    comparison = repo / "benchmarks" / "results" / "comparison"
    output = repo / "benchmarks" / "results" / "statevector_mlsys_current"
    tqd_root = output / "tqd_current"
    tqd2 = load(tqd_root / "tqd_31q_2gpu_2plus5.json")
    tqd4 = load(tqd_root / "tqd_31q_4gpu_2plus5.json")
    tqd8 = load(tqd_root / "tqd_31q_8gpu_2plus5.json")
    tqd16 = load(tqd_root / "tqd_31q_16gpu_2plus5.json")

    style()
    fig = fig0_final_scaling(comparison)
    # fig.axes contains: circuit header, panels a, b, c, d.
    runtime_ax = fig.axes[1]
    scaling_ax = fig.axes[2]
    phase_ax = fig.axes[3]
    relative_ax = fig.axes[4]

    formal_gpu = np.asarray([2, 4, 8, 16])
    formal_time = np.asarray(
        [
            tqd2["value_and_grad"]["median_seconds"],
            tqd4["value_and_grad"]["median_seconds"],
            tqd8["value_and_grad"]["median_seconds"],
            tqd16["value_and_grad"]["median_seconds"],
        ]
    )
    formal_samples = [
        np.asarray(tqd2["value_and_grad"]["samples_seconds"]),
        np.asarray(tqd4["value_and_grad"]["samples_seconds"]),
        np.asarray(tqd8["value_and_grad"]["samples_seconds"]),
        np.asarray(tqd16["value_and_grad"]["samples_seconds"]),
    ]
    formal_error = np.asarray(
        [
            [
                value - samples.min()
                for value, samples in zip(formal_time, formal_samples)
            ],
            [
                samples.max() - value
                for value, samples in zip(formal_time, formal_samples)
            ],
        ]
    )

    formal_handle = runtime_ax.errorbar(
        formal_gpu,
        formal_time,
        yerr=formal_error,
        fmt="D",
        capsize=2,
        color=PURPLE,
        label="TorchQuantum-Dist formal (2+5)",
        zorder=7,
    )
    runtime_ax.annotate(
        "TQD 1-GPU:\nmeasured OOM",
        (1, 105),
        xytext=(6, 0),
        textcoords="offset points",
        color=PURPLE,
        fontsize=5.6,
        va="center",
    )
    runtime_ax.text(
        15.2,
        670,
        "16 GPU: 100.4 s median\nrange 75–442 s; CV 74%\nfirst result ≈80 min",
        ha="right",
        va="top",
        color=PURPLE,
        fontsize=4.8,
    )
    runtime_ax.set_ylim(4, 3000)
    runtime_ax.get_legend().remove()
    runtime_ax.legend(
        [
            Line2D([], [], marker="o", color=GREEN, label="FlagQuantum"),
            Line2D([], [], marker="D", color=PURPLE, label="TorchQuantum-Dist"),
            Line2D(
                [],
                [],
                marker="s",
                color=GREY,
                linestyle="--",
                label="PennyLane",
            ),
        ],
        [
            "FlagQuantum (PyTorch Native)",
            "TorchQuantum-Dist (PyTorch Native)",
            "PennyLane (cuQuantum Runtime)",
        ],
        frameon=False,
        ncol=1,
        loc="upper left",
        fontsize=5.0,
        labelspacing=0.28,
        handletextpad=0.45,
    )

    tqd_speedup = np.asarray(
        [
            1.0,
            formal_time[0] / formal_time[1],
            formal_time[0] / formal_time[2],
            formal_time[0] / formal_time[3],
        ]
    )
    scaling_ax.plot(
        [2, 4, 8, 16],
        tqd_speedup,
        "D-",
        color=PURPLE,
        label="TQD vs. 2 GPUs",
        zorder=7,
    )
    scaling_handles, scaling_labels = scaling_ax.get_legend_handles_labels()
    scaling_order = sorted(
        range(len(scaling_labels)),
        key=lambda index: (
            0
            if "FlagQuantum" in scaling_labels[index]
            else 1
            if "TQD" in scaling_labels[index]
            else 2
            if "PennyLane" in scaling_labels[index]
            else 3
        ),
    )
    scaling_ax.legend(
        [scaling_handles[index] for index in scaling_order],
        [scaling_labels[index] for index in scaling_order],
        frameon=False,
        fontsize=5.2,
        loc="upper left",
        labelspacing=0.22,
        handletextpad=0.4,
    )

    fq8_payload = load(
        comparison / "flagquantum_31q_d8_8xa800_fused_final_warm2_rep5.json"
    )
    fq16_payload = load(
        comparison / "flagquantum_31q_d8_16xa800_fused_final_warm2_rep5.json"
    )
    pl8_payload = load(
        comparison
        / "pennylane_lightning_gpu_31q_d8_8xa800_mpi_final_warm2_rep5.json"
    )
    pl16_payload = load(
        comparison
        / "pennylane_lightning_gpu_31q_d8_16xa800_mpi_final_warm2_rep5.json"
    )
    phase_ax.clear()
    phase_positions = np.asarray([0.0, 0.72, 1.44, 2.45, 3.17, 3.89])
    phase_payloads = [
        ("FlagQuantum", fq8_payload, GREEN, BLUE),
        ("TQD", tqd8, PURPLE, ORANGE),
        ("PennyLane", pl8_payload, GREY, RED),
        ("FlagQuantum", fq16_payload, GREEN, BLUE),
        ("TQD", tqd16, PURPLE, ORANGE),
        ("PennyLane", pl16_payload, GREY, RED),
    ]
    for x_pos, (framework, payload, forward_color, backward_color) in zip(
        phase_positions, phase_payloads
    ):
        if payload is None:
            phase_ax.scatter(
                [x_pos],
                [51],
                marker="x",
                s=50,
                color=PURPLE,
                linewidth=1.5,
                zorder=5,
            )
            phase_ax.text(
                x_pos,
                42,
                "no completed\nforward",
                ha="center",
                va="top",
                color=PURPLE,
                fontsize=4.7,
            )
            continue
        total = payload["value_and_grad"]["median_seconds"]
        forward_pct = 100 * payload["forward"]["median_seconds"] / total
        backward_pct = 100 - forward_pct
        phase_ax.bar(x_pos, forward_pct, width=0.55, color=forward_color, zorder=3)
        phase_ax.bar(
            x_pos,
            backward_pct,
            bottom=forward_pct,
            width=0.55,
            color=backward_color,
            zorder=3,
        )
        bottom_label = "QNode\n+ adj." if framework == "PennyLane" else "Forward"
        top_label = "Torch handoff" if framework == "PennyLane" else "Backward"
        bottom_pct_label = (
            f"{forward_pct:.2f}%" if framework == "PennyLane" else f"{forward_pct:.0f}%"
        )
        phase_ax.text(
            x_pos,
            forward_pct / 2,
            f"{bottom_label}\n{bottom_pct_label}",
            ha="center",
            va="center",
            color="white",
            fontsize=(4.0 if framework == "PennyLane" else 4.6),
        )
        if backward_pct >= 5:
            phase_ax.text(
                x_pos,
                forward_pct + backward_pct / 2,
                f"{top_label}\n{backward_pct:.0f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=4.4,
            )
    phase_ax.axvline(1.95, color="#E9ECEF", linewidth=0.8)
    phase_ax.set(
        xticks=phase_positions,
        xticklabels=[
            "FlagQuantum\n8",
            "TQD\n8",
            "PennyLane\n8",
            "FlagQuantum\n16",
            "TQD\n16",
            "PennyLane\n16",
        ],
        ylabel="Framework-observed time share (%)",
        ylim=(0, 108),
    )
    phase_ax.text(
        0.245,
        0.95,
        "1 node",
        transform=phase_ax.transAxes,
        ha="center",
        fontsize=5.5,
    )
    phase_ax.text(
        0.755,
        0.95,
        "2 nodes",
        transform=phase_ax.transAxes,
        ha="center",
        fontsize=5.5,
    )
    phase_ax.set_title(
        r"$\bf{c}$  Where framework time is recorded", loc="left", pad=7
    )
    phase_ax.grid(True, axis="y", zorder=0)
    phase_ax.spines[["top", "right"]].set_visible(False)
    phase_ax.tick_params(axis="x", labelsize=4.5)

    fq = {
        count: load(
            comparison
            / f"flagquantum_31q_d8_{count}xa800_fused_final_warm2_rep5.json"
        )["value_and_grad"]["median_seconds"]
        for count in (2, 4, 8, 16)
    }
    formal_ratio = formal_time / np.asarray([fq[2], fq[4], fq[8], fq[16]])
    pl = {
        count: load(
            comparison
            / f"pennylane_lightning_gpu_31q_d8_{count}xa800_mpi_final_warm2_rep5.json"
        )["value_and_grad"]["median_seconds"]
        for count in (2, 4, 8, 16)
    }
    gpu = np.asarray([2, 4, 8, 16])
    pl_ratio = np.asarray([pl[count] / fq[count] for count in gpu])

    relative_ax.clear()
    relative_ax.axhline(1, color="black", linewidth=0.8, zorder=1)
    relative_ax.plot(
        gpu,
        pl_ratio,
        "s--",
        color=GREY,
        label="PennyLane / FlagQuantum",
        zorder=4,
    )
    relative_ax.plot(
        [2, 4, 8, 16],
        formal_ratio,
        "D-",
        color=PURPLE,
        label="TQD / FlagQuantum",
        zorder=7,
    )
    for count, value in zip(gpu, pl_ratio):
        relative_ax.annotate(
            f"{value:.2f}×",
            (count, value),
            xytext=((0, 5) if count < 16 else (0, -12)),
            textcoords="offset points",
            ha="center",
            va=("bottom" if count < 16 else "top"),
            color=GREY,
            fontsize=5.5,
            weight="bold",
        )
    tqd_ratio = formal_ratio
    for count, value in zip((2, 4, 8, 16), tqd_ratio):
        relative_ax.annotate(
            f"{value:.1f}×",
            (count, value),
            xytext=((7, 5) if count == 2 else (0, 5)),
            textcoords="offset points",
            ha=("left" if count == 2 else "center"),
            va="bottom",
            color=PURPLE,
            fontsize=5.5,
            weight="bold",
        )
    relative_ax.set_yscale("log")
    relative_ax.set_ylim(0.8, 85)
    relative_ax.set(
        xscale="log",
        xticks=gpu,
        xticklabels=[str(item) for item in gpu],
        xlim=(1.7, 18),
        xlabel="GPUs",
        ylabel="Runtime normalized to FlagQuantum",
    )
    relative_ax.set_title(r"$\bf{d}$  Relative performance", loc="left", pad=7)
    relative_ax.xaxis.set_minor_formatter(NullFormatter())
    relative_ax.grid(True, axis="both", zorder=0)
    relative_ax.spines[["top", "right"]].set_visible(False)
    relative_ax.legend(
        frameon=False,
        loc="upper left",
        fontsize=5.0,
        labelspacing=0.2,
        handletextpad=0.35,
    )
    relative_ax.text(
        0.69,
        0.965,
        "◆ formal 2+5 · error bars: min–max",
        transform=relative_ax.transAxes,
        ha="center",
        va="top",
        color=PURPLE,
        fontsize=4.8,
    )
    save(fig, output / "fig0_final_scaling_with_tqd")


if __name__ == "__main__":
    main()

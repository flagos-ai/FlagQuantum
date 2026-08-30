"""Render the README comparison between FlagQuantum and TensorCircuit-NG."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/results/legacy/tn_compare_tcng_20260731"
FIGURES = OUT / "figures"


def main() -> None:
    gpu = np.array([1, 2, 4, 8])
    fq_files = [
        "flagquantum_staged_full_tape_36q_1gpu.json",
        "flagquantum_staged_forward_36q_2gpu.json",
        "flagquantum_staged_forward_36q_4gpu.json",
        "flagquantum_staged_forward_36q_8gpu.json",
    ]
    fq_payloads = [json.loads((OUT / name).read_text()) for name in fq_files]
    fq = np.array([payload["max_execution_seconds"] for payload in fq_payloads])
    fq_slices = np.array([payload["slice_count"] for payload in fq_payloads])
    tcng = np.array(
        [
            json.loads((OUT / "tcng_36q_1gpu_fixed_path.json").read_text())["steady_step_seconds"],
            json.loads((OUT / "tcng_36q_2gpu.json").read_text())["steady_step_seconds"],
            json.loads((OUT / "tcng_36q_4gpu.json").read_text())["steady_step_seconds"],
            json.loads((OUT / "tcng_36q_8gpu.json").read_text())["steady_step_seconds"],
        ]
    )
    fq_speedup = fq[0] / fq
    tcng_speedup = tcng[0] / tcng

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(13.5, 7.6), facecolor="#07111f")
    grid = fig.add_gridspec(2, 2, hspace=0.37, wspace=0.28)
    ax_time = fig.add_subplot(grid[:, 0])
    ax_scale = fig.add_subplot(grid[0, 1])
    ax_plan = fig.add_subplot(grid[1, 1])
    cyan, coral, muted = "#35d0e8", "#ff8066", "#9dafc3"

    for ax in (ax_time, ax_scale, ax_plan):
        ax.set_facecolor("#0b1829")
        ax.grid(True, color="#29405b", alpha=0.52, linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color("#29405b")

    width = 0.34
    x = np.arange(len(gpu))
    bars_fq = ax_time.bar(x - width / 2, fq, width, color=cyan, label="FlagQuantum")
    bars_tc = ax_time.bar(x + width / 2, tcng, width, color=coral, label="TensorCircuit-NG")
    ax_time.set_yscale("log")
    ax_time.set_xticks(x, [str(v) for v in gpu])
    ax_time.set_xlabel("A800 GPUs")
    ax_time.set_ylabel("Steady value + gradient + SGD step (s, log scale)")
    ax_time.set_title("Absolute step time", loc="left", weight="bold")
    ax_time.legend(frameon=False, loc="upper right")
    for bars in (bars_fq, bars_tc):
        for bar in bars:
            value = bar.get_height()
            ax_time.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.08,
                f"{value:.2f}s",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    for index, ratio in enumerate(tcng / fq):
        ax_time.text(index, 2.0, f"FQ {ratio:.1f}× faster", ha="center", color="#f5d76e", fontsize=8.5)

    ax_scale.plot(gpu, gpu, "--", color=muted, label="Ideal")
    ax_scale.plot(gpu, fq_speedup, "o-", color=cyan, linewidth=2.4, label="FlagQuantum")
    ax_scale.plot(gpu, tcng_speedup, "o-", color=coral, linewidth=2.4, label="TensorCircuit-NG")
    ax_scale.set_xticks(gpu)
    ax_scale.set_xlabel("A800 GPUs")
    ax_scale.set_ylabel("Speedup vs 1 GPU")
    ax_scale.set_title("Observed scaling by execution plan", loc="left", weight="bold")
    ax_scale.legend(frameon=False, ncol=3, fontsize=8)

    labels = ["Native FQ\n(old)", "Canonical FQ\n(1–2 GPU)", "TC-NG"]
    plan_slices = [128, 2, 2]
    plan_seconds = [138.60234964918345, fq[0], tcng[0]]
    px = np.arange(2)
    px = np.arange(3)
    colors = ["#54758d", cyan, coral]
    bars = ax_plan.bar(px, plan_seconds, color=colors, width=0.58)
    ax_plan.set_yscale("log")
    ax_plan.set_xticks(px, labels)
    ax_plan.set_ylabel("1-GPU step time (s, log scale)")
    ax_plan.set_title("Unit-axis canonicalization impact", loc="left", weight="bold")
    for bar, seconds, slices in zip(bars, plan_seconds, plan_slices):
        ax_plan.text(
            bar.get_x() + bar.get_width() / 2,
            seconds * 1.13,
            f"{seconds:.2f}s · {slices} slices",
            ha="center",
            fontsize=8.5,
        )

    fig.suptitle("36-qubit TN training: FlagQuantum vs TensorCircuit-NG", x=0.06, ha="left", fontsize=19, weight="bold")
    fig.text(
        0.06,
        0.925,
        "4×9 grid · 4 entangling cycles · global Z expectation · complex128 · exact reverse gradient · A800 80GB",
        color=muted,
        fontsize=10.5,
    )
    fig.text(
        0.06,
        0.018,
        "FQ uses 2 slices on 1/2 GPUs and adaptive 8 slices on 4/8 GPUs; TC-NG uses one fixed 2-slice path. FQ gradients are owner-sharded.",
        color=muted,
        fontsize=9,
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(FIGURES / f"flagquantum_vs_tensorcircuit_ng_36q_a800.{suffix}", dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    summary = {
        "workload": "36q_4x9_4cycle_global_z_complex128_exact_gradient_sgd",
        "gpu_counts": gpu.tolist(),
        "flagquantum_seconds": fq.tolist(),
        "tensorcircuit_ng_seconds": tcng.tolist(),
        "flagquantum_advantage": (tcng / fq).tolist(),
        "flagquantum_speedup": fq_speedup.tolist(),
        "tensorcircuit_ng_speedup": tcng_speedup.tolist(),
        "flagquantum_slices": fq_slices.tolist(),
        "tensorcircuit_ng_slices": 2,
        "claim_boundary": "Single-node comparison. FlagQuantum changes from its minimum-work 2-slice path to an adaptive 8-slice path at 4/8 GPUs, so its curve includes a plan change and is not pure fixed-plan strong scaling.",
    }
    (FIGURES / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

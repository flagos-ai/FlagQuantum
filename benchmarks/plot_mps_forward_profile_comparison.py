"""Compare legacy and optimized PyTorch/CUPTI MPS Forward profiles."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-rank0", type=Path, required=True)
    parser.add_argument("--new-rank0", type=Path, required=True)
    parser.add_argument("--old-rank1", type=Path, required=True)
    parser.add_argument("--new-rank1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old = [load(args.old_rank0), load(args.old_rank1)]
    new = [load(args.new_rank0), load(args.new_rank1)]
    colors = ("#9aa0a6", "#238b45")

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.2))
    x = np.arange(2)
    width = 0.34
    old_wall = [item["forward_wall_ms"] / 1000 for item in old]
    new_wall = [item["forward_wall_ms"] / 1000 for item in new]
    for offset, values, label, color in (
        (-width / 2, old_wall, "Previous · eager gesvd", colors[0]),
        (width / 2, new_wall, "Optimized · compiled gesvda", colors[1]),
    ):
        bars = axes[0].bar(x + offset, values, width, label=label, color=color)
        axes[0].bar_label(bars, fmt="%.2f s", padding=3, fontsize=9)
    axes[0].set_xticks(x, ("Rank 0", "Rank 1"))
    axes[0].set_ylabel("Forward wall time under profiler (s)")
    axes[0].set_title("(a) Forward wall time", loc="left", weight="bold")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False)
    axes[0].text(
        0.5, 0.82,
        f"{old_wall[0]/new_wall[0]:.2f}× / {old_wall[1]/new_wall[1]:.2f}×",
        transform=axes[0].transAxes, ha="center", color=colors[1], weight="bold"
    )

    old_nccl = [item["cuda_work_ms"]["NCCL wait / communication"] / 1000 for item in old]
    new_nccl = [item["cuda_work_ms"]["NCCL wait / communication"] / 1000 for item in new]
    for offset, values, label, color in (
        (-width / 2, old_nccl, "Previous · eager gesvd", colors[0]),
        (width / 2, new_nccl, "Optimized · compiled gesvda", colors[1]),
    ):
        bars = axes[1].bar(x + offset, values, width, label=label, color=color)
        axes[1].bar_label(bars, fmt="%.2f s", padding=3, fontsize=9)
    axes[1].set_xticks(x, ("Rank 0", "Rank 1"))
    axes[1].set_ylabel("Aggregated NCCL CUDA time (s)")
    axes[1].set_title("(b) NCCL wait / communication", loc="left", weight="bold")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].text(
        0.5, 0.82,
        f"{old_nccl[0]/new_nccl[0]:.2f}× / {old_nccl[1]/new_nccl[1]:.2f}× lower",
        transform=axes[1].transAxes, ha="center", color=colors[1], weight="bold"
    )

    names = ("SVD", "QR", "Hamiltonian / env scan", "Contraction/gate + other")
    def detail(item: dict) -> list[float]:
        work = item["cuda_work_ms"]
        return [
            work["SVD"], work["QR"], work["Hamiltonian / env scan"],
            work["Fused contraction + gate"] + work["Truncate / layout"]
            + work["Other forward CUDA work"],
        ]
    y = np.arange(len(names))
    old_detail, new_detail = detail(old[1]), detail(new[1])
    axes[2].barh(y + width / 2, old_detail, width, label="Previous · eager gesvd", color=colors[0])
    axes[2].barh(y - width / 2, new_detail, width, label="Optimized · compiled gesvda", color=colors[1])
    axes[2].set_yticks(y, names)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Aggregated CUDA kernel time (ms)")
    axes[2].set_title("(c) Rank 1 compute detail", loc="left", weight="bold")
    axes[2].grid(axis="x", alpha=0.22)
    axes[2].legend(frameon=False)
    for index, (before, after) in enumerate(zip(old_detail, new_detail)):
        if before > 0 and before / max(after, 1e-12) > 1.05:
            axes[2].text(max(before, after) * 1.02, index, f"{before/after:.2f}×", va="center", fontsize=9, color=colors[1], weight="bold")

    fig.suptitle(
        "PyTorch/CUPTI Forward operator profile · previous vs optimized\n"
        "N=32 · p=2 · χ=512 · complex64 · 8×A800 · same attribution categories\n"
        "previous: eager exact gesvd · optimized: compiled approximate gesvda",
        fontsize=14, weight="bold",
    )
    fig.text(
        0.99, 0.015,
        "CUDA times are additive device work attributed by launch External id; they are not critical-path wall time.\n"
        "Optimized trace selects the last Forward scope to exclude torch.compile warmup.",
        ha="right", va="bottom", fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.89))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    summary = {
        "old_rank0": old[0], "new_rank0": new[0],
        "old_rank1": old[1], "new_rank1": new[1],
        "rank0_forward_wall_speedup": old_wall[0] / new_wall[0],
        "rank1_forward_wall_speedup": old_wall[1] / new_wall[1],
        "rank0_nccl_cuda_reduction_factor": old_nccl[0] / new_nccl[0],
        "rank1_nccl_cuda_reduction_factor": old_nccl[1] / new_nccl[1],
        "rank1_svd_cuda_reduction_factor": old_detail[0] / new_detail[0],
        "semantics": "PyTorch profiler Chrome trace with CUPTI CUDA kernels; additive work, not critical-path time",
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

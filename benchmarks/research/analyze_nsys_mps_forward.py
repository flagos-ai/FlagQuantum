"""Extract a per-rank MPS Forward critical-path view from Nsight Systems SQL."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt


def _merge(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _duration(intervals):
    return sum(end - start for start, end in intervals)


def _overlap_duration(left, right):
    first = second = total = 0
    while first < len(left) and second < len(right):
        total += max(
            0,
            min(left[first][1], right[second][1])
            - max(left[first][0], right[second][0]),
        )
        if left[first][1] < right[second][1]:
            first += 1
        else:
            second += 1
    return total


def analyze(path: Path) -> dict:
    connection = sqlite3.connect(path)
    forward_rows = list(
        connection.execute(
            "SELECT start, end, globalTid FROM NVTX_EVENTS "
            "WHERE text='flagquantum::mps::forward' ORDER BY globalTid"
        )
    )
    by_process = {}
    for start, end, global_tid in forward_rows:
        global_pid = (int(global_tid) >> 24) << 24
        by_process.setdefault(global_pid, []).append((start, end, global_tid))
    # Use the last Forward range per rank so compilation/warmup does not pollute
    # the steady-state overlap measurement.  A one-step trace is unchanged.
    forward = [sorted(rows)[-1] for rows in by_process.values()]
    ranks = []
    for start, end, global_tid in forward:
        global_pid = (int(global_tid) >> 24) << 24
        kernels = list(
            connection.execute(
                "SELECT k.start, k.end, s.value, k.deviceId "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                "JOIN StringIds s ON s.id=k.shortName "
                "WHERE k.globalPid=? AND k.end>? AND k.start<?",
                (global_pid, start, end),
            )
        )
        device = int(kernels[0][3]) if kernels else len(ranks)
        nccl = _merge(
            (max(start, left), min(end, right))
            for left, right, name, _ in kernels
            if str(name).startswith("nccl")
        )
        compute = _merge(
            (max(start, left), min(end, right))
            for left, right, name, _ in kernels
            if not str(name).startswith("nccl")
        )
        overlap = _overlap_duration(nccl, compute)
        compute_only = _duration(compute) - overlap
        nccl_only = _duration(nccl) - overlap
        idle = max(0, end - start - compute_only - nccl_only - overlap)
        totals = {
            "compute": compute_only,
            "nccl_wait_communication": nccl_only,
            "overlap": overlap,
            "idle": idle,
        }
        ranks.append(
            {
                "rank": device,
                "forward_seconds": (end - start) / 1e9,
                **{f"{key}_seconds": value / 1e9 for key, value in totals.items()},
            }
        )
    connection.close()
    ranks.sort(key=lambda item: item["rank"])
    return {
        "source": str(path),
        "profiler": "NVIDIA Nsight Systems",
        "time_semantics": "mutually_exclusive_gpu_timeline_intervals_clipped_to_forward_nvtx",
        "ranks": ranks,
        "critical_rank": max(ranks, key=lambda item: item["forward_seconds"])["rank"],
    }


def plot(result: dict, output: Path, title: str) -> None:
    ranks = result["ranks"]
    categories = (
        ("compute_seconds", "Compute", "#2468b4"),
        ("nccl_wait_communication_seconds", "NCCL wait / communication", "#d98c10"),
        ("overlap_seconds", "Compute–NCCL overlap", "#6a3d9a"),
        ("idle_seconds", "GPU idle / host gap", "#bdbdbd"),
    )
    fig, (timeline, detail) = plt.subplots(
        1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1.65, 1.0]}
    )
    bottoms = [0.0] * len(ranks)
    for key, label, color in categories:
        values = [item[key] for item in ranks]
        timeline.barh(
            [item["rank"] for item in ranks], values, left=bottoms,
            label=label, color=color, edgecolor="white", linewidth=0.6,
        )
        bottoms = [left + value for left, value in zip(bottoms, values)]
    timeline.invert_yaxis()
    timeline.set_xlabel("Forward critical-path time (s)")
    timeline.set_ylabel("GPU / rank")
    timeline.set_title("(a) Mutually exclusive Forward timeline", loc="left", weight="bold")
    timeline.grid(axis="x", alpha=0.2)
    timeline.legend(fontsize=8, ncol=2, loc="lower right")

    compute = [item["compute_seconds"] for item in ranks]
    nccl = [item["nccl_wait_communication_seconds"] for item in ranks]
    x = [item["rank"] for item in ranks]
    detail.plot(x, compute, "o-", label="Compute", color="#2468b4", linewidth=2)
    detail.plot(x, nccl, "s-", label="NCCL wait / communication", color="#d98c10", linewidth=2)
    detail.set_xlabel("GPU / rank")
    detail.set_ylabel("Time inside Forward (s)")
    detail.set_title("(b) Rank imbalance and serialization", loc="left", weight="bold")
    detail.grid(alpha=0.22)
    detail.legend(fontsize=9)
    overlap = sum(item["overlap_seconds"] for item in ranks)
    detail.text(
        0.97, 0.05,
        f"Measured compute–NCCL overlap\nacross all ranks: {overlap:.3f} s",
        transform=detail.transAxes, ha="right", va="bottom", fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "fc": "#f7f7f7", "ec": "#cccccc"},
    )
    fig.suptitle(title, x=0.04, ha="left", fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Distributed MPS Forward · Nsight Systems critical path")
    args = parser.parse_args()
    result = analyze(args.sqlite)
    plot(result, args.output, args.title)


if __name__ == "__main__":
    main()

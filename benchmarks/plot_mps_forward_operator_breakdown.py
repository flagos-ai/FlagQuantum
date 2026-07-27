"""Build an honest forward-operator breakdown from a PyTorch Chrome trace.

CUDA kernels are attributed to the innermost FlagQuantum record_function scope
that contains their CPU launch event.  Reported CUDA time is aggregated device
work, not wall time; overlapping kernels therefore remain additive.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


SCOPE_CATEGORY = {
    "flagquantum::mps::contraction_gate_fused": "Fused contraction + gate",
    "flagquantum::mps::qr": "QR",
    "flagquantum::mps::batched_qr": "QR",
    "flagquantum::mps::svd": "SVD",
    "flagquantum::mps::batched_svd": "SVD",
    "flagquantum::mps::truncate_layout": "Truncate / layout",
    "flagquantum::mps::factorization_retention": "Cache retention",
    "flagquantum::mps::heisenberg_mpo_forward": "Hamiltonian / env scan",
    "flagquantum::mps::left_environment_scan": "Hamiltonian / env scan",
    "flagquantum::mps::right_environment_scan": "Hamiltonian / env scan",
    "flagquantum::mps::environment_transfer": "Hamiltonian / env scan",
    "flagquantum::mps::environment_transfer_channels": "Hamiltonian / env scan",
    "flagquantum::mps::p2p_send": "NCCL wait / communication",
    "flagquantum::mps::p2p_recv": "NCCL wait / communication",
    "flagquantum::mps::p2p_batch_send": "NCCL wait / communication",
    "flagquantum::mps::p2p_batch_recv": "NCCL wait / communication",
    "flagquantum::mps::p2p_static_send": "NCCL wait / communication",
    "flagquantum::mps::p2p_static_recv": "NCCL wait / communication",
}

ORDER = [
    "Fused contraction + gate",
    "QR",
    "SVD",
    "Truncate / layout",
    "Cache retention",
    "Hamiltonian / env scan",
    "NCCL wait / communication",
    "Other forward CUDA work",
]


def _inside(event: dict, scope: dict) -> bool:
    start = float(event["ts"])
    return float(scope["ts"]) <= start <= float(scope["ts"]) + float(scope["dur"])


def parse_trace(path: Path, *, scope_selection: str = "longest") -> dict:
    events = json.loads(path.read_text())["traceEvents"]
    forward_scopes = [
        event
        for event in events
        if event.get("cat") == "user_annotation"
        and event.get("name") == "flagquantum::mps::forward"
        and event.get("ph") == "X"
    ]
    if not forward_scopes:
        raise ValueError(f"no FlagQuantum forward scope in {path}")
    # Multi-step compiled profiles select the last scope to exclude cold
    # compilation; legacy one-step/eager evidence retains longest-scope mode.
    forward = (
        max(forward_scopes, key=lambda event: float(event["ts"]))
        if scope_selection == "last"
        else max(forward_scopes, key=lambda event: float(event["dur"]))
    )
    category_scopes = [
        (event, SCOPE_CATEGORY[event["name"]])
        for event in events
        if event.get("cat") == "user_annotation"
        and event.get("name") in SCOPE_CATEGORY
        and event.get("ph") == "X"
        and _inside(event, forward)
    ]

    external_category: dict[int, str] = {}
    for event in events:
        args = event.get("args", {})
        external_id = args.get("External id")
        if external_id is None or event.get("ph") != "X" or not _inside(event, forward):
            continue
        containing = [
            (scope, category)
            for scope, category in category_scopes
            if _inside(event, scope)
        ]
        if containing:
            # Innermost scope prevents a broad Hamiltonian scope from hiding
            # a more specific environment-transfer scope.
            _, category = min(containing, key=lambda item: float(item[0]["dur"]))
            external_category[int(external_id)] = category

    forward_external_ids = {
        int(event.get("args", {})["External id"])
        for event in events
        if event.get("ph") == "X"
        and "External id" in event.get("args", {})
        and _inside(event, forward)
    }
    device_us: defaultdict[str, float] = defaultdict(float)
    forward_device_us = 0.0
    for event in events:
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            continue
        external_id = event.get("args", {}).get("External id")
        if external_id is None or int(external_id) not in forward_external_ids:
            continue
        category = external_category.get(int(external_id))
        # NCCL kernels are asynchronous and may outlive the short Python p2p
        # annotation which launched them.  Kernel identity is the reliable
        # attribution for their device-side wait/communication duration.
        if str(event.get("name", "")).startswith("ncclDevKernel"):
            category = "NCCL wait / communication"
        # Only kernels whose launch is mapped into the selected forward enter
        # this total; kernels from reverse and optimizer are excluded.
        if category is not None:
            device_us[category] += float(event["dur"])
            forward_device_us += float(event["dur"])
        elif int(external_id) in external_category:
            forward_device_us += float(event["dur"])

    # Determine all forward launch IDs separately so uncategorized CUDA work
    # remains visible instead of disappearing from the chart.
    total_forward_device_us = sum(
        float(event["dur"])
        for event in events
        if event.get("cat") == "kernel"
        and event.get("ph") == "X"
        and event.get("args", {}).get("External id") in forward_external_ids
    )
    device_us["Other forward CUDA work"] = max(
        0.0, total_forward_device_us - sum(device_us.values())
    )

    host_scope_us: defaultdict[str, float] = defaultdict(float)
    for scope, category in category_scopes:
        host_scope_us[category] += float(scope["dur"])
    return {
        "trace": str(path),
        "forward_wall_ms": float(forward["dur"]) / 1e3,
        "cuda_work_ms": {name: device_us[name] / 1e3 for name in ORDER},
        "annotated_host_scope_ms": {
            name: host_scope_us[name] / 1e3 for name in ORDER
        },
        "method": (
            "CUDA kernels attributed through External id to the innermost "
            "record_function scope containing the CPU launch event"
        ),
        "forward_scope_selection": scope_selection,
    }


def plot(result: dict, output: Path, title: str) -> None:
    values = [result["cuda_work_ms"][name] for name in ORDER]
    total = sum(values)
    shares = [100.0 * value / total if total else 0.0 for value in values]
    colors = [
        "#2468b4", "#57a0d3", "#d24b40", "#ef8a62", "#8c6bb1",
        "#238b45", "#d98c10", "#9aa0a6",
    ]

    fig, (ax, overview) = plt.subplots(
        1, 2, figsize=(13.2, 6.2), gridspec_kw={"width_ratios": [1.8, 1.0]}
    )
    compute_order = [name for name in ORDER if name != "NCCL wait / communication"]
    compute_values = [result["cuda_work_ms"][name] for name in compute_order]
    compute_total = sum(compute_values)
    compute_colors = [color for name, color in zip(ORDER, colors) if name != "NCCL wait / communication"]
    positions = list(range(len(compute_order)))
    bars = ax.barh(positions, compute_values, color=compute_colors, edgecolor="white", linewidth=0.8)
    ax.set_yticks(positions, compute_order)
    ax.invert_yaxis()
    ax.set_xlabel("Aggregated CUDA kernel time (ms)")
    ax.set_title("(a) Compute detail", loc="left", fontsize=12, weight="bold")
    ax.grid(axis="x", alpha=0.22)
    max_value = max(compute_values) if compute_values else 1.0
    ax.set_xlim(0, max_value * 1.38 if max_value else 1.0)
    for bar, name, value in zip(bars, compute_order, compute_values):
        share = 100.0 * value / compute_total if compute_total else 0.0
        label = f"{value:,.2f} ms  ({share:.1f}%)" if value >= 0.005 else "<0.01 ms"
        ax.text(
            bar.get_width() + max_value * 0.015,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=9,
        )

    nccl = result["cuda_work_ms"]["NCCL wait / communication"]
    compute = total - nccl
    overview.barh([0], [nccl], color="#d98c10", label="NCCL wait / communication")
    overview.barh([0], [compute], left=[nccl], color="#2468b4", label="All compute")
    overview.set_yticks([])
    overview.set_xlabel("Aggregated CUDA kernel time (ms)")
    overview.set_title("(b) Communication vs. compute", loc="left", fontsize=12, weight="bold")
    overview.grid(axis="x", alpha=0.22)
    overview.legend(fontsize=9, loc="upper right")
    for left, value, label in ((0.0, nccl, "NCCL"), (nccl, compute, "compute")):
        if value > total * 0.04:
            overview.text(
                left + value / 2, 0, f"{label}\n{100*value/total:.2f}%",
                ha="center", va="center", color="white", fontsize=9, weight="bold",
            )
        elif value > 0:
            overview.annotate(
                f"{label}: {value:,.1f} ms ({100*value/total:.2f}%)",
                xy=(left + value / 2, 0), xytext=(0.98, 0.72),
                textcoords="axes fraction", ha="right", va="center", fontsize=9,
                arrowprops={"arrowstyle": "->", "color": "#444444"},
            )

    note = (
        f"Forward wall time under profiler: {result['forward_wall_ms']/1e3:.2f} s\n"
        f"Attributed CUDA work: {total:,.2f} ms (additive; not critical-path time)\n"
        "Contraction and gate application are fused in the production kernel.\n"
        "Cache retention is graph/tensor bookkeeping, so it launches no standalone CUDA kernel."
    )
    fig.text(0.99, 0.015, note, ha="right", va="bottom", fontsize=8.5)
    fig.suptitle(title, x=0.04, ha="left", fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--title",
        default="Distributed MPS forward operator breakdown",
    )
    parser.add_argument(
        "--scope-selection", choices=("longest", "last"), default="longest"
    )
    args = parser.parse_args()
    result = parse_trace(args.trace, scope_selection=args.scope_selection)
    plot(result, args.output, args.title)


if __name__ == "__main__":
    main()

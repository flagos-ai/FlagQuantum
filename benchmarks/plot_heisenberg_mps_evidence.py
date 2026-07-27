"""Generate the FlagQuantum distributed-MPS Heisenberg evidence figure suite."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flagquantum-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks/results/heisenberg_vqe"
OUT = ROOT / "benchmarks/results/heisenberg_vqe/figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "blue": "#2563EB",
    "cyan": "#0891B2",
    "green": "#16A34A",
    "orange": "#EA580C",
    "red": "#DC2626",
    "purple": "#7C3AED",
    "gray": "#64748B",
    "light": "#E2E8F0",
    "dark": "#0F172A",
}


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text())


def metric(payload: dict, rank: int = 0, step: int = 0) -> dict:
    return payload["rank_records"][rank]["training"]["step_metrics"][step]


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.titleweight": "bold",
            "axes.edgecolor": COLORS["gray"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": COLORS["light"],
            "grid.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, stem: str, book: PdfPages) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    book.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def overview(capacity: dict, baseline: dict) -> plt.Figure:
    fig = plt.figure(figsize=(15, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(grid[0, 0])
    ax.axis("off")
    stages = ["Heisenberg\nHVA", "FlagQuantum\nIR", "Site-sharded\nMPS", "Reverse VJP\n+ sharded Adam"]
    xs = np.linspace(0.12, 0.88, len(stages))
    for i, (x, label) in enumerate(zip(xs, stages)):
        box = FancyBboxPatch(
            (x - 0.1, 0.38), 0.2, 0.25,
            boxstyle="round,pad=0.02", facecolor="#EFF6FF",
            edgecolor=COLORS["blue"], linewidth=1.8,
        )
        ax.add_patch(box)
        ax.text(x, 0.505, label, ha="center", va="center", fontsize=11, weight="bold")
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + 0.105, 0.505), (xs[i + 1] - 0.105, 0.505),
                                         arrowstyle="-|>", mutation_scale=15,
                                         color=COLORS["gray"], linewidth=1.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("A. End-to-end distributed training path", loc="left")

    ax = fig.add_subplot(grid[0, 1])
    ax.axis("off")
    facts = [
        ("256", "qubits"), ("16", "HVA layers"), ("512", "realized bond"),
        ("12,240", "two-qubit gates"), ("96", "two-qubit depth"),
        ("8×A800", "hardware"),
    ]
    for i, (value, label) in enumerate(facts):
        row, col = divmod(i, 3)
        x, y = 0.18 + col * 0.32, 0.72 - row * 0.42
        ax.text(x, y, value, ha="center", va="center", fontsize=23,
                color=COLORS["blue"], weight="bold")
        ax.text(x, y - 0.12, label, ha="center", va="center", color=COLORS["gray"])
    ax.set_title("B. Largest complete gradient workload", loc="left")

    ax = fig.add_subplot(grid[1, 0])
    peak = max(capacity["local_memory_bytes_by_rank"]) / 2**30
    oom = baseline["rank_records"][0]["peak_memory_bytes"] / 2**30
    ax.bar(["1×A800\nCUDA OOM", "8×A800\ncompleted"], [oom, peak],
           color=[COLORS["red"], COLORS["green"]], width=0.6)
    ax.axhline(79.25, linestyle="--", color=COLORS["gray"], label="A800 usable capacity")
    ax.set_ylabel("Peak memory (GiB)")
    ax.set_ylim(0, 88); ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper right")
    ax.text(0, oom + 2, f"{oom:.1f} GiB", ha="center", weight="bold")
    ax.text(1, peak + 2, f"{peak:.1f} GiB/rank", ha="center", weight="bold")
    ax.set_title("C. Measured capacity expansion", loc="left")

    ax = fig.add_subplot(grid[1, 1])
    labels = ["Forward", "Reverse", "Optimizer", "End-to-end"]
    values = [
        max(metric(capacity, rank)["forward_seconds"] for rank in range(8)),
        max(metric(capacity, rank)["reverse_seconds"] for rank in range(8)),
        max(metric(capacity, rank)["optimizer_seconds"] for rank in range(8)),
        max(record["elapsed_seconds"] for record in capacity["rank_records"]),
    ]
    bars = ax.barh(labels, values, color=[COLORS["blue"], COLORS["purple"], COLORS["orange"], COLORS["dark"]])
    ax.bar_label(bars, fmt="%.2f s", padding=4)
    ax.set_xlabel("Seconds"); ax.grid(axis="x"); ax.invert_yaxis()
    ax.set_title("D. Full-step phase timing", loc="left")
    fig.suptitle("FlagQuantum Distributed MPS: measured capability overview", fontsize=18, weight="bold")
    return fig


def capacity_memory(capacity: dict, baseline: dict) -> plt.Figure:
    ranks = np.arange(8)
    forward = np.array([metric(capacity, r)["forward_peak_memory_bytes"] for r in ranks]) / 2**30
    reverse = np.array([metric(capacity, r)["reverse_peak_memory_bytes"] for r in ranks]) / 2**30
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    w = 0.38
    axes[0].bar(ranks - w/2, forward, w, label="Forward peak", color=COLORS["blue"])
    axes[0].bar(ranks + w/2, reverse, w, label="Reverse peak", color=COLORS["purple"])
    axes[0].set_xticks(ranks); axes[0].set_xlabel("Rank / GPU"); axes[0].set_ylabel("GiB")
    axes[0].grid(axis="y"); axes[0].legend(frameon=False)
    axes[0].set_title("A. Phase-specific memory by rank")
    axes[1].barh(["8-GPU max/rank", "Single-GPU failure"],
                 [forward.max(), baseline["rank_records"][0]["peak_memory_bytes"] / 2**30],
                 color=[COLORS["green"], COLORS["red"]])
    axes[1].axvline(79.25, color=COLORS["gray"], linestyle="--", label="A800 usable capacity")
    axes[1].set_xlabel("Peak memory (GiB)"); axes[1].grid(axis="x")
    axes[1].legend(frameon=False); axes[1].set_title("B. Matched capacity baseline")
    fig.suptitle("256q, p16, χ512: distributed completion versus single-A800 OOM", fontsize=15, weight="bold")
    return fig


def scaling_figure(scaling: list[dict]) -> plt.Figure:
    gpu = np.array([1, 2, 4, 8])
    times = np.array([max(r["elapsed_seconds"] for r in p["rank_records"]) for p in scaling])
    peaks = np.array([
        max(
            max(
                metric(p, rank)["forward_peak_memory_bytes"],
                metric(p, rank)["reverse_peak_memory_bytes"],
            )
            for rank in range(len(p["rank_records"]))
        )
        for p in scaling
    ]) / 2**30
    speedup = times[0] / times
    efficiency = speedup / gpu * 100
    losses = np.array([p["rank_records"][0]["training"]["losses"][0] for p in scaling])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0,0].plot(gpu, times, "o-", color=COLORS["blue"], linewidth=2)
    axes[0,0].set_ylabel("End-to-end time (s)"); axes[0,0].set_xticks(gpu); axes[0,0].grid(); axes[0,0].set_title("A. Runtime")
    axes[0,1].plot(gpu, gpu, "--", color=COLORS["gray"], label="Ideal")
    axes[0,1].plot(gpu, speedup, "o-", color=COLORS["green"], linewidth=2, label="Measured")
    axes[0,1].set_ylabel("Speedup"); axes[0,1].set_xticks(gpu); axes[0,1].grid(); axes[0,1].legend(frameon=False); axes[0,1].set_title("B. Strong scaling")
    axes[1,0].plot(gpu, peaks, "o-", color=COLORS["purple"], linewidth=2, label="Measured")
    axes[1,0].plot(gpu, peaks[0] / gpu, "--", color=COLORS["gray"], label="Ideal 1/N")
    axes[1,0].set_xlabel("GPUs"); axes[1,0].set_ylabel("Phase peak memory / rank (GiB)"); axes[1,0].set_xticks(gpu); axes[1,0].grid(); axes[1,0].legend(frameon=False); axes[1,0].set_title("C. Sharded memory")
    axes[1,1].bar(gpu.astype(str), efficiency, color=COLORS["orange"])
    axes[1,1].set_xlabel("GPUs"); axes[1,1].set_ylabel("Apparent efficiency (%)"); axes[1,1].set_ylim(0, max(160, efficiency.max() * 1.1)); axes[1,1].grid(axis="y"); axes[1,1].set_title("D. Superlinear working-set relief")
    fig.suptitle(f"Large-workload strong scaling: fixed 192q, p2, χ512 (energy spread {np.ptp(losses):.2e})", fontsize=15, weight="bold")
    return fig


def checkpoint_figure(a: dict, b: dict) -> plt.Figure:
    pa = max(r["peak_memory_bytes"] for r in a["rank_records"]) / 2**30
    pb = max(r["peak_memory_bytes"] for r in b["rank_records"]) / 2**30
    ta = max(r["elapsed_seconds"] for r in a["rank_records"])
    tb = max(r["elapsed_seconds"] for r in b["rank_records"])
    phases_a = [max(metric(a,r)[k] for r in range(8)) for k in ("forward_seconds","reverse_seconds","optimizer_seconds")]
    phases_b = [max(metric(b,r)[k] for r in range(8)) for k in ("forward_seconds","reverse_seconds","optimizer_seconds")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot([pa, pb], [ta, tb], "o-", linewidth=2.5, markersize=9, color=COLORS["blue"])
    axes[0].annotate("Recompute", (pa, ta), xytext=(8, 8), textcoords="offset points")
    axes[0].annotate("Save factorization", (pb, tb), xytext=(8, -18), textcoords="offset points")
    axes[0].set_xlabel("Peak memory / rank (GiB)"); axes[0].set_ylabel("End-to-end time (s)"); axes[0].grid(); axes[0].set_title("A. Memory–time Pareto")
    x = np.arange(3); w = 0.36
    axes[1].bar(x-w/2, phases_a, w, label="Recompute", color=COLORS["gray"])
    axes[1].bar(x+w/2, phases_b, w, label="Save factorization", color=COLORS["green"])
    axes[1].set_xticks(x, ["Forward", "Reverse", "Optimizer"]); axes[1].set_ylabel("Seconds"); axes[1].grid(axis="y"); axes[1].legend(frameon=False); axes[1].set_title("B. Phase-level A/B")
    axes[1].text(1, max(phases_a[1], phases_b[1])*0.55, f"{phases_a[1]/phases_b[1]:.2f}× faster\nreverse", ha="center", weight="bold", color=COLORS["green"])
    fig.suptitle("Forward-factorization checkpoint: 64q, p2, χ512, 8 GPUs", fontsize=15, weight="bold")
    return fig


def rank_profile(capacity: dict) -> plt.Figure:
    ranks = np.arange(8)
    m = [metric(capacity, r) for r in ranks]
    forward = np.array([x["forward_seconds"] for x in m])
    reverse = np.array([x["reverse_seconds"] for x in m])
    tape = np.array([x["tape_memory_bytes"] for x in m]) / 2**30
    boundary = np.array([x["boundary_bytes"] for x in m]) / 2**20
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, y, title, ylabel, color in [
        (axes[0,0], forward, "A. Forward time", "Seconds", COLORS["blue"]),
        (axes[0,1], reverse, "B. Reverse time", "Seconds", COLORS["purple"]),
        (axes[1,0], tape, "C. Accounted tape", "GiB", COLORS["green"]),
        (axes[1,1], boundary, "D. Boundary payload", "MiB", COLORS["orange"]),
    ]:
        ax.bar(ranks, y, color=color); ax.set_xticks(ranks); ax.set_xlabel("Rank"); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(axis="y")
    fig.suptitle("Per-rank load and memory profile: 256q, p16, χ512", fontsize=15, weight="bold")
    return fig


def convergence_figure(payloads: list[tuple[str, dict]]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    for label, p in payloads:
        losses = np.array(p["rank_records"][0]["training"]["losses"])
        axes[0].plot(np.arange(1, len(losses)+1), losses, label=label, linewidth=1.8)
        exact = p.get("exact_ground_energy")
        if exact is not None:
            rel = np.abs(losses-exact)/abs(exact)
            axes[1].semilogy(np.arange(1, len(losses)+1), rel, label=label, linewidth=1.8)
    axes[0].set_xlabel("Optimizer step"); axes[0].set_ylabel("Energy"); axes[0].grid(); axes[0].legend(frameon=False, fontsize=8); axes[0].set_title("A. Variational energy")
    axes[1].axhline(1e-3, linestyle="--", color=COLORS["gray"], label="1e-3 target")
    axes[1].set_xlabel("Optimizer step"); axes[1].set_ylabel("Relative absolute energy error"); axes[1].grid(); axes[1].legend(frameon=False, fontsize=8); axes[1].set_title("B. Exact-reference error")
    fig.suptitle("Heisenberg VQE convergence: success and optimization limits", fontsize=15, weight="bold")
    return fig


def accuracy_summary(payloads: list[tuple[str, dict]]) -> plt.Figure:
    labels, errors, bests, exacts = [], [], [], []
    for label, p in payloads:
        losses = p["rank_records"][0]["training"]["losses"]
        exact = p["exact_ground_energy"]
        best = min(losses)
        labels.append(label); bests.append(best); exacts.append(exact); errors.append(abs(best-exact)/abs(exact))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    bars = axes[0].bar(labels, errors, color=[COLORS["green"], COLORS["orange"], COLORS["red"]])
    axes[0].set_yscale("log"); axes[0].axhline(1e-3, linestyle="--", color=COLORS["gray"]); axes[0].set_ylabel("Relative absolute error"); axes[0].grid(axis="y"); axes[0].set_title("A. Accuracy versus system size")
    for bar, value in zip(bars, errors): axes[0].text(bar.get_x()+bar.get_width()/2, value*1.25, f"{value:.2e}", ha="center", fontsize=9)
    x=np.arange(len(labels)); w=.35
    axes[1].bar(x-w/2, exacts, w, label="Exact", color=COLORS["dark"])
    axes[1].bar(x+w/2, bests, w, label="Best VQE", color=COLORS["blue"])
    axes[1].set_xticks(x, labels); axes[1].set_ylabel("Energy"); axes[1].grid(axis="y"); axes[1].legend(frameon=False); axes[1].set_title("B. Exact and variational energies")
    fig.suptitle("Scientific accuracy hierarchy (cutoff=0 for N=2/4/8)", fontsize=15, weight="bold")
    return fig


def circuit_scaling() -> plt.Figure:
    n = np.arange(16, 1025, 16)
    depths = [2, 8, 16, 32]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for p, color in zip(depths, [COLORS["cyan"], COLORS["green"], COLORS["blue"], COLORS["purple"]]):
        axes[0].plot(n, 3*p*(n-1), label=f"p={p}", color=color)
        axes[1].plot(n, 6*p*np.ones_like(n), label=f"p={p}", color=color)
    axes[0].scatter([256], [12240], s=90, color=COLORS["red"], zorder=5)
    axes[0].annotate("Measured 256q/p16\n12,240 gates", (256,12240), xytext=(340,10000), arrowprops={"arrowstyle":"->"})
    axes[0].set_xlabel("Qubits"); axes[0].set_ylabel("Two-qubit gate count"); axes[0].grid(); axes[0].legend(frameon=False); axes[0].set_title("A. Gate count = 3p(N−1)")
    axes[1].scatter([256], [96], s=90, color=COLORS["red"], zorder=5)
    axes[1].set_xlabel("Qubits"); axes[1].set_ylabel("Two-qubit gate depth"); axes[1].grid(); axes[1].legend(frameon=False); axes[1].set_title("B. Brickwork depth = 6p")
    fig.suptitle("Heisenberg HVA circuit-scale model", fontsize=15, weight="bold")
    return fig


def communication_topology(capacity: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(15, 4.8), constrained_layout=True)
    ax.axis("off")
    colors = plt.cm.Blues(np.linspace(.35,.85,8))
    for rank in range(8):
        x=.03+rank*.12
        box=FancyBboxPatch((x,.38),.095,.23,boxstyle="round,pad=.012",facecolor=colors[rank],edgecolor=COLORS["blue"])
        ax.add_patch(box); ax.text(x+.0475,.515,f"GPU {rank}",ha="center",weight="bold",color="white")
        ax.text(x+.0475,.435,f"q{rank*32}–q{(rank+1)*32-1}",ha="center",fontsize=9,color="white")
        if rank<7:
            ax.add_patch(FancyArrowPatch((x+.096,.495),(x+.119,.495),arrowstyle="<->",mutation_scale=14,color=COLORS["orange"],linewidth=2))
    ax.text(.5,.78,"Forward tensor exchange + reverse boundary-adjoint exchange",ha="center",fontsize=13,weight="bold")
    ax.text(.5,.18,f"Measured bidirectional boundary traffic: {capacity['communication_bytes']/2**30:.2f} GiB   |   NCCL batched isend/irecv   |   intra-node NVSwitch route",ha="center",fontsize=12)
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_title("Site-sharded ownership and executed boundary communication",loc="left",fontsize=15,weight="bold")
    return fig


def evidence_matrix() -> plt.Figure:
    rows = ["Correctness", "Full gradient", "Optimizer update", "True sharding", "Capacity expansion", "Strong scaling", "Physics convergence", "Multi-node"]
    cols = ["Measured", "Status / limitation"]
    status = [
        (1,"N=2 relative error 9.38e-6"), (1,"Reverse VJP executed"),
        (1,"Owner-sharded Adam executed"), (1,"Site/bond sharded across ranks"),
        (1,"Single A800 OOM; 8 GPUs complete"), (1,"10.14x on 8 GPUs for fixed 192q/p2/chi512"),
        (.45,"N=4/N=8 optimizer remains limited"), (0,"Not measured in this environment"),
    ]
    fig, ax = plt.subplots(figsize=(12,6), constrained_layout=True); ax.axis("off")
    for i,(row,(score,text)) in enumerate(zip(rows,status)):
        y=1-(i+1)/9
        ax.text(.03,y,row,va="center",weight="bold")
        color=COLORS["green"] if score>.8 else COLORS["orange"] if score>.2 else COLORS["red"]
        ax.add_patch(FancyBboxPatch((.25,y-.035),.12,.07,boxstyle="round,pad=.008",facecolor=color,edgecolor="none"))
        ax.text(.31,y,"PASS" if score>.8 else "PARTIAL" if score>.2 else "OPEN",ha="center",va="center",color="white",weight="bold")
        ax.text(.41,y,text,va="center")
        ax.plot([.02,.98],[y-.06,y-.06],color=COLORS["light"],linewidth=.8)
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_title("Evidence coverage and honest claim boundary",loc="left",fontsize=15,weight="bold")
    return fig


def main() -> None:
    configure()
    capacity = load("release_capacity_n256_p16_chi512_g8.json")
    baseline = load("release_capacity_n256_p16_chi512_g1_baseline.json")
    scaling = [load(f"scaling_large_n192_p2_chi512_g{g}.json") for g in (1,2,4,8)]
    checkpoint_a = load("checkpoint_ab_n64_p2_chi512_A_recompute.json")
    checkpoint_b = load("checkpoint_ab_n64_p2_chi512_B_save4g.json")
    n2 = load("accuracy_n2_phase_hva_normalized_100.json")
    n4 = load("accuracy_n4_p8_bond_resolved_exact_300.json")
    n8 = load("accuracy_n8_p16_chi16_exact_150.json")
    with PdfPages(OUT / "FlagQuantum_Distributed_MPS_Figure_Book.pdf") as book:
        figures = [
            (overview(capacity,baseline), "fig01_capability_overview"),
            (capacity_memory(capacity,baseline), "fig02_capacity_memory"),
            (scaling_figure(scaling), "fig03_strong_scaling"),
            (checkpoint_figure(checkpoint_a,checkpoint_b), "fig04_checkpoint_pareto"),
            (rank_profile(capacity), "fig05_per_rank_profile"),
            (convergence_figure([("N=2 phase-HVA",n2),("N=4 bond-HVA",n4),("N=8 HVA",n8)]), "fig06_vqe_convergence"),
            (accuracy_summary([("N=2",n2),("N=4",n4),("N=8",n8)]), "fig07_accuracy_hierarchy"),
            (circuit_scaling(), "fig08_circuit_scale"),
            (communication_topology(capacity), "fig09_sharding_communication"),
            (evidence_matrix(), "fig10_evidence_matrix"),
        ]
        for fig, stem in figures:
            save(fig, stem, book)
    manifest = {
        "figure_count": 10,
        "formats": ["png", "svg", "pdf"],
        "figure_book": "FlagQuantum_Distributed_MPS_Figure_Book.pdf",
        "figures": [f"fig{i:02d}" for i in range(1,11)],
        "source_data": str(DATA),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

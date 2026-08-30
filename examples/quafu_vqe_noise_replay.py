"""Replay a Quafu VQE trajectory with a calibration-derived noise model."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import torch

import flagquantum as fq


def logical_ansatz(theta: torch.Tensor | float, *, x_basis: bool = False) -> fq.Circuit:
    circuit = fq.Circuit(2)
    circuit.ry(0, theta).ry(1, theta)
    if x_basis:
        circuit.h(0).h(1)
    return circuit


def baihua_transpiled_ansatz(
    theta: torch.Tensor | float, *, x_basis: bool = False
) -> fq.Circuit:
    """Reproduce the one-U-per-wire circuit emitted by Baihua's compiler."""

    circuit = fq.Circuit(2)
    if x_basis:
        physical_theta = math.pi / 2 - float(theta)
        return circuit.u3(0, physical_theta, 0.0, math.pi).u3(
            1, physical_theta, 0.0, math.pi
        )
    return circuit.u3(0, theta, 0.0, 0.0).u3(1, theta, 0.0, 0.0)


def basis_expectation(probabilities: torch.Tensor, wires: tuple[int, ...]) -> float:
    values = probabilities.reshape(-1).tolist()
    n_wires = int(round(math.log2(len(values))))
    total = 0.0
    for index, probability in enumerate(values):
        bits = f"{index:0{n_wires}b}"
        sign = math.prod(1 if bits[wire] == "0" else -1 for wire in wires)
        total += float(probability) * sign
    return total


def energy_standard_error(source: dict) -> float:
    """Shot-noise standard error, retaining within-basis covariance."""

    def variance_of_mean(counts: dict[str, int], value) -> float:
        shots = sum(counts.values())
        mean = sum(count * value(bits) for bits, count in counts.items()) / shots
        variance = sum(
            count * (value(bits) - mean) ** 2 for bits, count in counts.items()
        ) / max(1, shots - 1)
        return variance / shots

    z_counts = source["quafu_counts"]["z_basis"]
    x_counts = source["quafu_counts"]["x_basis"]
    z_variance = variance_of_mean(
        z_counts,
        lambda bits: -(1 if bits[0] == bits[1] else -1),
    )
    x_variance = variance_of_mean(
        x_counts,
        lambda bits: -sum(1 if bit == "0" else -1 for bit in bits),
    )
    return math.sqrt(z_variance + x_variance)


def noisy_basis_probabilities(
    theta: float,
    *,
    x_basis: bool,
    model: fq.NoiseModel,
    physical_transpilation: bool = True,
) -> torch.Tensor:
    circuit = (
        baihua_transpiled_ansatz(theta, x_basis=x_basis)
        if physical_transpilation
        else logical_ansatz(theta, x_basis=x_basis)
    )
    rho = fq.noisy_density_matrix(circuit, model)
    probabilities = rho.diagonal(dim1=-2, dim2=-1).real
    return model.apply_readout_probabilities(probabilities, n_wires=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("artifacts/legacy/quafu_vqe/comparison.json"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("artifacts/legacy/quafu_vqe/baihua_q123_q124_calibration.json"),
    )
    parser.add_argument(
        "--spam",
        type=Path,
        default=Path("artifacts/legacy/quafu_vqe/spam_calibration.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/development/quafu_vqe/noise_replay.json"),
    )
    parser.add_argument(
        "--logical-gates",
        action="store_true",
        help="Replay pre-transpilation Ry/H gates (legacy comparison only).",
    )
    args = parser.parse_args()

    comparison = json.loads(args.comparison.read_text())
    calibration = json.loads(args.calibration.read_text())
    spam = json.loads(args.spam.read_text())
    model = fq.quafu_noise_model_from_chip_info(
        calibration,
        physical_qubits=spam["physical_qubits"],
        readout_confusion_matrices=spam.get("readout_confusion_matrices"),
        correlated_readout_confusion_matrix=spam.get(
            "correlated_readout_confusion_matrix"
        ),
    )

    points = []
    for source in comparison["points"]:
        z_probabilities = noisy_basis_probabilities(
            source["theta"],
            x_basis=False,
            model=model,
            physical_transpilation=not args.logical_gates,
        )
        x_probabilities = noisy_basis_probabilities(
            source["theta"],
            x_basis=True,
            model=model,
            physical_transpilation=not args.logical_gates,
        )
        zz = basis_expectation(z_probabilities, (0, 1))
        x0 = basis_expectation(x_probabilities, (0,))
        x1 = basis_expectation(x_probabilities, (1,))
        points.append(
            {
                "iteration": source["iteration"],
                "theta": source["theta"],
                "statevector_energy": source["statevector_energy"],
                "calibrated_noise_energy": -zz - x0 - x1,
                "quafu_energy": source["quafu_energy"],
                "quafu_standard_error": energy_standard_error(source),
                "calibrated_expectations": {"ZZ": zz, "X0": x0, "X1": x1},
            }
        )

    noise_errors = [
        point["calibrated_noise_energy"] - point["quafu_energy"] for point in points
    ]
    sv_errors = [
        point["statevector_energy"] - point["quafu_energy"] for point in points
    ]
    payload = {
        "schema": "flagquantum_quafu_vqe_noise_replay_v2",
        "ideal_simulation_engine": "FlagQuantum Statevector",
        "simulation_engine": "FlagQuantum NoiseModel",
        "noise_calibration_source": "Quafu Baihua Q123/Q124",
        "noise_replay_circuit": (
            "Baihua-transpiled physical U gates"
            if not args.logical_gates
            else "logical Ry/H gates"
        ),
        "noise_model_identity": model.identity,
        "device_profile": model.device_profile.to_dict(),
        "physical_qubits": spam["physical_qubits"],
        "calibration_time": calibration["calibration_time"],
        "spam_task_ids": {
            key: value["task_id"] for key, value in spam["prepared_results"].items()
        },
        "metrics": {
            "calibrated_noise_vs_quafu_mae": statistics.mean(map(abs, noise_errors)),
            "statevector_vs_quafu_mae": statistics.mean(map(abs, sv_errors)),
            "calibrated_noise_vs_quafu_rmse": math.sqrt(
                statistics.mean(value * value for value in noise_errors)
            ),
            "statevector_vs_quafu_rmse": math.sqrt(
                statistics.mean(value * value for value in sv_errors)
            ),
        },
        "points": points,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    import matplotlib.pyplot as plt

    iterations = [point["iteration"] for point in points]
    hardware = [point["quafu_energy"] for point in points]
    hardware_ci95 = [1.96 * point["quafu_standard_error"] for point in points]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "svg.fonttype": "none",
        }
    )
    figure = plt.figure(figsize=(7.2, 4.35), facecolor="white")
    grid = figure.add_gridspec(1, 2, width_ratios=(3.45, 1.18), wspace=0.13)
    left = grid[0, 0].subgridspec(2, 1, height_ratios=(3.0, 1), hspace=0.08)
    axis = figure.add_subplot(left[0, 0])
    residual = figure.add_subplot(left[1, 0], sharex=axis)
    information = figure.add_subplot(grid[0, 1])
    information.axis("off")
    axis.plot(
        iterations,
        [point["statevector_energy"] for point in points],
        color="#0072B2",
        marker="o",
        markersize=3.2,
        linewidth=1.25,
        label="Statevector",
        zorder=5,
    )
    axis.plot(
        iterations,
        [point["calibrated_noise_energy"] for point in points],
        color="#D55E00",
        marker="^",
        markersize=3.5,
        linewidth=1.25,
        linestyle="--",
        label="NoiseModel",
        zorder=6,
    )
    axis.errorbar(
        iterations,
        hardware,
        yerr=hardware_ci95,
        color="#009E73",
        marker="s",
        markerfacecolor="white",
        markeredgewidth=0.8,
        markersize=3.0,
        linewidth=0.75,
        elinewidth=0.65,
        capsize=1.8,
        label="Baihua (95% shot CI)",
        zorder=2,
    )
    axis.axhline(
        -2.0,
        color="black",
        linewidth=0.9,
        linestyle=":",
        label="Exact ansatz minimum",
        zorder=1,
    )
    axis.set_ylabel("Energy")
    axis.set_xlim(-0.35, 19.35)
    axis.set_ylim(-2.025, -1.285)
    axis.set_xticks(range(0, 20, 2))
    axis.tick_params(axis="x", labelbottom=False)
    axis.text(
        -0.12, 1.02, "(a)", transform=axis.transAxes, fontweight="bold", fontsize=8.5
    )
    metrics = payload["metrics"]
    handles, labels = axis.get_legend_handles_labels()
    information.text(
        0.02,
        0.96,
        "Legend",
        transform=information.transAxes,
        fontsize=7.2,
        fontweight="bold",
        color="black",
        va="top",
    )
    information.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.93),
        frameon=False,
        fontsize=6.6,
        handlelength=2.4,
        labelspacing=0.55,
        borderaxespad=0,
    )
    q123 = calibration["qubits_info"]["Q123"]
    q124 = calibration["qubits_info"]["Q124"]
    coupler = next(iter(calibration["couplers_info"].values()))
    shots_label = f"{comparison['shots_per_basis']:,}".replace(",", "\u2009")
    readout_shots_label = f"{spam['shots_per_state']:,}".replace(",", "\u2009")
    calibration_text = (
        "Baihua calibration\n"
        f"{calibration['calibration_time']}\n\n"
        f"Q123:  $T_1$={q123['T1']:.3f} μs\n"
        f"          $T_2$={q123['T2']:.3f} μs,  $F_{{1q}}$={q123['fidelity']:.3f}\n"
        f"Q124:  $T_1$={q124['T1']:.3f} μs\n"
        f"          $T_2$={q124['T2']:.3f} μs,  $F_{{1q}}$={q124['fidelity']:.3f}\n"
        f"CZ:     $F_{{CZ}}$={coupler['fidelity']:.3f},  {coupler['length'] * 1e9:.0f} ns\n"
        f"1Q duration: {q123['length'] * 1e9:.0f} ns\n\n"
        f"Hardware: {shots_label} shots/basis\n"
        f"Readout: 4 × {readout_shots_label} shots\n"
        "NoiseModel: exact expectation"
    )
    information.text(
        0.02,
        0.58,
        calibration_text,
        transform=information.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        linespacing=1.12,
        color="black",
    )
    metric_text = (
        "NoiseModel − hardware\n"
        f"MAE = {metrics['calibrated_noise_vs_quafu_mae']:.4f}\n"
        f"RMSE = {metrics['calibrated_noise_vs_quafu_rmse']:.4f}"
    )
    information.text(
        0.02,
        0.005,
        metric_text,
        transform=information.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.8,
        linespacing=1.18,
        color="black",
    )
    axis.grid(axis="y", color="0.88", linewidth=0.45)
    axis.grid(axis="x", visible=False)
    axis.spines[["top", "right"]].set_visible(False)
    noise_residual = [
        point["calibrated_noise_energy"] - point["quafu_energy"] for point in points
    ]
    sv_residual = [
        point["statevector_energy"] - point["quafu_energy"] for point in points
    ]
    residual.axhline(0.0, color="#263238", linewidth=1.0, linestyle=(0, (4, 3)))
    residual.errorbar(
        iterations,
        noise_residual,
        yerr=hardware_ci95,
        color="#D55E00",
        marker="^",
        markersize=2.8,
        linewidth=1.0,
        linestyle="--",
        elinewidth=0.6,
        capsize=1.5,
        label="NoiseModel − hardware",
    )
    residual.plot(
        iterations,
        sv_residual,
        color="#0072B2",
        marker="o",
        markersize=2.6,
        linewidth=1.0,
        label="Statevector − hardware",
    )
    residual.set_xlabel("VQE iteration")
    residual.set_ylabel("Residual")
    residual.text(
        -0.12,
        0.93,
        "(b)",
        transform=residual.transAxes,
        fontweight="bold",
        fontsize=8.5,
    )
    residual.set_xlim(-0.35, 19.35)
    residual.set_xticks(range(0, 20, 2))
    residual.grid(axis="y", color="0.88", linewidth=0.45)
    residual.spines[["top", "right"]].set_visible(False)
    figure.subplots_adjust(left=0.085, right=0.99, bottom=0.12, top=0.98)
    curve = args.output.with_name("noise_replay_curve.png")
    figure.savefig(
        curve, dpi=220, bbox_inches="tight", facecolor=figure.get_facecolor()
    )
    figure.savefig(curve.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(curve.with_suffix(".pdf"), bbox_inches="tight")
    print(json.dumps(payload["metrics"], sort_keys=True))
    print(f"saved={args.output} curve={curve}")


if __name__ == "__main__":
    main()

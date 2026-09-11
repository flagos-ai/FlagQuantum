"""End-to-end VQE: Triton fusion, PyTorch eager, and optional JAX.

This is a local single-device product-path benchmark, not scalability evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import flagquantum as fq  # noqa: E402


def _hamiltonian(n_wires: int) -> fq.Hamiltonian:
    return fq.Hamiltonian(
        [
            *(fq.pauli_term(0.7, "ZZ", (wire, wire + 1)) for wire in range(n_wires - 1)),
            *(fq.pauli_term(-0.25, "X", (wire,)) for wire in range(n_wires)),
            fq.pauli_term(0.05, "Z", (0,)),
        ]
    )


def _build_module(
    initial: torch.Tensor,
    *,
    n_wires: int,
    local_depth: int,
    hamiltonian,
    backend: str = "pytorch",
) -> fq.Module:
    def ansatz(parameters: torch.Tensor) -> fq.Circuit:
        circuit = fq.Circuit(n_wires, device=parameters.device)
        for wire in range(n_wires):
            circuit.h(wire)
        for layer in range(local_depth):
            for wire in range(n_wires):
                circuit.rx(wire, theta=parameters[layer, wire, 0])
                circuit.rz(wire, theta=parameters[layer, wire, 1])
        for wire in range(n_wires - 1):
            circuit.cx(wire, wire + 1)
        return circuit

    return fq.Module(
        ansatz,
        tuple(initial.shape),
        init=initial.detach().clone(),
        device=initial.device,
        hamiltonian=hamiltonian,
        policy=fq.RuntimePolicy(
            backend=backend,
            mode="statevector",
            observable="hamiltonian",
            allow_backend_fallback=False,
        ),
    )


def _train(
    module: fq.Module,
    *,
    iterations: int,
    lr: float,
    label: str,
    fused: bool = False,
):
    os.environ["FQ_TRITON_SINGLE_QUBIT_LOOP"] = "1" if fused else "0"
    optimizer = torch.optim.Adam(module.parameters(), lr=lr)
    cumulative = []
    energies = []
    peak_memory_bytes = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    for index in range(iterations):
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        loss = module().sum()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        cumulative.append(time.perf_counter() - started)
        energies.append(float(loss.detach()))
        peak_memory_bytes.append(int(torch.cuda.max_memory_allocated()))
        if index == 0 or (index + 1) % max(1, iterations // 20) == 0:
            print(
                f"heartbeat backend={label} iteration={index + 1}/{iterations} "
                f"elapsed={cumulative[-1]:.3f}s "
                f"peak_gib={peak_memory_bytes[-1] / 2**30:.3f}",
                flush=True,
            )
    return {
        "cumulative_seconds": cumulative,
        "energies": energies,
        "peak_memory_bytes": peak_memory_bytes,
    }


def _warmup(module: fq.Module, *, fused: bool = False) -> float:
    os.environ["FQ_TRITON_SINGLE_QUBIT_LOOP"] = "1" if fused else "0"
    module.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    module().sum().backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    module.zero_grad(set_to_none=True)
    return elapsed


def _write_chart(
    *,
    iterations: int,
    series: tuple[tuple[str, list[float], str], ...],
    title: str,
    y_label: str,
    annotation: str,
    path: Path,
) -> None:
    width, height = 900, 560
    left, right, top, bottom = 92, 35, 108, 76
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_max = max(max(values) for _, values, _ in series) * 1.08

    def points(values):
        return " ".join(
            f"{left + plot_w * index / max(1, iterations - 1):.2f},"
            f"{top + plot_h * (1.0 - value / y_max):.2f}"
            for index, value in enumerate(values)
        )

    grid = []
    for tick in range(6):
        y_value = y_max * tick / 5
        y = top + plot_h * (1.0 - tick / 5)
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" '
            'stroke="#d9dee8" stroke-width="1"/>'
            f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" '
            f'font-size="14" fill="#445">{y_value:.2f}</text>'
        )
    for tick in range(6):
        iteration = 1 + round((iterations - 1) * tick / 5)
        x = left + plot_w * (iteration - 1) / max(1, iterations - 1)
        grid.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 28}" text-anchor="middle" '
            f'font-size="14" fill="#445">{iteration}</text>'
        )
    polylines = "".join(
        f'<polyline points="{points(values)}" fill="none" stroke="{color}" '
        'stroke-width="3"/>'
        for _, values, color in series
    )
    legends = "".join(
        f'<line x1="{left + 18}" y1="{55 + offset * 23}" '
        f'x2="{left + 53}" y2="{55 + offset * 23}" stroke="{color}" '
        f'stroke-width="3"/><text x="{left + 62}" '
        f'y="{60 + offset * 23}" font-size="14">{label}</text>'
        for offset, (label, _, color) in enumerate(series)
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="27" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="600">{title}</text>
<g font-family="sans-serif">{''.join(grid)}
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>
{polylines}
<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-size="16">Number of Iterations</text>
<text x="22" y="{top + plot_h / 2}" text-anchor="middle" font-size="16" transform="rotate(-90 22 {top + plot_h / 2})">{y_label}</text>
{legends}
<text x="{left + plot_w - 8}" y="78" text-anchor="end" font-size="14" font-weight="600">{annotation}</text>
</g></svg>'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-wires", type=int, default=16)
    parser.add_argument("--local-depth", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--svg-output", type=Path, required=True)
    parser.add_argument("--speedup-svg-output", type=Path)
    parser.add_argument("--memory-svg-output", type=Path)
    parser.add_argument(
        "--include-jax",
        action="store_true",
        help="measure the same product path through the JAX backend",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    torch.manual_seed(17)
    initial = 0.05 * torch.randn(
        args.local_depth, args.n_wires, 2, device="cuda"
    )
    hamiltonian = _hamiltonian(args.n_wires)
    triton_module = _build_module(
        initial,
        n_wires=args.n_wires,
        local_depth=args.local_depth,
        hamiltonian=hamiltonian,
    )
    eager_module = _build_module(
        initial,
        n_wires=args.n_wires,
        local_depth=args.local_depth,
        hamiltonian=hamiltonian,
    )
    jax_module = (
        _build_module(
            initial,
            n_wires=args.n_wires,
            local_depth=args.local_depth,
            hamiltonian=hamiltonian,
            backend="jax",
        )
        if args.include_jax
        else None
    )
    torch.testing.assert_close(
        triton_module.parameters_tensor, eager_module.parameters_tensor
    )
    triton_startup_seconds = _warmup(triton_module, fused=True)
    eager_startup_seconds = _warmup(eager_module, fused=False)
    if jax_module is not None:
        torch.testing.assert_close(
            triton_module.parameters_tensor, jax_module.parameters_tensor
        )
        jax_startup_seconds = _warmup(jax_module)
    else:
        jax_startup_seconds = None
    triton = _train(
        triton_module,
        iterations=args.iterations,
        lr=args.lr,
        label="triton",
        fused=True,
    )
    eager = _train(
        eager_module, iterations=args.iterations, lr=args.lr, label="eager"
    )
    jax_result = (
        _train(jax_module, iterations=args.iterations, lr=args.lr, label="jax")
        if jax_module is not None
        else None
    )
    os.environ["FQ_TRITON_SINGLE_QUBIT_LOOP"] = "1"
    triton_runtime = dict(triton_module.execute().runtime)
    os.environ["FQ_TRITON_SINGLE_QUBIT_LOOP"] = "0"
    eager_runtime = dict(eager_module.execute().runtime)
    jax_runtime = dict(jax_module.execute().runtime) if jax_module is not None else None
    assert triton_runtime["triton_single_qubit_loop_regions"] == args.n_wires
    assert eager_runtime["triton_single_qubit_loop_regions"] == 0
    payload = {
        "benchmark": "flagquantum_vqe_triton_ir_runtime",
        "benchmark_evidence_class": "local",
        "non_release_evidence": True,
        "release_gate_allowed": False,
        "distribution_semantics": "single_device_fast_path",
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "task": "end_to_end_vqe_training",
        "api": "fq.Module -> FlagQuantum IR -> statevector -> Hamiltonian",
        "hamiltonian": "0.7 sum ZZ - 0.25 sum X + 0.05 Z0",
        "optimizer": "torch.optim.Adam",
        "n_wires": args.n_wires,
        "local_rx_rz_depth": args.local_depth,
        "parameters": int(initial.numel()),
        "iterations": args.iterations,
        "learning_rate": args.lr,
        "timing_boundary": {
            "training_curves": "post_warmup_steady_state",
            "compile_and_first_execution_excluded": True,
            "startup_seconds": {
                "triton": triton_startup_seconds,
                "eager": eager_startup_seconds,
                "jax": jax_startup_seconds,
            },
            "startup_measurement_note": (
                "One first forward+backward execution; may use persistent compiler caches."
            ),
        },
        "triton": triton,
        "eager": eager,
        "jax": jax_result,
        "triton_runtime": triton_runtime,
        "eager_runtime": eager_runtime,
        "jax_runtime": jax_runtime,
        "speedup_eager_over_triton": (
            eager["cumulative_seconds"][-1] / triton["cumulative_seconds"][-1]
        ),
        "cumulative_speedup_factor": [
            eager_time / triton_time
            for eager_time, triton_time in zip(
                eager["cumulative_seconds"], triton["cumulative_seconds"]
            )
        ],
        "final_energy_abs_delta": abs(
            eager["energies"][-1] - triton["energies"][-1]
        ),
        "limitations": [
            "single-GPU local engineering evidence",
            "fusion applies only to validated RX/RZ-only commuting IR regions",
            "does not claim arbitrary VQE or distributed scalability speedup",
        ],
    }
    if jax_result is not None:
        payload["runtime_ratio_jax_over_triton"] = (
            jax_result["cumulative_seconds"][-1]
            / triton["cumulative_seconds"][-1]
        )
        payload["cumulative_jax_over_triton_factor"] = [
            jax_time / triton_time
            for jax_time, triton_time in zip(
                jax_result["cumulative_seconds"], triton["cumulative_seconds"]
            )
        ]
        payload["final_energy_abs_delta_jax_triton"] = abs(
            jax_result["energies"][-1] - triton["energies"][-1]
        )
        payload["limitations"].append(
            "JAX GPU memory is not reported by torch.cuda.max_memory_allocated"
        )
    encoded = json.dumps(payload, indent=2)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(encoded + "\n", encoding="utf-8")
    _write_chart(
        iterations=args.iterations,
        series=(
            (
                "FlagQuantum IR + Triton fusion",
                triton["cumulative_seconds"],
                "#2468b4",
            ),
            ("FlagQuantum PyTorch eager", eager["cumulative_seconds"], "#d24b40"),
        )
        + (
            (("FlagQuantum JAX (jit)", jax_result["cumulative_seconds"], "#238b57"),)
            if jax_result is not None
            else ()
        ),
        title="FlagQuantum VQE: cumulative training runtime (post-warmup)",
        y_label="Total Runtime (seconds)",
        annotation=(
            f"eager/Triton: {payload['speedup_eager_over_triton']:.2f}x; "
            f"JAX/Triton: {payload['runtime_ratio_jax_over_triton']:.2f}x"
            if jax_result is not None
            else (
                f"eager/Triton: {payload['speedup_eager_over_triton']:.2f}x; "
                "JAX not measured"
            )
        ),
        path=args.svg_output,
    )
    if args.speedup_svg_output is not None:
        _write_chart(
            iterations=args.iterations,
            series=(
                (
                    "PyTorch eager / IR+Triton",
                    payload["cumulative_speedup_factor"],
                    "#6b3fa0",
                ),
            )
            + (
                (
                    (
                        "JAX jit / IR+Triton",
                        payload["cumulative_jax_over_triton_factor"],
                        "#238b57",
                    ),
                )
                if jax_result is not None
                else ()
            ),
            title="FlagQuantum VQE: post-warmup runtime ratio vs IR+Triton",
            y_label="Runtime Ratio vs IR+Triton (>1: Triton faster)",
            annotation=(
                f"final eager: {payload['speedup_eager_over_triton']:.2f}x; "
                f"JAX: {payload['runtime_ratio_jax_over_triton']:.2f}x"
                if jax_result is not None
                else f"final eager: {payload['speedup_eager_over_triton']:.2f}x"
            ),
            path=args.speedup_svg_output,
        )
    if args.memory_svg_output is not None:
        _write_chart(
            iterations=args.iterations,
            series=(
                (
                    "FlagQuantum IR + Triton fusion",
                    [value / 2**30 for value in triton["peak_memory_bytes"]],
                    "#2468b4",
                ),
                (
                    "FlagQuantum PyTorch eager",
                    [value / 2**30 for value in eager["peak_memory_bytes"]],
                    "#d24b40",
                ),
            ),
            title="FlagQuantum VQE: per-step peak GPU memory",
            y_label="Peak Allocated GPU Memory (GiB)",
            annotation="torch.cuda.max_memory_allocated",
            path=args.memory_svg_output,
        )
    print(encoded)


if __name__ == "__main__":
    main()

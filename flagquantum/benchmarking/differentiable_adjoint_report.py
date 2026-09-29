"""Markdown rendering for the method-matched adjoint benchmark track."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_WORKLOAD_LABELS = {
    "hardware_efficient_vqe": "Hardware-efficient VQE",
    "qaoa_path_maxcut": "QAOA path MaxCut",
}


def render_adjoint_markdown(payload: Mapping[str, Any], *, artifact_name: str) -> str:
    """Render the method-matched adjoint comparison without changing claims."""

    cases = payload["cases"]
    methodology = payload["methodology"]
    lines = [
        "# Adjoint differentiable simulator corpus (Apple arm64 CPU)",
        "",
        f"This report is generated from [`{artifact_name}`]({artifact_name}). It is a",
        "separate method-matched track from the backpropagation corpus: FlagQuantum's",
        "fused native reversible adjoint is compared with its Python fallback and",
        "PennyLane Lightning's adjoint.",
        "Both compute one exact weighted Z/ZZ expectation and its gradient with respect",
        "to every circuit parameter; construction and optimizer updates are excluded.",
        "",
        "## Results",
        "",
        "| Workload | Qubits | Gates | Params | FQ forward (ms) | FQ backward (ms) | FQ total (ms) | Python backward (ms) | Python total (ms) | Native backward speedup | PennyLane Lightning total (ms) | PennyLane Lightning / FlagQuantum total | Max gradient error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def milliseconds(engine: Mapping[str, Any], metric: str) -> float:
        return 1000.0 * float(engine[metric]["median_seconds"])

    ratios = []
    backward_speedups = []
    total_speedups = []
    errors = []
    for case in cases:
        workload = case["workload"]
        native = case["engines"]["flagquantum_adjoint"]
        python_fallback = case["engines"]["flagquantum_adjoint_python_fallback"]
        external = case["engines"]["pennylane_lightning_adjoint"]
        ratio = case["comparison"]["engine_over_flagquantum_median"][
            "pennylane_lightning_adjoint"
        ]["value_and_grad"]
        error = case["correctness"]["engines"]["pennylane_lightning_adjoint"][
            "gradient_max_abs_error"
        ]
        ratios.append(ratio)
        backward_speedup = (
            python_fallback["backward"]["median_seconds"]
            / native["backward"]["median_seconds"]
        )
        total_speedup = (
            python_fallback["value_and_grad"]["median_seconds"]
            / native["value_and_grad"]["median_seconds"]
        )
        backward_speedups.append(backward_speedup)
        total_speedups.append(total_speedup)
        errors.append(error)
        values = (
            _WORKLOAD_LABELS[workload["name"]],
            workload["n_wires"],
            workload["gate_count"],
            workload["parameter_count"],
            f"{milliseconds(native, 'forward'):.3f}",
            f"{milliseconds(native, 'backward'):.3f}",
            f"{milliseconds(native, 'value_and_grad'):.3f}",
            f"{milliseconds(python_fallback, 'backward'):.3f}",
            f"{milliseconds(python_fallback, 'value_and_grad'):.3f}",
            f"{backward_speedup:.2f}x",
            f"{milliseconds(external, 'value_and_grad'):.3f}",
            f"{ratio:.2f}x",
            f"{error:.3e}",
        )
        lines.append("| " + " | ".join(map(str, values)) + " |")

    lines.extend(
        (
            "",
            "A PennyLane Lightning/FlagQuantum ratio above one means FlagQuantum was "
            "faster. Peak RSS was",
            "not measured in this timing run. These are local, non-release",
            "single-device results, not a universal framework ranking or scaling claim.",
            "",
            "## Meaning and current level",
            "",
            f"All {len(cases)} cases passed the 1e-9 value/gradient tolerance; maximum "
            f"gradient error was {max(errors):.3e}. One-pass forward RZZ segments and "
            "native disjoint one-qubit H/rotation layers complement the fused "
            "RX/RY/RZ/RZZ adjoint operators, commuting shared-parameter RZZ segments, "
            "shared rotation-layer adjoints with native shared-gradient reduction, "
            "allocation-free reverse H blocks, same-wire forward "
            "composition, CX-sequence permutations, reuse of the observable diagonal, "
            "bounded structural planning reuse, cached accelerator capability probes, "
            "parameter-free adjoint IR template reuse, once-per-parameter binding "
            "validation, and analytic Pauli-rotation VJPs. The native adjoint operators "
            "accelerated backward by "
            f"{min(backward_speedups):.2f}x to {max(backward_speedups):.2f}x and total "
            f"value-and-gradient by {min(total_speedups):.2f}x to "
            f"{max(total_speedups):.2f}x. Total PennyLane Lightning/FlagQuantum ratios "
            "ranged from "
            f"{min(ratios):.2f}x to {max(ratios):.2f}x on this host. Hardware-efficient "
            "VQE stresses many independent rotation gradients; QAOA stresses repeated "
            "shared parameters. This establishes functional local adjoint support for real "
            "weighted Z/ZZ Hamiltonians at complex128, not a universal performance or "
            "general Pauli-support claim. The QAOA path now reduces repeated planning and "
            "shared-parameter bookkeeping overhead; repeated optimization steps also "
            "avoid rebuilding detached IR snapshots while preserving fresh parameter "
            "values and autograd contexts.",
            "",
            "## FlagQuantum example",
            "",
            "```python",
            "import torch",
            "import flagquantum as fq",
            "from flagquantum import algorithms as fqa",
            "",
            "theta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)",
            "circuit = fq.Circuit(3, dtype=torch.complex128).ry(0, theta).cx(0, 1)",
            "hamiltonian = fqa.Hamiltonian((",
            '    fqa.pauli_term(0.7, "ZZ", (0, 1)),',
            '    fqa.pauli_term(0.2, "Z", (2,)),',
            "))",
            'energy = hamiltonian.expectation(circuit, differentiation="adjoint")',
            "energy.backward()",
            "print(energy.item(), theta.grad)",
            "```",
            "",
            "## Reproduction",
            "",
            "```bash",
            "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\",
            "flagquantum-benchmark run differentiable_simulator_corpus \\",
            "  --engines flagquantum_adjoint "
            "flagquantum_adjoint_python_fallback pennylane_lightning_adjoint \\",
            "  --workloads hardware_efficient_vqe qaoa_path_maxcut \\",
            "  --n-wires " + " ".join(map(str, payload["n_wires"])) + " \\",
            f"  --layers {payload['layers']} --threads 1 --warmup {methodology['warmup']} \\",
            f"  --iterations {methodology['iterations']} --calls-per-sample "
            f"{methodology['calls_per_sample']} \\",
            "  --skip-memory-probe \\",
            f"  --json-output {artifact_name} --markdown-output REPORT.md",
            "```",
            "",
        )
    )
    return "\n".join(lines)

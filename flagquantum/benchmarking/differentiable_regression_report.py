"""Markdown reports for focused differentiable performance regressions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_WORKLOAD_LABELS = {
    "hardware_efficient_vqe": "Hardware-efficient VQE",
    "qaoa_path_maxcut": "QAOA path MaxCut",
}


def _milliseconds(engine: Mapping[str, Any], metric: str) -> float:
    return 1000 * float(engine[metric]["median_seconds"])


def render_forward_cx_markdown(
    payload: Mapping[str, Any], *, artifact_name: str
) -> str:
    """Render the paired forward-CX native/rollback comparison."""

    methodology = payload["methodology"]
    lines = [
        "# Native CPU forward CX gather comparison",
        "",
        f"This report is generated from [`{artifact_name}`]({artifact_name}). It compares",
        "FlagQuantum's reusable-output native CPU CX gather with the same adjoint path",
        "using allocating PyTorch `index_select`. The value and full parameter gradient",
        "must agree exactly; ratios above 1 mean the native path is faster.",
        "",
        "| Workload | Qubits | CX gates | Native forward (ms) | Rollback forward (ms) | Forward speedup | Native total (ms) | Rollback total (ms) | Total speedup | Max gradient error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in payload["cases"]:
        workload = case["workload"]
        native = case["engines"]["flagquantum_adjoint"]
        rollback = case["engines"]["flagquantum_adjoint_forward_cx_rollback"]
        ratio = case["comparison"]["engine_over_flagquantum_median"][
            "flagquantum_adjoint_forward_cx_rollback"
        ]
        error = case["correctness"]["engines"][
            "flagquantum_adjoint_forward_cx_rollback"
        ]["gradient_max_abs_error"]
        cx_count = case["features"]["opcode_histogram"].get("cx", 0)
        lines.append(
            f"| {_WORKLOAD_LABELS[workload['name']]} | {workload['n_wires']} | "
            f"{cx_count} | {_milliseconds(native, 'forward'):.3f} | "
            f"{_milliseconds(rollback, 'forward'):.3f} | {ratio['forward']:.2f}x | "
            f"{_milliseconds(native, 'value_and_grad'):.3f} | "
            f"{_milliseconds(rollback, 'value_and_grad'):.3f} | "
            f"{ratio['value_and_grad']:.2f}x | {error:.3e} |"
        )
    lines.extend(
        (
            "",
            "## Interpretation",
            "",
            "The hardware-efficient VQE case contains one 21-gate CX chain, so it",
            "exercises the optimized gather once per forward sweep. QAOA contains no",
            "CX gates and is a negative control: its small timing difference is ordinary",
            "run-to-run noise, not an optimization claim. This is non-release,",
            "single-device comparison evidence; it does not establish scaling claims.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "flagquantum-benchmark run differentiable_simulator_corpus \\",
            "  --workloads " + " ".join(payload["workloads"]) + " \\",
            "  --n-wires " + " ".join(map(str, payload["n_wires"])) + " \\",
            "  --engines flagquantum_adjoint flagquantum_adjoint_forward_cx_rollback \\",
            f"  --layers {payload['layers']} --threads {payload['environment']['torch_threads']} \\",
            f"  --warmup {methodology['warmup']} --iterations {methodology['iterations']} \\",
            f"  --calls-per-sample {methodology['calls_per_sample']} --skip-memory-probe \\",
            f"  --json-output {artifact_name} --markdown-output REPORT.md",
            "```",
            "",
        )
    )
    return "\n".join(lines)


def render_observable_cache_markdown(
    payload: Mapping[str, Any], *, artifact_name: str
) -> str:
    """Render steady-state observable-weight cache evidence."""

    methodology = payload["methodology"]
    rollback_name = "flagquantum_adjoint_observable_cache_rollback"
    lines = [
        "# Native CPU adjoint observable-weight cache comparison",
        "",
        f"This report is generated from [`{artifact_name}`]({artifact_name}). It measures",
        "steady-state repeated optimization steps with FlagQuantum's bounded observable",
        "diagonal cache enabled and disabled. Both paths use the same native adjoint",
        "implementation and must produce identical values and full parameter gradients.",
        "",
        "| Workload | Qubits | Observable terms | Cached forward (ms) | Rollback forward (ms) | Forward speedup | Cached total (ms) | Rollback total (ms) | Total speedup | Max gradient error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in payload["cases"]:
        workload = case["workload"]
        native = case["engines"]["flagquantum_adjoint"]
        rollback = case["engines"][rollback_name]
        ratio = case["comparison"]["engine_over_flagquantum_median"][rollback_name]
        error = case["correctness"]["engines"][rollback_name]["gradient_max_abs_error"]
        lines.append(
            f"| {_WORKLOAD_LABELS[workload['name']]} | {workload['n_wires']} | "
            f"{workload['observable_term_count']} | {_milliseconds(native, 'forward'):.3f} | "
            f"{_milliseconds(rollback, 'forward'):.3f} | {ratio['forward']:.2f}x | "
            f"{_milliseconds(native, 'value_and_grad'):.3f} | "
            f"{_milliseconds(rollback, 'value_and_grad'):.3f} | "
            f"{ratio['value_and_grad']:.2f}x | {error:.3e} |"
        )
    lines.extend(
        (
            "",
            "## Interpretation",
            "",
            "The first execution builds the real Z/ZZ observable diagonal. Later optimizer",
            "steps reuse it because the Hamiltonian structure is independent of trainable",
            "gate parameters. The cache is CPU-only, keyed by terms, width, dtype, device,",
            "and local amplitude count, and bounded to eight entries and 256 MiB. The",
            "warmups populate it before retained measurements. This is non-release local",
            "evidence and does not establish distributed or accelerator scaling claims.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "flagquantum-benchmark run differentiable_simulator_corpus \\",
            "  --workloads " + " ".join(payload["workloads"]) + " \\",
            "  --n-wires " + " ".join(map(str, payload["n_wires"])) + " \\",
            "  --engines flagquantum_adjoint " + rollback_name + " \\",
            f"  --layers {payload['layers']} --threads {payload['environment']['torch_threads']} \\",
            f"  --warmup {methodology['warmup']} --iterations {methodology['iterations']} \\",
            f"  --calls-per-sample {methodology['calls_per_sample']} --skip-memory-probe \\",
            f"  --json-output {artifact_name} --markdown-output REPORT.md",
            "```",
            "",
            "Set `FQ_STATEVECTOR_ADJOINT_OBSERVABLE_CACHE=0` for direct rollback.",
            "",
        )
    )
    return "\n".join(lines)


def render_observable_boundary_markdown(
    payload: Mapping[str, Any], *, artifact_name: str
) -> str:
    """Render native fused-observable-boundary evidence."""

    methodology = payload["methodology"]
    rollback_name = "flagquantum_adjoint_observable_boundary_rollback"
    lines = [
        "# Native CPU adjoint observable-boundary comparison",
        "",
        f"This report is generated from [`{artifact_name}`]({artifact_name}). It compares",
        "FlagQuantum's fused native observable expectation and adjoint-seed kernels",
        "against the prior eager PyTorch tensor expressions. Circuit execution, cached",
        "Z/ZZ observable weights, and the remaining adjoint sweep are identical.",
        "Ratios above 1 mean the fused path is faster.",
        "",
        "| Workload | Qubits | Observable terms | Native forward (ms) | Rollback forward (ms) | Forward speedup | Native backward (ms) | Rollback backward (ms) | Backward speedup | Native total (ms) | Rollback total (ms) | Total speedup | Max value error | Max gradient error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in payload["cases"]:
        workload = case["workload"]
        native = case["engines"]["flagquantum_adjoint"]
        rollback = case["engines"][rollback_name]
        ratio = case["comparison"]["engine_over_flagquantum_median"][rollback_name]
        correctness = case["correctness"]["engines"][rollback_name]
        lines.append(
            f"| {_WORKLOAD_LABELS[workload['name']]} | {workload['n_wires']} | "
            f"{workload['observable_term_count']} | {_milliseconds(native, 'forward'):.3f} | "
            f"{_milliseconds(rollback, 'forward'):.3f} | {ratio['forward']:.2f}x | "
            f"{_milliseconds(native, 'backward'):.3f} | "
            f"{_milliseconds(rollback, 'backward'):.3f} | {ratio['backward']:.2f}x | "
            f"{_milliseconds(native, 'value_and_grad'):.3f} | "
            f"{_milliseconds(rollback, 'value_and_grad'):.3f} | "
            f"{ratio['value_and_grad']:.2f}x | "
            f"{correctness['value_max_abs_error']:.3e} | "
            f"{correctness['gradient_max_abs_error']:.3e} |"
        )
    lines.extend(
        (
            "",
            "## What this measures",
            "",
            "Exact Z/ZZ expectations multiply every state probability by a cached real",
            "diagonal. Reverse-mode differentiation then seeds the adjoint with twice the",
            "state times that same diagonal. The native boundary performs each operation",
            "in one parallel traversal, avoiding eager full-state intermediates. It is",
            "automatic for contiguous complex64/complex128 CPU statevectors; unsupported",
            "layouts retain the existing PyTorch fallback. These are local single-device",
            "measurements, not distributed or cross-machine scalability claims.",
            "",
            "## FlagQuantum example",
            "",
            "```python",
            "import torch",
            "import flagquantum as fq",
            "from flagquantum import algorithms as fqa",
            "",
            "theta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)",
            "circuit = fq.Circuit(3, dtype=torch.complex128)",
            "for wire in range(3):",
            "    circuit.rx(wire, theta).ry(wire, theta).rz(wire, theta)",
            "circuit.cx(0, 1).cx(1, 2)",
            'hamiltonian = fqa.Hamiltonian((fqa.pauli_term(1.0, "ZZ", (0, 1)),))',
            'energy = hamiltonian.expectation(circuit, differentiation="adjoint")',
            "energy.backward()",
            "print(energy.item(), theta.grad)",
            "```",
            "",
            "## Reproduce",
            "",
            "```bash",
            "flagquantum-benchmark run differentiable_simulator_corpus \\",
            "  --workloads " + " ".join(payload["workloads"]) + " \\",
            "  --n-wires " + " ".join(map(str, payload["n_wires"])) + " \\",
            "  --engines flagquantum_adjoint " + rollback_name + " \\",
            f"  --layers {payload['layers']} --threads {payload['environment']['torch_threads']} \\",
            f"  --warmup {methodology['warmup']} --iterations {methodology['iterations']} \\",
            f"  --calls-per-sample {methodology['calls_per_sample']} --skip-memory-probe \\",
            f"  --json-output {artifact_name} --markdown-output REPORT.md",
            "```",
            "",
            "Set `FQ_NATIVE_CPU_OBSERVABLE_BOUNDARY=0` to restore the eager expectation",
            "and adjoint-seed expressions while retaining the rest of native adjoint.",
            "",
        )
    )
    return "\n".join(lines)


def render_rotation_tile_markdown(
    payload: Mapping[str, Any], *, artifact_name: str
) -> str:
    """Render the native wide/legacy adjoint rotation-tile comparison."""

    methodology = payload["methodology"]
    rollback_name = "flagquantum_adjoint_rotation_tile_rollback"
    lines = [
        "# Native CPU adjoint rotation-tile comparison",
        "",
        f"This report is generated from [`{artifact_name}`]({artifact_name}). It compares",
        "FlagQuantum's full-layer, structure-specialized native adjoint kernels with the",
        "legacy 48-gate, two-wire rotation path. The optimized path also accumulates one",
        "gradient subtotal per tile and applies adjacent fixed Hadamards to ket and",
        "adjoint together in 11-wire tiles. Both paths execute the same exact",
        "statevector-adjoint method and must return matching values and gradients.",
        "",
        "| Workload | Qubits | Rotations | Native backward (ms) | Rollback backward (ms) | Backward speedup | Native total (ms) | Rollback total (ms) | Total speedup | Max gradient error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in payload["cases"]:
        workload = case["workload"]
        native = case["engines"]["flagquantum_adjoint"]
        rollback = case["engines"][rollback_name]
        ratio = case["comparison"]["engine_over_flagquantum_median"][rollback_name]
        error = case["correctness"]["engines"][rollback_name]["gradient_max_abs_error"]
        opcodes = case["features"]["opcode_histogram"]
        rotations = sum(opcodes.get(name, 0) for name in ("rx", "ry", "rz"))
        lines.append(
            f"| {_WORKLOAD_LABELS[workload['name']]} | {workload['n_wires']} | "
            f"{rotations} | {_milliseconds(native, 'backward'):.3f} | "
            f"{_milliseconds(rollback, 'backward'):.3f} | {ratio['backward']:.2f}x | "
            f"{_milliseconds(native, 'value_and_grad'):.3f} | "
            f"{_milliseconds(rollback, 'value_and_grad'):.3f} | "
            f"{ratio['value_and_grad']:.2f}x | {error:.3e} |"
        )
    lines.extend(
        (
            "",
            "## Interpretation",
            "",
            "Hardware-efficient VQE has 66 adjacent RX/RY/RZ gates. The optimized",
            "kernel processes the complete layer in bounded statevector tiles and uses",
            "the known sparse rotation structure instead of general complex matrix",
            "multiplication. Its tile-local gradient subtotal avoids a hot memory update for",
            "every amplitude pair. QAOA has only 22 RX gates, but its 22 fixed Hadamards now",
            "use a paired ket/adjoint real-arithmetic kernel, reducing eight full-state",
            "kernel calls to two bounded tile calls. This is non-release single-device",
            "evidence and is not a scaling claim.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "flagquantum-benchmark run differentiable_simulator_corpus \\",
            "  --workloads " + " ".join(payload["workloads"]) + " \\",
            "  --n-wires " + " ".join(map(str, payload["n_wires"])) + " \\",
            "  --engines flagquantum_adjoint " + rollback_name + " \\",
            f"  --layers {payload['layers']} --threads {payload['environment']['torch_threads']} \\",
            f"  --warmup {methodology['warmup']} --iterations {methodology['iterations']} \\",
            f"  --calls-per-sample {methodology['calls_per_sample']} --skip-memory-probe \\",
            f"  --json-output {artifact_name} --markdown-output REPORT.md",
            "```",
            "",
            "Set `FQ_NATIVE_CPU_ADJOINT_WIDE_TILES=0` for direct rollback.",
            "",
        )
    )
    return "\n".join(lines)


def render_euler_triple_markdown(
    payload: Mapping[str, Any], *, artifact_name: str
) -> str:
    """Render the native Euler-triple/rollback adjoint comparison."""

    methodology = payload["methodology"]
    rollback_name = "flagquantum_adjoint_euler_triple_rollback"
    lines = [
        "# Native CPU adjoint Euler-triple comparison",
        "",
        f"This report is generated from [`{artifact_name}`]({artifact_name}). It compares",
        "FlagQuantum's native reverse sweep with and without the same-wire RZ/RY/RX",
        "Euler-triple and branchless amplitude-pair kernel. Both paths use exact",
        "statevector adjoint differentiation and must return matching values and full",
        "parameter gradients. Kernel tests also compare ket and adjoint states.",
        "Ratios above 1 mean the fused path is faster.",
        "",
        "| Workload | Qubits | Euler triples | Native backward (ms) | Rollback backward (ms) | Backward speedup | Native total (ms) | Rollback total (ms) | Total speedup | Max gradient error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in payload["cases"]:
        workload = case["workload"]
        native = case["engines"]["flagquantum_adjoint"]
        rollback = case["engines"][rollback_name]
        ratio = case["comparison"]["engine_over_flagquantum_median"][rollback_name]
        error = case["correctness"]["engines"][rollback_name]["gradient_max_abs_error"]
        opcodes = case["features"]["opcode_histogram"]
        triples = min(opcodes.get("rx", 0), opcodes.get("ry", 0), opcodes.get("rz", 0))
        lines.append(
            f"| {_WORKLOAD_LABELS[workload['name']]} | {workload['n_wires']} | "
            f"{triples} | {_milliseconds(native, 'backward'):.3f} | "
            f"{_milliseconds(rollback, 'backward'):.3f} | {ratio['backward']:.2f}x | "
            f"{_milliseconds(native, 'value_and_grad'):.3f} | "
            f"{_milliseconds(rollback, 'value_and_grad'):.3f} | "
            f"{ratio['value_and_grad']:.2f}x | {error:.3e} |"
        )
    lines.extend(
        (
            "",
            "## What this measures",
            "",
            "Hardware-efficient VQE applies RX, RY, and RZ consecutively to each",
            "qubit, which the reverse sweep visits as RZ/RY/RX. The fused kernel keeps",
            "one amplitude pair in registers while computing three gradients and undoing",
            "all three rotations, instead of loading and storing the pair once per gate.",
            "The pair loop uses an OpenMP SIMD reduction for its gradient totals. All other",
            "rotations use direct zero/one pair blocks without a per-amplitude bit test.",
            "QAOA has no Euler triples, so it isolates that branchless traversal gain.",
            "This is non-release single-device evidence, not a thread-scaling or",
            "cross-machine claim.",
            "",
            "## FlagQuantum example",
            "",
            "The optimization is automatic behind the existing adjoint API; user code",
            "does not select a kernel:",
            "",
            "```python",
            "import torch",
            "import flagquantum as fq",
            "from flagquantum import algorithms as fqa",
            "",
            "theta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)",
            "circuit = fq.Circuit(3, dtype=torch.complex128)",
            "for wire in range(3):",
            "    circuit.rx(wire, theta).ry(wire, theta).rz(wire, theta)",
            'hamiltonian = fqa.Hamiltonian((fqa.pauli_term(1.0, "ZZ", (0, 1)),))',
            'energy = hamiltonian.expectation(circuit, differentiation="adjoint")',
            "energy.backward()",
            "print(energy.item(), theta.grad)",
            "```",
            "",
            "## Reproduce",
            "",
            "```bash",
            "flagquantum-benchmark run differentiable_simulator_corpus \\",
            "  --workloads " + " ".join(payload["workloads"]) + " \\",
            "  --n-wires " + " ".join(map(str, payload["n_wires"])) + " \\",
            "  --engines flagquantum_adjoint " + rollback_name + " \\",
            f"  --layers {payload['layers']} --threads {payload['environment']['torch_threads']} \\",
            f"  --warmup {methodology['warmup']} --iterations {methodology['iterations']} \\",
            f"  --calls-per-sample {methodology['calls_per_sample']} --skip-memory-probe \\",
            f"  --json-output {artifact_name} --markdown-output REPORT.md",
            "```",
            "",
            "Set `FQ_NATIVE_CPU_ADJOINT_EULER_TRIPLES=0` to restore the prior",
            "per-gate, bit-tested pair traversal while retaining native adjoint.",
            "",
        )
    )
    return "\n".join(lines)

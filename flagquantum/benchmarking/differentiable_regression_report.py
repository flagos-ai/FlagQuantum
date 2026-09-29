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

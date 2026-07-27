"""Replicated per-rank capacity smoke for FlagQuantum MPS/TN on multi-GPU nodes.

This launcher starts one torchrun job per problem size. That is deliberate:
large MPS/TN cases can hit OOM or backend stalls, and process-level timeouts are
the safest way to keep a capacity scan moving.

Important: this is not a sharded multi-GPU scalability benchmark. Every rank
runs the same full case on its own GPU. Use this only for per-GPU capacity and
cluster health checks. Single-problem multi-GPU expansion must use a dedicated
sharded MPS/TN benchmark with ``distribution_semantics=sharded_across_ranks``.

Recommended 8xA100 commands
---------------------------
Native MPS forward replicated smoke:
  python benchmarks/capacity_sweep.py --nproc-per-node 8 --engine native --modes mps --start-wires 64 --stop-wires 512 --step-wires 32 --layers 2 --batch-size 1 --max-bond 32 --forward-only --case-timeout-seconds 600 --case-output-dir benchmarks/capacity_cases/native_mps_forward --json-output capacity_a100_8gpu_mps_forward.json

Native MPS value+grad replicated smoke:
  python benchmarks/capacity_sweep.py --nproc-per-node 8 --engine native --modes mps --start-wires 32 --stop-wires 256 --step-wires 32 --layers 2 --batch-size 1 --max-bond 32 --case-timeout-seconds 900 --case-output-dir benchmarks/capacity_cases/native_mps_grad --json-output capacity_a100_8gpu_mps_grad.json

Native tensor-network forward replicated smoke:
  python benchmarks/capacity_sweep.py --nproc-per-node 8 --engine native --modes tensor_network --start-wires 8 --stop-wires 80 --step-wires 4 --layers 2 --batch-size 1 --forward-only --case-timeout-seconds 600 --case-output-dir benchmarks/capacity_cases/native_tn_forward --json-output capacity_a100_8gpu_tn_forward.json

Native tensor-network value+grad replicated smoke:
  python benchmarks/capacity_sweep.py --nproc-per-node 8 --engine native --modes tensor_network --start-wires 8 --stop-wires 48 --step-wires 4 --layers 2 --batch-size 1 --case-timeout-seconds 900 --case-output-dir benchmarks/capacity_cases/native_tn_grad --json-output capacity_a100_8gpu_tn_grad.json

JAX MPS quantum-kernel replicated smoke:
  python benchmarks/capacity_sweep.py --nproc-per-node 8 --engine jax --modes mps --start-wires 32 --stop-wires 256 --step-wires 32 --layers 2 --batch-size 1 --max-bond 32 --case-timeout-seconds 900 --case-output-dir benchmarks/capacity_cases/jax_mps_grad --json-output capacity_a100_8gpu_jax_mps.json

JAX tensor-network quantum-kernel replicated smoke:
  python benchmarks/capacity_sweep.py --nproc-per-node 8 --engine jax --modes tensor_network --start-wires 8 --stop-wires 48 --step-wires 4 --layers 2 --batch-size 1 --case-timeout-seconds 900 --case-output-dir benchmarks/capacity_cases/jax_tn_grad --json-output capacity_a100_8gpu_jax_tn.json

Quick smoke:
  python benchmarks/capacity_sweep.py --nproc-per-node 2 --engine native --modes mps,tensor_network --start-wires 4 --stop-wires 8 --step-wires 4 --layers 1 --batch-size 1 --forward-only --case-timeout-seconds 120 --case-output-dir benchmarks/capacity_cases/native_smoke_forward

JAX quick smoke:
  python benchmarks/capacity_sweep.py --nproc-per-node 2 --engine jax --modes mps,tensor_network --start-wires 4 --stop-wires 8 --step-wires 4 --layers 1 --batch-size 1 --forward-only --case-timeout-seconds 120 --case-output-dir benchmarks/capacity_cases/jax_smoke_forward
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_SCRIPT = REPO_ROOT / "benchmarks" / "capacity_case.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flagquantum as fq  # noqa: E402


def _write_json_output(path_text: str, text: str) -> None:
    path = Path(path_text)
    if not path.is_absolute():
        path = Path("benchmarks") / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def _json_path(base_dir: Path, mode: str, n_wires: int) -> Path:
    safe_mode = "tn" if mode in {"tensor_network", "tn"} else mode
    return base_dir / f"capacity_case_{safe_mode}_{int(n_wires)}q.json"


def _load_case(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _tail(text: str, max_chars: int = 8000) -> str:
    if len(text) <= int(max_chars):
        return text
    return text[-int(max_chars) :]


def _run_case(args: argparse.Namespace, mode: str, n_wires: int, output_dir: Path) -> dict[str, Any]:
    case_json = _json_path(output_dir, mode, n_wires)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={int(args.nproc_per_node)}",
        str(CASE_SCRIPT),
        "--device",
        args.device,
        "--dist-backend",
        args.dist_backend,
        "--engine",
        args.engine,
        "--mode",
        mode,
        "--n-wires",
        str(int(n_wires)),
        "--layers",
        str(int(args.layers)),
        "--batch-size",
        str(int(args.batch_size)),
        "--observable",
        args.observable,
        "--entangler",
        args.entangler,
        "--iters",
        str(int(args.iters)),
        "--warmup",
        str(int(args.warmup)),
        "--jax-compute-dtype",
        args.jax_compute_dtype,
        "--jax-matmul-precision",
        args.jax_matmul_precision,
        "--torch-matmul-precision",
        args.torch_matmul_precision,
        "--dist-timeout-seconds",
        str(float(args.dist_timeout_seconds)),
        "--json-output",
        str(case_json),
    ]
    if mode == "mps":
        cmd.extend(["--max-bond", str(int(args.max_bond))])
    if args.forward_only:
        cmd.append("--forward-only")
    if args.no_jax_jit:
        cmd.append("--no-jax-jit")

    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    env.setdefault("USE_LIBUV", "0")
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            capture_output=True,
            timeout=float(args.case_timeout_seconds),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "mode": mode,
            "n_wires": int(n_wires),
            "seconds": time.perf_counter() - start,
            "timeout_seconds": float(args.case_timeout_seconds),
            "command": cmd,
            "stdout_tail": _tail(exc.stdout or ""),
            "stderr_tail": _tail(exc.stderr or ""),
        }

    elapsed = time.perf_counter() - start
    case_payload = _load_case(case_json)
    status = "ok" if completed.returncode == 0 and case_payload and case_payload.get("status") == "ok" else "error"
    result = {
        "status": status,
        "mode": mode,
        "n_wires": int(n_wires),
        "seconds": elapsed,
        "returncode": completed.returncode,
        "case_json": str(case_json),
        "command": cmd,
        "stdout_tail": _tail(completed.stdout or ""),
        "stderr_tail": _tail(completed.stderr or ""),
    }
    if case_payload is not None:
        result["case"] = case_payload
    return result


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode in sorted({str(item.get("mode")) for item in results}):
        mode_results = [item for item in results if item.get("mode") == mode]
        ok_results = [item for item in mode_results if item.get("status") == "ok"]
        max_ok = max([int(item["n_wires"]) for item in ok_results], default=None)
        first_failure = next((item for item in mode_results if item.get("status") != "ok"), None)
        summary[mode] = {
            "max_successful_n_wires": max_ok,
            "successful_cases": len(ok_results),
            "attempted_cases": len(mode_results),
            "first_failure": None
            if first_failure is None
            else {
                "n_wires": first_failure.get("n_wires"),
                "status": first_failure.get("status"),
                "returncode": first_failure.get("returncode"),
                "timeout_seconds": first_failure.get("timeout_seconds"),
                "case_json": first_failure.get("case_json"),
            },
        }
    return summary


def _default_case_output_dir(args: argparse.Namespace, modes: list[str]) -> Path:
    normalized_modes = ["tn" if mode == "tensor_network" else mode for mode in modes]
    mode_part = "_".join(normalized_modes) if normalized_modes else "unknown"
    pass_part = "forward" if args.forward_only else "grad"
    return REPO_ROOT / "benchmarks" / "capacity_cases" / f"{args.engine}_{mode_part}_{pass_part}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="mps,tensor_network")
    parser.add_argument("--start-wires", type=int, default=8)
    parser.add_argument("--stop-wires", type=int, default=128)
    parser.add_argument("--step-wires", type=int, default=8)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dist-backend", choices=("auto", "nccl", "gloo", "none"), default="nccl")
    parser.add_argument("--engine", choices=("native", "jax"), default="native")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-bond", type=int, default=32)
    parser.add_argument("--observable", choices=("z_sum",), default="z_sum")
    parser.add_argument("--entangler", choices=("chain", "brickwork", "ring", "none"), default="chain")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--no-jax-jit", action="store_true")
    parser.add_argument("--jax-compute-dtype", choices=("complex64", "complex128"), default="complex64")
    parser.add_argument("--jax-matmul-precision", default="highest")
    parser.add_argument("--torch-matmul-precision", default="highest")
    parser.add_argument("--dist-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--case-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--continue-after-failure", action="store_true")
    parser.add_argument(
        "--case-output-dir",
        default="",
        help="Directory for per-size case JSON files. Defaults to benchmarks/capacity_cases/<engine>_<modes>_<forward|grad>.",
    )
    parser.add_argument("--json-output", default="capacity_sweep.json")
    args = parser.parse_args()

    modes = [item.strip() for item in str(args.modes).split(",") if item.strip()]
    modes = ["tensor_network" if item == "tn" else item for item in modes]
    output_dir = _default_case_output_dir(args, modes) if not args.case_output_dir else Path(args.case_output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for mode in modes:
        for n_wires in range(int(args.start_wires), int(args.stop_wires) + 1, int(args.step_wires)):
            print(f"[capacity] mode={mode} n_wires={n_wires}", flush=True)
            result = _run_case(args, mode, n_wires, output_dir)
            results.append(result)
            print(
                f"[capacity] done mode={mode} n_wires={n_wires} status={result.get('status')} "
                f"seconds={float(result.get('seconds', 0.0)):.2f}",
                flush=True,
            )
            if result.get("status") != "ok" and not args.continue_after_failure:
                break

    payload = {
        "benchmark": "capacity_sweep",
        "distribution_semantics": "replicated_per_rank",
        "scalability_claim_allowed": False,
        "scalability_note": (
            "This sweep launches replicated capacity cases. It summarizes per-rank capacity/health "
            "and must not be used as evidence that one large MPS/TN workload is sharded across ranks."
        ),
        "status": "ok" if all(item.get("status") == "ok" for item in results) else "partial",
        "configuration": {
            "modes": modes,
            "start_wires": int(args.start_wires),
            "stop_wires": int(args.stop_wires),
            "step_wires": int(args.step_wires),
            "nproc_per_node": int(args.nproc_per_node),
            "engine": args.engine,
            "layers": int(args.layers),
            "batch_size": int(args.batch_size),
            "max_bond": int(args.max_bond),
            "forward_only": bool(args.forward_only),
            "iters": int(args.iters),
            "warmup": int(args.warmup),
            "case_timeout_seconds": float(args.case_timeout_seconds),
            "case_output_dir": str(output_dir),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "summary": _summarize(results),
        "results": results,
    }
    payload = fq.attach_distributed_scalability_audit(payload)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_output:
        _write_json_output(args.json_output, text)


if __name__ == "__main__":
    main()

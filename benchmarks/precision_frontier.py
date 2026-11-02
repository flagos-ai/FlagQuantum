"""Bounded complete-circuit precision/accuracy/time/memory comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmarks.precision_workloads import (
    Case,
    output_hash,
    reference,
    state_quality,
    synthesis_error,
)
from benchmarks.qng_hva_ablation import run_worker
from flagquantum import algorithms as fqa
from flagquantum.benchmarking.contract import runtime_metadata, write_json_atomic
from flagquantum.compute import resolve_platform_device
from flagquantum.runtime.executors.statevector.split_real_imag import (
    parameter_shift_split_real_imag_gradient,
)
from flagquantum.runtime.executors.statevector.split_real_imag_device_double_single import (
    parameter_shift_split_real_imag_device_double_single_gradient,
)
from flagquantum.runtime.executors.statevector.split_real_imag_double_single import (
    parameter_shift_split_real_imag_double_single_gradient,
)
from flagquantum.runtime.executors.statevector.split_real_imag_precision import (
    parameter_shift_split_real_imag_precision_gradient,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks/manifests/precision_frontier_v1.json"
SCHEMA = "flagquantum.precision_frontier.v1"
MODES = ("fp32", "double_single_p4", "fp64")
CONTROLS = ("selective_p2", "double_single_p3")


class UnsupportedRouteError(RuntimeError):
    """A requested device/stage is unavailable; substitution is prohibited."""


def source_identity() -> dict[str, Any]:
    """Reuse the established benchmark source contract and bind this producer."""
    from benchmarks.training_comparison import source_identity as training_source

    source = training_source(MANIFEST)
    for name in ("precision_frontier.py", "precision_workloads.py"):
        source["producer_sha256"][f"benchmarks/{name}"] = hashlib.sha256(
            (ROOT / "benchmarks" / name).read_bytes()
        ).hexdigest()
    return source


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
    elif device.startswith("mps"):
        torch.mps.synchronize()
    elif not device.startswith("cpu"):
        resolve_platform_device(device)
        backend = getattr(torch, torch.device(device).type, None)
        if backend is None or not hasattr(backend, "synchronize"):
            raise UnsupportedRouteError(
                f"no observable synchronization API for {device}"
            )
        backend.synchronize()


def require_device(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise UnsupportedRouteError("CUDA unavailable; no CPU fallback")
    if device.startswith("mps") and not torch.backends.mps.is_available():
        raise UnsupportedRouteError("MPS unavailable; no CPU fallback")
    try:
        physical = (
            torch.device(device)
            if device.startswith(("cpu", "cuda", "mps"))
            else resolve_platform_device(device)
        )
        torch.empty(1, dtype=torch.float32, device=physical)
    except (RuntimeError, ValueError, NotImplementedError) as exc:
        raise UnsupportedRouteError(str(exc)) from exc


def environment(device: str) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package_name in ("torch", "numpy", "triton"):
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = None
    name: str
    uuid: str | None
    reason: str | None
    name, uuid, reason = (
        platform.processor() or platform.machine(),
        None,
        "CPU has no accelerator UUID",
    )
    if device.startswith("cuda") and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.device(device))
        name = props.name
        uuid = str(getattr(props, "uuid", "")) or None
        reason = None if uuid else "UUID not exposed by this Torch build"
    elif device.startswith("mps"):
        try:
            name = (
                subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=5
                ).strip()
                + " / Apple Metal MPS"
            )
        except (OSError, subprocess.SubprocessError):
            name = "Apple Metal MPS"
        reason = "Torch MPS does not expose a physical accelerator UUID"
    driver = None
    if device.startswith("cuda") and torch.cuda.is_available():
        logical = torch.device(device).index or 0
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        selector = uuid or (
            visible[logical]
            if logical < len(visible) and visible[logical]
            else str(logical)
        )
        if uuid and not selector.startswith("GPU-"):
            selector = "GPU-" + selector
        try:
            record = (
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=uuid,driver_version",
                        "--format=csv,noheader",
                        "--id=" + selector,
                    ],
                    text=True,
                    timeout=10,
                )
                .strip()
                .split(",")
            )
            uuid, driver = record[0].strip(), record[1].strip()
            reason = None
        except (OSError, subprocess.SubprocessError, IndexError):
            pass
    return {
        "device": device,
        "device_name": name,
        "device_uuid": uuid,
        "device_uuid_unavailable_reason": reason,
        "versions": versions,
        "cuda_runtime": torch.version.cuda,
        "driver_version": driver,
        "driver_version_unavailable_reason": (
            None
            if driver is not None or not device.startswith("cuda")
            else "nvidia-smi query unavailable for selected physical GPU"
        ),
        "torch_build": torch.__config__.show(),
        "tf32_environment": {
            name: os.environ.get(name)
            for name in (
                "NVIDIA_TF32_OVERRIDE",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                "CUBLAS_WORKSPACE_CONFIG",
            )
        },
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "fma_policy": "eager_PyTorch_build_and_device_intrinsics_not_independently_overridden",
        "torch_compile": False,
        "torch_jit": False,
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "synchronization": (
            "synchronous_CPU"
            if device.startswith("cpu")
            else "backend_synchronize_at_every_sample_boundary"
        ),
    }


def probe_stages(device: str) -> list[dict[str, Any]]:
    """Exercise every existing stage before planning complete-circuit cases."""
    from flagquantum.runtime.executors.statevector.split_real_imag import (
        run_split_real_imag_conformance,
        run_split_real_imag_training_conformance,
    )
    from flagquantum.runtime.executors.statevector.split_real_imag_autograd_conformance import (
        run_split_real_imag_autograd_conformance,
    )
    from flagquantum.runtime.executors.statevector.split_real_imag_device_double_single_conformance import (
        run_split_real_imag_device_double_single_conformance,
    )
    from flagquantum.runtime.executors.statevector.split_real_imag_double_single_conformance import (
        run_split_real_imag_double_single_conformance,
    )
    from flagquantum.runtime.executors.statevector.split_real_imag_optimizer_conformance import (
        run_split_real_imag_optimizer_conformance,
    )
    from flagquantum.runtime.executors.statevector.split_real_imag_precision import (
        run_split_real_imag_precision_conformance,
    )
    from flagquantum.simulation.numerics.conformance import (
        run_double_single_conformance,
    )

    probes: list[tuple[str, Callable[[], Any]]] = [
        ("arithmetic", lambda: run_double_single_conformance(device)),
        ("P0", lambda: run_split_real_imag_conformance(device, depths=(2,))),
        (
            "P1",
            lambda: run_split_real_imag_training_conformance(
                device, depths=(2,), seeds=(0,)
            ),
        ),
        (
            "P2",
            lambda: run_split_real_imag_precision_conformance(
                device, depths=(2,), seeds=(0,)
            ),
        ),
        (
            "P3",
            lambda: run_split_real_imag_double_single_conformance(
                device, depths=(2,), seeds=(0,)
            ),
        ),
        (
            "P4",
            lambda: run_split_real_imag_device_double_single_conformance(
                device,
                depths=(2,),
                seeds=(0,),
                stability_depths=(4,),
                stability_seeds=(0,),
            ),
        ),
        (
            "P5_autograd",
            lambda: run_split_real_imag_autograd_conformance(
                device, depths=(2,), seeds=(0,)
            ),
        ),
        (
            "P5_optimizer",
            lambda: run_split_real_imag_optimizer_conformance(device, checkpoints=(1,)),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, function in probes:
        started = time.perf_counter()
        try:
            require_device(device)
            result = function().to_dict()
            sync(device)
            row = {
                "stage": name,
                "status": "completed",
                "conformance_passed": result.get("passed"),
                "report": result,
            }
        except (UnsupportedRouteError, NotImplementedError) as exc:
            row = {"stage": name, "status": "unsupported", "error": str(exc)}
        except KeyError as exc:
            row = {
                "stage": name,
                "status": (
                    "unsupported"
                    if "Unknown FlagQuantum platform" in str(exc)
                    else "failed"
                ),
                "error": f"{type(exc).__name__}: {exc}",
            }
        except Exception as exc:  # Probe boundary retains stage failures independently.
            row = {
                "stage": name,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        row["probe_seconds"] = time.perf_counter() - started
        rows.append(row)
        print(
            json.dumps({key: value for key, value in row.items() if key != "report"}),
            flush=True,
        )
    started = time.perf_counter()
    try:
        require_device(device)
        case = Case(2, 2, "noncommuting", 11)
        candidate, value, gradient = cpu_output(
            evaluate(case, "fp64", device, preflight=True)
        )
        expected, _, observables, derivative = reference(case)
        error = max(
            float(np.max(np.abs(candidate - expected))),
            abs(value - observables["objective"]),
            float(np.max(np.abs(gradient - derivative))),
        )
        native: dict[str, Any] = {
            "stage": "native_fp64",
            "status": "completed",
            "conformance_passed": error <= 1e-11,
            "independent_max_error": error,
        }
    except (UnsupportedRouteError, NotImplementedError) as exc:
        native = {"stage": "native_fp64", "status": "unsupported", "error": str(exc)}
    except KeyError as exc:
        native = {
            "stage": "native_fp64",
            "status": (
                "unsupported"
                if "Unknown FlagQuantum platform" in str(exc)
                else "failed"
            ),
            "error": f"{type(exc).__name__}: {exc}",
        }
    except Exception as exc:
        native = {
            "stage": "native_fp64",
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    native["probe_seconds"] = time.perf_counter() - started
    rows.append(native)
    print(json.dumps(native), flush=True)
    return rows


def verify_residency(result: dict[str, Any], mode: str, device: str) -> dict[str, Any]:
    """Observe actual output words and refuse host fallback or dtype demotion."""
    requested = torch.device(device)
    words = result["state_words"] + result["gradient_words"] + result["value_words"]
    expected_dtype = torch.float64 if mode == "fp64" else torch.float32
    for word in words:
        if word.device.type != requested.type or (
            requested.type != "cpu"
            and requested.index is not None
            and word.device.index != requested.index
        ):
            raise RuntimeError(f"candidate output escaped {device}: {word.device}")
        if word.dtype != expected_dtype:
            raise RuntimeError(
                f"candidate silently changed word dtype: {word.dtype}, expected {expected_dtype}"
            )
    result["summary"] = {
        **result["summary"],
        "observed_tensor_devices": sorted({str(word.device) for word in words}),
        "observed_word_dtypes": sorted({str(word.dtype) for word in words}),
        "output_device_and_dtype_verified": True,
    }
    return result


def evaluate(case: Case, mode: str, device: str, *, preflight: bool) -> dict[str, Any]:
    """Call existing stage executors; no new precision arithmetic or fallback."""
    if mode == "fp64":
        if device.startswith("mps"):
            raise UnsupportedRouteError(
                "Apple MPS has no FP64/complex128 kernels; CPU substitution is prohibited"
            )
        circuit = case.circuit(device=device, fp64=True)
        state = circuit.state().reshape(-1)
        observable = case.observable()
        term_values = torch.stack(
            [
                fqa.Hamiltonian([fqa.pauli_term(c, word, wires)])
                .expectation(state)
                .sum()
                .real
                for c, word, wires in case.terms()
            ]
        )
        value = term_values.sum()
        shifted = []
        for direction in (1, -1):
            shifted_state = (
                case.circuit(
                    device=device, fp64=True, theta=case.theta + direction * np.pi / 2
                )
                .state()
                .reshape(-1)
            )
            shifted.append(observable.expectation(shifted_state).sum().real)
        gradient = ((shifted[0] - shifted[1]) / 2).reshape(1)
        return verify_residency(
            {
                "state": state,
                "value": value,
                "gradient": gradient,
                "terms": term_values,
                "value_words": [value, term_values],
                "state_words": [state.real, state.imag],
                "gradient_words": [gradient],
                "summary": {
                    "backend": "FlagQuantum_Circuit_complex128",
                    "gradient_method": "parameter_shift",
                    "host_execution_fallback": False,
                    "parameter_host_fallback": None,
                    "host_gate_encoding": None,
                    "host_boundary_unavailable_reason": "native Circuit tensor output does not expose gate/parameter host-route tracing",
                    "provider_internal_route_audited": False,
                    "fallback_observation_scope": "returned state, observable and gradient words verified on requested device",
                    "automatic_precision_selection": False,
                },
            },
            mode,
            device,
        )
    functions: dict[str, Callable[..., Any]] = {
        "fp32": parameter_shift_split_real_imag_gradient,
        "selective_p2": parameter_shift_split_real_imag_precision_gradient,
        "double_single_p3": parameter_shift_split_real_imag_double_single_gradient,
        "double_single_p4": parameter_shift_split_real_imag_device_double_single_gradient,
    }
    options: dict[str, Any] = {
        "device": device,
        "preflight": preflight,
        "parameter_bindings": {
            "theta": torch.tensor(case.theta, dtype=torch.float32, device=device)
        },
    }
    if mode in ("double_single_p3", "double_single_p4"):
        options["renormalize_every"] = 0
    result = functions[mode](
        case.circuit(device=device, symbolic=True), case.observable(), **options
    )
    expectation = result.expectation
    state = expectation.state
    if mode in ("fp32", "selective_p2"):
        words = [state.real, state.imag]
    else:
        words = [
            state.state.real.high,
            state.state.real.low,
            state.state.imag.high,
            state.state.imag.low,
        ]
    gradient_words = (
        [result.gradient]
        if mode == "fp32"
        else [result.gradient.high, result.gradient.low]
    )
    return verify_residency(
        {
            "state": state,
            "value": expectation.value,
            "terms": expectation.term_expectations,
            "value_words": (
                [expectation.value, expectation.term_expectations]
                if mode == "fp32"
                else [
                    expectation.value.high,
                    expectation.value.low,
                    expectation.term_expectations.high,
                    expectation.term_expectations.low,
                ]
            ),
            "gradient": result.gradient,
            "state_words": words,
            "gradient_words": gradient_words,
            "summary": result.summary(),
        },
        mode,
        device,
    )


def cpu_output(result: dict[str, Any]) -> tuple[np.ndarray, float, np.ndarray]:
    state = result["state"]
    observed = (
        state.detach().cpu().to(torch.complex128)
        if isinstance(state, torch.Tensor)
        else state.cpu_complex128()
    )

    def scalar(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().double().numpy()
        return value.to("cpu").to_float64().detach().numpy()

    return (
        observed.numpy().reshape(-1),
        float(scalar(result["value"])),
        scalar(result["gradient"]).reshape(-1),
    )


def run_case(
    case: Case, mode: str, device: str, *, repeats: int, warmup: int
) -> dict[str, Any]:
    require_device(device)
    started = time.perf_counter()
    first = evaluate(case, mode, device, preflight=True)
    sync(device)
    first_seconds = time.perf_counter() - started
    warmup_started = time.perf_counter()
    for _ in range(warmup):
        evaluate(case, mode, device, preflight=False)
    sync(device)
    warmup_seconds = time.perf_counter() - warmup_started
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    samples, hashes = [], []
    # Diagnostics/host reconstruction occur after each synchronized timer stops.
    # Hashing is outside the timer and cannot be inferred from scalar errors.
    result = first
    for index in range(repeats):
        sync(device)
        sample_started = time.perf_counter()
        result = evaluate(case, mode, device, preflight=False)
        sync(device)
        samples.append(time.perf_counter() - sample_started)
        hashes.append(
            output_hash(
                result["state_words"] + result["gradient_words"] + result["value_words"]
            )
        )
        print(
            json.dumps(
                {
                    "case": case.name,
                    "mode": mode,
                    "sample": index + 1,
                    "seconds": samples[-1],
                }
            ),
            flush=True,
        )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    cuda_peak = (
        torch.cuda.max_memory_allocated(torch.device(device))
        if device.startswith("cuda")
        else None
    )
    observed, value, gradient = cpu_output(result)
    if (
        not np.isfinite(observed).all()
        or not np.isfinite(value)
        or not np.isfinite(gradient).all()
    ):
        raise ValueError(
            f"non-finite simulation output; retained output hashes: {hashes}"
        )
    expected, _, expected_observables, expected_gradient = reference(case)
    quality = state_quality(observed, expected)
    term_tensor = result["terms"]
    term_values = (
        term_tensor.detach().cpu().double().numpy()
        if isinstance(term_tensor, torch.Tensor)
        else term_tensor.to("cpu").to_float64().detach().numpy()
    )
    z_index = 1 if case.family == "cancellation_echo" else 0
    observable_values = {
        "Z0": float(term_values[z_index]),
        "X1": float(term_values[-2] / 0.5),
        "Z0Z1": float(term_values[-1] / 0.25),
        "objective": value,
    }
    errors = {
        name: abs(v - expected_observables[name])
        for name, v in observable_values.items()
    }
    quality.update(
        observable_max_absolute_error=max(errors.values()),
        gradient_max_absolute_error=float(np.max(np.abs(gradient - expected_gradient))),
    )
    quality["accuracy_error"] = max(
        quality["state_max_absolute_error"],
        quality["norm_drift"],
        quality["observable_max_absolute_error"],
        quality["gradient_max_absolute_error"],
    )
    intended_state, _, intended_observables, intended_gradient = reference(
        case, case.intended_theta
    )
    intended_quality = state_quality(observed, intended_state)
    intended_quality["observable_max_absolute_error"] = max(
        abs(value - intended_observables[name])
        for name, value in observable_values.items()
    )
    intended_quality["gradient_max_absolute_error"] = float(
        np.max(np.abs(gradient - intended_gradient))
    )
    intended_quality["accuracy_error"] = max(
        intended_quality["state_max_absolute_error"],
        intended_quality["norm_drift"],
        intended_quality["observable_max_absolute_error"],
        intended_quality["gradient_max_absolute_error"],
    )
    return {
        "status": "completed",
        "case": asdict(case),
        "mode": mode,
        "device": device,
        "execution": result["summary"],
        "initial_theta_intended_float64": case.intended_theta,
        "initial_theta_matched_float32": case.theta,
        "gate_count": len(case.recipe()),
        "quality": quality,
        "quality_vs_intended_inputs": intended_quality,
        "observables": observable_values,
        "reference_observables": expected_observables,
        "observable_errors": errors,
        "gradient": gradient.tolist(),
        "reference_gradient": expected_gradient,
        "output_state_real": observed.real.tolist(),
        "output_state_imag": observed.imag.tolist(),
        "output_word_hashes_sha256": hashes,
        "repeated_output_bit_identity": len(set(hashes)) == 1,
        "bit_identity_scope": "same_route_state_observables_gradient_words_dtype_shape_and_bytes_all_retained_samples",
        "synthesis": synthesis_error(case, mode, device),
        "timing": {
            "scope": "complete_circuit_build_gate_generation_state_expectation_parameter_shift_no_diagnostic_transfer",
            "first_call_including_preflight_seconds": first_seconds,
            "warmup_calls": warmup,
            "warmup_seconds": warmup_seconds,
            "samples_seconds": samples,
            "statistic": "median",
            "median_seconds": statistics.median(samples),
            "total_timed_seconds_including_first_call": first_seconds + sum(samples),
            "total_measured_seconds_including_warmup": first_seconds
            + warmup_seconds
            + sum(samples),
        },
        "memory": {
            "process_peak_rss_bytes_at_timing_end": int(
                rss if sys.platform == "darwin" else rss * 1024
            ),
            "cuda_peak_allocated_bytes": cuda_peak,
            "frontier_peak_memory_bytes": (
                cuda_peak
                if cuda_peak is not None
                else (
                    int(rss if sys.platform == "darwin" else rss * 1024)
                    if device.startswith("cpu")
                    else None
                )
            ),
            "accelerator_peak_unavailable_reason": (
                None if cuda_peak is not None else "CPU_or_MPS_no_allocator_peak_API"
            ),
            "state_payload_bytes": sum(
                word.numel() * word.element_size() for word in result["state_words"]
            ),
            "scope": "RSS_includes_imports_preflight_warmup_and_hash_transfers_before_oracle_diagnostics_CUDA_peak_covers_timed_samples",
        },
    }


def frontier(
    results: list[dict[str, Any]], targets: list[float]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if len({r["device"] for r in results if r["status"] == "completed"}) > 1:
        raise ValueError("frontiers cannot mix device campaigns")
    groups = {
        json.dumps(result["case"], sort_keys=True)
        for result in results
        if result["status"] == "completed"
    }
    for key in sorted(groups):
        group = [
            result
            for result in results
            if result["status"] == "completed"
            and json.dumps(result["case"], sort_keys=True) == key
        ]
        for target in targets:
            feasible = [r for r in group if r["quality"]["accuracy_error"] <= target]
            measured = [
                r
                for r in feasible
                if r["memory"]["frontier_peak_memory_bytes"] is not None
            ]
            selected = []
            for candidate in measured:
                cost = (
                    candidate["timing"]["median_seconds"],
                    candidate["memory"]["frontier_peak_memory_bytes"],
                )
                if not any(
                    all(
                        a <= b
                        for a, b in zip(
                            (
                                other["timing"]["median_seconds"],
                                other["memory"]["frontier_peak_memory_bytes"],
                            ),
                            cost,
                            strict=True,
                        )
                    )
                    and (
                        other["timing"]["median_seconds"],
                        other["memory"]["frontier_peak_memory_bytes"],
                    )
                    != cost
                    for other in measured
                ):
                    selected.append(candidate["mode"])
            rows.append(
                {
                    "case": json.loads(key),
                    "accuracy_scope": "matched_FP32_representable_inputs",
                    "tolerance": target,
                    "feasible_modes": [r["mode"] for r in feasible],
                    "pareto_modes": selected,
                    "cost_axes": [
                        "median_complete_gradient_seconds",
                        "observed_peak_memory_bytes",
                    ],
                    "peak_memory_scope": "CUDA_allocator_peak_or_CPU_worker_RSS_never_storage_substitution",
                }
            )
    return rows


def render_report(output: Path) -> str:
    freeze = json.loads((output / "freeze.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    expected = freeze["case_names"]
    cases = summary["cases"]
    if [row["case"] for row in cases] != expected or summary["source"] != freeze[
        "source"
    ]:
        raise ValueError("report inventory or frozen producer changed")
    results = []
    lines = [
        "# Complete-circuit precision frontier",
        "",
        f"Source: `{freeze['source']['code_version']}`. Device: `{freeze['device']}`. Smoke: `{freeze['smoke']}`.",
        "",
        "Development comparison only. No precision provider, capability maturity, vendor certification, simulation/GEMM speedup or general FP64 equivalence is established.",
        "",
        "All routes share the same initialized FP32-representable angles and exactly represented observable coefficients. Reference: independent CPU NumPy complex128 dense matrices and analytic initial-RY tangent. Repeated Z coefficients are coalesced before reference evaluation. CPU reference and diagnostic conversion are outside candidate timing, not a fallback execution path.",
        "",
        "Timing includes circuit construction, gate generation, state evolution, expectation and supported parameter-shift gradient. Samples are synchronized absolute seconds; the statistic is the median. First call includes preflight and is separate. Renormalization is disabled for every route. Payload memory is exact storage, not peak memory; worker RSS and CUDA allocator peak have separately stated scopes.",
        "",
        "| Case | Mode | Status | State max error | Norm drift | Observable max error | Gradient error | Median s | First s | RSS MiB | State bytes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in cases:
        path = output / row["result"]
        result = (
            json.loads(path.read_text())
            if path.exists()
            else {"status": row["status"], "error": "see retained worker log"}
        )
        if path.exists() and (
            result["source"] != freeze["source"]
            or result["protocol"] != freeze["protocol"]
        ):
            raise ValueError(f"producer/protocol changed: {row['case']}")
        if result["status"] != row["status"]:
            raise ValueError("worker status changed")
        results.append(result)
        if row["status"] != "completed":
            lines.append(
                f"| {row['case']} | {row['mode']} | {row['status']}: {result.get('error','')} | — | — | — | — | — | — | — | — |"
            )
            continue
        if (
            len(result["timing"]["samples_seconds"]) != freeze["repeats"]
            or len(result["output_word_hashes_sha256"]) != freeze["repeats"]
        ):
            raise ValueError("missing repeated timing samples or output hashes")
        samples = result["timing"]["samples_seconds"]
        if (
            result["timing"]["statistic"] != "median"
            or result["timing"]["median_seconds"] != statistics.median(samples)
            or not all(np.isfinite(value) and value > 0 for value in samples)
        ):
            raise ValueError("invalid timing samples or incorrectly named statistic")
        if result["repeated_output_bit_identity"] != (
            len(set(result["output_word_hashes_sha256"])) == 1
        ):
            raise ValueError("output bit identity does not match actual word hashes")
        q, timing, memory = result["quality"], result["timing"], result["memory"]
        lines.append(
            f"| [{row['case']}]({row['result']}) | {row['mode']} | completed | {q['state_max_absolute_error']:.5g} | {q['norm_drift']:.5g} | {q['observable_max_absolute_error']:.5g} | {q['gradient_max_absolute_error']:.5g} | {timing['median_seconds']:.6f} | {timing['first_call_including_preflight_seconds']:.6f} | {memory['process_peak_rss_bytes_at_timing_end']/2**20:.1f} | {memory['state_payload_bytes']} |"
        )
    points = frontier(results, freeze["protocol"]["accuracy_targets"])
    lines += [
        "",
        "## Accuracy/time/peak-memory frontier",
        "",
        "The frontier applies to the frozen matched FP32-representable inputs, not the counterfactual intended FP64 angle. Error against that intended angle is separately retained in `quality_vs_intended_inputs`; it can exceed a chosen tolerance even when the matched-input frontier is feasible. A route is feasible only when amplitude error, norm drift, every observable error and gradient error are all within the declared dimensionless tolerance. Among feasible routes, Pareto dominance uses median complete-circuit gradient time and observed peak memory (CUDA allocator peak or CPU worker RSS). Unobservable accelerator peaks are excluded from Pareto scoring. State storage is reported separately and never substituted for a peak. These are not exclusive-system-memory claims.",
        "",
        "| Workload | Tolerance | Feasible routes | Pareto routes |",
        "| --- | ---: | --- | --- |",
    ]
    for point in points:
        lines.append(
            f"| {Case(**point['case']).name} | {point['tolerance']:g} | {', '.join(point['feasible_modes']) or 'none'} | {', '.join(point['pareto_modes']) or 'none'} |"
        )
    lines += [
        "",
        "Slower and inaccurate completed cases remain in the table. Unsupported and failed routes receive no speedup score. Repeated bit identity is based on actual output dtype/shape/byte hashes; equal scalar errors alone never imply identical outputs.",
        "",
        "Double-Single retains binary32 exponent range, is not IEEE binary64, and Dekker splitting requires finite operands small enough that multiplication by 4097 does not overflow. P4 direct angles are bounded by |theta| <= 1024; only the existing fixed and RX/RY/RZ/RXX/RYY/RZZ gates are supported. P3 uses explicit CPU complex128 gate encoding. P5 autograd/optimizer boundaries are separately probed and are not promoted by this parameter-shift benchmark.",
        "",
    ]
    return "\n".join(lines)


def campaign(output: Path, device: str, *, smoke: bool, controls: bool) -> int:
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(MANIFEST.read_text())
    source = source_identity()
    modes = list(MODES) + (list(CONTROLS) if controls else [])
    shapes = protocol["shapes"][:1] if smoke else protocol["shapes"]
    families = protocol["families"][:1] if smoke else protocol["families"]
    seeds = protocol["seeds"][:1] if smoke else protocol["seeds"]
    descriptors = [
        (Case(**shape, family=family, seed=seed), mode)
        for shape in shapes
        for family in families
        for seed in seeds
        for mode in modes
    ]
    repeats = 2 if smoke else protocol["repeats"]
    metadata_record = runtime_metadata(
        runner="precision_frontier",
        schema=SCHEMA,
        source=source,
        protocol=protocol,
        device=device,
        smoke=smoke,
        scalability_claim_allowed=False,
        execution_semantics="single_device_fast_path",
        benchmark_evidence_class="comparison_non_release",
    )
    write_json_atomic(
        output / "freeze.json",
        {
            **metadata_record,
            "case_names": [f"{case.name}-{mode}" for case, mode in descriptors],
            "repeats": repeats,
            "controls": controls,
        },
    )
    stages = probe_stages(device)
    write_json_atomic(
        output / "stages.json",
        {**metadata_record, "environment": environment(device), "stages": stages},
    )
    rows: list[dict[str, Any]] = []
    for case, mode in descriptors:
        name = f"{case.name}-{mode}"
        result_path = output / f"{name}.json"
        command = [
            sys.executable,
            "-m",
            "benchmarks.precision_frontier",
            "--worker",
            "--device",
            device,
            "--qubits",
            str(case.qubits),
            "--depth",
            str(case.depth),
            "--family",
            case.family,
            "--seed",
            str(case.seed),
            "--mode",
            mode,
            "--repeats",
            str(repeats),
            "--output",
            str(result_path),
        ]
        env = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            PYTORCH_ENABLE_MPS_FALLBACK="0",
            NVIDIA_TF32_OVERRIDE="0",
            TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="0",
        )
        print(f"Starting {name}", flush=True)
        started = time.perf_counter()
        stage_name = {
            "fp32": "P1",
            "double_single_p4": "P4",
            "fp64": "native_fp64",
            "selective_p2": "P2",
            "double_single_p3": "P3",
        }[mode]
        probe = next(row for row in stages if row["stage"] == stage_name)
        worker_started = probe["status"] == "completed"
        with (output / f"{name}.log").open("w") as log:
            if not worker_started:
                status = probe["status"]
                result = {
                    **metadata_record,
                    "status": status,
                    "case": asdict(case),
                    "mode": mode,
                    "error": f"{stage_name} preflight: {probe.get('error', 'stage unavailable')}",
                    "worker_started": False,
                }
                write_json_atomic(result_path, result)
                log.write(result["error"] + "\n")
            else:
                try:
                    completed = run_worker(
                        command,
                        env=env,
                        log=log,
                        timeout=protocol["case_timeout_seconds"],
                    )
                    result = (
                        json.loads(result_path.read_text())
                        if result_path.exists()
                        else {"status": "failed"}
                    )
                    status = result["status"]
                    if completed.returncode and status == "completed":
                        status = "failed"
                except subprocess.TimeoutExpired:
                    status = "timeout"
        rows.append(
            {
                "case": name,
                "mode": mode,
                "status": status,
                "result": result_path.name,
                "process_wall_seconds": (
                    time.perf_counter() - started if worker_started else None
                ),
                "worker_started": worker_started,
            }
        )
        write_json_atomic(output / "summary.json", {**metadata_record, "cases": rows})
    (output / "REPORT.md").write_text(render_report(output))
    results = [
        json.loads((output / row["result"]).read_text())
        for row in rows
        if (output / row["result"]).exists()
    ]
    write_json_atomic(
        output / "frontier.json",
        {
            **metadata_record,
            "frontier": frontier(results, protocol["accuracy_targets"]),
        },
    )
    return int(any(row["status"] in ("failed", "timeout") for row in rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--staged-controls", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--qubits", type=int, default=2)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument(
        "--family",
        choices=("noncommuting", "cancellation_echo"),
        default="noncommuting",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--mode", choices=MODES + CONTROLS, default="fp32")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.report:
        (args.output / "REPORT.md").write_text(render_report(args.output))
        return 0
    if not args.worker:
        return campaign(
            args.output, args.device, smoke=args.smoke, controls=args.staged_controls
        )
    protocol = json.loads(MANIFEST.read_text())
    case = Case(args.qubits, args.depth, args.family, args.seed)
    if (
        {"qubits": args.qubits, "depth": args.depth} not in protocol["shapes"]
        or args.seed not in protocol["seeds"]
        or args.repeats not in (2, protocol["repeats"])
    ):
        parser.error("worker is outside frozen protocol")
    record = runtime_metadata(
        runner="precision_frontier",
        schema=SCHEMA,
        source=source_identity(),
        protocol=protocol,
        device=args.device,
        scalability_claim_allowed=False,
        benchmark_evidence_class="comparison_non_release",
        environment=environment(args.device),
    )
    try:
        result = run_case(
            case,
            args.mode,
            args.device,
            repeats=args.repeats,
            warmup=protocol["warmup_calls"],
        )
    except (UnsupportedRouteError, NotImplementedError) as exc:
        result = {"status": "unsupported", "error": str(exc)}
    except KeyError as exc:
        result = {
            "status": (
                "unsupported"
                if "Unknown FlagQuantum platform" in str(exc)
                else "failed"
            ),
            "error": f"{type(exc).__name__}: {exc}",
        }
    except Exception as exc:  # Isolated worker boundary retains real errors.
        traceback.print_exc()
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    write_json_atomic(args.output, {**record, **result})
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())

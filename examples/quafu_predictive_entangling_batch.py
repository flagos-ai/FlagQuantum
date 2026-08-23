"""Run a frozen-model entangling validation batch on Quafu Baihua.

The protocol is deliberately two stage: assignment calibration freezes the
NoiseModel before any held-out workload is submitted.  Every task is persisted
immediately and ``--resume`` never silently replaces a failed task.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from contextlib import redirect_stdout
from itertools import product
from pathlib import Path
from typing import Any

import torch

import flagquantum as fq

DEFAULT_MAPPINGS = {2: (36, 49), 3: (106, 119, 120)}
DEFAULT_DEPTHS = {2: (0, 1, 2, 4, 8), 3: (1, 2, 4)}
SCALE_MAPPINGS = {
    4: (123, 124, 125, 126),
    5: (123, 124, 125, 126, 127),
    6: (123, 124, 125, 126, 127, 128),
}
SCALE_DEPTHS = {n: (0, 1, 2, 4, 8, 12) for n in SCALE_MAPPINGS}
BASES = ("z", "x", "y")


def _dump(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def assignment_circuit(bits: str) -> fq.Circuit:
    circuit = fq.Circuit(len(bits))
    for wire, bit in enumerate(bits):
        if bit == "1":
            circuit.x(wire)
    return circuit


def workload_circuit(n_wires: int, depth: int, basis: str) -> fq.Circuit:
    circuit = fq.Circuit(n_wires)
    initial = tuple(0.37 + 0.31 * wire for wire in range(n_wires))
    for wire in range(n_wires):
        circuit.ry(wire, initial[wire])
    for layer in range(depth):
        for wire in range(n_wires):
            angle = 0.11 + 0.17 * (layer + 1) * (wire + 1)
            circuit.ry(wire, angle)
        for wire in range(n_wires - 1):
            circuit.cz(wire, wire + 1)
    if basis == "x":
        for wire in range(n_wires):
            circuit.h(wire)
    elif basis == "y":
        for wire in range(n_wires):
            circuit.sdg(wire).h(wire)
    elif basis != "z":
        raise ValueError(f"unsupported basis {basis!r}")
    return circuit


def _backend(provider: fq.QuafuProvider, n_wires: int) -> fq.CloudBackendProfile:
    return next(
        item for item in provider.discover_backends(n_wires) if item.name == "Baihua"
    )


def _logical_qasm(
    circuit: fq.Circuit, backend: fq.CloudBackendProfile, shots: int, name: str
) -> str:
    return fq.create_deployment_package(
        circuit, backend=backend, shots=shots, name=name
    ).qasm


def _precompile(qasm: str, chip_info: dict[str, Any], mapping: tuple[int, ...]):
    from quark.circuit import Backend, Transpiler

    with redirect_stdout(io.StringIO()):
        compiled = Transpiler(Backend(chip_info)).run(
            qasm, target_qubits=list(mapping), optimize_level=1
        )
    return {
        "openqasm2": str(compiled.to_openqasm2),
        "qlisp": compiled.to_qlisp,
        "depth": int(compiled.depth),
        "cz_count": int(compiled.ncz),
    }


def _physical_circuit(
    qlisp: list[Any], mapping: tuple[int, ...]
) -> fq.Circuit:
    physical_to_logical = {wire: index for index, wire in enumerate(mapping)}
    circuit = fq.Circuit(len(mapping))
    for item in qlisp:
        operation, target = item
        if isinstance(operation, (tuple, list)):
            name = str(operation[0]).lower()
            if name == "measure":
                continue
            if name != "u":
                raise ValueError(f"unsupported compiled operation {operation!r}")
            physical = int(str(target).lstrip("Qq"))
            circuit.u3(
                physical_to_logical[physical],
                float(operation[1]),
                float(operation[2]),
                float(operation[3]),
            )
            continue
        name = str(operation).lower()
        if name == "barrier":
            continue
        if name != "cz":
            raise ValueError(f"unsupported compiled operation {operation!r}")
        left, right = (int(str(wire).lstrip("Qq")) for wire in target)
        circuit.cz(physical_to_logical[left], physical_to_logical[right])
    return circuit


def _normalize_counts(raw_counts: dict[str, int], n_wires: int) -> dict[str, int]:
    states = ("".join(bits) for bits in product("01", repeat=n_wires))
    # Quafu serializes c[n-1]...c[0]; all archived analysis uses q0...q[n-1].
    return {
        state: int(raw_counts.get(state[::-1], 0))
        for state in states
    }


def _probabilities(counts: dict[str, int]) -> list[float]:
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("counts are empty")
    return [counts[state] / total for state in sorted(counts)]


def _simulate_probabilities(
    compiled: dict[str, Any],
    mapping: tuple[int, ...],
    model: fq.NoiseModel | None,
) -> list[float]:
    circuit = _physical_circuit(compiled["qlisp"], mapping)
    if model is None:
        probabilities = circuit.probabilities().reshape(-1)
    else:
        rho = fq.noisy_density_matrix(circuit, model)
        probabilities = rho.diagonal(dim1=-2, dim2=-1).real.reshape(-1)
        probabilities = model.apply_readout_probabilities(
            probabilities, n_wires=len(mapping)
        )
    return [float(value) for value in probabilities]


def _tv(left: list[float], right: list[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(left, right, strict=True))


def _run_task(
    provider: fq.QuafuProvider,
    backend: fq.CloudBackendProfile,
    circuit: fq.Circuit,
    *,
    name: str,
    shots: int,
    mapping: tuple[int, ...],
    physical_qasm: str | None = None,
) -> dict[str, Any]:
    if physical_qasm is not None:
        submitted_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        handle = provider.submit_physical_qasm(
            physical_qasm,
            chip=backend.name,
            name=name,
            shots=shots,
        )
        deadline = time.monotonic() + provider.result_timeout
        while True:
            status = provider.query_status(handle)
            if status in {"Finished", "Completed", "Done"}:
                result = provider.fetch_result(handle)
                break
            if status in {"Failed", "Cancelled", "Canceled"}:
                raise RuntimeError(f"Quafu task {handle.task_id} ended as {status}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Quafu task {handle.task_id} timed out")
            time.sleep(provider.poll_interval)
        return {
            "task_id": result.handle.task_id,
            "submitted_at": submitted_at,
            "physical_qasm": physical_qasm,
            "physical_qasm_sha256": handle.payload["physical_qasm_sha256"],
            "raw_quafu_counts_c_reverse": dict(result.counts),
            "result_metadata": dict(result.metadata),
        }
    package = fq.create_deployment_package(
        circuit,
        backend=backend,
        name=name,
        shots=shots,
        metadata={
            "provider_compile": True,
            "provider_options": {
                "compiler": "quarkcircuit",
                "correct": False,
                "open_dd": None,
                "target_qubits": list(mapping),
            },
        },
    )
    submitted_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    result = provider.run(package)
    return {
        "task_id": result.handle.task_id,
        "submitted_at": submitted_at,
        "logical_qasm": package.qasm,
        "raw_quafu_counts_c_reverse": dict(result.counts),
        "result_metadata": dict(result.metadata),
    }


def _manifest(
    provider: fq.QuafuProvider,
    chip_info: dict[str, Any],
    shots: int,
    physical_execution: bool,
    batch: str,
    mappings: dict[int, tuple[int, ...]],
    depths: dict[int, tuple[int, ...]],
) -> dict[str, Any]:
    workloads = []
    assignments = []
    for n_wires, mapping in mappings.items():
        backend = _backend(provider, n_wires)
        for bits in ("".join(value) for value in product("01", repeat=n_wires)):
            circuit = assignment_circuit(bits)
            name = f"fq_prxq_b01_ro_n{n_wires}_{bits}"
            qasm = _logical_qasm(circuit, backend, shots, name)
            assignments.append(
                {
                    "key": f"n{n_wires}:{bits}",
                    "n_wires": n_wires,
                    "mapping": list(mapping),
                    "prepared": bits,
                    "logical_qasm": qasm,
                    "compiled": _precompile(qasm, chip_info, mapping),
                }
            )
        for depth in depths[n_wires]:
            for basis in BASES:
                circuit = workload_circuit(n_wires, depth, basis)
                name = f"fq_prxq_b01_n{n_wires}_d{depth}_{basis}"
                qasm = _logical_qasm(circuit, backend, shots, name)
                workloads.append(
                    {
                        "key": f"n{n_wires}:d{depth}:{basis}",
                        "n_wires": n_wires,
                        "mapping": list(mapping),
                        "depth_parameter": depth,
                        "basis": basis,
                        "logical_qasm": qasm,
                        "compiled": _precompile(qasm, chip_info, mapping),
                    }
                )
    payload = {
        "schema": "flagquantum.quafu_predictive_batch_manifest.v1",
        "batch": batch,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "chip": "Baihua",
        "calibration_time": chip_info["calibration_time"],
        "shots_per_task": shots,
        "mappings": {str(key): list(value) for key, value in mappings.items()},
        "depths": {str(key): list(value) for key, value in depths.items()},
        "bases": list(BASES),
        "execution_mode": (
            "sealed_physical_qasm" if physical_execution else "provider_compile"
        ),
        "assignment_tasks": assignments,
        "heldout_workloads": workloads,
        "rules": {
            "model_inputs": "chip_info plus assignment tasks only",
            "heldout_tuning": "prohibited",
            "failed_tasks": "retained; no automatic resubmission",
            "provider_compile": not physical_execution,
        },
    }
    payload["manifest_sha256"] = _sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=10240)
    parser.add_argument("--batch", default="baihua_entangling_01")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/quafu_predictive_batch_01"),
    )
    parser.add_argument(
        "--physical-qasm",
        action="store_true",
        help="Execute the sealed precompiled OpenQASM with provider compilation off.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--scale-study",
        action="store_true",
        help="Freeze the prespecified 4--6-qubit, 0--12-layer scale study.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Freeze calibration, workloads, and physical compilations without submitting.",
    )
    args = parser.parse_args()
    if not os.getenv("QPU_API_TOKEN"):
        raise RuntimeError("QPU_API_TOKEN is required")
    if args.shots <= 0 or args.shots % 1024:
        raise ValueError("Quafu shots must be a positive multiple of 1024")

    args.output.mkdir(parents=True, exist_ok=True)
    mappings = SCALE_MAPPINGS if args.scale_study else DEFAULT_MAPPINGS
    depths = SCALE_DEPTHS if args.scale_study else DEFAULT_DEPTHS
    provider = fq.QuafuProvider(
        timeout=30, poll_interval=5, result_timeout=args.timeout
    )
    manifest_path = args.output / "frozen_manifest.json"
    chip_path = args.output / "chip_info.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(f"{manifest_path} exists; pass --resume")
        manifest = json.loads(manifest_path.read_text())
        chip_info = json.loads(chip_path.read_text())
        requested_mode = (
            "sealed_physical_qasm" if args.physical_qasm else "provider_compile"
        )
        if manifest.get("execution_mode") != requested_mode:
            raise ValueError(
                f"resume mode mismatch: manifest uses {manifest.get('execution_mode')}, "
                f"requested {requested_mode}"
            )
    else:
        chip_info = dict(provider.fetch_chip_info("Baihua"))
        _dump(chip_path, chip_info)
        manifest = _manifest(
            provider,
            chip_info,
            args.shots,
            args.physical_qasm,
            args.batch,
            mappings,
            depths,
        )
        _dump(manifest_path, manifest)
        print(f"frozen manifest={manifest['manifest_sha256']}", flush=True)

    if args.prepare_only:
        print(
            f"prepared {len(manifest['assignment_tasks'])} assignment and "
            f"{len(manifest['heldout_workloads'])} held-out tasks",
            flush=True,
        )
        return

    assignment_path = args.output / "assignment_results.json"
    assignment_payload = (
        json.loads(assignment_path.read_text())
        if args.resume and assignment_path.exists()
        else {
            "schema": "flagquantum.quafu_assignment_results.v1",
            "manifest_sha256": manifest["manifest_sha256"],
            "results": {},
        }
    )
    for specification in manifest["assignment_tasks"]:
        key = specification["key"]
        if key in assignment_payload["results"]:
            continue
        n_wires = int(specification["n_wires"])
        mapping = tuple(specification["mapping"])
        try:
            record = _run_task(
                provider,
                _backend(provider, n_wires),
                assignment_circuit(specification["prepared"]),
                name="fq_prxq_b01_ro_" + key.replace(":", "_"),
                shots=args.shots,
                mapping=mapping,
                physical_qasm=(
                    specification["compiled"]["openqasm2"]
                    if args.physical_qasm
                    else None
                ),
            )
            record["counts_q0_to_qn"] = _normalize_counts(
                record["raw_quafu_counts_c_reverse"], n_wires
            )
            record["status"] = "finished"
        except Exception as error:
            record = {"status": "failed", "error": repr(error)}
        assignment_payload["results"][key] = record
        _dump(assignment_path, assignment_payload)
        print(f"assignment {key} {record['status']}", flush=True)
        if record["status"] != "finished":
            raise RuntimeError(f"assignment task {key} failed; inspect artifact")

    matrices: dict[int, list[list[float]]] = {}
    active_mappings = {
        int(key): tuple(value) for key, value in manifest["mappings"].items()
    }
    for n_wires in active_mappings:
        rows = []
        for bits in ("".join(value) for value in product("01", repeat=n_wires)):
            counts = assignment_payload["results"][f"n{n_wires}:{bits}"][
                "counts_q0_to_qn"
            ]
            rows.append(_probabilities(counts))
        matrices[n_wires] = rows

    models = {
        n_wires: fq.quafu_noise_model_from_chip_info(
            chip_info,
            physical_qubits=mapping,
            correlated_readout_confusion_matrix=matrices[n_wires],
        )
        for n_wires, mapping in active_mappings.items()
    }
    frozen_model = {
        "schema": "flagquantum.quafu_frozen_models.v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "models": {
            str(n_wires): {
                "identity": model.identity,
                "specification": model.to_dict(),
                "assignment_matrix": matrices[n_wires],
            }
            for n_wires, model in models.items()
        },
    }
    frozen_model["frozen_models_sha256"] = _sha256(frozen_model)
    _dump(args.output / "frozen_models.json", frozen_model)
    print(f"frozen models={frozen_model['frozen_models_sha256']}", flush=True)

    heldout_path = args.output / "heldout_results.json"
    heldout_payload = (
        json.loads(heldout_path.read_text())
        if args.resume and heldout_path.exists()
        else {
            "schema": "flagquantum.quafu_heldout_results.v1",
            "manifest_sha256": manifest["manifest_sha256"],
            "frozen_models_sha256": frozen_model["frozen_models_sha256"],
            "results": {},
        }
    )
    for specification in manifest["heldout_workloads"]:
        key = specification["key"]
        if key in heldout_payload["results"]:
            continue
        n_wires = int(specification["n_wires"])
        mapping = tuple(specification["mapping"])
        circuit = workload_circuit(
            n_wires,
            int(specification["depth_parameter"]),
            specification["basis"],
        )
        ideal = _simulate_probabilities(specification["compiled"], mapping, None)
        noisy = _simulate_probabilities(
            specification["compiled"], mapping, models[n_wires]
        )
        try:
            record = _run_task(
                provider,
                _backend(provider, n_wires),
                circuit,
                name="fq_prxq_b01_" + key.replace(":", "_"),
                shots=args.shots,
                mapping=mapping,
                physical_qasm=(
                    specification["compiled"]["openqasm2"]
                    if args.physical_qasm
                    else None
                ),
            )
            record["counts_q0_to_qn"] = _normalize_counts(
                record["raw_quafu_counts_c_reverse"], n_wires
            )
            hardware = _probabilities(record["counts_q0_to_qn"])
            record.update(
                {
                    "status": "finished",
                    "statevector_probabilities": ideal,
                    "noise_model_probabilities": noisy,
                    "hardware_probabilities": hardware,
                    "tv_statevector_hardware": _tv(ideal, hardware),
                    "tv_noise_model_hardware": _tv(noisy, hardware),
                }
            )
        except Exception as error:
            record = {"status": "failed", "error": repr(error)}
        heldout_payload["results"][key] = record
        _dump(heldout_path, heldout_payload)
        print(f"heldout {key} {record['status']}", flush=True)
        if record["status"] != "finished":
            raise RuntimeError(f"heldout task {key} failed; inspect artifact")

    summary = {}
    for n_wires in active_mappings:
        rows = [
            value
            for key, value in heldout_payload["results"].items()
            if key.startswith(f"n{n_wires}:") and value["status"] == "finished"
        ]
        summary[str(n_wires)] = {
            "tasks": len(rows),
            "mean_tv_statevector_hardware": sum(
                row["tv_statevector_hardware"] for row in rows
            )
            / len(rows),
            "mean_tv_noise_model_hardware": sum(
                row["tv_noise_model_hardware"] for row in rows
            )
            / len(rows),
        }
    heldout_payload["summary"] = summary
    _dump(heldout_path, heldout_payload)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main()

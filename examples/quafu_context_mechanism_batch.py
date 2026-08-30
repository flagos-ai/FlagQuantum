"""Run a frozen six-qubit matched-CZ spatial-context experiment on Baihua."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import time
from itertools import product
from pathlib import Path

import torch

import flagquantum as fq

ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = Path(__file__).with_name("quafu_predictive_entangling_batch.py")
MAPPING = (124, 125, 126, 127, 128, 129)
CZ_COUNTS = (10, 20, 40, 60)
FAMILIES = ("hotspot", "matching")
BASES = ("z", "x", "y")


def _base():
    specification = importlib.util.spec_from_file_location("quafu_base", BASE_SCRIPT)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


def context_circuit(family: str, cz_count: int, basis: str) -> fq.Circuit:
    circuit = fq.Circuit(6)
    for wire in range(6):
        circuit.ry(wire, 0.29 + 0.23 * wire)
    matching = ((0, 1), (2, 3), (4, 5))
    for index in range(cz_count):
        if index % 5 == 0:
            for wire in range(6):
                circuit.ry(wire, 0.03 + 0.011 * (index + 1) * (wire + 1))
        if family == "hotspot":
            circuit.cz(2, 3)
        elif family == "matching":
            circuit.cz(*matching[index % len(matching)])
        else:
            raise ValueError(f"unsupported family {family!r}")
    if basis == "x":
        for wire in range(6):
            circuit.h(wire)
    elif basis == "y":
        for wire in range(6):
            circuit.sdg(wire).h(wire)
    elif basis != "z":
        raise ValueError(f"unsupported basis {basis!r}")
    return circuit


def _physical_program(
    *,
    prepared: str | None = None,
    family: str | None = None,
    cz_count: int = 0,
    basis: str = "z",
) -> dict:
    qlisp = []
    if prepared is not None:
        for logical, bit in enumerate(prepared):
            if bit == "1":
                qlisp.append(
                    (("U", math.pi, math.pi / 2, -math.pi / 2), f"Q{MAPPING[logical]}")
                )
    else:
        for logical, physical in enumerate(MAPPING):
            qlisp.append((("U", 0.29 + 0.23 * logical, 0.0, 0.0), f"Q{physical}"))
        matching = ((0, 1), (2, 3), (4, 5))
        for index in range(cz_count):
            if index % 5 == 0:
                for logical, physical in enumerate(MAPPING):
                    angle = 0.03 + 0.011 * (index + 1) * (logical + 1)
                    qlisp.append((("U", angle, 0.0, 0.0), f"Q{physical}"))
            edge = (2, 3) if family == "hotspot" else matching[index % 3]
            qlisp.append(("CZ", tuple(f"Q{MAPPING[wire]}" for wire in edge)))
        terminal = {
            "z": None,
            "x": (math.pi / 2, 0.0, math.pi),
            "y": (math.pi / 2, 0.0, math.pi / 2),
        }[basis]
        if terminal is not None:
            for physical in MAPPING:
                qlisp.append((("U", *terminal), f"Q{physical}"))
    qlisp.extend(
        (("Measure", logical), f"Q{physical}")
        for logical, physical in enumerate(MAPPING)
    )
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', "qreg q[130];", "creg c[6];"]
    for operation, target in qlisp:
        if isinstance(operation, tuple):
            if operation[0] == "Measure":
                lines.append(f"measure q[{int(str(target)[1:])}] -> c[{operation[1]}];")
            else:
                lines.append(
                    f"u({operation[1]},{operation[2]},{operation[3]}) "
                    f"q[{int(str(target)[1:])}];"
                )
        else:
            left, right = (int(str(wire)[1:]) for wire in target)
            lines.append(f"cz q[{left}],q[{right}];")
    return {
        "openqasm2": "\n".join(lines) + "\n",
        "qlisp": qlisp,
        "cz_count": cz_count,
    }


def _manifest(base, provider, chip_info, shots: int) -> dict:
    backend = base._backend(provider, 6)
    assignments = []
    for bits in ("".join(value) for value in product("01", repeat=6)):
        circuit = base.assignment_circuit(bits)
        qasm = base._logical_qasm(circuit, backend, shots, f"fq_ctx_ro_{bits}")
        assignments.append(
            {
                "key": f"ro:{bits}",
                "prepared": bits,
                "logical_qasm": qasm,
                "compiled": _physical_program(prepared=bits),
            }
        )
    workloads = []
    for family in FAMILIES:
        for cz_count in CZ_COUNTS:
            for basis in BASES:
                circuit = context_circuit(family, cz_count, basis)
                key = f"{family}:c{cz_count}:{basis}"
                qasm = base._logical_qasm(
                    circuit, backend, shots, "fq_ctx_" + key.replace(":", "_")
                )
                workloads.append(
                    {
                        "key": key,
                        "family": family,
                        "cz_count": cz_count,
                        "basis": basis,
                        "logical_qasm": qasm,
                        "compiled": _physical_program(
                            family=family, cz_count=cz_count, basis=basis
                        ),
                    }
                )
    payload = {
        "schema": "flagquantum.quafu_context_manifest.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "chip": "Baihua",
        "calibration_time": chip_info["calibration_time"],
        "shots_per_task": shots,
        "mapping": list(MAPPING),
        "families": list(FAMILIES),
        "cz_counts": list(CZ_COUNTS),
        "bases": list(BASES),
        "execution_mode": "compile_false_request",
        "assignment_tasks": assignments,
        "heldout_workloads": workloads,
        "rules": {
            "primary_contrast": "paired TV residual: hotspot minus matching",
            "model_inputs": "chip_info plus assignment tasks only",
            "heldout_tuning": "prohibited",
            "failed_tasks": "retained; no automatic resubmission",
        },
    }
    payload["manifest_sha256"] = base._sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=10240)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/development/quafu_context_batch"),
    )
    args = parser.parse_args()
    if not os.getenv("QPU_API_TOKEN"):
        raise RuntimeError("QPU_API_TOKEN is required")
    if args.shots <= 0 or args.shots % 1024:
        raise ValueError("shots must be a positive multiple of 1024")
    base = _base()
    args.output.mkdir(parents=True, exist_ok=True)
    provider = fq.QuafuProvider(
        timeout=30, poll_interval=5, result_timeout=args.timeout
    )
    backend = base._backend(provider, 6)
    manifest_path = args.output / "frozen_manifest.json"
    chip_path = args.output / "chip_info.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(f"{manifest_path} exists; pass --resume")
        manifest = json.loads(manifest_path.read_text())
        chip_info = json.loads(chip_path.read_text())
    else:
        chip_info = dict(provider.fetch_chip_info("Baihua"))
        base._dump(chip_path, chip_info)
        manifest = _manifest(base, provider, chip_info, args.shots)
        base._dump(manifest_path, manifest)
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
        else {"manifest_sha256": manifest["manifest_sha256"], "results": {}}
    )
    for specification in manifest["assignment_tasks"]:
        key = specification["key"]
        if key in assignment_payload["results"]:
            continue
        try:
            record = base._run_task(
                provider,
                backend,
                base.assignment_circuit(specification["prepared"]),
                name="fq_ctx_" + key.replace(":", "_"),
                shots=args.shots,
                mapping=MAPPING,
                physical_qasm=specification["compiled"]["openqasm2"],
            )
            record["counts_q0_to_qn"] = base._normalize_counts(
                record["raw_quafu_counts_c_reverse"], 6
            )
            record["status"] = "finished"
        except Exception as error:
            record = {"status": "failed", "error": repr(error)}
        assignment_payload["results"][key] = record
        base._dump(assignment_path, assignment_payload)
        print(f"assignment {key} {record['status']}", flush=True)
        if record["status"] != "finished":
            raise RuntimeError(f"assignment task {key} failed")

    matrix = []
    for bits in ("".join(value) for value in product("01", repeat=6)):
        counts = assignment_payload["results"][f"ro:{bits}"]["counts_q0_to_qn"]
        matrix.append(base._probabilities(counts))
    model = fq.quafu_noise_model_from_chip_info(
        chip_info,
        physical_qubits=MAPPING,
        correlated_readout_confusion_matrix=matrix,
    )
    frozen = {
        "manifest_sha256": manifest["manifest_sha256"],
        "identity": model.identity,
        "specification": model.to_dict(),
        "assignment_matrix": matrix,
    }
    frozen["frozen_model_sha256"] = base._sha256(frozen)
    base._dump(args.output / "frozen_model.json", frozen)
    print(f"frozen model={frozen['frozen_model_sha256']}", flush=True)

    heldout_path = args.output / "heldout_results.json"
    heldout_payload = (
        json.loads(heldout_path.read_text())
        if args.resume and heldout_path.exists()
        else {
            "manifest_sha256": manifest["manifest_sha256"],
            "frozen_model_sha256": frozen["frozen_model_sha256"],
            "results": {},
        }
    )
    for specification in manifest["heldout_workloads"]:
        key = specification["key"]
        if key in heldout_payload["results"]:
            continue
        circuit = context_circuit(
            specification["family"], specification["cz_count"], specification["basis"]
        )
        with torch.no_grad():
            ideal = base._simulate_probabilities(
                specification["compiled"], MAPPING, None
            )
            noisy = base._simulate_probabilities(
                specification["compiled"], MAPPING, model
            )
        try:
            record = base._run_task(
                provider,
                backend,
                circuit,
                name="fq_ctx_" + key.replace(":", "_"),
                shots=args.shots,
                mapping=MAPPING,
                physical_qasm=specification["compiled"]["openqasm2"],
            )
            record["counts_q0_to_qn"] = base._normalize_counts(
                record["raw_quafu_counts_c_reverse"], 6
            )
            hardware = base._probabilities(record["counts_q0_to_qn"])
            record.update(
                {
                    "status": "finished",
                    "statevector_probabilities": ideal,
                    "noise_model_probabilities": noisy,
                    "hardware_probabilities": hardware,
                    "tv_statevector_hardware": base._tv(ideal, hardware),
                    "tv_noise_model_hardware": base._tv(noisy, hardware),
                }
            )
        except Exception as error:
            record = {"status": "failed", "error": repr(error)}
        heldout_payload["results"][key] = record
        base._dump(heldout_path, heldout_payload)
        print(f"heldout {key} {record['status']}", flush=True)
        if record["status"] != "finished":
            raise RuntimeError(f"heldout task {key} failed")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main()

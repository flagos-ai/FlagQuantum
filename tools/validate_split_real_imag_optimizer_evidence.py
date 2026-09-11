#!/usr/bin/env python3
"""Validate checked-in P5 Double-Single SGD A800 portability evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "artifacts/split_real_imag_optimizer_a800_20260825.json"
CONTRACT = (
    ROOT
    / "contracts"
    / "split-real-imag-statevector-p5-autograd-optimizer-contract.toml"
)
HISTORICAL_OPERATOR_PROFILE = {
    "name": "split_real_imag_statevector_p4_device_double_single",
    "sha256": "beb0e58248a93d5f9ab17fe61b869f0a4c2925039c29eb9426101ef0c569ac73",
}


def _load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def evidence_errors(payload: dict[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if payload.get("schema") != (
        "flagquantum_split_real_imag_p5_optimizer_a800_evidence_v1"
    ):
        errors.append("unexpected split P5 optimizer A800 evidence schema")
    if payload.get("status") != "passed":
        errors.append("split P5 optimizer A800 evidence must have passed status")

    source = payload.get("source", {})
    if (
        len(str(source.get("revision", ""))) != 40
        or source.get("tree_dirty") is not False
        or len(str(source.get("archive_sha256", ""))) != 64
    ):
        errors.append("split P5 optimizer evidence source identity is incomplete")

    if payload.get("operator_profile") != HISTORICAL_OPERATOR_PROFILE:
        errors.append("split P5 optimizer historical operator profile identity drifted")

    contract = tomllib.loads(CONTRACT.read_text(encoding="utf-8"))
    acceptance = contract["acceptance_thresholds"]
    thresholds = payload.get("thresholds", {})
    if thresholds != {
        "max_optimizer_parameter_absolute_error": acceptance[
            "max_optimizer_parameter_absolute_error"
        ],
        "max_optimizer_loss_trajectory_absolute_error": acceptance[
            "max_optimizer_loss_trajectory_absolute_error"
        ],
        "double_single_must_not_exceed_float32_baseline": True,
    }:
        errors.append("split P5 optimizer evidence thresholds drifted")
    matrix = payload.get("matrix", {})
    expected_cases = {
        (workload, step)
        for workload in ("two_qubit_vqe", "three_qubit_qaoa")
        for step in contract["acceptance_matrix"]["optimizer_steps"]
    }
    if matrix != {
        "optimizer_steps": contract["acceptance_matrix"]["optimizer_steps"],
        "workloads": ["two_qubit_vqe", "three_qubit_qaoa"],
    }:
        errors.append("split P5 optimizer evidence matrix drifted")
    cases = payload.get("cases", ())
    if {(case.get("workload"), case.get("step")) for case in cases} != expected_cases:
        errors.append("split P5 optimizer trajectory coverage is incomplete")
    for case in cases:
        if not case.get("passed", False):
            errors.append("split P5 optimizer evidence contains a failed case")
        ds_parameter = float(
            case.get("double_single_parameter_absolute_error", float("inf"))
        )
        fp32_parameter = float(
            case.get("float32_parameter_absolute_error", float("-inf"))
        )
        ds_loss = float(case.get("double_single_loss_absolute_error", float("inf")))
        fp32_loss = float(case.get("float32_loss_absolute_error", float("-inf")))
        if (
            ds_parameter > float(acceptance["max_optimizer_parameter_absolute_error"])
            or ds_loss
            > float(acceptance["max_optimizer_loss_trajectory_absolute_error"])
            or ds_parameter > fp32_parameter
            or ds_loss > fp32_loss
        ):
            errors.append("split P5 optimizer trajectory exceeds its evidence envelope")

    cancellation = payload.get("cancellation", {})
    if (
        cancellation.get("steps") != 64
        or cancellation.get("double_single_absolute_error") != 0.0
        or float(cancellation.get("float32_absolute_error", 0.0)) <= 0.0
        or cancellation.get("passed") is not True
    ):
        errors.append("split P5 optimizer sub-ULP evidence is incomplete")

    environment = payload.get("environment", {})
    expected_locks = {
        "native_cuda_image_id": "sha256:a6a6d2c4d50c9ac503eeae562e841add24a1b71a7e70410709c6ca6a87722d7b",
        "flagos_image_id": "sha256:8244e20acbad1a3ede024404b13d129fa6d83933e1320f93b75e404a85efbb4b",
        "torch_fl_revision": "2e00b393cf80088706b460a187aef185d3a283f4",
        "torch_fl_source_archive_sha256": "efccb4f92615b8f159c17d0f03865de74801d437f2130e70901e54332bdcc916",
        "torch_fl_wheel_sha256": "cca7953ac2aed376d83fd7d0dc931c8b2ac0d59d70d4f841691b1f01217f4fec",
    }
    if any(environment.get(name) != value for name, value in expected_locks.items()):
        errors.append("split P5 optimizer environment lock identity drifted")

    routes = {route.get("name"): route for route in payload.get("routes", ())}
    common = {
        "status": "passed",
        "logical_device_residency": True,
        "parameter_word_count": 2,
        "tensor_grad_used": False,
        "accelerator_float64_tensor_materialized": False,
    }
    native = routes.get("native_cuda", {})
    if (
        native.get("device") != "cuda:0"
        or native.get("provider") != "pytorch_cuda"
        or any(native.get(name) != value for name, value in common.items())
        or len(str(native.get("raw_result_sha256", ""))) != 64
    ):
        errors.append("split P5 optimizer native CUDA route is incomplete")
    flagos = routes.get("torch_fl_flagos", {})
    if (
        flagos.get("device") != "flagos:0"
        or flagos.get("provider") != "torch_fl"
        or any(flagos.get(name) != value for name, value in common.items())
        or flagos.get("provider_internal_route_audited") is not False
        or flagos.get("flagcx_collectives_validated") is not False
        or len(str(flagos.get("raw_result_sha256", ""))) != 64
    ):
        errors.append("split P5 optimizer Torch-FL flagos route is incomplete")

    equivalence = payload.get("route_metric_equivalence", {})
    if equivalence != {
        "native_cuda_vs_torch_fl_flagos_max_absolute_delta": 0.0,
        "passed": True,
    }:
        errors.append("split P5 optimizer route equivalence evidence drifted")

    claims = payload.get("claims", {})
    required_true = {
        "double_single_optimizer",
        "sub_ulp_updates_retained",
        "native_cuda_portability_evidence",
        "torch_fl_flagos_portability_evidence",
    }
    required_false = {
        "complex128_equivalence",
        "torch_optimizer_compatibility",
        "convergence_certification",
        "hardware_certification",
        "domestic_accelerator_certification",
        "provider_internal_route_audited",
        "flagcx_collectives_validated",
        "scalability_claim_allowed",
        "performance_claim_allowed",
        "production_claim_allowed",
    }
    if claims.get("maturity") != "experimental" or any(
        claims.get(name) is not True for name in required_true
    ):
        errors.append("split P5 optimizer portability claims are incomplete")
    if any(claims.get(name) is not False for name in required_false):
        errors.append("split P5 optimizer evidence contains a forbidden promotion")
    return tuple(errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="?", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    errors = evidence_errors(_load_json(args.artifact))
    if errors:
        print("\n".join(errors))
        return 1
    print("Split real/imag P5 optimizer A800 evidence passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

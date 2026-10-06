"""Compute a credential-free QBoson submission plan for frozen QDiffusion."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, cast

from examples.qdiffusion_kaiwu.strict_json import loads_json_strict

SCHEMA = "flagquantum.qboson_qdiffusion_quota_plan"
SYSTEM_MAX_CALLS_PER_HOST = 10
SYSTEM_HOST_COUNT = 2
SMOKE_CALLS = 2


def protein_remote_call_breakdown(config: dict[str, Any]) -> dict[str, int] | None:
    """Return the pinned workflow's worst-case distinct calls for one seed."""

    dataset = config.get("dataset")
    training = config.get("training")
    generation = config.get("generation")
    if not all(
        isinstance(section, dict) for section in (dataset, training, generation)
    ):
        return None
    assert isinstance(dataset, dict)
    assert isinstance(training, dict)
    assert isinstance(generation, dict)
    integer_fields = (
        dataset.get("max_records"),
        training.get("epochs"),
        training.get("num_candidates"),
        training.get("validation_steps"),
        generation.get("sequence_count"),
        generation.get("max_steps"),
        generation.get("num_candidates"),
    )
    if any(type(value) is not int or value <= 0 for value in integer_fields):
        return None
    validation_ratio = dataset.get("validation_ratio")
    test_ratio = dataset.get("test_ratio")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (validation_ratio, test_ratio)
    ):
        return None
    selected = int(dataset["max_records"])
    validation_count = max(
        1, int(selected * float(cast(int | float, validation_ratio)))
    )
    test_count = max(1, int(selected * float(cast(int | float, test_ratio))))
    train_count = selected - validation_count - test_count
    if train_count <= 0 or generation["sequence_count"] != test_count:
        return None
    training_candidates = int(training["num_candidates"])
    generation_candidates = int(generation["num_candidates"])
    validation_steps = int(training["validation_steps"])
    epochs = int(training["epochs"])
    max_steps = int(generation["max_steps"])

    structural_calls = 2 + validation_steps
    training_calls_per_epoch = (train_count + validation_count) * (
        1 + training_candidates
    )
    training_calls = epochs * training_calls_per_epoch
    baseline_calls_per_record = 2 + max_steps
    guided_calls_per_record = (
        1 + generation_candidates + max_steps * generation_candidates
    )
    generation_calls = test_count * (
        baseline_calls_per_record + guided_calls_per_record
    )
    total = structural_calls + training_calls + generation_calls
    return {
        "selected_records": selected,
        "training_records": train_count,
        "validation_records": validation_count,
        "test_records": test_count,
        "structural_calls": structural_calls,
        "training_calls_per_epoch": training_calls_per_epoch,
        "training_calls": training_calls,
        "baseline_calls_per_record": baseline_calls_per_record,
        "guided_calls_per_record": guided_calls_per_record,
        "generation_calls": generation_calls,
        "total": total,
    }


def estimate_protein_remote_calls(config: dict[str, Any]) -> int | None:
    breakdown = protein_remote_call_breakdown(config)
    return breakdown["total"] if breakdown is not None else None


def portability_remote_call_breakdown(
    config: dict[str, Any],
) -> dict[str, int] | None:
    generation = config.get("generation")
    if not isinstance(generation, dict):
        return None
    candidates = generation.get("num_candidates")
    steps = generation.get("portability_steps")
    if type(candidates) is not int or candidates <= 0:
        return None
    if type(steps) is not int or steps <= 0:
        return None
    initialization_calls = 1 + candidates
    guided_step_calls = steps * candidates
    return {
        "initialization_calls": initialization_calls,
        "guided_step_calls": guided_step_calls,
        "total": initialization_calls + guided_step_calls,
    }


def estimate_portability_remote_calls(config: dict[str, Any]) -> int | None:
    breakdown = portability_remote_call_breakdown(config)
    return breakdown["total"] if breakdown is not None else None


def _positive_integer_or_required(value: Any, label: str) -> int | None:
    if value == "<required>":
        return None
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer or <required>")
    return value


def build_quota_plan(config: dict[str, Any], config_sha256: str) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{64}", config_sha256) is None:
        raise ValueError("config_sha256 must be a lowercase SHA-256 digest")
    protein = protein_remote_call_breakdown(config)
    portability = portability_remote_call_breakdown(config)
    if protein is None:
        raise ValueError("cannot derive the protein submission estimate")
    if portability is None:
        raise ValueError("cannot derive the portability submission estimate")
    seeds = config.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must contain unique integers")
    system_ceiling = _positive_integer_or_required(
        config.get("remote_call_budget"), "remote_call_budget"
    )
    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError("training must be an object")
    protein_ceiling = _positive_integer_or_required(
        training.get("remote_call_budget_per_seed"),
        "training.remote_call_budget_per_seed",
    )
    if system_ceiling is not None:
        if system_ceiling < SYSTEM_MAX_CALLS_PER_HOST:
            raise ValueError("remote_call_budget is below the system-slice estimate")
        if system_ceiling < portability["total"]:
            raise ValueError("remote_call_budget is below the portability estimate")
    if protein_ceiling is not None and protein_ceiling < protein["total"]:
        raise ValueError(
            "training.remote_call_budget_per_seed is below the protein estimate"
        )

    seed_count = len(seeds)
    estimated_total = (
        SMOKE_CALLS
        + SYSTEM_MAX_CALLS_PER_HOST * SYSTEM_HOST_COUNT
        + protein["total"] * seed_count
        + portability["total"]
    )
    declared_ceiling_total = (
        None
        if system_ceiling is None or protein_ceiling is None
        else SMOKE_CALLS
        + system_ceiling * SYSTEM_HOST_COUNT
        + protein_ceiling * seed_count
        + system_ceiling
    )
    return {
        "schema": SCHEMA,
        "version": "1.0",
        "experiment_config_sha256": config_sha256,
        "unit": "distinct_provider_task_submissions",
        "estimate_kind": "conservative_maximum",
        "includes_status_polls": False,
        "includes_result_retrievals": False,
        "smoke": {"estimated_max_calls": SMOKE_CALLS},
        "system": {
            "host_count": SYSTEM_HOST_COUNT,
            "estimated_max_calls_per_host": SYSTEM_MAX_CALLS_PER_HOST,
            "estimated_max_calls": SYSTEM_MAX_CALLS_PER_HOST * SYSTEM_HOST_COUNT,
            "declared_ceiling_per_host": system_ceiling,
            "declared_ceiling": (
                system_ceiling * SYSTEM_HOST_COUNT
                if system_ceiling is not None
                else None
            ),
        },
        "protein": {
            "seed_count": seed_count,
            "breakdown_per_seed": protein,
            "estimated_max_calls": protein["total"] * seed_count,
            "declared_ceiling_per_seed": protein_ceiling,
            "declared_ceiling": (
                protein_ceiling * seed_count if protein_ceiling is not None else None
            ),
        },
        "portability": {
            "breakdown": portability,
            "declared_ceiling": system_ceiling,
        },
        "totals": {
            "estimated_max_calls": estimated_total,
            "declared_ceiling": declared_ceiling_total,
            "budget_complete": declared_ceiling_total is not None,
        },
        "claim_boundary": (
            "Planning estimate only; it is not QBoson quota approval, provider "
            "evidence, execution evidence, or acceptance evidence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    arguments = parser.parse_args()
    raw = arguments.config.read_bytes()
    config = loads_json_strict(raw)
    if not isinstance(config, dict):
        parser.error("config must be a JSON object")
    try:
        plan = build_quota_plan(config, hashlib.sha256(raw).hexdigest())
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

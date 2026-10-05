"""Fail-closed validator for QBoson QDiffusion acceptance evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Any, cast

from examples.qdiffusion_kaiwu.source_preflight import (
    SCHEMA as SOURCE_PREFLIGHT_COMPONENT_SCHEMA,
)
from examples.qdiffusion_kaiwu.source_preflight import validate_source_preflight_record

HOSTS = {"jp-a800-171", "jp-a800-172"}
FULL_REVISION = re.compile(r"[0-9a-f]{40}")
CONFIG_SCHEMA = "flagquantum.qboson_qdiffusion_config"
RECORD_SCHEMA = "flagquantum.qboson_qdiffusion_acceptance"
MANIFEST_SCHEMA = "flagquantum.qboson_qdiffusion_manifest"
SYSTEM_COMPONENT_SCHEMA = "flagquantum.qboson_qdiffusion_system_live_probe"
PORTABILITY_COMPONENT_SCHEMA = "flagquantum.qboson_qdiffusion_portability_replay"
TRAINING_COMPONENT_SCHEMA = "flagquantum.qboson_qdiffusion_protein_training"
EVALUATION_COMPONENT_SCHEMA = "flagquantum.qboson_qdiffusion_protein_evaluation"
METRIC_NAMES = (
    "mean_cosine_distance",
    "median_cosine_distance",
    "mean_l2_distance",
    "median_l2_distance",
    "identity_to_reference_mean",
    "amino_acid_jsd",
    "kmer2_jsd",
    "kmer3_jsd",
    "uniqueness_ratio",
    "repeat_ratio_ge4",
    "length_match_ratio",
    "invalid_sequence_count",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any, label: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{label}: expected an object")
        return {}
    return value


def _finite_number(value: Any, label: str, errors: list[str]) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{label}: expected a finite number")
        return None
    number = float(value)
    if not math.isfinite(number):
        errors.append(f"{label}: expected a finite number")
        return None
    return number


def _estimate_protein_remote_calls(config: dict[str, Any]) -> int | None:
    """Estimate the workflow's worst-case distinct sampler calls per seed."""
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
    # Structural validation uses one candidate: one positive call, one negative
    # call, then one candidate-scoring call per generation step.
    structural_calls = 2 + validation_steps
    epoch_calls = epochs * (train_count + validation_count) * (1 + training_candidates)
    # Each baseline/guided record first executes objective(), then generate().
    baseline_calls_per_record = 2 + max_steps
    guided_calls_per_record = (
        1 + generation_candidates + max_steps * generation_candidates
    )
    generation_calls = test_count * (
        baseline_calls_per_record + guided_calls_per_record
    )
    return structural_calls + epoch_calls + generation_calls


def _estimate_portability_remote_calls(config: dict[str, Any]) -> int | None:
    generation = config.get("generation")
    if not isinstance(generation, dict):
        return None
    candidates = generation.get("num_candidates")
    steps = generation.get("portability_steps")
    if type(candidates) is not int or candidates <= 0:
        return None
    if type(steps) is not int or steps <= 0:
        return None
    return 1 + candidates + steps * candidates


def _validate_config(config: dict[str, Any], errors: list[str]) -> None:
    if config.get("schema") != CONFIG_SCHEMA or config.get("version") != "1.0":
        errors.append("config: unsupported schema or version")
    primary_host = config.get("primary_host")
    replay_host = config.get("replay_host")
    if primary_host not in HOSTS:
        errors.append("config.primary_host: expected one declared A800 host")
    if replay_host not in HOSTS or replay_host == primary_host:
        errors.append("config.replay_host: expected the other declared A800 host")
    preregistered_at = config.get("preregistered_at")
    try:
        timestamp = datetime.fromisoformat(str(preregistered_at).replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            raise ValueError
    except ValueError:
        errors.append("config.preregistered_at: expected a timezone-aware timestamp")
    seeds = config.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 3
        or any(type(seed) is not int for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        errors.append("config.seeds: expected at least three unique integer seeds")
    for name in (
        "dataset",
        "checkpoint",
        "tokenizer",
        "evaluation_model",
        "training",
        "generation",
        "evaluation",
    ):
        section = _mapping(config.get(name), f"config.{name}", errors)
        if not section or any(
            value is None or value == "" or value == "<required>"
            for value in section.values()
        ):
            errors.append(f"config.{name}: all frozen identity fields are required")
    for name in ("dataset", "checkpoint", "tokenizer", "evaluation_model"):
        digest = _mapping(config.get(name), f"config.{name}", errors).get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            errors.append(f"config.{name}.sha256: expected a SHA-256 digest")
    generation = _mapping(config.get("generation"), "config.generation", errors)
    for field in (
        "sequence_count",
        "max_steps",
        "num_candidates",
        "portability_steps",
    ):
        if type(generation.get(field)) is not int or generation.get(field, 0) <= 0:
            errors.append(f"config.generation.{field}: expected a positive integer")
    portability_seed = generation.get("portability_training_seed")
    if type(portability_seed) is not int or portability_seed not in config.get(
        "seeds", []
    ):
        errors.append(
            "config.generation.portability_training_seed: expected one frozen seed"
        )
    portability_index = generation.get("portability_fixture_index")
    if (
        type(portability_index) is not int
        or portability_index < 0
        or (
            type(generation.get("sequence_count")) is int
            and portability_index >= generation["sequence_count"]
        )
    ):
        errors.append(
            "config.generation.portability_fixture_index: outside the frozen sequence set"
        )
    for field in (
        "proposal_temperature",
        "proposal_noise_scale",
        "energy_temperature",
        "resample_ratio",
        "resample_top_p",
    ):
        value = _finite_number(
            generation.get(field), f"config.generation.{field}", errors
        )
        if value is not None and value < 0:
            errors.append(f"config.generation.{field}: expected a non-negative value")
    if type(generation.get("disable_resample")) is not bool:
        errors.append("config.generation.disable_resample: expected a boolean")
    for field in ("resample_ratio", "resample_top_p"):
        value = generation.get(field)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 1
        ):
            errors.append(f"config.generation.{field}: expected a value at most one")
    evaluation = _mapping(config.get("evaluation"), "config.evaluation", errors)
    if evaluation.get("pair_mode") != "order":
        errors.append("config.evaluation.pair_mode: expected order")
    if evaluation.get("pooling") != "mean":
        errors.append("config.evaluation.pooling: expected mean")
    if (
        type(evaluation.get("batch_size")) is not int
        or evaluation.get("batch_size", 0) <= 0
    ):
        errors.append("config.evaluation.batch_size: expected a positive integer")
    dataset = _mapping(config.get("dataset"), "config.dataset", errors)
    if dataset.get("split") != "deterministic-shuffle-v1":
        errors.append("config.dataset.split: expected deterministic-shuffle-v1")
    for field in ("min_length", "max_length", "max_records"):
        if type(dataset.get(field)) is not int or dataset.get(field, 0) <= 0:
            errors.append(f"config.dataset.{field}: expected a positive integer")
    if (
        type(dataset.get("min_length")) is int
        and type(dataset.get("max_length")) is int
        and dataset["min_length"] > dataset["max_length"]
    ):
        errors.append("config.dataset.min_length: cannot exceed max_length")
    split_ratios: list[float] = []
    for field in ("validation_ratio", "test_ratio"):
        value = _finite_number(dataset.get(field), f"config.dataset.{field}", errors)
        if value is not None:
            split_ratios.append(value)
            if not 0 < value < 1:
                errors.append(
                    f"config.dataset.{field}: expected a value between zero and one"
                )
    if len(split_ratios) == 2 and sum(split_ratios) >= 1:
        errors.append("config.dataset: validation and test ratios must sum below one")
    if (
        type(dataset.get("max_records")) is int
        and isinstance(dataset.get("test_ratio"), (int, float))
        and not isinstance(dataset.get("test_ratio"), bool)
        and type(generation.get("sequence_count")) is int
    ):
        expected_count = max(1, int(dataset["max_records"] * dataset["test_ratio"]))
        if generation["sequence_count"] != expected_count:
            errors.append(
                "config.generation.sequence_count: differs from the frozen test split"
            )
    training = _mapping(config.get("training"), "config.training", errors)
    if type(training.get("freeze_proposal")) is not bool:
        errors.append("config.training.freeze_proposal: expected a boolean")
    for field in (
        "epochs",
        "min_epochs",
        "batch_size",
        "num_candidates",
        "validation_steps",
        "scheduler_patience",
        "early_stop_patience",
        "remote_call_budget_per_seed",
    ):
        if type(training.get(field)) is not int or training.get(field, 0) <= 0:
            errors.append(f"config.training.{field}: expected a positive integer")
    for field in (
        "learning_rate",
        "weight_decay",
        "grad_clip_norm",
        "scheduler_factor",
    ):
        value = _finite_number(training.get(field), f"config.training.{field}", errors)
        if value is not None and value <= 0:
            errors.append(f"config.training.{field}: expected a positive value")
    if (
        type(training.get("epochs")) is int
        and type(training.get("min_epochs")) is int
        and training["min_epochs"] > training["epochs"]
    ):
        errors.append("config.training.min_epochs: cannot exceed epochs")
    estimated_calls = _estimate_protein_remote_calls(config)
    protein_budget = training.get("remote_call_budget_per_seed")
    if (
        estimated_calls is not None
        and type(protein_budget) is int
        and protein_budget < estimated_calls
    ):
        errors.append(
            "config.training.remote_call_budget_per_seed: below the "
            f"worst-case workflow estimate of {estimated_calls}"
        )
    software = _mapping(config.get("software"), "config.software", errors)
    for field in (
        "source_revision",
        "kaiwu_pytorch_plugin_revision",
        "python_version",
        "torch_version",
        "kaiwu_sdk_version",
    ):
        if software.get(field) in {None, "", "<required>"}:
            errors.append(f"config.software.{field}: frozen value is required")
    for field in ("source_revision", "kaiwu_pytorch_plugin_revision"):
        revision = software.get(field)
        if not isinstance(revision, str) or FULL_REVISION.fullmatch(revision) is None:
            errors.append(f"config.software.{field}: expected a full Git revision")
    precision = _mapping(
        config.get("precision_policy"), "config.precision_policy", errors
    )
    for field in ("name", "target_min", "target_max"):
        if field not in precision:
            errors.append(f"config.precision_policy: missing {field}")
    if config.get("primary_metric") != {
        "name": "mean_cosine_distance",
        "direction": "lower",
    }:
        errors.append(
            "config.primary_metric: expected preregistered lower mean_cosine_distance"
        )
    if (
        type(config.get("remote_call_budget")) is not int
        or config.get("remote_call_budget", 0) <= 0
    ):
        errors.append("config.remote_call_budget: expected a positive integer")
    portability_calls = _estimate_portability_remote_calls(config)
    if (
        portability_calls is not None
        and type(config.get("remote_call_budget")) is int
        and config["remote_call_budget"] < portability_calls
    ):
        errors.append(
            "config.remote_call_budget: below the portability replay estimate of "
            f"{portability_calls}"
        )
    thresholds = _mapping(config.get("thresholds"), "config.thresholds", errors)
    if thresholds.get("uniqueness_baseline_fraction_min") != 0.95:
        errors.append("config.thresholds: uniqueness floor must remain 0.95")
    if thresholds.get("repeat_ratio_absolute_increase_max") != 0.05:
        errors.append("config.thresholds: repeat-ratio allowance must remain 0.05")
    if thresholds.get("invalid_sequence_count_max") != 0:
        errors.append("config.thresholds: invalid sequence count must remain zero")


def _validate_system_record(
    record: dict[str, Any],
    *,
    config: dict[str, Any],
    config_sha256: str,
    label: str,
    errors: list[str],
) -> None:
    if record.get("schema") != RECORD_SCHEMA or record.get("version") != "1.0":
        errors.append(f"{label}: unsupported schema or version")
    for field in ("source_revision", "kaiwu_pytorch_plugin_revision"):
        value = record.get(field)
        if not isinstance(value, str) or FULL_REVISION.fullmatch(value) is None:
            errors.append(f"{label}.{field}: expected a full Git revision")
    for field in ("source_preflight_sha256", "transfer_manifest_sha256"):
        value = record.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            errors.append(f"{label}.{field}: expected a SHA-256 digest")
    software = _mapping(config.get("software"), "config.software", errors)
    for field in (
        "source_revision",
        "kaiwu_pytorch_plugin_revision",
        "python_version",
        "torch_version",
        "kaiwu_sdk_version",
    ):
        if record.get(field) != software.get(field):
            errors.append(f"{label}.{field}: differs from the frozen software lane")
    if record.get("experiment_config_sha256") != config_sha256:
        errors.append(f"{label}: frozen experiment config identity mismatch")
    if record.get("execution_host") not in HOSTS:
        errors.append(f"{label}.execution_host: unexpected host")
    if record.get("run_role") not in {"primary", "portability_replay"}:
        errors.append(f"{label}.run_role: unexpected role")
    if record.get("requested_cuda_device") != "cuda:0":
        errors.append(f"{label}: acceptance requires explicit cuda:0")
    if record.get("observed_tensor_device") != "cuda:0":
        errors.append(f"{label}: tensor work was not observed on cuda:0")
    if "A800" not in str(record.get("observed_gpu_model", "")):
        errors.append(f"{label}: observed GPU is not an NVIDIA A800")
    if record.get("transport") != "kaiwu_cim":
        errors.append(f"{label}: transport is not kaiwu_cim")
    if record.get("qboson_hardware_used") is not True:
        errors.append(f"{label}: QBoson hardware use is not proven")
    if record.get("real_provider_evidence") is not True:
        errors.append(f"{label}: real provider evidence is absent")
    if record.get("provider_reported_target") is not True:
        errors.append(f"{label}: target identity is not provider-reported")
    if (
        not isinstance(record.get("qboson_target"), str)
        or not record["qboson_target"].strip()
    ):
        errors.append(f"{label}: provider target is missing")
    task_ids = record.get("qboson_task_ids")
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or any(
            not isinstance(task_id, str) or not task_id.strip() for task_id in task_ids
        )
    ):
        errors.append(f"{label}: provider task identities are missing")
    if record.get("sampling_mode") != "sampling":
        errors.append(f"{label}: expected sampling mode")
    requested = _finite_number(
        record.get("requested_samples"), f"{label}.requested_samples", errors
    )
    returned = _finite_number(
        record.get("returned_samples"), f"{label}.returned_samples", errors
    )
    if requested is not None and returned is not None and requested != returned:
        errors.append(f"{label}: requested and returned sample counts differ")
    call_count = record.get("remote_call_count")
    call_budget = record.get("remote_call_budget")
    if (
        type(call_count) is not int
        or type(call_budget) is not int
        or call_count <= 0
        or call_count > call_budget
    ):
        errors.append(f"{label}: remote-call accounting violates its budget")
    if record.get("fallback_occurred") is not False:
        errors.append(f"{label}: fallback must be explicitly false")
    if record.get("retrieval_resubmitted") is not False:
        errors.append(f"{label}: retrieval must not resubmit")
    if record.get("secrets_redacted") is not True:
        errors.append(f"{label}: secret redaction is not proven")
    artifacts = _mapping(record.get("artifacts"), f"{label}.artifacts", errors)
    artifact_config_fields = {
        "dataset_sha256": "dataset",
        "base_checkpoint_sha256": "checkpoint",
        "tokenizer_sha256": "tokenizer",
        "evaluation_model_sha256": "evaluation_model",
    }
    for record_field, config_section in artifact_config_fields.items():
        expected_digest = _mapping(
            config.get(config_section), f"config.{config_section}", errors
        ).get("sha256")
        if artifacts.get(record_field) != expected_digest:
            errors.append(f"{label}.artifacts.{record_field}: differs from config")
    trained_digest = artifacts.get("trained_energy_checkpoint_sha256")
    if (
        not isinstance(trained_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", trained_digest) is None
    ):
        errors.append(
            f"{label}.artifacts.trained_energy_checkpoint_sha256: expected a SHA-256 digest"
        )
    precision = _mapping(
        record.get("precision_policy"), f"{label}.precision_policy", errors
    )
    expected_precision = _mapping(
        config.get("precision_policy"), "config.precision_policy", errors
    )
    for field in ("name", "target_min", "target_max"):
        if precision.get(field) != expected_precision.get(field):
            errors.append(f"{label}.precision_policy.{field}: differs from config")
    matrix_count = precision.get("matrix_count")
    if type(matrix_count) is not int or matrix_count <= 0:
        errors.append(
            f"{label}.precision_policy.matrix_count: expected a positive integer"
        )
    elif type(call_count) is int and matrix_count < call_count:
        errors.append(f"{label}.precision_policy: fewer reports than remote calls")
    scale_min = _finite_number(
        precision.get("scale_factor_min"),
        f"{label}.precision_policy.scale_factor_min",
        errors,
    )
    scale_max = _finite_number(
        precision.get("scale_factor_max"),
        f"{label}.precision_policy.scale_factor_max",
        errors,
    )
    max_error = _finite_number(
        precision.get("max_abs_error"),
        f"{label}.precision_policy.max_abs_error",
        errors,
    )
    mean_error = _finite_number(
        precision.get("mean_of_matrix_mean_abs_error"),
        f"{label}.precision_policy.mean_of_matrix_mean_abs_error",
        errors,
    )
    if scale_min is not None and scale_min <= 0:
        errors.append(f"{label}.precision_policy.scale_factor_min: expected positive")
    if scale_max is not None and scale_max <= 0:
        errors.append(f"{label}.precision_policy.scale_factor_max: expected positive")
    if scale_min is not None and scale_max is not None and scale_max < scale_min:
        errors.append(f"{label}.precision_policy: invalid scale-factor range")
    if max_error is not None and max_error < 0:
        errors.append(f"{label}.precision_policy.max_abs_error: expected non-negative")
    if mean_error is not None and mean_error < 0:
        errors.append(
            f"{label}.precision_policy.mean_of_matrix_mean_abs_error: expected non-negative"
        )
    if max_error is not None and mean_error is not None and mean_error > max_error:
        errors.append(f"{label}.precision_policy: mean error exceeds maximum error")
    training = _mapping(record.get("training"), f"{label}.training", errors)
    for field in ("energy_objective", "gradient_norm", "parameter_delta_max"):
        value = _finite_number(training.get(field), f"{label}.training.{field}", errors)
        if field != "energy_objective" and value is not None and value <= 0:
            errors.append(f"{label}.training.{field}: expected a positive value")
    generation = _mapping(record.get("generation"), f"{label}.generation", errors)
    if generation.get("token_constraints_passed") is not True:
        errors.append(f"{label}: token constraints did not pass")
    if generation.get("invalid_sequence_count") != 0:
        errors.append(f"{label}: invalid generated sequences were observed")
    transfers = _mapping(
        record.get("transfer_accounting"), f"{label}.transfer_accounting", errors
    )
    requested_device = record.get("requested_cuda_device")
    if transfers.get("matrix_origin_device") != requested_device:
        errors.append(f"{label}: matrix origin was not the requested CUDA device")
    if transfers.get("returned_sample_target_device") != requested_device:
        errors.append(f"{label}: returned samples did not target the CUDA device")
    boundaries = transfers.get("sampler_boundaries")
    if not isinstance(boundaries, list) or not boundaries:
        errors.append(f"{label}: sampler transfer accounting is missing")
    else:
        non_cached = 0
        for index, raw_boundary in enumerate(boundaries):
            boundary = _mapping(
                raw_boundary,
                f"{label}.transfer_accounting.sampler_boundaries[{index}]",
                errors,
            )
            if boundary.get("cache_hit") is False:
                non_cached += 1
            expected_fields = {
                "input_type": "numpy.ndarray",
                "input_device": "cpu",
                "canonical_device": "cpu",
                "canonical_dtype": "torch.float64",
                "submission_storage": "cpu_python_tuple",
                "returned_storage": "cpu_numpy",
                "returned_dtype": "int8",
            }
            for field, expected in expected_fields.items():
                if boundary.get(field) != expected:
                    errors.append(
                        f"{label}.transfer_accounting.sampler_boundaries[{index}]."
                        f"{field}: expected {expected}"
                    )
            matrix_shape = boundary.get("matrix_shape")
            returned_shape = boundary.get("returned_shape")
            if (
                not isinstance(matrix_shape, list)
                or len(matrix_shape) != 2
                or matrix_shape[0] != matrix_shape[1]
                or any(type(size) is not int or size <= 0 for size in matrix_shape)
            ):
                errors.append(f"{label}: invalid transferred matrix shape")
            if (
                not isinstance(returned_shape, list)
                or len(returned_shape) != 2
                or any(type(size) is not int or size <= 0 for size in returned_shape)
                or (
                    isinstance(matrix_shape, list)
                    and len(matrix_shape) == 2
                    and returned_shape[1] != matrix_shape[0]
                )
            ):
                errors.append(f"{label}: invalid returned sample shape")
        if type(call_count) is int and non_cached != call_count:
            errors.append(
                f"{label}: transfer accounting differs from remote-call count"
            )
    acceptance = _mapping(record.get("acceptance"), f"{label}.acceptance", errors)
    if acceptance.get("system") != "pass":
        errors.append(f"{label}: system acceptance did not pass")


def _metric(
    metrics: dict[str, Any], name: str, label: str, errors: list[str]
) -> float | None:
    return _finite_number(metrics.get(name), f"{label}.{name}", errors)


def _validate_application(
    primary: dict[str, Any], config: dict[str, Any], errors: list[str]
) -> None:
    baseline = _mapping(
        primary.get("baseline_metrics"), "primary.baseline_metrics", errors
    )
    guided = _mapping(primary.get("guided_metrics"), "primary.guided_metrics", errors)
    baseline_values = {
        name: _metric(baseline, name, "primary.baseline_metrics", errors)
        for name in METRIC_NAMES
    }
    guided_values = {
        name: _metric(guided, name, "primary.guided_metrics", errors)
        for name in METRIC_NAMES
    }
    baseline_primary = baseline_values["mean_cosine_distance"]
    guided_primary = guided_values["mean_cosine_distance"]
    if (
        baseline_primary is not None
        and guided_primary is not None
        and guided_primary >= baseline_primary
    ):
        errors.append("primary: guided mean cosine distance did not improve")
    baseline_unique = baseline_values["uniqueness_ratio"]
    guided_unique = guided_values["uniqueness_ratio"]
    thresholds = _mapping(config.get("thresholds"), "config.thresholds", errors)
    uniqueness_floor = _finite_number(
        thresholds.get("uniqueness_baseline_fraction_min"),
        "config.thresholds.uniqueness_baseline_fraction_min",
        errors,
    )
    if (
        baseline_unique is not None
        and guided_unique is not None
        and uniqueness_floor is not None
        and guided_unique < uniqueness_floor * baseline_unique
    ):
        errors.append("primary: guided uniqueness is below its preregistered floor")
    baseline_repeat = baseline_values["repeat_ratio_ge4"]
    guided_repeat = guided_values["repeat_ratio_ge4"]
    repeat_allowance = _finite_number(
        thresholds.get("repeat_ratio_absolute_increase_max"),
        "config.thresholds.repeat_ratio_absolute_increase_max",
        errors,
    )
    if (
        baseline_repeat is not None
        and guided_repeat is not None
        and repeat_allowance is not None
        and guided_repeat > baseline_repeat + repeat_allowance
    ):
        errors.append("primary: guided repeat ratio exceeds its preregistered limit")
    if guided_values["invalid_sequence_count"] != 0:
        errors.append("primary: guided generation contains invalid sequences")
    if primary.get("attempted_seeds") != config.get("seeds"):
        errors.append("primary: attempted seeds differ from the frozen seed set")
    application_evidence = _mapping(
        primary.get("application_evidence"), "primary.application_evidence", errors
    )
    if application_evidence.get("aggregation") != "arithmetic_mean_across_frozen_seeds":
        errors.append("primary: unexpected seed aggregation policy")
    evidence_records = application_evidence.get("records")
    if not isinstance(evidence_records, list) or len(evidence_records) != len(
        config.get("seeds", [])
    ):
        errors.append("primary: application evidence does not cover every seed")
    else:
        evidence_seeds = [record.get("seed") for record in evidence_records]
        if evidence_seeds != config.get("seeds"):
            errors.append(
                "primary: application evidence seed order differs from config"
            )
        for index, record in enumerate(evidence_records):
            if not isinstance(record, dict):
                errors.append(
                    f"primary.application_evidence.records[{index}]: expected object"
                )
                continue
            for field in (
                "training_record_sha256",
                "evaluation_record_sha256",
                "trained_energy_checkpoint_sha256",
            ):
                digest = record.get(field)
                if (
                    not isinstance(digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                ):
                    errors.append(
                        f"primary.application_evidence.records[{index}].{field}: "
                        "expected a SHA-256 digest"
                    )
    acceptance = _mapping(primary.get("acceptance"), "primary.acceptance", errors)
    if acceptance.get("application") != "pass":
        errors.append("primary: application acceptance did not pass")


def _validate_component_bundle(
    component_payloads: dict[str, dict[str, Any]],
    *,
    config: dict[str, Any],
    config_sha256: str,
    primary: dict[str, Any],
    replay: dict[str, Any],
    errors: list[str],
) -> None:
    """Independently prove that final records are derived from their components."""

    seeds = config.get("seeds", [])
    expected_schema_counts = {
        SOURCE_PREFLIGHT_COMPONENT_SCHEMA: 2,
        SYSTEM_COMPONENT_SCHEMA: 2,
        PORTABILITY_COMPONENT_SCHEMA: 1,
        TRAINING_COMPONENT_SCHEMA: len(seeds),
        EVALUATION_COMPONENT_SCHEMA: len(seeds),
    }
    observed_schema_counts = {
        schema: sum(
            payload.get("schema") == schema for payload in component_payloads.values()
        )
        for schema in expected_schema_counts
    }
    if observed_schema_counts != expected_schema_counts:
        errors.append("manifest: component record schema counts are incomplete")
        return

    for digest, payload in component_payloads.items():
        if payload.get("version") != "1.0":
            errors.append(f"component {digest}: unsupported version")
        if (
            payload.get("schema") != SOURCE_PREFLIGHT_COMPONENT_SCHEMA
            and payload.get("experiment_config_sha256") != config_sha256
        ):
            errors.append(
                f"component {digest}: frozen experiment config identity mismatch"
            )

    software = _mapping(config.get("software"), "config.software", errors)
    source_preflights: dict[str, tuple[str, dict[str, Any]]] = {}
    for digest, payload in component_payloads.items():
        if payload.get("schema") != SOURCE_PREFLIGHT_COMPONENT_SCHEMA:
            continue
        host = payload.get("verified_for_target_host")
        if not isinstance(host, str) or host in source_preflights:
            errors.append("manifest: source preflights contain an invalid host")
            continue
        try:
            validate_source_preflight_record(
                payload,
                execution_host=host,
                source_revision=software.get("source_revision", ""),
                plugin_revision=software.get("kaiwu_pytorch_plugin_revision", ""),
            )
        except ValueError as exc:
            errors.append(f"source preflight {host}: {exc}")
        source_preflights[host] = (digest, payload)
    if set(source_preflights) != HOSTS:
        errors.append("manifest: source preflights must cover both validation hosts")

    for digest, payload in component_payloads.items():
        if payload.get("schema") == SOURCE_PREFLIGHT_COMPONENT_SCHEMA:
            continue
        host = payload.get("execution_host")
        source_entry = source_preflights.get(host) if isinstance(host, str) else None
        if source_entry is None:
            errors.append(f"component {digest}: source preflight host is unavailable")
            continue
        source_digest, source_record = source_entry
        if payload.get("source_preflight_sha256") != source_digest:
            errors.append(f"component {digest}: source preflight identity mismatch")
        if payload.get("transfer_manifest_sha256") != source_record.get(
            "manifest_sha256"
        ):
            errors.append(f"component {digest}: transfer manifest identity mismatch")

    for final, label in ((primary, "primary"), (replay, "replay")):
        system_digest = final.get("system_evidence_sha256")
        component = (
            component_payloads.get(system_digest)
            if isinstance(system_digest, str)
            else None
        )
        if component is None or component.get("schema") != SYSTEM_COMPONENT_SCHEMA:
            errors.append(
                f"{label}: system evidence does not identify a system component"
            )
            continue
        copied_fields = (
            "source_revision",
            "kaiwu_pytorch_plugin_revision",
            "source_preflight_sha256",
            "transfer_manifest_sha256",
            "python_version",
            "torch_version",
            "kaiwu_sdk_version",
            "execution_host",
            "run_role",
            "requested_cuda_device",
            "observed_tensor_device",
            "observed_gpu_model",
            "transport",
            "qboson_hardware_used",
            "real_provider_evidence",
            "provider_reported_target",
            "qboson_target",
            "qboson_task_ids",
            "sampling_mode",
            "requested_samples",
            "returned_samples",
            "remote_call_budget",
            "remote_call_count",
            "fallback_occurred",
            "retrieval_resubmitted",
            "secrets_redacted",
            "precision_policy",
            "transfer_accounting",
        )
        for field in copied_fields:
            if component.get(field) != final.get(field):
                errors.append(
                    f"{label}: system component {field} differs from final record"
                )
        component_training = _mapping(
            component.get("training"), f"{label}.system_component.training", errors
        )
        final_training = _mapping(final.get("training"), f"{label}.training", errors)
        training_fields = {
            "objective": "energy_objective",
            "gradient_norm": "gradient_norm",
            "parameter_delta_max": "parameter_delta_max",
        }
        for component_field, final_field in training_fields.items():
            if component_training.get(component_field) != final_training.get(
                final_field
            ):
                errors.append(
                    f"{label}: system component training.{component_field} "
                    "differs from final record"
                )
        component_generation = _mapping(
            component.get("generation"),
            f"{label}.system_component.generation",
            errors,
        )
        final_generation = _mapping(
            final.get("generation"), f"{label}.generation", errors
        )
        if component_generation.get("token_constraints_passed") != final_generation.get(
            "token_constraints_passed"
        ):
            errors.append(
                f"{label}: system component generation result differs from final record"
            )
        if component.get("acceptance") != {"system": "pass", "application": "not_run"}:
            errors.append(f"{label}: system component did not pass its isolated gate")

    application = _mapping(
        primary.get("application_evidence"), "primary.application_evidence", errors
    )
    evidence_records = application.get("records")
    if not isinstance(evidence_records, list):
        return
    training_by_seed: dict[int, tuple[str, dict[str, Any]]] = {}
    evaluation_by_seed: dict[int, tuple[str, dict[str, Any]]] = {}
    for digest, payload in component_payloads.items():
        seed = payload.get("seed")
        if payload.get("schema") == TRAINING_COMPONENT_SCHEMA:
            if type(seed) is not int or seed in training_by_seed:
                errors.append(
                    "manifest: training components contain an invalid or duplicate seed"
                )
            else:
                training_by_seed[seed] = (digest, payload)
        elif payload.get("schema") == EVALUATION_COMPONENT_SCHEMA:
            if type(seed) is not int or seed in evaluation_by_seed:
                errors.append(
                    "manifest: evaluation components contain an invalid or duplicate seed"
                )
            else:
                evaluation_by_seed[seed] = (digest, payload)
    if list(training_by_seed) != seeds or list(evaluation_by_seed) != seeds:
        errors.append("manifest: component seed order differs from the frozen seed set")

    for evidence in evidence_records:
        if not isinstance(evidence, dict) or type(evidence.get("seed")) is not int:
            continue
        seed = evidence["seed"]
        training_entry = training_by_seed.get(seed)
        evaluation_entry = evaluation_by_seed.get(seed)
        if training_entry is None or evaluation_entry is None:
            continue
        training_digest, training = training_entry
        evaluation_digest, evaluation = evaluation_entry
        if evidence.get("training_record_sha256") != training_digest:
            errors.append(
                f"seed {seed}: final record references another training component"
            )
        if evidence.get("evaluation_record_sha256") != evaluation_digest:
            errors.append(
                f"seed {seed}: final record references another evaluation component"
            )
        if training.get("run_completed") is not True:
            errors.append(f"seed {seed}: training component did not complete")
        if training.get("execution_host") != config.get("primary_host"):
            errors.append(f"seed {seed}: training component is from the wrong host")
        checkpoint = training.get("trained_energy_checkpoint_sha256")
        if evidence.get("trained_energy_checkpoint_sha256") != checkpoint:
            errors.append(
                f"seed {seed}: final checkpoint differs from training component"
            )
        if evaluation.get("training_record_sha256") != training_digest:
            errors.append(
                f"seed {seed}: evaluation is linked to another training component"
            )
        if evaluation.get("execution_host") != config.get("primary_host"):
            errors.append(f"seed {seed}: evaluation component is from the wrong host")

    for field in ("baseline_metrics", "guided_metrics"):
        final_metrics = _mapping(primary.get(field), f"primary.{field}", errors)
        for metric in METRIC_NAMES:
            values: list[float] = []
            for seed in seeds:
                entry = evaluation_by_seed.get(seed)
                if entry is None:
                    continue
                metrics = _mapping(entry[1].get(field), f"seed {seed}.{field}", errors)
                value = _finite_number(
                    metrics.get(metric), f"seed {seed}.{field}.{metric}", errors
                )
                if value is not None:
                    values.append(value)
            if len(values) != len(seeds):
                continue
            expected = (
                sum(values) if metric == "invalid_sequence_count" else fmean(values)
            )
            if final_metrics.get(metric) != expected:
                errors.append(
                    f"primary.{field}.{metric}: differs from component aggregation"
                )

    portability_evidence = _mapping(
        replay.get("portability_evidence"), "replay.portability_evidence", errors
    )
    portability_digest = portability_evidence.get("record_sha256")
    portability = (
        component_payloads.get(portability_digest)
        if isinstance(portability_digest, str)
        else None
    )
    if portability is None or portability.get("schema") != PORTABILITY_COMPONENT_SCHEMA:
        errors.append(
            "replay: portability evidence does not identify a portability component"
        )
        return
    if portability.get("execution_host") != config.get("replay_host"):
        errors.append("replay: portability component is from the wrong host")
    if portability.get("acceptance") != {"portability": "pass"}:
        errors.append("replay: portability component did not pass")
    for field in ("training_record_sha256", "trained_energy_checkpoint_sha256"):
        if portability.get(field) != portability_evidence.get(field):
            errors.append(
                f"replay: portability component {field} differs from final record"
            )


def validate_acceptance(manifest_path: Path) -> list[str]:
    """Return all validation errors; an empty list means the evidence passes."""

    errors: list[str] = []
    root = manifest_path.resolve().parent
    manifest = _mapping(_read_json(manifest_path), "manifest", errors)
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("version") != "1.0":
        errors.append("manifest: unsupported schema or version")

    def resolve_member(value: Any, label: str) -> Path | None:
        if not isinstance(value, str) or not value:
            errors.append(f"{label}: expected a relative path")
            return None
        member = (root / value).resolve()
        if Path(value).is_absolute() or not member.is_relative_to(root):
            errors.append(f"{label}: path escapes the evidence directory")
            return None
        if not member.is_file():
            errors.append(f"{label}: file does not exist")
            return None
        return member

    config_entry = _mapping(manifest.get("config"), "manifest.config", errors)
    config_path = resolve_member(config_entry.get("path"), "manifest.config.path")
    if config_path is None:
        return errors
    config_hash = _sha256(config_path)
    if config_entry.get("sha256") != config_hash:
        errors.append("manifest.config: SHA-256 mismatch")
    config = _mapping(_read_json(config_path), "config", errors)
    _validate_config(config, errors)

    component_payloads: dict[str, dict[str, Any]] = {}
    raw_components = manifest.get("component_records")
    if raw_components is not None:
        if not isinstance(raw_components, list):
            errors.append("manifest.component_records: expected a list")
        else:
            for index, raw_entry in enumerate(raw_components):
                entry = _mapping(
                    raw_entry, f"manifest.component_records[{index}]", errors
                )
                component_path = resolve_member(
                    entry.get("path"),
                    f"manifest.component_records[{index}].path",
                )
                if component_path is None:
                    continue
                digest = _sha256(component_path)
                if entry.get("sha256") != digest:
                    errors.append(
                        f"manifest.component_records[{index}]: SHA-256 mismatch"
                    )
                    continue
                component_payloads[digest] = _mapping(
                    _read_json(component_path),
                    f"component_record[{index}]",
                    errors,
                )

    entries = manifest.get("records")
    if not isinstance(entries, list) or len(entries) != 2:
        errors.append("manifest.records: expected exactly two host records")
        return errors
    records: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry, f"manifest.records[{index}]", errors)
        record_path = resolve_member(
            entry.get("path"), f"manifest.records[{index}].path"
        )
        if record_path is None:
            continue
        if entry.get("sha256") != _sha256(record_path):
            errors.append(f"manifest.records[{index}]: SHA-256 mismatch")
        record = _mapping(_read_json(record_path), f"record[{index}]", errors)
        _validate_system_record(
            record,
            config=config,
            config_sha256=config_hash,
            label=f"record[{index}]",
            errors=errors,
        )
        records.append(record)
    if len(records) != 2:
        return errors

    by_host = {str(record.get("execution_host")): record for record in records}
    if set(by_host) != HOSTS:
        errors.append("manifest: records must cover jp-a800-171 and jp-a800-172")
        return errors
    primary_host = config.get("primary_host")
    replay_host = config.get("replay_host")
    if primary_host not in by_host or replay_host not in by_host:
        errors.append("manifest: configured host roles do not match the records")
        return errors
    primary = by_host[primary_host]
    replay = by_host[replay_host]
    if primary.get("component_bundle_required") is True:
        expected_component_count = 5 + 2 * len(config.get("seeds", []))
        if len(component_payloads) != expected_component_count:
            errors.append(
                "manifest: component bundle does not contain every source record"
            )
    if primary.get("run_role") != "primary":
        errors.append("manifest: configured primary host lacks the primary role")
    if replay.get("run_role") != "portability_replay":
        errors.append("manifest: configured replay host lacks the portability role")
    if (
        _mapping(replay.get("acceptance"), "replay.acceptance", errors).get(
            "application"
        )
        != "not_run"
    ):
        errors.append("replay: application acceptance must be explicitly not_run")
    portability_evidence = _mapping(
        replay.get("portability_evidence"), "replay.portability_evidence", errors
    )
    if portability_evidence.get("acceptance") != "pass":
        errors.append("replay: portability evidence did not pass")
    expected_portability_seed = _mapping(
        config.get("generation"), "config.generation", errors
    ).get("portability_training_seed")
    if portability_evidence.get("training_seed") != expected_portability_seed:
        errors.append("replay: portability evidence uses another training seed")
    for field in ("record_sha256", "training_record_sha256"):
        portability_digest = portability_evidence.get(field)
        if (
            not isinstance(portability_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", portability_digest) is None
        ):
            errors.append(f"replay.portability_evidence.{field}: expected SHA-256")
    revisions = {record.get("source_revision") for record in records}
    plugin_revisions = {
        record.get("kaiwu_pytorch_plugin_revision") for record in records
    }
    if len(revisions) != 1 or len(plugin_revisions) != 1:
        errors.append("manifest: both hosts must use identical source revisions")
    trained_checkpoint_digests = {
        _mapping(record.get("artifacts"), "record.artifacts", errors).get(
            "trained_energy_checkpoint_sha256"
        )
        for record in records
    }
    if len(trained_checkpoint_digests) != 1:
        errors.append(
            "manifest: both hosts must use the same trained energy checkpoint"
        )
    shared_checkpoint_digest = next(iter(trained_checkpoint_digests), None)
    if (
        portability_evidence.get("trained_energy_checkpoint_sha256")
        != shared_checkpoint_digest
    ):
        errors.append("replay: portability checkpoint differs from host records")
    application_evidence = _mapping(
        primary.get("application_evidence"), "primary.application_evidence", errors
    )
    evidence_records = application_evidence.get("records")
    if isinstance(evidence_records, list):
        selected = [
            record
            for record in evidence_records
            if isinstance(record, dict)
            and record.get("seed") == expected_portability_seed
        ]
        if (
            len(selected) != 1
            or selected[0].get("trained_energy_checkpoint_sha256")
            != shared_checkpoint_digest
            or selected[0].get("training_record_sha256")
            != portability_evidence.get("training_record_sha256")
        ):
            errors.append(
                "manifest: portability evidence is not linked to the selected seed checkpoint"
            )
    if primary.get("component_bundle_required") is True:
        referenced_component_hashes = {
            primary.get("source_preflight_sha256"),
            replay.get("source_preflight_sha256"),
            primary.get("system_evidence_sha256"),
            replay.get("system_evidence_sha256"),
            portability_evidence.get("record_sha256"),
        }
        if isinstance(evidence_records, list):
            for record in evidence_records:
                if isinstance(record, dict):
                    referenced_component_hashes.add(
                        record.get("training_record_sha256")
                    )
                    referenced_component_hashes.add(
                        record.get("evaluation_record_sha256")
                    )
        if referenced_component_hashes != set(component_payloads):
            errors.append(
                "manifest: final records do not reference the exact component bundle"
            )
        _validate_component_bundle(
            component_payloads,
            config=config,
            config_sha256=config_hash,
            primary=primary,
            replay=replay,
            errors=errors,
        )
    task_sets = []
    for record in records:
        raw_task_ids = record.get("qboson_task_ids")
        task_sets.append(
            {task_id for task_id in raw_task_ids if isinstance(task_id, str)}
            if isinstance(raw_task_ids, list)
            else set()
        )
    if task_sets[0] & task_sets[1]:
        errors.append(
            "manifest: host runs must have independent QBoson task identities"
        )
    if any(
        record.get("remote_call_budget") != config.get("remote_call_budget")
        for record in records
    ):
        errors.append("manifest: host call budgets differ from the frozen config")
    _validate_application(primary, config, errors)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    arguments = parser.parse_args()
    errors = validate_acceptance(arguments.manifest)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("QBoson QDiffusion acceptance evidence passed")


if __name__ == "__main__":
    main()

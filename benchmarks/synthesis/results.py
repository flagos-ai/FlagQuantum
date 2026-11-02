"""Fail-closed AO-00 artifact validation and deterministic resource comparison."""

import json
from collections import Counter
from dataclasses import fields
from pathlib import Path

from .expressions import corpus
from .oracle import output_width


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("expected a JSON object with string keys")
    return value


def _integer(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("expected a nonnegative plain integer")
    return value


def _cost(value: object, data_bits: int) -> dict[str, object]:
    from .arithmetic_baseline import Costs

    resource = _object(value)
    if set(resource) != {field.name for field in fields(Costs)}:
        raise ValueError("unexpected resource fields")
    identity = resource["circuit_hash"]
    if (
        not isinstance(identity, str)
        or len(identity) != 64
        or any(c not in "0123456789abcdef" for c in identity)
    ):
        raise ValueError("invalid circuit content hash")
    n = _integer(resource["allocated_qubits"])
    used = _integer(resource["used_qubits"])
    workspace = _integer(resource["workspace_qubits"])
    operations = _integer(resource["operations"])
    histogram = _object(resource["operation_counts"])
    if sum(_integer(count) for count in histogram.values()) != operations:
        raise ValueError("operation histogram does not match total")
    if used > n or workspace > n - data_bits:
        raise ValueError("invalid used/working qubit count")
    if resource["planned_complex128_state_bytes"] != 16 << n:
        raise ValueError("state buffer size does not match complex128 width")
    if (
        _integer(resource["depth"]) > operations
        or _integer(resource["two_qubit_operations"]) > operations
    ):
        raise ValueError("invalid depth or two-qubit count")
    return resource


def read_result(path: Path) -> dict[str, object]:
    """Validate fixed domains and resource arithmetic before comparing artifacts."""
    from .arithmetic_baseline import SCHEMA, Baseline, Configuration, Row

    payload = _object(json.loads(path.read_text()))
    if set(payload) != set(Baseline.__annotations__):
        raise ValueError("unexpected baseline fields")
    if set(_object(payload["configuration"])) != set(Configuration.__annotations__):
        raise ValueError("unexpected configuration fields")
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported baseline schema")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != len(corpus()):
        raise ValueError("baseline must contain the complete fixed corpus")
    for row_value, case in zip(rows, corpus(), strict=True):
        row = _object(row_value)
        if set(row) != {field.name for field in fields(Row)}:
            raise ValueError("unexpected case fields")
        widths = row.get("widths")
        if not isinstance(widths, list) or any(type(w) is not int for w in widths):
            raise ValueError("widths must be plain integers")
        _integer(row.get("output_width"))
        if (
            row.get("case"),
            row.get("expression"),
            row.get("widths"),
            row.get("output_width"),
        ) != (
            case.identifier,
            case.expression.text,
            list(case.widths),
            output_width(case),
        ):
            raise ValueError("case order, formula, domain or output width changed")
        status = row.get("status")
        if not isinstance(status, str) or status not in {
            "compiled",
            "refused",
            "timeout",
        }:
            raise ValueError("unknown completion status")
        if status != "compiled":
            if (
                row.get("resources") is not None
                or row.get("verification") is not None
                or not row.get("diagnostic_codes")
            ):
                raise ValueError(
                    "incomplete cases cannot claim resources or verification"
                )
            continue
        resource = _cost(row.get("resources"), sum(case.widths) + output_width(case))
        n = _integer(resource["allocated_qubits"])
        verification = _object(row.get("verification"))
        if _integer(verification.get("basis_checks")) == 0:
            raise ValueError("compiled cases require independent basis checks")
        exhaustive = sum(case.widths) <= 6
        if (
            verification.get("basis_domain_exhaustive") is not exhaustive
            or verification.get("nonzero_targets") is not True
        ):
            raise ValueError("basis verification scope is overstated")
        value_count = (1 << sum(case.widths)) if exhaustive else 18
        target_count = (
            16
            if case.identifier == "energy-2x2"
            else len({0, 1, (1 << output_width(case)) - 1})
        )
        if verification["basis_checks"] != value_count * target_count:
            raise ValueError("basis verification count does not match the fixed recipe")
        coherent = verification.get("coherent_states")
        if coherent not in (0, 2) or type(coherent) is not int:
            raise ValueError("invalid coherent verification count")
        if coherent != (2 if exhaustive and n <= 20 else 0):
            raise ValueError("coherent verification does not match the fixed recipe")
        if coherent == 0 and (
            verification.get("maximum_amplitude_error") is not None
            or verification.get("workspace_nonzero_probability") is not None
        ):
            raise ValueError("static cases cannot claim quantum state verification")
        if coherent == 2:
            if n > 20 or sum(case.widths) > 6:
                raise ValueError("state verification exceeds the declared bound")
            for name in ("maximum_amplitude_error", "workspace_nonzero_probability"):
                value = verification.get(name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not 0 <= value <= 1e-12
                ):
                    raise ValueError("quantum verification exceeded its tolerance")
    expected_summary = dict(Counter(_object(row)["status"] for row in rows))
    if payload.get("summary") != expected_summary:
        raise ValueError("summary does not match case outcomes")
    targets = payload.get("sentinel_targets")
    if not isinstance(targets, list) or len(targets) != 9:
        raise ValueError("missing target/reference sentinel rows")
    keys = set()
    for value in targets:
        row = _object(value)
        key = (row.get("implementation"), row.get("profile"))
        if key in keys:
            raise ValueError("duplicate target sentinel")
        keys.add(key)
        _cost(row.get("resources"), 8)
    if keys != {
        (name, profile)
        for name in ("python", "qmod", "manual")
        for profile in ("logical", "exact_basis_all_to_all", "exact_basis_line")
    }:
        raise ValueError("unexpected target sentinel configuration")
    variants = payload.get("spelling_variants")
    if not isinstance(variants, list) or len(variants) != 2:
        raise ValueError("missing spelling variants")
    for value, name in zip(variants, ("helper", "exchanged"), strict=True):
        row = _object(value)
        if (row.get("implementation"), row.get("profile")) != (name, "logical"):
            raise ValueError("unexpected spelling variant")
        _cost(row.get("resources"), 8)
    return payload


def compare(before: Path, after: Path) -> tuple[str, ...]:
    """Compare semantic configuration and costs; elapsed times are observational."""
    old, new = read_result(before), read_result(after)
    if old["configuration"] != new["configuration"]:
        raise ValueError("incompatible benchmark configuration")
    changes: list[str] = []
    old_rows, new_rows = old["rows"], new["rows"]
    assert isinstance(old_rows, list) and isinstance(new_rows, list)
    for left, right in zip(old_rows, new_rows, strict=True):
        a, b = _object(left), _object(right)
        if (a["status"], a["resources"]) != (b["status"], b["resources"]):
            changes.append(str(a["case"]))
    if old["spelling_variants"] != new["spelling_variants"]:
        changes.append("spelling_variants")
    if old["sentinel_targets"] != new["sentinel_targets"]:
        changes.append("sentinel_targets")
    return tuple(changes)

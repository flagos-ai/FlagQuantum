"""Native-only source capture and consistency over existing numerical plans."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch

from ..core.ir.program.model import ProgramRecord
from ..core.ir.program.physical import RydbergContext, RydbergPresence
from ..core.ir.program.physical_codec import (
    native_program_from_dict,
    native_program_to_dict,
)
from ..core.ir.state_space import ProductStateSpace
from ..errors import CapabilityError, SerializationError
from ..simulation.lindblad import _plan_density_matrix_evolution
from ..simulation.lindblad_request import _error
from ..simulation.rydberg_model import native_model

if TYPE_CHECKING:
    from ._plan import LindbladPlan

NATIVE_VERSION = "1.7"


def native_request(
    record: ProgramRecord,
    times: Any,
    *,
    n_qubits: int | None,
    dimensions: Sequence[int] | None,
    basis_labels: object,
    collapse_operators: object,
    dtype: torch.dtype,
    device: torch.device | str,
    parameter_values: Any,
    allow_presence: bool = False,
) -> tuple[dict[str, Any], list[dict[str, float]] | None]:
    if str(device).split(":", 1)[0] == "flagos":
        from ..compute import resolve_platform_device

        device = resolve_platform_device(device)
    for field, value in (
        ("n_qubits", n_qubits),
        ("basis_labels", basis_labels),
        ("collapse_operators", collapse_operators),
    ):
        if value is not None:
            raise _error(
                "invalid_native_request", f"native programs own {field}", field
            )
    context = record.physical_context
    if not isinstance(context, RydbergContext):
        raise CapabilityError("native Lindblad plans require timed Rydberg sources")
    expected = (3,) * len(context.atoms)
    if dimensions is not None:
        try:
            supplied = ProductStateSpace(dimensions).dimensions
        except (TypeError, ValueError) as exc:
            raise _error("invalid_native_request", str(exc), "dimensions") from exc
        if supplied != expected:
            raise _error(
                "invalid_native_request",
                "dimensions must match the native product space",
                "dimensions",
            )
    if any(
        isinstance(op.physical_data, RydbergPresence)
        for op in record.functions[0].blocks[0].operations
    ):
        if not allow_presence:
            raise CapabilityError("presence requires a sampled conditional trajectory")
        if dtype != torch.complex128 or str(device).split(":", 1)[0] not in {
            "cpu",
            "flagos",
        }:
            raise CapabilityError("presence dynamics requires CPU complex128")
        from ..simulation.rydberg_presence import _presence_model

        numeric, bindings, _ = _presence_model(
            record, times, parameter_values=parameter_values, device=device
        )
        return numeric, bindings
    return native_model(
        record, times, dtype=dtype, device=device, parameter_values=parameter_values
    )


def build_native_plan(
    record: ProgramRecord,
    initial_state: Any,
    times: Any,
    *,
    n_qubits: int | None,
    dimensions: Sequence[int] | None,
    basis_labels: object,
    collapse_operators: object,
    parameter_values: Any,
    observables: Any,
    **settings: Any,
) -> LindbladPlan:
    from ._plan import LindbladPlan, _identity, build_lindblad_plan

    numeric, bindings = native_request(
        record,
        times,
        n_qubits=n_qubits,
        dimensions=dimensions,
        basis_labels=basis_labels,
        collapse_operators=collapse_operators,
        parameter_values=parameter_values,
        dtype=settings["dtype"],
        device=settings["device"],
    )
    numeric_plan = build_lindblad_plan(
        initial_state=initial_state, observables=observables, **numeric, **settings
    )
    payload = numeric_plan.to_dict()
    payload.update(
        version=NATIVE_VERSION,
        program=native_program_to_dict(record),
        bindings=bindings,
    )
    payload["identity"] = _identity(payload)
    # Validate serialized meaning independently of the retained live graph.
    validated = LindbladPlan.from_dict(payload)
    return replace(validated, _live_request=numeric_plan._live_request)


def _numeric_values(value: object) -> None:
    if isinstance(value, list):
        for item in value:
            _numeric_values(item)
    elif type(value) not in (int, float):
        raise SerializationError(
            "native numeric payloads require finite real JSON numbers"
        )
    else:
        assert isinstance(value, (int, float))
        if not math.isfinite(value):
            raise SerializationError(
                "native numeric payloads require finite real JSON numbers"
            )


def validate_native_plan(payload: Mapping[str, Any]) -> None:
    from ._plan import _decode_request, _identity

    sampled = payload.get("version") == "1.8"
    expected_fields = {
        "schema",
        "version",
        "request",
        "decision",
        "program",
        "bindings",
        "identity",
    }
    if sampled:
        expected_fields.update(("options", "environment_requirements"))
    if set(payload) != expected_fields:
        raise SerializationError("unknown native plan fields")
    if sampled:
        from ..runtime._rydberg_presence import validate_presence_options

        validate_presence_options(payload)
    try:
        if payload.get("identity") != _identity(payload):
            raise SerializationError(
                "Lindblad plan identity does not match its content"
            )
        record = native_program_from_dict(payload["program"])
        if not isinstance(record.physical_context, RydbergContext):
            raise CapabilityError("native Lindblad plans require timed Rydberg sources")
        if payload["program"].get("version") != ("1.4" if sampled else "1.0"):
            raise SerializationError("native source and plan versions disagree")
        request, decision = payload["request"], payload["decision"]
        if not isinstance(request, Mapping) or not isinstance(decision, Mapping):
            raise SerializationError("native plans require request and decision maps")
        if sampled:
            from ..runtime._rydberg_presence import check_presence_budget

            check_presence_budget(
                record,
                len(request.get("times", [])),
                bool(request.get("return_density_matrices")),
            )
        precision = decision.get("numerics", {}).get("precision")
        dtype = {"complex64": torch.complex64, "complex128": torch.complex128}.get(
            precision
        )
        if dtype is None:
            raise SerializationError("unsupported native plan precision")
        bindings = payload["bindings"]
        if bindings is not None:
            if not isinstance(bindings, list) or any(
                not isinstance(row, Mapping) for row in bindings
            ):
                raise SerializationError(
                    "native bindings must be interval maps or null"
                )
            for row in bindings:
                if any(not isinstance(key, str) for key in row):
                    raise SerializationError("native binding names must be strings")
                for value in row.values():
                    _numeric_values(value)
                    if isinstance(value, list):
                        raise SerializationError("native bindings require scalars")
        with torch.no_grad():
            numeric, canonical_bindings = native_request(
                record,
                request.get("times"),
                n_qubits=request.get("n_qubits"),
                dimensions=request.get("dimensions"),
                basis_labels=request.get("basis_labels"),
                collapse_operators=None,
                dtype=dtype,
                device=decision.get("numerics", {}).get("device", "cpu"),
                parameter_values=bindings,
                allow_presence=sampled,
            )
            if bindings != canonical_bindings:
                raise SerializationError(
                    "native bindings must use the requested precision"
                )
            _numeric_values(request.get("times"))
            for key in ("initial_density_matrix",):
                value = request.get(key)
                if not isinstance(value, Mapping):
                    raise SerializationError("invalid native tensor encoding")
                _numeric_values(value.get("values"))
            for item in request.get("hamiltonians", []):
                if not isinstance(item, Mapping):
                    raise SerializationError("invalid native Hamiltonian encoding")
                _numeric_values(item.get("values"))
            for collection in ("collapse_operators", "observables"):
                for item in request.get(collection, []):
                    _numeric_values(item["operator"]["values"])
                    if collection == "collapse_operators":
                        _numeric_values(item["rate"])
            decoded = _decode_request(request, decision, version="1.5")
            if sampled:
                from ..runtime._rydberg_presence import check_presence_request

                check_presence_request(record, decoded)
            summary = _plan_density_matrix_evolution(**decoded)
            if summary.to_dict() != dict(decision):
                raise SerializationError(
                    "Lindblad plan decision does not match its request"
                )
            if tuple(decoded.get("dimensions", ())) != tuple(numeric["dimensions"]):
                raise SerializationError("native dimensions disagree with source")
            captured = decoded["bound_hamiltonians"]
            epsilon = (
                16
                * torch.finfo(
                    torch.float32 if dtype == torch.complex64 else torch.float64
                ).eps
            )
            if len(captured) != len(numeric["bound_hamiltonians"]):
                raise SerializationError("native interval count disagrees with source")
            # Verify the bounded serialized source payload on host; the live
            # Hamiltonians and evolution remain on the requested device.
            for actual, expected in zip(
                captured, numeric["bound_hamiltonians"], strict=True
            ):
                if not torch.allclose(
                    actual.detach().cpu(),
                    expected.detach().cpu(),
                    atol=epsilon,
                    rtol=epsilon,
                ):
                    raise SerializationError(
                        "native Hamiltonians disagree with source and bindings"
                    )
            jumps = numeric["collapse_operators"]
            if len(decoded["collapse_operators"]) != len(jumps):
                raise SerializationError("native jumps disagree with source")
            for (actual, rate), (expected, source_rate) in zip(
                decoded["collapse_operators"], jumps, strict=True
            ):
                if not math.isclose(
                    rate, source_rate, abs_tol=epsilon, rel_tol=epsilon
                ) or not torch.allclose(
                    actual.detach().cpu(),
                    expected.detach().cpu(),
                    atol=epsilon,
                    rtol=epsilon,
                ):
                    raise SerializationError("native loss model disagrees with source")
    except SerializationError:
        raise
    except (
        TypeError,
        ValueError,
        KeyError,
        AttributeError,
        RuntimeError,
        CapabilityError,
        OverflowError,
        RecursionError,
    ) as exc:
        raise SerializationError(f"invalid native Lindblad plan: {exc}") from exc

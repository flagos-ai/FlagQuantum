"""Admission rules shared by execution-input contract consumers."""

from __future__ import annotations

from collections.abc import Mapping

from ..core.ir import CircuitIR
from ..core.operator_schema import get_operator_schema
from ..errors import CapabilityError, ValidationError
from .dynamic._conditions import instruction_is_dynamic


def validate_input_program(program: CircuitIR) -> None:
    """Admit only arbitrary-input-safe ordinary static unitary programs."""
    from .planner import carries_noise_channels

    if program.observables or carries_noise_channels(program):
        raise CapabilityError(
            "external initial_state requires an ordinary unitary program"
        )
    if any(request.kind != "probabilities" for request in program.measurements):
        raise CapabilityError(
            "external initial_state supports statevectors and probabilities only"
        )
    for instruction in program.instructions:
        schema = get_operator_schema(instruction.name)
        if (
            schema is None
            or not schema.unitary
            or instruction.matrix is not None
            or instruction_is_dynamic(instruction)
        ):
            raise CapabilityError(
                "external initial_state supports ordinary static unitary opcodes only"
            )


def validate_input_decision(
    program: CircuitIR, decision: Mapping[str, object], resolved: Mapping[str, object]
) -> None:
    """Reject even rehashed plans whose marker contradicts the admitted scope."""
    validate_input_program(program)
    if (
        decision["mode"] != "statevector"
        or decision["backend"] != "pytorch"
        or decision["device"] != "cpu"
        or decision["batch_size"] != 1
        or decision["world_size"] != 1
        or decision["target"] not in ("auto", "state")
        or decision["allow_approximate"] is not False
        or decision["allow_backend_fallback"] is not False
        or resolved["shots"] is not None
        or resolved["mode"] not in ("auto", "statevector")
        or resolved["backend"] not in ("auto", "pytorch")
        or resolved["device"] not in ("auto", "cpu")
        or program.metadata.get("batch_size", 1) != 1
    ):
        raise CapabilityError(
            "external initial_state requires exact unbatched CPU PyTorch statevector execution"
        )
    for name in (
        "batch_size",
        "precision",
        "target",
        "require_gradients",
        "allow_approximate",
        "allow_backend_fallback",
        "memory_limit_bytes",
    ):
        if decision[name] != resolved[name]:
            raise ValidationError(
                f"initial_state decision disagrees with resolved {name}"
            )
    expected_bytes = (2**program.n_wires) * (
        16 if decision["precision"] == "complex128" else 8
    )
    if decision["state_bytes"] != expected_bytes:
        raise ValidationError(
            "initial_state plan must account for the complete register"
        )


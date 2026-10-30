"""Bounded Rydberg model assembly over Core physical source semantics."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from itertools import combinations
from numbers import Complex, Real
from typing import Any, Literal

import torch

from ..core.geometry import same_time
from ..core.ir import IRValidationError
from ..core.ir.program.model import ProgramRecord
from ..core.ir.program.physical import (
    RydbergContext,
    RydbergPulse,
    RydbergTransport,
    _boundary_times,
    _collision,
    _exact_position,
    is_physical_record,
)
from ..core.ir.program.verifier import verify_program_record
from ..core.ising import IsingBraidProgram
from ..core.parameters import (
    bind_parameter_value,
    parameter_names_in_value,
    value_to_tensor,
)
from ..core.rydberg import RydbergProgram
from ..errors import CapabilityError
from .lindblad_request import (
    EvolutionValidationError,
    _error,
    _normalize_parameter_mapping,
    _normalize_times,
)


def native_record(value: object) -> ProgramRecord | None:
    """Recognize native inputs without widening low-level mathematical entries."""
    if isinstance(value, RydbergProgram):
        return value.to_ir()
    if isinstance(value, IsingBraidProgram):
        value = value.to_ir()
    if not isinstance(value, ProgramRecord) or not is_physical_record(value):
        return None
    diagnostics = verify_program_record(value)
    if diagnostics:
        message = "; ".join(f"{d.code}: {d.message}" for d in diagnostics)
        if any(d.code == "physical.unsupported_structure" for d in diagnostics):
            raise CapabilityError(message)
        raise IRValidationError(message)
    if not isinstance(value.physical_context, RydbergContext):
        raise CapabilityError(
            "Lindblad requires timed Rydberg controls; discrete Ising braids "
            "do not define a Hamiltonian"
        )
    return value


def _invalid(message: str, field: str) -> EvolutionValidationError:
    return _error("invalid_native_request", message, field)


def _grid(
    record: ProgramRecord,
    times: Any,
    *,
    real_dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[float, float]]:
    try:
        grid = _normalize_times(
            times,
            device=torch.device("cpu"),
            real_dtype=real_dtype,
            allow_gradients=False,
        )
    except EvolutionValidationError as exc:
        if exc.code == "unsupported_time_gradient":
            raise
        raise _invalid(str(exc), "times") from exc
    if grid.numel() < 2:
        raise _invalid("native evolution requires at least two output times", "times")
    precision: Literal["float32", "float64"] = (
        "float32" if real_dtype == torch.float32 else "float64"
    )
    controls = tuple(
        op.physical_data
        for op in record.functions[0].blocks[0].operations
        if isinstance(op.physical_data, (RydbergTransport, RydbergPulse))
    )
    source = _boundary_times(controls)
    values = grid.tolist()
    if not same_time(values[0], 0.0, precision=precision):
        raise _invalid("native output times must start at zero", "times")
    matches: dict[float, float] = {}
    used: set[int] = set()
    for boundary in sorted(set(source.values())):
        normalized = float(torch.tensor(boundary, dtype=real_dtype).item())
        indices = [
            i
            for i, t in enumerate(values)
            if same_time(t, normalized, precision=precision)
        ]
        if len(indices) != 1 or indices[0] in used:
            raise _invalid(
                "each distinct physical boundary must match one distinct output time",
                "times",
            )
        used.add(indices[0])
        matches[boundary] = values[indices[0]]
    if values[-1] < max(matches.values()):
        raise _invalid("output grid must cover all controls", "times")
    return grid, {key: matches[value] for key, value in source.items()}


def _embed(operator: torch.Tensor, actor: int, count: int) -> torch.Tensor:
    result = torch.ones((1, 1), dtype=operator.dtype, device=operator.device)
    identity = torch.eye(3, dtype=operator.dtype, device=operator.device)
    for index in range(count):
        result = torch.kron(result, operator if index == actor else identity)
    return result


def native_model(
    record: ProgramRecord,
    times: Any,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    parameter_values: Any = None,
) -> tuple[dict[str, Any], list[dict[str, float]] | None]:
    """Assemble interval matrices; existing normalizers and engines execute them."""
    supported = {"rydberg.move": RydbergTransport, "rydberg.pulse": RydbergPulse}
    for operation in record.functions[0].blocks[0].operations:
        payload_type = supported.get(operation.opcode)
        if payload_type is None or not isinstance(
            operation.physical_data, payload_type
        ):
            raise CapabilityError(
                "unconditional native Rydberg evolution requires only pulse "
                "and transport controls; observation operations require a "
                "conditional trajectory"
            )
    context = record.physical_context
    assert isinstance(context, RydbergContext)
    count = len(context.atoms)
    if count > 2:
        raise CapabilityError(
            "native Rydberg dynamics currently supports one or two actors"
        )
    if str(device).split(":", 1)[0] not in {"cpu", "flagos"}:
        raise _error(
            "unsupported_device",
            "continuous-time evolution currently supports CPU only",
            "device",
        )
    if dtype not in (torch.complex64, torch.complex128):
        raise _error(
            "unsupported_dtype",
            "dtype must be torch.complex64 or torch.complex128",
            "dtype",
        )
    from ..compute import resolve_platform_device

    device = resolve_platform_device(device)
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    grid, boundaries = _grid(record, times, real_dtype=real_dtype)
    controls = tuple(
        op.physical_data for op in record.functions[0].blocks[0].operations
    )
    moves = tuple(c for c in controls if isinstance(c, RydbergTransport))
    collision = _collision(context, moves, boundaries)
    if collision is not None:
        raise _invalid(collision, "times")
    pulses = tuple(c for c in controls if isinstance(c, RydbergPulse))
    names = set(
        parameter_names_in_value([(p.amplitude, p.detuning, p.phase) for p in pulses])
    )
    intervals = grid.numel() - 1
    if names or parameter_values is not None:
        if (
            not isinstance(parameter_values, Sequence)
            or isinstance(parameter_values, (str, bytes))
            or len(parameter_values) != intervals
        ):
            raise _error(
                "invalid_parameter_values",
                "parameter_values must contain one mapping per interval",
                "parameter_values",
            )
    bindings: list[dict[str, float]] | None = [] if names else None
    local_n = torch.diag(torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device))
    local_drive = torch.zeros((3, 3), dtype=dtype, device=device)
    local_drive[0, 1] = 1
    populations = [_embed(local_n, i, count) for i in range(count)]
    drives = [_embed(local_drive, i, count) for i in range(count)]
    jumps = []
    for i in range(count):
        for level in (0, 1):
            jump = torch.zeros((3, 3), dtype=dtype, device=device)
            jump[2, level] = 1
            jumps.append((_embed(jump, i, count), context.loss_rate))
    order = {name: i for i, (name, _) in enumerate(context.atoms)}
    hamiltonians = []
    for interval in range(intervals):
        field = f"parameter_values[{interval}]"
        values: dict[str, Any] = {}
        if parameter_values is not None:
            raw = _normalize_parameter_mapping(
                parameter_values[interval],
                names,
                dtype=dtype,
                device=device,
                field=field,
            )
            for name, value in raw.items():
                if (
                    value.is_complex()
                    if isinstance(value, torch.Tensor)
                    else isinstance(value, Complex) and not isinstance(value, Real)
                ):
                    raise _error(
                        "invalid_coefficient",
                        "native controls require real scalar bindings",
                        field,
                    )
                tensor = value_to_tensor(value, dtype=real_dtype, device=device).clone()
                if not bool(torch.isfinite(tensor).item()):
                    raise _error(
                        "invalid_coefficient",
                        "native bindings must be finite at requested precision",
                        field,
                    )
                values[name] = tensor
            if bindings is not None:
                bindings.append(
                    {
                        name: float(value.detach().item())
                        for name, value in values.items()
                    }
                )
        # Fraction preserves the encoded midpoint without overflowing a+b.
        midpoint = (
            Fraction(float(grid[interval].item()))
            + Fraction(float(grid[interval + 1].item()))
        ) / 2
        matrix = torch.zeros((3**count, 3**count), dtype=dtype, device=device)
        for a, b in combinations(range(count), 2):
            pa = _exact_position(
                context, moves, context.atoms[a][0], midpoint, boundaries
            )
            pb = _exact_position(
                context, moves, context.atoms[b][0], midpoint, boundaries
            )
            squared = sum((x - y) ** 2 for x, y in zip(pa, pb, strict=True))
            try:
                interaction = float(Fraction(context.c6) / squared**3)
            except (OverflowError, ZeroDivisionError) as exc:
                raise _invalid(
                    "interaction cannot be represented at this geometry", "program"
                ) from exc
            matrix = matrix + interaction * (populations[a] @ populations[b])
        for pulse in pulses:
            if (
                not Fraction(boundaries[pulse.start])
                <= midpoint
                < Fraction(boundaries[pulse.start + pulse.duration])
            ):
                continue
            scalars = [
                value_to_tensor(
                    bind_parameter_value(v, values), dtype=real_dtype, device=device
                )
                for v in (pulse.amplitude, pulse.detuning, pulse.phase)
            ]
            amplitude, detuning, phase = scalars
            if any(not bool(torch.isfinite(v).item()) for v in scalars):
                raise _error(
                    "invalid_coefficient", "bound native controls must be finite", field
                )
            phase_factor = torch.exp(1j * phase)
            for actor in pulse.atoms:
                index = order[actor]
                drive = phase_factor * drives[index]
                matrix = (
                    matrix
                    + amplitude / 2 * (drive + drive.mH)
                    - detuning * populations[index]
                )
        hamiltonians.append(matrix)
    return {
        "hamiltonian": None,
        "n_qubits": None,
        "times": grid,
        "dimensions": (3,) * count,
        "collapse_operators": jumps,
        "bound_hamiltonians": tuple(hamiltonians),
    }, bindings

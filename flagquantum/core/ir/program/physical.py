"""The typed Rydberg profile of ProgramRecord, without a second program IR.

This module validates imposed controls and their geometry. It does not assemble
Hamiltonians, bind controls, evolve states or own any execution backend.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import combinations
from typing import TYPE_CHECKING, Literal, TypeAlias

from ...geometry import cartesian_pair, finite_real, same_time
from ...parameters import Parameter, ParameterExpression, parameter_names_in_value
from ..diagnostics import LevelDiagnostic, _diagnostic

if TYPE_CHECKING:
    from .model import Operation, ProgramRecord

ControlScalar: TypeAlias = float | Parameter | ParameterExpression


@dataclass(frozen=True)
class RydbergContext:
    atoms: tuple[tuple[str, str], ...]
    sites: tuple[tuple[str, tuple[float, float]], ...]
    length_unit: Literal["m", "um", "nm"]
    time_unit: Literal["s", "ms", "us"]
    c6: float
    loss_rate: float
    minimum_separation: float
    coordinate_frame: str
    transport_channels: tuple[str, ...]
    drive_channels: tuple[str, ...]
    geometry_sampling: Literal["interval_midpoint"]


@dataclass(frozen=True)
class RydbergTransport:
    atoms: tuple[str, ...]
    source: tuple[str, ...]
    destination: tuple[str, ...]
    start: float
    duration: float
    channel: str


@dataclass(frozen=True)
class RydbergPulse:
    atoms: tuple[str, ...]
    start: float
    duration: float
    amplitude: ControlScalar
    detuning: ControlScalar
    phase: ControlScalar
    channel: str


def identifier(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value:
        raise ValueError(f"{field} must be nonempty")
    return value


def control_scalar(value: object, field: str) -> ControlScalar:
    """Validate leaves; Core's existing expression implementation owns algebra."""
    if isinstance(value, Parameter):
        identifier(value.name, field)
        return value
    if isinstance(value, ParameterExpression):
        if not isinstance(value.args, tuple):
            raise TypeError(f"{field} expression operands must be immutable")
        for argument in value.args:
            control_scalar(argument, field)
        # Reuse the existing operator/arity validator, not another evaluator.
        value.bind(dict.fromkeys(parameter_names_in_value(value), 0.0))
        return value
    return finite_real(value, field)


def check_context(context: RydbergContext) -> None:
    """Validate immutable context data, raising a field-specific input error."""
    if not isinstance(context.atoms, tuple) or not context.atoms:
        raise ValueError("atoms must be a nonempty immutable roster")
    if not isinstance(context.sites, tuple) or not context.sites:
        raise ValueError("sites must be a nonempty immutable table")
    actors: list[str] = []
    occupied: list[str] = []
    sites: dict[str, tuple[float, float]] = {}
    for pair in context.sites:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("each site must be an immutable ID/coordinate pair")
        name = identifier(pair[0], "site ID")
        if not isinstance(pair[1], tuple):
            raise TypeError("site coordinates must be immutable pairs")
        point = cartesian_pair(pair[1], f"site {name!r}")
        if name in sites or point in sites.values():
            raise ValueError("site IDs and coordinate pairs must be unique")
        sites[name] = point
    if tuple(sites) != tuple(sorted(sites)):
        raise ValueError("site table must be sorted by ID")
    for atom_pair in context.atoms:
        if not isinstance(atom_pair, tuple) or len(atom_pair) != 2:
            raise ValueError("each atom must be an immutable atom/site pair")
        actor, site = identifier(atom_pair[0], "atom ID"), identifier(
            atom_pair[1], "initial site"
        )
        if actor in actors or site in occupied or site not in sites:
            raise ValueError("atom IDs/initial assignments must be unique and known")
        actors.append(actor)
        occupied.append(site)
    identifier(context.length_unit, "length_unit")
    identifier(context.time_unit, "time_unit")
    identifier(context.geometry_sampling, "geometry_sampling")
    if context.length_unit not in ("m", "um", "nm"):
        raise ValueError("length_unit must be m, um or nm")
    if context.time_unit not in ("s", "ms", "us"):
        raise ValueError("time_unit must be s, ms or us")
    identifier(context.coordinate_frame, "coordinate_frame")
    finite_real(context.c6, "c6")
    if finite_real(context.loss_rate, "loss_rate") < 0:
        raise ValueError("loss_rate must be nonnegative")
    if finite_real(context.minimum_separation, "minimum_separation") <= 0:
        raise ValueError("minimum_separation must be positive")
    if context.geometry_sampling != "interval_midpoint":
        raise ValueError("geometry_sampling must be interval_midpoint")
    for channels in (context.transport_channels, context.drive_channels):
        if not isinstance(channels, tuple):
            raise TypeError("declared channels must be immutable")
        for channel in channels:
            identifier(channel, "channel")
        if channels != tuple(sorted(set(channels))):
            raise ValueError("channels must be unique and sorted")
    if set(context.transport_channels) & set(context.drive_channels):
        raise ValueError("transport and drive channel names must not overlap")


def check_control(
    context: RydbergContext, data: RydbergTransport | RydbergPulse
) -> None:
    if not isinstance(data.atoms, tuple) or not data.atoms:
        raise ValueError("control atoms must be a nonempty immutable group")
    order = {actor: i for i, (actor, _) in enumerate(context.atoms)}
    for actor in data.atoms:
        identifier(actor, "control atom")
        if actor not in order:
            raise ValueError(f"unknown control atom {actor!r}")
    if len(set(data.atoms)) != len(data.atoms):
        raise ValueError("a control group cannot repeat an atom")
    if data.atoms != tuple(sorted(data.atoms, key=order.__getitem__)):
        raise ValueError("control groups must use the declared subsystem order")
    start = finite_real(data.start, "start")
    duration = finite_real(data.duration, "duration")
    if start < 0 or duration <= 0 or not math.isfinite(start + duration):
        raise ValueError(
            "start must be nonnegative and duration positive with finite end"
        )
    if start + duration == start:
        raise ValueError("duration must produce a distinct binary64 end boundary")
    identifier(data.channel, "channel")
    if isinstance(data, RydbergTransport):
        sites = dict(context.sites)
        for references in (data.source, data.destination):
            if not isinstance(references, tuple) or len(references) != len(data.atoms):
                raise ValueError(
                    "move atoms/source/destination must have matching lengths"
                )
            for site in references:
                identifier(site, "move site")
                if site not in sites:
                    raise ValueError(f"unknown move site {site!r}")
        if data.channel not in context.transport_channels:
            raise ValueError(f"undeclared transport channel {data.channel!r}")
    else:
        if data.channel not in context.drive_channels:
            raise ValueError(f"undeclared drive channel {data.channel!r}")
        for name in ("amplitude", "detuning", "phase"):
            control_scalar(getattr(data, name), name)


def operation_key(operation: Operation) -> tuple[float, str, tuple[str, ...], str]:
    data = operation.physical_data
    assert data is not None
    return data.start, operation.opcode, data.atoms, data.channel


def _overlap(
    left: RydbergTransport | RydbergPulse, right: RydbergTransport | RydbergPulse
) -> bool:
    start = max(left.start, right.start)
    end = min(left.start + left.duration, right.start + right.duration)
    return start < end and not same_time(start, end)


def _histories(
    context: RydbergContext, moves: tuple[RydbergTransport, ...]
) -> str | None:
    for atom, initial in context.atoms:
        current = initial
        previous: RydbergTransport | None = None
        for move in moves:
            if atom not in move.atoms:
                continue
            if previous is not None and _overlap(previous, move):
                return f"atom {atom!r} has overlapping moves"
            i = move.atoms.index(atom)
            if move.source[i] != current:
                return f"atom {atom!r} source {move.source[i]!r} disagrees with prior site {current!r}"
            current, previous = move.destination[i], move
    return None


def _boundary_times(
    controls: tuple[RydbergTransport | RydbergPulse, ...],
) -> dict[float, float]:
    """Resolve equivalent encoded boundaries once for all actors and groups.

    The source retains its original literals. This derived geometry view uses
    the approved ULP equivalence rather than giving one atom a different group
    start or introducing an infinitesimal discontinuity between adjacent moves.
    """
    times = sorted(
        {0.0, *(c.start for c in controls), *(c.start + c.duration for c in controls)}
    )
    result: dict[float, float] = {}
    representative = previous = times[0]
    for value in times:
        if same_time(value, representative):
            result[value] = representative
        elif same_time(value, previous):
            raise ValueError("source boundaries have ambiguous eight-ULP equivalence")
        else:
            representative = value
            result[value] = value
        previous = value
    if any(result[c.start] >= result[c.start + c.duration] for c in controls):
        raise ValueError(
            "control duration collapses under the source boundary precision"
        )
    return result


def _exact_position(
    context: RydbergContext,
    moves: tuple[RydbergTransport, ...],
    atom: str,
    time: Fraction,
    boundaries: dict[float, float],
) -> tuple[Fraction, Fraction]:
    sites = dict(context.sites)
    point = sites[dict(context.atoms)[atom]]
    for move in moves:
        if atom not in move.atoms:
            continue
        i = move.atoms.index(atom)
        start = Fraction(boundaries[move.start])
        end = Fraction(boundaries[move.start + move.duration])
        if time >= end:
            point = sites[move.destination[i]]
        elif time >= start:
            fraction = (time - start) / (end - start)
            source, destination = sites[move.source[i]], sites[move.destination[i]]
            return (
                (1 - fraction) * Fraction(source[0])
                + fraction * Fraction(destination[0]),
                (1 - fraction) * Fraction(source[1])
                + fraction * Fraction(destination[1]),
            )
        else:
            break
    return Fraction(point[0]), Fraction(point[1])


def commanded_position(
    context: RydbergContext,
    operations: tuple[Operation, ...],
    atom: str,
    time: float,
) -> tuple[float, float]:
    """Query the commanded path, independently of physical presence/loss."""
    moves = tuple(
        op.physical_data
        for op in operations
        if isinstance(op.physical_data, RydbergTransport)
    )
    controls = tuple(
        op.physical_data for op in operations if op.physical_data is not None
    )
    boundaries = _boundary_times(controls)
    matching = {value for value in boundaries.values() if same_time(time, value)}
    if len(matching) > 1:
        raise ValueError("query time matches multiple physical boundaries")
    resolved = next(iter(matching)) if matching else time
    x, y = _exact_position(context, moves, atom, Fraction(resolved), boundaries)
    return float(x), float(y)


def _collision(
    context: RydbergContext,
    moves: tuple[RydbergTransport, ...],
    boundary_map: dict[float, float],
) -> str | None:
    boundaries = sorted(set(boundary_map.values()))
    intervals = list(zip(boundaries, boundaries[1:], strict=False)) or [(0.0, 0.0)]
    threshold = Fraction(context.minimum_separation) ** 2
    for left, right in intervals:
        for (a, _), (b, _) in combinations(context.atoms, 2):
            a0 = _exact_position(context, moves, a, Fraction(left), boundary_map)
            b0 = _exact_position(context, moves, b, Fraction(left), boundary_map)
            a1 = _exact_position(context, moves, a, Fraction(right), boundary_map)
            b1 = _exact_position(context, moves, b, Fraction(right), boundary_map)
            delta = (a0[0] - b0[0], a0[1] - b0[1])
            slope = (a1[0] - b1[0] - delta[0], a1[1] - b1[1] - delta[1])
            speed = slope[0] ** 2 + slope[1] ** 2
            location = (
                max(
                    Fraction(0),
                    min(
                        Fraction(1),
                        -(delta[0] * slope[0] + delta[1] * slope[1]) / speed,
                    ),
                )
                if speed
                else Fraction(0)
            )
            distance = sum((delta[i] + location * slope[i]) ** 2 for i in (0, 1))
            if distance < threshold:
                return f"atoms {a!r}/{b!r} breach minimum_separation during [{left}, {right}]"
    return None


def is_physical_record(record: ProgramRecord) -> bool:
    """Recognize typed physical meaning without reserving circuit opcode names."""
    return record.physical_context is not None or any(
        operation.physical_data is not None
        for function in record.functions
        for block in function.blocks
        for operation in block.operations
    )


def _structural_errors(record: ProgramRecord) -> tuple[LevelDiagnostic, ...]:
    """Reuse the existing CFG/SSA verifier for the graph around native controls.

    Native controls define/consume no SSA values. Removing only those controls
    for this structural check leaves the original source intact and does not
    lower it to a circuit or another IR.
    """
    from .model import BlockRecord, FunctionRecord, Operation
    from .verifier import _verify_function

    if not isinstance(record.functions, tuple) or not record.functions:
        return (
            _diagnostic("program.function_arity", "a program record needs a function"),
        )
    diagnostics: list[LevelDiagnostic] = []
    for function in record.functions:
        if not isinstance(function, FunctionRecord) or not isinstance(
            function.blocks, tuple
        ):
            return (
                _diagnostic(
                    "physical.invalid_operation",
                    "functions and blocks must be immutable typed records",
                ),
            )
        blocks = []
        for block in function.blocks:
            if not isinstance(block, BlockRecord) or any(
                not isinstance(value, tuple)
                for value in (block.arguments, block.edges, block.operations)
            ):
                return (
                    _diagnostic(
                        "physical.invalid_operation",
                        "block fields must be immutable typed data",
                    ),
                )
            if any(
                not isinstance(operation, Operation) for operation in block.operations
            ):
                return (
                    _diagnostic(
                        "physical.invalid_operation",
                        "operations must be Operation records",
                    ),
                )
            ordinary = tuple(
                op
                for op in block.operations
                if op.physical_data is None
                and op.opcode not in ("rydberg.move", "rydberg.pulse")
            )
            blocks.append(replace(block, operations=ordinary))
        diagnostics.extend(_verify_function(replace(function, blocks=tuple(blocks))))
    return tuple(diagnostics)


def verify_physical_program(record: ProgramRecord) -> tuple[LevelDiagnostic, ...]:
    """Verify the one admitted physical profile through the existing verifier."""
    from .model import Operation

    context = record.physical_context
    if not isinstance(context, RydbergContext):
        return (
            _diagnostic(
                "physical.invalid_context", "a typed RydbergContext is required"
            ),
        )
    try:
        check_context(context)
    except (TypeError, ValueError) as exc:
        return (_diagnostic("physical.invalid_context", str(exc)),)
    structure_errors = _structural_errors(record)
    if structure_errors:
        return structure_errors
    if (
        len(record.functions) != 1
        or record.functions[0].name != "main"
        or record.functions[0].entry != "entry"
        or len(record.functions[0].blocks) != 1
    ):
        return (
            _diagnostic(
                "physical.unsupported_structure",
                "native controls require one main/entry function and block",
            ),
        )
    block = record.functions[0].blocks[0]
    if (
        block.name != "entry"
        or block.arguments
        or block.edges
        or block.terminator != "return"
        or record.canonical_payload is not None
        or any(
            op.physical_data is None
            and op.opcode not in ("rydberg.move", "rydberg.pulse")
            for op in block.operations
        )
    ):
        return (
            _diagnostic(
                "physical.unsupported_structure",
                "native controls cannot carry SSA arguments, edges, branches or a circuit payload",
            ),
        )
    controls: list[RydbergTransport | RydbergPulse] = []
    diagnostics: list[LevelDiagnostic] = []
    if not isinstance(block.operations, tuple):
        return (
            _diagnostic(
                "physical.invalid_operation", "operations must be an immutable tuple"
            ),
        )
    for i, operation in enumerate(block.operations):
        if not isinstance(operation, Operation):
            diagnostics.append(
                _diagnostic(
                    "physical.invalid_operation", f"operation {i} must be an Operation"
                )
            )
            continue
        data = operation.physical_data
        if (
            operation.operands
            or operation.results
            or operation.parameters
            or operation.literal is not None
            or operation.condition_clauses
        ):
            diagnostics.append(
                _diagnostic(
                    "physical.invalid_operation",
                    f"operation {i} mixes physical controls with SSA/circuit fields",
                )
            )
            continue
        expected = (
            RydbergTransport
            if operation.opcode == "rydberg.move"
            else RydbergPulse if operation.opcode == "rydberg.pulse" else None
        )
        if expected is None or type(data) is not expected:
            diagnostics.append(
                _diagnostic(
                    "physical.invalid_operation",
                    f"operation {i} needs the exact typed payload for {operation.opcode!r}",
                )
            )
            continue
        assert isinstance(data, (RydbergTransport, RydbergPulse))
        try:
            check_control(context, data)
        except (TypeError, ValueError) as exc:
            diagnostics.append(
                _diagnostic("physical.invalid_operation", f"operation {i}: {exc}")
            )
            continue
        controls.append(data)
    if diagnostics:
        return tuple(diagnostics)
    if tuple(sorted(block.operations, key=operation_key)) != block.operations:
        return (
            _diagnostic(
                "physical.invalid_operation",
                "timed operations must be in canonical order",
            ),
        )
    try:
        boundary_map = _boundary_times(tuple(controls))
    except ValueError as exc:
        return (_diagnostic("physical.invalid_operation", str(exc)),)
    moves = tuple(c for c in controls if isinstance(c, RydbergTransport))
    issue = _histories(context, moves)
    if issue is not None:
        diagnostics.append(_diagnostic("physical.occupancy_conflict", issue))
    channels: dict[str, list[RydbergTransport | RydbergPulse]] = defaultdict(list)
    for control in controls:
        channels[control.channel].append(control)
    for channel, scheduled in channels.items():
        if any(_overlap(a, b) for a, b in combinations(scheduled, 2)):
            diagnostics.append(
                _diagnostic(
                    "physical.channel_conflict",
                    f"channel {channel!r} has overlapping controls",
                )
            )
    pulses = tuple(c for c in controls if isinstance(c, RydbergPulse))
    if any(
        set(a.atoms) & set(b.atoms) and _overlap(a, b)
        for a, b in combinations(pulses, 2)
    ):
        diagnostics.append(
            _diagnostic(
                "physical.channel_conflict",
                "one atom cannot receive overlapping pulses",
            )
        )
    if issue is None:
        collision = _collision(context, moves, boundary_map)
        if collision is not None:
            diagnostics.append(_diagnostic("physical.path_collision", collision))
    return tuple(diagnostics)

"""Experimental Rydberg controls backed by the existing Core ProgramRecord.

This frontend constructs and verifies imposed paths and controls. Mathematical
tensor helpers are small CPU inputs, not a dynamics or hardware implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Literal

import torch

from ..errors import CapabilityError
from .geometry import cartesian_pair, finite_real
from .ir import IRValidationError
from .ir.program.model import BlockRecord, FunctionRecord, Operation, ProgramRecord
from .ir.program.physical import (
    RydbergContext,
    RydbergPulse,
    RydbergTransport,
    check_context,
    check_control,
    commanded_position,
    control_scalar,
    identifier,
    operation_key,
)
from .ir.program.verifier import verify_program_record
from .ir.state_space import ProductStateSpace
from .parameters import Parameter, ParameterExpression


def _ids(value: str | Sequence[str], field: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (identifier(value, field),)
    if not isinstance(value, Sequence) or isinstance(value, bytes):
        raise TypeError(f"{field} must be an ID or ordered sequence of IDs")
    return tuple(identifier(item, field) for item in value)


class RydbergProgram:
    """An imposed ground/Rydberg/lost control program with named atom identity.

    ``move`` and ``pulse`` append controls without making append order physical
    time. Full history/resource checks occur when taking ``to_ir()`` or querying
    geometry/tensors, so a program may be assembled in timestamp-independent
    order. Each snapshot is immutable; later edits leave earlier snapshots intact.
    """

    def __init__(
        self,
        *,
        atoms: Sequence[tuple[str, str]],
        sites: Mapping[str, tuple[float, float]],
        length_unit: Literal["m", "um", "nm"],
        time_unit: Literal["s", "ms", "us"],
        c6: float,
        loss_rate: float,
        minimum_separation: float,
        coordinate_frame: str = "program_local",
        transport_channels: Sequence[str] = (),
        drive_channels: Sequence[str] = (),
        geometry_sampling: Literal["interval_midpoint"] = "interval_midpoint",
    ) -> None:
        if not isinstance(atoms, Sequence) or isinstance(atoms, (str, bytes)):
            raise TypeError("atoms must be an ordered sequence of atom/site pairs")
        if not isinstance(sites, Mapping):
            raise TypeError("sites must map site IDs to Cartesian pairs")
        if any(
            not isinstance(channels, Sequence) or isinstance(channels, (str, bytes))
            for channels in (transport_channels, drive_channels)
        ):
            raise TypeError("channel declarations must be ordered sequences of names")
        try:
            roster: list[tuple[str, str]] = []
            for pair in atoms:
                if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)):
                    raise TypeError("each atom must be an atom/site pair")
                if len(pair) != 2:
                    raise ValueError("each atom needs exactly one initial site")
                roster.append(
                    (
                        identifier(pair[0], "atom ID"),
                        identifier(pair[1], "initial site"),
                    )
                )
            coordinates = tuple(
                sorted(
                    (
                        identifier(name, "site ID"),
                        cartesian_pair(point, f"site {name!r}"),
                    )
                    for name, point in sites.items()
                )
            )
            context = RydbergContext(
                tuple(roster),
                coordinates,
                length_unit,
                time_unit,
                finite_real(c6, "c6"),
                finite_real(loss_rate, "loss_rate"),
                finite_real(minimum_separation, "minimum_separation"),
                identifier(coordinate_frame, "coordinate_frame"),
                tuple(sorted(_ids(transport_channels, "transport_channels"))),
                tuple(sorted(_ids(drive_channels, "drive_channels"))),
                geometry_sampling,
            )
            check_context(context)
        except ValueError as exc:
            raise IRValidationError(str(exc)) from exc
        self._record = ProgramRecord(
            (FunctionRecord("main", (BlockRecord("entry"),)),), physical_context=context
        )
        self.to_ir()

    @property
    def atom_ids(self) -> tuple[str, ...]:
        return tuple(actor for actor, _ in self._context.atoms)

    @property
    def dimensions(self) -> tuple[int, ...]:
        return ProductStateSpace((3,) * len(self.atom_ids)).dimensions

    @property
    def _context(self) -> RydbergContext:
        assert isinstance(self._record.physical_context, RydbergContext)
        return self._record.physical_context

    def _append(
        self, opcode: str, data: RydbergTransport | RydbergPulse
    ) -> RydbergProgram:
        try:
            check_control(self._context, data)
        except ValueError as exc:
            raise IRValidationError(str(exc)) from exc
        function = self._record.functions[0]
        block = function.blocks[0]
        operations = tuple(
            sorted(
                (*block.operations, Operation(opcode, physical_data=data)),
                key=operation_key,
            )
        )
        self._record = replace(
            self._record,
            functions=(
                replace(function, blocks=(replace(block, operations=operations),)),
            ),
        )
        return self

    def move(
        self,
        atoms: str | Sequence[str],
        *,
        source: str | Sequence[str],
        destination: str | Sequence[str],
        start: float,
        duration: float,
        channel: str,
    ) -> RydbergProgram:
        try:
            actors, sources, destinations = (
                _ids(atoms, "move atoms"),
                _ids(source, "source"),
                _ids(destination, "destination"),
            )
            if len(actors) != len(sources) or len(actors) != len(destinations):
                raise ValueError(
                    "move atoms/source/destination must have matching lengths"
                )
            order = {name: i for i, name in enumerate(self.atom_ids)}
            if any(actor not in order for actor in actors):
                raise ValueError("move references an unknown atom")
            pairs = sorted(
                zip(actors, sources, destinations, strict=True),
                key=lambda item: order[item[0]],
            )
            data = RydbergTransport(
                tuple(item[0] for item in pairs),
                tuple(item[1] for item in pairs),
                tuple(item[2] for item in pairs),
                finite_real(start, "start"),
                finite_real(duration, "duration"),
                identifier(channel, "channel"),
            )
        except ValueError as exc:
            raise IRValidationError(str(exc)) from exc
        return self._append("rydberg.move", data)

    def pulse(
        self,
        atoms: str | Sequence[str],
        *,
        start: float,
        duration: float,
        amplitude: float | Parameter | ParameterExpression,
        detuning: float | Parameter | ParameterExpression = 0.0,
        phase: float | Parameter | ParameterExpression = 0.0,
        channel: str,
    ) -> RydbergProgram:
        try:
            actors = _ids(atoms, "pulse atoms")
            order = {name: i for i, name in enumerate(self.atom_ids)}
            if any(actor not in order for actor in actors):
                raise ValueError("pulse references an unknown atom")
            data = RydbergPulse(
                tuple(sorted(actors, key=order.__getitem__)),
                finite_real(start, "start"),
                finite_real(duration, "duration"),
                control_scalar(amplitude, "amplitude"),
                control_scalar(detuning, "detuning"),
                control_scalar(phase, "phase"),
                identifier(channel, "channel"),
            )
        except ValueError as exc:
            raise IRValidationError(str(exc)) from exc
        return self._append("rydberg.pulse", data)

    def to_ir(self) -> ProgramRecord:
        """Return a validated immutable snapshot of the sole source record."""
        diagnostics = verify_program_record(self._record)
        if diagnostics:
            message = "; ".join(f"{item.code}: {item.message}" for item in diagnostics)
            if all(
                item.code == "physical.unsupported_structure" for item in diagnostics
            ):
                raise CapabilityError(message)
            raise IRValidationError(message)
        return self._record

    def commanded_position(self, atom: str, time: float) -> tuple[float, float]:
        record = self.to_ir()
        try:
            identifier(atom, "atom")
        except ValueError as exc:
            raise IRValidationError(str(exc)) from exc
        if atom not in self.atom_ids:
            raise IRValidationError(f"unknown atom {atom!r}")
        try:
            value = finite_real(time, "time")
        except ValueError as exc:
            raise IRValidationError(str(exc)) from exc
        if value < 0:
            raise IRValidationError("time must be nonnegative")
        try:
            return commanded_position(
                self._context, record.functions[0].blocks[0].operations, atom, value
            )
        except ValueError as exc:
            raise IRValidationError(str(exc)) from exc

    def _space(self) -> ProductStateSpace:
        self.to_ir()
        if len(self.atom_ids) > 2:
            raise CapabilityError(
                "Rydberg tensor helpers support one or two atoms before allocation"
            )
        return ProductStateSpace(self.dimensions)

    def basis_state(self, occupations: Sequence[str]) -> torch.Tensor:
        space = self._space()
        if not isinstance(occupations, Sequence) or isinstance(
            occupations, (str, bytes)
        ):
            raise TypeError("occupations must be an ordered sequence")
        levels = {"g": 0, "r": 1, "lost": 2}
        if any(not isinstance(level, str) for level in occupations):
            raise TypeError("occupations must contain state names")
        if any(level not in levels for level in occupations):
            raise IRValidationError("occupations must be g, r or lost")
        try:
            index = space.basis_index(tuple(levels[level] for level in occupations))
        except ValueError as exc:
            raise IRValidationError(str(exc)) from exc
        result = torch.zeros(space.dimension, dtype=torch.complex128, device="cpu")
        result[index] = 1
        return result

    def projector(self, occupations: Mapping[str, str]) -> torch.Tensor:
        self._space()
        if not isinstance(occupations, Mapping):
            raise TypeError("occupations must map atom IDs to g/r/lost")
        levels = {"g": 0, "r": 1, "lost": 2}
        for atom, level in occupations.items():
            try:
                identifier(atom, "atom")
            except ValueError as exc:
                raise IRValidationError(str(exc)) from exc
            if not isinstance(level, str):
                raise TypeError("occupations must contain state names")
            if (
                atom not in self.atom_ids
                or not isinstance(level, str)
                or level not in levels
            ):
                raise IRValidationError(
                    f"unknown atom/state selection {atom!r}: {level!r}"
                )
        result = torch.ones((1, 1), dtype=torch.complex128, device="cpu")
        for atom in self.atom_ids:
            local = torch.eye(3, dtype=torch.complex128, device="cpu")
            if atom in occupations:
                local.zero_()
                index = levels[occupations[atom]]
                local[index, index] = 1
            result = torch.kron(result, local)
        return result

    def survival_operator(self) -> torch.Tensor:
        return self.no_flag_operator(1.0)

    def no_flag_operator(self, efficiency: float) -> torch.Tensor:
        self._space()
        try:
            value = finite_real(efficiency, "efficiency")
        except ValueError as exc:
            raise IRValidationError(str(exc)) from exc
        if not 0 <= value <= 1:
            raise IRValidationError("efficiency must lie in [0, 1]")
        local = torch.diag(
            torch.tensor([1.0, 1.0, 1 - value], dtype=torch.complex128, device="cpu")
        )
        result = torch.ones((1, 1), dtype=torch.complex128, device="cpu")
        for _ in self.atom_ids:
            result = torch.kron(result, local)
        return result

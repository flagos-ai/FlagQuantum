# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ProgrammableRotationGateArrayBase.
# Source: https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/rotations/programmable_rotation_gate_array.py

"""Programmable rotations through successive XOR updates of one angle word."""

from collections.abc import Callable, Iterator

from ....core.ir import Instruction
from ..bits.bitwise import _validate_wires
from ..lookup.source_qrom import source_qrom_ir, source_qrom_workspace_size


def _rotation_batches(
    angles: tuple[tuple[int, ...], ...], width: int
) -> Iterator[tuple[tuple[int, ...], tuple[tuple[int, int], ...]]]:
    """Yield each XOR delta and its (angle index, bit power) columns."""
    columns = tuple(
        (index, power)
        for index, row in enumerate(angles)
        for power in reversed(range(max(row).bit_length()))
    )
    previous = (0,) * len(angles[0])
    for start in range(0, len(columns), width):
        batch = columns[start : start + width]
        words = tuple(
            sum(
                ((angles[index][address] >> power) & 1) << (width - position - 1)
                for position, (index, power) in enumerate(batch)
            )
            for address in range(len(previous))
        )
        yield tuple(old ^ new for old, new in zip(previous, words, strict=True)), batch
        previous = words
    yield previous, ()


def programmable_rotation_array_ir(
    selection: tuple[int, ...],
    load: tuple[int, ...],
    target: tuple[int, ...],
    angles: tuple[tuple[int, ...], ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    *,
    rotation: Callable[[int, tuple[int, ...], int], tuple[Instruction, ...]],
    interleaved: tuple[tuple[Instruction, ...], ...],
    interleaved_target: tuple[int, ...] = (),
) -> tuple[Instruction, ...]:
    """Emit the pinned rotation-array generator on bound registers.

    Selection and load are big endian. Supply one row per rotation stage
    and one entry per valid selection. The rotation callback emits the
    controlled base gate to power 2**(-1-bit_power), preserving its phase
    convention. Interleavers may act on target and interleaved_target.
    Load and QROM workspace start and finish clean.
    """
    if (
        not angles
        or not angles[0]
        or any(len(row) != len(angles[0]) for row in angles)
        or any(type(value) is not int or value < 0 for row in angles for value in row)
    ):
        raise ValueError(
            "Rotation array requires equally sized nonempty unsigned angle rows"
        )
    extent = len(angles[0])
    if not load or not target or len(selection) != (extent - 1).bit_length():
        raise ValueError(
            "Rotation array requires source selection width and nonempty load and target"
        )
    if len(interleaved) != len(angles) - 1:
        raise ValueError(
            "Rotation array requires one interleaver per adjacent angle row"
        )
    _validate_wires((*selection, *load, *target, *workspace, *interleaved_target))
    if (
        len(classical_bits) != len(workspace)
        or len(set(classical_bits)) != len(classical_bits)
        or any(type(bit) is not int or bit < 0 for bit in classical_bits)
    ):
        raise ValueError(
            "Rotation array requires one distinct classical slot per work wire"
        )
    interleave_wires = {*target, *interleaved_target}
    if any(
        wire not in interleave_wires
        for stage in interleaved
        for gate in stage
        for wire in gate.wires
    ):
        raise ValueError("Rotation array interleaver used a wire outside its targets")

    gates: list[Instruction] = []
    last_index = 0
    for delta, batch in _rotation_batches(angles, len(load)):
        size = source_qrom_workspace_size(0, (len(selection),), (delta,))
        if len(workspace) < size:
            raise ValueError(
                f"Rotation array requires at least {size} clean work wires"
            )
        gates.extend(
            source_qrom_ir(
                (),
                (selection,),
                (extent,),
                ((load,),),
                (delta,),
                workspace[:size],
                classical_bits[:size],
            )
        )
        for control, (index, power) in zip(load[: len(batch)], batch, strict=True):
            if index != last_index:
                gates.extend(interleaved[last_index])
                last_index = index
            stage = rotation(control, target, power)
            if any(
                wire not in (control, *target) for gate in stage for wire in gate.wires
            ):
                raise ValueError(
                    "Rotation array gate used a wire outside its control and target"
                )
            gates.extend(stage)
    return tuple(gates)

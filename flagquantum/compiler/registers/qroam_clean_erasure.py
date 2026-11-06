# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Uses the locked forward QROAM layout and measured XOR uncomputation.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/data_loading/qroam_clean.py

"""Measurement-assisted erasure on matching signed clean-bank outputs."""

from ...core.ir import Instruction
from .measurement import ParityPhase, measure_x_clear_ir
from .source_qroam_clean import (
    QroamCleanTable,
    _source_qroam_batch_ir,
    _source_qroam_route_ir,
)


def qroam_clean_measurement_erase_ir(
    table: QroamCleanTable,
    controls: tuple[int, ...],
    selections: tuple[tuple[int, ...], ...],
    banks: tuple[tuple[tuple[int, ...], ...], ...],
    workspace: tuple[int, ...],
    classical_bits: tuple[int, ...],
    bank_bits: tuple[int, ...],
) -> tuple[Instruction, ...]:
    """Erase matching clean-bank outputs with X measurements and phase feedback.

    Bank bits follow dataset, block cell, then big-endian wire order. The
    bank outcomes occupy slots distinct from measured traversal scratch.
    First reverse signed routing so the remaining bank state is the literal
    XOR load. Every bank branch then acts as the coherent inverse scaled by
    its positive projection amplitude. The source's abstract erasure cost
    describes a separate decomposition.
    """
    load, low = _source_qroam_batch_ir(
        table, controls, selections, banks, workspace, classical_bits
    )
    bank_wires = tuple(wire for bank in banks for word in bank for wire in word)
    if (
        len(bank_bits) != len(bank_wires)
        or any(type(bit) is not int or bit < 0 for bit in bank_bits)
        or len(set((*bank_bits, *classical_bits)))
        != len(bank_bits) + len(classical_bits)
    ):
        raise ValueError(
            "QROAM bank outcomes require distinct slots separate from scratch"
        )
    outcomes = dict(zip(bank_wires, bank_bits, strict=True))
    route = _source_qroam_route_ir(table, low, banks, reverse=True)
    clear = tuple(
        gate for wire, bit in outcomes.items() for gate in measure_x_clear_ir(wire, bit)
    )
    feedback = []
    scalar_bits = []
    for gate in load:
        target = gate.wires[-1]
        if target not in outcomes:
            feedback.append(gate)
        elif gate.name == "cx":
            feedback.extend(ParityPhase(gate.wires[:1], (outcomes[target],)).lower())
        else:
            assert gate.name == "x"
            scalar_bits.append(outcomes[target])
    if scalar_bits:
        # Every bank is now zero, so one can temporarily carry a scalar phase.
        anchor = bank_wires[0]
        flip = Instruction("x", (anchor,))
        feedback.append(flip)
        for bit in scalar_bits:
            feedback.extend(ParityPhase((anchor,), (bit,)).lower())
        feedback.append(flip)
    return (*route, *clear, *feedback)

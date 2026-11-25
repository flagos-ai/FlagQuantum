# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Reuses native measured erasure for locked Qualtran v0.7.0 QROAM layouts.

"""Bind the explicit native erasure recipe without filling a source model leaf."""

from math import prod

import numpy as np

from ....core.ir import CircuitIR
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register, RegisterSide
from ....core.ir.program.signatures import Signature
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from .qroam_clean_erasure import qroam_clean_measurement_erase_ir
from .qroam_clean_metadata import qroam_clean_signature
from .source_qroam_clean import QroamCleanTable


def qroam_measurement_erasure_program(
    table: QroamCleanTable,
    target_widths: tuple[int, ...],
    *,
    selection_widths: tuple[int, ...] | None = None,
    num_controls: int = 0,
) -> ProgramRecord:
    """Consume matching signed output banks, returning selectors and clean scratch.

    Inputs use the forward layout with LEFT bank words and THRU selectors.
    This is the independently certified native measurement recipe. The original
    model-only QROAMCleanAdjoint declaration remains bodyless in source formats.
    Bank outcomes follow dataset/cell/MSB order, separate from predicate slots.
    """
    widths = (
        selection_widths
        if selection_widths is not None
        else tuple((n - 1).bit_length() for n in table.shape)
    )
    if len(target_widths) != len(table.datasets) or any(
        type(n) is not int or n < 0 for n in target_widths
    ):
        raise ValueError("QROAM erasure requires one nonnegative width per dataset")
    if type(num_controls) is not int or num_controls < 0:
        raise ValueError("QROAM erasure requires a nonnegative control count")
    table._high_widths(widths)
    signature = qroam_clean_signature(
        table.shape,
        target_widths,
        table.log_block_sizes,
        selection_widths=widths,
        controls=num_controls,
    ).adjoint()
    padded = tuple(
        n * k for n, k in zip(table.batch_shape, table.block_shape, strict=True)
    )
    indices = qroam_clean_signature(
        padded,
        (),
        table.log_block_sizes,
        selection_widths=widths,
        controls=num_controls,
    )
    leaf_signature = Signature(
        (
            *indices,
            *(
                Register(
                    f"target{i}_",
                    signature.get_left(f"target{i}_").dtype,
                    shape=table.block_shape,
                    side=RegisterSide.LEFT,
                )
                for i, width in enumerate(target_widths)
                if width
            ),
        )
    )
    leaf = _native_erasure_leaf(
        table, target_widths, widths, num_controls, leaf_signature
    )
    builder = ComponentProgramBuilder(signature)
    values = {
        reg.name: builder.inputs[reg.name]
        for reg in signature
        if reg.side != RegisterSide.LEFT
    }
    for i, width in enumerate(target_widths):
        if width:
            cells = (
                builder.inputs[f"target{i}_"],
                *np.asarray(builder.inputs.get(f"junk_target{i}_", ())).flat,
            )
            values[f"target{i}_"] = np.asarray(cells, dtype=object).reshape(
                table.block_shape
            )
    return builder.finalize(
        builder.call_program(leaf, values, entry=leaf.functions[0].name)
    )


def _native_erasure_leaf(
    table: QroamCleanTable,
    target_widths: tuple[int, ...],
    widths: tuple[int, ...],
    num_controls: int,
    signature: Signature,
) -> ProgramRecord:
    """Implement the explicit native recipe behind the source-shaped leaf ports."""
    builder = ComponentProgramBuilder(signature, name="native_qroam_erase_leaf")
    if not any(target_widths):
        return builder.finalize(dict(builder.inputs))
    erased_table = QroamCleanTable(
        table.shape,
        table.log_block_sizes,
        tuple(
            data
            for data, width in zip(table.datasets, target_widths, strict=True)
            if width
        ),
    )
    work = erased_table.workspace_size(num_controls, widths)
    bits: list[RegisterBinding] = []
    ports: dict[str, tuple[tuple[int, ...], ...]] = {}
    for reg in signature:
        coordinates = []
        for word in np.asarray(builder.inputs[reg.name]).flat:
            assert isinstance(word, RegisterBinding)
            split = tuple(builder.split(word))
            coordinates.append(tuple(range(len(bits), len(bits) + len(split))))
            bits.extend(split)
        ports[reg.name] = tuple(coordinates)
    logical = len(bits)
    for _ in range(work):
        bits.append(builder.allocate())
    controls = ports["control"][0] if num_controls else ()
    indices = iter(
        words[0]
        for name, words in ports.items()
        if name != "control" and not name.startswith(("target", "junk_"))
    )
    selections = tuple(next(indices) if width else () for width in widths)
    banks = tuple(
        ports[f"target{i}_"] for i, width in enumerate(target_widths) if width
    )
    bank_size = prod(table.block_shape) * sum(target_widths)
    gates = qroam_clean_measurement_erase_ir(
        erased_table,
        controls,
        selections,
        banks,
        tuple(range(logical, logical + work)),
        tuple(range(work)),
        tuple(range(work, work + bank_size)),
    )
    atom = bit_gate_function(
        "native_qroam_measured_erase",
        (Register("q", quantum_bit_type(), (len(bits),)),),
        CircuitIR(len(bits), gates),
    )
    returned = builder.call_program(
        ProgramRecord((atom,)), {"q": np.asarray(bits, dtype=object)}, entry=atom.name
    )["q"]
    flat = tuple(np.asarray(returned).flat)
    outputs = {}
    for reg in signature:
        elements = np.empty(reg.shape, dtype=object)
        for position, index in enumerate(reg.all_idxs()):
            group = tuple(flat[i] for i in ports[reg.name][position])
            if reg.side == RegisterSide.LEFT:
                for bit in group:
                    builder.free(bit)
            else:
                elements[index] = builder.join(group, dtype=reg.dtype)
        if reg.side != RegisterSide.LEFT:
            outputs[reg.name] = elements if reg.shape else elements[()]
    for bit in flat[logical:]:
        builder.free(bit)
    return builder.finalize(outputs)

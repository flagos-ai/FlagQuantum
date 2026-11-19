# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 QROAMClean and its model-only adjoint boundary.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/data_loading/qroam_clean.py

"""Own scalar clean lookup banks around the existing lookup and erase emitters."""

import numpy as np

from ....core.ir import CircuitIR
from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import quantum_bit_type
from ....core.ir.program.registers import Register
from ..component_builder import ComponentProgramBuilder
from ..component_gate_function import bit_gate_function
from .qroam_clean_erasure import qroam_clean_measurement_erase_ir
from .qroam_clean_metadata import qroam_clean_signature
from .source_qroam_clean import QroamCleanTable, source_qroam_clean_ir


def clean_lookup_program(
    table: QroamCleanTable, word_width: int, *, erase: bool = False
) -> ProgramRecord:
    """Load or measurement-erase one scalar dataset on a power-of-two axis.

    Selection and bank words are MSB first. Forward target and optional junk
    ports transfer owned clean work to the caller. Erasure consumes those
    matching signed banks. Scratch starts and ends zero. The measured erase
    is an executable dependency with a separately reported primitive cost,
    while the upstream adjoint supplies an abstract model-only leaf.
    """
    if (
        len(table.shape) != 1
        or len(table.datasets) != 1
        or table.shape[0] < 2
        or table.shape[0] & (table.shape[0] - 1)
    ):
        raise ValueError("Scalar clean lookup requires one power-of-two data axis")
    if type(word_width) is not int or word_width < 1 or type(erase) is not bool:
        raise ValueError(
            "Clean lookup requires a positive word width and boolean direction"
        )
    selection_width = (table.shape[0] - 1).bit_length()
    signature = qroam_clean_signature(table.shape, (word_width,), table.log_block_sizes)
    if erase:
        signature = signature.adjoint()
    builder = ComponentProgramBuilder(signature)
    selection = builder.inputs["selection"]
    assert isinstance(selection, RegisterBinding)
    target_register = (
        signature.get_left("target0_") if erase else signature.get_right("target0_")
    )
    dtype = target_register.dtype
    cells = table.block_shape[0]
    if erase:
        target = builder.inputs["target0_"]
        assert isinstance(target, RegisterBinding)
        junk = (
            list(np.asarray(builder.inputs["junk_target0_"]).flat) if cells > 1 else []
        )
    else:
        junk = [builder.allocate(dtype=dtype) for _ in range(cells - 1)]
        target = builder.allocate(dtype=dtype)
    banks = [target, *junk]
    scratch_count = table.workspace_size(0, (selection_width,))
    scratch = [builder.allocate() for _ in range(scratch_count)]
    words = [selection, *banks, *scratch]
    bindings = np.concatenate([builder.split(word) for word in words])
    selector_wires = tuple(range(selection_width))
    offset = selection_width
    bank_wires = tuple(
        tuple(range(offset + i * word_width, offset + (i + 1) * word_width))
        for i in range(cells)
    )
    offset += cells * word_width
    work_wires = tuple(range(offset, offset + scratch_count))
    scratch_bits = tuple(range(scratch_count))
    args = (table, (), (selector_wires,), (bank_wires,), work_wires, scratch_bits)
    gates = (
        qroam_clean_measurement_erase_ir(
            *args, tuple(range(scratch_count, scratch_count + cells * word_width))
        )
        if erase
        else source_qroam_clean_ir(*args)
    )
    count = offset + scratch_count
    leaf = bit_gate_function(
        "lookup_bank_erase" if erase else "lookup_bank_load",
        (Register("bits", quantum_bit_type(), shape=(count,)),),
        CircuitIR(count, gates),
    )
    builder.define(leaf)
    bits = np.asarray(builder.call_dict(leaf.name, {"bits": bindings})["bits"])
    restored = []
    offset = 0
    for word in words:
        size = int(word.register.bitsize)
        restored.append(builder.join(bits[offset : offset + size], word.register.dtype))
        offset += size
    selection, *restored = restored
    banks, scratch = restored[:cells], restored[cells:]
    for word in scratch:
        builder.free(word)
    outputs: dict[str, object] = {"selection": selection}
    if erase:
        for word in banks:
            builder.free(word)
    else:
        outputs["target0_"] = banks[0]
        if cells > 1:
            outputs["junk_target0_"] = np.array(banks[1:])
    return builder.finalize(outputs)

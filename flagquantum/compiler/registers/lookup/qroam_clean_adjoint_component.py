# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 QROAMCleanAdjointWrapper.build_composite_bloq.

"""Preserve the source bank reshape around its explicitly supplied erase leaf."""

from math import prod

import numpy as np

from ....core.ir.program import FunctionRecord, ProgramRecord
from ....core.ir.program.signatures import Signature
from ..component_builder import BindingCollection, ComponentProgramBuilder


def qroam_clean_adjoint_wrapper_program(
    forward_signature: Signature,
    erase_leaf: FunctionRecord,
    block_shape: tuple[int, ...],
    *,
    name: str = "main",
) -> ProgramRecord:
    """Consume all loaded words through the source's single bank-shaped call.

    Bank dimensions must be concrete to reshape the port collections; word
    widths may remain symbolic. The caller supplies the true source erase
    declaration. This wrapper does not infer measurement gates or a price.
    """
    signature = forward_signature.adjoint()
    builder = ComponentProgramBuilder(signature, (erase_leaf,), name=name)
    values: dict[str, BindingCollection] = {
        register.name: builder.inputs[register.name]
        for register in signature
        if not register.name.startswith(("target", "junk_"))
    }
    for target in signature:
        if not target.name.startswith("target"):
            continue
        junk = (
            np.asarray(builder.inputs["junk_" + target.name]).flat
            if prod(block_shape) > 1
            else ()
        )
        values[target.name] = np.asarray(
            [builder.inputs[target.name], *junk], dtype=object
        ).reshape(block_shape)
    return builder.finalize(builder.call_dict(erase_leaf.name, values))

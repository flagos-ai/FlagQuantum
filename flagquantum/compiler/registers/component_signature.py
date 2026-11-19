# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 BloqBuilder.finalize.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/composite_bloq.py

"""Infer ad hoc return metadata without allocating or consuming register values."""

from collections.abc import Iterator, Mapping

import numpy as np

from ...core.ir.program.register_bindings import RegisterBinding
from ...core.ir.program.registers import Register, RegisterSide


def infer_output_registers(
    registers: tuple[Register, ...], bindings: Mapping[str, object]
) -> Iterator[Register]:
    """Yield undeclared RIGHT ports using the first supplied handle's dtype."""
    declared = {reg.name for reg in registers if reg.side & RegisterSide.RIGHT}
    for name, value in bindings.items():
        if name in declared:
            continue
        if isinstance(value, RegisterBinding):
            dtype, shape = value.register.dtype, ()
        else:
            array = np.asarray(value)
            first = array.reshape(-1)[0]
            if not isinstance(first, RegisterBinding):
                raise ValueError(
                    f"Cannot infer output register {name!r} from non-binding values"
                )
            dtype, shape = first.register.dtype, array.shape
        yield Register(name, dtype, shape, RegisterSide.RIGHT)

# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _DirtyOutOfPlaceMontgomeryModMulImpl.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic/mod_multiplication.py

"""Compose Montgomery windows, retained indices and the source final reduction."""

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.register_types import (
    montgomery_register_type,
    quantum_bit_type,
    unsigned_register_type,
)
from ....core.ir.program.signatures import Signature
from ..arithmetic.constant_addition_component import constant_addition_program
from ..comparison._component_gates import comparison_bit_functions
from ..comparison.constant_component import less_than_constant_program
from ..component_builder import BindingCollection, ComponentProgramBuilder
from .single_window_component import single_window_program


def _reduction(
    builder: ComponentProgramBuilder,
    target: BindingCollection,
    reduced: BindingCollection,
    width: int,
    modulus: int,
    adjoint: bool,
) -> tuple[BindingCollection, BindingCollection]:
    correction = constant_addition_program(
        unsigned_register_type(width), modulus, control_value=1, adjoint=adjoint
    )
    if adjoint:
        restored = builder.call_program(correction, {"ctrl": reduced, "x": target})
        target, reduced = restored["x"], restored["ctrl"]
    else:
        reduced = builder.call_dict("comparison_x", {"q": reduced})["q"]
    compared = builder.call_program(
        less_than_constant_program(width, modulus), {"x": target, "target": reduced}
    )
    target, reduced = compared["x"], compared["target"]
    if adjoint:
        reduced = builder.call_dict("comparison_x", {"q": reduced})["q"]
    else:
        corrected = builder.call_program(correction, {"ctrl": reduced, "x": target})
        target, reduced = corrected["x"], corrected["ctrl"]
    return target, reduced


def windowed_multiplication_program(
    width: int, window: int, modulus: int, *, adjoint: bool = False
) -> ProgramRecord:
    """Transform through product/index/reduced ports with source arithmetic.

    Inputs are MSB-first Montgomery words. Forward target, indices and reduced
    start zero. Window input slices reverse x's bit order. The carry prefix is
    clean at release on the source carry-safe domain. The final correction
    intentionally retains the original addition/callback disagreement.
    """
    step = single_window_program(width, window, modulus, adjoint=adjoint)
    if modulus <= 1 or 2 * modulus - 1 >= 1 << width:
        raise ValueError(
            "Windowed Montgomery product requires an odd carry-safe modulus"
        )
    dtype = montgomery_register_type(width, modulus)
    builder = ComponentProgramBuilder(
        Signature.build_from_dtypes(
            x=dtype,
            y=dtype,
            target=dtype,
            qrom_indices=montgomery_register_type(width, modulus),
            reduced=quantum_bit_type(),
        ),
        comparison_bit_functions(),
    )
    x, y = builder.inputs["x"], builder.inputs["y"]
    target, indices, reduced = (
        builder.inputs["target"],
        builder.inputs["qrom_indices"],
        builder.inputs["reduced"],
    )
    if adjoint:
        target, reduced = _reduction(builder, target, reduced, width, modulus, True)
    assert isinstance(x, RegisterBinding) and isinstance(target, RegisterBinding)
    assert isinstance(indices, RegisterBinding)
    xs = builder.split(x)[::-1]
    target_array = np.concatenate(
        [builder.split(builder.allocate(window)), builder.split(target)]
    )
    index_array = builder.split(indices)
    positions = (
        reversed(range(0, width, window)) if adjoint else range(0, width, window)
    )
    for i in positions:
        result = builder.call_program(
            step,
            {
                "x": xs[i : i + window],
                "y": y,
                "target": target_array,
                "qrom_index": builder.join(index_array[i : i + window]),
            },
        )
        xs[i : i + window], y, target_array = (
            result["x"],
            result["y"],
            np.asarray(result["target"]),
        )
        value = result["qrom_index"]
        assert isinstance(value, RegisterBinding)
        index_array[i : i + window] = builder.split(value)
    builder.free(builder.join(target_array[:-width]))
    x = builder.join(xs[::-1])
    indices = builder.join(index_array)
    target = builder.join(target_array[-width:])
    if not adjoint:
        target, reduced = _reduction(builder, target, reduced, width, modulus, False)
    return builder.finalize(
        {"x": x, "y": y, "target": target, "qrom_indices": indices, "reduced": reduced}
    )

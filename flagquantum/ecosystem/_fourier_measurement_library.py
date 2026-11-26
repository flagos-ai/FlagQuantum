# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Declaration follows Qualtran v0.7.0 cryptography/_factoring_shims.py.

"""Share original Fourier-measurement declarations across factoring algorithms."""

from collections.abc import Mapping

from ..algorithms.primitives.measure_qft import source_measure_qft_signature
from ..compiler.formats.legacy_component_resolution import (
    ComponentFactory,
    ResolvedArgument,
)
from ..compiler.formats.legacy_modular_addition_factories import _integer
from ..core.ir.program import FunctionRecord
from ..core.ir.program.registers import Register
from ..errors import SerializationError

_MEASURE = "qualtran.bloqs.cryptography._factoring_shims.MeasureQFT"


def _measure(
    name: str,
    registers: tuple[Register, ...] | None,
    arguments: Mapping[str, ResolvedArgument],
) -> FunctionRecord:
    if set(arguments) != {"n"}:
        raise SerializationError("MeasureQFT requires its original n field")
    ports = tuple(source_measure_qft_signature(_integer(arguments, "n")))
    if registers is not None and registers != ports:
        raise SerializationError("Invalid original MeasureQFT register signature")
    return FunctionRecord(name, (), registers=ports)


def fourier_measurement_factories() -> dict[str, ComponentFactory]:
    """Keep the original unused-CBit failure separate from explicit readout."""
    return {_MEASURE: _measure}

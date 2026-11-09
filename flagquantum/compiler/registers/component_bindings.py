# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Binding traversal adapted from Qualtran v0.7.0 _process_soquets.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/composite_bloq.py

"""Source-ordered validation of indexed component port bindings.

The caller owns the callback's connection and consumption effects. This helper
retains their order relative to array indexing and dtype compatibility failures.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

import numpy as np

from ...core.ir.program.register_bindings import RegisterBinding
from ...core.ir.program.register_compatibility import check_register_types_consistent
from ...core.ir.program.registers import Register


class RegisterBindingError(ValueError):
    """A required component port is missing, surplus or type-inconsistent."""


def process_register_bindings(
    registers: Iterable[Register],
    bindings: Mapping[str, object],
    context: str,
    visit: Callable[[RegisterBinding, Register, tuple[int, ...]], None],
) -> None:
    """Visit indexed bindings in signature order with source failure timing.

    The callback runs before dtype consistency is checked. Earlier callback
    effects remain observable if a later index, type or surplus-name check fails.
    Arrays use NumPy indexing without additional broadcasting or shape guards.
    """
    unchecked_names = set(bindings)
    for register in registers:
        try:
            elements = np.asarray(bindings[register.name])
        except KeyError:
            raise RegisterBindingError(
                f"{context} requires a binding named `{register.name}`."
            ) from None
        unchecked_names.remove(register.name)
        for index in register.all_idxs():
            binding = elements[index]
            assert isinstance(binding, RegisterBinding), binding
            visit(binding, register, index)
            if not check_register_types_consistent(
                binding.register.dtype, register.dtype
            ):
                detail = (
                    f"{binding.register.name}: {binding.register.dtype} vs "
                    f"{register.name}: {register.dtype}"
                )
                raise RegisterBindingError(
                    f"{context} register dtypes are not consistent {detail}."
                ) from None
    if unchecked_names:
        raise RegisterBindingError(
            f"{context} does not accept bindings: {unchecked_names}."
        ) from None

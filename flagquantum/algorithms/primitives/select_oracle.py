# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 multiplexers/select_base.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/multiplexers/select_base.py

"""Internal ordered register roles for SELECT oracles."""

from abc import ABC, abstractmethod
from functools import cached_property

from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature


class SelectOracle(ABC):
    """Declare control, selection and target descriptors in source order."""

    @property
    @abstractmethod
    def control_registers(self) -> tuple[Register, ...]:
        """Return the ordered control descriptors."""

    @property
    @abstractmethod
    def selection_registers(self) -> tuple[Register, ...]:
        """Return the ordered address descriptors."""

    @property
    @abstractmethod
    def target_registers(self) -> tuple[Register, ...]:
        """Return the ordered system descriptors."""

    @cached_property
    def signature(self) -> Signature:
        """Preserve control, selection and target groups, including their sides."""
        return Signature(
            (*self.control_registers, *self.selection_registers, *self.target_registers)
        )

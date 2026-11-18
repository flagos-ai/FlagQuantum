# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 state_preparation/prepare_base.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/state_preparation/prepare_base.py

"""Internal register and normalization contract for PREPARE oracles."""

from abc import ABC, abstractmethod
from functools import cached_property

from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicReal


class PrepareOracle(ABC):
    """Declare selection ports, directed auxiliary registers, and optional L1 norm."""

    @property
    @abstractmethod
    def selection_registers(self) -> tuple[Register, ...]:
        """Return the ordered selection descriptors."""

    @cached_property
    def junk_registers(self) -> tuple[Register, ...]:
        """Return auxiliary descriptors with their original sides, empty by default."""
        return ()

    @cached_property
    def signature(self) -> Signature:
        """Preserve each selection descriptor followed by each junk descriptor."""
        return Signature((*self.selection_registers, *self.junk_registers))

    @property
    def l1_norm_of_coeffs(self) -> SymbolicReal:
        """Return the sum of absolute coefficients when provided by the oracle."""
        raise NotImplementedError

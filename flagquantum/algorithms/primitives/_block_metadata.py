# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 BlockEncoding.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/block_encoding/block_encoding_base.py

"""Structural block declarations and the concrete binding specialization."""

from __future__ import annotations

from typing import Protocol, TypeVar

from ...compiler.registers.layout.packed_block_component import packed_block_signature
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt, SymbolicReal
from .black_box_prepare import BlackBoxPrepare

_Width = TypeVar("_Width", bound=SymbolicInt, covariant=True)
_Signal = TypeVar("_Signal", covariant=True)


class _BlockDeclaration(Protocol[_Width, _Signal]):
    """The source six properties without a physical execution requirement."""

    @property
    def system_bitsize(self) -> _Width: ...

    @property
    def ancilla_bitsize(self) -> _Width: ...

    @property
    def resource_bitsize(self) -> _Width: ...

    @property
    def alpha(self) -> SymbolicReal: ...

    @property
    def epsilon(self) -> SymbolicReal: ...

    @property
    def signal_state(self) -> _Signal: ...


class _BlockMetadata(_BlockDeclaration[int, BlackBoxPrepare], Protocol):
    """Concrete widths and a bound signal preparation for current emitters."""


def _packed_block_signature(
    system: SymbolicInt, ancilla: SymbolicInt, resource: SymbolicInt
) -> Signature:
    """Declare ordered opaque ports, omitting the source's zero widths."""
    return packed_block_signature(system, ancilla, resource)

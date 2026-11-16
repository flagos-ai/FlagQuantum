# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 for_testing/atom.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/for_testing/atom.py

"""Small atomic fixtures for source tensor and resource parity tests."""

from dataclasses import dataclass, replace
from typing import Literal

import torch

from flagquantum.compiler._component_trace.resource_model import ComponentCost, GateCost
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.gate_matrix import gate_matrix
from flagquantum.simulation.tensor_network.register_tensor import (
    register_tensor_from_unitary,
)

AtomKind = Literal["x", "cx", "identity"]


@dataclass(frozen=True)
class AtomicRegisterFixture:
    """Describe an atomic test leaf using the ordinary gate and tensor owners."""

    kind: AtomKind
    tag: str | None = None
    is_adjoint: bool = False

    @property
    def signature(self) -> Signature:
        return (
            Signature.build(ctrl=1, target=1)
            if self.kind == "cx"
            else Signature.build(q=1)
        )

    @property
    def instructions(self) -> tuple[Instruction, ...]:
        if self.kind == "identity":
            return ()
        width = 2 if self.kind == "cx" else 1
        return (Instruction(self.kind, tuple(range(width))),)

    @property
    def label(self) -> str:
        label = {"x": "Atom", "cx": "TwoBitOp", "identity": "GWRAtom"}[self.kind]
        if not self.tag:
            return label
        tag = repr(self.tag) if self.kind == "x" else self.tag
        return f"{label}({tag})"

    @property
    def declared_cost(self) -> ComponentCost | None:
        if self.kind == "cx":
            return None
        return ComponentCost(self.label, 1, GateCost(t=100))

    def adjoint(self) -> "AtomicRegisterFixture":
        return replace(self, is_adjoint=not self.is_adjoint)

    def tensor(self) -> torch.Tensor:
        instructions = self.instructions
        matrix = (
            gate_matrix(instructions[0], bsz=1, device="cpu", dtype=torch.complex128)
            if instructions
            else torch.eye(2, dtype=torch.complex128)
        )
        return register_tensor_from_unitary(
            matrix.reshape(2 ** len(self.signature), -1), self.signature
        )

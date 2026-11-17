# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Always in Qualtran v0.7.0.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/bookkeeping/always.py

"""Explicitly bypass inherited controls for a component within a composition."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Generic, ParamSpec, TypeVar

from ....core.ir import Instruction
from ....core.ir.program.signatures import Signature
from ..._component_trace.resource_model import ComponentCost

Parameters = ParamSpec("Parameters")
Controls = TypeVar("Controls", bound=tuple[object, ...])


@dataclass(frozen=True)
class Always(Generic[Parameters]):
    """Use a component's own forward and inverse emitters under any control."""

    forward: Callable[Parameters, tuple[Instruction, ...]]
    inverse: Callable[Parameters, tuple[Instruction, ...]]
    signature: Signature | None = None

    def controlled(self, controls: Controls) -> tuple[Controls, "Always[Parameters]"]:
        """Pass control wires to the caller unchanged, without gating this child."""
        return controls, self

    def adjoint(self) -> "Always[Parameters]":
        """Reverse the wrapped component using its declared inverse."""
        signature = None if self.signature is None else self.signature.adjoint()
        return Always(self.inverse, self.forward, signature)

    def emit(
        self, *args: Parameters.args, **kwargs: Parameters.kwargs
    ) -> tuple[Instruction, ...]:
        return self.forward(*args, **kwargs)


def always_component_cost(
    component: ComponentCost | Callable[[], ComponentCost],
) -> ComponentCost:
    """Delegate an independent cost declaration, including any child error."""
    cost = component() if callable(component) else component
    return replace(cost, name=f"Always({cost.name})")

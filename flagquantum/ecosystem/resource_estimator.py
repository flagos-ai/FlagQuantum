# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 msft_resource_estimator_interop.ipynb at
# bcb17bb25360ed68ce3bc084645db3e929799389.

"""Translate declared logical resources to Microsoft's known-count schema."""

from typing import TypedDict

from ..compiler._component_trace.gate_budget import legacy_gate_budget
from ..compiler._component_trace.resource_model import ModeledResources


class MicrosoftLogicalCounts(TypedDict):
    """The four source fields accepted by the optional Q# resource estimator."""

    numQubits: int
    tCount: int
    rotationCount: int
    rotationDepth: int


def microsoft_logical_counts(
    logical: ModeledResources, *, rotation_depth: int | None = None
) -> MicrosoftLogicalCounts:
    """Export concrete counts without importing or invoking an external SDK.

    The caller owns the declared qubit scope. Supply peak logical width to
    include work qubits. The source notebook instead uses signature width.
    Rotation depth defaults to rotation count, as in that notebook. The legacy
    T convention folds Toffoli, CSwap and forward AND as four, seven and four T.
    """
    budget = legacy_gate_budget(logical.gates)
    width, ts, rotations = logical.peak_logical_qubits, budget["t"], budget["rotations"]
    depth = rotations if rotation_depth is None else rotation_depth
    if (
        type(width) is not int
        or type(ts) is not int
        or type(rotations) is not int
        or type(depth) is not int
    ):
        raise ValueError("Estimator counts require concrete nonnegative integers")
    if min(width, ts, rotations, depth) < 0:
        raise ValueError("Estimator counts require concrete nonnegative integers")
    return MicrosoftLogicalCounts(
        numQubits=width, tCount=ts, rotationCount=rotations, rotationDepth=depth
    )

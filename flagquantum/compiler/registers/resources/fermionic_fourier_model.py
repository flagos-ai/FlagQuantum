# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 TwoBitFFFT.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/qft/two_bit_ffft.py

"""Original fermionic Fourier calls and independent gate classification."""

from ..._component_trace.resource_classification import classify_resource_gate
from ..._component_trace.resource_generalization import (
    CliffordCall,
    PhaseCall,
    RotationCall,
)
from ..._component_trace.resource_model import ComponentCost, GateCost
from ..._component_trace.symbolic_math import SymbolicReal
from ..phase.fermionic_fourier import _fourier_angle


def source_two_bit_fermionic_fourier_calls(
    k: int, n: int, *, epsilon: SymbolicReal = 1e-10
) -> dict[RotationCall | PhaseCall | CliffordCall, int]:
    """Keep the source's precision and its combined T/adjoint-T declaration."""
    return {
        RotationCall("rz", _fourier_angle(k, n), precision=epsilon): 1,
        PhaseCall("s"): 3,
        CliffordCall("h", 1): 6,
        PhaseCall("t"): 2,
        CliffordCall("cx", 2): 3,
    }


def source_two_bit_fermionic_fourier_cost(k: int, n: int) -> ComponentCost:
    """Count the locked Rz, three S, six H, two T and three CNOT calls."""
    phase = classify_resource_gate(RotationCall("rz", _fourier_angle(k, n)))
    return ComponentCost(
        "TwoBitFFFT",
        2,
        GateCost(
            t=2 + int(phase == "t"),
            clifford=12 + int(phase == "clifford"),
            rotation=int(phase == "rotation"),
        ),
    )

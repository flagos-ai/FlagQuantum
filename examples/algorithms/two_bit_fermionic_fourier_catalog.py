# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Qualtran v0.7.0 TwoBitFFFT notebook.

"""The original two-bit fermionic Fourier instance and declared graph."""

from flagquantum.algorithms.primitives.fermionic_fourier import (
    source_two_bit_fermionic_fourier_calls,
)
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.resource_generalization import RotationCall

TWO_BIT_FOURIER_EXAMPLE = (2, 3)


def two_bit_fermionic_fourier_graph(k: int, n: int) -> SymbolicCallGraph[str]:
    """Display the source F(k, n) identity and precision-bearing rotation."""
    root = f"F({k}, {n})"
    calls = {
        f"Rz({gate.parameter}, epsilon={gate.precision})"
        if isinstance(gate, RotationCall)
        else gate.opcode.upper(): count
        for gate, count in source_two_bit_fermionic_fourier_calls(k, n).items()
    }
    return symbolic_call_graph(root, {root: calls}, max_depth=1)

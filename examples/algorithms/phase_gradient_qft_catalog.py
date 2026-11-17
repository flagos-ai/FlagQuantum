# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Qualtran v0.7.0 phase-gradient QFT notebook.

"""Native presentation for a Fourier transform driven by a gradient register."""

from flagquantum.algorithms.primitives.qft_phase_gradient_resource_model import (
    PhaseGradientQFTCall,
    PhaseGradientQFTNode,
    ProductAccumulationCall,
)


def phase_gradient_qft_label(node: PhaseGradientQFTNode) -> str:
    """Give the source's mathematical callees short graphical labels."""
    if isinstance(node, PhaseGradientQFTCall):
        return f"Phase-gradient QFT ({node.width})"
    if isinstance(node, ProductAccumulationCall):
        return f"Product accumulation ({node.left}, {node.right}; out {node.result})"
    return "Hadamard" if node.opcode == "h" else "Swap"

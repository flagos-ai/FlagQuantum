# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Register-group presentation adapted from Qualtran v0.7.0 graphviz.ipynb.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/drawing/graphviz.ipynb

"""Draw standalone signature metadata without inventing executable bodies."""

from ..core.ir.program.registers import RegisterSide
from ..core.ir.program.signatures import Signature
from .component_score import _label
from .register_graph import export_register_graph_dot
from .register_score import RegisterScore, RegisterStep, layout_register_score


def signature_register_score(signature: Signature, name: str) -> RegisterScore:
    """Project positive concrete quantum words into one display-only instance.

    Separate LEFT/RIGHT ports with the same name retain different lifetimes,
    even when their shape or width differs. THRU words share their display lane.
    Output-only names are prefixed with ``out:`` until unique. Metadata drawing
    has no numerical meaning, allocation promise or implied unitary completion.
    """
    reserved = {
        _label(register.name, index)
        for register in signature.lefts()
        for index in register.all_idxs()
    }
    widths: dict[str, int] = {}
    inputs = []
    outputs = []
    for register in signature:
        if (
            type(register.bitsize) is not int
            or register.bitsize < 1
            or register.dtype.classical_bits
        ):
            raise ValueError("signature graphs require positive concrete quantum words")
        for index in register.all_idxs():
            label = _label(register.name, index)
            if register.side == RegisterSide.RIGHT:
                while label in widths or label in reserved:
                    label = "out:" + label
            widths[label] = register.bitsize
            if register.side != RegisterSide.RIGHT:
                inputs.append(label)
            if register.side != RegisterSide.LEFT:
                outputs.append(label)
    return layout_register_score(
        widths, inputs, (RegisterStep(name, tuple(inputs), tuple(outputs)),)
    )


def export_signature_graph_dot(signature: Signature, name: str) -> str:
    """Render standalone metadata through the shared register graph renderer."""
    return export_register_graph_dot(signature_register_score(signature, name))

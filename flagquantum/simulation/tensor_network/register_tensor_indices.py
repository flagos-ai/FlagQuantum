# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Naming adapted from Qualtran v0.7.0 _get_outer_indices/_get_outer_superindices.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/tensor/_quimb.py

"""Describe external register tensor indices without renaming execution edges."""

from collections.abc import Mapping

from ...core.ir.program.signatures import Signature
from .conjugate_pairs import conjugate_pair_labels
from .register_tensor_binding import register_tensor_label_groups


def register_tensor_outer_indices(
    signature: Signature,
    incoming: Mapping[str, tuple[tuple[int, ...], ...]],
    outgoing: Mapping[str, tuple[tuple[int, ...], ...]],
    *,
    superoperator: bool = False,
    friendly: bool = False,
) -> dict[int, tuple[str, tuple[int, ...], int, str] | str]:
    """Map external integer edges to source-compatible tuples or display names.

    Bindings contain original undoubled labels, as in the existing register
    binder. Tuple values are (register name, array index, bit index, group).
    Groups are l/r for pure tensors or lf/lb/rf/rb for doubled tensors. Friendly
    values use `name{index}_{bit}{group}`, omitting an empty scalar index.

    This is a diagnostic side table: no nodes, data or contraction labels change.
    The caller identifies external ports; topology and discard are not inferred.
    Existing binding validation applies, and repeated external labels raise
    ValueError because one edge cannot describe distinct exposed endpoints.
    Display names need not be unique and must not be used as execution labels.
    """
    if type(superoperator) is not bool or type(friendly) is not bool:
        raise ValueError("index naming modes require built-in booleans")
    register_tensor_label_groups(signature, incoming, outgoing)
    result: dict[int, tuple[str, tuple[int, ...], int, str] | str] = {}
    for side, registers, bindings in (
        ("l", signature.lefts(), incoming),
        ("r", signature.rights(), outgoing),
    ):
        for register in registers:
            if not register.bitsize or not tuple(register.all_idxs()):
                continue
            for index, word in zip(
                register.all_idxs(), bindings[register.name], strict=True
            ):
                directions = (False, True) if superoperator else (False,)
                for backward in directions:
                    labels = (
                        conjugate_pair_labels(word, backward=backward)
                        if superoperator
                        else word
                    )
                    group = side + (("b" if backward else "f") if superoperator else "")
                    for bit, label in enumerate(labels):
                        if label in result:
                            raise ValueError("external tensor indices must be distinct")
                        result[label] = (
                            f"{register.name}{index if index else ''}_{bit}{group}"
                            if friendly
                            else (register.name, index, bit, group)
                        )
    return result

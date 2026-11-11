# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Axis ordering adapted from Qualtran v0.7.0 _dense.py.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/simulation/tensor/_dense.py

"""Order declared tensor ports and actual external register indices."""

from collections import defaultdict
from collections.abc import Iterable, Mapping

from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from .register_tensor import register_tensor_shape


def _port_labels(
    registers: Iterable[Register],
    bindings: Mapping[str, tuple[tuple[int, ...], ...]],
) -> tuple[int, ...]:
    labels: list[int] = []
    for register in registers:
        width = register.bitsize
        assert isinstance(width, int)
        count = len(tuple(register.all_idxs()))
        # The source never looks up connections when no bits are visited.
        if not width or not count:
            continue
        words = bindings[register.name]
        if len(words) != count or any(len(word) != width for word in words):
            raise ValueError("tensor bindings must match register element/bit counts")
        for word in words:
            for label in word:
                if type(label) is not int or label < 0:
                    raise ValueError("tensor labels must be nonnegative integers")
                labels.append(label)
    return tuple(labels)


def register_tensor_label_groups(
    signature: Signature,
    incoming: Mapping[str, tuple[tuple[int, ...], ...]],
    outgoing: Mapping[str, tuple[tuple[int, ...], ...]],
) -> tuple[tuple[int, ...], ...]:
    """Order nonempty output and input bit groups as the pinned dense protocol.

    Each port maps to one tuple of bit labels per row-major word element, even
    for a scalar word. Labels refer to the caller's existing tensor graph, not
    physical wire ownership. Zero-width/empty ports require no mapping. Extra
    keys are ignored, while missing live ports raise KeyError. No labels are
    allocated, renamed or deduplicated here.
    """
    register_tensor_shape(signature)
    right = _port_labels(signature.rights(), outgoing)
    left = _port_labels(signature.lefts(), incoming)
    return tuple(group for group in (right, left) if group)


def group_register_outer_indices(
    signature: Signature,
    outer_indices: Mapping[tuple[str, tuple[int, ...], int, str], int],
    *,
    superoperator: bool = False,
) -> tuple[tuple[int, ...], ...]:
    """Order existing external edges using the pinned source's complete group rule.

    The mapping describes actual exposed edges, not all declared ports. Values
    are existing native integer labels. Extra descriptors are ignored; a missing
    live descriptor raises KeyError. Source traversal checks left groups before
    right groups, while the result places right groups first. Classical bits
    contribute axes, and empty groups are omitted. No edges are allocated.
    """
    by_group: dict[str, dict[tuple[str, tuple[int, ...], int], int]] = defaultdict(dict)
    for (name, index, bit, group), label in outer_indices.items():
        by_group[group][name, index, bit] = label
    groups: dict[str, tuple[int, ...]] = {}
    traversal = ("lf", "lb", "rf", "rb") if superoperator else ("l", "r")
    for group in traversal:
        registers = signature.lefts() if group.startswith("l") else signature.rights()
        groups[group] = tuple(
            by_group[group][register.name, index, bit]
            for register in registers
            for index in register.all_idxs()
            for bit in range(register.dtype.num_bits)
        )
    order = ("rf", "rb", "lf", "lb") if superoperator else ("r", "l")
    return tuple(groups[group] for group in order if groups[group])

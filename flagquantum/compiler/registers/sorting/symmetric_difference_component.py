# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Qualtran v0.7.0 SymmetricDifference supplies the signature and cost graph.
# The frozen root has no executable decomposition.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/lists/symmetric_difference.py

"""Compose the set oracle from owned merge, count and compaction components."""

import numpy as np
from numpy.typing import NDArray

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.registers import RegisterType
from ..bits.bitwise_component import bitwise_word_program
from ..bits.hamming_weight_component import hamming_weight_program
from ..comparison.equality_component import equality_constant_program, equality_program
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..control.multi_control_component import multi_control_x_program
from .bitonic_merge_component import bitonic_merge_program
from .bitonic_sort_component import bitonic_sort_program
from .symmetric_difference import _layout
from .symmetric_difference_declaration import symmetric_difference_signature


def _copy_payloads(
    builder: ComponentProgramBuilder,
    keys: list[NDArray[np.object_]],
    words: NDArray[np.object_],
    width: int,
) -> NDArray[np.object_]:
    dtype = RegisterType("uint", width)
    for i, value in enumerate(words):
        assert isinstance(value, RegisterBinding)
        original_dtype = value.register.dtype
        if original_dtype != dtype:
            value = builder.join(builder.split(value), dtype)
        copied = builder.call_program(
            bitwise_word_program(dtype, "xor"),
            {"x": value, "y": builder.join(keys[i][-width:], dtype)},
        )
        source = copied["x"]
        assert isinstance(source, RegisterBinding)
        words[i] = (
            builder.join(builder.split(source), original_dtype)
            if original_dtype != dtype
            else source
        )
        payload = copied["y"]
        assert isinstance(payload, RegisterBinding)
        keys[i][-width:] = builder.split(payload)
    return words


def _bias_payload_signs(
    builder: ComponentProgramBuilder, keys: list[NDArray[np.object_]], width: int
) -> None:
    for key in keys:
        key[-width] = builder.call_program(
            multi_control_x_program(()), {"target": key[-width]}
        )["target"]


def _compare_adjacent(
    builder: ComponentProgramBuilder,
    keys: list[NDArray[np.object_]],
    flags: NDArray[np.object_],
    width: int,
) -> NDArray[np.object_]:
    dtype = RegisterType("uint", width)
    for i in range(len(flags)):
        compared = builder.call_program(
            equality_program(dtype),
            {
                "x": builder.join(keys[i][-width:], dtype),
                "y": builder.join(keys[i + 1][-width:], dtype),
                "target": flags[i],
            },
        )
        a, b = compared["x"], compared["y"]
        assert isinstance(a, RegisterBinding) and isinstance(b, RegisterBinding)
        keys[i][-width:], keys[i + 1][-width:] = builder.split(a), builder.split(b)
        flags[i] = compared["target"]
    return flags


def _tag_duplicates(
    builder: ComponentProgramBuilder,
    keys: list[NDArray[np.object_]],
    flags: NDArray[np.object_],
    offset: int,
) -> NDArray[np.object_]:
    for i in range(len(flags)):
        for j in (i, i + 1):
            tagged = builder.call_program(
                multi_control_x_program((1,)),
                {"controls": flags[i : i + 1], "target": keys[j][offset]},
            )
            flags[i] = np.asarray(tagged["controls"])[0]
            keys[j][offset] = tagged["target"]
    return flags


def _compact(
    builder: ComponentProgramBuilder,
    keys: list[NDArray[np.object_]],
    history: BindingCollection | None = None,
) -> tuple[list[NDArray[np.object_]], BindingCollection | None]:
    width = len(keys[0])
    words = np.array(
        [builder.join(key, RegisterType("uint", width)) for key in keys], dtype=object
    )
    inputs: dict[str, BindingCollection] = {"xs": words}
    if history is not None:
        inputs["junk"] = history
    result = builder.call_program(
        bitonic_sort_program(len(keys), width, adjoint=history is not None), inputs
    )
    return [builder.split(key) for key in np.asarray(result["xs"])], result.get("junk")


def symmetric_difference_program(
    left: int, right: int, width: int, size: int, *, signed: bool = False
) -> ProgramRecord:
    """XOR ascending set difference and cardinality match into occupied targets.

    All words are MSB first. S and T must each be ascending distinct sets.
    The root owns padded copies, equality flags and the success bit. Shared
    merge/sort/Hamming histories are consumed by exact adjoints. All work
    returns clean. This is the existing native paper construction, since the
    frozen source supplies no root gates. Invalid shapes raise ValueError.
    Signed words retain two's-complement order without biasing occupied outputs.
    """
    layout = _layout(left, right, width, size)
    if type(signed) is not bool:
        raise ValueError("Set difference signed flag requires a boolean")
    dtype = RegisterType("signed" if signed else "uint", width)
    builder = ComponentProgramBuilder(
        symmetric_difference_signature(left, right, size, dtype)
    )
    s, t, output = (
        np.asarray(builder.inputs[name]).copy() for name in ("S", "T", "diff")
    )
    flag = builder.inputs["flag"]
    if layout is None:
        if right == 0 and size == left:
            for i in range(left):
                copied = builder.call_program(
                    bitwise_word_program(dtype, "xor"), {"x": s[i], "y": output[i]}
                )
                s[i], output[i] = copied["x"], copied["y"]
            flag = builder.call_program(multi_control_x_program(()), {"target": flag})[
                "target"
            ]
        return builder.finalize({"S": s, "T": t, "diff": output, "flag": flag})

    key_width = width + layout.padding + layout.tagging
    keys = [
        builder.split(builder.allocate(dtype=RegisterType("uint", key_width)))
        for _ in range(2 * layout.half)
    ]
    active = [*keys[:left], *keys[layout.half : layout.half + right]]
    words = _copy_payloads(builder, active, np.concatenate((s, t)), width)
    if signed:
        _bias_payload_signs(builder, active, width)
    for i in (*range(left, layout.half), *range(layout.half + right, len(keys))):
        toggled = builder.call_program(
            multi_control_x_program(()), {"target": keys[i][0]}
        )
        keys[i][0] = toggled["target"]
    # Keep tag bits outside the merge key, ahead of the MSB-first payload.
    positions = [*range(layout.padding), *range(key_width - width, key_width)]
    merge_words = np.array(
        [
            builder.join(key[positions], RegisterType("uint", len(positions)))
            for key in keys
        ],
        dtype=object,
    )
    merged = builder.call_program(
        bitonic_merge_program(layout.half, len(positions)),
        {"xs": merge_words[: layout.half], "ys": merge_words[layout.half :]},
    )
    for key, value in zip(keys, np.asarray(merged["result"]), strict=True):
        key[positions] = builder.split(value)
    flags = np.array([builder.allocate() for _ in range(layout.adjacent)], dtype=object)
    flags = _compare_adjacent(builder, keys, flags, width)
    if layout.tagging:
        flags = _tag_duplicates(builder, keys, flags, layout.padding)
    counted = builder.call_program(
        hamming_weight_program(layout.adjacent),
        {"x": builder.join(flags, RegisterType("uint", layout.adjacent))},
    )
    success: BindingCollection = builder.allocate()
    predicate = equality_constant_program(layout.count, (left + right - size) // 2)
    matched = builder.call_program(predicate, {"x": counted["out"], "target": success})
    count, success = matched["x"], matched["target"]
    compact_history = None
    if layout.tagging:
        keys, compact_history = _compact(builder, keys)
    for i in range(size):
        target = output[i]
        assert isinstance(target, RegisterBinding)
        bits = builder.split(target)
        for j in range(width):
            copied = builder.call_program(
                multi_control_x_program((1, 1)),
                {
                    "controls": np.array([success, keys[i][-width + j]], dtype=object),
                    "target": bits[j],
                },
            )
            success, keys[i][-width + j] = np.asarray(copied["controls"])
            bits[j] = copied["target"]
        if signed:
            corrected = builder.call_program(
                multi_control_x_program((1,)),
                {"controls": np.array([success], dtype=object), "target": bits[0]},
            )
            success = np.asarray(corrected["controls"])[0]
            bits[0] = corrected["target"]
        output[i] = builder.join(bits, dtype)
    toggled = builder.call_program(
        multi_control_x_program((1,)),
        {"controls": np.array([success], dtype=object), "target": flag},
    )
    success, flag = np.asarray(toggled["controls"])[0], toggled["target"]
    if layout.tagging:
        keys, _ = _compact(builder, keys, compact_history)
    unmatched = builder.call_program(predicate, {"x": count, "target": success})
    success = unmatched["target"]
    assert isinstance(success, RegisterBinding)
    builder.free(success)
    restored = builder.call_program(
        hamming_weight_program(layout.adjacent, adjoint=True),
        {"x": counted["x"], "junk": counted["junk"], "out": unmatched["x"]},
    )
    restored_flags = restored["x"]
    assert isinstance(restored_flags, RegisterBinding)
    flags = builder.split(restored_flags)
    if layout.tagging:
        flags = _tag_duplicates(builder, keys, flags, layout.padding)
    flags = _compare_adjacent(builder, keys, flags, width)
    for value in flags:
        builder.free(value)
    merge_words = np.array(
        [
            builder.join(key[positions], RegisterType("uint", len(positions)))
            for key in keys
        ],
        dtype=object,
    )
    unmerged = builder.call_program(
        bitonic_merge_program(layout.half, len(positions), adjoint=True),
        {"result": merge_words, "junk": merged["junk"]},
    )
    for key, value in zip(
        keys,
        np.concatenate((np.asarray(unmerged["xs"]), np.asarray(unmerged["ys"]))),
        strict=True,
    ):
        key[positions] = builder.split(value)
    active = [*keys[:left], *keys[layout.half : layout.half + right]]
    if signed:
        _bias_payload_signs(builder, active, width)
    words = _copy_payloads(builder, active, words, width)
    for i in (*range(left, layout.half), *range(layout.half + right, len(keys))):
        toggled = builder.call_program(
            multi_control_x_program(()), {"target": keys[i][0]}
        )
        keys[i][0] = toggled["target"]
    for key in keys:
        builder.free(builder.join(key, RegisterType("uint", key_width)))
    return builder.finalize(
        {"S": words[:left], "T": words[left:], "diff": output, "flag": flag}
    )

# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Qualtran v0.7.0 SortInPlace supplies the signature and Lemma 4.12 contract.
# Its frozen source has no executable root decomposition.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/lists/sort_in_place.py

"""Compose distinct-word sorting from bitonic copy and permutation indexing."""

import numpy as np
from numpy.typing import NDArray

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.registers import RegisterType
from ..bits.bitwise_component import bitwise_word_program
from ..component_builder import ComponentProgramBuilder
from .bitonic_sort_component import bitonic_sort_program
from .in_place_sort import in_place_sort_requirements
from .in_place_sort_declaration import in_place_sort_signature
from .permutation_index_component import permutation_index_program


def _xor_words(
    builder: ComponentProgramBuilder,
    source: NDArray[np.object_],
    target: NDArray[np.object_],
    dtype: RegisterType,
) -> tuple[NDArray[np.object_], NDArray[np.object_]]:
    child = bitwise_word_program(dtype, "xor")
    for i in range(len(source)):
        copied = builder.call_program(child, {"x": source[i], "y": target[i]})
        source[i], target[i] = copied["x"], copied["y"]
    return source, target


def _copy_payload(
    builder: ComponentProgramBuilder,
    keys: NDArray[np.object_],
    words: NDArray[np.object_],
    width: int,
    *,
    into_keys: bool,
) -> tuple[NDArray[np.object_], NDArray[np.object_]]:
    word = RegisterType("uint", width)
    for i in range(len(words)):
        key = keys[i]
        assert isinstance(key, RegisterBinding)
        bits = builder.split(key)
        payload = builder.join(bits[-width:], word)
        source, target = (words[i], payload) if into_keys else (payload, words[i])
        copied = builder.call_program(
            bitwise_word_program(word, "xor"), {"x": source, "y": target}
        )
        updated = copied["y"] if into_keys else copied["x"]
        assert isinstance(updated, RegisterBinding)
        keys[i] = builder.join(
            [*bits[:-width], *builder.split(updated)], key.register.dtype
        )
        words[i] = copied["x"] if into_keys else copied["y"]
    return keys, words


def _sorted_copy(
    builder: ComponentProgramBuilder,
    words: NDArray[np.object_],
    table: NDArray[np.object_],
    width: int,
) -> tuple[NDArray[np.object_], NDArray[np.object_]]:
    count = len(words)
    padded = 1 << (count - 1).bit_length()
    key_width = width + int(padded != count)
    dtype = RegisterType("uint", key_width)
    keys = np.array(
        [builder.allocate(dtype=dtype) for _ in range(padded)], dtype=object
    )
    padding = bitwise_word_program(dtype, "xor_constant", pattern=1 << (key_width - 1))
    keys, words = _copy_payload(builder, keys, words, width, into_keys=True)
    for i in range(count, padded):
        keys[i] = builder.call_program(padding, {"x": keys[i]})["x"]
    sorted_values = builder.call_program(
        bitonic_sort_program(padded, key_width), {"xs": keys}
    )
    keys = np.asarray(sorted_values["xs"])
    keys, table = _copy_payload(builder, keys, table, width, into_keys=False)
    restored = builder.call_program(
        bitonic_sort_program(padded, key_width, adjoint=True),
        {"xs": keys, "junk": sorted_values["junk"]},
    )
    keys = np.asarray(restored["xs"])
    keys, words = _copy_payload(builder, keys, words, width, into_keys=True)
    for i in range(count, padded):
        keys[i] = builder.call_program(padding, {"x": keys[i]})["x"]
    for key in keys:
        builder.free(key)
    return words, table


def in_place_sort_program(
    count: int, width: int, *, adjoint: bool = False, signed: bool = False
) -> ProgramRecord:
    """Sort distinct integer THRU words with clean, bounded THRU permutation pi.

    Inputs and pi use MSB-first order. Forward requires pi zero. Inverse requires
    ascending distinct words and the corresponding original-to-sorted ranks,
    restores the original words and clears pi. The root owns its sorted copy,
    padded keys and child histories. All work returns clean. This composes the
    existing paper construction, since frozen SortInPlace supplies no root gates.
    Signed words use two's-complement order through reversible sign-bit bias.
    """
    rank_width, _, _ = in_place_sort_requirements(count, width)
    if type(adjoint) is not bool or type(signed) is not bool:
        raise ValueError("In-place sort direction and signed flag require booleans")
    if signed:
        return _signed_sort_program(count, width, adjoint=adjoint)
    dtype = RegisterType("uint", width)
    builder = ComponentProgramBuilder(in_place_sort_signature(count, dtype))
    words = np.asarray(builder.inputs["xs"]).copy()
    original_ranks = np.asarray(builder.inputs["pi"])
    if count == 1:
        return builder.finalize({"xs": words, "pi": original_ranks})
    rank = RegisterType("uint", rank_width)
    ranks = np.array(
        [builder.join(builder.split(value), rank) for value in original_ranks],
        dtype=object,
    )
    table = np.array(
        [builder.allocate(dtype=dtype) for _ in range(count)], dtype=object
    )
    if adjoint:
        words, table = _xor_words(builder, words, table, dtype)
        table, words = _xor_words(builder, table, words, dtype)
    else:
        words, table = _sorted_copy(builder, words, table, width)
    for operation in (("words", "rank") if adjoint else ("rank", "words")):
        result = builder.call_program(
            permutation_index_program(count, width, operation),
            {"table": table, "words": words, "ranks": ranks},
        )
        table, words, ranks = (
            np.asarray(result[name]) for name in ("table", "words", "ranks")
        )
    if adjoint:
        words, table = _sorted_copy(builder, words, table, width)
    else:
        table, words = _xor_words(builder, table, words, dtype)
        words, table = _xor_words(builder, words, table, dtype)
    for value in table:
        builder.free(value)
    pi = np.array(
        [
            builder.join(builder.split(value), original_ranks[i].register.dtype)
            for i, value in enumerate(ranks)
        ],
        dtype=object,
    )
    return builder.finalize({"xs": words, "pi": pi})


def _signed_sort_program(count: int, width: int, *, adjoint: bool) -> ProgramRecord:
    dtype = RegisterType("signed", width)
    unsigned = RegisterType("uint", width)
    builder = ComponentProgramBuilder(in_place_sort_signature(count, dtype))
    if count == 1:
        return builder.finalize(builder.inputs)
    mask = bitwise_word_program(unsigned, "xor_constant", pattern=1 << (width - 1))

    def bias(
        words: NDArray[np.object_], output_dtype: RegisterType
    ) -> NDArray[np.object_]:
        result = words.copy()
        for i, word in enumerate(words):
            raw = builder.join(builder.split(word), unsigned)
            changed = builder.call_program(mask, {"x": raw})["x"]
            assert isinstance(changed, RegisterBinding)
            result[i] = builder.join(builder.split(changed), output_dtype)
        return result

    words = bias(np.asarray(builder.inputs["xs"]), unsigned)
    sorted_values = builder.call_program(
        in_place_sort_program(count, width, adjoint=adjoint),
        {"xs": words, "pi": builder.inputs["pi"]},
    )
    restored = bias(np.asarray(sorted_values["xs"]), dtype)
    return builder.finalize({"xs": restored, "pi": sorted_values["pi"]})

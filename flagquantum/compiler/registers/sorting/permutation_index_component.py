# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Rank contract from Qualtran v0.7.0 SortInPlace and its cited Lemma 4.12.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/lists/sort_in_place.py

"""Store permutation indexing over shared equality and controlled XOR calls."""

from typing import Literal

import numpy as np

from ....core.ir.program import ProgramRecord
from ....core.ir.program.register_bindings import RegisterBinding
from ....core.ir.program.registers import Register, RegisterType
from ....core.ir.program.signatures import Signature
from ..bits.bitwise_component import bitwise_word_program
from ..bits.bitwise_component_control import controlled_bitwise_word_program
from ..comparison.equality_component import equality_constant_program, equality_program
from ..component_builder import BindingCollection, ComponentProgramBuilder
from ..control.multi_control_component import multi_control_x_program
from .permutation_index import permutation_index_requirements


def permutation_index_program(
    count: int, width: int, operation: Literal["rank", "words"]
) -> ProgramRecord:
    """XOR table ranks or indexed entries into caller-owned THRU arrays.

    Words and ranks are MSB first. Rank lookup requires distinct table entries
    and table membership for its logical permutation meaning. Indexed-word XOR
    preserves arbitrary ranks, selecting no entry for out-of-range bit patterns.
    All predicate and carry work returns clean. Singleton ranks are empty.
    """
    rank_width, _, _ = permutation_index_requirements(count, width)
    if operation not in {"rank", "words"}:
        raise ValueError("Permutation indexing operation must be rank or words")
    word = RegisterType("uint", width)
    rank = RegisterType("uint", rank_width) if rank_width else RegisterType("opaque", 0)
    builder = ComponentProgramBuilder(
        Signature(
            (
                Register("table", word, (count,)),
                Register("words", word, (count,)),
                Register("ranks", rank, (count,)),
            )
        )
    )
    table, words, ranks = (
        np.asarray(builder.inputs[name]).copy() for name in ("table", "words", "ranks")
    )
    if count == 1:
        if operation == "words":
            copied = builder.call_program(
                bitwise_word_program(word, "xor"), {"x": table[0], "y": words[0]}
            )
            table[0], words[0] = copied["x"], copied["y"]
        return builder.finalize({"table": table, "words": words, "ranks": ranks})
    flag: BindingCollection = builder.allocate()
    for j in range(count):
        for i in range(1 if operation == "rank" else 0, count):
            if operation == "rank":
                equal = equality_program(word)
                compared = builder.call_program(
                    equal, {"x": table[i], "y": words[j], "target": flag}
                )
                table[i], words[j], flag = (
                    compared["x"],
                    compared["y"],
                    compared["target"],
                )
                copied = builder.call_program(
                    controlled_bitwise_word_program(rank, "xor_constant", pattern=i),
                    {"ctrl": flag, "x": ranks[j]},
                )
                flag, ranks[j] = copied["ctrl"], copied["x"]
                restored = builder.call_program(
                    equal, {"x": table[i], "y": words[j], "target": flag}
                )
                table[i], words[j], flag = (
                    restored["x"],
                    restored["y"],
                    restored["target"],
                )
            else:
                equal = equality_constant_program(rank_width, i)
                compared = builder.call_program(equal, {"x": ranks[j], "target": flag})
                ranks[j], flag = compared["x"], compared["target"]
                entry, target = table[i], words[j]
                assert isinstance(entry, RegisterBinding) and isinstance(
                    target, RegisterBinding
                )
                source_bits, target_bits = builder.split(entry), builder.split(target)
                for k in range(width):
                    copied = builder.call_program(
                        multi_control_x_program((1, 1)),
                        {
                            "controls": np.array([flag, source_bits[k]], dtype=object),
                            "target": target_bits[k],
                        },
                    )
                    flag, source_bits[k] = np.asarray(copied["controls"])
                    target_bits[k] = copied["target"]
                table[i], words[j] = (
                    builder.join(source_bits, word),
                    builder.join(target_bits, word),
                )
                restored = builder.call_program(equal, {"x": ranks[j], "target": flag})
                ranks[j], flag = restored["x"], restored["target"]
    assert isinstance(flag, RegisterBinding)
    builder.free(flag)
    return builder.finalize({"table": table, "words": words, "ranks": ranks})

# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 AutoPartition.build_composite_bloq.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/bookkeeping/auto_partition.py

"""Compose numerical partition, child callback and restoration in source order."""

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass

from .classical_partition import (
    PartitionValuePort,
    partition_classical_value,
    unpartition_classical_values,
)


@dataclass(frozen=True)
class PackedValueBinding:
    """An already-bound external packed port and its native numerical codecs."""

    name: str
    ports: tuple[PartitionValuePort, ...]
    encode: Callable[[object], Sequence[int]]
    decode: Callable[[Sequence[int]], object]


def auto_partition_classical_values(
    values: Mapping[str, object],
    bindings: Sequence[PackedValueBinding],
    child_inputs: Collection[str],
    child: Callable[[dict[str, object]], dict[str, object]],
    *,
    left_only: bool = False,
) -> dict[str, object]:
    """Split packed inputs, retain unused values, call the child once and restore.

    Compiler owns binding/constructor legality, including unused names and
    left-only restrictions. This numerical composition follows the original
    builder with native callbacks; it is not a generic graph interpreter.
    """
    parts: dict[str, PackedValueBinding] = {}
    inputs: dict[str, object] = {}
    unused: dict[str, object] = {}
    for binding in bindings:
        parts[binding.name] = binding
        registers = partition_classical_value(
            values[binding.name], binding.encode, binding.ports
        )
        for name, value in registers.items():
            if name in child_inputs:
                inputs[name] = value
            else:
                unused[name] = value
    outputs = child(inputs)
    if left_only:
        return outputs
    restored: dict[str, object] = {}
    for name in values:
        binding = parts[name]
        restored[name] = unpartition_classical_values(
            {port.name: (outputs | unused)[port.name] for port in binding.ports},
            binding.ports,
            binding.decode,
        )
    return restored

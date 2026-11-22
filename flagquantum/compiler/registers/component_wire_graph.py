# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from locked Qualtran v0.7.0 cirq_interop/_cirq_to_bloq.py.

"""Rebuild typed component calls from finite, vendor-neutral wire identities."""

from collections.abc import Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

from ...core.ir.program import ProgramRecord
from ...core.ir.program.register_bindings import RegisterBinding
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.register_verifier import executable_register_width
from ...core.ir.program.registers import Register, RegisterSide, RegisterType
from ...core.ir.program.signatures import Signature
from .component_builder import BindingCollection, ComponentProgramBuilder
from .layout.component_cast import cast_word_function

WireCall = tuple[ProgramRecord, Mapping[str, tuple[int, ...]]]


class _WireWords:
    """Keep word insertion and source key dtypes through lazy Split/Join."""

    def __init__(self, builder: ComponentProgramBuilder) -> None:
        self.builder = builder
        self.words: dict[tuple[int, ...], tuple[RegisterType, RegisterBinding]] = {}
        self.bit = quantum_bit_type()

    def ensure(self, wires: tuple[int, ...], dtype: RegisterType) -> RegisterBinding:
        present = {wire for group in self.words for wire in group}
        missing = tuple(wire for wire in wires if wire not in present)
        if missing:
            key_dtype = (
                self.bit if len(missing) == 1 else RegisterType("opaque", len(missing))
            )
            self.words[missing] = key_dtype, self.builder.allocate(len(missing))
        if wires in self.words:
            return self.words[wires][1]
        requested = set(wires)
        split: dict[tuple[int, ...], tuple[RegisterType, RegisterBinding]] = {}
        for group, (kind, word) in self.words.items():
            if len(group) > 1 and requested.intersection(group):
                split.update(
                    ((wire,), (self.bit, bit))
                    for wire, bit in zip(group, self.builder.split(word), strict=True)
                )
            else:
                split[group] = kind, word
        self.words.clear()
        bits: dict[int, RegisterBinding] = {}
        for group, (kind, word) in split.items():
            if group and group[0] in requested:
                assert len(group) == 1
                if len(wires) > 1:
                    if kind != self.bit:
                        word = self.cast(word, kind, self.bit)
                    bits[group[0]] = word
                else:
                    assert kind == self.bit
                    if dtype != self.bit:
                        word = self.cast(word, self.bit, dtype)
                    self.words[group] = dtype, word
            else:
                self.words[group] = kind, word
        if bits:
            self.words[wires] = dtype, self.builder.join(
                np.array([bits[w] for w in wires], dtype=object), dtype
            )
        return self.words[wires][1]

    def cast(
        self, word: RegisterBinding, source: RegisterType, target: RegisterType
    ) -> RegisterBinding:
        function = cast_word_function("wire_cast", source, target)
        result = self.builder.call_program(
            ProgramRecord((function,)), {"reg": word}, entry=function.name
        )["reg"]
        assert isinstance(result, RegisterBinding)
        return result

    def gather(
        self, registers: Iterable[Register], ports: Mapping[str, tuple[int, ...]]
    ) -> dict[str, BindingCollection]:
        result: dict[str, BindingCollection] = {}
        for register in registers:
            groups = _groups(register, ports[register.name])
            values = np.empty(register.shape, dtype=object)
            for index in register.all_idxs():
                values[index] = self.ensure(tuple(groups[index]), register.dtype)
            result[register.name] = values if register.shape else values[()]
        return result

    def store(
        self,
        registers: Iterable[Register],
        ports: Mapping[str, tuple[int, ...]],
        values: Mapping[str, BindingCollection],
    ) -> None:
        for register in registers:
            groups = _groups(register, ports[register.name])
            bindings = np.asarray(values[register.name])
            for index in register.all_idxs():
                key = tuple(groups[index])
                kind = self.words[key][0] if key in self.words else register.dtype
                self.words[key] = kind, bindings[index]


def _groups(register: Register, wires: tuple[int, ...]) -> NDArray[np.object_]:
    width = executable_register_width(register.dtype)
    if (
        width < 1
        or any(type(wire) is not int or wire < 0 for wire in wires)
        or len(set(wires)) != len(wires)
    ):
        raise ValueError(
            "Wire component ports require distinct nonnegative wires and positive quantum width"
        )
    try:
        return np.asarray(wires, dtype=object).reshape((*register.shape, width))
    except ValueError as error:
        raise ValueError(
            f"Wire count does not match register {register.name}"
        ) from error


def wire_component_program(
    signature: Signature,
    inputs: Mapping[str, tuple[int, ...]],
    outputs: Mapping[str, tuple[int, ...]],
    calls: Iterable[WireCall],
    *,
    barriers: tuple[int, ...] = (),
) -> ProgramRecord:
    """Recreate earliest-layer calls and lazy typed views without an SDK.

    Physical identities span both sides of every call. RIGHT-only ports own new
    wires and LEFT-only ports consume them. Barrier positions start a new
    earliest-layer group, preserving explicitly nested source circuit moments.
    Existing component bodies continue
    owning gates and measurement feedback. Unreturned live words are freed.
    """
    calls = tuple(calls)
    if any(
        type(position) is not int or not 0 <= position < len(calls)
        for position in barriers
    ) or len(set(barriers)) != len(barriers):
        raise ValueError("Wire barriers require distinct valid call positions")
    builder = ComponentProgramBuilder(signature)
    words = _WireWords(builder)
    input_wires = [wire for group in inputs.values() for wire in group]
    if len(set(input_wires)) != len(input_wires):
        raise ValueError("Wire component inputs overlap")
    words.store(signature.lefts(), inputs, builder.inputs)
    layers: dict[int, list[WireCall]] = {}
    latest: dict[int, int] = {}
    floor = 0
    for position, (record, ports) in enumerate(calls):
        if position in barriers:
            floor = 1 + max(latest.values(), default=-1)
        root = record.functions[0]
        if root.registers is None:
            raise ValueError("Wire calls require typed component signatures")
        wires = tuple(
            wire
            for name in dict.fromkeys(register.name for register in root.registers)
            for wire in ports[name]
        )
        if len(set(wires)) != len(wires):
            raise ValueError("Wire call ports overlap")
        for register in root.registers:
            _groups(register, ports[register.name])
        layer = max(
            floor, 1 + max((latest.get(wire, -1) for wire in wires), default=-1)
        )
        layers.setdefault(layer, []).append((record, ports))
        latest.update((wire, layer) for wire in wires)
    for layer in sorted(layers):
        for record, ports in layers[layer]:
            root = record.functions[0]
            assert root.registers is not None
            child = Signature(root.registers)
            returned = builder.call_program(
                record, words.gather(child.lefts(), ports), entry=root.name
            )
            for register in child:
                if register.side == RegisterSide.LEFT:
                    for group in _groups(register, ports[register.name]).reshape(
                        -1, register.bitsize
                    ):
                        del words.words[tuple(group)]
            words.store(child.rights(), ports, returned)
    returned = words.gather(signature.rights(), outputs)
    live = {
        binding for value in returned.values() for binding in np.asarray(value).flat
    }
    for _, binding in words.words.values():
        if binding not in live:
            builder.free(binding)
    return builder.finalize(returned)

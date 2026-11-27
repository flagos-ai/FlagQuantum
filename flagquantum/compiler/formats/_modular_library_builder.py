# Copyright 2023-2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Ordered recipes adapted from Qualtran v0.7.0 mod_addition.py/mod_subtraction.py.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mod_arithmetic

"""Pair typed source calls with constructor metadata at the Compiler boundary."""

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from ...core.ir.program import FunctionRecord, ProgramRecord
from ...core.ir.program.register_bindings import RegisterBinding
from ...core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
)
from ...core.ir.program.registers import Register, RegisterSide, RegisterType
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt
from ..registers.component_builder import BindingCollection, ComponentProgramBuilder
from .legacy_component_arguments import (
    ComponentArgument,
    ComponentArgumentValue,
    ComponentReference,
    ControlSpecification,
)
from .legacy_component_export import component_entry_from_program
from .legacy_component_library import (
    ComponentDeclaration,
    ComponentLibrary,
    ComponentLibraryEntry,
)

_BIT = quantum_bit_type()
_BASIC = "qualtran.bloqs.basic_gates."
_BOOK = "qualtran.bloqs.bookkeeping."


class _ModularGraph:
    """Pair the existing typed call builder with its constructor declarations."""

    def __init__(self, root: ComponentDeclaration) -> None:
        assert root.registers is not None
        self.builder = ComponentProgramBuilder(Signature(root.registers))
        self.declarations = {"main": root}

    def declare(
        self,
        identity: str,
        ports: tuple[Register, ...],
        arguments: Mapping[str, ComponentArgumentValue] | None = None,
    ) -> str:
        name = f"modular_child_{len(self.declarations)}"
        self.declarations[name] = ComponentDeclaration(
            identity,
            tuple(
                ComponentArgument(key, value)
                for key, value in (arguments or {}).items()
            ),
            ports,
        )
        self.builder.define(FunctionRecord(name, (), registers=ports))
        return name

    def call(
        self,
        identity: str,
        ports: tuple[Register, ...],
        inputs: Mapping[str, object],
        arguments: Mapping[str, ComponentArgumentValue] | None = None,
    ) -> dict[str, BindingCollection]:
        return self.builder.call_dict(self.declare(identity, ports, arguments), inputs)

    def add_constant(
        self,
        dtype: RegisterType,
        value: SymbolicInt,
        inputs: Mapping[str, object],
        *,
        controlled: bool = False,
        adjoint: bool = False,
    ) -> dict[str, BindingCollection]:
        ports: tuple[Register, ...] = (Register("x", dtype),)
        if controlled and adjoint:
            raise ValueError("Constant declaration requires one wrapper direction")
        child = self.declare(
            "qualtran.bloqs.arithmetic.addition.AddK",
            ports,
            {"dtype": dtype, "k": value},
        )
        if controlled or adjoint:
            reference = ComponentReference(tuple(self.declarations).index(child))
            args: dict[str, ComponentArgumentValue] = {"subbloq": reference}
            wrapper = "qualtran._infra.adjoint.Adjoint"
            if controlled:
                wrapper = "qualtran._infra.controlled.Controlled"
                args["ctrl_spec"] = ControlSpecification(((_BIT, np.asarray(1)),))
                ports = (Register("ctrl", _BIT), *ports)
            child = self.declare(wrapper, ports, args)
        return self.builder.call_dict(child, inputs)

    def declare_adjoint(self, child: str) -> str:
        """Declare the source generic wrapper, without inferring a gate body."""
        declaration = self.declarations[child]
        assert declaration.registers is not None
        return self.declare(
            "qualtran._infra.adjoint.Adjoint",
            tuple(Signature(declaration.registers).adjoint()),
            {"subbloq": ComponentReference(tuple(self.declarations).index(child))},
        )

    def split(self, word: BindingCollection) -> NDArray[np.object_]:
        assert isinstance(word, RegisterBinding)
        dtype = word.register.dtype
        ports = (
            Register("reg", dtype, side=RegisterSide.LEFT),
            Register("reg", _BIT, (int(dtype.num_qubits),), RegisterSide.RIGHT),
        )
        value = self.call(
            _BOOK + "split.Split", ports, {"reg": word}, {"dtype": dtype}
        )["reg"]
        assert isinstance(value, np.ndarray)
        return value

    def join(self, bits: object, dtype: RegisterType) -> BindingCollection:
        ports = (
            Register("reg", _BIT, (int(dtype.num_qubits),), RegisterSide.LEFT),
            Register("reg", dtype, side=RegisterSide.RIGHT),
        )
        return self.call(_BOOK + "join.Join", ports, {"reg": bits}, {"dtype": dtype})[
            "reg"
        ]

    def allocate(self, dtype: RegisterType | None = None) -> BindingCollection:
        if dtype is None:
            dtype = opaque_register_type(1)
        return self.call(
            _BOOK + "allocate.Allocate",
            (Register("reg", dtype, side=RegisterSide.RIGHT),),
            {},
            {"dtype": dtype, "dirty": False},
        )["reg"]

    def free(self, bit: BindingCollection) -> None:
        assert isinstance(bit, RegisterBinding)
        dtype = bit.register.dtype
        self.call(
            _BOOK + "free.Free",
            (Register("reg", dtype, side=RegisterSide.LEFT),),
            {"reg": bit},
            {"dtype": dtype, "dirty": False},
        )

    def x(self, bit: BindingCollection) -> BindingCollection:
        return self.call(_BASIC + "x_basis.XGate", (Register("q", _BIT),), {"q": bit})[
            "q"
        ]

    def inline(
        self,
        record: ProgramRecord,
        declarations: Mapping[str, ComponentDeclaration],
        bindings: Mapping[str, object],
    ) -> dict[str, BindingCollection]:
        """Reuse typed inlining while rebasing its owned constructor references."""
        names: dict[str, str] = {}
        identifiers: dict[int, int] = {}
        for index, (name, declaration) in enumerate(declarations.items()):
            if name == "main":
                continue
            arguments = {
                argument.name: (
                    ComponentReference(identifiers[argument.value.identifier])
                    if isinstance(argument.value, ComponentReference)
                    else argument.value
                )
                for argument in declaration.arguments
            }
            assert declaration.registers is not None
            identifiers[index] = len(self.declarations)
            names[name] = self.declare(
                declaration.name, declaration.registers, arguments
            )
        functions = tuple(
            replace(
                function,
                name=names.get(function.name, function.name),
                blocks=tuple(
                    replace(
                        block,
                        operations=tuple(
                            (
                                replace(operation, callee=names[operation.callee])
                                if operation.callee is not None
                                else operation
                            )
                            for operation in block.operations
                        ),
                    )
                    for block in function.blocks
                ),
            )
            for function in record.functions
        )
        return self.builder.inline_calls(replace(record, functions=functions), bindings)

    def finish(self, outputs: Mapping[str, object]) -> ComponentLibrary:
        record = self.builder.finalize(outputs)
        identifiers = {name: i for i, name in enumerate(self.declarations)}
        root = component_entry_from_program(
            record,
            "main",
            identifiers,
            self.declarations["main"],
        )
        return ComponentLibrary(
            "modular logical calls",
            (
                root,
                *(
                    ComponentLibraryEntry(identifiers[name], declaration)
                    for name, declaration in self.declarations.items()
                    if name != "main"
                ),
            ),
        )

# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Call construction adapted from Qualtran v0.7.0 BloqBuilder.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/_infra/composite_bloq.py

"""Build finite register calls in the existing Core Program representation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from ...core.ir.program import (
    BlockRecord,
    FunctionRecord,
    Operation,
    ProgramRecord,
    ValueRef,
    verify_program_record,
)
from ...core.ir.program.register_bindings import RegisterBinding
from ...core.ir.program.register_verifier import register_port_types
from ...core.ir.program.registers import Register, RegisterSide, RegisterType
from ...core.ir.program.signatures import Signature
from .component_bindings import (
    RegisterBindingError,
    create_register_bindings,
    process_register_bindings,
)
from .component_rebindings import join_word_function, split_word_function
from .layout.component_workspace import allocate_word_function, free_word_function

BindingCollection = RegisterBinding | NDArray[np.object_]


class ComponentProgramBuilder:
    """Construct an owned word function from finite typed component definitions.

    Inputs and call outputs use independent SSA identities per port element.
    Failed calls reserve their operation and retain consumed input prefixes.
    Core independently verifies every definition at finalization. Workspace
    construction is explicit. Physical scheduling and classical handles require
    separate admission.
    """

    def __init__(
        self,
        signature: Signature,
        functions: Iterable[FunctionRecord] = (),
        name: str = "main",
    ) -> None:
        self._registers = tuple(signature)
        self._name = name
        self._functions: dict[str, FunctionRecord] = {}
        for function in functions:
            if function.name == name or function.name in self._functions:
                raise ValueError(f"conflicting component definition {function.name!r}")
            self._functions[function.name] = function
        declaration = FunctionRecord(name, (), registers=self._registers)
        register_port_types(declaration, RegisterSide.LEFT)
        self._next_index = 0
        self._available: set[RegisterBinding] = set()
        self._operations: list[Operation] = []
        self._returned: list[ValueRef] = []
        self.inputs = self._outputs(Signature(self._registers).lefts())
        self._arguments = tuple(
            binding.value
            for value in self.inputs.values()
            for binding in np.asarray(value).flat
        )

    def _new_value(self) -> ValueRef:
        value = ValueRef("entry", self._next_index, "register")
        self._next_index += 1
        return value

    def _outputs(self, registers: Iterable[Register]) -> dict[str, BindingCollection]:
        return {
            register.name: create_register_bindings(
                self._new_value, register, self._available
            )
            for register in registers
        }

    def _consume(self, binding: RegisterBinding) -> None:
        try:
            self._available.remove(binding)
        except KeyError:
            raise RegisterBindingError(
                f"{binding!r} is not an available register binding."
            ) from None

    def call_dict(
        self, callee: str, bindings: Mapping[str, object]
    ) -> dict[str, BindingCollection]:
        """Consume named inputs and return outputs in declaration order.

        Missing/surplus names, incompatible dtypes and unavailable handles raise
        RegisterBindingError after preserving earlier successful consumption.
        """
        try:
            function = self._functions[callee]
        except KeyError:
            raise ValueError(f"unknown component definition {callee!r}") from None
        register_port_types(function, RegisterSide.LEFT)
        assert function.registers is not None
        signature = Signature(function.registers)
        position = len(self._operations)
        self._operations.append(Operation("call", callee=callee))
        operands: list[ValueRef] = []

        def consume(
            binding: RegisterBinding, register: Register, index: tuple[int, ...]
        ) -> None:
            self._consume(binding)
            operands.append(binding.value)

        try:
            process_register_bindings(signature.lefts(), bindings, callee, consume)
        finally:
            self._operations[position] = replace(
                self._operations[position], operands=tuple(operands)
            )
        outputs = self._outputs(signature.rights())
        self._operations[position] = replace(
            self._operations[position],
            results=tuple(
                binding.value
                for value in outputs.values()
                for binding in np.asarray(value).flat
            ),
        )
        return outputs

    def call_tuple(
        self, callee: str, bindings: Mapping[str, object]
    ) -> tuple[BindingCollection, ...]:
        """Return a tuple even for zero or one declared right port."""
        return tuple(self.call_dict(callee, bindings).values())

    def call(
        self, callee: str, bindings: Mapping[str, object]
    ) -> BindingCollection | tuple[BindingCollection, ...] | None:
        """Return None, a single collection or a tuple according to port arity."""
        outputs = self.call_tuple(callee, bindings)
        if not outputs:
            return None
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def finalize(self, bindings: Mapping[str, object]) -> ProgramRecord:
        """Consume declared returns and verify the complete finite module.

        Failed returns retain their consumption prefix. A complete signature
        cannot hide invalid supplied bodies, unfinished calls or unused handles.
        """

        def consume(
            binding: RegisterBinding, register: Register, index: tuple[int, ...]
        ) -> None:
            self._consume(binding)
            self._returned.append(binding.value)

        process_register_bindings(
            Signature(self._registers).rights(), bindings, "Finalizing", consume
        )
        if self._available:
            raise RegisterBindingError(
                f"During finalization, {self._available!r} register bindings were not used."
            )
        block = BlockRecord(
            "entry",
            self._arguments,
            operations=tuple(self._operations),
            return_values=tuple(self._returned),
        )
        function = FunctionRecord(self._name, (block,), registers=self._registers)
        module = ProgramRecord((function, *self._functions.values()))
        diagnostics = verify_program_record(module)
        if diagnostics:
            raise ValueError(f"invalid constructed register Program: {diagnostics}")
        return module

    def _rebinding_name(self) -> str:
        index = len(self._functions)
        name = f"__fq_rebinding_{index}"
        while name in self._functions or name == self._name:
            index += 1
            name = f"__fq_rebinding_{index}"
        return name

    def split(self, binding: RegisterBinding) -> NDArray[np.object_]:
        """Consume one quantum word and return its ordered bit-word handles."""
        if not isinstance(binding, RegisterBinding):
            raise ValueError("split expects a single register binding")
        function = split_word_function(self._rebinding_name(), binding.register.dtype)
        self._functions[function.name] = function
        outputs = self.call_dict(function.name, {"reg": binding})
        bits = outputs["reg"]
        assert isinstance(bits, np.ndarray)
        return bits

    def join(
        self, bindings: object, dtype: RegisterType | None = None
    ) -> RegisterBinding:
        """Consume a rank-one bit-word collection in order, retaining failures."""
        try:
            values = np.asarray(bindings)
            (width,) = values.shape
        except (AttributeError, ValueError):
            raise ValueError("join expects a 1-d array of input bindings") from None
        if not all(binding.register.bitsize == 1 for binding in values):
            raise ValueError("join can only join bindings of width 1")
        target = RegisterType("opaque", width) if dtype is None else dtype
        function = join_word_function(self._rebinding_name(), target)
        self._functions[function.name] = function
        outputs = self.call_dict(function.name, {"reg": values})
        result = outputs["reg"]
        assert isinstance(result, RegisterBinding)
        return result

    def allocate(
        self, n: int = 1, dtype: RegisterType | None = None, dirty: bool = False
    ) -> RegisterBinding:
        """Allocate one owned word, with an explicit dtype taking precedence over n.

        Concrete quantum descriptors and built-in boolean flags are required.
        This records a lifetime. Physical scheduling is a separate boundary.
        """
        target = RegisterType("opaque", n) if dtype is None else dtype
        function = allocate_word_function(self._rebinding_name(), target, dirty)
        self._functions[function.name] = function
        outputs = self.call_dict(function.name, {})
        result = outputs["reg"]
        assert isinstance(result, RegisterBinding)
        return result

    def free(self, binding: RegisterBinding, dirty: bool = False) -> None:
        """Consume one scalar owned word, retaining stale-call failure timing."""
        if not isinstance(binding, RegisterBinding):
            raise ValueError("free expects a single register binding")
        function = free_word_function(
            self._rebinding_name(), binding.register.dtype, dirty
        )
        self._functions[function.name] = function
        self.call_dict(function.name, {"reg": binding})

# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 _infra/Bloqs-Tutorial.ipynb and rsa_mod_exp.py.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389

"""Native declarations and connected decompositions for the component tutorial."""

from dataclasses import dataclass

import numpy as np

from flagquantum.algorithms.primitives.modular_exponentiation import (
    ModularExponentiation,
)
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.register_types import (
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature


def cnot_definition(*, implemented: bool = False) -> FunctionRecord:
    """Declare the tutorial's control/target ports, optionally supplying CX."""
    registers = (
        Register("control", quantum_bit_type()),
        Register("target", quantum_bit_type()),
    )
    if implemented:
        return bit_gate_function(
            "CNOT", registers, CircuitIR(2, (Instruction("cx", (0, 1)),))
        )
    return FunctionRecord("CNOT", (), registers=registers)


def bit_swap_program(*, implemented: bool = False) -> ProgramRecord:
    """Store exactly the source's three calls with alternating operands."""
    signature = Signature.build(x=1, y=1)
    builder = ComponentProgramBuilder(
        signature, (cnot_definition(implemented=implemented),), name="SwapTwoBits"
    )
    x, y = builder.inputs["x"], builder.inputs["y"]
    first = builder.call_dict("CNOT", {"control": x, "target": y})
    second = builder.call_dict(
        "CNOT", {"control": first["target"], "target": first["control"]}
    )
    third = builder.call_dict(
        "CNOT", {"control": second["target"], "target": second["control"]}
    )
    return builder.finalize({"x": third["control"], "y": third["target"]})


@dataclass(frozen=True)
class _BoundWordSwap:
    wires: tuple[int, ...]
    circuit: CircuitIR

    def append_to(self, circuit: Circuit) -> Circuit:
        return circuit.compose(self.circuit, qubits=self.wires)


@dataclass(frozen=True)
class WordSwap:
    """Keep the wide declaration compact until its decomposition is requested."""

    width: int

    @property
    def signature(self) -> Signature:
        return Signature.build(x=self.width, y=self.width)

    @property
    def declaration(self) -> FunctionRecord:
        return FunctionRecord("Swap", (), registers=tuple(self.signature))

    def make_on(self, x: tuple[int, ...], y: tuple[int, ...]) -> _BoundWordSwap:
        """Bind executable CX decomposition for the ordinary append workflow."""
        if len(x) != self.width or len(y) != self.width:
            raise ValueError("swap operands must match the declared word width")
        wires = (*x, *y)
        if any(type(wire) is not int or wire < 0 for wire in wires) or len(
            set(wires)
        ) != len(wires):
            raise ValueError("swap operands require distinct nonnegative wires")
        bound = bind_register_program(
            self.decompose(implemented=True),
            (tuple(range(self.width)), tuple(range(self.width, 2 * self.width))),
            entry="Swap",
        )
        return _BoundWordSwap(wires, bound.circuit)

    def decompose(
        self, *, shaped: bool = False, implemented: bool = False
    ) -> ProgramRecord:
        """Compose per-bit swaps through explicit arrays or split/join calls."""
        if shaped:
            signature = Signature(
                Register(name, quantum_bit_type(), shape=(self.width,))
                for name in ("x", "y")
            )
        else:
            signature = self.signature
        builder = ComponentProgramBuilder(
            signature,
            bit_swap_program(implemented=implemented).functions,
            name="SwapManyBits" if shaped else "Swap",
        )
        x, y = builder.inputs["x"], builder.inputs["y"]
        if shaped:
            assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
            xs, ys = x.copy(), y.copy()
        else:
            assert isinstance(x, RegisterBinding) and isinstance(y, RegisterBinding)
            xs, ys = builder.split(x), builder.split(y)
        for index in range(self.width):
            outputs = builder.call_dict("SwapTwoBits", {"x": xs[index], "y": ys[index]})
            xs[index], ys[index] = outputs["x"], outputs["y"]
        if shaped:
            return builder.finalize({"x": xs, "y": ys})
        return builder.finalize({"x": builder.join(xs), "y": builder.join(ys)})


def reallocation_declaration() -> FunctionRecord:
    """Describe directional ports without inventing their numerical action."""
    return FunctionRecord(
        "ReAlloc",
        (),
        registers=(
            Register("input_only", quantum_bit_type(), side=RegisterSide.LEFT),
            Register("output_only", quantum_bit_type(), side=RegisterSide.RIGHT),
        ),
    )


def modular_exponentiation_program(operation: ModularExponentiation) -> ProgramRecord:
    """Store the source one-level diagram through the shared Compiler owner.

    Multiplication leaves deliberately remain declarations in this teaching
    path. Source constructor execution uses the same owner with full bodies.
    """
    from flagquantum.compiler.registers.modular.modular_exponentiation_component import (
        modular_exponentiation_program as shared_program,
    )

    operation.workspace_requirements()
    return shared_program(
        int(operation.base),
        int(operation.modulus),
        int(operation.exponent_width),
        int(operation.output_width),
        executable=False,
    )

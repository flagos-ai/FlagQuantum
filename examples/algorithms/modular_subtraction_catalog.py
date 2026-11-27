# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 mod_subtraction.py and its notebook.

"""Register signatures and direct calls for the modular subtraction tutorial."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sympy import Symbol

from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler.formats.legacy_component_library import ComponentLibrary
from flagquantum.compiler.formats.symbolic_modular_library import (
    modular_component_library,
)
from flagquantum.compiler.registers.modular.controlled_modular_addition import (
    controlled_modular_requirements,
)
from flagquantum.compiler.registers.modular.controlled_modular_negation import (
    controlled_modular_negate_adjoint_ir,
    controlled_modular_negation_requirements,
)
from flagquantum.compiler.registers.modular.controlled_modular_negation_component import (
    controlled_modular_negation_program,
)
from flagquantum.compiler.registers.modular.controlled_modular_subtraction import (
    controlled_modular_subtract_adjoint_ir,
)
from flagquantum.compiler.registers.modular.controlled_modular_subtraction_component import (
    controlled_modular_subtraction_program,
)
from flagquantum.compiler.registers.modular.modular_addition import (
    modular_add_requirements,
)
from flagquantum.compiler.registers.modular.modular_negation import (
    modular_negate_adjoint_ir,
    modular_negation_requirements,
)
from flagquantum.compiler.registers.modular.modular_negation_component import (
    modular_negation_program,
)
from flagquantum.compiler.registers.modular.modular_subtraction import (
    modular_subtract_adjoint_ir,
)
from flagquantum.compiler.registers.modular.modular_subtraction_component import (
    modular_subtraction_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.register_types import (
    quantum_bit_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.core.value_math import SymbolicInt

SubtractionKind = Literal["ModNeg", "CModNeg", "ModSub", "CModSub"]


@dataclass(frozen=True)
class ModularSubtractionExample:
    kind: SubtractionKind
    width: SymbolicInt
    modulus: SymbolicInt
    control_value: int = 1
    adjoint: bool = False

    @property
    def signature(self) -> Signature:
        dtype = unsigned_register_type(self.width)
        registers = []
        if self.kind in ("CModNeg", "CModSub"):
            registers.append(Register("ctrl", quantum_bit_type()))
        registers.append(Register("x", dtype))
        if self.kind in ("ModSub", "CModSub"):
            registers.append(Register("y", dtype))
        return Signature(registers)

    def component_library(self) -> ComponentLibrary:
        """Return the forward ordered source recipe, without physical admission."""
        if self.adjoint:
            raise ValueError("Ordered source libraries require the forward operation")
        return modular_component_library(
            self.kind,
            unsigned_register_type(self.width),
            self.modulus,
            control_value=self.control_value,
        )

    def call_graph(self) -> SymbolicCallGraph[str]:
        n, p, cv = self.width, self.modulus, self.control_value
        add_constant = f"AddK(QUInt({n}), {p + 1})"
        if self.kind in ("ModNeg", "CModNeg"):
            calls = {
                f"MultiTargetCNOT({n})": 1,
                f"C[{add_constant}](1)": 1,
                "X": 2,
            }
            if self.kind == "ModNeg":
                calls[f"C[X](QUInt({n}), 0)"] = 2
            else:
                calls[f"MultiControlX({n}, 0)"] = 2
                calls[f"And({cv}, 1)"] = 1
                calls[f"And({cv}, 1)†"] = 1
        else:
            add = (
                f"CModAdd(QUInt({n}), {p}, {cv})"
                if self.kind == "CModSub"
                else f"ModAdd({n}, {p})"
            )
            calls = {
                f"BitwiseNot(QUInt({n}))": 2,
                add_constant: 1,
                f"{add_constant}†": 1,
                add: 1,
            }
        return symbolic_call_graph(self.kind, {self.kind: calls}, max_depth=1)

    def source_ccz_count(self) -> SymbolicInt:
        """Return the source forward Toffoli-equivalent estimate."""
        n = self.width
        return {
            "ModNeg": 3 * (n - 1),
            "CModNeg": 3 * (n - 1) + 1,
            "ModSub": 6 * n - 3,
            "CModSub": 7 * n - 1,
        }[self.kind]

    def _parameters(self) -> tuple[int, int]:
        n, p = self.width, self.modulus
        if type(n) is not int or type(p) is not int:
            raise ValueError("Append requires concrete modular parameters")
        return n, p

    def _requirements(self, n: int) -> tuple[int, int]:
        return {
            "ModNeg": modular_negation_requirements,
            "CModNeg": controlled_modular_negation_requirements,
            "ModSub": modular_add_requirements,
            "CModSub": controlled_modular_requirements,
        }[self.kind](n)

    @property
    def qubit_count(self) -> int:
        n, _ = self._parameters()
        return (
            sum(int(register.bitsize) for register in self.signature)
            + self._requirements(n)[0]
        )

    def append_to(self, circuit: Circuit) -> Circuit:
        """Bind control and big-endian data ports before caller-owned clean work."""
        n, p = self._parameters()
        controlled = self.kind in ("CModNeg", "CModSub")
        subtract = self.kind in ("ModSub", "CModSub")
        offset = int(controlled)
        x = tuple(range(offset, offset + n))
        y = tuple(range(offset + n, offset + 2 * n))
        data_end = offset + n * (2 if subtract else 1)
        work_count, slots = self._requirements(n)
        if circuit.n_qubits < data_end + work_count:
            raise ValueError("Modular operation requires its data and clean work wires")
        work = tuple(range(data_end, data_end + work_count))
        bits = tuple(range(slots))
        if self.adjoint:
            if self.kind == "CModSub":
                gates = controlled_modular_subtract_adjoint_ir(
                    0, x, y, p, work, bits, control_value=self.control_value
                )
            elif self.kind == "ModSub":
                gates = modular_subtract_adjoint_ir(x, y, p, work, bits)
            elif self.kind == "CModNeg":
                gates = controlled_modular_negate_adjoint_ir(
                    0, x, p, work, bits, control_value=self.control_value
                )
            else:
                gates = modular_negate_adjoint_ir(x, p, work, bits)
        else:
            dtype = unsigned_register_type(n)
            if self.kind == "CModSub":
                graph = controlled_modular_subtraction_program(
                    dtype, p, control_value=self.control_value
                )
            elif self.kind == "ModSub":
                graph = modular_subtraction_program(dtype, p)
            elif self.kind == "CModNeg":
                graph = controlled_modular_negation_program(
                    dtype, p, control_value=self.control_value
                )
            else:
                graph = modular_negation_program(dtype, p)
            inputs: tuple[tuple[int, ...], ...] = ((0,),) if controlled else ()
            inputs += (x, y) if subtract else (x,)
            gates = bind_register_program(
                graph, inputs, clean=work
            ).circuit.instructions
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def modular_subtraction_examples() -> dict[str, ModularSubtractionExample]:
    n, p = Symbol("n"), Symbol("p")
    return {
        "mod_neg": ModularSubtractionExample("ModNeg", 32, p),
        "cmod_neg": ModularSubtractionExample("CModNeg", 32, p),
        "modsub_symb": ModularSubtractionExample("ModSub", n, p),
        "cmodsub_symb": ModularSubtractionExample("CModSub", n, p),
    }

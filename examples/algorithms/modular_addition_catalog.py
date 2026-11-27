# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 mod_addition.ipynb and mod_addition.py.

"""Concrete and symbolic fixtures from the modular-addition tutorial."""

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
from flagquantum.compiler.registers.modular.bounded_modular_addition import (
    bounded_modular_add_ir,
    bounded_modular_add_requirements,
)
from flagquantum.compiler.registers.modular.controlled_modular_addition import (
    controlled_modular_requirements,
)
from flagquantum.compiler.registers.modular.controlled_modular_addition_component import (
    controlled_modular_addition_program,
)
from flagquantum.compiler.registers.modular.modular_addition import (
    modular_add_requirements,
)
from flagquantum.compiler.registers.modular.modular_addition_component import (
    modular_addition_program,
)
from flagquantum.compiler.registers.modular.modular_constant_addition import (
    modular_constant_requirements,
)
from flagquantum.compiler.registers.modular.modular_constant_component import (
    controlled_modular_constant_program,
)
from flagquantum.compiler.registers.modular.modular_scaled_addition import (
    modular_scaled_requirements,
)
from flagquantum.compiler.registers.modular.scaled_modular_addition_component import (
    scaled_modular_addition_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.register_types import (
    montgomery_register_type,
    quantum_bit_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.core.value_math import SymbolicInt

AddKind = Literal["ModAdd", "ModAddK", "CModAdd", "CModAddK", "CtrlScaleModAdd"]


@dataclass(frozen=True)
class ModularAdditionExample:
    kind: AddKind
    width: SymbolicInt
    modulus: SymbolicInt
    constant: SymbolicInt | None = None

    @property
    def signature(self) -> Signature:
        dtype = (
            montgomery_register_type(self.width, self.modulus)
            if self.kind == "ModAdd"
            else unsigned_register_type(self.width)
        )
        ports = []
        if self.kind in ("CModAdd", "CModAddK", "CtrlScaleModAdd"):
            ports.append(Register("ctrl", quantum_bit_type()))
        ports.append(Register("x", dtype))
        if self.kind in ("ModAdd", "CModAdd", "CtrlScaleModAdd"):
            ports.append(Register("y", dtype))
        return Signature(ports)

    def component_library(self) -> ComponentLibrary:
        """Return ordered finite source calls with owned constructor metadata.

        Symbolic arithmetic remains metadata until it is bound. ModAddK has no
        source decomposition, and unresolved split widths cannot be expanded.
        """
        if self.kind == "ModAddK":
            raise ValueError("The source constant permutation has no ordered body")
        if self.kind in ("CModAddK", "CtrlScaleModAdd") and self.constant is None:
            raise ValueError("A modular constant or scale must be supplied")
        return modular_component_library(
            self.kind,
            unsigned_register_type(self.width),
            self.modulus,
            constant=0 if self.constant is None else self.constant,
        )

    def call_graph(self) -> SymbolicCallGraph[str]:
        """Return the original root declaration with parameterized child labels."""
        n, p = self.width, self.modulus
        calls: dict[str, SymbolicInt]
        if self.kind == "ModAddK":
            calls = {f"Add({n}, {n})": 5}
        elif self.kind == "CModAddK":
            calls = {f"CModAdd({n}, {p}, 1)": 1}
        elif self.kind == "CtrlScaleModAdd":
            calls = {f"CModAddK({n}, {p}, _k0)": n, "And": n, "And†": n}
        else:
            controlled = self.kind == "CModAdd"
            add = f"CAdd({n}, {n + 1}, 1)" if controlled else f"Add({n + 1}, {n + 1})"
            comparison = (
                f"CLinearDepthGreaterThan({n}, 1)"
                if controlled
                else f"LinearDepthGreaterThan({n})"
            )
            calls = {
                add: 1,
                f"AddK({n + 1}, {-p})": 1,
                f"C[AddK]({n}, {p}, 1)": 1,
                comparison: 1,
                "X": 1,
            }
        return symbolic_call_graph(self.kind, {self.kind: calls}, max_depth=1)

    def workspace_requirements(self) -> tuple[int, int]:
        n, p = self._physical_parameters()
        if self.kind == "ModAddK":
            return bounded_modular_add_requirements(n, p)
        requirements = {
            "ModAdd": modular_add_requirements,
            "CModAdd": controlled_modular_requirements,
            "CModAddK": modular_constant_requirements,
            "CtrlScaleModAdd": modular_scaled_requirements,
        }
        return requirements[self.kind](n)

    def _physical_parameters(self) -> tuple[int, int]:
        n, p = self.width, self.modulus
        if type(n) is not int or type(p) is not int:
            raise ValueError("Append requires concrete modular parameters")
        return n, p

    def append_to(self, circuit: Circuit) -> Circuit:
        """Bind the source-order live ports before caller-owned clean work."""
        n, p = self._physical_parameters()
        control = self.kind in ("CModAdd", "CModAddK", "CtrlScaleModAdd")
        offset = int(control)
        x = tuple(range(offset, offset + n))
        pair = self.kind in ("ModAdd", "CModAdd", "CtrlScaleModAdd")
        y = tuple(range(offset + n, offset + 2 * n))
        end = offset + n * (2 if pair else 1)
        work, slots = self.workspace_requirements()
        if circuit.n_qubits < end + work:
            raise ValueError("Modular addition requires its data and clean work wires")
        workspace = tuple(range(end, end + work))
        classical = tuple(range(slots))
        if p == 1 << n and self.kind != "ModAddK":
            raise ValueError("Modulus must be a positive integer fitting the operand")
        if self.kind == "ModAdd":
            graph = modular_addition_program(n, p)
        elif self.kind == "CModAdd":
            graph = controlled_modular_addition_program(unsigned_register_type(n), p)
        else:
            k = self.constant
            if type(k) is not int:
                raise ValueError("Append requires a concrete modular constant")
            if self.kind == "ModAddK":
                gates = bounded_modular_add_ir(x, k, p, workspace, classical)
                return circuit.compose(
                    CircuitIR(
                        circuit.n_qubits, gates, metadata={"batch_size": circuit.bsz}
                    )
                )
            if self.kind == "CModAddK":
                if not 0 <= k <= p:
                    raise ValueError(
                        "Modular constant must be a concrete integer in [0,modulus]"
                    )
                graph = controlled_modular_constant_program(n, p, k)
            else:
                graph = scaled_modular_addition_program(n, p, k)
        inputs: tuple[tuple[int, ...], ...] = ((0,),) if control else ()
        inputs += (x, y) if pair else (x,)
        gates = bind_register_program(
            graph, inputs, clean=workspace
        ).circuit.instructions
        return circuit.compose(
            CircuitIR(circuit.n_qubits, gates, metadata={"batch_size": circuit.bsz})
        )


def modular_addition_examples() -> dict[str, ModularAdditionExample]:
    """Retain all ten original examples in tutorial order."""
    n, p, m, k = (Symbol(name) for name in ("n", "p", "m", "k"))
    return {
        "mod_add": ModularAdditionExample("ModAdd", n, p),
        "mod_add_k": ModularAdditionExample("ModAddK", n, m, k),
        "mod_add_k_small": ModularAdditionExample("ModAddK", 4, 7, 1),
        "mod_add_k_large": ModularAdditionExample("ModAddK", 64, 500, 23),
        "cmodadd_example": ModularAdditionExample("CModAdd", 32, 10**9 + 7),
        "cmodadd_symbolic": ModularAdditionExample("CModAdd", n, p),
        "cmod_add_k": ModularAdditionExample("CModAddK", n, m, k),
        "cmod_add_k_small": ModularAdditionExample("CModAddK", 4, 7, 1),
        "ctrl_scale_mod_add": ModularAdditionExample("CtrlScaleModAdd", n, m, k),
        "ctrl_scale_mod_add_small": ModularAdditionExample("CtrlScaleModAdd", 4, 7, 1),
    }

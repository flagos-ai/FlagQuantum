"""Square-lattice Majorana selection with shared bounded traversal."""

# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 HubbardMajorannaOperator.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/hubbard_model/qubitization/select_hubbard.py

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from ...circuit import Circuit
from ...compiler._component_trace.resource_model import ComponentCost, GateCost
from ...compiler.registers.bits.bitwise import _validate_wires
from ...compiler.registers.layout.partition import RegisterLayout, unpartition_register
from ...compiler.registers.lookup.bounded_majorana_select import (
    bounded_majorana_ir,
    bounded_majorana_workspace,
)
from ...core.classical_control import validate_classical_control
from ...core.ir import CircuitIR, Instruction
from ...dynamic import DynamicCircuit
from .hubbard_qubitization_resource_model import source_hubbard_majorana_formula


@dataclass(frozen=True)
class HubbardMajorana:
    """Select X or Y after the Jordan–Wigner prefix, with optional control.

    The MSB-first address is ordered x, y, spin. Its flattened target index
    is (x * dimension + y) * 2 + spin. Workspace is clean and restored.
    """

    dimension: int
    pauli: Literal["X", "Y"] = "Y"
    controlled: bool = False
    control_value: int = field(default=1, kw_only=True)

    def __post_init__(self) -> None:
        if type(self.dimension) is not int or self.dimension < 1:
            raise ValueError("Hubbard Majorana requires positive square side length")
        if self.pauli not in ("X", "Y"):
            raise ValueError("Hubbard Majorana requires X or Y")
        if type(self.controlled) is not bool:
            raise ValueError("Hubbard Majorana control flag must be boolean")
        if type(self.control_value) is not int or self.control_value not in (0, 1):
            raise ValueError("Hubbard Majorana control value must be zero or one")
        if not self.controlled and self.control_value == 0:
            raise ValueError("Hubbard Majorana control on zero needs a control wire")

    @property
    def layout(self) -> RegisterLayout:
        coordinate = (self.dimension - 1).bit_length()
        control = (("control", 1),) if self.controlled else ()
        return (
            *control,
            ("x", coordinate),
            ("y", coordinate),
            ("spin", 1),
            ("target", 2 * self.dimension**2),
        )

    def requirements(self) -> tuple[int, int]:
        """Return clean accumulator/tree wires and reusable measurement slots."""
        d = self.dimension
        work = bounded_majorana_workspace(
            2 * d**2,
            controlled=self.controlled,
            axis_extents=(d, d, 2) if d > 1 else (),
            selection_width=1 if d == 1 else None,
        )
        return work, work - 1

    def source_resource_model(self) -> ComponentCost:
        """Keep the source declaration and add X gates for native zero control.

        The source declaration omits Pauli costs. Zero control uses a native
        X sandwich because the original zero-controlled body fails to build.
        """
        formula = source_hubbard_majorana_formula(
            self.dimension, controlled=self.controlled
        )
        return ComponentCost(
            f"HubbardMajorannaOperator({self.pauli})",
            int(formula.logical_qubits),
            GateCost(
                and_bloq=int(formula.and_bloq),
                clifford=int(formula.clifford) + 2 * int(self.control_value == 0),
                measurement=int(formula.measurement),
            ),
        )

    def _emit(
        self,
        ports: Mapping[str, tuple[int, ...]],
        work: tuple[int, ...],
        slots: tuple[int, ...],
    ) -> tuple[Instruction, ...]:
        d = self.dimension
        control = ports["control"][0] if self.controlled else None
        gates = bounded_majorana_ir(
            (*ports["x"], *ports["y"], *ports["spin"]),
            ports["target"],
            control,
            self.pauli,
            work,
            slots,
            axis_extents=(d, d, 2) if d > 1 else (),
            selection_width=1 if d == 1 else None,
        )
        if control is not None and self.control_value == 0:
            flip = Instruction("x", (control,))
            return (flip, *gates, flip)
        return gates

    def make_on(
        self, ports: Mapping[str, tuple[int, ...]], *, workspace: tuple[int, ...]
    ) -> "_BoundHubbardMajorana":
        """Bind a standalone involution for circuit.append(op)."""
        logical = unpartition_register(ports, self.layout)
        _validate_wires((*logical, *workspace))
        if len(workspace) != self.requirements()[0]:
            raise ValueError("Hubbard Majorana requires matching clean work")
        return _BoundHubbardMajorana(
            self, dict(ports), workspace, max((*logical, *workspace)) + 1
        )


@dataclass(frozen=True)
class _BoundHubbardMajorana:
    operation: HubbardMajorana
    ports: Mapping[str, tuple[int, ...]]
    workspace: tuple[int, ...]
    required_qubits: int

    def append_to(self, circuit: Circuit) -> Circuit:
        if circuit.n_qubits < self.required_qubits or not isinstance(
            circuit, DynamicCircuit
        ):
            raise ValueError(
                "Hubbard Majorana needs a DynamicCircuit of sufficient width"
            )
        start = validate_classical_control(
            circuit.to_ir().instructions, n_qubits=circuit.n_qubits
        )
        count = self.operation.requirements()[1]
        gates = self.operation._emit(
            self.ports, self.workspace, tuple(range(start, start + count))
        )
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits, gates, metadata={"batch_size": circuit.bsz}
            ).validate()
        )

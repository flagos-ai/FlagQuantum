# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 select_pauli_lcu.py.

"""The locked four-row Pauli SELECT tutorial fixture."""

from dataclasses import dataclass

from flagquantum.algorithms.primitives.multiplexers.select_pauli_lcu import (
    SelectPauliLCU,
    _BoundedSelectPauliLCU,
)
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.compiler.registers.lookup.unary_iteration import (
    unary_iteration_workspace,
)
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature


@dataclass(frozen=True)
class PauliSelectExample:
    control_value: int | None = None

    @property
    def select(self) -> SelectPauliLCU:
        return SelectPauliLCU(
            2, 4, ("XIXI", "YIYI", "ZZZZ", "ZXYZ"), self.control_value
        )

    @property
    def signature(self) -> Signature:
        control = (
            (Register("control", quantum_bit_type()),)
            if self.control_value is not None
            else ()
        )
        return Signature(
            (
                *control,
                Register("selection", bounded_unsigned_register_type(2, 4)),
                Register("target", opaque_register_type(4)),
            )
        )

    @property
    def qubit_count(self) -> int:
        return (
            6
            + int(self.control_value is not None)
            + unary_iteration_workspace(2, self.control_value)
        )

    def _bound(self) -> _BoundedSelectPauliLCU:
        offset = int(self.control_value is not None)
        return self.select.make_bounded_on(
            control=tuple(range(offset)),
            selection=tuple(range(offset, offset + 2)),
            target=tuple(range(offset + 2, offset + 6)),
            workspace=tuple(range(offset + 6, self.qubit_count)),
            classical_bits=tuple(range(self.qubit_count - offset - 6)),
        )

    def append_to(self, circuit: Circuit) -> Circuit:
        return circuit.append(self._bound())

    def resource_model(self) -> ComponentCost:
        return self._bound().resource_model()


def pauli_select_example_call_graph() -> SymbolicCallGraph[str]:
    """Retain the original controlled Pauli identities after identity removal."""
    root = "Pauli SELECT"
    return symbolic_call_graph(
        root,
        {
            root: {
                "X": 2,
                "And(1,0)": 2,
                "CNOT": 2,
                "And(1,1)†": 2,
                "C[XX]": 1,
                "C[YY]": 1,
                "C[ZZZZ]": 1,
                "C[ZXYZ]": 1,
            }
        },
        max_depth=1,
    )

# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the locked Qualtran v0.7.0 ControlledViaAnd notebook.

"""Run the notebook's Hadamard on grouped predicates and query its source graph."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler.registers.component_builder import (
    BindingCollection,
    ComponentProgramBuilder,
)
from flagquantum.compiler.registers.component_control import (
    control_component_via_predicate,
    quantum_control_signature,
)
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.control.controlled_component import (
    controlled_component_ir,
)
from flagquantum.compiler.registers.primitives.controlled_hadamard import (
    controlled_hadamard_ir,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import (
    quantum_bit_type,
    signed_register_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit


def _add_positive_hadamard(
    builder: ComponentProgramBuilder,
    control: BindingCollection,
    values: dict[str, BindingCollection],
) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
    control, target = builder.call_tuple("ch", {"ctrl": control, "target": values["q"]})
    return control, (target,)


@dataclass(frozen=True)
class ControlledHadamardExample:
    """Bind a locked fixture using flattened big-endian words followed by target."""

    kind: Literal["bits", "integers", "nested"] = "bits"

    @cached_property
    def groups(self) -> tuple[ControlRegister, ...]:
        if self.kind == "integers":
            return (
                ControlRegister(4, (0, 1, 2, 3)),
                ControlRegister(4, (0, 1, -1, -2), "twos_complement"),
            )
        bits = ControlRegister(1, (0, 1, 1, 0))
        return (ControlRegister(1, (1, 1)), bits) if self.kind == "nested" else (bits,)

    @cached_property
    def signature(self) -> Signature:
        bit = quantum_bit_type()
        controls: tuple[tuple[RegisterType, tuple[int, ...]], ...]
        if self.kind == "integers":
            controls = (
                (unsigned_register_type(4), (4,)),
                (signed_register_type(4), (4,)),
            )
        else:
            controls = tuple((bit, (len(group.values),)) for group in self.groups)
        return quantum_control_signature(Signature((Register("q", bit),)), controls)

    def call_graph(self, *, depth: int | None = None) -> SymbolicCallGraph[str]:
        """Declare the source predicate ladder without expanding a physical circuit."""
        root = "ControlledViaAnd"
        bits = tuple(bit for group in self.groups for bit in group.bits())
        forward = {"And": len(bits) - 1, "XGate": 2 * bits.count(0)}
        inverse = {"And†": forward["And"], "XGate": forward["XGate"]}
        return symbolic_call_graph(
            root,
            {
                root: {"CHadamard": 1, "CtrlSpecAnd": 1, "CtrlSpecAnd†": 1},
                "CtrlSpecAnd": {"MultiAnd": 1},
                "CtrlSpecAnd†": {"MultiAnd†": 1},
                "MultiAnd": forward,
                "MultiAnd†": inverse,
            },
            max_depth=depth,
        )

    def program(self) -> ProgramRecord:
        """Retain the source predicate, positive CH and measured erase calls."""
        bit = quantum_bit_type()
        positive = bit_gate_function(
            "h", (Register("q", bit),), CircuitIR(1, (Instruction("h", (0,)),))
        )
        controlled = bit_gate_function(
            "ch",
            (Register("ctrl", bit), Register("target", bit)),
            CircuitIR(2, controlled_hadamard_ir((0,), 1)),
        )
        builder = ComponentProgramBuilder(
            Signature((Register("q", bit),)), (positive, controlled)
        )
        record = builder.finalize(builder.call_dict("h", {"q": builder.inputs["q"]}))
        return control_component_via_predicate(
            record,
            {"main": _add_positive_hadamard},
            tuple(self.signature)[:-1],
            self.groups,
        )

    def append_to(self, circuit: Circuit) -> Circuit:
        sizes = tuple(len(group.bits()) for group in self.groups)
        width = sum(sizes)
        if not isinstance(circuit, DynamicCircuit) or circuit.n_qubits < 2 * width:
            raise ValueError(
                "Controlled Hadamard example needs a DynamicCircuit and clean work"
            )
        controls = tuple(
            tuple(range(sum(sizes[:i]), sum(sizes[: i + 1]))) for i in range(len(sizes))
        )
        body = controlled_component_ir(
            controls,
            self.groups,
            lambda predicate: (Instruction("ch", (predicate, width)),),
            tuple(range(width + 1, 2 * width)),
            tuple(range(width - 1)),
        )
        return circuit.compose(
            CircuitIR(circuit.n_qubits, body, metadata={"batch_size": circuit.bsz})
        )

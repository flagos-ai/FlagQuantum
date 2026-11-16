# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 specialized_ctrl.ipynb.

"""Register-program bodies for the specialized control tutorial."""

from dataclasses import dataclass
from math import pi

from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_builder import (
    BindingCollection,
    ComponentProgramBuilder,
)
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.control.component_adjoint_control import (
    adjoint_component_control_system,
)
from flagquantum.compiler.registers.control.component_control_selection import (
    select_component_control_system,
)
from flagquantum.compiler.registers.control.component_extension import (
    StructuredControlAdder,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.control.specialized_component_control import (
    multi_controlled_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature


def phase_conjugated_x_program(*, positive_control: bool = False) -> ProgramRecord:
    """Store T, then X or its positive specialization, then T† on the target."""
    bit = quantum_bit_type()
    q = Register("q", bit)
    ctrl = Register("ctrl", bit)
    ports = (ctrl, q) if positive_control else (q,)
    t = bit_gate_function("t", (q,), CircuitIR(1, (Instruction("t", (0,)),)))
    tdg = bit_gate_function("tdg", (q,), CircuitIR(1, (Instruction("tdg", (0,)),)))
    flip = Instruction("cx", (0, 1)) if positive_control else Instruction("x", (0,))
    x = bit_gate_function(
        "cx" if positive_control else "x",
        ports,
        CircuitIR(len(ports), (flip,)),
    )
    builder = ComponentProgramBuilder(Signature(ports), (t, tdg, x))
    values = dict(builder.inputs)
    values["q"] = builder.call_dict("t", {"q": values["q"]})["q"]
    outputs = builder.call_dict(x.name, values)
    outputs["q"] = builder.call_dict("tdg", {"q": outputs["q"]})["q"]
    return builder.finalize(outputs)


def doubly_controlled_phase_conjugated_x() -> ProgramRecord:
    """Bind both positive controls through one shared predicate and erase."""
    return multi_controlled_component_program(
        phase_conjugated_x_program(positive_control=True), "ctrl", (1, 1)
    )


def controlled_adjoint_phase_conjugated_x(*, specialize: bool) -> ProgramRecord:
    """Compare direct positive specialization with controlling each inverse gate."""
    forward = phase_conjugated_x_program()
    positive = phase_conjugated_x_program(positive_control=True)
    bit = quantum_bit_type()
    ctrl, q = Register("ctrl", bit), Register("q", bit)
    ports = (ctrl, q)
    patterns = (ControlRegister(1, (1,)),)

    def fallback(
        _controls: tuple[Register, ...], _patterns: tuple[ControlRegister, ...]
    ) -> tuple[ProgramRecord, StructuredControlAdder]:
        # This fixture is self-adjoint. Expanding the inverse gate by gate
        # still yields T, X, T† in execution order.
        body = (
            ("ct", Instruction("cphase", (0, 1), {"theta": pi / 4})),
            ("cx", Instruction("cx", (0, 1))),
            ("ctdg", Instruction("cphase", (0, 1), {"theta": -pi / 4})),
        )
        leaves = tuple(
            bit_gate_function(name, ports, CircuitIR(2, (gate,))) for name, gate in body
        )
        builder = ComponentProgramBuilder(Signature(ports), leaves)
        values = builder.inputs
        for leaf in leaves:
            values = builder.call_dict(leaf.name, values)
        module = builder.finalize(values)

        def add(
            caller: ComponentProgramBuilder,
            controls: tuple[BindingCollection, ...],
            data: dict[str, BindingCollection],
        ) -> tuple[tuple[BindingCollection, ...], tuple[BindingCollection, ...]]:
            (control,) = controls
            outputs = caller.call_dict("main", {**data, "ctrl": control})
            return (outputs["ctrl"],), (outputs["q"],)

        return module, add

    def select(
        controls: tuple[Register, ...], values: tuple[ControlRegister, ...]
    ) -> tuple[ProgramRecord, StructuredControlAdder]:
        return select_component_control_system(
            controls,
            values,
            lambda cv: (positive, "ctrl") if cv == 1 else None,
            fallback,
        )

    module, _ = adjoint_component_control_system(
        forward,
        (ctrl,),
        patterns,
        frozenset((1,)) if specialize else frozenset(),
        select,
        fallback,
        {"t": "tdg", "tdg": "t", "cx": "cx"},
    )
    return module


@dataclass(frozen=True)
class SpecializedControlExample:
    """Append a bound tutorial Program with its caller-owned workspace."""

    program: ProgramRecord
    controls: int

    def append_to(self, circuit: Circuit) -> Circuit:
        inputs = tuple((i,) for i in range(self.controls + 1))
        clean = (self.controls + 1,) if self.controls == 2 else ()
        bound = bind_register_program(
            self.program, inputs, clean=clean, measured_workspace="reset"
        )
        if circuit.n_qubits < bound.circuit.n_wires:
            raise ValueError(
                "Specialized control example requires its data and work wires"
            )
        return circuit.compose(
            CircuitIR(
                circuit.n_qubits,
                bound.circuit.instructions,
                metadata={"batch_size": circuit.bsz},
            )
        )

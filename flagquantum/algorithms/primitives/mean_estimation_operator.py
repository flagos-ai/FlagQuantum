# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 MeanEstimationOperator.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/mean_estimation/mean_estimation_operator.py

"""Mean-estimation walk from a phase oracle and the preparation reflection."""

from dataclasses import dataclass, fields, replace
from typing import cast

from ...circuit import Circuit, _CircuitOperation
from ...compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from ...compiler._component_trace.qec_gate_cost import QECGateCost
from ...compiler._component_trace.resource_model import (
    ComponentCost,
    estimate_component_resources,
)
from ...compiler.registers.component_control import quantum_control_signature
from ...compiler.registers.preparation.prepare_reflection import (
    prepare_reflection_ir,
    prepare_reflection_requirements,
    source_uncontrolled_prepare_reflection_cost,
)
from ...compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from ...core.ir import CircuitIR
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt, smax
from ._block_binding import _BoundBlockOperation, _capture_block, _validate_block_work
from .complex_phase_oracle import ComplexPhaseOracle


@dataclass(frozen=True)
class CodeForRandomVariable:
    """Pair the encoder layout with the synthesizer's same selection registers."""

    synthesizer_selection_registers: tuple[Register, ...]
    encoder_selection_registers: tuple[Register, ...]
    encoder_target_registers: tuple[Register, ...]

    def __post_init__(self) -> None:
        if self.synthesizer_selection_registers != self.encoder_selection_registers:
            raise ValueError(
                "Encoder and synthesizer must use the same selection registers"
            )


@dataclass(frozen=True)
class MeanEstimationOperator:
    """Apply the encoded phase and then reflect about the synthesized state."""

    code: CodeForRandomVariable
    arctan_bitsize: int = 32
    control_value: int | None = None
    repetitions: int = 1

    def __post_init__(self) -> None:
        if self.control_value is not None and (
            type(self.control_value) is not int or self.control_value not in (0, 1)
        ):
            raise ValueError("Mean estimation control activation must be binary")
        if type(self.repetitions) is not int or self.repetitions == 0:
            raise ValueError("Mean estimation power must be a nonzero integer")

    def controlled(self, control_value: int = 1) -> "MeanEstimationOperator":
        if self.control_value is not None:
            raise ValueError("Mean estimation currently takes one external control")
        if type(control_value) is not int or control_value not in (0, 1):
            raise ValueError("Mean estimation control activation must be binary")
        return replace(self, control_value=control_value)

    def __pow__(self, exponent: int) -> "MeanEstimationOperator":
        if type(exponent) is not int or exponent == 0:
            raise ValueError("Mean estimation power must be a nonzero integer")
        return replace(self, repetitions=self.repetitions * exponent)

    def wire_symbols(self) -> tuple[str, ...]:
        width = self._selection_width()
        power = abs(self.repetitions)
        label = "U_ko" if power == 1 else f"U_ko^{power}"
        controls = (
            ()
            if self.control_value is None
            else ("@" if self.control_value else "(0)",)
        )
        return (*controls, *((label,) * width))

    @property
    def signature(self) -> Signature:
        selection = Signature(self.code.encoder_selection_registers)
        if self.control_value is None:
            return selection
        return quantum_control_signature(selection, ((quantum_bit_type(), ()),))

    @property
    def select(self) -> ComplexPhaseOracle:
        return ComplexPhaseOracle(
            (),
            self.code.encoder_selection_registers,
            self.code.encoder_target_registers,
            self.arctan_bitsize,
        )

    def _selection_width(self) -> int:
        width = Signature(self.code.encoder_selection_registers).n_qubits()
        if not isinstance(width, int):
            raise ValueError(
                "Mean estimation execution needs a concrete selection width"
            )
        return width

    def workspace_requirements(self) -> tuple[int, int]:
        controlled = self.control_value is not None
        phase_wires, phase_slots = self.select.workspace_requirements(
            controlled=controlled
        )
        reflection_wires, reflection_slots = prepare_reflection_requirements(
            self._selection_width(), controlled=controlled
        )
        return max(phase_wires, reflection_wires), max(phase_slots, reflection_slots)

    def source_call_graph(self) -> SymbolicCallGraph[str]:
        """Describe the logical phase and reflection calls independently of lowering."""
        step = "MeanEstimationOperator"
        phase = "ComplexPhaseOracle"
        if self.repetitions < 0:
            step += "†"
            phase += "†"
        if self.control_value is not None:
            step = f"Controlled({step})"
            phase = f"Controlled({phase})"
        graph: dict[str, dict[str, SymbolicInt]] = {
            step: {phase: 1, "ReflectionUsingPrepare": 1}
        }
        root = step
        power = abs(self.repetitions)
        if power != 1:
            root = f"{step}^{power}"
            graph[root] = {step: power}
        if self.control_value == 0:
            zero_control = f"ControlOnZero({root})"
            graph[zero_control] = {root: 1, "X": 2}
            root = zero_control
        return symbolic_call_graph(root, graph)

    def source_resource_model(
        self,
        encoder: SymbolicResourceFormula,
        encoder_adjoint: SymbolicResourceFormula,
        prepare: ComponentCost,
        unprepare: ComponentCost,
    ) -> SymbolicResourceFormula:
        """Price uncontrolled signed powers from one step's explicit child models.

        Both phase directions retain the same source peak after private phase
        outputs are released. Sequential repetitions multiply gates while keeping
        the step's peak width. Controlled source estimates remain unavailable.
        """
        if self.control_value is not None:
            raise ValueError(
                "This source model describes one forward uncontrolled step"
            )
        phase = self.select.source_resource_model(encoder, encoder_adjoint)
        reflection = estimate_component_resources(
            source_uncontrolled_prepare_reflection_cost(
                self._selection_width(),
                0,
                prepare,
                unprepare,
            )
        )
        count = abs(self.repetitions)
        costs = QECGateCost().plus(phase, count).plus(reflection.gates, count)
        peak = cast(
            SymbolicInt, smax(phase.logical_qubits, reflection.peak_logical_qubits)
        )
        return SymbolicResourceFormula(
            logical_qubits=peak,
            **{field.name: getattr(costs, field.name) for field in fields(costs)},
        )

    def make_on(
        self,
        synthesizer: _CircuitOperation,
        encoder: _CircuitOperation,
        *,
        selection: tuple[int, ...],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
        adjoint: bool = False,
        control: int | None = None,
    ) -> _BoundBlockOperation:
        """Bind both children and reuse the cleaned phase work for the reflection."""
        if type(adjoint) is not bool:
            raise ValueError("Mean estimation direction must be boolean")
        if len(selection) != self._selection_width():
            raise ValueError(
                "Mean estimation binding must match its selection signature"
            )
        controlled = control is not None
        if controlled != (self.control_value is not None):
            raise ValueError(
                "Mean estimation control wire must match its control specification"
            )
        inverse_direction = adjoint != (self.repetitions < 0)
        value = 1 if self.control_value is None else self.control_value
        work, slots = self.workspace_requirements()
        if len(workspace) != work or len(classical_bits) != slots:
            raise ValueError("Mean estimation binding must supply its exact workspace")
        public = selection if control is None else (control, *selection)
        _validate_block_work(public, workspace, classical_bits)
        required = max((*public, *workspace)) + 1
        phase_work, phase_slots = self.select.workspace_requirements(
            controlled=controlled
        )
        oracle = self.select.make_on(
            encoder,
            through=selection,
            workspace=workspace[:phase_work],
            classical_bits=classical_bits[:phase_slots],
            adjoint=inverse_direction,
            phase_control=control,
            control_value=value,
        )
        forward = _capture_block(synthesizer, required, set(selection))
        inverse = Circuit(required).compose(CircuitIR(required, forward)).adjoint()
        reflect_work, reflect_slots = prepare_reflection_requirements(
            len(selection), controlled=controlled
        )
        reflection = prepare_reflection_ir(
            selection,
            workspace[:reflect_work],
            classical_bits[:reflect_slots],
            prepare=forward,
            unprepare=inverse.to_ir().instructions,
            global_phase=-1,
            control=control,
            control_value=value,
        )
        gates = (
            (*reflection, *oracle.instructions)
            if inverse_direction
            else (*oracle.instructions, *reflection)
        )
        return _BoundBlockOperation(gates * abs(self.repetitions), required)

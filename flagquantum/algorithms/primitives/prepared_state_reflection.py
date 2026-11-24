# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 reflections/reflection_using_prepare.py.

"""Declare prepared-state reflections and bind their existing compiler recipe."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import isclose, pi

from ...compiler._component_trace.resource_model import ComponentCost
from ...compiler.registers.layout.partition import unpartition_register
from ...compiler.registers.preparation.prepare_reflection import (
    prepare_reflection_ir,
    source_reflection_declared_cost,
)
from ...compiler.registers.preparation.prepared_reflection_component import (
    prepared_reflection_program,
)
from ...compiler.registers.primitives.global_phase import global_phase_exponent
from ...compiler.registers.primitives.global_phase_component import scalar_phase_program
from ...core.ir.program import ProgramRecord
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt
from ._block_binding import _BoundBlockOperation, _validate_block_work
from .black_box_prepare import (
    BlackBoxPrepare,
    PackedPrepareDeclaration,
    PrepareEmitter,
    _concrete_layout,
)
from .identity_prepare import IdentityPrepare, identity_prepare_program
from .prepare_oracle import PrepareOracle


@dataclass(frozen=True)
class PreparedStateReflection:
    """Preserve the preparation's selection ports and optional binary control."""

    prepare_gate: PrepareOracle | BlackBoxPrepare
    control_value: int | None = None
    global_phase: complex = 1
    precision: float = 1e-11

    def __post_init__(self) -> None:
        if self.control_value is not None and (
            type(self.control_value) is not int or self.control_value not in (0, 1)
        ):
            raise ValueError("Reflection control activation must be zero or one")

    @property
    def control_registers(self) -> tuple[Register, ...]:
        return (
            ()
            if self.control_value is None
            else (Register("control", quantum_bit_type()),)
        )

    @property
    def selection_registers(self) -> tuple[Register, ...]:
        return self.prepare_gate.selection_registers

    @property
    def signature(self) -> Signature:
        return Signature((*self.control_registers, *self.selection_registers))

    @classmethod
    def around_zero(
        cls,
        bitsizes: Sequence[SymbolicInt],
        control_value: int | None = None,
        global_phase: complex = 1,
        precision: float = 1e-11,
    ) -> "PreparedStateReflection":
        return cls(
            IdentityPrepare.from_bitsizes(bitsizes),
            control_value,
            global_phase,
            precision,
        )

    def adjoint(self) -> "PreparedStateReflection":
        """Retain the source-declared adjoint, including its nonreal-phase defect.

        Physical mathematical inversion is requested with make_on(adjoint=True).
        """
        return self

    def controlled(self, control_value: int = 1) -> "PreparedStateReflection":
        """Specialize the source's first single-bit control."""
        if self.control_value is not None:
            raise ValueError(
                "Additional reflection controls require the shared control binder"
            )
        return replace(self, control_value=control_value)

    def wire_symbols(self) -> tuple[str, ...]:
        layout = _concrete_layout(self.selection_registers)
        controls = (
            ()
            if self.control_value is None
            else ("@" if self.control_value else "@(0)",)
        )
        return (*controls, *(("R_L",) * sum(width for _, width in layout)))

    def resource_model(
        self,
        prepare_cost: ComponentCost,
        unprepare_cost: ComponentCost,
        *,
        prepare_is_self_adjoint: bool | None = None,
    ) -> ComponentCost:
        """Query the original declared graph independently from the emitted scan."""
        selected = sum(width for _, width in _concrete_layout(self.selection_registers))
        junk = sum(
            width for _, width in _concrete_layout(self.prepare_gate.junk_registers)
        )
        return source_reflection_declared_cost(
            selected,
            junk,
            prepare_cost,
            unprepare_cost,
            control_value=self.control_value,
            global_phase=self.global_phase,
            precision=self.precision,
            prepare_is_self_adjoint=(
                isinstance(self.prepare_gate, IdentityPrepare)
                if prepare_is_self_adjoint is None
                else prepare_is_self_adjoint
            ),
        )

    def make_on(
        self,
        *,
        selection: Mapping[str, tuple[int, ...]],
        reflection_workspace: tuple[int, ...],
        reflection_classical_bits: tuple[int, ...] = (),
        prepare_emit: PrepareEmitter | None = None,
        prepare_junk: tuple[int, ...] = (),
        prepare_workspace: tuple[int, ...] = (),
        prepare_classical_bits: tuple[int, ...] = (),
        control: int | None = None,
        adjoint: bool = False,
    ) -> _BoundBlockOperation:
        """Bind explicit preparation storage and the shared zero-reflection recipe.

        Preparation junk follows the child's lifetime contract. It can remain
        live after the reflection and is retained for a restoring inverse step.
        PrepareEmitter supplies both directions independently, including feedback.

        Examples:
            >>> import flagquantum as fq
            >>> op = PreparedStateReflection.around_zero((1,), global_phase=-1)
            >>> circuit = fq.Circuit(2).append(op.make_on(
            ...     selection={"reg0_": (0,)}, reflection_workspace=(1,)))
        """
        if (control is not None) != (self.control_value is not None):
            raise ValueError("Reflection control wire must match its declaration")
        selection_wires = unpartition_register(
            selection, _concrete_layout(self.selection_registers)
        )
        if isinstance(self.prepare_gate, BlackBoxPrepare):
            if prepare_emit is not None:
                raise ValueError("Packed PREPARE already owns its emitter")
            preparation = self.prepare_gate
        elif prepare_emit is not None:
            preparation = PackedPrepareDeclaration(self.prepare_gate).bind(prepare_emit)
        elif isinstance(self.prepare_gate, IdentityPrepare):
            preparation = self.prepare_gate.bind()
        else:
            raise ValueError("Named PREPARE requires a bound directional emitter")
        public = (
            *selection_wires,
            *prepare_junk,
            *((control,) if control is not None else ()),
        )
        _validate_block_work(public, prepare_workspace, prepare_classical_bits)
        _validate_block_work(public, reflection_workspace, reflection_classical_bits)
        inverse = preparation.emit(
            selection_wires,
            prepare_junk,
            prepare_workspace,
            prepare_classical_bits,
            adjoint=True,
        )
        forward = preparation.emit(
            selection_wires, prepare_junk, prepare_workspace, prepare_classical_bits
        )
        gates = prepare_reflection_ir(
            selection_wires,
            reflection_workspace,
            reflection_classical_bits,
            prepare=forward,
            unprepare=inverse,
            control=control,
            control_value=1 if self.control_value is None else self.control_value,
            global_phase=self.global_phase,
            adjoint=adjoint,
        )
        required = (
            max((*public, *prepare_workspace, *reflection_workspace), default=0) + 1
        )
        return _BoundBlockOperation(gates, required)


def zero_state_reflection_program(reflection: PreparedStateReflection) -> ProgramRecord:
    """Store the concrete identity-PREPARE reflection and its shared dependencies.

    Selection words are MSB-first THRU storage. A missing control owns one clean
    phase target. MultiControlZ owns and erases predicate work. Uncontrolled
    nontrivial scalar phases use the shared exact private carrier owner.
    """
    prepare = reflection.prepare_gate
    if not isinstance(prepare, IdentityPrepare):
        raise ValueError("Stored zero reflection requires IdentityPrepare")
    if not isclose(abs(reflection.global_phase), 1, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Reflection global phase must be a unit coefficient")
    preparation = identity_prepare_program(prepare)
    return prepared_reflection_program(
        preparation,
        preparation,
        tuple(reg.name for reg in prepare.selection_registers),
        control_value=reflection.control_value,
        global_phase=reflection.global_phase,
        scalar_phase=(
            scalar_phase_program(
                float(global_phase_exponent(reflection.global_phase)) * pi
            )
            if reflection.control_value is None and reflection.global_phase != 1
            else None
        ),
    )

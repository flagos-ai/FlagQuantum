"""Bind the measured Hubbard SELECT and PREPARE owners to Hamiltonian GQSP."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ...compiler.registers.control.control_predicate import ControlRegister
from ...compiler.registers.preparation.prepare_reflection import (
    prepare_reflection_requirements,
)
from ...core.ir import Instruction
from .black_box_prepare import BlackBoxPrepare
from .hamiltonian_gqsp import HamiltonianGQSP, _BoundHamiltonianGQSP
from .hubbard_prepare import PrepareHubbard
from .hubbard_select import SelectHubbard
from .hubbard_walk import HubbardWalk
from .qubitization_walk import QubitizationWalk
from .select_block_encoding import SelectBlockEncoding


@dataclass(frozen=True)
class HubbardHamiltonianGQSP:
    """Evolve with the source selection-only reflection and explicit feedback."""

    prepare: PrepareHubbard
    time: float
    precision: float

    def requirements(self) -> tuple[int, int]:
        """Return reusable work width and feedback width, excluding both temp words."""
        prep_work, prep_slots = self.prepare.requirements()
        select_work, select_slots = SelectHubbard(
            self.prepare.dimension, controlled=True
        ).requirements()
        reflect_work, reflect_slots = prepare_reflection_requirements(
            sum(width for _, width in self.prepare.selection_layout), controlled=True
        )
        return max(prep_work, select_work, reflect_work), max(
            prep_slots, select_slots, reflect_slots
        )

    def make_on(
        self,
        *,
        signal: int,
        target: tuple[int, ...],
        prepare_temp: tuple[int, ...],
        reflection_temp: tuple[int, ...],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
    ) -> _BoundHamiltonianGQSP:
        """Keep outer PREPARE temp live and reuse predicate scratch between stages.

        All supplied work starts at zero. Reflection temp and predicate work
        return to zero. Outer PREPARE temp is retained as output because the
        locked Hubbard sequence can leave it entangled with the result.
        """
        work_width, slot_width = self.requirements()
        if (
            len(workspace) != work_width
            or len(classical_bits) != slot_width
            or len(set(classical_bits)) != slot_width
            or any(type(bit) is not int or bit < 0 for bit in classical_bits)
        ):
            raise ValueError(
                "Hubbard GQSP requires matching work and unique feedback slots"
            )
        prep = self.prepare
        source_walk = HubbardWalk(prep, reflect_directions=False)
        select = SelectHubbard(prep.dimension)
        controlled_select = SelectHubbard(prep.dimension, controlled=True)
        select_work, select_slots = controlled_select.requirements()
        prep_work, prep_slots = prep.requirements()
        reflect_work, reflect_slots = prepare_reflection_requirements(
            sum(width for _, width in prep.selection_layout), controlled=True
        )
        prepared = BlackBoxPrepare(
            prep.selection_layout, (("temp", 2),), prep._emit, prep.alpha
        )

        def emit_select(
            ports: Mapping[str, tuple[int, ...]],
            work: tuple[int, ...],
            slots: tuple[int, ...],
            adjoint: bool,
        ) -> tuple[Instruction, ...]:
            return select._emit(ports, work, slots, adjoint=adjoint)

        def emit_controlled(
            value: int,
            ports: Mapping[str, tuple[int, ...]],
            work: tuple[int, ...],
            slots: tuple[int, ...],
            adjoint: bool,
        ) -> tuple[Instruction, ...]:
            assert value == 1
            return controlled_select._emit(ports, work, slots, adjoint=adjoint)

        walk = QubitizationWalk(
            SelectBlockEncoding(
                prep.selection_layout,
                (source_walk.source_layout[-1],),
                emit_select,
                prepared,
            )
        )

        def emit_walk(
            control: int,
            ports: Mapping[str, tuple[int, ...]],
            work: tuple[int, ...],
            value: int | None,
            adjoint: bool,
        ) -> tuple[Instruction, ...]:
            assert value in (0, 1)
            temp, scratch = work[:2], work[2:]
            return walk.make_on(
                selection={name: ports[name] for name, _ in prep.selection_layout},
                target=ports["target"],
                signal_junk=temp,
                select_workspace=scratch[:select_work],
                select_classical_bits=classical_bits[:select_slots],
                prepare_workspace=scratch[:prep_work],
                prepare_classical_bits=classical_bits[:prep_slots],
                reflection_workspace=scratch[:reflect_work],
                reflection_classical_bits=classical_bits[:reflect_slots],
                adjoint=adjoint,
                controls=((control,),),
                control_registers=(ControlRegister(1, (value,)),),
                controlled_select=emit_controlled,
            ).instructions

        if len(reflection_temp) != 2:
            raise ValueError("Hubbard GQSP reflection temp must contain two wires")
        return HamiltonianGQSP(
            source_walk.source_layout, emit_walk, prepared, self.time, self.precision
        ).make_on(
            signal=signal,
            target=target,
            prepare_junk=prepare_temp,
            prepare_workspace=workspace[:prep_work],
            prepare_classical_bits=classical_bits[:prep_slots],
            walk_workspace=(*reflection_temp, *workspace),
        )

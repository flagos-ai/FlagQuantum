# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 LinearCombination.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/block_encoding/linear_combination.py

"""Compose coefficient PREPARE, signed SELECT and PREPARE-adjoint."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import cast

from ...circuit import _CircuitOperation
from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
)
from ...compiler._component_trace.symbolic_math import SymbolicInt, SymbolicReal, smax
from ...compiler.registers.control_predicate import (
    ControlRegister,
    bind_control_registers,
)
from ...compiler.registers.controlled_component import (
    controlled_component_ir,
    extend_controlled_component_ir,
)
from ...compiler.registers.controlled_component_resource_model import (
    controlled_component_cost,
    extended_controlled_component_cost,
)
from ...compiler.registers.partition import partition_register
from ...core.ir import Instruction
from ._block_binding import _BoundBlockOperation, _validate_block_work
from ._block_metadata import _BlockMetadata
from .black_box_prepare import BlackBoxPrepare, _identity_prepare
from .linear_combination_preparation import (
    LinearCombinationPreparation,
    _rescaled_lcu_coefficients,
)
from .linear_combination_select import LinearCombinationSelect
from .select_block_encoding import ControlledSelectEmitter, SelectBlockEncoding


@dataclass(frozen=True)
class LinearCombinationBlockEncoding:
    """Source linear combination with default or supplied packed oracles.

    Ancilla packs coefficient selection then constituent signal ancilla.
    Resource packs constituent resource then coefficient preparation junk.
    Child factories own their positive-controlled directional implementations.
    """

    block_encodings: tuple[_BlockMetadata, ...]
    coefficients: tuple[float, ...]
    coefficient_bits: SymbolicInt
    prepare_override: BlackBoxPrepare | None = None
    select_override: SelectBlockEncoding | None = None
    controlled_select: ControlledSelectEmitter | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_encodings", tuple(self.block_encodings))
        object.__setattr__(self, "coefficients", tuple(self.coefficients))
        self.select  # Validate child signatures, coefficients and signal states.
        if (
            self.prepare_override is None
            and isinstance(self.coefficient_bits, (int, float))
            and (type(self.coefficient_bits) is not int or self.coefficient_bits < 1)
        ):
            raise ValueError("LCU coefficient precision must be a positive integer")
        if self.controlled_select is not None and self.select_override is None:
            raise ValueError("Controlled custom SELECT requires a SELECT override")
        if self.prepare_override is not None or self.select_override is not None:
            selection_width = (
                sum(width for _, width in self.select_override.selection_layout)
                if self.select_override is not None
                else self.select.selection_bitsize
            )
            if self.prepare.selection_width != selection_width:
                raise ValueError(
                    "Custom PREPARE and SELECT must have equal packed selection widths"
                )
            target_width = (
                self.system_bitsize
                + self.select.ancilla_bitsize
                + self.select.resource_bitsize
            )
            if (
                self.select_override is not None
                and self.select_override.target_layout != (("system", target_width),)
            ):
                raise ValueError(
                    "Custom SELECT must pack system, constituent ancilla and resource as system"
                )

    @cached_property
    def select(self) -> LinearCombinationSelect:
        return LinearCombinationSelect(self.block_encodings, self.coefficients)

    def _concrete_normalizations(self) -> tuple[float, ...]:
        alphas = tuple(block.alpha for block in self.block_encodings)
        if any(not isinstance(alpha, (int, float)) for alpha in alphas):
            raise ValueError(
                "Default LCU preparation requires concrete block normalizations"
            )
        return cast(tuple[float, ...], alphas)

    @cached_property
    def coefficient_preparation(self) -> LinearCombinationPreparation:
        if type(self.coefficient_bits) is not int:
            raise ValueError(
                "Default LCU preparation requires concrete coefficient precision"
            )
        return LinearCombinationPreparation(
            self.coefficients, self._concrete_normalizations(), self.coefficient_bits
        )

    @property
    def prepare(self) -> BlackBoxPrepare:
        """Expose the supplied packed preparation or the source default factory."""
        return (
            self.coefficient_preparation.prepare
            if self.prepare_override is None
            else self.prepare_override
        )

    @property
    def system_bitsize(self) -> int:
        return self.select.system_bitsize

    @property
    def ancilla_bitsize(self) -> int:
        return self.select.ancilla_bitsize + self.prepare.selection_width

    @property
    def resource_bitsize(self) -> int:
        return self.select.resource_bitsize + self.prepare.junk_width

    @property
    def alpha(self) -> SymbolicReal:
        return sum(
            abs(coefficient) * block.alpha
            for coefficient, block in zip(
                self.coefficients, self.block_encodings, strict=True
            )
        )

    @property
    def epsilon(self) -> SymbolicReal:
        # Retain the implementation's normalized weights, not the docstring's
        # unnormalized coefficient bound.
        return sum(
            abs(weight)
            for weight in _rescaled_lcu_coefficients(
                self.coefficients, self._concrete_normalizations()
            )
        ) * smax(block.epsilon for block in self.block_encodings)

    @property
    def signal_state(self) -> BlackBoxPrepare:
        return BlackBoxPrepare(
            (("ancilla", self.ancilla_bitsize),), (), _identity_prepare
        )

    def resource_model(
        self,
        select_cost: ComponentCost,
        prepare_cost: ComponentCost,
        unprepare_cost: ComponentCost,
        *,
        controlled: bool = False,
        existing_control: bool = False,
        adjoint: bool = False,
        control_registers: tuple[ControlRegister, ...] = (),
        legacy_shims: bool = False,
    ) -> ComponentCost:
        """Compose directional child models and live bystanders, without emission.

        Supply the same selected direction and QEC profile in the child models.
        Structured controls require the positive singly-controlled SELECT cost.
        legacy_shims selects predicate costs; supplied children use that profile.
        existing_control adds the source conjunction to a precontrolled LCU.
        Original generic adjoint child-cost failures remain caller-visible.
        """
        if (
            type(controlled) is not bool
            or type(existing_control) is not bool
            or type(adjoint) is not bool
            or type(legacy_shims) is not bool
        ):
            raise ValueError(
                "LCU model control, direction and QEC profile must be booleans"
            )
        pattern = tuple(
            bit for register in control_registers for bit in register.bits()
        )
        if control_registers and not pattern:
            raise ValueError("Structured LCU control requires nonempty control words")
        if existing_control and not pattern:
            raise ValueError(
                "Existing LCU control extension requires new control words"
            )
        if pattern:
            if controlled:
                raise ValueError(
                    "Choose direct positive control or structured LCU controls"
                )
            positive = self.resource_model(
                select_cost,
                prepare_cost,
                unprepare_cost,
                controlled=True,
                adjoint=adjoint,
                legacy_shims=legacy_shims,
            )
            payload = self.system_bitsize + self.ancilla_bitsize + self.resource_bitsize
            if existing_control:
                return extended_controlled_component_cost(
                    positive,
                    control_registers,
                    current_control_value=1,
                    target_width=payload,
                    legacy_shims=legacy_shims,
                )
            return controlled_component_cost(
                positive,
                control_registers,
                target_width=payload,
                legacy_shims=legacy_shims,
            )
        prepare = self.prepare
        if self.ancilla_bitsize == 0:
            raise ValueError(
                "Original LCU decomposition requires a nonempty ancilla port"
            )
        prep_ports = prepare.selection_width + prepare.junk_width
        ports = (
            self.system_bitsize
            + self.ancilla_bitsize
            + self.resource_bitsize
            + int(controlled)
        )
        select_ports = ports - prepare.junk_width
        if (
            prepare_cost.qubits < prep_ports
            or unprepare_cost.qubits < prep_ports
            or select_cost.qubits < select_ports
        ):
            raise ValueError("LCU child model omits its through ports")
        return ComponentCost(
            "LinearCombination†" if adjoint else "LinearCombination",
            ports,
            calls=(
                CostCall(prepare_cost, bystander_qubits=ports - prep_ports),
                CostCall(select_cost, bystander_qubits=prepare.junk_width),
                CostCall(unprepare_cost, bystander_qubits=ports - prep_ports),
            ),
        )

    def make_on(
        self,
        bind_children: tuple[
            Callable[
                [
                    int,
                    Mapping[str, tuple[int, ...]],
                    tuple[int, ...],
                    tuple[int, ...],
                    bool,
                ],
                _CircuitOperation,
            ],
            ...,
        ] = (),
        *,
        system: tuple[int, ...],
        ancilla: tuple[int, ...],
        resource: tuple[int, ...],
        control: tuple[int, ...] = (),
        existing_control: tuple[int, ...] = (),
        prepare_workspace: tuple[int, ...],
        prepare_classical_bits: tuple[int, ...],
        traversal_workspace: tuple[int, ...],
        traversal_classical_bits: tuple[int, ...],
        workspaces: tuple[tuple[int, ...], ...] = (),
        classical_bits: tuple[tuple[int, ...], ...] = (),
        adjoint: bool = False,
        controls: tuple[tuple[int, ...], ...] = (),
        control_registers: tuple[ControlRegister, ...] = (),
        control_workspace: tuple[int, ...] = (),
        control_classical_bits: tuple[int, ...] = (),
    ) -> _BoundBlockOperation:
        """Bind the original conjugation for circuit.append(op).

        PREPARE is unconditional even when SELECT has an external control.
        Private work may be shared across nonoverlapping PREPARE/SELECT calls.
        Logical selection, junk and signal outputs are never treated as scratch.
        Structured controls keep one predicate live around the whole block,
        separate from all stage scratch. existing_control binds an already
        positive-controlled LCU before adding the structured controls. With
        adjoint=True this takes the adjoint of that specialized root, not a
        generic control wrapper around an already-adjoint operation.
        """
        if len(existing_control) > 1 or (existing_control and control):
            raise ValueError(
                "LCU extension needs one existing positive control and no direct control"
            )
        outer, pattern = bind_control_registers(controls, control_registers)
        if not pattern and (
            controls
            or control_registers
            or control_workspace
            or control_classical_bits
            or existing_control
        ):
            raise ValueError(
                "Structured LCU controls require nonempty activation words"
            )
        if pattern:
            if control:
                raise ValueError(
                    "Choose direct positive control or structured LCU controls"
                )
            logical = (*outer, *existing_control, *system, *ancilla, *resource)
            _validate_block_work(logical, control_workspace, control_classical_bits)
            for work, bits in (
                (prepare_workspace, prepare_classical_bits),
                (traversal_workspace, traversal_classical_bits),
                *tuple(zip(workspaces, classical_bits, strict=True)),
            ):
                _validate_block_work((*logical, *control_workspace), work, bits)

            def emit(predicate: int) -> tuple[Instruction, ...]:
                return self.make_on(
                    bind_children,
                    system=system,
                    ancilla=ancilla,
                    resource=resource,
                    control=(predicate,),
                    prepare_workspace=prepare_workspace,
                    prepare_classical_bits=prepare_classical_bits,
                    traversal_workspace=traversal_workspace,
                    traversal_classical_bits=traversal_classical_bits,
                    workspaces=workspaces,
                    classical_bits=classical_bits,
                    adjoint=adjoint,
                ).instructions

            gates = (
                extend_controlled_component_ir(
                    controls,
                    control_registers,
                    (existing_control[0], 1),
                    emit,
                    control_workspace,
                    control_classical_bits,
                )
                if existing_control
                else controlled_component_ir(
                    controls,
                    control_registers,
                    emit,
                    control_workspace,
                    control_classical_bits,
                )
            )
            required = (
                max(
                    (
                        *logical,
                        *control_workspace,
                        *prepare_workspace,
                        *traversal_workspace,
                        *(q for work in workspaces for q in work),
                    ),
                    default=-1,
                )
                + 1
            )
            return _BoundBlockOperation(gates, required)
        if (
            type(adjoint) is not bool
            or len(system) != self.system_bitsize
            or len(control) > 1
        ):
            raise ValueError(
                "LCU needs matching system width, boolean direction and at most one control"
            )
        prep = self.prepare
        if self.ancilla_bitsize == 0:
            raise ValueError(
                "Original LCU decomposition requires a nonempty ancilla port"
            )
        anc = partition_register(
            ancilla,
            (
                ("selection", prep.selection_width),
                ("ancilla", self.select.ancilla_bitsize),
            ),
        )
        res = partition_register(
            resource,
            (("resource", self.select.resource_bitsize), ("junk", prep.junk_width)),
        )
        through = (*control, *system, *ancilla, *resource)
        _validate_block_work(through, prepare_workspace, prepare_classical_bits)
        _validate_block_work(through, traversal_workspace, traversal_classical_bits)
        for work, bits in zip(workspaces, classical_bits, strict=True):
            _validate_block_work(through, work, bits)
        if self.select_override is not None:
            if bind_children or workspaces or classical_bits:
                raise ValueError(
                    "Custom SELECT owns its binding; child factories/work groups must be empty"
                )
            if control and self.controlled_select is None:
                raise ValueError(
                    "Controlled custom SELECT needs its positive-controlled emitter"
                )
        before = prep.emit(
            anc["selection"], res["junk"], prepare_workspace, prepare_classical_bits
        )
        if self.select_override is None:
            middle = self.select.make_on(
                bind_children,
                selection=anc["selection"],
                system=system,
                ancilla=anc["ancilla"],
                resource=res["resource"],
                control=control,
                traversal_workspace=traversal_workspace,
                traversal_classical_bits=traversal_classical_bits,
                workspaces=workspaces,
                classical_bits=classical_bits,
                adjoint=adjoint,
            )
        else:
            selected = self.select_override
            if control and self.controlled_select is not None:
                selected = selected.with_control(1, self.controlled_select)
            middle_gates = selected.emit(
                (*control, *anc["selection"]),
                (*system, *anc["ancilla"], *res["resource"]),
                traversal_workspace,
                traversal_classical_bits,
                adjoint=adjoint,
            )
            middle = _BoundBlockOperation(
                middle_gates, max((*through, *traversal_workspace), default=-1) + 1
            )
        after = prep.emit(
            anc["selection"],
            res["junk"],
            prepare_workspace,
            prepare_classical_bits,
            adjoint=True,
        )
        required = max(
            middle.required_qubits, max((*through, *prepare_workspace), default=-1) + 1
        )
        return _BoundBlockOperation((*before, *middle.instructions, *after), required)

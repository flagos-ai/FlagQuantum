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
from ...compiler._component_trace.resource_model import ComponentCost, CostCall
from ...compiler._component_trace.symbolic_math import SymbolicReal, smax
from ...compiler.registers.partition import partition_register
from ._block_binding import _BoundBlockOperation, _validate_block_work
from ._block_metadata import _BlockMetadata
from .black_box_prepare import BlackBoxPrepare, _identity_prepare
from .linear_combination_preparation import LinearCombinationPreparation
from .linear_combination_select import LinearCombinationSelect


@dataclass(frozen=True)
class LinearCombinationBlockEncoding:
    """Source default-oracle linear combination on packed through registers.

    Ancilla packs coefficient selection then constituent signal ancilla.
    Resource packs constituent resource then coefficient preparation junk.
    Child factories own their positive-controlled directional implementations.
    """

    block_encodings: tuple[_BlockMetadata, ...]
    coefficients: tuple[float, ...]
    coefficient_bits: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_encodings", tuple(self.block_encodings))
        object.__setattr__(self, "coefficients", tuple(self.coefficients))
        self.select  # Validate child signatures, coefficients and signal states.
        if type(self.coefficient_bits) is not int or self.coefficient_bits < 1:
            raise ValueError("LCU coefficient precision must be a positive integer")

    @cached_property
    def select(self) -> LinearCombinationSelect:
        return LinearCombinationSelect(self.block_encodings, self.coefficients)

    @cached_property
    def coefficient_preparation(self) -> LinearCombinationPreparation:
        alphas = tuple(block.alpha for block in self.block_encodings)
        if any(not isinstance(alpha, (int, float)) for alpha in alphas):
            raise ValueError(
                "Default LCU preparation requires concrete block normalizations"
            )
        return LinearCombinationPreparation(
            self.coefficients, cast(tuple[float, ...], alphas), self.coefficient_bits
        )

    @property
    def system_bitsize(self) -> int:
        return self.select.system_bitsize

    @property
    def ancilla_bitsize(self) -> int:
        return (
            self.select.ancilla_bitsize
            + self.coefficient_preparation.prepare.selection_width
        )

    @property
    def resource_bitsize(self) -> int:
        return (
            self.select.resource_bitsize
            + self.coefficient_preparation.prepare.junk_width
        )

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
            abs(weight) for weight in self.coefficient_preparation.rescaled_coefficients
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
        adjoint: bool = False,
    ) -> ComponentCost:
        """Compose directional child models and live bystanders, without emission.

        Supply the same selected direction and QEC profile in the child models.
        Original generic adjoint child-cost failures remain caller-visible.
        """
        if type(controlled) is not bool or type(adjoint) is not bool:
            raise ValueError("LCU model control and direction must be boolean")
        prepare = self.coefficient_preparation.prepare
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
        ],
        *,
        system: tuple[int, ...],
        ancilla: tuple[int, ...],
        resource: tuple[int, ...],
        control: tuple[int, ...] = (),
        prepare_workspace: tuple[int, ...],
        prepare_classical_bits: tuple[int, ...],
        traversal_workspace: tuple[int, ...],
        traversal_classical_bits: tuple[int, ...],
        workspaces: tuple[tuple[int, ...], ...],
        classical_bits: tuple[tuple[int, ...], ...],
        adjoint: bool = False,
    ) -> _BoundBlockOperation:
        """Bind the original conjugation for circuit.append(op).

        PREPARE is unconditional even when SELECT has an external control.
        Private work may be shared across nonoverlapping PREPARE/SELECT calls.
        Logical selection, junk and signal outputs are never treated as scratch.
        """
        if (
            type(adjoint) is not bool
            or len(system) != self.system_bitsize
            or len(control) > 1
        ):
            raise ValueError(
                "LCU needs matching system width, boolean direction and at most one control"
            )
        prep = self.coefficient_preparation.prepare
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
        before = prep.emit(
            anc["selection"], res["junk"], prepare_workspace, prepare_classical_bits
        )
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

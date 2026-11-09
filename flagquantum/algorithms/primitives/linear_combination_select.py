# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 LinearCombination.signed_block_encodings/select.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/block_encoding/linear_combination.py

"""Select signed, padded block encodings through shared register traversal."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite

from ...circuit import _CircuitOperation
from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler.registers.control.indexed_component import indexed_component_ir
from ...compiler.registers.control.indexed_component_resource_model import (
    indexed_component_cost,
)
from ...compiler.registers.layout.auto_partition import AutoPartition, PartitionGroups
from ...core.ir import Instruction
from ._block_binding import _BoundBlockOperation, _capture_block, _validate_block_work
from ._block_metadata import _BlockMetadata
from .black_box_prepare import _identity_prepare


@dataclass(frozen=True)
class LinearCombinationSelect:
    """Retain term order, sign phase and unused child port tails.

    Child factories provide their existing positive-controlled operation and
    own its directional decomposition. They must preserve the predicate and
    clean their private workspace, including any measured uncomputation.
    """

    block_encodings: tuple[_BlockMetadata, ...]
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_encodings", tuple(self.block_encodings))
        object.__setattr__(self, "coefficients", tuple(self.coefficients))
        if len(self.block_encodings) < 2 or len(self.coefficients) != len(
            self.block_encodings
        ):
            raise ValueError(
                "Signed SELECT needs at least two paired blocks and coefficients"
            )
        if any(not isfinite(value) for value in self.coefficients) or not any(
            self.coefficients
        ):
            raise ValueError(
                "Signed SELECT coefficients must be finite and not all zero"
            )
        if any(
            block.system_bitsize != self.system_bitsize
            for block in self.block_encodings
        ):
            raise ValueError("Signed SELECT blocks must share the system width")
        if any(
            block.signal_state.prepare is not _identity_prepare
            for block in self.block_encodings
        ):
            raise ValueError("Signed SELECT requires zero-signal constituent blocks")

    @property
    def selection_bitsize(self) -> int:
        return (len(self.block_encodings) - 1).bit_length()

    @property
    def system_bitsize(self) -> int:
        return self.block_encodings[0].system_bitsize

    @property
    def ancilla_bitsize(self) -> int:
        return max(block.ancilla_bitsize for block in self.block_encodings)

    @property
    def resource_bitsize(self) -> int:
        return max(block.resource_bitsize for block in self.block_encodings)

    def _partition(self, index: int) -> AutoPartition:
        block = self.block_encodings[index]
        groups: list[tuple[str, tuple[tuple[str | None, int], ...]]] = []
        for name in ("system", "ancilla", "resource"):
            width = getattr(self, name + "_bitsize")
            child_width = getattr(block, name + "_bitsize")
            if width:
                segments: tuple[tuple[str | None, int], ...] = (
                    ((name, child_width),) if child_width else ()
                )
                if width > child_width:
                    segments += ((None, width - child_width),)
                groups.append((name, segments))
        typed: PartitionGroups = tuple(groups)
        return AutoPartition(typed)

    def resource_model(
        self,
        children: tuple[ComponentCost, ...],
        *,
        controlled: bool = False,
        adjoint: bool = False,
        legacy_shims: bool = False,
    ) -> ComponentCost:
        """Pad supplied controlled child models and retain controlled sign costs."""
        if len(children) != len(self.block_encodings) or type(controlled) is not bool:
            raise ValueError("Signed SELECT needs one controlled model per child")
        target_width = (
            self.system_bitsize + self.ancilla_bitsize + self.resource_bitsize
        )
        padded = []
        for block, coefficient, model in zip(
            self.block_encodings, self.coefficients, children, strict=True
        ):
            child_width = (
                block.system_bitsize + block.ancilla_bitsize + block.resource_bitsize
            )
            if model.qubits < child_width + 1:
                raise ValueError("Child cost omits its predicate or quantum ports")
            calls = [CostCall(model, bystander_qubits=target_width - child_width)]
            if coefficient < 0:
                sign = CostCall(
                    ComponentCost("ControlledGlobalPhase(1)", 1, GateCost(clifford=1)),
                    bystander_qubits=target_width,
                )
                calls.append(sign)
            padded.append(
                ComponentCost("PaddedSignedBlock", target_width + 1, calls=tuple(calls))
            )
        return indexed_component_cost(
            (self.selection_bitsize,),
            (len(children),),
            tuple(padded),
            target_width=target_width,
            control_count=int(controlled),
            adjoint=adjoint,
            legacy_shims=legacy_shims,
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
        selection: tuple[int, ...],
        system: tuple[int, ...],
        ancilla: tuple[int, ...] = (),
        resource: tuple[int, ...] = (),
        control: tuple[int, ...] = (),
        traversal_workspace: tuple[int, ...],
        traversal_classical_bits: tuple[int, ...],
        workspaces: tuple[tuple[int, ...], ...],
        classical_bits: tuple[tuple[int, ...], ...],
        adjoint: bool = False,
    ) -> _BoundBlockOperation:
        """Bind original bounded traversal and caller-owned controlled children.

        A single positive external control is supported. Padded addresses retain
        the locked bounded-tree aliases; they are not changed to identity.
        """
        count = len(self.block_encodings)
        if (
            type(adjoint) is not bool
            or len(control) > 1
            or any(
                len(group) != count
                for group in (bind_children, workspaces, classical_bits)
            )
        ):
            raise ValueError(
                "Signed SELECT needs one binding/work/slot group per child and at most one control"
            )
        if any(
            len(word) != width
            for word, width in (
                (selection, self.selection_bitsize),
                (system, self.system_bitsize),
                (ancilla, self.ancilla_bitsize),
                (resource, self.resource_bitsize),
            )
        ):
            raise ValueError("Signed SELECT packed widths do not match block metadata")
        through = (*control, *selection, *system, *ancilla, *resource)
        _validate_block_work(through, traversal_workspace, traversal_classical_bits)
        for work, bits in zip(workspaces, classical_bits, strict=True):
            _validate_block_work((*through, *traversal_workspace), work, bits)
        required = (
            max(
                (
                    *through,
                    *traversal_workspace,
                    *(q for work in workspaces for q in work),
                ),
                default=-1,
            )
            + 1
        )
        packed = {"system": system}
        if ancilla:
            packed["ancilla"] = ancilla
        if resource:
            packed["resource"] = resource

        def emit(index: int, predicate: int) -> tuple[Instruction, ...]:
            ports, _ = self._partition(index).bind(packed)
            allowed = {
                predicate,
                *(q for word in ports.values() for q in word),
                *workspaces[index],
            }
            child = bind_children[index](
                predicate, ports, workspaces[index], classical_bits[index], adjoint
            )
            gates = _capture_block(child, required, allowed)
            phase = (
                (Instruction("z", (predicate,)),)
                if self.coefficients[index] < 0
                else ()
            )
            return (*gates, *phase)

        instructions = indexed_component_ir(
            (selection,),
            (count,),
            emit,
            controls=control,
            workspace=traversal_workspace,
            classical_bits=traversal_classical_bits,
            adjoint=adjoint,
        )
        return _BoundBlockOperation(instructions, required)

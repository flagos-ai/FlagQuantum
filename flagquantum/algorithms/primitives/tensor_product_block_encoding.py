# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 TensorProduct.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/block_encoding/tensor_product.py

"""Compose disjoint native block encodings through packed source partitions."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import prod
from typing import cast

from ...circuit import _CircuitOperation
from ...compiler._component_trace.resource_model import ComponentCost, CostCall
from ...compiler._component_trace.symbolic_math import SymbolicReal
from ...compiler.registers.bits.bitwise import _validate_wires
from ...compiler.registers.layout.partition import partition_register
from ...core.ir import Instruction
from ._block_binding import _BoundBlockOperation, _capture_block, _validate_block_work
from ._block_metadata import _BlockMetadata
from .black_box_prepare import BlackBoxPrepare, _identity_prepare


@dataclass(frozen=True)
class TensorProductBlockEncoding:
    """Retain packed register order and independent child work ownership."""

    block_encodings: tuple[_BlockMetadata, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_encodings", tuple(self.block_encodings))
        if not self.block_encodings:
            raise ValueError("Tensor product requires at least one block")

    @property
    def system_bitsize(self) -> int:
        return sum(child.system_bitsize for child in self.block_encodings)

    @property
    def ancilla_bitsize(self) -> int:
        return sum(child.ancilla_bitsize for child in self.block_encodings)

    @property
    def resource_bitsize(self) -> int:
        return sum(child.resource_bitsize for child in self.block_encodings)

    @property
    def alpha(self) -> SymbolicReal:
        return cast(SymbolicReal, prod(child.alpha for child in self.block_encodings))

    @property
    def epsilon(self) -> SymbolicReal:
        return sum(child.alpha * child.epsilon for child in self.block_encodings)

    @property
    def signal_state(self) -> BlackBoxPrepare:
        if any(
            child.signal_state.prepare is not _identity_prepare
            for child in self.block_encodings
        ):
            raise NotImplementedError(
                "Tensor product signal requires identity child preparations"
            )
        return BlackBoxPrepare(
            (("ancilla", self.ancilla_bitsize),), (), _identity_prepare
        )

    def resource_model(
        self, children: tuple[ComponentCost, ...], *, adjoint: bool = False
    ) -> ComponentCost:
        """Use one directional model per child, retaining all other live ports."""
        if len(children) != len(self.block_encodings) or type(adjoint) is not bool:
            raise ValueError(
                "Tensor product needs one child model and a boolean direction"
            )
        total = self.system_bitsize + self.ancilla_bitsize + self.resource_bitsize
        calls = []
        indices = reversed(range(len(children))) if adjoint else range(len(children))
        for index in indices:
            block, model = self.block_encodings[index], children[index]
            ports = (
                block.system_bitsize + block.ancilla_bitsize + block.resource_bitsize
            )
            if model.qubits < ports:
                raise ValueError("Tensor child model omits its through ports")
            calls.append(CostCall(model, bystander_qubits=total - ports))
        return ComponentCost(
            "TensorProduct†" if adjoint else "TensorProduct", total, calls=tuple(calls)
        )

    def make_on(
        self,
        bind_children: tuple[
            Callable[
                [Mapping[str, tuple[int, ...]], tuple[int, ...], tuple[int, ...], bool],
                _CircuitOperation,
            ],
            ...,
        ],
        *,
        system: tuple[int, ...],
        ancilla: tuple[int, ...] = (),
        resource: tuple[int, ...] = (),
        workspaces: tuple[tuple[int, ...], ...],
        classical_bits: tuple[tuple[int, ...], ...],
        adjoint: bool = False,
    ) -> _BoundBlockOperation:
        """Bind and capture children without a numerical execution path.

        Work/results are per child and may be reused between sequential calls.
        Each factory binds its child in the requested direction. All packed
        through ports are disjoint; each child's emitted IR must stay in its
        partition plus that child's workspace.
        """
        count = len(self.block_encodings)
        if type(adjoint) is not bool or any(
            len(group) != count for group in (bind_children, workspaces, classical_bits)
        ):
            raise ValueError(
                "Tensor product needs one binding/work/result group per child and boolean direction"
            )
        packed = {"system": system, "ancilla": ancilla, "resource": resource}
        through = (*system, *ancilla, *resource)
        if through:
            _validate_wires(through)
        parts = {
            name: partition_register(
                word,
                tuple(
                    (str(index), getattr(child, name + "_bitsize"))
                    for index, child in enumerate(self.block_encodings)
                ),
            )
            for name, word in packed.items()
        }
        for work, bits in zip(workspaces, classical_bits, strict=True):
            _validate_block_work(through, work, bits)
        all_wires = (*through, *(wire for work in workspaces for wire in work))
        if not all_wires:
            raise ValueError(
                "Tensor numerical binding requires an existing quantum wire"
            )
        required = max(all_wires) + 1
        gates: list[Instruction] = []
        indices = reversed(range(count)) if adjoint else range(count)
        for index in indices:
            ports = {name: words[str(index)] for name, words in parts.items()}
            operation = bind_children[index](
                ports, workspaces[index], classical_bits[index], adjoint
            )
            allowed = {wire for word in ports.values() for wire in word} | set(
                workspaces[index]
            )
            gates.extend(_capture_block(operation, required, allowed))
        return _BoundBlockOperation(tuple(gates), required)

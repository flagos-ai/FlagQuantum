# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Product.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/block_encoding/product.py

"""Source ordered block products with shared ancilla and explicit success flags."""

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from math import prod
from typing import Literal, cast

from ...circuit import _CircuitOperation
from ...compiler._component_trace.qec_gate_cost import query_qec_gate_cost
from ...compiler._component_trace.resource_generalization import (
    CliffordCall,
    PredicateControlledCall,
)
from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler._component_trace.symbolic_math import SymbolicReal
from ...compiler.registers.bitwise import _validate_wires
from ...compiler.registers.multi_control import multi_control_x_ir
from ...compiler.registers.partition import partition_register
from ...core.ir import Instruction
from ._block_binding import _BoundBlockOperation, _capture_block, _validate_block_work
from ._block_metadata import _BlockMetadata
from .black_box_prepare import BlackBoxPrepare, _identity_prepare


@dataclass(frozen=True)
class ProductBlockEncoding:
    """Encode tuple-order operator multiplication without flattening child calls."""

    block_encodings: tuple[_BlockMetadata, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_encodings", tuple(self.block_encodings))
        if not self.block_encodings:
            raise ValueError("Product requires at least one block")
        if any(
            child.system_bitsize != self.system_bitsize
            for child in self.block_encodings
        ):
            raise ValueError("Product blocks must share the same system width")
        if any(
            child.signal_state.prepare is not _identity_prepare
            for child in self.block_encodings
        ):
            raise ValueError("Product requires identity child signal preparation")

    @property
    def system_bitsize(self) -> int:
        return self.block_encodings[0].system_bitsize

    @property
    def ancilla_bitsize(self) -> int:
        return (
            max(child.ancilla_bitsize for child in self.block_encodings)
            + len(self.block_encodings)
            - 1
        )

    @property
    def resource_bitsize(self) -> int:
        return max(child.resource_bitsize for child in self.block_encodings)

    @property
    def alpha(self) -> SymbolicReal:
        return cast(SymbolicReal, prod(child.alpha for child in self.block_encodings))

    @property
    def epsilon(self) -> SymbolicReal:
        return sum(child.alpha * child.epsilon for child in self.block_encodings)

    @property
    def signal_state(self) -> BlackBoxPrepare:
        return BlackBoxPrepare(
            (("ancilla", self.ancilla_bitsize),), (), _identity_prepare
        )

    def _check_resource_partition(self) -> None:
        if any(
            0 < child.resource_bitsize < self.resource_bitsize
            for child in self.block_encodings
        ):
            raise ValueError(
                "Locked Product cannot partition unequal nonzero resource widths"
            )

    def resource_model(
        self,
        children: tuple[ComponentCost, ...],
        *,
        adjoint: bool = False,
        legacy_shims: bool = False,
        flag_model: Literal["call_graph", "decomposition"] = "call_graph",
    ) -> ComponentCost:
        """Preserve either source flag-cost route, separately from emitted IR.

        The direct call graph replaces flag polarities by ones. The executable
        decomposition retains zeros; caller-supplied child models must follow
        the requested QEC profile and direction.
        """
        self._check_resource_partition()
        if (
            type(adjoint) is not bool
            or type(legacy_shims) is not bool
            or len(children) != len(self.block_encodings)
            or flag_model not in ("call_graph", "decomposition")
        ):
            raise ValueError(
                "Product needs directional child costs, boolean profile and a source flag model"
            )
        total = self.system_bitsize + self.ancilla_bitsize + self.resource_bitsize
        calls = []
        order = range(len(children)) if adjoint else reversed(range(len(children)))
        for index in order:
            block, child = self.block_encodings[index], children[index]
            ports = (
                block.system_bitsize + block.ancilla_bitsize + block.resource_bitsize
            )
            if child.qubits < ports:
                raise ValueError("Product child model omits its through ports")
            flag = None
            if index > 0 and block.ancilla_bitsize:
                size = block.ancilla_bitsize
                values = ((1,) if flag_model == "call_graph" else (0,)) * size
                cost = query_qec_gate_cost(
                    PredicateControlledCall(CliffordCall("cx", 2), values),
                    legacy_shims=legacy_shims,
                )
                gates = GateCost(
                    **{name: int(value) for name, value in asdict(cost).items()}
                ).plus(GateCost(clifford=1), 1)
                flag = CostCall(
                    ComponentCost("SuccessFlag", 2 * size, gates),
                    bystander_qubits=total - size - 1,
                )
            if adjoint and flag is not None:
                calls.append(flag)
            calls.append(CostCall(child, bystander_qubits=total - ports))
            if not adjoint and flag is not None:
                calls.append(flag)
        return ComponentCost(
            f"Product[{flag_model}]" + ("†" if adjoint else ""),
            total,
            calls=tuple(calls),
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
        flag_workspace: tuple[int, ...] = (),
        flag_classical_bits: tuple[int, ...] = (),
        adjoint: bool = False,
    ) -> _BoundBlockOperation:
        """Bind shared source words and child-private work for circuit.append.

        Flag work and child work may overlap between sequential lifetimes.
        Resource registers obey the source zero-restoration assumption.
        """
        self._check_resource_partition()
        count = len(self.block_encodings)
        if type(adjoint) is not bool or any(
            len(group) != count for group in (bind_children, workspaces, classical_bits)
        ):
            raise ValueError(
                "Product needs one binding/work/result group per child and boolean direction"
            )
        partition_register(system, (("system", self.system_bitsize),))
        if (
            len(ancilla) != self.ancilla_bitsize
            or len(resource) != self.resource_bitsize
        ):
            raise ValueError("Product auxiliary ports do not match declared widths")
        through = (*system, *ancilla, *resource)
        if through:
            _validate_wires(through)
        flags, shared = ancilla[: count - 1], ancilla[count - 1 :]
        need = max(
            (max(0, b.ancilla_bitsize - 1) for b in self.block_encodings[1:]), default=0
        )
        if len(flag_workspace) != need or len(flag_classical_bits) != need:
            raise ValueError(
                f"Product flags require {need} clean work wires and result slots"
            )
        for work, bits in zip(
            (*workspaces, flag_workspace),
            (*classical_bits, flag_classical_bits),
            strict=True,
        ):
            _validate_block_work(through, work, bits)
        all_wires = (
            *through,
            *(w for work in workspaces for w in work),
            *flag_workspace,
        )
        if not all_wires:
            raise ValueError(
                "Product numerical binding requires an existing quantum wire"
            )
        required = max(all_wires) + 1
        gates: list[Instruction] = []

        def flag(index: int, inverse: bool) -> None:
            child = self.block_encodings[index]
            size = child.ancilla_bitsize
            if index == 0 or size == 0:
                return
            target = flags[count - 1 - index]
            flip = Instruction("x", (target,))
            if inverse:
                gates.append(flip)
            gates.extend(
                multi_control_x_ir(
                    shared[:size],
                    (0,) * size,
                    target,
                    flag_workspace[: size - 1],
                    flag_classical_bits[: size - 1],
                )
            )
            if not inverse:
                gates.append(flip)

        order = range(count) if adjoint else reversed(range(count))
        for index in order:
            child = self.block_encodings[index]
            if adjoint:
                flag(index, True)
            ports = {
                "system": system,
                "ancilla": shared[: child.ancilla_bitsize],
                "resource": resource[: child.resource_bitsize],
            }
            operation = bind_children[index](
                ports, workspaces[index], classical_bits[index], adjoint
            )
            allowed = {wire for word in ports.values() for wire in word} | set(
                workspaces[index]
            )
            gates.extend(_capture_block(operation, required, allowed))
            if not adjoint:
                flag(index, False)
        return _BoundBlockOperation(tuple(gates), required)

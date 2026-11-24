# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 Product.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/block_encoding/product.py

"""Source ordered block products with shared ancilla and explicit success flags."""

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
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
from ...compiler._component_trace.symbolic_math import SymbolicInt, SymbolicReal, smax
from ...compiler.registers.bits.bitwise import _validate_wires
from ...compiler.registers.control.control_predicate_resource_model import (
    control_predicate_resource_formula,
)
from ...compiler.registers.control.multi_control import multi_control_x_ir
from ...compiler.registers.layout.partition import partition_register
from ...compiler.registers.layout.product_block_component import (
    check_product_resource_partition as _check_product_resource_partition,
)
from ...compiler.registers.layout.product_block_component import product_block_signature
from ...compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from ...core.ir import Instruction
from ...core.ir.program.signatures import Signature
from ._block_binding import _BoundBlockOperation, _capture_block, _validate_block_work
from ._block_metadata import _BlockMetadata
from .black_box_prepare import BlackBoxPrepare, _identity_prepare


def source_product_block_signature(
    system: SymbolicInt,
    ancillas: tuple[SymbolicInt, ...],
    resources: tuple[SymbolicInt, ...],
) -> Signature:
    """Declare the common system, packed success flags and shared child pools."""
    return product_block_signature(system, ancillas, resources)


def source_symbolic_product_resource_formula(
    system: SymbolicInt,
    ancillas: tuple[SymbolicInt, ...],
    resources: tuple[SymbolicInt, ...],
    children: tuple[SymbolicResourceFormula, ...],
    *,
    legacy_shims: bool = False,
) -> SymbolicResourceFormula:
    """Use the locked call-graph cost when symbolic widths prevent decomposition.

    The source peak falls back to the maximum over the signature and
    individual callees. It does not keep the entire product live across calls.
    The supplied child peak already includes its own through ports.
    """
    signature = source_product_block_signature(system, ancillas, resources)
    if len(children) != len(ancillas) or type(legacy_shims) is not bool:
        raise ValueError(
            "Product needs one child formula per declaration and a boolean profile"
        )
    if all(type(width) is int for width in (system, *ancillas, *resources)):
        raise ValueError("Use the concrete Product model for resolved widths")
    shared_ancilla = cast(SymbolicInt, smax(ancillas))
    _check_product_resource_partition(resources, cast(SymbolicInt, smax(resources)))
    peaks: list[SymbolicInt] = [signature.n_qubits()]
    counts = {
        field.name: sum(getattr(child, field.name) for child in children)
        for field in fields(GateCost)
    }
    for index, (child, width) in enumerate(zip(children, ancillas, strict=True)):
        if width == 0:
            peaks.append(child.logical_qubits)
            continue
        padding = shared_ancilla - width
        peaks.append(cast(SymbolicInt, smax(0, child.logical_qubits)) + padding)
        if index == 0:
            continue
        controls = (1,) * width if type(width) is int else width
        flag = query_qec_gate_cost(
            PredicateControlledCall(CliffordCall("cx", 2), controls),
            legacy_shims=legacy_shims,
        )
        for name in counts:
            counts[name] += getattr(flag, name)
        counts["clifford"] += 1
        if type(width) is int:
            peaks.append(2 * width)
        else:
            peaks.extend(
                (
                    width + 1,
                    control_predicate_resource_formula(
                        width, legacy_shims=legacy_shims
                    ).logical_qubits,
                )
            )
    return SymbolicResourceFormula(
        **counts, logical_qubits=cast(SymbolicInt, smax(peaks))
    )


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

    @classmethod
    def of(cls, *block_encodings: _BlockMetadata) -> "ProductBlockEncoding":
        """Construct a product from an ordered sequence of constituent blocks."""
        return cls(block_encodings)

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
    def signature(self) -> Signature:
        return source_product_block_signature(
            self.system_bitsize,
            tuple(child.ancilla_bitsize for child in self.block_encodings),
            tuple(child.resource_bitsize for child in self.block_encodings),
        )

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
        _check_product_resource_partition(
            tuple(child.resource_bitsize for child in self.block_encodings),
            self.resource_bitsize,
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

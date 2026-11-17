# Copyright 2023 Google LLC
# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the Qualtran v0.7.0 swap network examples.

"""Bind the original swap-network examples to FlagQuantum circuits."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler._component_trace.resource_decomposition import (
    decomposed_resource_callees,
)
from flagquantum.compiler._component_trace.resource_generalization import (
    CliffordCall,
    ControlledSwapCall,
    ControlPredicateCall,
)
from flagquantum.compiler._component_trace.resource_model import ComponentCost
from flagquantum.compiler._component_trace.resource_width import (
    query_resource_logical_width,
)
from flagquantum.compiler._component_trace.symbolic_math import SymbolicInt
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.lookup.multiplexed_swap import (
    multiplexed_cswap_ir,
    source_multiplexed_cswap_cost,
)
from flagquantum.compiler.registers.lookup.swap_with_zero import swap_with_zero_ir
from flagquantum.compiler.registers.primitives.relative_controlled_swap import (
    PositiveControlledSwapChild,
    control_relative_swap_ir,
    relative_controlled_swap_ir,
    source_outer_relative_swap_calls,
)
from flagquantum.compiler.registers.primitives.relative_controlled_swap import (
    relative_swap_signature as relative_swap_signature,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.register_types import (
    opaque_register_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature


def relative_swap_call_graph(width: SymbolicInt) -> SymbolicCallGraph[str]:
    root = "CSwapApprox"
    calls = {
        (
            f"ArbitraryClifford({child.width})"
            if isinstance(child, CliffordCall)
            else "T"
        ): count
        for child, count in decomposed_resource_callees(
            ControlledSwapCall(width, approximate=True), None
        )
    }
    return symbolic_call_graph(root, {root: calls}, max_depth=1)


def relative_swap_logical_width(width: SymbolicInt) -> SymbolicInt:
    return query_resource_logical_width(ControlledSwapCall(width, approximate=True))


@dataclass(frozen=True)
class RelativeSwapExample:
    width: int
    extra_controls: int | tuple[int, ...] | None = None
    adjoint: bool = False

    @property
    def _activation(self) -> tuple[int, ...]:
        if self.extra_controls is None:
            return ()
        if isinstance(self.extra_controls, tuple):
            return self.extra_controls
        return (self.extra_controls,)

    @property
    def signature(self) -> Signature:
        controls = self._activation
        ports = (
            (
                Register(
                    "ctrl2",
                    quantum_bit_type(),
                    shape=(len(controls),)
                    if isinstance(self.extra_controls, tuple)
                    else (),
                ),
            )
            if controls
            else ()
        )
        return relative_swap_signature(self.width, extra_controls=ports)

    @property
    def qubit_count(self) -> int:
        controls = len(self._activation)
        return 2 * self.width + 1 + controls + max(0, controls - 1)

    def outer_calls(
        self,
    ) -> dict[
        PositiveControlledSwapChild | ControlPredicateCall | CliffordCall, SymbolicInt
    ]:
        """Describe the added control layer before abstract resource expansion."""
        return source_outer_relative_swap_calls(
            self.width,
            (ControlRegister(1, self._activation),),
            scalar_control=isinstance(self.extra_controls, int),
            adjoint=self.adjoint,
        )

    def append_to(self, circuit: Circuit) -> Circuit:
        values = self._activation
        outside = len(values)
        left = tuple(range(outside + 1, outside + self.width + 1))
        right = tuple(range(outside + self.width + 1, outside + 2 * self.width + 1))
        if values:
            work = tuple(range(outside + 2 * self.width + 1, self.qubit_count))
            gates = control_relative_swap_ir(
                (tuple(range(outside)),),
                (ControlRegister(1, values),),
                outside,
                left,
                right,
                work,
                tuple(range(len(work))),
                adjoint=self.adjoint,
            )
        else:
            gates = relative_controlled_swap_ir(
                outside, left, right, adjoint=self.adjoint
            )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


def swap_with_zero_signature(
    widths: tuple[SymbolicInt, ...],
    word: SymbolicInt,
    shape: tuple[SymbolicInt, ...],
) -> Signature:
    live_widths = tuple(width for width in widths if width != 0)
    return Signature(
        (
            *(
                Register(
                    "selection" if len(live_widths) == 1 else f"selection{i}_",
                    opaque_register_type(width),
                )
                for i, width in enumerate(live_widths)
            ),
            Register("targets", opaque_register_type(word), shape=shape),
        )
    )


def swap_with_zero_call_graph(
    word: SymbolicInt, shape: tuple[SymbolicInt, ...]
) -> SymbolicCallGraph[str]:
    root = "SwapWithZero"
    return symbolic_call_graph(
        root, {root: {f"CSwapApprox({word})": prod(shape) - 1}}, max_depth=1
    )


@dataclass(frozen=True)
class SwapWithZeroExample:
    widths: tuple[int, ...]
    word: int
    shape: tuple[int, ...]
    adjoint: bool = False

    @property
    def signature(self) -> Signature:
        return swap_with_zero_signature(self.widths, self.word, self.shape)

    @property
    def selections(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(range(sum(self.widths[:i]), sum(self.widths[: i + 1])))
            for i in range(len(self.widths))
        )

    @property
    def targets(self) -> tuple[tuple[int, ...], ...]:
        start = sum(self.widths)
        return tuple(
            tuple(range(offset, offset + self.word))
            for offset in range(start, self.qubit_count, self.word)
        )

    @property
    def qubit_count(self) -> int:
        return sum(self.widths) + prod(self.shape) * self.word

    def append_to(self, circuit: Circuit) -> Circuit:
        gates = swap_with_zero_ir(
            self.selections, self.shape, self.targets, adjoint=self.adjoint
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))


@dataclass(frozen=True)
class MultiplexedSwapExample:
    width: int = 3
    rows: int = 5
    word: int = 2
    adjoint: bool = False

    @property
    def signature(self) -> Signature:
        return Signature(
            (
                Register("selection", opaque_register_type(self.width)),
                Register(
                    "targets", opaque_register_type(self.word), shape=(self.rows,)
                ),
                Register("output", opaque_register_type(self.word)),
            )
        )

    @property
    def data_width(self) -> int:
        return self.width + (self.rows + 1) * self.word

    @property
    def qubit_count(self) -> int:
        return self.data_width + self.width - 1

    def resource_model(self) -> ComponentCost:
        return source_multiplexed_cswap_cost((self.rows,), (self.width,), 0, self.word)

    def call_graph(self) -> SymbolicCallGraph[str]:
        root = "MultiplexedCSwap"
        counts = self.resource_model().gates
        return symbolic_call_graph(
            root,
            {
                root: {
                    f"CSwap({self.word})": self.rows,
                    "And(1,0)": counts.and_bloq,
                    "CNOT": counts.and_bloq,
                    "And†(1,1)": counts.measurement,
                    "X": 2,
                }
            },
            max_depth=1,
        )

    def append_to(self, circuit: Circuit) -> Circuit:
        targets = tuple(
            tuple(range(offset, offset + self.word))
            for offset in range(self.width, self.data_width - self.word, self.word)
        )
        gates = multiplexed_cswap_ir(
            (),
            (tuple(range(self.width)),),
            (self.rows,),
            targets,
            tuple(range(self.data_width - self.word, self.data_width)),
            tuple(range(self.data_width, self.qubit_count)),
            tuple(range(self.width - 1)),
            adjoint=self.adjoint,
        )
        return circuit.compose(CircuitIR(circuit.n_qubits, gates))

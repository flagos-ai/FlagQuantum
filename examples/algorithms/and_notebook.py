# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 mcmt/and_bloq.ipynb.

"""And and MultiAnd source-level declarations and native notebook examples."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from examples.algorithms.conjunction_tensors import (
    conjunction_program,
    conjunction_providers,
)
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.control.multi_control import multi_and_ir
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.core.value_math import SymbolicInt
from flagquantum.simulation.tensor_network.component_dense import (
    contract_component_tensors,
)


@dataclass(frozen=True)
class AndLadder:
    """Append the existing native ladder using controls before prefix outputs."""

    values: tuple[int, ...]
    erase: bool = False

    def append_to(self, circuit: Circuit) -> Circuit:
        n = len(self.values)
        body = multi_and_ir(
            tuple(range(n)),
            self.values,
            tuple(range(n, 2 * n - 1)),
            classical_bits=tuple(range(n - 1)) if self.erase else (),
            uncompute=self.erase,
        )
        return circuit.compose(
            CircuitIR(circuit.n_qubits, body, metadata={"batch_size": circuit.bsz})
        )


def and_signature(width: SymbolicInt, *, multi: bool, erase: bool = False) -> Signature:
    """Preserve the source array shapes and reversal of the output sides."""
    bit = quantum_bit_type()
    output_side = RegisterSide.LEFT if erase else RegisterSide.RIGHT
    registers = [Register("ctrl", bit, (width,))]
    if multi:
        registers.append(Register("junk", bit, (width - 2,), side=output_side))
    registers.append(Register("target", bit, side=output_side))
    return Signature(registers)


def and_source_graph(controls: tuple[int, ...] | SymbolicInt) -> SymbolicCallGraph[str]:
    """Keep And atomic and MultiAnd's generalized declaration independent of gates."""
    width = len(controls) if isinstance(controls, tuple) else controls
    if isinstance(controls, tuple) and width == 2:
        return symbolic_call_graph("And", {})
    calls: dict[str, SymbolicInt] = {"And": width - 1}
    if isinstance(controls, tuple) and controls.count(0):
        calls["XGate"] = 2 * controls.count(0)
    return symbolic_call_graph("MultiAnd", {"MultiAnd": calls})


def projected_conjunction(
    values: tuple[int, ...], inputs: tuple[int, ...]
) -> torch.Tensor:
    """Contract through equal basis state/effect selections on the control axes."""
    program = conjunction_program(values)
    like = torch.empty((), dtype=torch.complex128)
    providers = conjunction_providers(values, like)
    order = tuple(i for i, _, _ in iter_component_calls(program, "multi_and"))
    matrix = contract_component_tensors(
        program, order, providers.supply, like, "multi_and"
    )
    width = len(values)
    basis = int("".join(map(str, inputs)), 2)
    return matrix.reshape(1 << width, 1 << (width - 1), 1 << width)[basis, :, basis]

# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 MultiplexedCSwap3D.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/pbc/first_quantization/select_and_prepare.py

"""Components for first-quantized plane-wave chemistry."""

from collections.abc import Sequence
from dataclasses import dataclass

from ..circuit import Circuit
from ..compiler._component_trace.resource_model import ComponentCost, CostCall
from ..compiler.registers.multiplexed_swap import (
    multiplexed_cswap_ir,
    source_multiplexed_cswap_cost,
)
from ..core.ir import CircuitIR, Instruction
from ..errors import ValidationError


@dataclass(frozen=True)
class MultiplexedCSwap3D:
    """Exchange an addressed particle's three momentum coordinates with junk.

    Coordinates are ordered x, y, z. Each coordinate and the selection word
    are big-endian. Junk can carry arbitrary quantum data. The explicit
    traversal workspace starts and ends in zero.
    """

    selection: tuple[int, ...]
    targets: tuple[tuple[tuple[int, ...], ...], ...]
    junk: tuple[tuple[int, ...], ...]
    workspace: tuple[int, ...] = ()
    classical_bits: tuple[int, ...] = ()
    adjoint: bool = False

    @classmethod
    def make_on(
        cls,
        *,
        selection: Sequence[int],
        targets: Sequence[Sequence[Sequence[int]]],
        junk: Sequence[Sequence[int]],
        workspace: Sequence[int] = (),
        classical_bits: Sequence[int] = (),
        adjoint: bool = False,
    ) -> "MultiplexedCSwap3D":
        """Bind particle, junk and work registers to qubit indices."""
        return cls(
            tuple(selection),
            tuple(tuple(tuple(word) for word in particle) for particle in targets),
            tuple(tuple(word) for word in junk),
            tuple(workspace),
            tuple(classical_bits),
            adjoint,
        )

    def emit(self) -> tuple[Instruction, ...]:
        """Bind the three coordinates as one contiguous logical word per row."""
        if len(self.targets) < 2:
            raise ValidationError("Momentum exchange requires at least two particles.")
        if len(self.selection) != (len(self.targets) - 1).bit_length():
            raise ValidationError(
                "Momentum exchange selection width disagrees with the particle count."
            )
        if len(self.junk) != 3 or not self.junk[0]:
            raise ValidationError(
                "Momentum exchange requires three nonempty coordinates."
            )
        width = len(self.junk[0])
        if any(
            len(particle) != 3 or any(len(word) != width for word in particle)
            for particle in (*self.targets, self.junk)
        ):
            raise ValidationError("Momentum coordinates must have equal widths.")
        return multiplexed_cswap_ir(
            (),
            (self.selection,),
            (len(self.targets),),
            tuple(tuple(wire for word in row for wire in word) for row in self.targets),
            tuple(wire for word in self.junk for wire in word),
            self.workspace,
            self.classical_bits,
            adjoint=self.adjoint,
        )

    def append_to(self, circuit: Circuit) -> Circuit:
        """Append the bound exchange and its measurement feedback."""
        ir = CircuitIR(
            circuit.n_qubits, self.emit(), metadata={"batch_size": circuit.bsz}
        )
        return circuit.compose(ir.validate())


def source_multiplexed_cswap_3d_cost(eta: int, num_bits_p: int) -> ComponentCost:
    """Estimate the source reshaping wrapper and its shared exchange leaf."""
    if type(eta) is not int or eta < 2 or type(num_bits_p) is not int or num_bits_p < 1:
        raise ValueError(
            "Momentum exchange requires at least two particles and a positive coordinate width"
        )
    selection_width = (eta - 1).bit_length()
    child = source_multiplexed_cswap_cost((eta,), (selection_width,), 0, 3 * num_bits_p)
    ports = selection_width + 3 * num_bits_p * (eta + 1)
    return ComponentCost("MultiplexedCSwap3D", ports, calls=(CostCall(child),))

# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 MeasureQFT and its QFTTextBook declaration.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/cryptography/_factoring_shims.py

"""Textbook forward Fourier measurement with explicitly bound result slots."""

from dataclasses import dataclass

from ...circuit import Circuit
from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler._component_trace.symbolic_math import SymbolicInt, smax
from ...compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program.register_types import quantum_bit_type
from ...core.ir.program.registers import Register, RegisterSide
from ...core.ir.program.signatures import Signature
from ...dynamic import DynamicCircuit
from .qft import qft


def source_measure_qft_signature(width: SymbolicInt) -> Signature:
    """Describe the consumed input array in the locked shim."""
    return Signature(
        (Register("x", quantum_bit_type(), shape=(width,), side=RegisterSide.LEFT),)
    )


def source_measure_qft_symbolic_formula(width: SymbolicInt) -> SymbolicResourceFormula:
    """Use the source's grouped phase-gradient rule for symbolic widths."""
    if isinstance(width, int):
        raise TypeError("Use the concrete MeasureQFT model for an integer width")
    swaps = width // 2
    phases = (width - 1) * swaps
    return SymbolicResourceFormula(
        and_bloq=phases,
        clifford=width + phases + swaps,
        measurement=width + phases,
        rotation=phases,
        logical_qubits=smax(3, width),
    )


def source_measure_qft_cost(width: int) -> ComponentCost:
    """Preserve the concrete source call graph and model its textbook QFT child."""
    if type(width) is not int or width < 0:
        raise ValueError("Fourier measurement width must be a nonnegative integer")
    hadamard = CostCall(ComponentCost("Hadamard", 1, GateCost(clifford=1)), width)
    swap = CostCall(ComponentCost("TwoBitSwap", 2, GateCost(clifford=1)), width // 2)
    gradients = tuple(
        CostCall(
            ComponentCost(
                f"PhaseGradientUnitary({bits})",
                bits + 2,
                GateCost(
                    t=int(bits > 1),
                    and_bloq=bits,
                    clifford=bits + 1,
                    rotation=max(bits - 2, 0),
                    measurement=bits,
                ),
            ),
            bystander_qubits=width - bits - 1,
        )
        for bits in range(1, width)
    )
    transform = ComponentCost("QFTTextBook", width, calls=(hadamard, *gradients, swap))
    measure = ComponentCost("MeasureZ", 1, GateCost(measurement=1))
    return ComponentCost(
        "MeasureQFT", width, calls=(CostCall(transform), CostCall(measure, width))
    )


@dataclass(frozen=True)
class _BoundMeasureQFT:
    x: tuple[int, ...]
    instructions: tuple[Instruction, ...]

    def resource_model(self) -> ComponentCost:
        return source_measure_qft_cost(len(self.x))

    def append_to(self, circuit: Circuit) -> Circuit:
        if not self.x:
            return circuit
        if not isinstance(circuit, DynamicCircuit):
            raise ValueError("Fourier measurement requires a DynamicCircuit")
        local = CircuitIR(
            len(self.x), self.instructions, metadata={"batch_size": circuit.bsz}
        ).validate()
        return circuit.compose(local, qubits=self.x)


@dataclass(frozen=True)
class MeasureQFT:
    """Bind a forward QFT followed by Z readout in the supplied wire order."""

    width: int

    def __post_init__(self) -> None:
        if type(self.width) is not int or self.width < 0:
            raise ValueError("Fourier measurement width must be a nonnegative integer")

    @property
    def source_signature(self) -> Signature:
        return source_measure_qft_signature(self.width)

    def make_on(
        self, *, x: tuple[int, ...], result_bits: tuple[int, ...]
    ) -> _BoundMeasureQFT:
        """Bind an ordered big-endian value and its equally wide result slots."""
        if len(x) != self.width or len(result_bits) != self.width:
            raise ValueError(
                "Fourier measurement requires matching wire and result widths"
            )
        for values in (x, result_bits):
            if len(set(values)) != len(values) or any(
                type(value) is not int or value < 0 for value in values
            ):
                raise ValueError(
                    "Fourier measurement requires distinct nonnegative wires and result slots"
                )
        if not self.width:
            return _BoundMeasureQFT(x, ())
        readout = tuple(
            Instruction(
                "measure",
                (index,),
                metadata={"is_dynamic": True, "classical_bit": slot},
            )
            for index, slot in enumerate(result_bits)
        )
        return _BoundMeasureQFT(x, (*qft(self.width).to_ir().instructions, *readout))

# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0 ModExp.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/cryptography/rsa/rsa_mod_exp.py

"""Right-to-left binary modular exponentiation on explicit register bindings."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, gcd, log2

from numpy.random import RandomState

from ...circuit import Circuit
from ...compiler._component_trace.call_graph import (
    SymbolicCallGraph,
    symbolic_call_graph,
)
from ...compiler.registers.layout.classical_storage import bind_classical_scratch
from ...compiler.registers.modular.modular_exponentiation_component import (
    modular_exponentiation_program,
)
from ...compiler.registers.modular.modular_multiplication import (
    modular_multiplication_requirements,
)
from ...compiler.registers.register_binding import bind_register_program
from ...core.ir import CircuitIR, Instruction
from ...core.ir.program.register_types import unsigned_register_type
from ...core.ir.program.registers import Register, RegisterSide
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt
from ...core.value_parameters import is_symbolic_value
from ...dynamic import DynamicCircuit


def shor_factoring_parameters(
    modulus: SymbolicInt, base: SymbolicInt | None, random_state: RandomState | None
) -> tuple[SymbolicInt, SymbolicInt]:
    """Return the source word width and optional seeded base for either factory."""
    if is_symbolic_value(modulus):
        from sympy import Symbol, ceiling, log

        return ceiling(log(modulus, 2)), Symbol("g") if base is None else base
    value = int(modulus)
    width = ceil(log2(value))
    if base is not None:
        return width, base
    if value <= 2:
        raise ValueError("Random factoring base selection requires modulus > 2")
    rng = RandomState() if random_state is None else random_state
    candidate = int(rng.randint(2, value))
    while gcd(candidate, value) != 1:
        candidate = int(rng.randint(2, value))
    return width, candidate


@dataclass(frozen=True)
class _BoundModularExponentiation:
    wires: tuple[int, ...]
    instructions: tuple[Instruction, ...]

    def append_to(self, circuit: Circuit) -> Circuit:
        if not isinstance(circuit, DynamicCircuit):
            raise ValueError("Modular exponentiation requires a DynamicCircuit")
        local = CircuitIR(
            len(self.wires), self.instructions, metadata={"batch_size": circuit.bsz}
        ).validate()
        return circuit.compose(local, qubits=self.wires)


@dataclass(frozen=True)
class ModularExponentiation:
    """Compute base**exponent modulo modulus into a fresh zero word.

    Exponent and output are big-endian. The exponent is unchanged. Work
    starts and ends zero. Each multiplier uses the shared measurement slots
    and the source's control, including when its squared constant is one.
    """

    base: SymbolicInt
    modulus: SymbolicInt
    exponent_width: SymbolicInt
    output_width: SymbolicInt

    def __post_init__(self) -> None:
        if (
            not is_symbolic_value(self.base, self.modulus)
            and gcd(int(self.base), int(self.modulus)) != 1
        ):
            raise ValueError("Modular exponentiation base and modulus must be coprime")

    @classmethod
    def for_shor(
        cls,
        modulus: SymbolicInt,
        base: SymbolicInt | None = None,
        *,
        random_state: RandomState | None = None,
    ) -> ModularExponentiation:
        """Use the source factoring widths and its optional seeded base selection."""
        width, base = shor_factoring_parameters(modulus, base, random_state)
        return cls(base, modulus, 2 * width, width)

    @property
    def source_signature(self) -> Signature:
        return Signature(
            (
                Register("exponent", unsigned_register_type(self.exponent_width)),
                Register(
                    "x",
                    unsigned_register_type(self.output_width),
                    side=RegisterSide.RIGHT,
                ),
            )
        )

    @property
    def source_label(self) -> str:
        return f"{self.base}^e % {self.modulus}"

    def source_call_graph(self) -> SymbolicCallGraph[str]:
        return symbolic_call_graph(
            "ModExp", {"ModExp": {"CModMulK": self.exponent_width, "|1>": 1}}
        )

    def classical_values(
        self, exponent: SymbolicInt
    ) -> tuple[SymbolicInt, SymbolicInt]:
        if (
            all(type(v) is int for v in (self.base, self.modulus, exponent))
            and int(exponent) >= 0
        ):
            return exponent, pow(int(self.base), int(exponent), int(self.modulus))
        return exponent, (self.base**exponent) % self.modulus

    def workspace_requirements(self) -> tuple[int, int]:
        if type(self.exponent_width) is not int or self.exponent_width < 0:
            raise ValueError(
                "Modular exponentiation requires a nonnegative concrete exponent width"
            )
        if type(self.output_width) is not int or self.output_width < 1:
            raise ValueError(
                "Modular exponentiation requires a positive concrete output width"
            )
        if (
            type(self.modulus) is not int
            or not 1 < self.modulus < 1 << self.output_width
        ):
            raise ValueError(
                "Modular exponentiation modulus must fit the output and exceed one"
            )
        if type(self.base) is not int:
            raise ValueError("Modular exponentiation requires a concrete base")
        if not self.exponent_width:
            return 0, 0
        return modular_multiplication_requirements(self.output_width)

    def make_on(
        self,
        *,
        exponent: tuple[int, ...],
        target: tuple[int, ...],
        workspace: tuple[int, ...],
        classical_bits: tuple[int, ...],
    ) -> _BoundModularExponentiation:
        q, c = self.workspace_requirements()
        if len(exponent) != self.exponent_width or len(target) != self.output_width:
            raise ValueError("Modular exponentiation requires matching operand widths")
        if len(workspace) != q or len(classical_bits) != c:
            raise ValueError("Modular exponentiation requires exact workspace sizes")
        wires = (*exponent, *target, *workspace)
        for values in (wires, classical_bits):
            if len(set(values)) != len(values) or any(
                type(v) is not int or v < 0 for v in values
            ):
                raise ValueError(
                    "Modular exponentiation requires distinct nonnegative wires and slots"
                )
        e, n = len(exponent), len(target)
        record = modular_exponentiation_program(int(self.base), int(self.modulus), e, n)
        bound = bind_register_program(
            record,
            (tuple(range(e)),),
            entry="ModExp",
            clean=tuple(range(e, len(wires))),
            allow_workspace_outputs=True,
        )
        if bound.outputs != (tuple(range(e)), tuple(range(e, e + n))):
            raise ValueError("Modular exponentiation changed its operand binding")
        bound = bind_classical_scratch(bound, tuple(classical_bits))
        return _BoundModularExponentiation(wires, bound.circuit.instructions)

"""Generate bounded native Python phase oracles from signed 3-SAT clauses."""

import runpy
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import flagquantum as fq
from flagquantum import synth

MAX_VARIABLES = 68


@dataclass(frozen=True)
class Formula:
    """One clause per variable, with two column permutations."""

    negative: tuple[int, ...]
    positive: tuple[int, ...]

    def __post_init__(self) -> None:
        width = self.width
        if not 3 <= width <= MAX_VARIABLES:
            raise ValueError("The oracle admits 3 to 68 variables")
        if any(type(bit) is not int for bit in self.negative + self.positive):
            raise ValueError("Clause indices must be integers")
        if sorted(self.negative) != list(range(width)) or sorted(self.positive) != list(
            range(width)
        ):
            raise ValueError("Both literal columns must be permutations")
        if any(
            len({index, left, right}) != 3
            for index, (left, right) in enumerate(
                zip(self.negative, self.positive, strict=True)
            )
        ):
            raise ValueError("Every clause must contain three distinct variables")

    @property
    def width(self) -> int:
        return len(self.negative)

    def satisfies(self, word: int) -> bool:
        """Evaluate the clause truth table on an independent classical word."""
        if not 0 <= word < (1 << self.width):
            raise ValueError("Word outside the formula register")
        bits = tuple(bool(word & (1 << wire)) for wire in range(self.width))
        return all(
            bits[index] or not bits[left] or bits[right]
            for index, (left, right) in enumerate(
                zip(self.negative, self.positive, strict=True)
            )
        )


def native_source(formula: Formula, *, word: int = 0, uniform: bool = False) -> str:
    """Emit source-retaining Python using only validated integer wire indices."""
    if type(word) is not int or not 0 <= word < (1 << formula.width):
        raise ValueError("Word outside the formula register")
    if uniform and word:
        raise ValueError("Uniform preparation starts at word zero")
    preparation = (
        ["    hadamard_transform(variables)"]
        if uniform
        else [
            f"    X(variables[{wire}])"
            for wire in range(formula.width)
            if word & (1 << wire)
        ]
    )
    literals = list(zip(formula.negative, formula.positive, strict=True))
    clauses = [
        f"(variables[{index}] or not variables[{left}] or variables[{right}])"
        for index, (left, right) in enumerate(literals)
    ]
    body = [
        "    control(",
        "        " + "\n        and ".join(clauses) + ",",
        "        lambda: phase(1, pi),",
        "    )",
    ]
    return "\n".join(
        [
            "from math import pi",
            "from flagquantum.synth import Output, QArray, QBit, X, allocate, control, hadamard_transform, phase, qfunc",
            "",
            "@qfunc",
            "def main(variables: Output[QArray[QBit]]) -> None:",
            f"    allocate({formula.width}, variables)",
            *preparation,
            *body,
            "",
        ]
    )


def build_program(
    formula: Formula, *, word: int = 0, uniform: bool = False, max_qubits: int = 220
) -> synth.Program:
    """Synthesize a generic signed CNF; the basis benchmark admits only exact storage."""
    if uniform and formula.width > 5:
        raise ValueError("Dense coherent validation is limited to five data wires")
    with TemporaryDirectory(prefix="fq-native-sat-") as directory:
        path = Path(directory) / "phase_oracle.py"
        path.write_text(native_source(formula, word=word, uniform=uniform))
        namespace = runpy.run_path(str(path))
        return synth.synthesize(
            namespace["main"],
            max_qubits=max_qubits,
            max_operations=10_000,
            memory_limit_bytes=64 * 1024**2 if formula.width <= 8 else 32 * max_qubits,
        )


def clause_circuit(formula: Formula, *, uniform: bool = False) -> fq.Circuit:
    """Build a reversible clause network with one reused three-literal work bit."""
    width = formula.width
    circuit = fq.Circuit(3 * width, dtype=torch.complex128)
    if uniform:
        for wire in range(width):
            circuit.h(wire)
    temporary = 3 * width - 1
    gates = [
        (width - 1 - index, width - 1 - left, width - 1 - right, width + index)
        for index, (left, right) in enumerate(
            zip(formula.negative, formula.positive, strict=True)
        )
    ]

    def toggle_clause(wires: tuple[int, int, int, int]) -> None:
        positive, negative, other, flag = wires
        circuit.x(flag).x(positive).x(other)
        circuit.ccx(positive, negative, temporary)
        circuit.ccx(temporary, other, flag)
        circuit.ccx(positive, negative, temporary)
        circuit.x(other).x(positive)

    for wires in gates:
        toggle_clause(wires)
    for index in range(width - 1):
        previous = width if index == 0 else 2 * width + index - 1
        circuit.ccx(previous, width + index + 1, 2 * width + index)
    circuit.z(3 * width - 2)
    for index in reversed(range(width - 1)):
        previous = width if index == 0 else 2 * width + index - 1
        circuit.ccx(previous, width + index + 1, 2 * width + index)
    for wires in reversed(gates):
        toggle_clause(wires)
    return circuit

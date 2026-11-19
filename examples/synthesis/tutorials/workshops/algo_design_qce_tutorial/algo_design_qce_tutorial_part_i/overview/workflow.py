"""Full source-sized QCE part I teaching stages and decoded actual observations."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

import torch

import flagquantum as fq
from examples.synthesis.tutorials.basics.quantum_programming.qmod_tutorial_part1.overview.workflow import (
    bell,
    ghz,
    signed_ghz,
)
from examples.synthesis.tutorials.basics.quantum_programming.qmod_tutorial_part1.overview.workflow import (
    reference_circuit as teaching_reference,
)
from examples.synthesis.tutorials.workshops.algo_design_qce_tutorial.algo_design_qce_tutorial_part_i.comparisons.circuit import (
    reference_circuit as square_reference,
)
from examples.synthesis.tutorials.workshops.algo_design_qce_tutorial.algo_design_qce_tutorial_part_i.main import (
    main as fourier_square,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate

from .ansatz import fractional_sum, grover, integer_sum, product_phase, signed_condition

STAGES = (
    "bell",
    "ghz",
    "signed_ghz",
    "integer_sum",
    "fractional_sum",
    "signed_condition",
    "grover",
    "product_phase",
    "fourier_square",
)


@dataclass(frozen=True)
class StateCheck:
    name: str
    program: synth.Program
    maximum_state_error: float
    dirty_workspace_probability: float


@dataclass(frozen=True)
class Observation:
    values: tuple[Fraction, ...]
    count: int
    frequency: float


@dataclass(frozen=True)
class NumericAttributes:
    size: int
    signed: bool
    fraction_digits: int


def build_program(name: str) -> synth.Program:
    definitions = dict(
        bell=bell,
        ghz=ghz,
        signed_ghz=signed_ghz,
        integer_sum=integer_sum,
        fractional_sum=fractional_sum,
        signed_condition=signed_condition,
        grover=grover,
        product_phase=product_phase,
        fourier_square=fourier_square,
    )
    return synth.synthesize(definitions[name], memory_limit_bytes=64 * 1024**2)


def numeric_attributes(program: synth.Program, name: str) -> NumericAttributes:
    formats = program.report.get("numeric_formats", ())
    assert isinstance(formats, tuple)
    signed, fraction = False, 0
    for item in formats:
        assert isinstance(item, tuple) and len(item) == 3
        if item[0] == name:
            assert isinstance(item[1], bool) and type(item[2]) is int
            signed, fraction = item[1], item[2]
    return NumericAttributes(len(program.registers[name]), signed, fraction)


def distribution(program: synth.Program) -> Mapping[tuple[Fraction, ...], float]:
    result = fq.run(program)
    names = tuple(program.registers)
    if program.report.get("numeric_formats"):
        return program.numeric_probabilities(result, registers=names)
    return {
        tuple(Fraction(v) for v in values): probability
        for values, probability in program.probabilities(
            result, registers=names
        ).items()
    }


def sample(
    program: synth.Program, *, shots: int = 1000, seed: int = 1970
) -> tuple[Observation, ...]:
    if type(shots) is not int or not 1 <= shots <= 100000:
        raise ValueError("shots must lie in 1..100000")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    values = distribution(program)
    keys = tuple(values)
    counts = torch.bincount(
        torch.multinomial(
            torch.tensor(tuple(values.values()), dtype=torch.float64),
            shots,
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        ),
        minlength=len(keys),
    )
    return tuple(
        Observation(key, int(counts[i]), int(counts[i]) / shots)
        for i, key in enumerate(keys)
        if int(counts[i])
    )


def reference_circuit(name: str) -> fq.Circuit:
    if name in {"bell", "ghz", "signed_ghz"}:
        return teaching_reference(name)
    if name == "fourier_square":
        return square_reference()
    if name in {"integer_sum", "fractional_sum"}:
        width = 4 if name == "integer_sum" else 5
        circuit = fq.Circuit(4 + width, dtype=torch.complex128)
        for wire in range(4):
            circuit.h(wire)
        controlled_x = torch.eye(32, dtype=torch.complex128)
        controlled_x[30:, 30:] = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
        for a in range(4):
            for b in range(4):
                word = 4 * a + b
                zeros = tuple(i for i in range(4) if not (word >> (3 - i)) & 1)
                for wire in zeros:
                    circuit.x(wire)
                encoded = 3 * a + b if name == "integer_sum" else 6 * a + b
                for bit in range(width):
                    if (encoded >> bit) & 1:
                        circuit.any(
                            0, 1, 2, 3, 4 + width - 1 - bit, matrix=controlled_x
                        )
                for wire in zeros:
                    circuit.x(wire)
        return circuit
    if name == "signed_condition":
        circuit = fq.Circuit(4, dtype=torch.complex128)
        for wire in range(3):
            circuit.h(wire)
        permutation = torch.zeros((16, 16), dtype=torch.complex128)
        for word in range(8):
            x = Fraction(word - 8 if word >= 4 else word, 4)
            for flag in range(2):
                permutation[
                    (word << 1) | (flag ^ int(x < Fraction(1, 2))), (word << 1) | flag
                ] = 1
        return circuit.any(*range(4), matrix=permutation)
    if name == "grover":
        circuit = fq.Circuit(6, dtype=torch.complex128)
        for wire in range(6):
            circuit.h(wire)
        diagonal = torch.tensor(
            [
                -1 if 3 * (w >> 4) + ((w >> 2) & 3) + 2 * (w & 3) == 9 else 1
                for w in range(64)
            ],
            dtype=torch.complex128,
        )
        reflection = (
            torch.eye(64, dtype=torch.complex128)
            - torch.ones((64, 64), dtype=torch.complex128) / 32
        )
        for _ in range(2):
            circuit.any(*range(6), matrix=torch.diag(diagonal))
            circuit.any(*range(6), matrix=reflection)
        return circuit
    if name != "product_phase":
        raise ValueError("unknown teaching stage")
    circuit = fq.Circuit(4, dtype=torch.complex128)
    for wire in range(4):
        circuit.h(wire)
    diagonal = torch.tensor(
        [math.pi / 8 * (w >> 2) * (w & 3) for w in range(16)], dtype=torch.float64
    )
    return circuit.any(*range(4), matrix=torch.diag(torch.exp(1j * diagonal)))


def declared_amplitudes(program: synth.Program) -> torch.Tensor:
    state = fq.run(program).statevector()
    assert isinstance(state, torch.Tensor)
    width = state.numel().bit_length() - 1
    logical = tuple(wire for wires in program.registers.values() for wire in wires)
    hidden = tuple(wire for wire in range(width) if wire not in logical)
    return (
        state.reshape((2,) * width)
        .permute((*logical, *hidden))
        .reshape(1 << len(logical), -1)
    )


def phase_table(program: synth.Program) -> tuple[tuple[int, int, complex], ...]:
    if tuple(program.registers) != ("x", "y") or any(
        len(wires) != 2 for wires in program.registers.values()
    ):
        raise ValueError("the product-phase table needs two two-bit named registers")
    amplitudes = declared_amplitudes(program)
    if float(amplitudes[:, 1:].abs().square().sum()) > 1e-22:
        raise AssertionError("product-phase workspace must be zero")
    return tuple(
        (x, y, complex(amplitudes[4 * x + y, 0])) for x in range(4) for y in range(4)
    )


def verify_program(name: str, program: synth.Program) -> StateCheck:
    expected = fq.run(reference_circuit(name)).statevector()
    assert isinstance(expected, torch.Tensor)
    amplitudes = declared_amplitudes(program)
    error = float((amplitudes[:, 0] - expected.flatten()).abs().max())
    dirty = float(amplitudes[:, 1:].abs().square().sum())
    if error > 2e-11 or dirty > 1e-22:
        raise AssertionError(
            "native absolute state or workspace differs from independent Circuit"
        )
    return StateCheck(name, program, error, dirty)


def verify(name: str) -> StateCheck:
    return verify_program(name, build_program(name))


def resources(check: StateCheck) -> ResourceEstimate:
    resource = check.program.report["resources"]
    assert isinstance(resource, ResourceEstimate)
    return resource

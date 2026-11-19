"""Complete native teaching stages with independent logical-state references."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Annotated

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import (
    CX,
    RX,
    H,
    Output,
    QArray,
    QBit,
    QNum,
    X,
    allocate,
    control,
    hadamard_transform,
    inplace_prepare_state,
    prepare_state,
    qfunc,
)

PROBABILITIES = [0.3, 0.2, 0.05, 0.13, 0.15, 0.05, 0.1, 0.02]


@qfunc
def bell_helper(pair: QArray[QBit]) -> None:
    H(pair[0])
    CX(pair[0], pair[1])


@qfunc
def bell(res: Output[QArray[QBit]]) -> None:
    allocate(2, res)
    bell_helper(res)


@qfunc
def ghz_helper(qubits: QArray[QBit]) -> None:
    H(qubits[0])
    CX(qubits[0], qubits[1])
    CX(qubits[1], qubits[2])


@qfunc
def ghz(res: Output[QArray[QBit]]) -> None:
    allocate(3, res)
    ghz_helper(res)


@qfunc
def signed_ghz(x: Output[Annotated[QNum, (3, True, 0)]]) -> None:
    allocate(x)
    ghz_helper(x)


@qfunc
def distributions(
    probabilities: list[float], a: Output[QArray[QBit]], b: Output[QArray[QBit]]
) -> None:
    prepare_state(probabilities, 0.01, a)
    allocate(3, b)
    inplace_prepare_state(probabilities, 0.01, b)


@qfunc
def invalid_prep(q: QBit) -> None:
    H(q)


@qfunc
def invalid_initialization(q: Output[QBit]) -> None:
    invalid_prep(q)


@qfunc
def owned_prep(q: Output[QBit]) -> None:
    allocate(q)
    H(q)


@qfunc
def corrected_initialization(q: Output[QBit]) -> None:
    owned_prep(q)


@qfunc
def rotate(turns: float, index: int, qarr: QArray[QBit]) -> None:
    RX(2 * math.pi * turns, qarr[index])


@qfunc
def rotations(qarr: Output[QArray[QBit]]) -> None:
    allocate(2, qarr)
    rotate(0.3, 0, qarr)
    rotate(0.2, 1, qarr)


@qfunc
def parameterized(alpha: float, beta: float, qarr: Output[QArray[QBit]]) -> None:
    allocate(2, qarr)
    rotate(alpha, 0, qarr)
    rotate(beta, 1, qarr)


@qfunc
def rotate_layer(turns: list[float], qarr: QArray[QBit]) -> None:
    for index in range(len(turns)):
        rotate(turns[index], index, qarr)


@qfunc
def layer(qarr: Output[QArray[QBit]]) -> None:
    allocate(3, qarr)
    rotate_layer([0.1, 0.3, 0.5], qarr)


@qfunc
def arithmetic(
    x: Output[QNum], y: Output[QNum], total: Output[QNum], prod: Output[QNum]
) -> None:
    allocate(2, x)
    allocate(2, y)
    hadamard_transform(x)
    hadamard_transform(y)
    total |= x + y
    prod |= x * y


@qfunc
def accumulation(y: Output[QNum], acc: Output[QNum]) -> None:
    allocate(2, y)
    hadamard_transform(y)
    acc |= 4
    acc += y


@qfunc
def conditional(x: Output[QNum], target: Output[QBit]) -> None:
    allocate(3, x)
    hadamard_transform(x)
    allocate(target)
    control(x == 5, lambda: X(target))


@dataclass(frozen=True)
class Observation:
    values: tuple[int, ...]
    count: int
    probability: float


@dataclass(frozen=True)
class Check:
    name: str
    program: synth.Program
    maximum_error: float
    dirty_workspace_probability: float


def build_program(name: str, *, alpha: float = 0.3, beta: float = 0.2) -> synth.Program:
    """Build a complete solution; classical parameters bind at synthesis time."""
    definitions = {
        "bell": bell,
        "ghz": ghz,
        "signed_ghz": signed_ghz,
        "distributions": distributions,
        "corrected_initialization": corrected_initialization,
        "rotations": rotations,
        "parameterized": parameterized,
        "layer": layer,
        "arithmetic": arithmetic,
        "accumulation": accumulation,
        "conditional": conditional,
    }
    return synth.synthesize(
        definitions[name],
        classical_inputs=(
            {"alpha": alpha, "beta": beta}
            if name == "parameterized"
            else {"probabilities": PROBABILITIES} if name == "distributions" else None
        ),
        memory_limit_bytes=64 * 1024**2,
    )


def reference_circuit(
    name: str, *, alpha: float = 0.3, beta: float = 0.2
) -> fq.Circuit:
    """Build independent gates or a normalized mathematical reference vector."""
    if name in {"bell", "ghz", "signed_ghz", "corrected_initialization"}:
        width = 2 if name == "bell" else 1 if name == "corrected_initialization" else 3
        circuit = fq.Circuit(width, dtype=torch.complex128)
        circuit.h(0)
        for bit in range(1, width):
            circuit.cx(bit - 1, bit)
        return circuit
    if name in {"rotations", "parameterized", "layer"}:
        turns = (
            (0.1, 0.3, 0.5)
            if name == "layer"
            else (alpha, beta) if name == "parameterized" else (0.3, 0.2)
        )
        circuit = fq.Circuit(len(turns), dtype=torch.complex128)
        for bit, value in enumerate(turns):
            circuit.rx(len(turns) - 1 - bit, 2 * math.pi * value)
        return circuit
    if name == "arithmetic":
        # Enumerate the independent classical truth table into four-control X
        # gates. All references stay local to five wires rather than constructing
        # an eleven-wire dense unitary.
        circuit = fq.Circuit(11, dtype=torch.complex128)
        for wire in range(4):
            circuit.h(wire)
        controlled_x = torch.eye(32, dtype=torch.complex128)
        controlled_x[30:, 30:] = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
        for x in range(4):
            for y in range(4):
                word = (x << 2) | y
                zeros = tuple(wire for wire in range(4) if not (word >> (3 - wire)) & 1)
                for wire in zeros:
                    circuit.x(wire)
                for start, size, value in ((4, 3, x + y), (7, 4, x * y)):
                    for bit in range(size):
                        if (value >> bit) & 1:
                            circuit.any(
                                0, 1, 2, 3, start + size - 1 - bit, matrix=controlled_x
                            )
                for wire in zeros:
                    circuit.x(wire)
        return circuit
    widths = {"distributions": 6, "accumulation": 5, "conditional": 4}
    width = widths[name]
    state = torch.zeros(1 << width, dtype=torch.complex128)
    if name == "distributions":
        for a, pa in enumerate(PROBABILITIES):
            for b, pb in enumerate(PROBABILITIES):
                state[(a << 3) | b] = math.sqrt(pa * pb)
    elif name == "accumulation":
        for y in range(4):
            state[(y << 3) | (4 + y)] = 0.5
    else:
        for x in range(8):
            state[(x << 1) | int(x == 5)] = 1 / math.sqrt(8)
    circuit = fq.Circuit(width, dtype=torch.complex128)
    # A Householder reflection is a full unitary with first column equal to the
    # real normalized oracle vector. It shares no synth arithmetic/preparation.
    delta = -state.clone()
    delta[0] += 1
    norm = float(delta.norm())
    if norm:
        delta /= norm
        matrix = torch.eye(1 << width, dtype=torch.complex128) - 2 * torch.outer(
            delta, delta.conj()
        )
        circuit.any(*range(width), matrix=matrix)
    return circuit


def verify(name: str, *, alpha: float = 0.3, beta: float = 0.2) -> Check:
    """Check absolute amplitudes and every compiler workspace amplitude."""
    program = build_program(name, alpha=alpha, beta=beta)
    return verify_program(name, program, alpha=alpha, beta=beta)


def verify_program(
    name: str, program: synth.Program, *, alpha: float = 0.3, beta: float = 0.2
) -> Check:
    """Verify a supplied Notebook program against the independent oracle."""
    expected = (
        fq.run(reference_circuit(name, alpha=alpha, beta=beta)).statevector().flatten()
    )
    actual = fq.run(program).statevector().flatten()
    width = actual.numel().bit_length() - 1
    logical = tuple(bit for value in program.registers.values() for bit in value)
    hidden = tuple(bit for bit in range(width) if bit not in logical)
    ordered = (
        actual.reshape((2,) * width)
        .permute((*logical, *hidden))
        .reshape(expected.numel(), -1)
    )
    error = float((ordered[:, 0] - expected).abs().max())
    dirty = float(ordered[:, 1:].abs().square().sum())
    if error > 1e-12 or dirty > 1e-24:
        raise AssertionError(
            f"{name}: absolute state error {error}, dirty work {dirty}"
        )
    return Check(name, program, error, dirty)


def sample(
    program: synth.Program, *, shots: int = 1000, seed: int = 19
) -> tuple[Observation, ...]:
    """Draw actual seeded local observations from the native result distribution."""
    if type(shots) is not int or shots < 1 or type(seed) is not int or seed < 0:
        raise ValueError("shots must be positive and seed must be nonnegative integers")
    probabilities = dict(
        program.probabilities(fq.run(program), registers=tuple(program.registers))
    )
    keys = tuple(probabilities)
    weights = torch.tensor(tuple(probabilities.values()), dtype=torch.float64)
    counts = torch.bincount(
        torch.multinomial(
            weights,
            shots,
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        ),
        minlength=len(keys),
    )
    return tuple(
        Observation(values, int(counts[i]), int(counts[i]) / shots)
        for i, values in enumerate(keys)
    )


def signed_distribution(program: synth.Program) -> Mapping[tuple[Fraction, ...], float]:
    return program.numeric_probabilities(fq.run(program), registers=("x",))


def initialization_refusal() -> str:
    """The original deliberate ownership error must fail before execution."""
    try:
        synth.synthesize(invalid_initialization)
    except ValueError as error:
        if "construction.register.uninitialized" not in str(error):
            raise
        return str(error)
    raise AssertionError("uninitialized borrowed storage was accepted")

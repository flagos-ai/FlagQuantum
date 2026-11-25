"""Run the sampled and synthesis-only ECDLP workflows at the original scale."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.algorithms.number_theory_and_cryptography.elliptic_curves.elliptic_curve_discrete_log.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.algorithms.number_theory_and_cryptography.elliptic_curves.elliptic_curve_discrete_log.curve import (
    add,
    translation_steps,
)
from examples.synthesis.algorithms.number_theory_and_cryptography.elliptic_curves.elliptic_curve_discrete_log.main import (
    build_program,
    translation,
)
from flagquantum.algorithms.primitives.qft import append_qft
from flagquantum.compiler import ResourceEstimate
from flagquantum.core.ir import Instruction
from flagquantum.synth import (
    Output,
    Program,
    QArray,
    QBit,
    X,
    allocate,
    qfunc,
    synthesize,
)

from .arithmetic import Gate, circuit_from_gates
from .kaliski import PointWorkspace, point_add_gates

SEED = (4, 2)
GENERATOR = (0, 5)
TARGET = (0, 2)
ORDER = 5
SHOTS = 2048
X1 = (2, 1, 0)
X2 = (5, 4, 3)
POINT_X = (11, 10, 9)
POINT_Y = (8, 7, 6)


@dataclass(frozen=True)
class Frequency:
    left: int
    right: int
    point: tuple[int, int]
    count: int

    def logarithm(self) -> int | None:
        left = round(self.left * ORDER / 8)
        right = round(self.right * ORDER / 8)
        if right % ORDER == 0:
            return None
        return -left * pow(right, -1, ORDER) % ORDER


@qfunc
def mock_point_add(steps: list[int], point: Output[QArray[QBit]]) -> None:
    allocate(6, point)
    X(point[2])
    X(point[4])
    translation(steps, point)


def sampled_point_add() -> tuple[tuple[int, int], ResourceEstimate]:
    program = synthesize(
        mock_point_add,
        classical_inputs={"steps": translation_steps(GENERATOR)},
        max_qubits=9,
        max_workspace=3,
        memory_limit_bytes=2**20,
    )
    probabilities = program.probabilities(fq.run(program), registers=("point",))
    words = tuple(probabilities)
    sample = torch.multinomial(
        torch.tensor([probabilities[key] for key in words], dtype=torch.float64),
        1,
        generator=torch.Generator().manual_seed(5),
    )
    word = words[int(sample.item())][0]
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    return (word % 8, word // 8), resources


def sampled_period_finding(
    program: Program | None = None,
) -> tuple[tuple[Frequency, ...], ResourceEstimate, float]:
    if program is None:
        program = build_program()
    result = fq.run(program)
    physical_state = result.statevector().reshape(4096, -1)
    error = float(
        (physical_state[:, 0] - fq.run(reference_circuit()).statevector().flatten())
        .abs()
        .max()
    )
    if error > 3e-12 or float(physical_state[:, 1:].abs().max()) > 3e-12:
        raise AssertionError("mock point-translation state or workspace differs")
    probabilities = program.probabilities(result, registers=("x1", "x2", "point"))
    words = tuple(probabilities)
    indices = torch.multinomial(
        torch.tensor([probabilities[word] for word in words], dtype=torch.float64),
        SHOTS,
        replacement=True,
        generator=torch.Generator().manual_seed(7),
    ).tolist()
    counts = Counter(words[index] for index in indices)
    rows = tuple(
        Frequency(left, right, (point % 8, point // 8), count)
        for (left, right, point), count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    )
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    return rows, resources, error


def _packed(value: int, wires: tuple[int, ...]) -> int:
    return sum(((value >> bit) & 1) << wire for bit, wire in enumerate(wires))


def _read(word: int, wires: tuple[int, ...]) -> int:
    return sum(((word >> wire) & 1) << bit for bit, wire in enumerate(wires))


def _powers(point: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    twice = add(point, point)
    if twice is None:
        raise ValueError("degenerate generator")
    fourth = add(twice, twice)
    if fourth is None:
        raise ValueError("degenerate generator")
    return point, twice, fourth


def point_add_circuit() -> fq.Circuit:
    workspace = PointWorkspace.starting_at(6)
    x, y = (5, 4, 3), (2, 1, 0)
    gates: list[Gate] = [((), x[2]), ((), y[1])]
    gates.extend(point_add_gates(x, y, GENERATOR, workspace))
    return circuit_from_gates(workspace.ladder[-1] + 1, gates)


def period_finding_circuit() -> fq.Circuit:
    """Build the full source circuit, including all six controlled additions."""
    workspace = PointWorkspace.starting_at(12)
    circuit = fq.Circuit(workspace.ladder[-1] + 1)
    circuit.x(POINT_X[2]).x(POINT_Y[1])
    for wire in X1 + X2:
        circuit.h(wire)
    negative_target = (TARGET[0], -TARGET[1] % 7)
    for scalar, addend in ((X1, GENERATOR), (X2, negative_target)):
        for control, power in zip(scalar, _powers(addend), strict=True):
            for controls, wire in point_add_gates(
                POINT_X, POINT_Y, power, workspace, control=control
            ):
                circuit.gate(("x", "cx", "ccx")[len(controls)], (*controls, wire))
    append_qft(circuit, X1[::-1], inverse=True)
    append_qft(circuit, X2[::-1], inverse=True)
    return circuit


def trace_basis(
    instructions: Sequence[Instruction], width: int, inputs: Sequence[int]
) -> tuple[int, ...]:
    """Evaluate the actual X/CX/CCX instructions on many basis inputs in parallel."""
    all_inputs = (1 << len(inputs)) - 1
    masks = [
        sum(((word >> wire) & 1) << index for index, word in enumerate(inputs))
        for wire in range(width)
    ]
    for instruction in instructions:
        if instruction.name not in ("x", "cx", "ccx"):
            raise ValueError(f"basis trace does not support {instruction.name}")
        *controls, target = instruction.wires
        enabled = all_inputs
        for wire in controls:
            enabled &= masks[wire]
        masks[target] ^= enabled
    return tuple(
        sum(((mask >> index) & 1) << wire for wire, mask in enumerate(masks))
        for index in range(len(inputs))
    )


def checked_point_add(circuit: fq.Circuit) -> tuple[int, int]:
    (word,) = trace_basis(circuit.to_ir().instructions, circuit.n_qubits, (0,))
    if word >> 6:
        raise AssertionError("point addition left dirty workspace")
    return _read(word, (5, 4, 3)), _read(word, (2, 1, 0))


def checked_period_finding(circuit: fq.Circuit) -> float:
    """Check all branches and run the actual Fourier suffix on the clean subspace."""
    instructions = circuit.to_ir().instructions
    initial = instructions[:8]
    if [(op.name, op.wires) for op in initial] != [
        ("x", (POINT_X[2],)),
        ("x", (POINT_Y[1],)),
        *[("h", (wire,)) for wire in X1 + X2],
    ]:
        raise AssertionError("unexpected point seed or scalar preparation")
    fourier_start = len(instructions) - 14
    inputs = tuple(
        _packed(left, X1)
        | _packed(right, X2)
        | _packed(SEED[0], POINT_X)
        | _packed(SEED[1], POINT_Y)
        for left in range(8)
        for right in range(8)
    )
    words = trace_basis(instructions[8:fourier_start], circuit.n_qubits, inputs)
    if any(word >> 12 for word in words):
        raise AssertionError("point arithmetic left dirty workspace")
    for row, word in enumerate(words):
        if (_read(word, X1), _read(word, X2)) != divmod(row, 8):
            raise AssertionError("point arithmetic changed a scalar register")
    amplitudes = torch.zeros(4096, dtype=torch.complex128)
    for word in words:
        # The trace packs wire zero first; Circuit state indexes it as the MSB.
        index = int(f"{word:012b}"[::-1], 2)
        amplitudes[index] = 1 / 8
    compact = fq.Circuit(12, dtype=torch.complex128, inputs=amplitudes)
    for op in instructions[fourier_start:]:
        compact.gate(op.name, op.wires, params=op.params)
    error = float(
        (
            compact.state().flatten()
            - fq.run(reference_circuit()).statevector().flatten()
        )
        .abs()
        .max()
    )
    if error > 3e-12:
        raise AssertionError(f"arithmetic period finding differs by {error}")
    return error

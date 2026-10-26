"""Approved finite coherent-reference algebra, not a production branch engine."""

from __future__ import annotations

import cmath
import itertools
import math
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
import torch

import flagquantum as fq
from flagquantum.benchmarking.ftqc_mixed_reference import _reference

pytestmark = pytest.mark.integration

_Instruction = tuple[str, tuple[int, ...]]
_MATRIX = tuple(itertools.product((2, 3), (0, 1, 2, 4)))
_B = (cmath.exp(1j * math.pi / 4) - 1) / (1j - 1)
_A = 1 - _B
_ATOL = 1e-12


@dataclass(frozen=True)
class _Reference:
    n: int
    instructions: tuple[_Instruction, ...]
    coefficients: tuple[complex, ...]
    branches: tuple[torch.Tensor, ...]
    direct: torch.Tensor

    def coherent(self) -> torch.Tensor:
        return sum(
            (
                coefficient * branch
                for coefficient, branch in zip(
                    self.coefficients, self.branches, strict=True
                )
            ),
            torch.zeros_like(self.direct),
        )


def _matrix_description(n: int, t: int) -> tuple[_Instruction, ...]:
    if type(n) is not int or type(t) is not int or (n, t) not in _MATRIX:
        raise ValueError("outside the approved eight-case matrix")
    instructions: list[_Instruction] = [("h", (qubit,)) for qubit in range(n)]
    for layer in range(4):
        instructions.append(("s", ((layer + 1) % n,)))
        instructions.extend(("cx", (qubit, qubit + 1)) for qubit in range(n - 1))
        if layer < t:
            instructions.append(("t", (layer % n,)))
    instructions.extend(("h", (qubit,)) for qubit in range(n))
    return tuple(instructions)


def _clifford_vector(n: int, instructions: tuple[_Instruction, ...]) -> torch.Tensor:
    # Canonical scalar amplitudes; no production matrices, compiler or evolution.
    amplitudes = [0j] * (1 << n)
    amplitudes[0] = 1 + 0j
    for gate, qubits in instructions:
        mask = 1 << (n - 1 - qubits[0])
        if gate == "h":
            previous = amplitudes[:]
            amplitudes = [
                (
                    previous[index & ~mask]
                    + (-1 if index & mask else 1) * previous[index | mask]
                )
                / math.sqrt(2)
                for index in range(1 << n)
            ]
        elif gate == "s":
            amplitudes = [
                value * (1j if index & mask else 1)
                for index, value in enumerate(amplitudes)
            ]
        elif gate == "cx":
            target = 1 << (n - 1 - qubits[1])
            amplitudes = [
                amplitudes[index ^ target if index & mask else index]
                for index in range(1 << n)
            ]
        elif gate == "x":
            amplitudes = [amplitudes[index ^ mask] for index in range(1 << n)]
        elif gate in ("p0", "p1"):
            # Private projected terms retain their unnormalized amplitudes.
            outcome = gate == "p1"
            amplitudes = [
                value if bool(index & mask) == outcome else 0j
                for index, value in enumerate(amplitudes)
            ]
        else:
            raise ValueError("private branch contains a non-Clifford instruction")
    return torch.tensor(amplitudes, dtype=torch.complex128)


def _expand(
    n: int, instructions: tuple[_Instruction, ...], direct: torch.Tensor
) -> _Reference:
    t = sum(gate == "t" for gate, _ in instructions)
    coefficients: list[complex] = []
    branches: list[torch.Tensor] = []
    for choices in itertools.product((0, 1), repeat=t):
        position = 0
        coefficient = 1 + 0j
        clifford: list[_Instruction] = []
        for gate, qubits in instructions:
            if gate == "t":
                choice = choices[position]
                position += 1
                coefficient *= _B if choice else _A
                if choice:
                    clifford.append(("s", qubits))
            else:
                clifford.append((gate, qubits))
        coefficients.append(coefficient)
        branches.append(_clifford_vector(n, tuple(clifford)))
    return _Reference(n, instructions, tuple(coefficients), tuple(branches), direct)


def _matrix_reference(n: int, t: int) -> _Reference:
    instructions = _matrix_description(n, t)
    # This existing basis oracle uses direct T phases and independent gate order.
    return _expand(n, instructions, _reference(n, t, "chain"))


def _unit_density(state: torch.Tensor, dimension: int) -> torch.Tensor:
    if (
        state.shape != (dimension,)
        or state.dtype != torch.complex128
        or state.device.type != "cpu"
        or not bool(torch.isfinite(state).all())
    ):
        raise ValueError("reference requires finite ordered CPU complex128 amplitudes")
    squared_norm = float(state.abs().square().sum())
    if squared_norm == 0:
        raise ValueError("cannot normalize a zero reference")
    if abs(squared_norm - 1) > _ATOL:
        raise ValueError("reference lost unit norm")
    # Check norm before the outer product; no normalization hides loss or gain.
    return state[:, None] * state.conj()[None, :]


@pytest.fixture(scope="module", autouse=True)
def _one_cpu_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def references() -> dict[str, _Reference]:
    cases = {f"n{n}t{t}": _matrix_reference(n, t) for n, t in _MATRIX}
    phase = cmath.exp(1j * math.pi / 4)
    cases["hth"] = _expand(
        1,
        (("h", (0,)), ("t", (0,)), ("h", (0,))),
        torch.tensor([1 + phase, 1 - phase], dtype=torch.complex128) / 2,
    )
    cases["bell"] = _expand(
        2,
        (("h", (0,)), ("cx", (0, 1)), ("t", (0,)), ("h", (0,)), ("h", (1,))),
        torch.tensor(
            [1 + phase, 1 - phase, 1 - phase, 1 + phase], dtype=torch.complex128
        )
        / (2 * math.sqrt(2)),
    )
    assert sum(len(case.branches) for case in cases.values()) == 50
    return cases


@pytest.mark.parametrize("name", [f"n{n}t{t}" for n, t in _MATRIX] + ["hth", "bell"])
def test_coherent_reference_matches_direct_and_unexpanded_public_execution(
    name: str, references: dict[str, _Reference]
) -> None:
    case = references[name]
    coherent = case.coherent()
    torch.testing.assert_close(coherent, case.direct, atol=_ATOL, rtol=0)
    for branch in case.branches:
        _unit_density(branch, 1 << case.n)
    density = _unit_density(coherent, 1 << case.n)
    torch.testing.assert_close(
        density, _unit_density(case.direct, 1 << case.n), atol=_ATOL, rtol=0
    )
    circuit = fq.Circuit(case.n)
    for gate, qubits in case.instructions:
        if gate == "h":
            circuit.h(qubits[0])
        elif gate == "s":
            circuit.s(qubits[0])
        elif gate == "t":
            circuit.t(qubits[0])
        else:
            assert gate == "cx"
            circuit.cx(*qubits)
    result = fq.run(
        circuit,
        outputs=fq.probabilities(list(range(case.n))),
        options=fq.ExecutionOptions(
            mode="statevector",
            device="cpu",
            precision="complex128",
            allow_approximate=False,
            allow_backend_fallback=False,
        ),
    )
    state = result.to_statevector().reshape(-1)
    torch.testing.assert_close(
        _unit_density(state, 1 << case.n), density, atol=_ATOL, rtol=0
    )
    probabilities = result.probabilities
    assert probabilities is not None and probabilities.shape == (1, 1 << case.n)
    assert probabilities.dtype == torch.float64 and probabilities.device.type == "cpu"
    assert bool(torch.isfinite(probabilities).all())
    torch.testing.assert_close(
        probabilities[0], coherent.abs().square(), atol=_ATOL, rtol=0
    )
    assert abs(float(probabilities.sum()) - 1) <= _ATOL
    if name in ("hth", "bell"):
        high, low = (2 + math.sqrt(2)), (2 - math.sqrt(2))
        analytic = (
            [high / 4, low / 4]
            if name == "hth"
            else [high / 8, low / 8, low / 8, high / 8]
        )
        torch.testing.assert_close(
            probabilities[0],
            torch.tensor(analytic, dtype=torch.float64),
            atol=_ATOL,
            rtol=0,
        )


def test_compensated_branch_phase_and_common_phase_preserve_interference(
    references: dict[str, _Reference],
) -> None:
    case = references["bell"]
    first, second = case.branches
    a, b = case.coefficients
    phase = cmath.exp(0.73j)
    compensated = (a / phase) * (phase * first) + b * second
    torch.testing.assert_close(compensated, case.coherent(), atol=_ATOL, rtol=0)
    torch.testing.assert_close(
        _unit_density(phase * case.coherent(), 4),
        _unit_density(case.coherent(), 4),
        atol=_ATOL,
        rtol=0,
    )
    wrong = a * (phase * first) + b * second
    assert (
        float((wrong.abs().square() - case.coherent().abs().square()).abs().max())
        > 0.01
    )


def test_squared_coefficient_mixture_and_omitted_cross_terms_are_wrong(
    references: dict[str, _Reference],
) -> None:
    case = references["bell"]
    diagonal = sum(
        (
            abs(coefficient) ** 2 * _unit_density(branch, 4)
            for coefficient, branch in zip(
                case.coefficients, case.branches, strict=True
            )
        ),
        torch.zeros((4, 4), dtype=torch.complex128),
    )
    trace = torch.trace(diagonal).real
    assert abs(float(trace) - 1) > 0.1
    mixture = diagonal.diag().real / trace
    torch.testing.assert_close(
        mixture,
        torch.tensor([3 / 8, 1 / 8, 1 / 8, 3 / 8], dtype=torch.float64),
        atol=_ATOL,
        rtol=0,
    )
    assert float((mixture - case.coherent().abs().square()).abs().max()) > 0.05


def test_cancelled_reference_is_refused_before_normalization(
    references: dict[str, _Reference],
) -> None:
    branch = references["bell"].branches[0]
    zero = (1 + 0j) * branch + (-1 + 0j) * branch
    assert bool((zero == 0).all())
    with pytest.raises(ValueError, match="zero reference"):
        _unit_density(zero, 4)


@pytest.mark.parametrize("n,t", [(1, 1), (4, 0), (3, 3), (2, 8), (True, 0), (2, False)])
def test_outside_matrix_is_refused_before_branch_allocation(n: int, t: int) -> None:
    with pytest.raises(ValueError, match="approved eight-case matrix"):
        _matrix_reference(n, t)


@pytest.mark.parametrize("defect", ["shape", "nonfinite", "norm", "dtype"])
def test_invalid_reference_cannot_be_hidden_by_density_normalization(
    defect: str, references: dict[str, _Reference]
) -> None:
    state = references["bell"].coherent().clone()
    if defect == "shape":
        state = state.reshape(2, 2)
    elif defect == "nonfinite":
        state[0] = torch.tensor(complex(float("nan"), 0), dtype=state.dtype)
    elif defect == "norm":
        state *= 2
    else:
        state = state.to(torch.complex64)
    with pytest.raises(ValueError):
        _unit_density(state, 4)

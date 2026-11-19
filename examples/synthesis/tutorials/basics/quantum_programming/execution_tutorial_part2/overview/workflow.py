"""Run symbolic native templates and verify each bound two-qubit state."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from examples.synthesis.tutorials.basics.quantum_programming.execution_tutorial_part2.main import (
    main as native_bell,
)
from flagquantum.algorithms import CobylaOptimizer
from flagquantum.synth import synthesize

SHOTS = 2048
ANGLE_COUNT = 50
BUDGET = 200

PROJECTOR = 0.25 * (
    fq.I() + (fq.X(0) @ fq.X(1)) - (fq.Y(0) @ fq.Y(1)) + (fq.Z(0) @ fq.Z(1))
)


@dataclass(frozen=True)
class Observation:
    angle: float
    counts: dict[str, int]
    expectation: float
    sampled_expectation: float
    projector_shots: int
    state_error: float


@dataclass(frozen=True)
class Trial:
    angles: tuple[float, float]
    fidelity: float


@dataclass(frozen=True)
class Optimization:
    trials: tuple[Trial, ...]
    converged: bool
    budget_stop: bool
    state_error: float


class EvaluationBudgetError(Exception):
    """Stop before executing more than the source's query budget."""


def template() -> fq.Circuit:
    return (
        fq.Circuit(2, dtype=torch.complex128)
        .x(0)
        .ry(0, theta=fq.Parameter("angle"))
        .cx(0, 1)
    )


def bell_template() -> fq.Circuit:
    return (
        fq.Circuit(2, dtype=torch.complex128)
        .rx(0, theta=fq.Parameter("rx"))
        .ry(0, theta=fq.Parameter("ry"))
        .cx(0, 1)
    )


def angles() -> tuple[float, ...]:
    return tuple(
        float(value)
        for value in torch.linspace(0, math.tau, ANGLE_COUNT, dtype=torch.float64)
    )


def bind(circuit: fq.Circuit, angle: float) -> fq.Circuit:
    if not math.isfinite(angle):
        raise ValueError("the runtime angle must be finite")
    result = circuit.bind_parameters({"angle": angle})
    if tuple(circuit.parameter_names) != ("angle",) or result.parameter_names:
        raise AssertionError("binding must preserve the original symbolic template")
    return result


def _fidelity(circuit: fq.Circuit) -> float:
    actual = fq.run(circuit, outputs=fq.expectation(PROJECTOR, name="bell_fidelity"))
    return float(actual.expectation("bell_fidelity").reshape(()))


def sample(circuit: fq.Circuit, angle: float, *, seed: int) -> dict[str, int]:
    bound = bind(circuit, angle)
    counts = bound.counts(shots=SHOTS, generator=torch.Generator().manual_seed(seed))[0]
    return {
        str(word): count
        for word, count in sorted(counts.items(), key=lambda row: str(row[0]))
    }


def observe(circuit: fq.Circuit, angle: float) -> tuple[float, float]:
    bound = bind(circuit, angle)
    state = fq.run(bound).statevector().flatten()
    signed_sine, cosine = -math.sin(angle / 2), math.cos(angle / 2)
    expected_state = torch.tensor((signed_sine, 0, 0, cosine), dtype=torch.complex128)
    error = float((state - expected_state).abs().max())
    fidelity = _fidelity(bound)
    assert error < 3e-12 and abs(fidelity - 0.5 * (signed_sine + cosine) ** 2) < 3e-12
    return fidelity, error


def sample_projector(
    circuit: fq.Circuit, angle: float, *, seed: int
) -> tuple[float, int]:
    """Estimate the native Bell projector with actual shots in its eigenbasis."""
    bound = bind(circuit, angle)
    bound.cx(0, 1).h(0)
    observed = bound.counts(shots=SHOTS, generator=torch.Generator().manual_seed(seed))[
        0
    ]
    shots = sum(observed.values())
    assert shots == SHOTS
    return observed.get("00", 0) / shots, shots


def sweep(circuit: fq.Circuit) -> tuple[Observation, ...]:
    rows = []
    for seed, angle in enumerate(angles(), 711):
        fidelity, error = observe(circuit, angle)
        estimate, projector_shots = sample_projector(circuit, angle, seed=seed + 100)
        rows.append(
            Observation(
                angle,
                sample(circuit, angle, seed=seed),
                fidelity,
                estimate,
                projector_shots,
                error,
            )
        )
    return tuple(rows)


def optimize() -> Optimization:
    circuit = bell_template()
    trace: list[Trial] = []
    maximum_error = 0.0
    for rx, ry in ((0.23, 0.49), (-0.37, 0.84)):
        retained = synthesize(native_bell, classical_inputs={"angles": [rx, ry]})
        expected = fq.run(retained).statevector().flatten()
        bound = circuit.bind_parameters({"rx": rx, "ry": ry})
        candidate = fq.run(bound).statevector().flatten()
        difference = float((candidate - expected).abs().max())
        assert difference < 3e-12
        maximum_error = max(maximum_error, difference)

    def evaluate(point: torch.Tensor) -> torch.Tensor:
        nonlocal maximum_error
        if len(trace) == BUDGET:
            raise EvaluationBudgetError
        rx, ry = (float(value) for value in point)
        bound = circuit.bind_parameters({"rx": rx, "ry": ry})
        stored = fq.run(bound).statevector().flatten()
        a, b = math.cos(rx / 2), -1j * math.sin(rx / 2)
        c, d = math.cos(ry / 2), math.sin(ry / 2)
        reference = torch.tensor(
            (c * a - d * b, 0, 0, d * a + c * b), dtype=torch.complex128
        )
        error = float((stored - reference).abs().max())
        fidelity = _fidelity(bound)
        expected = 0.5 * abs((c + d) * a + (c - d) * b) ** 2
        assert error < 3e-12 and abs(fidelity - expected) < 3e-12
        maximum_error = max(maximum_error, error)
        trace.append(Trial((rx, ry), fidelity))
        return torch.tensor(-fidelity, dtype=torch.float64)

    converged = stopped = False
    try:
        result = CobylaOptimizer(maxiter=BUDGET, rhobeg=1.0, rhoend=1e-8).minimize(
            evaluate,
            torch.zeros(2, dtype=torch.float64),
        )
        converged = result.converged
    except EvaluationBudgetError:
        stopped = True
    return Optimization(tuple(trace), converged, stopped, maximum_error)

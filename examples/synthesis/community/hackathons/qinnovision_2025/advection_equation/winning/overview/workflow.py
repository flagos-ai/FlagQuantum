"""Run the complete original winning advection exercise and both resource sweeps."""

from dataclasses import dataclass

import torch

from . import physics, quantum, reference

VARIANTS: tuple[quantum.Variant, ...] = ("proposed", "custom", "trotter")


@dataclass(frozen=True)
class Curve:
    name: str
    corrected: tuple[float, ...]
    fidelity: float
    phase: float
    sign: float
    absolute_state_error: float
    width: int
    depth: int
    cx: int


@dataclass(frozen=True)
class Result:
    axis: tuple[float, ...]
    norm: float
    classical_times: tuple[float, ...]
    classical_profiles: tuple[tuple[float, ...], ...]
    reference: tuple[float, ...]
    exponential: tuple[float, ...]
    curves: tuple[Curve, ...]
    pauli_terms: int
    decomposition_error: float
    qubit_widths: tuple[int, ...]
    qubit_cx: tuple[tuple[int, ...], ...]
    qubit_theory: tuple[int, ...]
    step_counts: tuple[int, ...]
    step_cx: tuple[tuple[int, ...], ...]


def correct(
    state: torch.Tensor, baseline: torch.Tensor
) -> tuple[torch.Tensor, float, float]:
    index = int(state.abs().argmax())
    phase = float(state[index].angle())
    corrected = (
        state * torch.exp(torch.tensor(-1j * phase, dtype=torch.complex128))
    ).real
    sign = float(corrected[index].sign() * baseline[index].sign())
    return corrected * sign, phase, sign


def run() -> Result:
    problem = physics.create()
    classical = physics.rk4(problem)
    baseline = classical[:, int(1 / 0.038)]
    exact = physics.exponential(problem)[:, -1]
    terms = quantum.paulis(problem)
    decomposition = float(
        (reference.reassemble(problem, terms) + 1j * problem.derivative).abs().max()
    )
    curves = []
    for variant in VARIANTS:
        circuit = quantum.build(problem, variant)
        state = quantum.state(circuit)
        expected = (
            reference.trotter(problem, terms)
            if variant == "trotter"
            else reference.proposed(problem)
        )
        corrected, phase, sign = correct(state, baseline)
        fidelity = float(abs(corrected @ (baseline / baseline.norm())) ** 2)
        resource = quantum.metrics(circuit)
        curves.append(
            Curve(
                variant,
                tuple(float(v) for v in corrected),
                fidelity,
                phase,
                sign,
                float((state - expected).abs().max()),
                resource.n_qubits,
                resource.depth,
                resource.operation_counts.get("cx", 0),
            )
        )
    widths = tuple(range(3, 7))
    width_cx = tuple(
        tuple(
            quantum.metrics(
                quantum.build(physics.create(n), variant, steps=1, prepare=False)
            ).operation_counts.get("cx", 0)
            for n in widths
        )
        for variant in VARIANTS
    )
    steps = (1, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    step_cx = tuple(
        tuple(
            quantum.metrics(
                quantum.build(physics.create(3), variant, steps=s)
            ).operation_counts.get("cx", 0)
            for s in steps
        )
        for variant in VARIANTS
    )
    times = (0.0, 0.5, 1.0)
    profiles = tuple(
        tuple(float(v) * problem.norm for v in classical[:, int(t / 0.038)])
        for t in times
    )
    return Result(
        tuple(float(v) for v in problem.axis),
        problem.norm,
        times,
        profiles,
        tuple(float(v) for v in baseline),
        tuple(float(v) for v in exact),
        tuple(curves),
        len(terms),
        decomposition,
        widths,
        width_cx,
        tuple(9 * n * n - 33 * n + 34 for n in widths),
        steps,
        step_cx,
    )

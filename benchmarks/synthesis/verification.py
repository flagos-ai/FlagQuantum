"""Bounded independent gate interpretation and complex-amplitude checks."""

import random
from dataclasses import dataclass
from itertools import product

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.core.ir import CircuitIR

from .expressions import Case
from .oracle import evaluate, expected_amplitudes

SEED = 20261102


@dataclass(frozen=True)
class Verification:
    basis_checks: int
    basis_domain_exhaustive: bool
    nonzero_targets: bool
    coherent_states: int
    maximum_amplitude_error: float | None
    workspace_nonzero_probability: float | None
    scope: str


def encode(values: tuple[int, ...], wires: tuple[tuple[int, ...], ...]) -> int:
    """Encode with wire zero as the least-significant bit for gate interpretation."""
    state = 0
    for value, register in zip(values, wires, strict=True):
        for bit, wire in enumerate(reversed(register)):
            state |= ((value >> bit) & 1) << wire
    return state


def classical_apply(circuit: CircuitIR, state: int) -> int:
    """Interpret exact X/CX/CCX permutations without calling synthesis kernels."""
    for gate in circuit.instructions:
        if (
            gate.name not in {"x", "cx", "ccx"}
            or gate.matrix is not None
            or gate.params
            or gate.metadata
        ):
            raise ValueError("basis oracle requires exact static X/CX/CCX gates")
        controls, target = gate.wires[:-1], gate.wires[-1]
        if all(state & (1 << wire) for wire in controls):
            state ^= 1 << target
    return state


def basis_checks(case: Case, program: synth.Program) -> tuple[int, bool]:
    bits = sum(case.widths)
    maxima = tuple((1 << width) - 1 for width in case.widths)
    rng = random.Random(SEED)
    values = (
        list(product(*(range(limit + 1) for limit in maxima)))
        if bits <= 6
        else [tuple(0 for _ in maxima), maxima]
        + [tuple(rng.randrange(limit + 1) for limit in maxima) for _ in range(16)]
    )
    wires = tuple(
        program.registers[name] for name in (*case.expression.arguments, "result")
    )
    target_max = (1 << len(wires[-1])) - 1
    # Every target on the two-bit sentinel; other cases have three fixed targets.
    targets = (
        tuple(range(target_max + 1))
        if case.identifier == "energy-2x2"
        else tuple(sorted({0, 1, target_max}))
    )
    for inputs in values:
        answer = evaluate(case.expression.name, inputs)
        for target in targets:
            initial = encode((*inputs, target), wires)
            expected = encode((*inputs, target ^ answer), wires)
            if classical_apply(program.arithmetic, initial) != expected:
                raise AssertionError(
                    f"basis/XOR/cleanup mismatch: {case.identifier}, {inputs}, {target}"
                )
    return len(values) * len(targets), bits <= 6


def coherent_check(
    case: Case, program: synth.Program, *, phased: bool
) -> tuple[float, float]:
    state = fq.run(program).statevector().reshape(-1)
    expected = expected_amplitudes(
        case, program.registers, program.circuit.n_wires, phased=phased
    )
    residual = state.clone()
    for index, amplitude in expected.items():
        residual[index] -= amplitude
    error = float(residual.abs().max().item())
    logical = {wire for ports in program.registers.values() for wire in ports}
    indices = torch.arange(state.numel(), device=state.device)
    dirty = torch.zeros(state.numel(), dtype=torch.bool, device=state.device)
    for wire in range(program.circuit.n_wires):
        if wire not in logical:
            dirty |= indices.bitwise_and(1 << (program.circuit.n_wires - 1 - wire)) != 0
    leakage = float(state[dirty].abs().square().sum().item())
    if error > 1e-12 or leakage > 1e-12:
        raise AssertionError(
            f"complex amplitude/cleanup mismatch: {case.identifier}: {error}, {leakage}"
        )
    return error, leakage

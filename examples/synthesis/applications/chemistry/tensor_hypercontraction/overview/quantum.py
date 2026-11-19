"""Compile the source-sized PREPARE, SELECT, reflection, and clock stages."""

import math
from dataclasses import dataclass

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.algorithms.primitives.qft import append_qft
from flagquantum.synth import (
    Output,
    QArray,
    QBit,
    allocate,
    inplace_prepare_amplitudes,
    qfunc,
)

from .layout import Layout, allocate_layout
from .model import CLOCK_STEPS, N_EPSILON, N_MN, N_UD, Model
from .reversible import (
    Leaf,
    less_than_xor,
    mcx,
    positive_phase_flip,
    qrom_rows,
    table_walk,
    wide_phase_flip,
    xor_polynomial,
)


@dataclass(frozen=True)
class Compilation:
    layout: Layout
    prepare_one: fq.Circuit
    prepare_two: fq.Circuit
    select: fq.Circuit
    complete: fq.Circuit


def empty(layout: Layout) -> fq.Circuit:
    return fq.Circuit(layout.width, dtype=torch.complex128)


def hadamards(circuit: fq.Circuit, qubits: tuple[int, ...]) -> None:
    for qubit in qubits:
        circuit.h(qubit)


def equal_superposition(model: Model, layout: Layout) -> fq.Circuit:
    circuit = empty(layout)
    hadamards(circuit, layout.mu + layout.nu)
    allowed, marker = layout.comparison[:2]
    predicate = empty(layout)
    less_than_xor(predicate, layout.nu, layout.mu, allowed, layout.comparison[2:])
    predicate.x(allowed)
    # The unsigned five-bit nu is always <= 31 and never equals 32.
    # The remaining condition is !(m1 and mu >= 3).
    flags = [wire for wire in layout.mu[2:]]
    for subset in range(1, 1 << len(flags)):
        controls = tuple(wire for i, wire in enumerate(flags) if subset >> i & 1)
        mcx(predicate, (layout.m1, *controls), marker, layout.ladder)
        mcx(
            predicate,
            (layout.m1, layout.mu[0], layout.mu[1], *controls),
            marker,
            layout.ladder,
        )
    mcx(predicate, (layout.m1, layout.mu[0], layout.mu[1]), marker, layout.ladder)
    predicate.x(marker)
    undo = predicate.adjoint()

    circuit.ry(layout.r_br, theta=model.rotation).compose(predicate)
    positive_phase_flip(circuit, (allowed, marker), layout.r_br, layout.ladder)
    circuit.compose(undo).ry(layout.r_br, theta=-model.rotation)
    hadamards(circuit, layout.mu + layout.nu)
    positive_phase_flip(circuit, layout.mu + layout.nu, layout.r_br, layout.ladder)
    hadamards(circuit, layout.mu + layout.nu)
    circuit.compose(predicate).ccx(allowed, marker, layout.succ).compose(undo)
    return circuit


def index_arithmetic(layout: Layout) -> fq.Circuit:
    circuit = empty(layout)
    mask = (1 << N_MN) - 1
    values = []
    for pair in range(1 << (2 * N_MN)):
        mu, nu = pair & mask, pair >> N_MN
        values.append(nu * ((nu >> 1) + 1) + mu)
    xor_polynomial(values, layout.mu + layout.nu, layout.s, layout.ladder, circuit)
    return circuit


def finish_prepare(model: Model, layout: Layout) -> fq.Circuit:
    circuit = index_arithmetic(layout)
    hadamards(circuit, layout.sigma)
    targets = layout.mu_alt + layout.nu_alt + layout.keep + (layout.theta,)
    rows = tuple(
        mu + nu + keep + (theta,)
        for mu, nu, keep, theta in zip(
            model.mu_alt, model.nu_alt, model.keep, model.theta, strict=True
        )
    )
    table_walk(circuit, layout.s, qrom_rows(rows, targets), layout.path[:N_MN])

    flag = layout.comparison[0]
    compare = empty(layout)
    less_than_xor(compare, layout.keep, layout.sigma, flag, layout.comparison[2:])
    circuit.compose(compare).cz(layout.theta_alt, flag)
    circuit.x(flag).cz(layout.theta, flag).x(flag)
    for first, second in zip(
        layout.mu + layout.nu, layout.mu_alt + layout.nu_alt, strict=True
    ):
        circuit.cswap(flag, first, second)
    circuit.compose(compare.adjoint())
    circuit.h(layout.aux).x(layout.m1)
    swap_flag = layout.comparison[1]
    circuit.ccx(layout.aux, layout.m1, swap_flag)
    for mu, nu in zip(layout.mu, layout.nu, strict=True):
        circuit.cswap(swap_flag, mu, nu)
    circuit.ccx(layout.aux, layout.m1, swap_flag).x(layout.m1)
    return circuit


def basis_change(layout: Layout) -> fq.Circuit:
    circuit = empty(layout)
    for i in range(N_UD - 1):
        first, second = layout.down[i : i + 2]
        bits = layout.rotation[i * N_EPSILON : (i + 1) * N_EPSILON]
        for x_wire, y_wire, sign in ((first, second, 1), (second, first, -1)):
            circuit.h(x_wire).sdg(y_wire).h(y_wire).cx(x_wire, y_wire)
            for position, bit in enumerate(bits):
                circuit.crz(
                    bit, y_wire, theta=sign * math.pi * 2 ** (-position - N_EPSILON)
                )
            circuit.cx(x_wire, y_wire).h(y_wire).s(y_wire).h(x_wire)
    return circuit


def select_block(model: Model, layout: Layout) -> fq.Circuit:
    circuit = empty(layout)
    digital = basis_change(layout).adjoint()
    for address, rows, control, swap_control in (
        (layout.mu, model.rot_mu, layout.m1, layout.plus1),
        (layout.nu, model.rot_nu, None, layout.plus2),
    ):
        block = empty(layout)
        table_walk(
            block,
            address,
            qrom_rows(rows, layout.rotation),
            layout.path[:N_MN],
            control=control,
        )
        for down, up in zip(layout.down, layout.up, strict=True):
            block.cswap(swap_control, down, up)
        block.compose(digital)
        circuit.compose(block)
        if control is None:
            circuit.x(layout.m1).cphase(layout.succ, layout.m1, theta=math.pi / 2)
            positive_phase_flip(
                circuit, (layout.succ, layout.m1), layout.down[0], layout.ladder
            )
            circuit.x(layout.m1)
        else:
            circuit.s(layout.succ).cz(layout.succ, layout.down[0])
        circuit.compose(block.adjoint())
    return circuit


@qfunc
def clock_state(weights: list[float], target: Output[QArray[QBit]]) -> None:
    allocate(4, target)
    inplace_prepare_amplitudes(weights, 0, target)


def full_walk(
    model: Model,
    layout: Layout,
    prepare: fq.Circuit,
    inverse: fq.Circuit,
    select: fq.Circuit,
) -> fq.Circuit:
    def leaf(with_select: bool) -> Leaf:
        def walk(circuit: fq.Circuit, control: int) -> None:
            if with_select:
                circuit.compose(select)
            circuit.compose(inverse)
            for wire in (*layout.ell, control):
                circuit.x(wire)
            first, second = layout.comparison[:2]
            wide_phase_flip(circuit, layout.ell, control, (first, second))
            for wire in (*layout.ell, control):
                circuit.x(wire)
            circuit.compose(prepare)

        return walk

    complete = empty(layout)
    initialized = synth.synthesize(
        clock_state,
        classical_inputs={"weights": list(model.clock)},
        max_qubits=4,
        max_workspace=0,
        memory_limit_bytes=1024**2,
    )
    complete.compose(initialized.circuit, qubits=layout.clock[::-1])
    complete.h(layout.plus1).h(layout.plus2)
    table_walk(
        complete,
        layout.clock,
        (leaf(False),) + (leaf(True),) * (CLOCK_STEPS - 1),
        layout.path[N_MN:],
    )
    append_qft(complete, layout.clock[::-1], inverse=True)
    return complete


def compile_all(model: Model) -> Compilation:
    layout = allocate_layout()
    early = equal_superposition(model, layout)
    late = finish_prepare(model, layout)
    select = select_block(model, layout)
    prepare = empty(layout).compose(early).compose(late)
    inverse = prepare.adjoint()

    complete = full_walk(model, layout, prepare, inverse, select)
    return Compilation(layout, early, late, select, complete)

"""Verify emitted reversible subcircuits independently of the construction."""

from collections.abc import Sequence
from dataclasses import dataclass

import flagquantum as fq

from .layout import Layout
from .model import N_ALEPH, N_EPSILON, N_MN, ROWS, Model
from .quantum import empty, index_arithmetic
from .reversible import Leaf, less_than_xor, qrom_rows, table_walk, wide_phase_flip


def pack_word(word: int, wires: Sequence[int]) -> int:
    return sum(((word >> index) & 1) << wire for index, wire in enumerate(wires))


def trace(circuit: fq.Circuit, words: Sequence[int]) -> tuple[int, ...]:
    """Apply only reversible Boolean instructions to all input witnesses."""
    columns = [
        sum(((word >> wire) & 1) << index for index, word in enumerate(words))
        for wire in range(circuit.n_qubits)
    ]
    mask = (1 << len(words)) - 1
    for operation in circuit.to_ir().instructions:
        match operation.name:
            case "x":
                columns[operation.wires[0]] ^= mask
            case "cx" | "ccx":
                *controls, target = operation.wires
                active = mask
                for wire in controls:
                    active &= columns[wire]
                columns[target] ^= active
            case _:
                raise ValueError(f"unexpected Boolean-trace opcode {operation.name}")
    return tuple(
        sum(((column >> index) & 1) << wire for wire, column in enumerate(columns))
        for index in range(len(words))
    )


def arithmetic_check(layout: Layout) -> int:
    circuit = index_arithmetic(layout)
    words = []
    expected = []
    size = 1 << N_MN
    limit = (1 << len(layout.s)) - 1
    for mu in range(size):
        for nu in range(size):
            residue = (17 * mu + nu) & limit
            operands = pack_word(mu, layout.mu) | pack_word(nu, layout.nu)
            words.append(operands | pack_word(residue, layout.s))
            value = (nu * (nu // 2 + 1) + mu) & limit
            expected.append(operands | pack_word(residue ^ value, layout.s))
    assert trace(circuit, words) == tuple(expected)
    return len(words)


def _selected_row(address: int, depth: int, entries: int) -> int:
    """Decode a compressed source-compatible table without compiling a circuit."""
    offset = 0
    for bit in range(depth):
        if entries == 1:
            break
        top = 1 << ((entries - 1).bit_length() - 1)
        if address & (1 << bit):
            offset += top
            entries -= top
        else:
            entries = top
    return offset


def qrom_check(model: Model, layout: Layout) -> int:
    witnesses = 0
    for address, data, control in (
        (layout.mu, model.rot_mu, layout.m1),
        (layout.nu, model.rot_nu, None),
    ):
        circuit = empty(layout)
        table_walk(
            circuit,
            address,
            qrom_rows(data, layout.rotation),
            layout.path[:N_MN],
            control,
        )
        words = []
        expected = []
        for enabled in ((0, 1) if control is not None else (1,)):
            for word in range(1 << N_MN):
                address_bits = pack_word(word, address)
                base = (
                    address_bits
                    if control is None
                    else address_bits | enabled << control
                )
                initial = word * 17
                case = base | pack_word(initial, layout.rotation)
                row = data[_selected_row(word, N_MN, ROWS)]
                lookup = sum(int(flag) << i for i, flag in enumerate(row))
                case_target = initial ^ lookup if enabled else initial
                words.append(case)
                expected.append(base | pack_word(case_target, layout.rotation))
        assert trace(circuit, words) == tuple(expected)
        witnesses += len(words)
    return witnesses


def clock_check(layout: Layout) -> int:
    circuit = empty(layout)
    witness = layout.succ

    def leaf(index: int) -> Leaf:
        def record(tape: fq.Circuit, active: int) -> None:
            if index == 0:
                tape.cx(active, witness)
            else:
                tape.cx(active, layout.ell[index + 20])

        return record

    entries = (1 << N_EPSILON) - 1
    table_walk(
        circuit,
        layout.clock,
        tuple(leaf(i) for i in range(entries)),
        layout.path[N_MN:],
    )
    words = [pack_word(i, layout.clock) for i in range(1 << N_EPSILON)]
    expected = []
    for index, word in enumerate(words):
        selected = _selected_row(index, N_EPSILON, entries)
        wire = witness if selected == 0 else layout.ell[selected + 20]
        expected.append(word | 1 << wire)
    assert trace(circuit, words) == tuple(expected)
    return len(words)


def comparator_check(layout: Layout) -> int:
    circuit = empty(layout)
    less_than_xor(circuit, layout.keep, layout.sigma, layout.succ, layout.comparison)
    values = (
        0,
        1,
        42,
        (1 << (N_ALEPH - 1)) - 1,
        1 << (N_ALEPH - 1),
        (1 << N_ALEPH) - 1,
    )
    words = []
    expected = []
    for left in values:
        for right in values:
            for target in (0, 1):
                operands = pack_word(left, layout.keep) | pack_word(right, layout.sigma)
                words.append(operands | target << layout.succ)
                expected.append(operands | (target ^ int(left < right)) << layout.succ)
    assert trace(circuit, words) == tuple(expected)
    return len(words)


def reflection_check(layout: Layout) -> int:
    """Check the H-conjugated saturation on every single-zero control witness."""
    target = layout.clock[0]
    first, second = layout.comparison[:2]
    circuit = empty(layout)
    wide_phase_flip(circuit, layout.ell, target, (first, second))
    ops = circuit.to_ir().instructions
    assert (ops[0].name, ops[-1].name) == ("ccx", "ccx")
    classical = empty(layout)
    for op in ops:
        if op.name == "h":
            assert op.wires == (target,)
            continue
        classical.gate(op.name, op.wires)
    saturation = sum(1 << wire for wire in layout.ell)
    patterns = (0, saturation, *(saturation ^ (1 << wire) for wire in layout.ell))
    words = tuple(pattern | bit << target for pattern in patterns for bit in (0, 1))
    expected = tuple(
        word ^ (1 << target) if word & saturation == saturation else word
        for word in words
    )
    assert trace(classical, words) == expected
    return len(words)


@dataclass(frozen=True)
class Checks:
    arithmetic: int
    qrom: int
    clock: int
    comparison: int
    reflection: int


def verify(model: Model, layout: Layout) -> Checks:
    return Checks(
        arithmetic_check(layout),
        qrom_check(model, layout),
        clock_check(layout),
        comparator_check(layout),
        reflection_check(layout),
    )

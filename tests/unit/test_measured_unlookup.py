"""Certify source unlookup composition, allocator reuse and coherent erasure."""

import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path

import pytest
import torch

from flagquantum.compiler.artifacts.reversible_phase import ReversiblePhaseProgram
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.finite_field import ClassicalFieldCoefficient as C
from flagquantum.compiler.registers.measured_unlookup import (
    measured_unlookup_operations,
)
from flagquantum.compiler.registers.recursive_lookup import recursive_lookup_operations
from flagquantum.core.classical_dataflow import ClassicalXor
from tests.unit.measurement_checks import expand_reset_branches, forced_branch

pytestmark = pytest.mark.unit


def _table(mode: int, rows: int, width: int) -> tuple[tuple[bool | C, ...], ...]:
    values = [
        (
            bool((j * 5 + 1) % 3 == 0)
            if mode == 0
            else (
                C(j % 2)
                if mode == 1
                else False if j % 3 == 0 else True if j % 3 == 1 else C(j % 2)
            )
        )
        for j in range(rows * width)
    ]
    return tuple(
        tuple(reversed(values[k * width : (k + 1) * width])) for k in range(rows)
    )


def _index(value: int, ports: tuple[int, ...], width: int) -> int:
    return sum(
        ((value >> bit) & 1) << (width - 1 - port)
        for bit, port in enumerate(reversed(ports))
    )


def _certify(
    n: int,
    w: int,
    rows: int,
    mode: int,
    classical: tuple[int, int],
    permuted: bool,
    round_trip: bool = False,
) -> None:
    h = n // 2
    needed = max(0, (1 << h) - h - w) + n - h
    if round_trip:
        needed = max(needed, n)
    qubits = n + w + 1 + needed
    mapping = tuple(reversed(range(qubits))) if permuted else tuple(range(qubits))
    address = tuple(mapping[:n])
    output = tuple(mapping[n : n + w])
    spectator = mapping[n + w]
    workspace = tuple(mapping[n + w + 1 :])
    table = _table(mode, rows, w)
    phase = tuple(range(2, 2 + (1 << n)))
    result = phase[-1] + 1
    ops = measured_unlookup_operations(
        table,
        address,
        output,
        workspace,
        phase,
        measurement_bit=result,
        reused_phase_bits=(phase[0],) if round_trip and n else (),
    )
    if round_trip:
        ops = (
            *recursive_lookup_operations(
                table, address, output, workspace, measurement_bit=phase[0]
            ),
            *ops,
        )
    program = ReversiblePhaseProgram(qubits, result + 1, ops, input_bits=(0, 1))
    assert decode_kmx(encode_kmx(program)) == program
    generator = torch.Generator().manual_seed(600 + n + mode)
    psi = torch.complex(
        torch.randn((1 << n, 2), generator=generator, dtype=torch.float64),
        torch.randn((1 << n, 2), generator=generator, dtype=torch.float64),
    )
    psi /= psi.norm()
    initial = torch.zeros(1 << qubits, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for a in range(1 << n):
        word = 0
        if a < rows:
            for value in table[a]:
                word = 2 * word + (
                    int(value) if type(value) is bool else classical[value.slot]
                )
        for s in range(2):
            index = _index(a, address, qubits) | _index(s, (spectator,), qubits)
            initial[index | (0 if round_trip else _index(word, output, qubits))] = psi[
                a, s
            ]
            expected[index] = psi[a, s]
    setup = tuple(
        ClassicalXor((), slot, bool(value)).instruction()
        for slot, value in enumerate(classical)
    )
    gates = (
        *setup,
        *expand_reset_branches(
            bind_reversible_phase_execution(program), first_slot=result + 10
        ),
    )
    events = sum(gate.name == "measure" for gate in gates)
    for outcomes in product((0, 1), repeat=events):
        actual = forced_branch(initial, gates, outcomes)
        torch.testing.assert_close(
            actual, expected / 2 ** (events / 2), atol=5e-14, rtol=5e-14
        )


@pytest.mark.parametrize("n", [0, 1, 2])
@pytest.mark.parametrize("mode", range(3))
@pytest.mark.parametrize("classical", list(product((0, 1), repeat=2)))
@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_every_small_complex_branch_clears_output_and_work(
    n: int, mode: int, classical: tuple[int, int], permuted: bool, partial: bool
) -> None:
    _certify(
        n, 2, max(0, (1 << n) - 1) if partial else 1 << n, mode, classical, permuted
    )


@pytest.mark.parametrize("n", [3, 4])
def test_every_branch_with_high_low_split_and_extra_quantum_work(n: int) -> None:
    _certify(n, 1, 1 << n, 2, (1, 0), False)


@pytest.mark.parametrize("mode", range(3))
@pytest.mark.parametrize("n", [0, 1, 2])
def test_full_lookup_then_unlookup_all_branches(mode: int, n: int) -> None:
    _certify(n, 1, 1 << n, mode, (1, 0), False, round_trip=True)


def test_all_complete_source_fingerprints_including_recycled_slots() -> None:
    fixture = json.loads(
        (
            Path(__file__).parents[1] / "fixtures/kickmix_measured_unlookup.json"
        ).read_text()
    )
    assert len(fixture["cases"]) == 783
    for case in fixture["cases"]:
        n, w = case["address_bits"], case["output_bits"]
        h = n // 2
        extra = max(0, (1 << h) - h - w)
        work = tuple(
            range(
                n + w,
                n
                + w
                + (max(n, extra + n - h) if case["repeat"] == 0 else extra + n - h),
            )
        )
        phase = tuple(range(2, 2 + (1 << n)))
        table = _table(case["mode"], case["rows"], w)
        args = (
            table,
            tuple(reversed(range(n))),
            tuple(reversed(range(n, n + w))),
            work,
            phase,
        )
        ops = measured_unlookup_operations(
            *args,
            measurement_bit=2 + (1 << n),
            reused_phase_bits=(phase[0],) if case["repeat"] == 0 and n else (),
        )
        if case["repeat"] == 0:
            ops = (
                *recursive_lookup_operations(
                    table, args[1], args[2], work, measurement_bit=phase[0]
                ),
                *ops,
            )
        if case["repeat"] == 2:
            ops = (
                *ops,
                *measured_unlookup_operations(
                    *args, measurement_bit=2 + (1 << n), reused_phase_bits=phase
                ),
            )
        program = ReversiblePhaseProgram(
            max(1, n + w + len(work)), 3 + (1 << n), ops, input_bits=(0, 1)
        )
        lines = encode_kmx(program).splitlines()[1:]
        assert len(lines) == case["instructions"]
        assert (
            hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
            == case["sha256"]
        )


def test_empty_output_is_source_early_return() -> None:
    assert measured_unlookup_operations((), (), (), (), (), measurement_bit=-1) == ()


@pytest.mark.parametrize(
    "phase,result", [((0,), 0), ((0, 0), 2), ((0,), True), ((0,), -1), ((-1,), 1)]
)
def test_invalid_phase_slot_ownership(phase: tuple[int, ...], result: int) -> None:
    with pytest.raises(ValueError):
        measured_unlookup_operations(
            ((True,),), (), (0,), (), phase, measurement_bit=result
        )


def test_classical_table_inputs_and_quantum_table_rejection() -> None:
    with pytest.raises(ValueError, match="overwrite"):
        measured_unlookup_operations(((C(0),),), (), (0,), (), (0,), measurement_bit=1)
    with pytest.raises(ValueError, match="quantum data"):
        measured_unlookup_operations(((2,),), (), (0,), (), (0,), measurement_bit=1)
    with pytest.raises(ValueError, match="Reused"):
        measured_unlookup_operations(
            ((True,),), (), (0,), (), (0,), measurement_bit=1, reused_phase_bits=(2,)
        )


def test_source_and_expanded_resource_counts_are_separate() -> None:
    ops = measured_unlookup_operations(
        ((C(0), C(0)),) * 4, (0, 1), (2, 3), (4,), (2, 3, 4, 5), measurement_bit=6
    )
    assert Counter(op.opcode for op in ops) == {
        "bit_store0": 4,
        "hmr": 4,
        "push_condition": 2,
        "bit_invert": 8,
        "pop_condition": 2,
        "r": 2,
        "swap": 2,
        "x": 4,
        "cx": 3,
        "cz": 4,
        "z": 1,
        "neg": 1,
    }
    assert len(ops) == 37
    gates = bind_reversible_phase_execution(
        ReversiblePhaseProgram(5, 7, ops, input_bits=(0, 1))
    )
    assert Counter(g.name for g in gates) == {
        "classical_xor": 19,
        "h": 6,
        "measure": 4,
        "x": 10,
        "reset": 2,
        "swap": 2,
        "cx": 3,
        "cz": 4,
        "z": 3,
    }
    assert len(gates) == 53
    assert (
        max(
            int(g.metadata["classical_bit"])
            for g in gates
            if "classical_bit" in g.metadata
        )
        == 7
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 6])
@pytest.mark.parametrize("w", [1, 2])
@pytest.mark.parametrize("mode", range(3))
@pytest.mark.parametrize("partial", [False, True])
def test_native_exact_phases_output_clear_and_input_preservation(
    n: int, w: int, mode: int, partial: bool
) -> None:
    from fractions import Fraction
    from random import Random

    km = pytest.importorskip("kickmix")
    assert hasattr(km, "Simulator")
    rows = (1 << n) // 2 if partial else 1 << n
    table = _table(mode, rows, w)
    h = n // 2
    extra = max(0, (1 << h) - h - w)
    work = tuple(range(n + w, n + w + extra + n - h))
    phase = tuple(range(2, 2 + (1 << n)))
    ops = measured_unlookup_operations(
        table,
        tuple(reversed(range(n))),
        tuple(reversed(range(n, n + w))),
        work,
        phase,
        measurement_bit=phase[-1] + 1,
    )
    program = ReversiblePhaseProgram(
        max(1, n + w + len(work)), phase[-1] + 2, ops, input_bits=(0, 1)
    )
    rng = Random(700 + n)
    addresses = [rng.randrange(1 << n) for _ in range(64)]
    bits = [[rng.randrange(2) for _ in range(64)] for _ in range(2)]
    sim = km.Simulator(batch_size=64, seed=21)
    for q in range(n):
        sim.write_across_shots(
            km.q(q),
            sum(((value >> q) & 1) << shot for shot, value in enumerate(addresses)),
        )
    for bit in range(2):
        sim.write_across_shots(
            km.b(bit), sum(value << shot for shot, value in enumerate(bits[bit]))
        )
    for j in range(w):
        packed = 0
        for shot, address in enumerate(addresses):
            value = False if address >= rows else table[address][w - 1 - j]
            data = value if type(value) is bool else bits[value.slot][shot]
            packed |= int(data) << shot
        sim.write_across_shots(km.q(n + j), packed)
    sim.do(km.Circuit(encode_kmx(program)))
    for q in range(n):
        assert sim.read_across_shots(km.q(q), out=int) == sum(
            ((value >> q) & 1) << shot for shot, value in enumerate(addresses)
        )
    for q in (*range(n, n + w), *work):
        assert sim.read_across_shots(km.q(q), out=int) == 0
    for bit in range(2):
        assert sim.read_across_shots(km.b(bit), out=int) == sum(
            value << shot for shot, value in enumerate(bits[bit])
        )
    assert all(Fraction(sim.read_shot_phase(shot)) % 2 == 0 for shot in range(64))


def test_runnable_compute_then_erase_example() -> None:
    from examples.algorithms.measured_unlookup import main

    main()


@pytest.mark.parametrize("n", range(9))
def test_logical_resource_counts_close_against_tree_shapes(n: int) -> None:
    h = n // 2
    high = n - h
    size = 1 << h
    extra = max(0, size - h - 1)
    workspace = tuple(range(n + 1, n + 1 + extra + high))
    phase = tuple(range(2, 2 + (1 << n)))
    ops = measured_unlookup_operations(
        ((C(0),),) * (1 << n),
        tuple(range(n)),
        (n,),
        workspace,
        phase,
        measurement_bit=phase[-1] + 1,
    )
    counts = Counter(op.opcode for op in ops)
    nodes = (1 << high) - 1
    assert counts["ccx"] == max(0, (1 << high) - 2)
    assert counts["cswap"] == size - h - 1
    assert counts["r"] == nodes + size - h
    assert counts["hmr"] == 1 + nodes + size - h
    assert counts["bit_store0"] == counts["bit_invert"] == 1 << n
    assert len(workspace) == extra + high
    assert {q for op in ops for q in op.qubits}.intersection(workspace) == set(
        workspace
    )


def test_shared_components_can_be_replaced_without_changing_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.compiler.artifacts.reversible_phase import (
        ReversiblePhaseOperation as Op,
    )
    from flagquantum.compiler.registers import measured_unlookup as consumer

    calls: list[tuple[str, object]] = []
    buffer = (5, 4, 2, 3)
    first, second, third = Op("z", (5,)), Op("z", (4,)), Op("z", (2,))

    def encode(target: tuple[int, ...]) -> tuple[Op, ...]:
        calls.append(("encode", target))
        return (first,)

    def lookup(
        table: object,
        address: tuple[int, ...],
        output: tuple[int, ...],
        work: tuple[int, ...],
        *,
        pauli: str,
        measurement_bit: int,
    ) -> tuple[Op, ...]:
        calls.append(("lookup", (table, address, output, work, pauli, measurement_bit)))
        return (second,)

    def erase(target: tuple[int, ...], *, measurement_bit: int) -> tuple[Op, ...]:
        calls.append(("erase", (target, measurement_bit)))
        return (third,)

    monkeypatch.setattr(consumer, "binary_to_unary_operations", encode)
    monkeypatch.setattr(consumer, "recursive_lookup_operations", lookup)
    monkeypatch.setattr(consumer, "unary_to_binary_operations", erase)
    phase = tuple(range(2, 18))
    operations = consumer.measured_unlookup_operations(
        ((True,),) * 16, (0, 1, 2, 3), (4,), (5, 6, 7), phase, measurement_bit=18
    )
    expected_table = tuple(
        tuple(C(slot) for slot in reversed(phase[start : start + 4]))
        for start in range(0, 16, 4)
    )
    assert calls == [
        ("encode", buffer),
        ("lookup", (expected_table, (0, 1), buffer, (6, 7), "z", 18)),
        ("erase", (buffer, 18)),
    ]
    assert operations[-3:] == (first, second, third)

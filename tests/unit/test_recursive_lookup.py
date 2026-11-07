"""Verify locked recursive lookup streams and expose incomplete-table phases."""

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
from flagquantum.compiler.registers.recursive_lookup import (
    TableBit,
    recursive_lookup_operations,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from tests.unit.measurement_checks import expand_reset_branches, forced_branch

pytestmark = pytest.mark.unit


def _table(
    mode: int, rows: int, width: int, data: int
) -> tuple[tuple[TableBit, ...], ...]:
    values: list[TableBit] = []
    for j in range(rows * width):
        if mode == 0:
            value: TableBit = (j * 5 + 1) % 3 == 0
        elif mode == 1:
            value = C(j % 2)
        elif mode == 2:
            value = data + j % 2
        else:
            value = (
                False
                if j % 4 == 0
                else True if j % 4 == 1 else C(j % 2) if j % 4 == 2 else data + j % 2
            )
        values.append(value)
    return tuple(
        tuple(reversed(values[row * width : (row + 1) * width])) for row in range(rows)
    )


def _value(index: int, ports: tuple[int, ...], width: int) -> int:
    result = 0
    for wire in ports:
        result = 2 * result + ((index >> (width - 1 - wire)) & 1)
    return result


def _ideal(
    psi: torch.Tensor,
    table: tuple[tuple[TableBit, ...], ...],
    n: int,
    width: int,
    controlled: bool,
    pauli: str,
    classical: tuple[int, int],
) -> torch.Tensor:
    data_width = psi.numel().bit_length() - 1
    expected = torch.zeros(psi.numel() << n, dtype=torch.complex128)
    for index, amplitude in enumerate(psi):
        enabled = not controlled or _value(index, (n,), data_width)
        row = _value(index, tuple(range(n)), data_width)
        target = index
        phase = 1
        if enabled and row < len(table):
            for entry, wire in zip(
                table[row], range(n + 1, n + 1 + width), strict=True
            ):
                value = (
                    entry
                    if type(entry) is bool
                    else (
                        classical[entry.slot]
                        if isinstance(entry, C)
                        else _value(index, (entry,), data_width)
                    )
                )
                if value:
                    if pauli == "x":
                        target ^= 1 << (data_width - 1 - wire)
                    elif _value(index, (wire,), data_width):
                        phase *= -1
        expected[target << n] = amplitude * phase
    return expected


@pytest.mark.parametrize("n", [0, 1, 2])
@pytest.mark.parametrize("mode", range(4))
@pytest.mark.parametrize("pauli", ["x", "z"])
@pytest.mark.parametrize("controlled", [False, True])
@pytest.mark.parametrize("classical", [(0, 1), (1, 0)])
def test_full_tables_every_complex_branch(
    n: int, mode: int, pauli: str, controlled: bool, classical: tuple[int, int]
) -> None:
    width = 2 if n == 1 and mode == 3 else 1
    data = n + 1 + width
    data_width = data + 3  # two quantum table bits and an entangled spectator
    table = _table(mode, 1 << n, width, data)
    ops = recursive_lookup_operations(
        table,
        tuple(range(n)),
        tuple(range(n + 1, n + 1 + width)),
        tuple(range(data_width, data_width + n)),
        pauli=pauli,
        control=n if controlled else None,
        measurement_bit=2,
    )
    program = ReversiblePhaseProgram(data_width + n, 3, ops, input_bits=(0, 1))
    assert decode_kmx(encode_kmx(program)) == program
    generator = torch.Generator().manual_seed(110 + n + mode)
    psi = torch.complex(
        torch.randn(1 << data_width, generator=generator, dtype=torch.float64),
        torch.randn(1 << data_width, generator=generator, dtype=torch.float64),
    )
    psi /= psi.norm()
    initial = torch.zeros(psi.numel() << n, dtype=torch.complex128)
    initial[:: 1 << n] = psi
    expected = _ideal(psi, table, n, width, controlled, pauli, classical)
    assignments = tuple(
        ClassicalXor((), slot, constant=bool(value)).instruction()
        for slot, value in enumerate(classical)
    )
    gates = (
        *assignments,
        *expand_reset_branches(bind_reversible_phase_execution(program), first_slot=10),
    )
    events = sum(g.name == "measure" for g in gates)
    for outcomes in product((0, 1), repeat=events):
        actual = forced_branch(initial, gates, outcomes)
        torch.testing.assert_close(
            actual, expected / 2 ** (events / 2), atol=4e-14, rtol=4e-14
        )
        if n:
            assert torch.count_nonzero(actual.reshape(-1, 1 << n)[:, 1:]) == 0


@pytest.mark.parametrize("controlled", [False, True])
def test_incomplete_source_subtree_retains_its_branch_phase_defect(
    controlled: bool,
) -> None:
    # The unchanged source computes !address but erases with address after a
    # low-only subtree. Outcome one adds Z(control), or a global minus sign.
    ops = recursive_lookup_operations(
        ((True,),), (0,), (2,), (3,), control=1 if controlled else None
    )
    program = ReversiblePhaseProgram(4, 1, ops)
    gates = expand_reset_branches(
        bind_reversible_phase_execution(program), first_slot=10
    )
    initial = torch.zeros(16, dtype=torch.complex128)
    initial[0] = initial[4] = 2**-0.5
    ideal = torch.zeros_like(initial)
    if controlled:
        ideal[0] = ideal[6] = 2**-0.5
    else:
        ideal[2] = ideal[6] = 2**-0.5
    for reset, erased in product((0, 1), repeat=2):
        expected = ideal.clone()
        if erased:
            if controlled:
                expected[6] *= -1
            else:
                expected *= -1
        actual = forced_branch(initial, gates, (reset, erased))
        torch.testing.assert_close(actual, expected / 2, rtol=2e-14, atol=2e-14)
    if controlled:
        branch_one = forced_branch(initial, gates, (0, 1))
        assert abs(torch.vdot(ideal, branch_one)) < 1e-14


def test_complete_native_fingerprints() -> None:
    fixture = json.loads(
        (
            Path(__file__).parents[1] / "fixtures/kickmix_recursive_lookup.json"
        ).read_text()
    )
    assert len(fixture["cases"]) == 480
    for case in fixture["cases"]:
        n, w = case["address_bits"], case["output_bits"]
        data = n + 1 + w
        table = _table(case["mode"], case["rows"], w, data)
        ops = recursive_lookup_operations(
            table,
            tuple(reversed(range(n))),
            tuple(reversed(range(n + 1, n + 1 + w))),
            tuple(range(data + 2, data + 2 + n)),
            pauli=case["pauli"],
            control=n if case["controlled"] else None,
            measurement_bit=2,
        )
        program = ReversiblePhaseProgram(data + 2 + n, 3, ops, input_bits=(0, 1))
        lines = encode_kmx(program).splitlines()[1:]
        assert len(lines) == case["instructions"]
        assert (
            hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
            == case["sha256"]
        )


def test_resource_counts_and_unowned_work() -> None:
    ops = recursive_lookup_operations(
        ((True,),) * 4, (0, 1), (2,), (3, 4, 5), control=6
    )
    counts = Counter(op.opcode for op in ops)
    assert counts == {"x": 6, "r": 3, "ccx": 3, "cx": 7, "hmr": 3, "cz": 3}
    assert 5 not in {q for op in ops for q in op.qubits}
    assert {op.result_bit for op in ops if op.opcode == "hmr"} == {0}
    emitted = bind_reversible_phase_execution(ReversiblePhaseProgram(7, 1, ops))
    assert Counter(gate.name for gate in emitted) == {
        "classical_xor": 1,
        "x": 9,
        "h": 6,
        "reset": 3,
        "ccx": 3,
        "cx": 7,
        "measure": 3,
        "cz": 3,
    }
    assert len(emitted) == 35


def test_source_empty_output_early_return() -> None:
    assert recursive_lookup_operations((), (), (), ()) == ()


@pytest.mark.parametrize(
    "table,address,output,work",
    [
        ((), (0,), (1,), (2,)),
        (((True, False),), (0,), (1,), (2,)),
        (((True,),) * 3, (0,), (1,), (2,)),
        (((True,),), (0,), (1,), ()),
        (((True,),), (0,), (1,), (0,)),
        (((1,),), (0,), (1,), (2,)),
        (((True,),), tuple(range(31)), (31,), tuple(range(32, 63))),
    ],
)
def test_invalid_layouts(
    table: tuple[tuple[TableBit, ...], ...],
    address: tuple[int, ...],
    output: tuple[int, ...],
    work: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        recursive_lookup_operations(table, address, output, work)


def test_result_cannot_alias_classical_table_input() -> None:
    with pytest.raises(ValueError, match="overwrite"):
        recursive_lookup_operations(((C(0),),), (0,), (1,), (2,))


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("mode", range(4))
@pytest.mark.parametrize("pauli", ["x", "z"])
@pytest.mark.parametrize("controlled", [False, True])
def test_native_simulator_full_table_outputs_and_exact_phases(
    mode: int, pauli: str, controlled: bool
) -> None:
    from fractions import Fraction
    from random import Random

    km = pytest.importorskip("kickmix")
    assert hasattr(km, "Simulator")
    n, width = 2, 2
    data = n + 1 + width
    table = _table(mode, 4, width, data)
    program = ReversiblePhaseProgram(
        data + 4,
        3,
        recursive_lookup_operations(
            table,
            (0, 1),
            (3, 4),
            (data + 2, data + 3),
            pauli=pauli,
            control=2 if controlled else None,
            measurement_bit=2,
        ),
        input_bits=(0, 1),
    )
    rng = Random(111)
    quantum = [[rng.randrange(2) for _ in range(64)] for _ in range(data + 2)]
    classical = [[rng.randrange(2) for _ in range(64)] for _ in range(2)]
    sim = km.Simulator(batch_size=64, seed=15)
    for wire, values in enumerate(quantum):
        sim.write_across_shots(
            km.q(wire), sum(value << shot for shot, value in enumerate(values))
        )
    for bit, values in enumerate(classical):
        sim.write_across_shots(
            km.b(bit), sum(value << shot for shot, value in enumerate(values))
        )
    sim.do(km.Circuit(encode_kmx(program)))
    outputs = [sim.read_across_shots(km.q(q), out=int) for q in range(data + 4)]
    for shot in range(64):
        row = 2 * quantum[0][shot] + quantum[1][shot]
        expected = [values[shot] for values in quantum] + [0, 0]
        phase = 0
        if not controlled or quantum[2][shot]:
            for entry, wire in zip(table[row], (3, 4), strict=True):
                value = (
                    entry
                    if type(entry) is bool
                    else (
                        classical[entry.slot][shot]
                        if isinstance(entry, C)
                        else quantum[entry][shot]
                    )
                )
                if value:
                    if pauli == "x":
                        expected[wire] ^= 1
                    else:
                        phase ^= quantum[wire][shot]
        assert [(column >> shot) & 1 for column in outputs] == expected
        assert Fraction(sim.read_shot_phase(shot)) % 2 == phase
    for bit in range(2):
        assert sim.read_across_shots(km.b(bit), out=int) == sum(
            value << shot for shot, value in enumerate(classical[bit])
        )


@pytest.mark.ftqc_reference
def test_native_simulator_independently_exposes_low_only_control_phase() -> None:
    from fractions import Fraction

    km = pytest.importorskip("kickmix")
    assert hasattr(km, "Simulator")
    # This all-false one-row table has the identity ideal lookup map.
    program = ReversiblePhaseProgram(
        4, 1, recursive_lookup_operations(((False,),), (0,), (2,), (3,), control=1)
    )
    sim = km.Simulator(batch_size=64, seed=15)
    controls = [shot % 2 for shot in range(64)]
    sim.write_across_shots(
        km.q(1), sum(value << shot for shot, value in enumerate(controls))
    )
    sim.do(km.Circuit(encode_kmx(program)))
    packed = sim.read_across_shots(km.b(0), out=int)
    results = [(packed >> shot) & 1 for shot in range(64)]
    phases = [Fraction(sim.read_shot_phase(shot)) % 2 for shot in range(64)]
    assert phases == [
        control * result for control, result in zip(controls, results, strict=True)
    ]
    assert any(phases)


@pytest.mark.parametrize("pauli", ["x", "z"])
def test_user_append_example(pauli: str) -> None:
    from examples.algorithms.recursive_lookup import verify_lookup

    verify_lookup(pauli)


@pytest.mark.parametrize("pauli", ["y", "X", ""])
def test_invalid_pauli(pauli: str) -> None:
    with pytest.raises(ValueError, match="pauli"):
        recursive_lookup_operations(((True,),), (), (0,), (), pauli=pauli)


@pytest.mark.parametrize("slot", [-1, True, 0.5])
def test_invalid_measurement_slot(slot: int) -> None:
    with pytest.raises(ValueError, match="result slot"):
        recursive_lookup_operations(((True,),), (), (0,), (), measurement_bit=slot)


def test_invalid_table_value_and_coherent_control() -> None:
    with pytest.raises(ValueError, match="entries"):
        recursive_lookup_operations(((0.5,),), (), (0,), ())
    with pytest.raises(ValueError, match="integer wires"):
        recursive_lookup_operations(((True,),), (), (0,), (), control=True)

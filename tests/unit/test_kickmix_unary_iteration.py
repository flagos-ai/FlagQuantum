"""Certify source traversal, coherent erasure branches and explicit ownership."""

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.artifacts.reversible_phase import (
    ReversiblePhaseOperation as Op,
)
from flagquantum.compiler.artifacts.reversible_phase import (
    ReversiblePhaseProgram,
)
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.compiler.registers.kickmix_unary_iteration import (
    UnaryAddressCursor,
    unary_iteration_operations,
)
from tests.unit.measurement_checks import expand_reset_branches, forced_branch

pytestmark = pytest.mark.unit


def _program(
    n: int, controlled: bool, values: tuple[int, ...]
) -> ReversiblePhaseProgram:
    return ReversiblePhaseProgram(
        n + 3 + max(1, n),
        1,
        unary_iteration_operations(
            tuple(reversed(range(n))),
            tuple(range(n + 3, n + 3 + max(1, n))),
            values,
            lambda value, match: (Op("cx", (match, n + 1)),),
            control=n if controlled else None,
        ),
    )


@pytest.mark.parametrize("n", [0, 1, 2])
@pytest.mark.parametrize("park", [False, True])
@pytest.mark.parametrize("controlled", [False, True])
def test_all_reset_and_erasure_branches_preserve_complex_amplitudes(
    n: int, controlled: bool, park: bool
) -> None:
    values = (
        ((1 << n) - 1, 0, 0, 1 << n, (1 << n) + 1, 0) if park else (0, (1 << n) - 1)
    )
    program = _program(n, controlled, values)
    assert decode_kmx(encode_kmx(program)) == program
    gates = expand_reset_branches(
        bind_reversible_phase_execution(program), first_slot=10
    )
    work = max(1, n)
    data_width = n + 3
    generator = torch.Generator().manual_seed(910 + n)
    psi = torch.complex(
        torch.randn(1 << data_width, generator=generator, dtype=torch.float64),
        torch.randn(1 << data_width, generator=generator, dtype=torch.float64),
    )
    psi /= psi.norm()
    state = torch.zeros(1 << program.n_qubits, dtype=torch.complex128)
    state[:: 1 << work] = psi
    expected = torch.zeros_like(state)
    for index, amplitude in enumerate(psi):
        address = sum(
            ((index >> (data_width - 1 - bit)) & 1) << bit for bit in range(n)
        )
        enabled = not controlled or bool(index & (1 << (data_width - 1 - n)))
        flip = enabled and bool(sum(value == address for value in values) % 2)
        target = index ^ ((1 << (data_width - 2 - n)) if flip else 0)
        expected[target << work] = amplitude
    events = sum(g.name == "measure" for g in gates)
    for outcomes in product((0, 1), repeat=events):
        actual = forced_branch(state, gates, outcomes)
        torch.testing.assert_close(
            actual, expected / (2 ** (events / 2)), atol=2e-14, rtol=2e-14
        )
        assert torch.count_nonzero(actual.reshape(-1, 1 << work)[:, 1:]) == 0


def test_jump_duplicates_parking_and_explicit_close() -> None:
    cursor = UnaryAddressCursor((1, 0), (4, 5, 6), control=2, measurement_bit=7)
    assert cursor.workspace == (4, 5)
    assert cursor.current == cursor.address_space_size == 4
    assert cursor.match_qubit == 4
    assert cursor.move_to(10) == cursor.close() == ()
    assert cursor.move_to(0)
    assert cursor.move_to(0) == ()
    forward = cursor.move_to(1)
    backward = cursor.move_to(0)
    assert forward == backward == (Op("cx", (5, 4)),)
    erased = cursor.close()
    assert [op.result_bit for op in erased if op.opcode == "hmr"] == [7, 7]
    assert cursor.close() == ()


def test_empty_visits_do_not_allocate_and_parked_callbacks_still_run() -> None:
    assert unary_iteration_operations((), (), (), lambda v, m: ()) == ()
    seen = []

    def body(value: int, match: int) -> tuple[Op, ...]:
        seen.append((value, match))
        return (Op("cx", (match, 2)),)

    operations = unary_iteration_operations((0,), (1,), (2, 9), body)
    assert seen == [(2, 1), (9, 1)]
    assert operations == (Op("cx", (1, 2)),) * 2


def test_source_counts_are_separate_from_expanded_ir() -> None:
    program = _program(2, True, (0, 1, 2, 3))
    assert Counter(op.opcode for op in program.operations) == {
        "x": 6,
        "r": 3,
        "ccx": 3,
        "cx": 7,
        "hmr": 3,
        "cz": 3,
    }
    gates = bind_reversible_phase_execution(program)
    assert Counter(g.name for g in gates) == {
        "classical_xor": 1,
        "x": 9,
        "h": 6,
        "reset": 3,
        "ccx": 3,
        "cx": 7,
        "measure": 3,
        "cz": 3,
    }
    assert len(program.operations) == 25


@pytest.mark.parametrize(
    "address,work,control,slot",
    [
        ((0,), (), None, 0),
        ((0, 0), (1, 2), None, 0),
        ((0,), (0,), None, 0),
        ((0,), (1,), 1, 0),
        ((0,), (1,), True, 0),
        ((0,), (1,), None, -1),
        (tuple(range(64)), tuple(range(64, 128)), None, 0),
    ],
)
def test_invalid_ownership(
    address: tuple[int, ...], work: tuple[int, ...], control: int | None, slot: int
) -> None:
    with pytest.raises(ValueError):
        UnaryAddressCursor(address, work, control=control, measurement_bit=slot)


@pytest.mark.parametrize("value", [-1, 1 << 64, True, 0.5])
def test_invalid_visits_leave_cursor_parked(value: int) -> None:
    cursor = UnaryAddressCursor((0,), (1,))
    with pytest.raises(ValueError):
        cursor.move_to(value)
    assert cursor.current == 2


def test_append_example() -> None:
    from examples.algorithms.unary_address_traversal import main

    main()


def test_locked_native_stream_fingerprints() -> None:
    import hashlib
    import json
    from pathlib import Path

    fixture = json.loads(
        (
            Path(__file__).parents[1] / "fixtures/kickmix_unary_iteration.json"
        ).read_text()
    )
    assert fixture["source_blob"] == "ec97900dbce6ac31ca66a33eb63e95ea7073fb1d"
    assert len(fixture["cases"]) == 90
    for case in fixture["cases"]:
        n = case["width"]
        program = ReversiblePhaseProgram(
            n + 2 + max(1, n),
            1,
            unary_iteration_operations(
                tuple(reversed(range(n))),
                tuple(range(n + 2, n + 2 + max(1, n))),
                tuple(case["values"]),
                lambda value, match, n=n: (Op("cx", (match, n + 1)),),
                control=n if case["controlled"] else None,
            ),
        )
        lines = encode_kmx(program).splitlines()[1:]
        assert len(lines) == case["instructions"]
        assert (
            hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
            == case["sha256"]
        )

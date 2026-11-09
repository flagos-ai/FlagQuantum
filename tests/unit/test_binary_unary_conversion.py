"""Prove source binary/unary streams and coherent measured channels."""

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
from flagquantum.compiler.registers.bits.binary_unary_conversion import (
    binary_to_unary_operations,
    unary_to_binary_operations,
)
from tests.unit.measurement_checks import expand_reset_branches, forced_branch

pytestmark = pytest.mark.unit


def _target(bits: int, permuted: bool) -> tuple[int, ...]:
    size = 1 << bits
    return tuple(reversed(range(size))) if permuted else tuple(range(size))


def _index(value: int, ports: tuple[int, ...], width: int) -> int:
    return sum(
        ((value >> bit) & 1) << (width - 1 - wire)
        for bit, wire in enumerate(reversed(ports))
    )


def _states(bits: int, ports: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    size = len(ports)
    width = size + 1
    generator = torch.Generator().manual_seed(303 + bits)
    psi = torch.complex(
        torch.randn((1 << bits, 2), generator=generator, dtype=torch.float64),
        torch.randn((1 << bits, 2), generator=generator, dtype=torch.float64),
    )
    psi /= psi.norm()
    binary = torch.zeros(1 << width, dtype=torch.complex128)
    unary = torch.zeros_like(binary)
    for value in range(1 << bits):
        for spectator in range(2):
            binary[_index(value, ports[-bits:] if bits else (), width) | spectator] = (
                psi[value, spectator]
            )
            unary[_index(1 << value, ports, width) | spectator] = psi[value, spectator]
    return binary, unary


@pytest.mark.parametrize("bits", range(4))
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("permuted", [False, True])
def test_every_complex_reset_and_erasure_branch(
    bits: int, inverse: bool, permuted: bool
) -> None:
    ports = _target(bits, permuted)
    binary, unary = _states(bits, ports)
    operations = (
        unary_to_binary_operations(ports)
        if inverse
        else binary_to_unary_operations(ports)
    )
    program = ReversiblePhaseProgram(len(ports) + 1, int(inverse), operations)
    assert decode_kmx(encode_kmx(program)) == program
    gates = expand_reset_branches(
        bind_reversible_phase_execution(program), first_slot=10
    )
    events = sum(g.name == "measure" for g in gates)
    assert events == len(ports) - bits
    initial, expected = (unary, binary) if inverse else (binary, unary)
    for outcomes in product((0, 1), repeat=events):
        actual = forced_branch(initial, gates, outcomes)
        torch.testing.assert_close(
            actual, expected / 2 ** (events / 2), rtol=3e-14, atol=3e-14
        )
    if inverse:
        active = set(ports[-bits:]) if bits else set()
        allowed = torch.tensor(
            [
                not any(
                    (index >> (program.n_qubits - 1 - q)) & 1
                    for q in ports
                    if q not in active
                )
                for index in range(initial.numel())
            ]
        )
        assert torch.count_nonzero(expected[~allowed]) == 0


@pytest.mark.parametrize("bits", range(3))
def test_full_round_trip_all_branches(bits: int) -> None:
    ports = _target(bits, False)
    initial, _ = _states(bits, ports)
    operations = (
        *binary_to_unary_operations(ports),
        *unary_to_binary_operations(ports),
    )
    program = ReversiblePhaseProgram(len(ports) + 1, 1, operations)
    gates = expand_reset_branches(
        bind_reversible_phase_execution(program), first_slot=10
    )
    events = sum(g.name == "measure" for g in gates)
    for outcomes in product((0, 1), repeat=events):
        actual = forced_branch(initial, gates, outcomes)
        torch.testing.assert_close(
            actual, initial / 2 ** (events / 2), rtol=3e-14, atol=3e-14
        )


def test_locked_complete_native_streams() -> None:
    fixture = json.loads(
        (Path(__file__).parents[1] / "fixtures/kickmix_binary_unary.json").read_text()
    )
    assert len(fixture["cases"]) == 56
    for case in fixture["cases"]:
        size = 1 << case["binary_bits"]
        ports = tuple(
            reversed(tuple(case["offset"] + case["stride"] * k for k in range(size)))
        )
        operations = (
            unary_to_binary_operations(ports)
            if case["inverse"]
            else binary_to_unary_operations(ports)
        )
        program = ReversiblePhaseProgram(
            case["offset"] + case["stride"] * size, int(case["inverse"]), operations
        )
        lines = encode_kmx(program).splitlines()[1:]
        assert len(lines) == case["instructions"]
        assert (
            hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
            == case["sha256"]
        )


def test_resource_counts_keep_measurement_model_separate() -> None:
    ports = tuple(range(8))
    forward = binary_to_unary_operations(ports)
    inverse = unary_to_binary_operations(ports)
    assert Counter(op.opcode for op in forward) == {
        "r": 5,
        "swap": 3,
        "x": 1,
        "cswap": 4,
        "cx": 7,
    }
    assert Counter(op.opcode for op in inverse) == {
        "swap": 3,
        "cx": 11,
        "hmr": 5,
        "cz": 4,
        "neg": 1,
    }
    assert {op.result_bit for op in inverse if op.opcode == "hmr"} == {0}
    program = ReversiblePhaseProgram(8, 1, inverse)
    emitted = bind_reversible_phase_execution(program)
    assert sum(g.name == "measure" for g in emitted) == 5
    assert all(q < 8 for g in emitted for q in g.wires)
    assert not any(g.name in {"t", "tdg"} for g in emitted)


@pytest.mark.parametrize("target", [(), (0, 1, 2), (0, 0), (-1,), (True,)])
def test_invalid_target(target: tuple[int, ...]) -> None:
    for implementation in (binary_to_unary_operations, unary_to_binary_operations):
        with pytest.raises(ValueError):
            implementation(target)


@pytest.mark.parametrize("slot", [-1, True, 0.5])
def test_invalid_result_slot(slot: int) -> None:
    with pytest.raises(ValueError):
        unary_to_binary_operations((0,), measurement_bit=slot)


def test_runnable_append_example() -> None:
    from examples.algorithms.binary_unary_conversion import main

    main()

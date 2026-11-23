"""Replay locked builder result-slot reuse through KMX and dynamic IR."""

from itertools import product
from pathlib import Path

import pytest
import torch

from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.formats.kmx_builder import KmxCircuitBuilder
from flagquantum.compiler.lowering.reversible_phase_execution import (
    bind_reversible_phase_execution,
)
from flagquantum.core.classical_dataflow import ClassicalXor, parse_classical_xor
from tests.unit.measurement_checks import forced_branch

pytestmark = pytest.mark.unit
_HEADER = 'INFO_TAG "@flagquantum:reversible-phase-v1 wires=3 bits=1"\n'
_LEASES = "BIT_STORE1 b0\nX q0 if b0\nBIT_STORE0 b0\nX q1 if b0\n"
_HMR = (
    "HMR q0 b0\n"
    "PUSH_CONDITION if b0\n"
    "Z q1\n"
    "POP_CONDITION\n"
    "HMR q1 b0\n"
    "NEG if b0\n"
)


@pytest.mark.ftqc_reference
def test_source_clean_and_dirty_reuse_replays_without_expanding_constant_store() -> (
    None
):
    km = pytest.importorskip("kickmix")
    assert km.__version__ == "0.1" and hasattr(km, "Simulator")
    assert list(Path(km.__file__).parent.glob("*.so"))
    program = decode_kmx(_HEADER + _LEASES)
    encoded = encode_kmx(program)
    assert encoded == _HEADER + _LEASES
    assert str(km.Circuit(encoded)).strip() == encoded.strip()
    sim = km.Simulator(batch_size=1, seed=1)
    sim.do(km.Circuit(encoded))
    assert sim.read_across_shots(km.q(0), out=int) == 1
    assert sim.read_across_shots(km.q(1), out=int) == 0
    assert sim.read_across_shots(km.b(0), out=int) == 0
    gates = bind_reversible_phase_execution(program, classical_offset=4)
    assert [gate.name for gate in gates if gate.wires] == ["x", "x"]
    assignments = [parse_classical_xor(gate) for gate in gates if not gate.wires]
    assert [assignment.target for assignment in assignments] == [4, 4, 4]
    assert [assignment.constant for assignment in assignments] == [False, True, False]
    initial = torch.zeros(8, dtype=torch.complex128)
    initial[0] = 1
    expected = torch.zeros_like(initial)
    expected[4] = 1
    torch.testing.assert_close(forced_branch(initial, gates, ()), expected)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("construct", [False, True])
def test_recycled_hmr_slot_preserves_all_complex_measurement_branches(
    construct: bool,
) -> None:
    km = pytest.importorskip("kickmix")
    body = _HMR
    if construct:
        builder = KmxCircuitBuilder()
        with builder.measure_result(0, condition=True):
            builder.append_kmx_line("Z q1")
        with builder.measure_result(1) as result:
            builder.append_kmx_line(f"NEG if b{result.bit.value}")
        body = builder.freeze().to_kmx() + "\n"
        assert body == _HMR
    program = decode_kmx(_HEADER + body)
    encoded = encode_kmx(program)
    assert encoded == _HEADER + _HMR
    assert str(km.Circuit(encoded)).strip() == encoded.strip()
    gates = bind_reversible_phase_execution(program, classical_offset=4)
    assert [
        gate.metadata["classical_bit"] for gate in gates if gate.name == "measure"
    ] == [4, 4]
    snapshots = [
        parse_classical_xor(gate).target
        for gate in gates
        if gate.name == "classical_xor" and parse_classical_xor(gate).sources
    ]
    assert snapshots == [5]

    index = torch.arange(8)
    initial = (index + 1).to(torch.complex128) + 1j * (8 - index)
    initial /= initial.norm()
    weights = initial.reshape(2, 2, 2)
    norms = 0.0
    for first, second in product((0, 1), repeat=2):
        expected = torch.zeros_like(initial)
        for a, b in product((0, 1), repeat=2):
            sign = (-1) ** (first * a + (first ^ second) * b + second)
            expected[:2] += weights[a, b] * sign / 2
        branch = forced_branch(initial, gates, (first, second))
        torch.testing.assert_close(branch, expected, rtol=0, atol=2e-15)
        norms += branch.norm().item() ** 2
    assert norms == pytest.approx(1.0)


@pytest.mark.parametrize(
    "payload,reason",
    [
        ("BIT_STORE1 b3", "Invalid classical result bit"),
        (
            "BIT_STORE0 b0\nBIT_STORE1 b1 if b0",
            "Conditional classical initialization",
        ),
        (
            "BIT_STORE0 b0\nPUSH_CONDITION if b0\nBIT_STORE1 b1\nPOP_CONDITION",
            "cannot initialize",
        ),
        ("X q0 if b0", "read before definition"),
    ],
)
def test_initialization_keeps_existing_definition_rules(
    payload: str, reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        decode_kmx(
            'INFO_TAG "@flagquantum:reversible-phase-v1 wires=3 bits=3"\n' + payload
        )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("clear", [False, True])
def test_guarded_stores_overwrite_only_defined_destinations(clear: bool) -> None:
    km = pytest.importorskip("kickmix")
    assert km.__version__ == "0.1" and hasattr(km, "Simulator")
    assert list(Path(km.__file__).parent.glob("*.so"))
    header = 'INFO_TAG "@flagquantum:reversible-phase-v1 wires=1 bits=3 inputs=0,1"\n'
    if clear:
        # The inline guard aliases its destination. The enclosing guard
        # snapshots b0 before the next store overwrites it.
        payload = (
            "BIT_STORE1 b2\n"
            "PUSH_CONDITION if b0\n"
            "BIT_STORE0 b0\n"
            "BIT_STORE0 b2 if b2\n"
            "POP_CONDITION\n"
            "X q0 if b2\n"
        )
    else:
        # Locked CircuitBuilder::bit_store_and plus a read of the result.
        payload = (
            "BIT_STORE0 b2\n"
            "PUSH_CONDITION if b0\n"
            "BIT_STORE1 b2 if b1\n"
            "POP_CONDITION\n"
            "X q0 if b2\n"
        )
    program = decode_kmx(header + payload)
    assert encode_kmx(program) == header + payload
    gates = bind_reversible_phase_execution(program, classical_offset=2)
    initial = torch.tensor([1, 2j], dtype=torch.complex128)
    initial /= initial.norm()
    for first, second in product((0, 1), repeat=2):
        expected = 1 ^ first if clear else first & second
        sim = km.Simulator(batch_size=1, seed=1)
        sim.write_across_shots(km.b(0), first)
        sim.write_across_shots(km.b(1), second)
        sim.do(km.Circuit(payload))
        assert sim.read_across_shots(km.b(2), out=int) == expected
        assert sim.read_across_shots(km.q(0), out=int) == expected
        assert sim.read_shot_phase(0) == 0

        setup = (
            ClassicalXor((), 2, bool(first)).instruction(),
            ClassicalXor((), 3, bool(second)).instruction(),
        )
        state = forced_branch(initial, (*setup, *gates), ())
        torch.testing.assert_close(state, initial.flip((0,)) if expected else initial)

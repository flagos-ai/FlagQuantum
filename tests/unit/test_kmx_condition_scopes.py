"""Check KMX condition scopes against the pinned native executor."""

from itertools import product

import pytest

from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.measurement import measure_x_clear_ir
from flagquantum.core.ir import CircuitIR, Instruction

pytestmark = pytest.mark.unit
_HEADER = 'INFO_TAG "@flagquantum:reversible-phase-v1 wires=2 bits=3"\n'


@pytest.mark.parametrize(
    "payload,reason",
    [
        ("PUSH_CONDITION if b0", "defined classical bit"),
        ("PUSH_CONDITION", "defined classical bit"),
        ("POP_CONDITION", "condition pop"),
        ("BIT_STORE0 b0\nPUSH_CONDITION if b0", "Unclosed"),
        ("BIT_STORE0 b0\nPUSH_CONDITION if b0\nPOP_CONDITION if b0", "condition pop"),
        (
            "BIT_STORE0 b0\nPUSH_CONDITION if b0\nHMR q0 b1\nPOP_CONDITION\nX q1 if b1",
            "read before definition",
        ),
        ("BIT_STORE0 b0\nHMR q0 b1 if b0", "requires a defined result"),
        ("R q0 if b0", "read before definition"),
        (
            "BIT_STORE0 b0\nPUSH_CONDITION if b0\nBIT_STORE0 b1\nPOP_CONDITION",
            "cannot initialize",
        ),
    ],
)
def test_decoder_rejects_invalid_condition_scopes(payload: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        decode_kmx(_HEADER + payload)


@pytest.mark.parametrize(
    "metadata",
    [
        {"condition_clauses": (((0, 1),), ((1, 1),))},
    ],
)
def test_disjunctive_quantum_predicates_lower_to_one_gate_mask(metadata: dict) -> None:
    ir = CircuitIR(
        2,
        (
            *measure_x_clear_ir(0, 0),
            *measure_x_clear_ir(0, 1),
            Instruction("x", (1,), metadata=metadata),
        ),
    )
    program = lower_reversible_phase(ir)
    assert program.n_bits == 4
    assert [op.opcode for op in program.operations if op.qubits] == ["hmr", "hmr", "x"]


@pytest.mark.ftqc_reference
def test_nested_conjunction_round_trips_and_matches_native_truth_table() -> None:
    km = pytest.importorskip("kickmix")
    assert km.__version__ in {"0.1", "0.1.dev0"} and hasattr(km, "Simulator")
    ir = CircuitIR(
        2,
        (
            *(gate for bit in range(3) for gate in measure_x_clear_ir(0, bit)),
            Instruction("x", (0,), metadata={"conditions": ((0, 1), (1, 1), (2, 1))}),
            Instruction("x", (1,), metadata={"conditions": ((2, 1),)}),
        ),
    )
    lowered = lower_reversible_phase(ir)
    text = encode_kmx(lowered)
    assert decode_kmx(text) == lowered
    expected = [
        "PUSH_CONDITION if b0",
        "PUSH_CONDITION if b1",
        "X q0 if b2",
        "POP_CONDITION",
        "POP_CONDITION",
        "X q1 if b2",
    ]
    assert text.splitlines()[4:] == expected
    assert lowered.n_bits == 3
    for inputs in product((0, 1), repeat=3):
        setup = [f"BIT_STORE0 b{bit}" for bit in range(3)]
        setup.extend(f"BIT_INVERT b{bit}" for bit, value in enumerate(inputs) if value)
        sim = km.Simulator(batch_size=1, seed=1)
        sim.do(km.Circuit("\n".join((*setup, *expected))))
        assert sim.read_across_shots(km.q(0), out=int) == int(all(inputs))
        assert sim.read_across_shots(km.q(1), out=int) == inputs[2]
        assert (
            tuple(sim.read_across_shots(km.b(i), out=int) for i in range(3)) == inputs
        )


def test_guarded_effect_definitions_follow_nested_scope_lifetimes() -> None:
    payload = (
        "BIT_STORE0 b0\n"
        "PUSH_CONDITION if b0\n"
        "HMR q0 b1\n"
        "PUSH_CONDITION if b1\n"
        "HMR q1 b2\n"
        "X q0 if b2\n"
        "R q1\n"
        "POP_CONDITION\n"
        "X q1 if b1\n"
        "POP_CONDITION\n"
    )
    assert encode_kmx(decode_kmx(_HEADER + payload)) == _HEADER + payload
    escaped = payload.replace("X q1 if b1", "X q1 if b2")
    with pytest.raises(ValueError, match="read before definition"):
        decode_kmx(_HEADER + escaped)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("inline", [False, True])
def test_guarded_hmr_preserves_disabled_wires_and_old_results(inline: bool) -> None:
    km = pytest.importorskip("kickmix")
    assert hasattr(km, "Simulator")
    initialize = "BIT_STORE0 b0\nBIT_STORE0 b1\nBIT_STORE0 b2\n"
    if inline:
        body = "HMR q0 b2 if b0\nR q1 if b0\n"
    else:
        body = (
            "PUSH_CONDITION if b0\nPUSH_CONDITION if b1\n"
            "HMR q0 b2\nR q1\nPOP_CONDITION\nPOP_CONDITION\n"
        )
    encoded = encode_kmx(decode_kmx(_HEADER + initialize + body))
    assert encoded == _HEADER + initialize + body
    shots = 64
    full = (1 << shots) - 1
    first = sum(1 << shot for shot in range(shots) if shot % 2 == 0)
    second = sum(1 << shot for shot in range(shots) if shot % 4 < 2)
    active = first if inline else first & second
    inactive = full ^ active
    sim = km.Simulator(batch_size=shots, seed=41)
    for bit, value in enumerate((first, second, full)):
        sim.write_across_shots(km.b(bit), value)
    for wire in (0, 1):
        sim.write_across_shots(km.q(wire), full)
    sim.do(km.Circuit(body))
    assert all(
        sim.read_across_shots(km.q(wire), out=int) == inactive for wire in (0, 1)
    )
    outcome = sim.read_across_shots(km.b(2), out=int)
    assert outcome & inactive == inactive
    assert 0 < (outcome & active).bit_count() < active.bit_count()
    assert all(sim.read_shot_phase(i) == 0 for i in range(shots) if (inactive >> i) & 1)


@pytest.mark.ftqc_reference
def test_hmr_overwriting_guard_keeps_scope_snapshot() -> None:
    km = pytest.importorskip("kickmix")
    assert hasattr(km, "Simulator")
    setup = "BIT_STORE0 b0\nBIT_INVERT b0\n"
    body = (
        "PUSH_CONDITION if b0\n"
        "HMR q0 b0\n"
        "X q1\n"
        "PUSH_CONDITION if b0\n"
        "X q0\n"
        "POP_CONDITION\n"
        "POP_CONDITION\n"
    )
    text = _HEADER + setup + body
    encoded = encode_kmx(decode_kmx(text))
    assert encoded == text
    sim = km.Simulator(batch_size=64, seed=41)
    sim.do(km.Circuit(encoded))
    outcome = sim.read_across_shots(km.b(0), out=int)
    full = (1 << 64) - 1
    assert 0 < outcome < full
    assert sim.read_across_shots(km.q(1), out=int) == full
    assert sim.read_across_shots(km.q(0), out=int) == outcome

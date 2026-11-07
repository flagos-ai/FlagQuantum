"""Verify exact scalar and three-wire phase lowering to the locked target."""

from fractions import Fraction

import pytest

from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.global_phase import ccz_ir, scalar_sign_ir
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.errors import CapabilityError

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("condition", [None, 0])
def test_exact_phase_stream_round_trips_and_runs_natively(
    condition: int | None,
) -> None:
    km = pytest.importorskip("kickmix")
    assert km.__version__ in {"0.1", "0.1.dev0"} and hasattr(km, "Simulator")
    for value in (0, 1):
        prefix = [ClassicalXor((), 0).instruction()]
        program = lower_reversible_phase(
            CircuitIR(
                3,
                (
                    *prefix,
                    *scalar_sign_ir(
                        1, conditions=() if condition is None else ((0, 1),)
                    ),
                    *ccz_ir((0, 1, 2)),
                ),
            )
        )
        expected = [
            "BIT_STORE0 b0",
            "NEG" if condition is None else "NEG if b0",
            "CCZ q0 q1 q2",
        ]
        text = encode_kmx(program)
        assert text.splitlines()[1:] == expected
        assert decode_kmx(text) == program
        if value:
            expected.insert(1, "BIT_INVERT b0")
        for basis in (0, 7):
            setup = [f"X q{q}" for q in range(3) if basis >> q & 1]
            sim = km.Simulator(batch_size=1, seed=1)
            sim.do(km.Circuit("\n".join((*setup, *expected))))
            negated = (condition is None or bool(value)) ^ (basis == 7)
            assert sim.read_shot_phase(0) == Fraction(int(negated))
            assert tuple(
                sim.read_across_shots(km.q(q), out=int) for q in range(3)
            ) == tuple((basis >> q) & 1 for q in range(3))


def test_scalar_pattern_requires_identical_wires_and_predicates() -> None:
    sign = scalar_sign_ir(0)
    different_wire = (sign[0], Instruction("z", (1,)), *sign[2:])
    prefix = ClassicalXor((), 0).instruction()
    different_condition = (
        prefix,
        *sign[:3],
        Instruction("z", (0,), metadata={"conditions": ((0, 1),)}),
    )
    for gates in (different_wire, different_condition):
        assert all(
            op.opcode != "neg"
            for op in lower_reversible_phase(CircuitIR(2, gates)).operations
        )


def test_mismatched_ccz_conjugation_does_not_accept_standalone_h() -> None:
    before, controlled, _ = ccz_ir((0, 1, 2))
    with pytest.raises(CapabilityError, match="cannot represent 'h'"):
        lower_reversible_phase(
            CircuitIR(3, (before, controlled, Instruction("h", (1,))))
        )

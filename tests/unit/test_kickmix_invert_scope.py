"""Mixed physical inversion and refusal of an unbalanced source scope."""

from itertools import product
from pathlib import Path

import pytest
import torch

from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient,
)
from flagquantum.compiler.registers.primitives._mixed_gates import _invert_operand
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR, Instruction
from tests.unit.measurement_checks import forced_branch

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
def test_mixed_inversion_matches_locked_alias_stream() -> None:
    km = pytest.importorskip("kickmix")
    assert km.__version__ == "0.1" and hasattr(km, "Simulator")
    assert list(Path(km.__file__).parent.glob("*.so"))
    coefficient = ClassicalFieldCoefficient(1)
    operands = (True, 3, coefficient, False, coefficient)
    inverted = tuple(_invert_operand(operand) for operand in operands)
    assert tuple(operand for operand, _ in inverted) == (
        False,
        3,
        coefficient,
        True,
        coefficient,
    )
    assert operands[0] is True and operands[3] is False
    gates = tuple(gate for _, body in inverted for gate in body)
    source = "X q3\nBIT_INVERT b1\nBIT_INVERT b1\n"
    basis = torch.arange(16)
    initial = (basis + 1).to(torch.complex128) + 1j * (16 - basis)
    initial /= initial.norm()
    for value in (0, 1):
        setup = (ClassicalXor((), 1, bool(value)).instruction(),)
        emitted = encode_kmx(lower_reversible_phase(CircuitIR(4, (*setup, *gates))))
        prefix = encode_kmx(lower_reversible_phase(CircuitIR(4, setup)))
        assert emitted == prefix + source
        sim = km.Simulator(batch_size=1, seed=1)
        sim.do(km.Circuit(emitted))
        assert sim.read_across_shots(km.b(1), out=int) == value
        assert sim.read_across_shots(km.q(3), out=int) == 1
        feedback = Instruction("x", (0,), metadata={"conditions": ((1, 1),)})
        state = forced_branch(initial, (*setup, *gates, feedback), ())
        torch.testing.assert_close(state, initial[basis ^ 1 ^ (value << 3)])


@pytest.mark.ftqc_reference
def test_source_move_assignment_leaks_a_scope_and_native_refuses() -> None:
    km = pytest.importorskip("kickmix")
    # The locked C++ fixture move-assigns one active RAII guard over another,
    # then destroys the surviving guard. Only one pop is emitted.
    source = "PUSH_CONDITION if b0\nPUSH_CONDITION if b1\nX q0\nPOP_CONDITION\n"
    header = 'INFO_TAG "@flagquantum:reversible-phase-v1 wires=2 bits=2 inputs=0,1"\n'
    with pytest.raises(ValueError, match="Unclosed"):
        decode_kmx(header + source)
    leaking = km.Circuit(source + "X q1\n")
    for first, second in product((0, 1), repeat=2):
        sim = km.Simulator(batch_size=1, seed=1)
        sim.write_across_shots(km.b(0), first)
        sim.write_across_shots(km.b(1), second)
        sim.do(leaking)
        assert sim.read_across_shots(km.q(0), out=int) == first & second
        assert sim.read_across_shots(km.q(1), out=int) == first

"""Locked control resolution, clean and borrowed schedules, and scope defects."""

from itertools import product
from pathlib import Path

import pytest
import torch

from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient,
)
from flagquantum.compiler.registers.primitives._mixed_gates import _flip, _XBasisTarget
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR, Instruction
from tests.unit.fixtures.kickmix_source_oracle import normalized_stream
from tests.unit.measurement_checks import forced_branch

pytestmark = pytest.mark.unit
_G1 = ClassicalFieldCoefficient(1)
_G3 = ClassicalFieldCoefficient(3)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("phase,clean", tuple(product((False, True), repeat=2)))
def test_high_arity_mixed_controls_agree_with_locked_source_and_all_branches(
    phase: bool, clean: bool
) -> None:
    km = pytest.importorskip("kickmix")
    assert km.__version__ == "0.1" and hasattr(km, "Simulator")
    assert list(Path(km.__file__).parent.glob("*.so"))
    quantum = (5, 1, 3, 4) if phase else (5, 1, 3)
    gates = _flip(
        None if phase else 4,
        _G3,
        *quantum,
        _G1,
        scratch=7 if clean else None,
        measurement_bit=4 if clean else None,
    )
    helper = 7 if clean else 0
    pair = [f"CCX q5 q1 q{helper}", f"CC{'Z' if phase else 'X'} q{helper} q3 q4"]
    body = (
        [f"R q{helper}", *pair, f"HMR q{helper} b4", "CZ q5 q1 if b4"]
        if clean
        else pair * 2
    )
    source = "\n".join(
        (
            "PUSH_CONDITION if b1",
            "PUSH_CONDITION if b3",
            *body,
            "POP_CONDITION",
            "POP_CONDITION",
        )
    )
    assert str(km.Circuit(source)).strip() == source
    index = torch.arange(256)
    initial = (index + 1).to(torch.complex128) + 1j * (256 - index)
    if clean:
        initial[1::2] = 0
    initial /= initial.norm()
    fully_enabled = ((index >> 2) & (index >> 6) & (index >> 4) & 1).bool()
    for first, second in product((0, 1), repeat=2):
        setup = (
            ClassicalXor((), 1, bool(first)).instruction(),
            ClassicalXor((), 3, bool(second)).instruction(),
            ClassicalXor((), 4, True).instruction(),
        )
        prefix = encode_kmx(lower_reversible_phase(CircuitIR(8, setup)))
        lowered = encode_kmx(lower_reversible_phase(CircuitIR(8, (*setup, *gates))))
        assert normalized_stream(
            lowered.removeprefix(prefix).strip()
        ) == normalized_stream(source)
        enabled = bool(first & second)
        active = fully_enabled & enabled
        if phase:
            expected = initial * torch.where(active & ((index >> 3) & 1).bool(), -1, 1)
        else:
            expected = initial[index ^ (active.long() << 3)]
        for history in ((0,), (1,)) if clean else ((),):
            if clean and enabled:
                branch = expected / (2**0.5)
            elif clean and history == (1,):
                branch = torch.zeros_like(expected)
            else:
                branch = expected
            actual = forced_branch(initial, (*setup, *gates), history)
            torch.testing.assert_close(actual, branch, rtol=0, atol=2e-15)
        if clean and not enabled:
            read = Instruction("x", (6,), metadata={"conditions": ((4, 1),)})
            actual = forced_branch(initial, (*setup, *gates, read), (0,))
            torch.testing.assert_close(actual, initial[index ^ 2], rtol=0, atol=2e-15)
            sim = km.Simulator(batch_size=1, seed=1)
            sim.do(km.Circuit(prefix + source))
            assert sim.read_across_shots(km.b(4), out=int) == 1


def test_resolving_x_basis_targets_has_no_state_to_leak() -> None:
    setup = tuple(ClassicalXor((), bit).instruction() for bit in range(4))
    gates = (
        *_flip(0, _G3, 2, _G1, True),
        *_flip(4),
        *_flip(_XBasisTarget(ClassicalFieldCoefficient(2)), 0, _G3),
    )
    native = encode_kmx(lower_reversible_phase(CircuitIR(5, (*setup, *gates))))
    expected = (
        "PUSH_CONDITION if b1\nCX q2 q0 if b3\nPOP_CONDITION\nX q4\n"
        "PUSH_CONDITION if b2\nZ q0 if b3\nPOP_CONDITION"
    )
    assert normalized_stream("\n".join(native.splitlines()[5:])) == normalized_stream(
        expected
    )

    assert _flip(_XBasisTarget(False), 2) == ()
    assert _flip(0) == (Instruction("x", (0,)),)
    with pytest.raises(ValueError, match="control count"):
        _flip(4, 0, 1, 2, 3, _G1, _G3)
    with pytest.raises(ValueError, match="Unclosed"):
        decode_kmx(
            'INFO_TAG "@flagquantum:reversible-phase-v1 wires=6 bits=2 inputs=1"\n'
            "PUSH_CONDITION if b1\nX q5\n"
        )
    with pytest.raises(ValueError, match="overwrite a guard"):
        _flip(4, 5, 1, 3, _G1, scratch=7, measurement_bit=1)

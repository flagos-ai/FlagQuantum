"""Validate exact monomial-channel certificates against independent primitive execution."""

from math import pi, sqrt

import pytest
import torch

from flagquantum.compiler.registers.primitives.and_compute import (
    and_compute_ir,
    and_uncompute_ir,
)
from flagquantum.compiler.registers.primitives.controlled_swap import (
    controlled_swap_bit_ir,
)
from flagquantum.core.ir import Instruction
from tests.unit.measurement_checks import certified_and_monomial_map, forced_branch

pytestmark = pytest.mark.unit


def test_monomial_certificate_complex_phases_and_every_erasure_outcome() -> None:
    gates = (
        *and_compute_ir(0, 1, 3),
        Instruction("cy", (3, 2)),
        *and_uncompute_ir(0, 1, 3, 0),
        *controlled_swap_bit_ir(0, 1, 2),
        *(
            Instruction(name, (i % 3,))
            for i, name in enumerate(("y", "z", "s", "sdg", "t", "tdg", "i"))
        ),
        Instruction("phase", (0,), params={"theta": pi / 3}),
        Instruction("rz", (1,), params={"theta": pi / 7}),
        Instruction("cz", (1, 2)),
        Instruction("swap", (0, 2)),
        Instruction("ccx", (0, 1, 2)),
    )
    basis = tuple(
        tuple((value >> (2 - i)) & 1 for i in range(3)) + (0,) for value in range(8)
    )
    outputs, phases, events = certified_and_monomial_map(gates, basis)
    assert events == 1
    for column, bits in enumerate(outputs):
        initial = torch.zeros(16, dtype=torch.complex128)
        initial[column * 2] = 1
        expected = torch.zeros_like(initial)
        index = sum(bit << (3 - i) for i, bit in enumerate(bits))
        expected[index] = phases[column] / sqrt(2)
        for outcome in (0, 1):
            torch.testing.assert_close(
                forced_branch(initial, gates, (outcome,)), expected, atol=2e-14, rtol=0
            )
    with pytest.raises(AssertionError):
        certified_and_monomial_map(and_uncompute_ir(0, 1, 3, 0), ((1, 1, 0, 0),))
    with pytest.raises(AssertionError):
        certified_and_monomial_map((Instruction("h", (0,)),), ((0,),))

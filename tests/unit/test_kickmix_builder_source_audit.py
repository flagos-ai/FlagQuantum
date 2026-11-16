"""Locked circuit-builder gate schedules and coherent scratch behavior."""

import pytest
import torch

from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.phase.masked_prefix_phase import _four_operand_phase
from flagquantum.compiler.registers.primitives._mixed_gates import (
    _flip,
    _parity_controlled_flip,
)
from flagquantum.core.ir import CircuitIR
from tests.unit.measurement_checks import forced_branch

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("phase", [False, True])
@pytest.mark.parametrize("alias", ["first", "second", "equal_parity"])
def test_builder_parity_alias_order_and_complex_amplitudes(
    phase: bool, alias: str
) -> None:
    first = 5
    second = first if alias == "equal_parity" else 1
    control = {"first": first, "second": second, "equal_parity": 3}[alias]
    target = 4
    ir = _parity_controlled_flip(
        None if phase else target,
        first,
        second,
        control,
        phase_operand=target if phase else True,
    )
    suffix = "Z" if phase else "X"
    original = (
        []
        if alias == "equal_parity"
        else [f"CC{suffix} q5 q1 q4", f"C{suffix} q{control} q4"]
    )
    program = lower_reversible_phase(CircuitIR(6, ir))
    assert encode_kmx(program).splitlines()[1:] == original
    assert program.n_bits == 0

    basis = torch.arange(64)
    initial = (basis + 1).to(torch.complex128) + 1j * (64 - basis)
    initial /= initial.norm()
    parity = ((basis >> (5 - first)) ^ (basis >> (5 - second))) & 1
    active = parity & ((basis >> (5 - control)) & 1)
    if phase:
        expected = initial * torch.where((active & ((basis >> 1) & 1)).bool(), -1, 1)
    else:
        expected = initial[basis ^ (active << 1)]
    torch.testing.assert_close(
        forced_branch(initial, ir, ()), expected, rtol=0, atol=1e-16
    )


@pytest.mark.parametrize("phase", [False, True])
@pytest.mark.parametrize("clean", [False, True])
def test_builder_three_control_stream_and_all_branch_amplitudes(
    phase: bool, clean: bool
) -> None:
    scratch = 7 if clean else None
    bit = 0 if clean else None
    ir = (
        _four_operand_phase(5, 1, 3, 4, clean=scratch, measurement_bit=bit)
        if phase
        else _flip(4, 5, 1, 3, scratch=scratch, measurement_bit=bit)
    )
    helper = 7 if clean else 0
    pair = [f"CCX q5 q1 q{helper}", f"{'CCZ' if phase else 'CCX'} q{helper} q3 q4"]
    # Recorded from the unchanged builder. Native clean leases enter in zero,
    # satisfying the source's leading R without emitting a redundant reset.
    original = ["R q7", *pair, "HMR q7 b0", "CZ q5 q1 if b0"] if clean else pair * 2
    program = lower_reversible_phase(CircuitIR(8, ir))
    assert encode_kmx(program).splitlines()[1:] == original[int(clean) :]
    assert program.n_bits == int(clean)

    basis = torch.arange(256)
    initial = (basis + 1).to(torch.complex128) + 1j * (256 - basis)
    if clean:
        initial[1::2] = 0
    initial /= initial.norm()
    active = (basis & 0b01010100) == 0b01010100
    if phase:
        expected = initial * torch.where(active & ((basis & 8) != 0), -1, 1)
    else:
        expected = initial[basis ^ (active.to(torch.int64) * 8)]
    for history in ((0,), (1,)) if clean else ((),):
        branch = forced_branch(initial, ir, history)
        torch.testing.assert_close(
            branch, expected / (2**0.5 if clean else 1), atol=3e-15, rtol=0
        )
        if clean:
            assert torch.count_nonzero(branch[1::2]) == 0

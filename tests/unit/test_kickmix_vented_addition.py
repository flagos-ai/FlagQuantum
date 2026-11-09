"""Check the measured carries and retained phase of locked vented addition."""

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.arithmetic.vented_addition import (
    vented_addition_ir,
)
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "width,promote,omit_low,controlled,measurements",
    [
        (2, False, False, True, 1),
        (3, False, True, True, 2),
        (4, False, False, False, 2),
        (4, True, False, True, 3),
        (6, False, True, True, 4),
    ],
)
def test_vented_addition_matches_every_full_complex_branch(
    width: int,
    promote: bool,
    omit_low: bool,
    controlled: bool,
    measurements: int,
) -> None:
    copy_count = width + 1 if promote else 2
    copy = tuple(range(width, width + copy_count))
    copy_ports = (*copy[:-1], None) if omit_low else copy
    high_copy = copy[0] if promote else None
    copied = copy[1:] if promote else copy
    low_offset, carry, control = range(width + copy_count, width + copy_count + 3)
    data = control + 1
    total = data + 2
    effective = width + int(promote)
    # Quantum low coefficient, a measured-data slot initialized to one,
    # and leading zeros inserted by the emitter.
    rhs = (ClassicalFieldCoefficient(0), low_offset)
    vents = tuple(range(1, effective - 1))
    measurement_bit = effective
    setup = tuple(
        ClassicalXor((), bit, constant=True).instruction() for bit in (0, *vents)
    )
    gates = vented_addition_ir(
        tuple(range(width)),
        rhs,
        carry_copy=copy_ports,
        clean=(data, data + 1),
        vent_bits=vents,
        carry_in=carry,
        control=control if controlled else True,
        measurement_bit=measurement_bit,
    )

    basis = torch.arange(1 << data)
    amplitudes = torch.complex((basis + 1).double(), (3 * basis - 5).double())
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    initial[::4] = amplitudes
    circuit = DynamicCircuit(total, inputs=initial, dtype=torch.complex128).append(
        FieldOperation(total, (*setup, *gates))
    )
    shots = 192
    result = run_dynamic(circuit, shots=shots, seed=41, strategy="trajectory")
    assert result.statistics["measurement_count"] == measurements * shots
    assert result.statistics["branch_count"] == 1 << measurements

    current = basis >> (data - width)
    if high_copy is not None:
        current |= ((basis >> (data - 1 - high_copy)) & 1) << width
    offset = 2 + ((basis >> (data - 1 - low_offset)) & 1)
    carry_in = (basis >> (data - 1 - carry)) & 1
    enabled = (basis & 1) if controlled else 1
    value = (current + (offset + carry_in) * enabled) % (1 << effective)
    carries = (current ^ value ^ offset) * enabled
    outputs = basis ^ (((current ^ value) % (1 << width)) << (data - width))
    if high_copy is not None:
        outputs ^= ((current ^ value) >> width) << (data - 1 - high_copy)
    for k, dest in enumerate(reversed(copied)):
        if not (omit_low and k == 0):
            outputs ^= ((carries >> k) & 1) << (data - 1 - dest)

    observed = result.classical_bits[:, list(vents)]
    assert torch.all(observed[:, 0] == 1) if vents else observed.shape[-1] == 0
    for values in observed.unique(dim=0).tolist() if vents else [[]]:
        parity = torch.zeros_like(basis)
        for k, value in enumerate(values):
            parity ^= ((carries >> k) & 1) * (value ^ 1)
        expected = torch.zeros_like(initial)
        expected[outputs * 4] = amplitudes * (1 - 2 * parity)
        match = (observed == torch.tensor(values, dtype=observed.dtype)).all(dim=1)
        actual = result.final_states.reshape(shots, -1)[match]
        torch.testing.assert_close(
            actual, expected.expand_as(actual), atol=1e-12, rtol=0
        )


def test_vent_slots_lower_to_native_conditional_inversions() -> None:
    setup = tuple(ClassicalXor((), bit).instruction() for bit in (0, 1, 2))
    gates = vented_addition_ir(
        (4, 3, 2, 1, 0),
        (False, True, False, True, False),
        clean=(5, 6),
        vent_bits=(0, 1, 2),
        measurement_bit=3,
    )
    code = encode_kmx(
        lower_reversible_phase(CircuitIR(7, (*setup, *gates)))
    ).splitlines()
    assert [line for line in code if line.startswith("BIT_INVERT")] == [
        "BIT_INVERT b1 if b3",
        "BIT_INVERT b2 if b3",
    ]
    assert sum(line.startswith("HMR") for line in code) == 3


def test_vented_addition_checks_port_ownership() -> None:
    with pytest.raises(ValueError, match="low carry-copy"):
        vented_addition_ir((0, 1), (True,), clean=(2, 3), carry_copy=(None, 4))
    with pytest.raises(ValueError, match="vent slots"):
        vented_addition_ir((0, 1, 2), (True,), clean=(3, 4))
    with pytest.raises(ValueError, match="distinct nonnegative"):
        vented_addition_ir(
            (0, 1, 2), (True,), clean=(3, 4), vent_bits=(0,), measurement_bit=0
        )

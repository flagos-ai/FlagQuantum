"""Locked mixed AND deletion and repeated result allocation."""

from itertools import product
from pathlib import Path

import pytest
import torch

from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.finite_field.coefficients import (
    ClassicalFieldCoefficient,
)
from flagquantum.compiler.registers.primitives._mixed_gates import (
    _delete_and_ir,
    _flip,
)
from flagquantum.compiler.registers.primitives.measurement import (
    delete_zero_register_ir,
    reset_x_register_ir,
)
from flagquantum.core.classical_dataflow import ClassicalXor
from flagquantum.core.ir import CircuitIR
from tests.unit.fixtures.kickmix_source_oracle import normalized_stream
from tests.unit.measurement_checks import forced_branch

pytestmark = pytest.mark.unit

_DELETE = {
    (1, 1): "NEG if b2",
    (1, 2): "PUSH_CONDITION if b2\nNEG if b1\nPOP_CONDITION",
    (1, 3): "Z q1 if b2",
    (2, 1): "PUSH_CONDITION if b2\nNEG if b0\nPOP_CONDITION",
    (
        2,
        2,
    ): "PUSH_CONDITION if b0\nPUSH_CONDITION if b2\nNEG if b1\nPOP_CONDITION\nPOP_CONDITION",
    (2, 3): "PUSH_CONDITION if b2\nZ q1 if b0\nPOP_CONDITION",
    (3, 1): "Z q0 if b2",
    (3, 2): "PUSH_CONDITION if b2\nZ q0 if b1\nPOP_CONDITION",
    (3, 3): "CZ q0 q1 if b2",
}
_RESET = {
    (1, 1): "X q2",
    (1, 2): "X q2 if b1",
    (1, 3): "CX q1 q2",
    (2, 1): "X q2 if b0",
    (2, 2): "PUSH_CONDITION if b0\nX q2 if b1\nPOP_CONDITION",
    (2, 3): "CX q1 q2 if b0",
    (3, 1): "CX q0 q2",
    (3, 2): "CX q0 q2 if b1",
    (3, 3): "CCX q0 q1 q2",
}


@pytest.mark.parametrize("left,right", tuple(product(range(4), repeat=2)))
def test_mixed_and_matches_locked_reset_and_all_erasure_branches(
    left: int, right: int
) -> None:
    first = (False, True, ClassicalFieldCoefficient(0), 0)[left]
    second = (False, True, ClassicalFieldCoefficient(1), 1)[right]
    erase = _delete_and_ir(2, 2, first, second)
    compute = (*reset_x_register_ir((2,)), *_flip(2, first, second))
    base = torch.arange(8)
    zero_indices = ((base & 6) << 1) | (base & 1)
    amplitudes = (base + 1).to(torch.complex128) + 1j * (8 - base)
    amplitudes /= amplitudes.norm()
    expected = torch.zeros(16, dtype=torch.complex128)
    expected[zero_indices] = amplitudes
    for b0, b1 in product((0, 1), repeat=2):
        setup = tuple(
            ClassicalXor((), slot, bool(value)).instruction()
            for slot, value in enumerate((b0, b1))
        )
        prefix = encode_kmx(lower_reversible_phase(CircuitIR(4, setup))).splitlines()[
            1:
        ]
        reset = encode_kmx(lower_reversible_phase(CircuitIR(4, (*setup, *compute))))
        delete = encode_kmx(lower_reversible_phase(CircuitIR(4, (*setup, *erase))))
        start = len(prefix) + 1
        assert reset.splitlines()[start] == "R q2"
        assert normalized_stream("\n".join(reset.splitlines()[start + 1 :])) == (
            normalized_stream(_RESET.get((left, right), ""))
        )
        assert normalized_stream(
            "\n".join(delete.splitlines()[start:])
        ) == normalized_stream(("HMR q2 b2\n" + _DELETE.get((left, right), "")).strip())

        left_active = (0, 1, b0, (base >> 2) & 1)[left]
        right_active = (0, 1, b1, (base >> 1) & 1)[right]
        prepared = torch.zeros_like(expected)
        prepared[zero_indices + 2 * (left_active & right_active)] = amplitudes
        for result in (0, 1):
            actual = forced_branch(prepared, (*setup, *erase), (result,))
            torch.testing.assert_close(actual, expected / (2**0.5), rtol=0, atol=1e-15)


@pytest.mark.ftqc_reference
def test_delete_zero_broadcast_reuses_one_source_result_slot() -> None:
    km = pytest.importorskip("kickmix")
    assert km.__version__ == "0.1" and hasattr(km, "Simulator")
    assert list(Path(km.__file__).parent.glob("*.so"))
    source = "HMR q3 b0\nHMR q1 b0\nHMR q3 b0"
    gates = delete_zero_register_ir((3, 1, 3), 0)
    program = lower_reversible_phase(CircuitIR(4, gates))
    assert program.n_bits == 1
    assert encode_kmx(program).splitlines()[1:] == str(km.Circuit(source)).splitlines()
    initial = torch.zeros(16, dtype=torch.complex128)
    initial[8] = torch.tensor(1j, dtype=torch.complex128)
    for branch in product((0, 1), repeat=3):
        actual = forced_branch(initial, gates, branch)
        torch.testing.assert_close(actual, initial / (2**1.5), rtol=0, atol=1e-15)


def test_delete_and_rejects_overwriting_classical_control() -> None:
    with pytest.raises(ValueError, match="cannot overwrite a control"):
        _delete_and_ir(2, 0, ClassicalFieldCoefficient(0), True)

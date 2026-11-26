"""Compare actual source coefficient normalization before unitary admission."""

from typing import Any

import pytest
import torch

from flagquantum.ecosystem._pauli_select_component_library import (
    capture_pauli_select_pair,
)
from flagquantum.ecosystem.cirq._select_pauli_lcu import select_pauli_lcu_from_cirq
from flagquantum.errors import SerializationError
from tests.unit.pauli_select_reference import compile_pauli_select
from tests.unit.pauli_select_reference import source as source
from tests.unit.test_chebyshev_component import assert_all_small_raw_histories
from tests.unit.test_default_linear_combination_oracles import bind

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

COEFFICIENTS = (
    complex(float("inf"), 0),
    complex(-float("inf"), 0),
    complex(1, float("inf")),
    complex(-1, -float("inf")),
    complex(1, float("nan")),
    complex(-1, float("nan")),
)


@pytest.mark.parametrize("coefficient", COEFFICIENTS)
@pytest.mark.parametrize("control", (None, 0, 1))
def test_actual_source_sign_and_both_conversion_paths(
    source: Any, coefficient: complex, control: int | None
) -> None:
    cirq = pytest.importorskip("cirq")

    sign = -1 if coefficient.real < 0 else 1
    rows = tuple(
        cirq.DensePauliString(word, coefficient=coefficient) for word in ("XY", "ZI")
    )
    original = source.SelectPauliLCU(1, 2, rows, control)
    normalized = source.SelectPauliLCU(
        1,
        2,
        tuple(cirq.DensePauliString(word, coefficient=sign) for word in ("XY", "ZI")),
        control,
    )
    wire, *targets = cirq.LineQubit.range(3)
    for index in range(2):
        operation = original.nth_operation(None, index, wire, targets)
        expected = normalized.nth_operation(None, index, wire, targets)
        assert cirq.has_unitary(operation)
        torch.testing.assert_close(
            torch.as_tensor(cirq.unitary(operation)),
            torch.as_tensor(cirq.unitary(expected)),
            rtol=0,
            atol=0,
        )
    assert select_pauli_lcu_from_cirq(original) == select_pauli_lcu_from_cirq(
        normalized
    )
    # Identical complete Programs reuse their accepted call/feedback/work proofs.
    assert capture_pauli_select_pair(original) == capture_pauli_select_pair(normalized)


@pytest.mark.parametrize("control", (None, 0, 1))
def test_complete_nonfinite_metadata_bank_full_complex_branches_and_resources(
    source: Any, control: int | None
) -> None:
    cirq = pytest.importorskip("cirq")

    words = ("IX", "-XY", "YZ", "-ZI")
    coefficients = (COEFFICIENTS[0], COEFFICIENTS[1], COEFFICIENTS[4], COEFFICIENTS[3])
    original = source.SelectPauliLCU(
        2,
        2,
        tuple(
            cirq.DensePauliString(word.removeprefix("-"), coefficient=coefficient)
            for word, coefficient in zip(words, coefficients, strict=True)
        ),
        control,
    )
    # Ordinary append, actual source primitive dependencies and independent cost.
    compile_pauli_select(original, words)
    expected = torch.as_tensor(original.tensor_contract(), dtype=torch.complex128)
    for record in capture_pauli_select_pair(original):
        bound, logical = bind(record)
        assert_all_small_raw_histories(bound.circuit, logical, expected)


@pytest.mark.parametrize("coefficient", (0j, -0j, 1j, -1j, complex(float("nan"), 1)))
def test_nonunitary_normalized_source_leaves_remain_refused(
    source: Any, coefficient: complex
) -> None:
    cirq = pytest.importorskip("cirq")

    original = source.SelectPauliLCU(
        1,
        1,
        (cirq.DensePauliString("X", coefficient=coefficient),) * 2,
        1,
    )
    control, target = cirq.LineQubit.range(2)
    assert not cirq.has_unitary(original.nth_operation(None, 0, control, (target,)))
    with pytest.raises(ValueError, match="coefficient"):
        select_pauli_lcu_from_cirq(original)
    with pytest.raises(SerializationError, match="not unitary"):
        capture_pauli_select_pair(original)

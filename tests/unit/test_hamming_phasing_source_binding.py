"""Bind complete Hamming-phase objects without changing reference failures."""

from cmath import exp
from collections import Counter
from dataclasses import replace
from math import pi, sqrt
from typing import Any

import pytest
import torch

from examples.algorithms.hamming_phasing_catalog import HammingPhasingExample
from flagquantum import Circuit
from flagquantum.algorithms.primitives.circuit_register_emitter import (
    CircuitRegisterEmitter,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_hamming_phasing_file import source as source
from tests.unit.test_phase_gradient_source_binding import through_operators

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("exponent", [0.375, -0.375])
def test_complete_direct_source_object_and_native_callback_replacement(
    source: Any, width: int, exponent: float
) -> None:
    original = source.HammingWeightPhasing(width, exponent, 0.2)
    native = HammingPhasingExample(width, exponent, 0.2)
    assert_source_model(native.resource_model(), original)
    actual = compare_binding(original, native)
    diagonal = torch.tensor(
        [exp(1j * pi * exponent * x.bit_count()) for x in range(1 << width)],
        dtype=torch.complex128,
    )
    torch.testing.assert_close(actual[0], torch.diag(diagonal), atol=1e-11, rtol=0)


def compare_binding(
    original: Any, native: HammingPhasingExample
) -> tuple[torch.Tensor, torch.Tensor]:
    imported = import_register_emitter(original)
    captured = replace(
        imported,
        snapshots=(
            CircuitIR.from_json(imported.snapshots[0].to_json()),
            CircuitIR.from_json(imported.snapshots[1].to_json()),
        ),
    )
    forward = Circuit(native.qubit_count).append(native).to_ir()
    backward = Circuit(native.qubit_count).append(replace(native, adjoint=True)).to_ir()
    replacement = CircuitRegisterEmitter(captured.layout, (forward, backward))
    actual = through_operators(captured)
    expected = through_operators(replacement)
    for index in (0, 1):
        torch.testing.assert_close(actual[index], expected[index], atol=1e-11, rtol=0)
        assert Counter(
            g.name for g in captured.snapshots[index].instructions
        ) == Counter(g.name for g in replacement.snapshots[index].instructions)
    return actual


@pytest.mark.parametrize("width", [2, 3])
@pytest.mark.parametrize("exponent", [1.0, -1.1934341])
def test_executable_original_gradient_cases_complete_operator(
    source: Any, width: int, exponent: float
) -> None:
    original = source.HammingWeightPhasingViaPhaseGradient(width, exponent, 0.2)
    native = HammingPhasingExample(width, exponent, 0.2, via_gradient=True)
    actual, _ = compare_binding(original, native)
    size = 1 << native.gradient_width
    fourier = torch.tensor(
        [exp(-2j * pi * k / size) / sqrt(size) for k in range(size)],
        dtype=torch.complex128,
    )
    for x in range(1 << width):
        initial = torch.zeros(
            1 << (width + native.gradient_width), dtype=torch.complex128
        )
        initial[x * size : (x + 1) * size] = fourier
        expected = initial * exp(1j * pi * exponent * x.bit_count())
        result = actual @ initial
        error = torch.linalg.vector_norm(result - expected).item()
        if exponent == 1.0:
            assert error < 1e-11
        else:
            # Preserve the locked source's finite-precision counterexample.
            phase_steps = (0, 7, 13, 4)
            rounded = initial * exp(2j * pi * phase_steps[x.bit_count()] / 16)
            torch.testing.assert_close(result, rounded, atol=1e-11, rtol=0)
            if x.bit_count() == 1:
                assert error == pytest.approx(0.21457825186131635, abs=1e-12)
                assert error > original.eps
            if x.bit_count() == 3:
                assert error == pytest.approx(0.25160865359628104, abs=1e-12)
                assert error > original.eps


@pytest.mark.parametrize("gradient", [False, True])
def test_original_one_bit_empty_junk_boundary_is_preserved(
    source: Any, gradient: bool
) -> None:
    cls = (
        source.HammingWeightPhasingViaPhaseGradient
        if gradient
        else source.HammingWeightPhasing
    )
    with pytest.raises(KeyError, match=r"QAny\(bitsize=0\)"):
        import_register_emitter(cls(1, 0.375, 0.2))


@pytest.mark.parametrize("width", [2, 3])
@pytest.mark.parametrize("exponent", [0.375, -0.375])
def test_original_gradient_mixed_format_boundary_is_preserved(
    source: Any, width: int, exponent: float
) -> None:
    from qualtran import BloqError

    original = source.HammingWeightPhasingViaPhaseGradient(width, exponent, 0.2)
    with pytest.raises(BloqError, match="register dtypes are not consistent"):
        import_register_emitter(original)

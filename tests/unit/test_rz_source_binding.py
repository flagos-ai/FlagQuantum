"""Bind complete gradient Rz objects across bit-preserving source dtypes."""

from cmath import exp
from collections import Counter
from dataclasses import replace
from math import pi, sqrt
from typing import Any

import pytest
import torch

from examples.algorithms.rz_phase_gradient_catalog import GradientRzExample
from flagquantum import Circuit
from flagquantum.algorithms.primitives.circuit_register_emitter import (
    CircuitRegisterEmitter,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.test_phase_gradient_source_binding import through_operators
from tests.unit.test_rz_phase_gradient_file import (
    assert_resource_model,
)
from tests.unit.test_rz_phase_gradient_file import (
    source_module as source_module,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("angle_width,gradient_width", [(1, 1), (2, 2), (2, 3)])
@pytest.mark.parametrize("kind", ["uint", "any", "fixed", "signed", "bounded"])
def test_symbol_resolved_generic_dtype_rz_complete_source_replacement(
    source_module: Any, angle_width: int, gradient_width: int, kind: str
) -> None:
    from attrs import evolve
    from qualtran import BQUInt, QAny, QDType, QFxp, QInt, QUInt
    from sympy import Symbol

    declaration = source_module.RzViaPhaseGradient(
        QUInt(Symbol("a", positive=True, integer=True)),
        QUInt(Symbol("b", positive=True, integer=True)),
    )
    with pytest.raises(ValueError, match="concrete positive shapes"):
        import_register_emitter(declaration)
    angle: QDType
    gradient: QDType
    if kind == "uint":
        angle, gradient = QUInt(angle_width), QUInt(gradient_width)
    elif kind == "any":
        angle, gradient = QAny(angle_width), QAny(gradient_width)
    elif kind == "fixed":
        angle = QFxp(angle_width, angle_width - 1)
        gradient = QFxp(gradient_width, gradient_width)
    elif kind == "signed":
        angle = QInt(angle_width)
        gradient = (
            QInt(1) if gradient_width == 1 else QFxp(gradient_width, 0, signed=True)
        )
    else:
        angle = BQUInt(angle_width, (1 << angle_width) - 1)
        gradient = BQUInt(gradient_width, (1 << gradient_width) - 1)
    original = evolve(declaration, angle_dtype=angle, phasegrad_dtype=gradient)
    native = GradientRzExample(angle_width, gradient_width)
    for inverse in (False, True):
        assert_resource_model(
            replace(native, adjoint=inverse),
            original.adjoint() if inverse else original,
        )
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
    gradient_size = 1 << gradient_width
    fourier = torch.tensor(
        [
            exp(-2j * pi * k / gradient_size) / sqrt(gradient_size)
            for k in range(gradient_size)
        ],
        dtype=torch.complex128,
    )
    # Casts use raw bit patterns, including signed and bounded padding words.
    for data in (0, 1):
        for raw in range(1 << angle_width):
            initial = torch.zeros(
                1 << (1 + angle_width + gradient_width), dtype=torch.complex128
            )
            start = ((data << angle_width) | raw) * gradient_size
            initial[start : start + gradient_size] = fourier
            phase = exp((1 if data else -1) * 2j * pi * raw / gradient_size)
            torch.testing.assert_close(
                actual[0] @ initial, initial * phase, atol=1e-11, rtol=0
            )


def test_source_rejects_angle_wider_than_gradient(source_module: Any) -> None:
    from qualtran import QAny

    original = source_module.RzViaPhaseGradient(QAny(3), QAny(2))
    with pytest.raises(ValueError, match="less than or equal"):
        import_register_emitter(original)

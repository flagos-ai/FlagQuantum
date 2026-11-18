"""Resolve complete QVR source objects at their named optional SDK boundary."""

from cmath import exp
from collections import Counter
from dataclasses import replace
from math import pi
from typing import Any

import pytest
import torch

from examples.algorithms.qvr_catalog import QvrExample
from flagquantum import Circuit
from flagquantum.algorithms.primitives.circuit_register_emitter import (
    CircuitRegisterEmitter,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.test_first_quantization_resource_model import assert_source_model
from tests.unit.test_phase_gradient_source_binding import through_operators
from tests.unit.test_qvr_file_audit import source_module as source_module

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "width,fraction,gradient,coefficient",
    [
        (2, 2, False, 0.375),
        (2, 2, False, -0.375),
        (3, 2, False, 0.375),
        (3, 2, False, -0.375),
        (2, 2, True, 0.375),
        (2, 2, True, -0.375),
        (3, 2, True, 1.0),
        (3, 0, True, 1.0),
    ],
)
def test_symbolic_qvr_named_port_resolution_and_complete_native_replacement(
    source_module: Any,
    width: int,
    fraction: int,
    gradient: bool,
    coefficient: float,
) -> None:
    from attrs import evolve
    from qualtran import QFxp, Register
    from sympy import Symbol

    cls = source_module.QvrPhaseGradient if gradient else source_module.QvrZPow
    declaration = cls(
        Register("energy", QFxp(Symbol("n", integer=True, positive=True), 2)),
        Symbol("gamma", real=True),
        Symbol("epsilon", positive=True),
    )
    with pytest.raises(ValueError, match="concrete positive shapes"):
        import_register_emitter(declaration)
    original = evolve(
        declaration,
        cost_reg=Register("energy", QFxp(width, fraction)),
        gamma=coefficient,
        eps=0.2,
    )
    native = QvrExample(width, coefficient, 0.2, fraction, gradient)
    if fraction == width:
        assert_source_model(native.resource_model(), original)
    imported = import_register_emitter(original)
    captured = replace(
        imported,
        snapshots=(
            CircuitIR.from_json(imported.snapshots[0].to_json()),
            CircuitIR.from_json(imported.snapshots[1].to_json()),
        ),
    )
    assert captured.layout[0] == ("energy", width)
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
    if not gradient:
        diagonal = torch.tensor(
            [
                exp(2j * pi * coefficient * x / (1 << fraction))
                for x in range(1 << width)
            ],
            dtype=torch.complex128,
        )
        torch.testing.assert_close(actual[0], torch.diag(diagonal), atol=1e-11, rtol=0)


@pytest.mark.parametrize("gradient", [False, True])
def test_original_twelve_bit_factories_capture_both_source_directions(
    source_module: Any, gradient: bool
) -> None:
    factory = source_module._qvr_phase_gradient if gradient else source_module._qvr_zpow
    original = factory.make()
    emitter = import_register_emitter(original)
    native = QvrExample(12, 0.1, 1e-4 if gradient else 1e-2, via_gradient=gradient)
    assert_source_model(native.resource_model(), original)
    assert emitter.layout[0] == ("x", 12)
    for index in (0, 1):
        ir = emitter.snapshots[index]
        assert CircuitIR.from_json(ir.to_json()).to_json() == ir.to_json()
        expected = (
            Circuit(native.qubit_count)
            .append(replace(native, adjoint=bool(index)))
            .to_ir()
        )
        assert Counter(g.name for g in ir.instructions) == Counter(
            g.name for g in expected.instructions
        )
        counts = Counter(g.name for g in ir.instructions)
        if gradient:
            assert emitter.layout[1] == ("phase_grad", 19)
            assert counts["t"] + counts["tdg"] == 484
            assert counts["cx"] == 1244 and counts["measure"] == 121
        else:
            assert counts == {"phase": 6}


@pytest.mark.parametrize("fraction", [0, 2])
@pytest.mark.parametrize("coefficient", [0.375, -0.375])
def test_original_qvr_mixed_type_failures_remain_visible(
    source_module: Any, fraction: int, coefficient: float
) -> None:
    from qualtran import BloqError, QFxp, Register

    original = source_module.QvrPhaseGradient(
        Register("energy", QFxp(3, fraction)), coefficient, 0.2
    )
    with pytest.raises(BloqError, match="register dtypes are not consistent"):
        import_register_emitter(original)

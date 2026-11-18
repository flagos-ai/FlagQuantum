"""Resolve all constant Z-power factories at the original register boundary."""

from cmath import exp
from collections import Counter
from dataclasses import fields, replace
from itertools import product
from math import pi, sqrt
from typing import Any

import pytest
import torch

from examples.algorithms.zpow_phase_gradient_catalog import GradientZPowExample
from flagquantum import Circuit
from flagquantum.algorithms.primitives.circuit_register_emitter import (
    CircuitRegisterEmitter,
)
from flagquantum.compiler._component_trace.resource_model import GateCost
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_phase_gradient_source_binding import restored, through_operators
from tests.unit.test_zpow_phase_gradient_file import assert_bound_stream
from tests.unit.test_zpow_phase_gradient_file import source_module as source_module

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("kind", ["concrete", "symb_prec", "symb_angle"])
@pytest.mark.parametrize(
    "width,exponent", [(1, 0.375), (2, 0.375), (3, 0.375), (3, 1.25), (3, 3.0)]
)
def test_complete_factory_binding_full_operators_and_histories(
    source_module: Any, kind: str, width: int, exponent: float
) -> None:
    from attrs import evolve
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    name = "_zpow_const_via_phase_grad"
    if kind != "concrete":
        name += "_" + kind
    declaration = getattr(source_module, name).make()
    if kind != "concrete":
        with pytest.raises(
            ValueError, match="concrete positive shapes|cannot be decomposed"
        ):
            import_register_emitter(declaration)
    source = evolve(declaration, exponent=exponent, phase_grad_bitsize=width)
    native = GradientZPowExample(exponent, width)
    model = native.resource_model()
    source_cost = get_cost_value(source, QECGatesCost())
    assert model.logical_qubits == get_cost_value(source, QubitCount())
    for field in fields(GateCost):
        assert getattr(model, field.name) == getattr(source_cost, field.name)
    imported = restored(source)
    streams = tuple(
        assert_bound_stream(replace(native, adjoint=inverse), source)
        for inverse in (False, True)
    )
    replacement = CircuitRegisterEmitter(
        imported.layout,
        (
            CircuitIR(native.qubit_count, streams[0]),
            CircuitIR(native.qubit_count, streams[1]),
        ),
    )
    actual = through_operators(imported)
    expected = through_operators(replacement)
    logical = width + 1
    size = 1 << width
    constant = int(source._load_bloq.k)
    fourier = torch.tensor(
        [exp(-2j * pi * y / size) / sqrt(size) for y in range(size)],
        dtype=torch.complex128,
    )
    initial = torch.zeros(1 << native.qubit_count, dtype=torch.complex128)
    stride = 1 << (native.qubit_count - logical)
    initial[::stride] = torch.exp(
        0.17j * torch.arange(1 << logical, dtype=torch.float64)
    ) / sqrt(1 << logical)
    for inverse in (False, True):
        direction = int(inverse)
        sign = -1 if inverse else 1
        torch.testing.assert_close(
            actual[direction], expected[direction], atol=1e-11, rtol=0
        )
        assert Counter(
            g.name for g in imported.snapshots[direction].instructions
        ) == Counter(g.name for g in streams[direction])
        permutation = torch.zeros_like(actual[direction])
        for q, y in product((0, 1), range(size)):
            permutation[q * size + (y + sign * q * constant) % size, q * size + y] = 1
        torch.testing.assert_close(actual[direction], permutation, atol=1e-11, rtol=0)
        for q in (0, 1):
            gradient_state = torch.zeros(1 << logical, dtype=torch.complex128)
            gradient_state[q * size : (q + 1) * size] = fourier
            torch.testing.assert_close(
                actual[direction] @ gradient_state,
                gradient_state * exp(sign * 2j * pi * q * constant / size),
                atol=1e-11,
                rtol=0,
            )
        output = torch.zeros_like(initial)
        output[::stride] = permutation @ initial[::stride]
        for history in product((0, 1), repeat=width - 1):
            torch.testing.assert_close(
                forced_branch(initial, streams[direction], history),
                output * 2 ** (-(width - 1) / 2),
                atol=1e-11,
                rtol=0,
            )


def test_unsigned_source_negative_angle_rejection_is_preserved(
    source_module: Any,
) -> None:
    source = source_module.ZPowConstViaPhaseGradient(-0.375, 3)
    with pytest.raises(ValueError, match="unsigned"):
        import_register_emitter(source)
    with pytest.raises(ValueError, match="nonnegative"):
        Circuit(9).append(GradientZPowExample(-0.375, 3))

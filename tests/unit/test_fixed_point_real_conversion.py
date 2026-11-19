# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Real conversion call-chain conformance against the unmodified source."""

from collections.abc import Callable
from copy import deepcopy
from hashlib import sha1
from itertools import islice, product
from pathlib import Path
from typing import Any

import pytest

from flagquantum.ecosystem import fixed_point as native
from flagquantum.simulation.integer_encoding import integer_from_bits
from tests.unit.test_fixed_point_encoding import SHAPES
from tests.unit.test_integer_encoding import compare_call as original_compare_call

np = pytest.importorskip("numpy")
sympy = pytest.importorskip("sympy")
pytest.importorskip("qualtran")

compare_call: Callable[[Callable[[Any], Any], Callable[[Any], Any], Any], None] = (
    original_compare_call
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def source(width: Any, fraction: Any, signed: Any) -> Any:
    from qualtran._infra.data_types import _Fxp

    return _Fxp(width, fraction, signed)


def snapshot(value: Any) -> tuple[Any, ...]:
    """Include shape, configuration and stored bits of external values."""
    return (
        value.dtype,
        value.bin(),
        np.asarray(value.get_val()).tolist(),
        value.config.op_sizing,
        value.config.const_op_sizing,
        value.config.shifting,
        value.config.overflow,
        value.config.rounding,
    )


def test_source_and_optional_dependency_identity() -> None:
    from importlib.metadata import version

    assert version("fxpmath") == "0.4.10"
    root = Path(__file__).resolve().parents[3] / "ftqc-reference-env/qualtran/_infra"
    for name, expected in (
        ("data_types.py", "fe570ea7ee74571d7d5202fef52ac1a215bea1a5"),
        ("data_types_test.py", "355ed49d57e9733f09a1d1ee3302ce2329a776be"),
    ):
        raw = (root / name).read_bytes()
        actual = sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        assert actual == expected


@pytest.mark.parametrize("width,fraction,signed", SHAPES)
def test_template_and_lazy_real_domain(width: Any, fraction: Any, signed: Any) -> None:
    compare_call(
        lambda _: snapshot(source(width, fraction, signed).fxp_dtype_template()),
        lambda _: snapshot(native.fixed_point_template(width, fraction, signed=signed)),
        None,
    )
    compare_call(
        lambda _: [
            snapshot(v)
            for v in islice(source(width, fraction, signed)._get_domain_fxp(), 5)
        ],
        lambda _: [
            snapshot(v)
            for v in islice(
                native.fixed_point_real_domain(width, fraction, signed=signed), 5
            )
        ],
        None,
    )


@pytest.mark.parametrize("width,fraction,signed", SHAPES)
@pytest.mark.parametrize(
    "require_exact,complement", tuple(product((False, True), repeat=2))
)
def test_bits_integer_and_external_mutation(
    width: Any, fraction: Any, signed: Any, require_exact: bool, complement: bool
) -> None:
    fxpmath = pytest.importorskip("fxpmath")

    values = [
        -4.375,
        -1.5,
        -0.7,
        -0.0,
        0,
        0.1,
        0.375,
        0.7,
        1.5,
        4.625,
        32.5,
        np.float64(0.7),
        fxpmath.Fxp(0.7),
        fxpmath.Fxp(-0.7),
    ]
    for value in values:
        for method, implementation in (
            ("_fxp_to_bits", native.fixed_point_real_to_bits),
            ("to_fixed_width_int", native.fixed_point_real_to_integer),
        ):

            def original_call(x: Any, method: str = method) -> Any:
                return getattr(source(width, fraction, signed), method)(
                    x, require_exact=require_exact, complement=complement
                )

            def native_call(
                x: Any, implementation: Callable[..., Any] = implementation
            ) -> Any:
                return implementation(
                    x,
                    width,
                    fraction,
                    signed=signed,
                    require_exact=require_exact,
                    complement=complement,
                )

            compare_call(original_call, native_call, value)
        if isinstance(value, fxpmath.Fxp):
            left, right = deepcopy(value), deepcopy(value)
            for callback, argument in (
                (
                    lambda v: source(width, fraction, signed)._fxp_to_bits(
                        v, require_exact=require_exact, complement=complement
                    ),
                    left,
                ),
                (
                    lambda v: native.fixed_point_real_to_bits(
                        v,
                        width,
                        fraction,
                        signed=signed,
                        require_exact=require_exact,
                        complement=complement,
                    ),
                    right,
                ),
            ):
                try:
                    callback(argument)
                except (ValueError, TypeError, OverflowError):
                    pass
            assert snapshot(left) == snapshot(right)


@pytest.mark.parametrize("width,fraction,signed", SHAPES)
def test_exactness_and_decode_diagnostics(
    width: Any, fraction: Any, signed: Any
) -> None:
    for value in (-1, 0, 0.1, 0.7, 10, sympy.Rational(1, 2), np.array([0.5, 0.7])):
        compare_call(
            lambda x: source(width, fraction, signed)._assert_valid_val(x),
            lambda x: native.validate_fixed_point_real(
                x, width, fraction, signed=signed
            ),
            value,
        )
    for bits in ([], [0], [1], [0, 1, 0], [1, 0, 0], [1, 9], [-1, 0], [True, False]):
        compare_call(
            lambda x: snapshot(source(width, fraction, signed)._from_bits_to_fxp(x)),
            lambda x: snapshot(
                native.fixed_point_bits_to_real(x, width, fraction, signed=signed)
            ),
            bits,
        )


@pytest.mark.parametrize(
    "width,fraction,signed",
    [
        (4, 3, True),
        (3, 3, False),
        (7, 3, False),
        (7, 3, True),
        (3, 0, True),
        (3, -1, False),
    ],
)
def test_complete_small_real_domains(width: int, fraction: int, signed: bool) -> None:
    original = source(width, fraction, signed)
    expected = list(original._get_domain_fxp())
    actual = list(native.fixed_point_real_domain(width, fraction, signed=signed))
    assert [snapshot(v) for v in actual] == [snapshot(v) for v in expected]
    for left, right in zip(expected, actual, strict=True):
        bits = native.fixed_point_real_to_bits(right, width, fraction, signed=signed)
        assert bits == original._fxp_to_bits(left)
        assert snapshot(
            native.fixed_point_bits_to_real(bits, width, fraction, signed=signed)
        ) == snapshot(original._from_bits_to_fxp(bits))
        assert native.fixed_point_real_to_integer(
            right, width, fraction, signed=signed
        ) == original.to_fixed_width_int(left)


def test_existing_integer_owner_is_replaceable(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[int], int, bool]] = []
    decoder = integer_from_bits

    def tracked(bits: Any, width: Any, *, signed: bool = False) -> int:
        calls.append((list(bits), width, signed))
        return decoder(bits, width, signed=signed)

    monkeypatch.setattr(native, "integer_from_bits", tracked)
    assert native.fixed_point_real_to_integer(-1.5, 6, 4, signed=True) == -24
    assert calls == [([1, 0, 1, 0, 0, 0], 6, True)]


@pytest.mark.parametrize("value", [-0.7, 0.1, 0.625])
def test_quantized_consumer_gate_stream_and_every_measurement_branch(
    value: float,
) -> None:
    from collections import Counter
    from math import pi

    import torch
    from qualtran.bloqs.rotations.phase_gradient import PhaseGradientUnitary
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    from examples.algorithms.classical_fixed_point_values import Gradient
    from flagquantum.circuit import Circuit
    from flagquantum.compiler.registers.phase.phase_gradient import (
        phase_gradient_unitary_ir,
    )
    from flagquantum.simulation.fixed_point_encoding import fixed_point_real_value
    from tests.unit.measurement_checks import forced_branch
    from tests.unit.test_register_phase_gradient import source_gradient_gates
    from tests.unit.test_register_subtraction import assert_source

    word = native.fixed_point_real_to_integer(value, 8, 7, signed=True)
    assert word == source(8, 7, True).to_fixed_width_int(value)
    exponent = float(fixed_point_real_value(word, 8, 7, signed=True))
    gradient = (0, 1, 2)
    original = PhaseGradientUnitary(3, exponent=exponent, is_controlled=True)
    ports = {"phase_grad": gradient, "ctrl": (3,)}
    reference, _ = source_gradient_gates(original, ports, ports, work=4, bit=17)
    gates = phase_gradient_unitary_ir(
        gradient, exponent=exponent, control=3, work=4, classical_bit=17
    )
    assert_source(gates, reference)
    model = get_cost_value(original, QECGatesCost())
    counts = Counter(g.name for g in gates)
    assert (
        counts["t"] + counts["tdg"],
        counts["measure"],
        counts["reset"],
        counts["cx"],
        counts["phase"],
    ) == (12, 3, 3, 18, 3)
    assert (model.and_bloq, model.measurement) == (3, 3)

    addresses = torch.arange(64)
    weights = addresses.to(torch.float64) + 1
    initial = torch.complex(weights, weights * (weights % 5 - 2))
    initial[(addresses & 2) != 0] = 0
    initial /= initial.norm()
    phases = (
        2
        * pi
        * exponent
        / 8
        * (addresses >> 3).to(torch.float64)
        * ((addresses >> 2) & 1)
    )
    expected = initial * torch.exp(1j * phases)
    for history in product((0, 1), repeat=3):
        actual = forced_branch(initial, gates, history)
        upstream = forced_branch(initial, reference, history)
        torch.testing.assert_close(actual, upstream, rtol=0, atol=2e-12)
        torch.testing.assert_close(actual, expected * 2 ** (-1.5), rtol=0, atol=2e-12)
        assert torch.count_nonzero(actual[(addresses & 2) != 0]) == 0

    uncontrolled = phase_gradient_unitary_ir(gradient, exponent=exponent)
    circuit = Circuit(
        4,
        inputs=initial.reshape(8, 8)[:, :2].reshape(-1).clone(),
        dtype=torch.complex128,
    )
    before = circuit.state().clone()
    circuit.append(Gradient(uncontrolled))
    circuit.append(Gradient(phase_gradient_unitary_ir(gradient, exponent=-exponent)))
    torch.testing.assert_close(circuit.state(), before, rtol=0, atol=2e-12)


def test_missing_optional_library_fails_at_the_conversion_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> Any:
        assert name == "fxpmath"
        raise ModuleNotFoundError("fxpmath unavailable")

    monkeypatch.setattr(native, "import_module", missing)
    with pytest.raises(ModuleNotFoundError, match="fxpmath unavailable"):
        native.fixed_point_real_to_integer(0.5, 4, 3)

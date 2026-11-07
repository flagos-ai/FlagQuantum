"""Verify fixed-wire field primitives against the locked Kickmix binary."""

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.kickmix_field_linear import (
    kickmix_field_constant_scale_ir,
    kickmix_field_frobenius_ir,
    kickmix_field_monomial_scale_ir,
)
from tests.unit.test_field_linear import _source_type

pytestmark = pytest.mark.unit


def _gate_stream(text: str) -> list[tuple[str, tuple[int, ...]]]:
    return [
        (op.lower(), tuple(int(wire[1:]) for wire in operands))
        for op, *operands in map(str.split, text.splitlines())
        if op in {"CX", "SWAP"}
    ]


def _block(text: str, name: str) -> str:
    return text.split(f'INFO_TAG "@enter:{name}"\n', 1)[1].split(
        f'INFO_TAG "@exit:{name}"', 1
    )[0]


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "width,modulus,constant", [(2, 7, 2), (4, 19, 7), (8, 283, 83)]
)
def test_constant_scale_matches_pinned_kickmix(
    width: int, modulus: int, constant: int
) -> None:
    km = pytest.importorskip("kickmix")
    assert km.__version__ in {"0.1", "0.1.dev0"}
    builder = km.CircuitBuilder()
    target = builder.create_quantum_register(width, name="target")
    builder.gf2_imul(constant, target=target, field=km.GF2Field(width, modulus))
    wires = tuple(reversed(range(width)))
    expected = _gate_stream(str(builder.finish_circuit()).strip())
    actual = kickmix_field_constant_scale_ir(wires, constant, modulus=modulus)
    assert [(gate.name, gate.wires) for gate in actual] == expected


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width,modulus", [(2, 7), (3, 11), (4, 19), (8, 283)])
def test_linear_leaves_match_the_pinned_division_stream(
    width: int, modulus: int
) -> None:
    km = pytest.importorskip("kickmix")
    assert km.__version__ in {"0.1", "0.1.dev0"}
    builder = km.CircuitBuilder()
    left, right, target = (
        builder.create_quantum_register(width, name=name)
        for name in ("left", "right", "target")
    )
    builder.ixor_gf2_div(left, right, target=target, field=km.GF2Field(width, modulus))
    reference = str(builder.finish_circuit())
    work_wires = tuple(reversed(range(3 * width, 4 * width)))
    target_wires = tuple(reversed(range(2 * width, 3 * width)))
    product = _block(
        reference.split('INFO_TAG "@exit:gf_inverse_chain"', 1)[1], "gf_mul"
    )
    step = (width + 1) // 2
    for name, actual in (
        ("gf_ifrobenius", kickmix_field_frobenius_ir(work_wires, modulus=modulus)),
        (
            "gf_ifrobenius_adjoint",
            kickmix_field_frobenius_ir(work_wires, modulus=modulus, adjoint=True),
        ),
        (
            "gf_imul_x",
            kickmix_field_monomial_scale_ir(target_wires, step, modulus=modulus),
        ),
        (
            "gf_idiv_x",
            kickmix_field_monomial_scale_ir(target_wires, -step, modulus=modulus),
        ),
    ):
        source = _block(
            reference if name.startswith("gf_ifrobenius") else product, name
        )
        assert [(gate.name, gate.wires) for gate in actual] == _gate_stream(
            source.strip()
        )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width,modulus", [(3, 11), (4, 19), (8, 283)])
def test_fixed_wire_transforms_preserve_complex_amplitudes(
    width: int, modulus: int
) -> None:
    field = _source_type(width, modulus).gf_type
    target = tuple(range(width))
    indices = torch.arange(1 << width, dtype=torch.float64)
    state = torch.complex(indices + 1, 3 * indices - 2)
    state /= state.norm()
    for power in (-100, -2, 0, 1, 100):
        gates = kickmix_field_monomial_scale_ir(target, power, modulus=modulus)
        expected = torch.empty_like(state)
        factor = field(2) ** power
        for value in range(1 << width):
            expected[int(field(value) * factor)] = state[value]
        circuit = Circuit(width, inputs=state, dtype=torch.complex128)
        actual = circuit.append(FieldOperation(width, gates)).state().flatten()
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    forward = kickmix_field_frobenius_ir(target, 3, modulus=modulus)
    adjoint = kickmix_field_frobenius_ir(target, 3, modulus=modulus, adjoint=True)
    expected = torch.empty_like(state)
    for value in range(1 << width):
        expected[int(field(value) ** 8)] = state[value]
    circuit = Circuit(width, inputs=state, dtype=torch.complex128)
    transformed = circuit.append(FieldOperation(width, forward))
    torch.testing.assert_close(
        transformed.state().flatten(), expected, rtol=0, atol=1e-12
    )
    restored = transformed.append(FieldOperation(width, adjoint)).state().flatten()
    torch.testing.assert_close(restored, state, rtol=0, atol=1e-12)

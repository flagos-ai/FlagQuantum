"""Verify fixed-wire field primitives against the locked Kickmix binary."""

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.finite_field.binary_linear_map import (
    binary_linear_map_ir,
    field_constant_scale_ir,
    field_frobenius_ir,
    field_monomial_scale_ir,
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
    actual = field_constant_scale_ir(wires, constant, modulus=modulus)
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
        ("gf_ifrobenius", field_frobenius_ir(work_wires, modulus=modulus)),
        (
            "gf_ifrobenius_adjoint",
            field_frobenius_ir(work_wires, modulus=modulus, adjoint=True),
        ),
        (
            "gf_imul_x",
            field_monomial_scale_ir(target_wires, step, modulus=modulus),
        ),
        (
            "gf_idiv_x",
            field_monomial_scale_ir(target_wires, -step, modulus=modulus),
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
        gates = field_monomial_scale_ir(target, power, modulus=modulus)
        expected = torch.empty_like(state)
        factor = field(2) ** power
        for value in range(1 << width):
            expected[int(field(value) * factor)] = state[value]
        circuit = Circuit(width, inputs=state, dtype=torch.complex128)
        actual = circuit.append(FieldOperation(width, gates)).state().flatten()
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    forward = field_frobenius_ir(target, 3, modulus=modulus)
    adjoint = field_frobenius_ir(target, 3, modulus=modulus, adjoint=True)
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


@pytest.mark.parametrize(
    "columns,forward,adjoint",
    [
        (
            (6, 1, 4),
            "CX q0 q2\nSWAP q0 q1",
            "CX q1 q2\nSWAP q0 q1",
        ),
        (
            (9, 4, 5, 6),
            "CX q2 q0\nCX q2 q1\nCX q3 q1\nCX q0 q2\nSWAP q1 q3\nSWAP q2 q3",
            "CX q1 q0\nCX q2 q0\nCX q3 q0\nCX q2 q1\nCX q1 q2\nCX q0 q1\nSWAP q0 q3\nSWAP q1 q3\nSWAP q2 q3",
        ),
    ],
)
def test_arbitrary_linear_map_matches_pinned_cpp_fixtures(
    columns: tuple[int, ...], forward: str, adjoint: str
) -> None:
    # Generated by the locked gen_linear_map and gen_linear_map_adjoint
    # entry points with the target registered in increasing wire order.
    target = tuple(reversed(range(len(columns))))
    for inverse, expected in ((False, forward), (True, adjoint)):
        gates = binary_linear_map_ir(target, columns, adjoint=inverse)
        assert [(gate.name, gate.wires) for gate in gates] == _gate_stream(expected)


@pytest.mark.parametrize("columns", [(6, 1, 4), (9, 4, 5, 6), (28, 5, 19, 6, 14)])
def test_arbitrary_binary_map_and_adjoint_preserve_complex_amplitudes(
    columns: tuple[int, ...],
) -> None:
    width = len(columns)
    target = tuple(range(width))
    basis = torch.arange(1 << (width + 1), dtype=torch.float64)
    state = torch.complex(basis + 1, 3 * basis - 2)
    state /= state.norm()
    expected = torch.empty_like(state)
    for value in range(1 << width):
        mapped = 0
        for i, column in enumerate(columns):
            if value >> i & 1:
                mapped ^= column
        expected[2 * mapped : 2 * mapped + 2] = state[2 * value : 2 * value + 2]
    forward = binary_linear_map_ir(target, columns)
    adjoint = binary_linear_map_ir(target, columns, adjoint=True)
    circuit = Circuit(width + 1, inputs=state, dtype=torch.complex128)
    transformed = circuit.append(FieldOperation(width + 1, forward))
    torch.testing.assert_close(
        transformed.state().flatten(), expected, rtol=0, atol=1e-12
    )
    restored = transformed.append(FieldOperation(width + 1, adjoint))
    torch.testing.assert_close(restored.state().flatten(), state, rtol=0, atol=1e-12)


def test_arbitrary_binary_map_validates_shape_and_invertibility() -> None:
    assert binary_linear_map_ir((), (), adjoint=True) == ()
    assert binary_linear_map_ir(tuple(range(8)), tuple(1 << i for i in range(8))) == ()
    with pytest.raises(ValueError, match="square"):
        binary_linear_map_ir((0, 1), (1,))
    with pytest.raises(ValueError, match="square"):
        binary_linear_map_ir((0, 1), (1, 4))
    for adjoint in (False, True):
        with pytest.raises(ValueError, match="invertible"):
            binary_linear_map_ir((0, 1), (1, 1), adjoint=adjoint)

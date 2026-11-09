"""Verify the pinned controlled field product and its measured XOR inverse."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.finite_field.polynomial_product import (
    kickmix_controlled_field_product_xor_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_field_linear import _source_type
from tests.unit.test_kickmix_field_inverse import _apply_phase_certified_block
from tests.unit.test_kickmix_field_linear import _block

pytestmark = pytest.mark.unit


def _words(width: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(range(start, start + width))
        for start in (0, width, 2 * width, 3 * width + 1)
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [1, 2, 3, 4, 8, 17])
def test_controlled_product_and_unproduct_match_locked_stream(width: int) -> None:
    km = pytest.importorskip("kickmix")
    assert km.__version__ in {"0.1", "0.1.dev0"}
    field = km.GF2Field(width, modulus=0x11B if width == 8 else None)
    builder = km.CircuitBuilder()
    lhs, rhs, target = (
        builder.create_quantum_register(width, name=name)
        for name in ("lhs", "rhs", "target")
    )
    control = builder.create_quantum_register(1, name="control")[0]
    builder.ixor_gf2_mul(lhs, rhs, target=target, field=field, control=control)
    builder.del_gf2_mul(lhs, rhs, target=target, field=field, control=control)
    source = str(builder.finish_circuit())
    remaining = source.split('INFO_TAG "@exit:c_gf_mul"', 1)[1]
    words = _words(width)
    ir = kickmix_controlled_field_product_xor_ir(
        3 * width,
        *(word[::-1] for word in words),
        tuple(reversed(range(width))),
        modulus=field.modulus,
        parity_bit=width,
    )
    lowered = lower_reversible_phase(CircuitIR(4 * width + 1, ir))
    actual = encode_kmx(lowered).splitlines()[1:]
    for is_first, stream in ((True, source), (False, remaining)):
        lines = [
            line
            for line in _block(stream, "c_gf_mul").splitlines()
            if not line.startswith("INFO_TAG")
        ]
        assert lines[:width] == [f"R q{wire}" for wire in words[-1]]
        reference = lines[width:]
        first_toggle = next(
            (i for i, line in enumerate(reference) if line.startswith("BIT_INVERT ")),
            None,
        )
        if is_first and first_toggle is not None:
            reference.insert(first_toggle, f"BIT_STORE0 b{width}")
        assert actual == reference
    counts = Counter(op.opcode for op in lowered.operations)
    assert counts["hmr"] == width
    if width == 8:
        assert counts["ccx"] == 35
        assert counts["cz"] == 64
        assert lowered.n_bits == 9


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width,modulus", [(1, 3), (2, 7), (3, 11), (8, 283)])
def test_both_control_values_and_every_branch_phase(width: int, modulus: int) -> None:
    np = pytest.importorskip("numpy")
    field = _source_type(width, modulus).gf_type
    size = 1 << width
    lhs = np.repeat(np.arange(size), 2 * size)
    rhs = np.tile(np.repeat(np.arange(size), 2), size)
    control = np.tile(np.arange(2), size * size)
    target = lhs ^ rhs
    delta = np.asarray(field(lhs) * field(rhs)) * control
    wires = [
        (values >> shift) & 1
        for values in (lhs, rhs, target)
        for shift in reversed(range(width))
    ]
    wires.append(control.copy())
    wires.extend(np.zeros_like(control) for _ in range(width))
    original = np.array(wires)
    ir = kickmix_controlled_field_product_xor_ir(
        3 * width,
        *_words(width),
        tuple(reversed(range(width))),
        modulus=modulus,
        parity_bit=width,
    )
    _apply_phase_certified_block(np, wires, ir, width)
    value = sum(
        wires[wire] << shift for shift, wire in enumerate(_words(width)[2][::-1])
    )
    np.testing.assert_array_equal(value, target ^ delta)
    np.testing.assert_array_equal(wires[3 * width :], original[3 * width :])
    _apply_phase_certified_block(np, wires, ir, width)
    np.testing.assert_array_equal(wires, original)


@pytest.mark.parametrize("repetitions", [1, 2])
def test_append_preserves_control_coherence(repetitions: int) -> None:
    width, modulus = 2, 7
    total = 4 * width + 1
    basis = torch.arange(1 << (3 * width + 1))
    amplitudes = torch.complex((basis + 1).double(), (3 * basis - 5).double())
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << total, dtype=torch.complex128)
    state[:: 1 << width] = amplitudes
    circuit = DynamicCircuit(total, inputs=state, dtype=torch.complex128)
    ir = kickmix_controlled_field_product_xor_ir(
        3 * width, *_words(width), (1, 0), modulus=modulus, parity_bit=width
    )
    circuit.append(FieldOperation(total, ir * repetitions))
    expected = state.clone()
    if repetitions == 1:
        table = torch.tensor([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]])
        delta = table[basis >> 5, (basis >> 3) & 3] * (basis & 1)
        expected[:: 1 << width] = amplitudes[basis ^ (delta << 1)]
    result = run_dynamic(circuit, shots=32, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(32, -1), expected.expand(32, -1), rtol=0, atol=1e-12
    )

"""Check the locked quotient paths, including their measured uncomputation."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.finite_field.division_chain import (
    field_division_chain_erase_ir,
    field_division_chain_ir,
    field_division_erase_ir,
    field_division_workspace_size,
    field_division_xor_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_field_linear import _source_type
from tests.unit.test_kickmix_field_inverse import _apply_phase_certified_block
from tests.unit.test_kickmix_field_linear import _block

pytestmark = pytest.mark.unit


def _registers(
    width: int, *, little_endian_ids: bool = False
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    total = 3 * width + field_division_workspace_size(width)
    words = [tuple(range(start, start + width)) for start in range(0, total, width)]
    if little_endian_ids:
        words = [word[::-1] for word in words]
    return words[0], words[1], words[2], sum(words[3:], ())


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 8, 9, 17])
@pytest.mark.parametrize("retain", [True, False])
def test_division_matches_locked_stream(width: int, retain: bool) -> None:
    km = pytest.importorskip("kickmix")
    assert km.__version__ in {"0.1", "0.1.dev0"}
    field = km.GF2Field(width, modulus=0x11B if width == 8 else None)
    builder = km.CircuitBuilder()
    left, right, target = (
        builder.create_quantum_register(width, name=name)
        for name in ("left", "right", "target")
    )
    registers = _registers(width, little_endian_ids=True)
    workspace = registers[3]
    bits = tuple(reversed(range(width)))
    inverse_bits = bits if len(workspace) > width else bits[-1:]
    if retain:
        target, scaffold = builder.init_gf2_div_with_scaffold(
            left, right, target=target, field=field
        )
        assert len(scaffold) == len(workspace)
        builder.del_gf2_div_with_scaffold(
            left, right, target=target, scaffold=scaffold, field=field
        )
        forward = field_division_chain_ir(*registers, modulus=field.modulus)
        erase = field_division_chain_erase_ir(
            *registers, bits, modulus=field.modulus, parity_bit=width
        )
    else:
        builder.ixor_gf2_div(left, right, target=target, field=field)
        builder.del_gf2_div(left, right, target=target, field=field)
        forward = field_division_xor_ir(
            *registers, inverse_bits, modulus=field.modulus, parity_bit=width
        )
        erase = field_division_erase_ir(
            *registers, bits, modulus=field.modulus, parity_bit=width
        )

    reference_circuit = str(builder.finish_circuit())
    parity_seen = False
    for name, block in (("gf_div", forward), ("gf_undiv", erase)):
        source = [
            line
            for line in _block(reference_circuit, name).splitlines()
            if not line.startswith("INFO_TAG")
        ]
        resets = [line for line in source if line.startswith("R ")]
        forward_block = name == "gf_div"
        reset_words = workspace if forward_block or not retain else ()
        expected = [
            f"R q{wire}"
            for start in range(0, len(reset_words), width)
            for wire in reset_words[start : start + width][::-1]
        ]
        if (width - 1).bit_count() > 1 and (not forward_block or not retain):
            expected.extend(f"R q{wire}" for wire in workspace[:width][::-1])
        assert resets == expected
        reference = [line for line in source if not line.startswith("R ")]
        first_toggle = next(
            (i for i, line in enumerate(reference) if line.startswith("BIT_INVERT ")),
            None,
        )
        if first_toggle is not None and not parity_seen:
            reference.insert(first_toggle, f"BIT_STORE0 b{width}")
            parity_seen = True
        lowered = lower_reversible_phase(CircuitIR(3 * width + len(workspace), block))
        assert encode_kmx(lowered).splitlines()[1:] == reference
        counts = Counter(op.opcode for op in lowered.operations)
        if width == 8:
            assert counts["ccx"] == (135 if forward_block else 0 if retain else 108)
            assert counts["hmr"] == (
                0 if retain and forward_block else 40 if forward_block else 48
            )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width,modulus", [(1, 3), (2, 7), (3, 11), (4, 19), (8, 283)])
@pytest.mark.parametrize("retain", [True, False])
def test_quotient_and_every_measurement_phase(
    width: int, modulus: int, retain: bool
) -> None:
    np = pytest.importorskip("numpy")
    field = _source_type(width, modulus).gf_type
    size = 1 << width
    lhs = np.repeat(np.arange(size), size)
    rhs = np.tile(np.arange(size), size)
    expected = np.zeros_like(lhs)
    nonzero = rhs != 0
    expected[nonzero] = field(lhs[nonzero]) / field(rhs[nonzero])
    initial_target = np.zeros_like(lhs) if retain else lhs ^ rhs
    registers = _registers(width)
    target, workspace = registers[2:]
    bits = tuple(reversed(range(width)))
    if retain:
        forward = field_division_chain_ir(*registers, modulus=modulus)
        erase = field_division_chain_erase_ir(
            *registers, bits, modulus=modulus, parity_bit=width
        )
    else:
        inverse_bits = bits if len(workspace) > width else bits[-1:]
        forward = field_division_xor_ir(
            *registers, inverse_bits, modulus=modulus, parity_bit=width
        )
        erase = field_division_erase_ir(
            *registers, bits, modulus=modulus, parity_bit=width
        )
    wires = [
        (values >> shift) & 1
        for values in (lhs, rhs, initial_target)
        for shift in reversed(range(width))
    ]
    original_inputs = np.array(wires[: 2 * width])
    wires.extend(np.zeros_like(lhs) for _ in workspace)
    _apply_phase_certified_block(np, wires, forward, width)
    actual = sum(wires[wire] << shift for shift, wire in enumerate(target[::-1]))
    np.testing.assert_array_equal(actual, expected ^ initial_target)
    if not retain:
        np.testing.assert_array_equal(wires[3 * width :], 0)
        for shift, wire in enumerate(target[::-1]):
            wires[wire] ^= (initial_target >> shift) & 1
    _apply_phase_certified_block(np, wires, erase, width)
    np.testing.assert_array_equal(wires[: 2 * width], original_inputs)
    np.testing.assert_array_equal(wires[2 * width :], 0)


@pytest.mark.parametrize("shared_inputs", [False, True])
def test_append_preserves_complex_amplitudes(shared_inputs: bool) -> None:
    width, modulus = 2, 7
    left, right, target, workspace = _registers(width)
    total = 3 * width + len(workspace)
    if shared_inputs:
        right = left
    registers = left, right, target, workspace
    forward = field_division_chain_ir(*registers, modulus=modulus)
    erase = field_division_chain_erase_ir(
        *registers, (1, 0), modulus=modulus, parity_bit=2
    )
    state = torch.zeros(1 << total, dtype=torch.complex128)
    basis = torch.arange(1 << (2 * width), dtype=torch.float64)
    amplitudes = torch.complex(basis + 1, 3 * basis - 5)
    state[:: 1 << (total - 2 * width)] = amplitudes / amplitudes.norm()
    circuit = DynamicCircuit(total, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(total, (*forward, *erase)))
    result = run_dynamic(circuit, shots=32, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(32, -1), state.expand(32, -1), rtol=0, atol=1e-12
    )


def test_division_validates_workspace_ownership() -> None:
    with pytest.raises(ValueError, match="workspace"):
        field_division_chain_ir((0, 1), (2, 3), (4, 5), (6,), modulus=7)
    with pytest.raises(ValueError, match="distinct"):
        field_division_chain_ir((0, 1), (2, 3), (4, 5), (3, 6), modulus=7)

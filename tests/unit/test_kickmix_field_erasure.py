"""Check Kickmix product erasure and every conditional phase."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.kmx import decode_kmx, encode_kmx
from flagquantum.compiler.lowering.reversible_phase import lower_reversible_phase
from flagquantum.compiler.registers.kickmix_field_erasure import (
    kickmix_field_product_erase_ir,
    kickmix_field_product_phase_ir,
)
from flagquantum.compiler.registers.kickmix_field_product import (
    kickmix_field_product_xor_ir,
)
from flagquantum.core.ir import CircuitIR
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_field_linear import _source_type
from tests.unit.test_kickmix_field_linear import _block

pytestmark = pytest.mark.unit


def _registers(width: int) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(range(width)),
        tuple(range(width, 2 * width)),
        tuple(range(2 * width, 3 * width)),
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "width,modulus",
    [
        (1, None),
        (2, None),
        (3, None),
        (4, None),
        (5, None),
        (8, None),
        (8, 0x11B),
        (17, None),
    ],
)
def test_erase_matches_pinned_hmr_phase_and_bit_lifetime(
    width: int, modulus: int | None
) -> None:
    km = pytest.importorskip("kickmix")
    assert km.__version__ in {"0.1", "0.1.dev0"}
    field = km.GF2Field(width, modulus=modulus)
    builder = km.CircuitBuilder()
    left, right, target = (
        builder.create_quantum_register(width, name=name)
        for name in ("left", "right", "target")
    )
    builder.del_gf2_mul(left, right, target=target, field=field)
    reference = [
        line
        for line in _block(str(builder.finish_circuit()), "gf_unmul").splitlines()
        if not line.startswith("INFO_TAG")
    ]
    a, b, result = (word[::-1] for word in _registers(width))
    slots = tuple(reversed(range(width)))
    ir = kickmix_field_product_erase_ir(
        a, b, result, slots, modulus=field.modulus, parity_bit=width
    )
    program = lower_reversible_phase(CircuitIR(3 * width, ir))
    encoded = encode_kmx(program)
    actual = encoded.splitlines()[1:]
    # The source's first RAII allocation starts at zero implicitly.
    first_toggle = next(
        (i for i, line in enumerate(reference) if line.startswith("BIT_INVERT ")), None
    )
    if first_toggle is not None:
        reference.insert(first_toggle, f"BIT_STORE0 b{width}")
    assert actual == reference
    assert decode_kmx(encoded) == program
    counts = Counter(op.opcode for op in program.operations)
    assert counts["hmr"] == width and counts["cz"] == width**2
    assert program.n_bits == width + int(first_toggle is not None)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width,modulus", [(1, 3), (2, 7), (3, 11), (5, 37), (8, 283)])
def test_erase_cancels_every_branch_for_all_field_inputs(
    width: int, modulus: int
) -> None:
    np = pytest.importorskip("numpy")
    field = _source_type(width, modulus).gf_type
    size = 1 << width
    a = np.repeat(np.arange(size), size)
    b = np.tile(np.arange(size), size)
    product = np.asarray(field(a) * field(b), dtype=np.int64)
    left, right, target = _registers(width)
    wires = [
        (value >> shift) & 1
        for value in (a, b, product)
        for shift in reversed(range(width))
    ]
    original = [values.copy() for values in wires[: 2 * width]]
    ir = kickmix_field_product_erase_ir(
        left, right, target, tuple(range(width)), modulus=modulus, parity_bit=width
    )
    operations = lower_reversible_phase(CircuitIR(3 * width, ir)).operations

    # Each bit of the accumulated mask tracks one independent measurement
    # outcome's contribution to the phase on every computational input.
    phase = np.zeros(size**2, dtype=np.int64)
    slots = {}
    for op in operations:
        if op.opcode == "hmr":
            (wire,) = op.qubits
            mask = 1 << op.result_bit
            slots[op.result_bit] = mask
            phase ^= wires[wire] * mask
            wires[wire] = np.zeros_like(phase)
        elif op.opcode == "cz":
            x, y = op.qubits
            phase ^= (wires[x] & wires[y]) * slots[op.condition_bit]
        elif op.opcode == "bit_store0":
            slots[op.result_bit] = 0
        else:
            assert op.opcode == "bit_invert"
            slots[op.result_bit] ^= slots[op.condition_bit]
    np.testing.assert_array_equal(phase, 0)
    np.testing.assert_array_equal(wires[: 2 * width], original)
    np.testing.assert_array_equal(wires[2 * width :], 0)
    assert slots.get(width, 0) == 0


@pytest.mark.parametrize("width,modulus", [(2, 7), (3, 11)])
def test_dynamic_roundtrip_restores_complex_input_in_each_branch(
    width: int, modulus: int
) -> None:
    left, right, target = _registers(width)
    bits = tuple(range(width))
    gates = (
        *kickmix_field_product_xor_ir(left, right, target, modulus=modulus),
        *kickmix_field_product_erase_ir(
            left, right, target, bits, modulus=modulus, parity_bit=width
        ),
    )
    state = torch.zeros(1 << (3 * width), dtype=torch.complex128)
    basis = torch.arange(1 << (2 * width), dtype=torch.float64)
    amplitudes = torch.complex(basis + 1, 3 * basis - 5)
    state[:: 1 << width] = amplitudes / amplitudes.norm()
    circuit = DynamicCircuit(3 * width, inputs=state, dtype=torch.complex128)
    circuit.append(FieldOperation(3 * width, gates))
    result = run_dynamic(circuit, shots=128, seed=41, strategy="trajectory")
    assert len(set(map(tuple, result.classical_bits[:, :width].tolist()))) == 1 << width
    assert torch.all(result.classical_bits[:, width] == 0)
    torch.testing.assert_close(
        result.final_states.reshape(128, -1), state.expand(128, -1), rtol=0, atol=1e-12
    )


def test_parity_slot_is_only_required_for_multi_bit_columns() -> None:
    assert len(kickmix_field_product_phase_ir((0,), (1,), (0,), modulus=3)) == 1
    with pytest.raises(ValueError, match="needs a classical parity slot"):
        kickmix_field_product_phase_ir((0, 1), (2, 3), (0, 1), modulus=7)
    with pytest.raises(ValueError, match="distinct"):
        kickmix_field_product_phase_ir((0, 1), (2, 3), (0, 1), modulus=7, parity_bit=0)

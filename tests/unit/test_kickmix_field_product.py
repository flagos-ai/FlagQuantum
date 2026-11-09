"""Compare fixed-wire product accumulation with the pinned Kickmix generator."""

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.finite_field.polynomial_product import (
    kickmix_field_product_xor_ir,
)
from tests.unit.test_field_linear import _source_type
from tests.unit.test_kickmix_field_linear import _block

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 6, 8, 9, 16, 17])
def test_product_xor_matches_locked_full_gate_stream(width: int) -> None:
    km = pytest.importorskip("kickmix")
    assert km.__version__ in {"0.1", "0.1.dev0"}
    field = km.GF2Field(width)
    builder = km.CircuitBuilder()
    left, right, target = (
        builder.create_quantum_register(width, name=name)
        for name in ("left", "right", "target")
    )
    builder.ixor_gf2_mul(left, right, target=target, field=field)
    reference = []
    for line in _block(str(builder.finish_circuit()), "gf_mul").splitlines():
        opcode, *operands = line.split()
        if opcode == "INFO_TAG":
            continue
        assert opcode in {"CX", "CCX", "SWAP"}
        reference.append((opcode.lower(), tuple(int(wire[1:]) for wire in operands)))

    left_wires, right_wires, target_wires = (
        tuple(reversed(range(i * width, (i + 1) * width))) for i in range(3)
    )
    gates = kickmix_field_product_xor_ir(
        left_wires, right_wires, target_wires, modulus=field.modulus
    )
    assert [(gate.name, gate.wires) for gate in gates] == reference
    assert all(wire < 3 * width for gate in gates for wire in gate.wires)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width,modulus", [(1, 3), (3, 11), (4, 19)])
def test_product_xor_into_arbitrary_target_preserves_complex_amplitudes(
    width: int, modulus: int
) -> None:
    field = _source_type(width, modulus).gf_type
    n_qubits = 3 * width + 1
    left = tuple(range(width))
    right = tuple(range(width, 2 * width))
    target = tuple(range(2 * width, 3 * width))
    basis = torch.arange(1 << n_qubits)
    state = torch.complex((basis + 1).double(), (3 * basis - 2).double())
    state /= state.norm()
    mask = (1 << width) - 1
    a = field((basis >> (2 * width + 1)).numpy())
    b = field(((basis >> (width + 1)) & mask).numpy())
    delta = torch.tensor([int(value) for value in a * b]) << 1
    expected = state[basis ^ delta]
    circuit = Circuit(n_qubits, inputs=state, dtype=torch.complex128)
    operation = FieldOperation(
        n_qubits, kickmix_field_product_xor_ir(left, right, target, modulus=modulus)
    )
    result = circuit.append(operation)
    torch.testing.assert_close(result.state().flatten(), expected, rtol=0, atol=1e-12)
    torch.testing.assert_close(
        result.append(operation).state().flatten(), state, rtol=0, atol=1e-12
    )


def test_product_xor_rejects_aliasing_and_mismatched_width() -> None:
    with pytest.raises(ValueError, match="distinct"):
        kickmix_field_product_xor_ir((0, 1), (0, 2), (3, 4), modulus=7)
    with pytest.raises(ValueError, match="equal"):
        kickmix_field_product_xor_ir((0,), (1, 2), (3, 4), modulus=7)

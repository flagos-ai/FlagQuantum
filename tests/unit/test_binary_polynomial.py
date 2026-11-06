"""Pinned polynomial networks on arbitrary accumulators and clean workspace."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.polynomial import (
    binary_polynomial_product_ir,
    binary_polynomial_workspace_size,
)
from flagquantum.core.ir import Instruction
from tests.unit.test_field_multiplication import assert_gate_streams, reference_gates

pytestmark = pytest.mark.unit
CASES = [(n, k) for n in (1, 2, 3, 4, 5, 6, 8) for k in (None, *range(1, n + 2))]


def registers(
    width: int, shift: int | None
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    size = 2 * width - 1 if shift is None else 2 * shift + max(0, 2 * width - shift - 1)
    return (
        tuple(range(width)),
        tuple(range(width, 2 * width)),
        tuple(range(2 * width, 2 * width + size)),
        tuple(
            range(
                2 * width + size,
                2 * width + size + binary_polynomial_workspace_size(width),
            )
        ),
    )


@pytest.mark.parametrize("width,shift", CASES)
def test_literal_source_gate_dependencies_outputs_and_resources(
    width: int, shift: int | None
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import (
        BinaryPolynomialMultiplication,
        MultiplyPolyByOnePlusXk,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    left, right, target, work = registers(width, shift)
    gates = binary_polynomial_product_ir(left, right, target, work, shift=shift)
    source = (
        BinaryPolynomialMultiplication(width)
        if shift is None
        else MultiplyPolyByOnePlusXk(width, shift)
    )
    # Upstream QBit arrays are little-endian coefficients; our boundary is big-endian.
    expected, outputs = reference_gates(
        source, {"f": left[::-1], "g": right[::-1], "h": target[::-1]}, work
    )
    assert_gate_streams([(g.name, g.wires) for g in gates], expected)
    assert outputs == {"f": left[::-1], "g": right[::-1], "h": target[::-1]}
    model = get_cost_value(source, QECGatesCost())
    counts = Counter(g.name for g in gates)
    assert counts["ccx"] == model.toffoli
    assert counts["cx"] == model.clifford
    assert set(counts) <= {"cx", "ccx"}
    if work:
        # The literal source allocates padding even when no emitted gate touches it.
        with pytest.raises(IndexError):
            reference_gates(
                source,
                {"f": left[::-1], "g": right[::-1], "h": target[::-1]},
                work[:-1],
            )


def carryless_product(
    left: torch.Tensor, right: torch.Tensor, width: int
) -> torch.Tensor:
    result = torch.zeros_like(left)
    for bit in range(width):
        result ^= (left << bit) * ((right >> bit) & 1)
    return result


def execute_sparse(
    gates: tuple[Instruction, ...], indices: torch.Tensor, n_qubits: int
) -> torch.Tensor:
    result = indices.clone()
    for gate in gates:
        controls = torch.ones_like(result)
        for wire in gate.wires[:-1]:
            controls &= (result >> (n_qubits - wire - 1)) & 1
        result ^= controls << (n_qubits - gate.wires[-1] - 1)
    return result


@pytest.mark.parametrize("width,shift", CASES)
def test_all_input_pairs_nonzero_targets_sparse_amplitudes_and_clean_work(
    width: int, shift: int | None
) -> None:
    left, right, target, work = registers(width, shift)
    gates = binary_polynomial_product_ir(left, right, target, work, shift=shift)
    pairs = torch.arange(1 << (2 * width), dtype=torch.int64)
    a, b = pairs >> width, pairs & ((1 << width) - 1)
    product = carryless_product(a, b, width)
    if shift is not None:
        product ^= product << shift
    target_mask = (1 << len(target)) - 1
    # Every input pair has a distinct nonzero accumulator, including its top bit.
    accumulator = ((pairs * 19 + 7) & target_mask) | (1 << (len(target) - 1))
    indices = ((pairs << len(target)) | accumulator) << len(work)
    expected = ((pairs << len(target)) | (accumulator ^ product)) << len(work)
    n_qubits = 2 * width + len(target) + len(work)
    actual = execute_sparse(gates, indices, n_qubits)
    assert torch.equal(actual, expected)
    assert torch.equal(execute_sparse(gates, actual, n_qubits), indices)
    phase = torch.arange(len(pairs), dtype=torch.float64)
    amplitudes = torch.complex(phase + 1, phase.remainder(31) - 15)
    amplitudes /= amplitudes.norm()
    # Canonical sparse complex states compare amplitudes at their output basis keys.
    actual_order, expected_order = actual.argsort(), expected.argsort()
    assert torch.equal(actual[actual_order], expected[expected_order])
    torch.testing.assert_close(
        amplitudes[actual_order], amplitudes[expected_order], rtol=0, atol=0
    )
    if work:
        assert not torch.any(actual & ((1 << len(work)) - 1))


@pytest.mark.integration
@pytest.mark.parametrize(
    "width,shift", [(1, None), (2, None), (3, None), (2, 1), (3, 1)]
)
def test_append_full_complex_accumulator_and_round_trip(
    width: int, shift: int | None
) -> None:
    left, right, target, work = registers(width, shift)
    gates = binary_polynomial_product_ir(left, right, target, work, shift=shift)
    data_bits = 2 * width + len(target)
    values = torch.arange(1 << data_bits, dtype=torch.int64)
    a = values >> (width + len(target))
    b = (values >> len(target)) & ((1 << width) - 1)
    product = carryless_product(a, b, width)
    if shift is not None:
        product ^= product << shift
    mapped = values ^ product
    coeff = values.to(torch.float64)
    amplitudes = torch.complex(coeff + 1, 3 * coeff - 11)
    amplitudes /= amplitudes.norm()
    n_qubits = data_bits + len(work)
    state = torch.zeros(1 << n_qubits, dtype=torch.complex128)
    state[values << len(work)] = amplitudes
    expected = torch.zeros_like(state)
    expected[mapped << len(work)] = amplitudes
    operation = FieldOperation(n_qubits, gates)
    circuit = Circuit(n_qubits, inputs=state, dtype=torch.complex128).append(operation)
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=1e-12)
    torch.testing.assert_close(
        circuit.append(operation).state().flatten(), state, rtol=0, atol=1e-12
    )


def test_validation_and_minimum_workspace() -> None:
    for bad in (0, -1, True):
        with pytest.raises(ValueError, match="positive concrete"):
            binary_polynomial_workspace_size(bad)
    left, right, target, work = registers(3, None)
    assert len(work) == 1
    with pytest.raises(ValueError, match="clean work"):
        binary_polynomial_product_ir(left, right, target, ())
    with pytest.raises(ValueError, match="equal"):
        binary_polynomial_product_ir(left, right[:-1], target, work)
    with pytest.raises(ValueError, match="target requires"):
        binary_polynomial_product_ir(left, right, target[:-1], work)
    with pytest.raises(ValueError, match="distinct"):
        binary_polynomial_product_ir(left, right, target, left)
    for bad in (0, -1, 5, True):
        with pytest.raises(ValueError, match="shift"):
            binary_polynomial_product_ir(left, right, target, work, shift=bad)

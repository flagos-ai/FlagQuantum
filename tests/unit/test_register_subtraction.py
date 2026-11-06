"""Pinned subtraction directions, sign extension, phase and clean work."""

from collections import Counter
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.subtraction import subtract_from_ir, subtract_ir
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_register_constant_addition import reference_arithmetic_gates

pytestmark = pytest.mark.unit
WIDTHS = [(1, 1), (2, 2), (3, 3), (5, 5), (8, 8), (1, 3), (2, 5), (3, 8)]


def registers(m: int, n: int) -> tuple[tuple[int, ...], ...]:
    return (
        tuple(range(m)),
        tuple(range(m, m + n)),
        tuple(range(m + n, 2 * n)),
        tuple(range(2 * n, 3 * n - 1)),
        tuple(range(40, 40 + n - 1)),
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("m,n", WIDTHS)
@pytest.mark.parametrize("dtype", ["unsigned", "signed", "montgomery"])
def test_subtract_source_gates_and_resources(m: int, n: int, dtype: str) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic.subtraction import Subtract
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    cls = {"unsigned": QUInt, "signed": QInt, "montgomery": QMontgomeryUInt}[dtype]
    a, b, prefix, carry, bits = registers(m, n)
    source = Subtract(cls(m), cls(n))
    actual = subtract_ir(a, b, prefix, carry, bits, signed=dtype == "signed")
    expected = reference_arithmetic_gates(
        source, {"a": a, "b": b}, (*prefix, *carry), dict(zip(carry, bits, strict=True))
    )
    assert_source(actual, expected)
    counts = Counter(g.name for g in actual)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * (n - 1)
    assert counts["measure"] == model.measurement == n - 1
    assert counts["x"] == 3 * n


def assert_source(
    actual: tuple[Instruction, ...], expected: tuple[Instruction, ...]
) -> None:
    def key(gate: Instruction) -> tuple[Any, ...]:
        return gate.name, gate.wires, tuple(sorted(gate.metadata.items()))

    assert Counter(map(key, actual)) == Counter(map(key, expected))
    for wire in {w for g in expected for w in g.wires}:
        assert [key(g) for g in actual if wire in g.wires] == [
            key(g) for g in expected if wire in g.wires
        ]


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("dtype", ["unsigned", "signed", "montgomery"])
def test_subtract_from_source_gates_and_resources(width: int, dtype: str) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic.subtraction import SubtractFrom
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    cls = {"unsigned": QUInt, "signed": QInt, "montgomery": QMontgomeryUInt}[dtype]
    a, b, _, carry, bits = registers(width, width)
    source = SubtractFrom(cls(width))
    actual = subtract_from_ir(a, b, carry, bits)
    expected = reference_arithmetic_gates(
        source, {"a": a, "b": b}, carry, dict(zip(carry, bits, strict=True))
    )
    assert_source(actual, expected)
    counts = Counter(g.name for g in actual)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * (width - 1)
    assert counts["measure"] == model.measurement == width - 1
    assert counts["x"] == 2 * width


@pytest.mark.integration
@pytest.mark.parametrize(
    "m,n,signed,from_right",
    [
        (1, 1, False, False),
        (2, 2, False, False),
        (3, 3, False, False),
        (1, 3, False, False),
        (1, 3, True, False),
        (2, 3, True, False),
        (1, 1, False, True),
        (2, 2, False, True),
        (3, 3, False, True),
    ],
)
def test_all_entangled_complex_amplitudes_work_cleanup_and_round_trip(
    m: int, n: int, signed: bool, from_right: bool
) -> None:
    a, b, prefix, carry, bits = registers(m, n)
    gates = (
        subtract_from_ir(a, b, carry, bits)
        if from_right
        else subtract_ir(a, b, prefix, carry, bits, signed=signed)
    )
    work = len(prefix) + len(carry)
    n_qubits = m + n + work + 1
    values = torch.arange(1 << (m + n + 1), dtype=torch.int64)
    spectator = values & 1
    pairs = values >> 1
    left, right = pairs >> n, pairs & ((1 << n) - 1)
    extended = left - (((left >> (m - 1)) & 1) << m) if signed else left
    output = (right - left if from_right else extended - right) % (1 << n)
    coeff = values.to(torch.float64)
    amplitudes = torch.complex(coeff + 1, 3 * coeff - 5)
    amplitudes /= amplitudes.norm()
    indices = (pairs << (work + 1)) | spectator
    state = torch.zeros(1 << n_qubits, dtype=torch.complex128)
    state[indices] = amplitudes
    expected = torch.zeros_like(state)
    expected[(((left << n) | output) << (work + 1)) | spectator] = amplitudes
    circuit = DynamicCircuit(n_qubits, inputs=state, dtype=torch.complex128).append(
        FieldOperation(n_qubits, gates)
    )
    result = run_dynamic(circuit, shots=64, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(64, -1), expected.expand(64, -1), rtol=0, atol=1e-12
    )
    # Subtract is an involution; SubtractFrom is inverted by the already validated Add.
    if from_right:
        from flagquantum.compiler.registers.arithmetic import add_ir

        inverse = add_ir(a, b, carry, bits)
    else:
        inverse = gates
    restored = run_dynamic(
        circuit.append(FieldOperation(n_qubits, inverse)),
        shots=64,
        seed=41,
        strategy="trajectory",
    )
    torch.testing.assert_close(
        restored.final_states.reshape(64, -1), state.expand(64, -1), rtol=0, atol=1e-12
    )
    if n > 1:
        outcomes = result.classical_bits[:, 40 : 40 + n - 1].tolist()
        assert len(set(map(tuple, outcomes))) == 1 << (n - 1)


def test_invalid_subtraction_bindings() -> None:
    a, b, prefix, carry, bits = registers(2, 3)
    with pytest.raises(ValueError, match="prefix"):
        subtract_ir(a, b, (), carry, bits)
    with pytest.raises(ValueError, match="distinct"):
        subtract_ir(a, b, a[:1], carry, bits)
    with pytest.raises(ValueError, match="classical slots"):
        subtract_ir(a, b, prefix, carry, (-1, 2))
    with pytest.raises(ValueError, match="boolean"):
        subtract_ir(a, b, prefix, carry, bits, signed=1)
    with pytest.raises(ValueError, match="prefix"):
        subtract_from_ir(a, b, carry, bits)
    with pytest.raises(ValueError, match="clean carry"):
        subtract_ir(a, b, prefix, (), ())

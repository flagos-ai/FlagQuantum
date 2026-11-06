"""Pinned controlled greater-than, including signed boundary amplitudes."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.controlled_comparison import (
    controlled_greater_ir,
    controlled_greater_requirements,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("dtype", ["unsigned", "signed", "montgomery"])
@pytest.mark.parametrize("cv", [0, 1])
def test_controlled_greater_unmodified_source_and_model(
    n: int, dtype: str, cv: int
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic.comparison import CLinearDepthGreaterThan
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    cls = {"unsigned": QUInt, "signed": QInt, "montgomery": QMontgomeryUInt}[dtype]
    q, c = controlled_greater_requirements(n)
    a = tuple(range(1, n + 1))
    b = tuple(range(n + 1, 2 * n + 1))
    t = 2 * n + 1
    w = tuple(range(t + 1, t + q + 1))
    bits = tuple(range(40, 40 + c))
    actual = controlled_greater_ir(
        0, a, b, t, w, bits, signed=dtype == "signed", control_value=cv
    )
    source = CLinearDepthGreaterThan(cls(n), cv=cv)
    measured = {
        g.wires[0]: g.metadata["classical_bit"] for g in actual if g.name == "measure"
    }
    # Flattened Bloq allocation discovers sum bits least-significant first;
    # bind those names to the caller's big-endian sum register.
    expected = reference_arithmetic_gates(
        source,
        {"ctrl": (0,), "a": a, "b": b, "target": (t,)},
        (*w[:2], *w[2:-1][::-1], *w[-1:]),
        measured,
    )
    assert_source(actual, expected)
    counts = Counter(g.name for g in actual)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * (n + 2)
    assert counts["measure"] == model.measurement == n + 2


@pytest.mark.integration
@pytest.mark.parametrize(
    "n,signed,cv",
    [(n, s, cv) for n in (1, 2) for s in (False, True) for cv in (0, 1)]
    + [(3, False, 1), (3, True, 1)],
)
def test_all_control_input_target_amplitudes_signed_boundaries_and_cleanup(
    n: int, signed: bool, cv: int
) -> None:
    q, c = controlled_greater_requirements(n)
    a = tuple(range(1, n + 1))
    b = tuple(range(n + 1, 2 * n + 1))
    t = 2 * n + 1
    w = tuple(range(t + 1, t + q + 1))
    bits = tuple(range(40, 40 + c))
    gates = controlled_greater_ir(0, a, b, t, w, bits, signed=signed, control_value=cv)
    data = torch.arange(1 << (2 * n + 2), dtype=torch.int64)
    control = data >> (2 * n + 1)
    left = (data >> (n + 1)) & ((1 << n) - 1)
    right = (data >> 1) & ((1 << n) - 1)
    z = data & 1
    av = left - ((left >> (n - 1)) << n) if signed else left
    bv = right - ((right >> (n - 1)) << n) if signed else right
    mapped = (data & ~1) | (z ^ ((control == cv) & (av > bv)))
    coeff = data.to(torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    n_qubits = 2 * n + 2 + q
    state = torch.zeros(1 << n_qubits, dtype=torch.complex128)
    state[data << q] = amps
    expected = torch.zeros_like(state)
    expected[mapped << q] = amps
    circuit = DynamicCircuit(n_qubits, inputs=state, dtype=torch.complex128).append(
        FieldOperation(n_qubits, gates)
    )
    result = run_dynamic(circuit, shots=256, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(256, -1),
        expected.expand(256, -1),
        rtol=0,
        atol=1e-12,
    )
    assert (
        len(set(map(tuple, result.classical_bits[:, 40 : 40 + c].tolist()))) == 1 << c
    )
    restored = run_dynamic(
        circuit.append(FieldOperation(n_qubits, gates)),
        shots=8,
        seed=41,
        strategy="trajectory",
    )
    torch.testing.assert_close(
        restored.final_states.reshape(8, -1), state.expand(8, -1), rtol=0, atol=1e-12
    )


def test_controlled_greater_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        controlled_greater_requirements(0)
    a = (1,)
    b = (2,)
    w = tuple(range(4, 10))
    bits = (40, 41, 42)
    with pytest.raises(ValueError, match="equal"):
        controlled_greater_ir(0, a, (2, 3), 10, w, bits)
    with pytest.raises(ValueError, match="boolean"):
        controlled_greater_ir(0, a, b, 3, w, bits, signed=1)
    with pytest.raises(ValueError, match="distinct"):
        controlled_greater_ir(0, a, b, 0, w, bits)
    with pytest.raises(ValueError, match="clean work"):
        controlled_greater_ir(0, a, b, 3, w[:-1], bits)
    with pytest.raises(ValueError, match="classical"):
        controlled_greater_ir(0, a, b, 3, w, (40, 40, 41))

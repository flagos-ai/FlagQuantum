"""Source equality calls compose the shared control ladder without rewriting."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.comparison.equality import (
    equality_constant_ir,
    equality_ir,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit
CASES = [
    (n, k)
    for n in (1, 2, 3, 5, 8)
    for k in sorted({0, 1, (1 << n) - 1, 83 if n == 8 else 1})
]


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width,value", CASES)
def test_constant_equality_source_gate_dependencies_and_model(
    width: int, value: int
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.arithmetic.comparison import EqualsAConstant
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    x = tuple(range(width))
    work = tuple(range(width + 1, 2 * width))
    bits = tuple(range(40, 40 + len(work)))
    source = EqualsAConstant(width, value)
    actual = equality_constant_ir(x, value, width, work, bits)
    expected = reference_arithmetic_gates(
        source, {"x": x, "target": (width,)}, work, dict(zip(work, bits, strict=True))
    )
    assert_source(actual, expected)
    counts = Counter(g.name for g in actual)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * (width - 1)
    assert counts["measure"] == model.measurement == width - 1


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("dtype", ["unsigned", "signed", "montgomery", "bits"])
def test_register_equality_source_gates_and_model(width: int, dtype: str) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QAny, QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic.comparison import Equals
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    cls = {
        "unsigned": QUInt,
        "signed": QInt,
        "montgomery": QMontgomeryUInt,
        "bits": QAny,
    }[dtype]
    a = tuple(range(width))
    b = tuple(range(width, 2 * width))
    target = 2 * width
    work = tuple(range(target + 1, 3 * width))
    bits = tuple(range(40, 40 + len(work)))
    source = Equals(cls(width))
    actual = equality_ir(a, b, target, work, bits)
    expected = reference_arithmetic_gates(
        source,
        {"x": a, "y": b, "target": (target,)},
        work,
        dict(zip(work, bits, strict=True)),
    )
    assert_source(actual, expected)
    counts = Counter(g.name for g in actual)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * (width - 1)
    assert counts["measure"] == model.measurement == width - 1
    assert counts["cx"] == 2 * width + 6 * (width - 1) + 1


@pytest.mark.integration
@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("constant", [None, 0, 1])
def test_all_complex_inputs_targets_spectator_and_clean_work(
    width: int, constant: int | None
) -> None:
    if constant is None:
        a = tuple(range(width))
        b = tuple(range(width, 2 * width))
        target = 2 * width
        work = tuple(range(target + 1, 3 * width))
        bits = tuple(range(40, 40 + len(work)))
        gates = equality_ir(a, b, target, work, bits)
        data_bits = 2 * width + 1
    else:
        a = tuple(range(width))
        target = width
        work = tuple(range(width + 1, 2 * width))
        bits = tuple(range(40, 40 + len(work)))
        gates = equality_constant_ir(a, constant, target, work, bits)
        data_bits = width + 1
    n_qubits = data_bits + len(work) + 1
    data = torch.arange(1 << (data_bits + 1), dtype=torch.int64)
    spectator = data & 1
    prefix = data >> 1
    values = prefix >> 1
    z = prefix & 1
    predicate = (
        ((values >> width) == (values & ((1 << width) - 1)))
        if constant is None
        else values == constant
    )
    mapped = (values << 1) | (z ^ predicate)
    coeff = data.to(torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << n_qubits, dtype=torch.complex128)
    state[(prefix << (len(work) + 1)) | spectator] = amps
    expected = torch.zeros_like(state)
    expected[(mapped << (len(work) + 1)) | spectator] = amps
    circuit = DynamicCircuit(n_qubits, inputs=state, dtype=torch.complex128).append(
        FieldOperation(n_qubits, gates)
    )
    result = run_dynamic(circuit, shots=128, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(128, -1),
        expected.expand(128, -1),
        rtol=0,
        atol=1e-12,
    )
    restored = run_dynamic(
        circuit.append(FieldOperation(n_qubits, gates)),
        shots=32,
        seed=41,
        strategy="trajectory",
    )
    torch.testing.assert_close(
        restored.final_states.reshape(32, -1), state.expand(32, -1), rtol=0, atol=1e-12
    )
    if bits:
        assert len(
            set(map(tuple, result.classical_bits[:, 40 : 40 + len(work)].tolist()))
        ) == 1 << len(work)


def test_equality_rejects_invalid_inputs() -> None:
    for value in (-1, 8, True):
        with pytest.raises(ValueError, match="constant"):
            equality_constant_ir((0, 1, 2), value, 3, (4, 5), (40, 41))
    with pytest.raises(ValueError, match="positive"):
        equality_ir((0,), (1, 2), 3, (), ())
    with pytest.raises(ValueError, match="distinct"):
        equality_ir((0,), (0,), 1, (), ())
    with pytest.raises(ValueError, match="clean work"):
        equality_ir((0, 1), (2, 3), 4, (), ())

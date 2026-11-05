"""Pinned integer negation, including phase, entanglement and clean work."""

from collections import Counter
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.arithmetic import negate_ir
from flagquantum.core.ir import Instruction
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_register_constant_addition import layout, source_gates

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("width", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("dtype", ["unsigned", "signed", "montgomery"])
def test_negation_unmodified_source_gates_and_cost(width: int, dtype: str) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic.negate import Negate
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    cls = {"unsigned": QUInt, "signed": QInt, "montgomery": QMontgomeryUInt}[dtype]
    source = Negate(cls(width))
    if width == 1 and dtype == "signed":
        # Upstream cannot load signed +1 in one bit; retain the counterexample.
        with pytest.raises(ValueError):
            source_gates(width, 1, source=source)
        return
    expected, _ = source_gates(width, 1, source=source)
    target, constant, carries, bits = layout(width)
    actual = negate_ir(target, constant, carries, bits)

    def key(gate: Instruction) -> tuple[Any, ...]:
        return gate.name, gate.wires, tuple(sorted(gate.metadata.items()))

    assert Counter(map(key, actual)) == Counter(map(key, expected))
    for wire in (*target, *constant, *carries):
        assert [key(g) for g in actual if wire in g.wires] == [
            key(g) for g in expected if wire in g.wires
        ]
    counts = Counter(g.name for g in actual)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * (width - 1)
    assert counts["measure"] == model.measurement == width - 1
    assert counts["reset"] == width - 1
    assert counts["x"] == width + 2
    assert not any(g.name == "swap" for g in actual)


@pytest.mark.integration
@pytest.mark.parametrize("width", [1, 2, 3, 4])
@pytest.mark.parametrize("repetitions", [1, 2])
def test_negation_all_complex_amplitudes_entangled_spectator_and_clean_work(
    width: int, repetitions: int
) -> None:
    target, constant, carries, bits = layout(width)
    gates = negate_ir(target, constant, carries, bits)
    work = len(constant) + len(carries)
    n_qubits = width + work + 1
    data = torch.arange(1 << (width + 1), dtype=torch.int64)
    x, spectator = data >> 1, data & 1
    phase = data.to(torch.float64)
    amplitudes = torch.complex(phase + 1, 3 * phase - 5)
    amplitudes /= amplitudes.norm()
    state = torch.zeros(1 << n_qubits, dtype=torch.complex128)
    state[(x << (work + 1)) | spectator] = amplitudes
    mapped = (-x) % (1 << width) if repetitions == 1 else x
    expected = torch.zeros_like(state)
    expected[(mapped << (work + 1)) | spectator] = amplitudes
    circuit = DynamicCircuit(n_qubits, inputs=state, dtype=torch.complex128)
    for _ in range(repetitions):
        circuit = circuit.append(FieldOperation(n_qubits, gates))
    result = run_dynamic(circuit, shots=64, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(64, -1), expected.expand(64, -1), rtol=0, atol=1e-12
    )
    if width > 1:
        outcomes = result.classical_bits[:, 40 : 40 + width - 1].tolist()
        assert len(set(map(tuple, outcomes))) == 1 << (width - 1)


def test_negation_inherits_strict_work_and_alias_validation() -> None:
    target, constant, carries, bits = layout(3)
    with pytest.raises(ValueError, match="distinct"):
        negate_ir(target, target, carries, bits)
    with pytest.raises(ValueError, match="clean carry"):
        negate_ir(target, constant, (), ())
    with pytest.raises(ValueError, match="equal nonempty"):
        negate_ir((), (), (), ())

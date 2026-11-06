"""Pinned controlled adder and shared measured And on coherent controls."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.and_compute import and_compute_ir, and_uncompute_ir
from flagquantum.compiler.registers.controlled_addition import controlled_add_ir
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit


def layout(m: int, n: int) -> tuple[tuple[int, ...], ...]:
    return (
        tuple(range(1, 1 + m)),
        tuple(range(1 + m, 1 + m + n)),
        tuple(range(1 + m + n, 1 + 2 * m + n)),
        tuple(range(1 + 2 * m + n, 2 * m + 2 * n)),
        tuple(range(40, 40 + m)),
        tuple(range(40 + m, 39 + m + n)),
    )


@pytest.mark.parametrize(
    "m,n", [(1, 1), (2, 2), (3, 3), (5, 5), (8, 8), (1, 3), (3, 8)]
)
@pytest.mark.parametrize("cv", [0, 1])
@pytest.mark.parametrize("dtype", ["unsigned", "signed", "montgomery"])
def test_controlled_addition_literal_source_gates_and_model(
    m: int, n: int, cv: int, dtype: str
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import QInt, QMontgomeryUInt, QUInt
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    cls = {"unsigned": QUInt, "signed": QInt, "montgomery": QMontgomeryUInt}[dtype]
    a, b, work, carry, wb, cb = layout(m, n)
    source = CAdd(cls(m), cls(n), cv=cv)
    actual = controlled_add_ir(0, a, b, work, carry, wb, cb, control_value=cv)
    expected = reference_arithmetic_gates(
        source,
        {"ctrl": (0,), "a": a, "b": b},
        (*work, *carry),
        dict(zip((*work, *carry), (*wb, *cb), strict=True)),
    )
    assert_source(actual, expected)
    counts = Counter(g.name for g in actual)
    model = get_cost_value(source, QECGatesCost())
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * (m + n - 1)
    assert counts["measure"] == model.measurement == m + n - 1
    assert counts["reset"] == m + n - 1
    assert counts["x"] == (4 * m if cv == 0 else 0)


@pytest.mark.parametrize("values", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_shared_and_literal_source_and_clean_coherent_erase(
    values: tuple[int, int],
) -> None:
    pytest.importorskip("qualtran")
    import numpy as np
    from qualtran import BloqBuilder, Signature
    from qualtran.bloqs.mcmt.and_bloq import And

    bb, inputs = BloqBuilder.from_signature(Signature.build(a=1, b=1))
    ctrl, target = bb.add(And(*values), ctrl=np.array([inputs["a"], inputs["b"]]))
    ctrl = bb.add(And(*values).adjoint(), ctrl=ctrl, target=target)
    source = bb.finalize(a=ctrl[0], b=ctrl[1])
    actual = (
        *and_compute_ir(0, 1, 2, values=values),
        *and_uncompute_ir(0, 1, 2, 40, values=values),
    )
    expected = reference_arithmetic_gates(source, {"a": (0,), "b": (1,)}, (2,), {2: 40})
    assert_source(actual, expected)
    state = torch.zeros(8, dtype=torch.complex128)
    amps = torch.tensor([1 + 2j, 3 - 1j, -4 + 5j, 2 - 7j], dtype=torch.complex128)
    amps /= amps.norm()
    state[::2] = amps
    expected_compute = torch.zeros_like(state)
    for controls in range(4):
        target = int((controls >> 1, controls & 1) == values)
        expected_compute[(controls << 1) | target] = amps[controls]
    computed = Circuit(3, inputs=state, dtype=torch.complex128).append(
        FieldOperation(3, and_compute_ir(0, 1, 2, values=values))
    )
    torch.testing.assert_close(
        computed.state().flatten(), expected_compute, rtol=0, atol=1e-12
    )

    circuit = DynamicCircuit(3, inputs=state, dtype=torch.complex128).append(
        FieldOperation(3, actual)
    )
    result = run_dynamic(circuit, shots=32, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(32, -1), state.expand(32, -1), rtol=0, atol=1e-12
    )
    assert set(result.classical_bits[:, 40].tolist()) == {0, 1}


@pytest.mark.integration
@pytest.mark.parametrize("m,n", [(1, 1), (2, 2), (3, 3), (1, 3), (2, 3)])
@pytest.mark.parametrize("cv", [0, 1])
def test_all_amplitudes_control_superposition_clean_work_and_inverse(
    m: int, n: int, cv: int
) -> None:
    a, b, work, carry, wb, cb = layout(m, n)
    gates = controlled_add_ir(0, a, b, work, carry, wb, cb, control_value=cv)
    # Invert the complete instrument by complementing b before and after another CAdd.
    from flagquantum.core.ir import Instruction

    flips = tuple(Instruction("x", (wire,)) for wire in b)
    inverse = (*flips, *gates, *flips)
    work_count = m + n - 1
    n_qubits = 1 + m + n + work_count + 1
    data = torch.arange(1 << (1 + m + n + 1), dtype=torch.int64)
    spectator = data & 1
    prefix = data >> 1
    control = prefix >> (m + n)
    left = (prefix >> n) & ((1 << m) - 1)
    right = prefix & ((1 << n) - 1)
    mapped = (right + left * (control == cv)) % (1 << n)
    coeff = data.to(torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << n_qubits, dtype=torch.complex128)
    state[(prefix << (work_count + 1)) | spectator] = amps
    expected = torch.zeros_like(state)
    expected[(((prefix >> n) << n | mapped) << (work_count + 1)) | spectator] = amps
    circuit = DynamicCircuit(n_qubits, inputs=state, dtype=torch.complex128).append(
        FieldOperation(n_qubits, gates)
    )
    result = run_dynamic(circuit, shots=512, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(512, -1),
        expected.expand(512, -1),
        rtol=0,
        atol=1e-12,
    )
    restored = run_dynamic(
        circuit.append(FieldOperation(n_qubits, inverse)),
        shots=64,
        seed=41,
        strategy="trajectory",
    )
    torch.testing.assert_close(
        restored.final_states.reshape(64, -1), state.expand(64, -1), rtol=0, atol=1e-12
    )
    assert (
        len(set(map(tuple, result.classical_bits[:, 40 : 40 + work_count].tolist())))
        == 1 << work_count
    )


def test_invalid_work_slots_polarities_and_aliases() -> None:
    a, b, w, c, wb, cb = layout(2, 3)
    for cv in (-1, 2, True):
        with pytest.raises(ValueError, match="control value"):
            controlled_add_ir(0, a, b, w, c, wb, cb, control_value=cv)
    with pytest.raises(ValueError, match="loaded wire"):
        controlled_add_ir(0, a, b, w[:-1], c, wb, cb)
    with pytest.raises(ValueError, match="distinct nonnegative quantum"):
        controlled_add_ir(a[0], a, b, w, c, wb, cb)
    with pytest.raises(ValueError, match="classical slots"):
        controlled_add_ir(0, a, b, w, c, wb, wb)
    with pytest.raises(ValueError, match="classical slots"):
        controlled_add_ir(0, a, b, w, c, (-1, 4), cb)
    with pytest.raises(ValueError, match="clean carry"):
        controlled_add_ir(0, a, b, w, (), wb, ())
    with pytest.raises(ValueError, match="distinct"):
        and_compute_ir(0, 0, 1)
    with pytest.raises(ValueError, match="control values"):
        and_compute_ir(0, 1, 2, values=(True, 1))
    with pytest.raises(ValueError, match="classical slot"):
        and_uncompute_ir(0, 1, 2, -1)


def test_pinned_narrow_signed_source_semantics_exception() -> None:
    pytest.importorskip("qualtran")
    from qualtran import QInt
    from qualtran.bloqs.arithmetic.controlled_addition import CAdd

    source = CAdd(QInt(1), QInt(2))
    # The source executable Add zero-extends the narrow bit pattern, while its
    # classical signed action interprets the same one-bit input as -1.
    classical = source.on_classical_vals(ctrl=1, a=-1, b=0)
    assert classical["b"] == -1
    a, b, work, carry, wb, cb = layout(1, 2)
    gates = controlled_add_ir(0, a, b, work, carry, wb, cb)
    state = torch.zeros(64, dtype=torch.complex128)
    state[48] = 1
    circuit = DynamicCircuit(6, inputs=state, dtype=torch.complex128).append(
        FieldOperation(6, gates)
    )
    result = run_dynamic(circuit, shots=16, seed=41, strategy="trajectory")
    expected = torch.zeros_like(state)
    expected[52] = 1
    torch.testing.assert_close(
        result.final_states.reshape(16, -1), expected.expand(16, -1), rtol=0, atol=1e-12
    )
    assert int(classical["b"]) % 4 != 1

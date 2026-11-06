"""Pinned controlled AddK without controlling the shared carry network."""

from collections import Counter

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.registers.arithmetic import add_constant_ir
from flagquantum.compiler.registers.controlled_addition import (
    controlled_add_constant_ir,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = pytest.mark.unit
CASES = [(n, k) for n in (1, 2, 3, 5, 8) for k in (0, 1, (1 << n) - 1, -3)]


def layout(width: int) -> tuple[tuple[int, ...], ...]:
    return (
        tuple(range(1, width + 1)),
        tuple(range(width + 1, 2 * width + 1)),
        tuple(range(2 * width + 1, 3 * width)),
        tuple(range(40, 39 + width)),
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("width,value", CASES)
@pytest.mark.parametrize("cv", [0, 1])
def test_controlled_constant_unmodified_source_gates_and_resource_model(
    width: int, value: int, cv: int
) -> None:
    pytest.importorskip("qualtran")
    from qualtran import CtrlSpec, QUInt
    from qualtran.bloqs.arithmetic import AddK
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    target, constant, carry, bits = layout(width)
    source = AddK(QUInt(width), value).controlled(ctrl_spec=CtrlSpec(cvs=cv))
    actual = controlled_add_constant_ir(
        0, target, value, constant, carry, bits, control_value=cv
    )
    expected = reference_arithmetic_gates(
        source,
        {"ctrl": (0,), "x": target},
        (*constant, *carry),
        dict(zip(carry, bits, strict=True)),
    )
    assert_source(actual, expected)
    model = get_cost_value(source, QECGatesCost())
    counts = Counter(g.name for g in actual)
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * (width - 1)
    assert counts["measure"] == model.measurement == width - 1
    assert counts["reset"] == width - 1
    assert counts["x"] == (2 if cv == 0 else 0)
    # The carry network is literally the same uncontrolled Add on either branch.
    plain = add_constant_ir(target, value, constant, carry, bits)
    assert [g for g in actual if g.name in {"t", "tdg", "measure", "reset", "cz"}] == [
        g for g in plain if g.name in {"t", "tdg", "measure", "reset", "cz"}
    ]


@pytest.mark.integration
@pytest.mark.parametrize(
    "width,value", [(n, k) for n in (1, 2, 3, 4) for k in (0, 1, -3)]
)
@pytest.mark.parametrize("cv", [0, 1])
def test_all_complex_control_target_spectator_amplitudes_and_clean_inverse(
    width: int, value: int, cv: int
) -> None:
    target, constant, carry, bits = layout(width)
    gates = controlled_add_constant_ir(
        0, target, value, constant, carry, bits, control_value=cv
    )
    work = 2 * width - 1
    n_qubits = 1 + width + work + 1
    values = torch.arange(1 << (width + 2), dtype=torch.int64)
    spectator = values & 1
    prefix = values >> 1
    control = prefix >> width
    x = prefix & ((1 << width) - 1)
    mapped = (x + value * (control == cv)) % (1 << width)
    coeff = values.to(torch.float64)
    amps = torch.complex(coeff + 1, 3 * coeff - 5)
    amps /= amps.norm()
    state = torch.zeros(1 << n_qubits, dtype=torch.complex128)
    state[(prefix << (work + 1)) | spectator] = amps
    expected = torch.zeros_like(state)
    expected[(((control << width) | mapped) << (work + 1)) | spectator] = amps
    circuit = DynamicCircuit(n_qubits, inputs=state, dtype=torch.complex128).append(
        FieldOperation(n_qubits, gates)
    )
    result = run_dynamic(circuit, shots=64, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(64, -1), expected.expand(64, -1), rtol=0, atol=1e-12
    )
    inverse = controlled_add_constant_ir(
        0, target, (-value) % (1 << width), constant, carry, bits, control_value=cv
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
    if width > 1:
        assert len(
            set(map(tuple, result.classical_bits[:, 40 : 40 + width - 1].tolist()))
        ) == 1 << (width - 1)


def test_controlled_constant_rejects_invalid_bindings() -> None:
    t, w, c, b = layout(3)
    for cv in (-1, 2, True):
        with pytest.raises(ValueError, match="control value"):
            controlled_add_constant_ir(0, t, 1, w, c, b, control_value=cv)
    for control in (-1, True):
        with pytest.raises(ValueError, match="control wire"):
            controlled_add_constant_ir(control, t, 1, w, c, b)
    with pytest.raises(ValueError, match="distinct"):
        controlled_add_constant_ir(t[0], t, 1, w, c, b)
    with pytest.raises(ValueError, match="Additive constant"):
        controlled_add_constant_ir(0, t, 8, w, c, b)
    with pytest.raises(ValueError, match="clean carry"):
        controlled_add_constant_ir(0, t, 1, w, (), ())

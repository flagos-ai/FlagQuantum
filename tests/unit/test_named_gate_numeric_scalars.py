"""Cirq scalar metadata becomes native literals before named-gate compilation."""

import pytest
import torch

from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import verify_program_record
from flagquantum.ecosystem.cirq._named_gate_definitions import _NamedGateDefinitions
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("kind", ("Rx", "XPow"))
def test_sdk_numpy_scalars_keep_exact_native_literal_and_complex_action(
    kind: str,
) -> None:
    cirq = pytest.importorskip("cirq")
    np = pytest.importorskip("numpy")
    value = np.float64(0.375)
    gate = cirq.rx(value) if kind == "Rx" else cirq.XPowGate(exponent=value)
    canonical = (
        cirq.rx(float(value)) if kind == "Rx" else cirq.XPowGate(exponent=float(value))
    )
    assert isinstance(gate._rads if kind == "Rx" else gate.exponent, np.generic)
    definitions = _NamedGateDefinitions()
    module = definitions._module(definitions[gate])
    expected_definitions = _NamedGateDefinitions()
    expected = expected_definitions._module(expected_definitions[canonical])
    assert module == expected and verify_program_record(module) == ()
    bound = bind_register_program(module, ((0,),), entry=module.functions[0].name)
    torch.testing.assert_close(
        logical_operator(bound.circuit, 1),
        torch.as_tensor(cirq.unitary(gate), dtype=torch.complex128),
        atol=2e-12,
        rtol=0,
    )


def test_float32_angle_precision_boundary_is_not_silently_widened() -> None:
    cirq = pytest.importorskip("cirq")
    np = pytest.importorskip("numpy")
    gate = cirq.rx(np.float32(0.375))
    canonical = cirq.rx(0.375)
    assert np.max(np.abs(cirq.unitary(gate) - cirq.unitary(canonical))) > 1e-10
    with pytest.raises(ValueError, match="real scalar"):
        _NamedGateDefinitions()[gate]

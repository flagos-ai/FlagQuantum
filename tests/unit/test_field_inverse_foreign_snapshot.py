"""Capture the complete original inverse factory and retain its source failures."""

import builtins
from collections import Counter
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.primitives.circuit_binding import (
    bind_primitive_circuit,
    restore_output_order_ir,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.ecosystem.cirq._component_snapshot import _snapshot
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.test_field_multiplication import assert_gate_streams, reference_gates

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("reversed_binding", (False, True))
def test_actual_inverse_factory_forward_full_complex_original_graph_and_logical_junk(
    reversed_binding: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cirq
    import numpy as np
    from qualtran.bloqs.gf_arithmetic.gf2_inverse import _gf16_inverse
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = _gf16_inverse.make()
    layout = tuple((reg.name, int(reg.total_bits())) for reg in source.signature)
    assert layout == (("x", 4), ("result", 4), ("junk", 4))
    snapshot, swaps = _snapshot(source, layout, local_ports=("result", "junk"))
    assert snapshot.n_wires == 12 and swaps == 12
    raw, outputs = reference_gates(
        source, {"x": (0, 1, 2, 3)}, tuple(range(4, 32)), repair_adjoint=False
    )
    locations = list(range(12))
    normalized: list[tuple[str, tuple[int, ...]]] = []
    for gate in snapshot.instructions:
        if gate.name == "swap":
            left, right = gate.wires
            locations[left], locations[right] = locations[right], locations[left]
        else:
            normalized.append((gate.name, tuple(locations[w] for w in gate.wires)))
    assert_gate_streams(normalized, raw)
    assert outputs == {
        "x": tuple(locations[:4]),
        "result": tuple(locations[4:8]),
        "junk": tuple(locations[8:12]),
    }
    counts = Counter(gate.name for gate in snapshot.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert counts == {"cx": model.clifford, "ccx": model.toffoli, "swap": 12}
    assert (counts["cx"], counts["ccx"]) == (104, 18)
    assert get_cost_value(source, QubitCount()) == 12

    # Logical junk is an original RIGHT output. All twelve physically used
    # wires are logical ports, with no private work or measurement events.
    n = 13
    wires = tuple(reversed(range(12))) if reversed_binding else tuple(range(12))

    def pack(value: int, spectator: int) -> int:
        return spectator | sum(
            ((value >> (3 - bit)) & 1) << (n - 1 - wire)
            for bit, wire in enumerate(wires[:4])
        )

    initial = torch.zeros(1 << n, dtype=torch.complex128)
    for value in range(16):
        for spectator in (0, 1):
            initial[pack(value, spectator)] = torch.tensor(
                complex(value + 1, 3 * spectator - value), dtype=torch.complex128
            )
    initial /= initial.norm()
    aliases = tuple(w for name, _ in layout for w in outputs[name])
    original_gates = tuple(Instruction(name, ws) for name, ws in raw)
    original_gates += restore_output_order_ir(aliases)
    original_gates = bind_primitive_circuit(CircuitIR(12, original_gates), wires)
    qubits = tuple(cirq.LineQubit.range(n))
    names = {"cx": cirq.CNOT, "ccx": cirq.CCX, "swap": cirq.SWAP}
    original_circuit = cirq.Circuit(
        names[gate.name].on(*(qubits[w] for w in gate.wires)) for gate in original_gates
    )
    expected = (
        cirq.Simulator(dtype=np.complex128)
        .simulate(original_circuit, qubit_order=qubits, initial_state=initial.numpy())
        .final_state_vector
    )
    expected_state = torch.tensor(expected, dtype=torch.complex128)
    original_import = builtins.__import__

    def owned_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq", "galois", "kickmix"}:
            raise AssertionError(f"Captured inverse execution imported {name}")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", owned_import)
        restored = CircuitIR.from_json(snapshot.to_json())
        bound = bind_primitive_circuit(restored, wires)
        actual = Circuit(n, inputs=initial, dtype=torch.complex128).append(
            FieldOperation(n, bound)
        )
        torch.testing.assert_close(
            actual.state().flatten(), expected_state, atol=3e-12, rtol=0
        )

    # The exact original linear-adjoint defect also affects this root. Keep
    # its mathematical callback separate from the unchanged captured recipe.
    mathematical = torch.zeros_like(initial)
    for value in range(16):
        result = source.on_classical_vals(x=source.qgf.gf_type(value))
        words = (int(result["x"]), int(result["result"]), int(result["junk"][0]))
        destination = sum(
            ((word >> (3 - bit)) & 1) << (n - 1 - wires[4 * port + bit])
            for port, word in enumerate(words)
            for bit in range(4)
        )
        for spectator in (0, 1):
            mathematical[destination | spectator] = initial[pack(value, spectator)]
    assert not torch.allclose(expected_state, mathematical)


def test_complete_inverse_factory_failures_original_adjoint_and_codec() -> None:
    from qualtran import BloqExample, DecomposeTypeError
    from qualtran.bloqs.gf_arithmetic import gf2_inverse as source
    from qualtran.serialization.bloq import bloqs_to_proto

    examples = {
        name for name, value in vars(source).items() if isinstance(value, BloqExample)
    }
    assert examples == {"_gf16_inverse", "_gf2_inverse_symbolic"}
    original = source._gf16_inverse.make()
    with pytest.raises(TypeError, match="QGF"):
        bloqs_to_proto(original, max_depth=0)
    # Original backward construction calls measured product erasure. Its generic
    # Cirq wrapper fails at the already independently witnessed MeasureX arity.
    with pytest.raises(ValueError, match="MeasureX.*Expected 1 qubits"):
        import_register_emitter(original, local_ports=("result", "junk"))
    symbolic = source._gf2_inverse_symbolic.make()
    with pytest.raises(DecomposeTypeError, match="symbolic"):
        symbolic.decompose_bloq()
    with pytest.raises(ValueError, match="concrete positive shapes"):
        import_register_emitter(symbolic, local_ports=("result", "junk"))

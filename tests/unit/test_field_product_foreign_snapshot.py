"""Capture the actual binary-field product without changing its broken codec."""

import builtins
from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.layout.partition import partition_register
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.test_field_multiplication import assert_gate_streams, reference_gates

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width,accumulate", tuple(product((2, 3, 4), (False, True))))
def test_actual_field_product_snapshot_full_complex_append_and_adjoint(
    width: int, accumulate: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qualtran import QGF
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import (
        GF2Multiplication,
        _gf16_multiplication,
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.serialization.bloq import bloqs_to_proto

    source = (
        _gf16_multiplication.make()
        if width == 4 and accumulate
        else GF2Multiplication(QGF(2, width), plus_equal_prod=accumulate)
    )
    with pytest.raises(TypeError, match="QGF"):
        bloqs_to_proto(source, max_depth=0)
    emitter = import_register_emitter(
        source, local_ports=() if accumulate else ("result",)
    )
    assert emitter.layout == (("x", width), ("y", width), ("result", width))
    assert emitter.requirements() == (0, 0)
    assert emitter.binding_swaps == (0, 0)
    logical = 3 * width
    ports = partition_register(tuple(reversed(range(logical))), emitter.layout)
    forward = emitter(ports, (), (), False)
    backward = emitter(ports, (), (), True)
    for direction, stream in ((source, forward), (source.adjoint(), backward)):
        consume = accumulate or direction is not source
        inputs = ports if consume else {"x": ports["x"], "y": ports["y"]}
        expected_stream, outputs = reference_gates(
            direction, inputs, () if consume else ports["result"], repair_adjoint=False
        )
        assert outputs == (
            ports
            if accumulate or direction is source
            else {"x": ports["x"], "y": ports["y"]}
        )
        assert_gate_streams(
            [(gate.name, gate.wires) for gate in stream], expected_stream
        )
    counts = Counter(gate.name for gate in forward)
    assert set(counts) <= {"cx", "ccx"}
    declared = get_cost_value(source, QECGatesCost())
    assert counts["ccx"] == declared.toffoli == width**2
    assert counts["cx"] == declared.clifford

    # A RIGHT output is initially clean. A THRU accumulator admits every word.
    field = source.qgf.gf_type
    n = logical + 1
    initial = torch.zeros(1 << n, dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    target_values = range(1 << width) if accumulate else (0,)

    def pack(x: int, y: int, target: int, spectator: int) -> int:
        values = {"x": x, "y": y, "result": target}
        return spectator | sum(
            ((values[name] >> (width - 1 - bit)) & 1) << (n - 1 - wire)
            for name, wires in ports.items()
            for bit, wire in enumerate(wires)
        )

    for x, y, target, spectator in product(
        range(1 << width), range(1 << width), target_values, (0, 1)
    ):
        amp = complex(x + target + spectator + 1, 2 * y - target - spectator)
        initial[pack(x, y, target, spectator)] = torch.tensor(
            amp, dtype=torch.complex128
        )
        expected[pack(x, y, target ^ int(field(x) * field(y)), spectator)] = (
            torch.tensor(amp, dtype=torch.complex128)
        )
    norm = initial.norm()
    initial /= norm
    expected /= norm
    original_import = builtins.__import__

    def owned_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq", "kickmix", "galois"}:
            raise AssertionError(f"Captured field product imported {name}")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", owned_import)
        artifact = CircuitIR.from_json(CircuitIR(n, forward).to_json())
        circuit = Circuit(n, inputs=initial, dtype=torch.complex128).append(
            FieldOperation(n, artifact.instructions)
        )
        torch.testing.assert_close(
            circuit.state().flatten(), expected, atol=2e-12, rtol=0
        )
        restored = circuit.append(FieldOperation(n, backward))
        torch.testing.assert_close(
            restored.state().flatten(), initial, atol=2e-12, rtol=0
        )

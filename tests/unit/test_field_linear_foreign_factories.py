"""Actual complete field addition and square factory consumers."""

import builtins
from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.layout.partition import partition_register
from flagquantum.compiler.registers.primitives.circuit_binding import (
    restore_output_order_ir,
)
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.ecosystem.cirq._register_emitter import import_register_emitter
from tests.unit.test_field_multiplication import assert_gate_streams, reference_gates

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("family", ("addition", "constant", "square"))
@pytest.mark.parametrize("reversed_binding", (False, True))
def test_actual_linear_factory_complex_states_original_directions_and_native_append(
    family: str, reversed_binding: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cirq
    import numpy as np
    from qualtran.bloqs.gf_arithmetic import gf2_add_k, gf2_addition, gf2_square
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    factories = {
        "addition": gf2_addition._gf16_addition,
        "constant": gf2_add_k._gf16_add_k,
        "square": gf2_square._gf16_square,
    }
    source = factories[family].make()
    emitter = import_register_emitter(source)
    logical = sum(width for _, width in emitter.layout)
    wires = (
        tuple(reversed(range(logical))) if reversed_binding else tuple(range(logical))
    )
    ports = partition_register(wires, emitter.layout)
    assert emitter.requirements() == (0, 0)
    assert get_cost_value(source, QubitCount()) == logical
    n = logical + 1
    initial = torch.arange(1 << n, dtype=torch.float64)
    initial = torch.complex(initial + 1, initial.remainder(17) - 8)
    initial /= initial.norm()
    original_import = builtins.__import__

    def owned_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq", "galois", "kickmix"}:
            raise AssertionError(f"Captured field factory imported {name}")
        return original_import(name, *args, **kwargs)

    def pack(values: dict[str, int], spectator: int) -> int:
        return spectator | sum(
            ((values[name] >> (3 - bit)) & 1) << (n - wire - 1)
            for name, word in ports.items()
            for bit, wire in enumerate(word)
        )

    for adjoint, direction in enumerate((source, source.adjoint())):
        if family == "constant":
            graph, returned = (
                direction.decompose_bloq()
                .flatten()
                .to_cirq_circuit_and_quregs(
                    **{"x": np.array([cirq.LineQubit(w) for w in ports["x"]])}
                )
            )
            raw = []
            for op in graph.all_operations():
                assert op.gate == cirq.X
                raw.append(("x", tuple(q.x for q in op.qubits)))
            outputs = {"x": tuple(q.x for q in returned["x"].flat)}
        else:
            raw, outputs = reference_gates(direction, ports, (), repair_adjoint=False)
        permutation = tuple(
            wires.index(w) for name, _ in emitter.layout for w in outputs[name]
        )
        swaps = tuple(
            Instruction(gate.name, tuple(wires[w] for w in gate.wires))
            for gate in restore_output_order_ir(permutation)
        )
        assert emitter.binding_swaps[adjoint] == len(swaps)
        stream = emitter(ports, (), (), bool(adjoint))
        assert_gate_streams(
            [(gate.name, gate.wires) for gate in stream if gate.name != "swap"], raw
        )
        assert tuple(
            frozenset(gate.wires) for gate in stream if gate.name == "swap"
        ) == tuple(frozenset(gate.wires) for gate in swaps)
        model = get_cost_value(direction, QECGatesCost())
        counts = Counter(gate.name for gate in stream)
        assert counts["x"] + counts["cx"] == model.clifford
        assert counts["swap"] == len(swaps)
        assert model.total_t_count() == 0
        expected = torch.zeros_like(initial)
        ranges = [range(16) for _ in emitter.layout]
        field = source.qgf.gf_type
        for inputs in product(*ranges):
            values = dict(
                zip((name for name, _ in emitter.layout), inputs, strict=True)
            )
            # The source square adjoint repeats its forward linear body.
            callback = direction if family == "addition" else source
            result = callback.on_classical_vals(
                **{name: field(v) for name, v in values.items()}
            )
            out = {name: int(v) for name, v in result.items()}
            for spectator in (0, 1):
                expected[pack(out, spectator)] = initial[pack(values, spectator)]
        with monkeypatch.context() as patch:
            patch.setattr(builtins, "__import__", owned_import)
            artifact = CircuitIR.from_json(CircuitIR(n, stream).to_json())
            actual = Circuit(n, inputs=initial, dtype=torch.complex128).append(
                FieldOperation(n, artifact.instructions)
            )
            torch.testing.assert_close(
                actual.state().flatten(), expected, atol=2e-12, rtol=0
            )
    forward, backward = (emitter(ports, (), (), a) for a in (False, True))
    restored = (
        Circuit(n, inputs=initial, dtype=torch.complex128)
        .append(FieldOperation(n, forward))
        .append(FieldOperation(n, backward))
    )
    if family == "square":
        assert not torch.allclose(restored.state().flatten(), initial)
    else:
        torch.testing.assert_close(
            restored.state().flatten(), initial, atol=2e-12, rtol=0
        )


@pytest.mark.parametrize("family", ("addition", "constant", "square"))
def test_complete_linear_factories_symbolic_models_original_codec_and_construction(
    family: str,
) -> None:
    from qualtran import BloqExample, DecomposeTypeError
    from qualtran.bloqs.gf_arithmetic import gf2_add_k, gf2_addition, gf2_square
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.serialization.bloq import bloqs_to_proto

    modules = {"addition": gf2_addition, "constant": gf2_add_k, "square": gf2_square}
    module = modules[family]
    examples = {
        name: value
        for name, value in vars(module).items()
        if isinstance(value, BloqExample)
    }
    assert len(examples) == 2
    for name, example in examples.items():
        original = example.make()
        with pytest.raises(TypeError, match="QGF"):
            bloqs_to_proto(original, max_depth=0)
        if name.endswith("symbolic"):
            with pytest.raises(DecomposeTypeError, match="symbolic"):
                original.decompose_bloq()
            with pytest.raises(ValueError, match="concrete positive shapes"):
                import_register_emitter(original)
            model = get_cost_value(original, QECGatesCost())
            width = original.bitsize
            assert model.clifford == (width**2 - width if family == "square" else width)
            assert model.total_t_count() == 0
        else:
            assert import_register_emitter(original).requirements() == (0, 0)
    if family == "addition":
        with pytest.warns(DeprecationWarning, match="bitsize"):
            deprecated = gf2_addition.GF2Addition(bitsize=4)
        assert deprecated == gf2_addition._gf16_addition.make()

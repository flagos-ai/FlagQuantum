"""Cost-free source composition preserves stage calls and sided extra ownership."""

from collections import Counter
from dataclasses import replace
from math import pi
from typing import Any

import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.formats.legacy_cost_phasing_factories import (
    cost_phasing_factories,
)
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_gate_function import bit_gate_function
from flagquantum.compiler.registers.phase.cost_function_component import (
    cost_function_phase_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.test_component_graph import _native_trace, _source_trace
from tests.unit.test_multiplexed_source_binding import logical_operator
from tests.unit.test_register_component_calls import SourceCalls
from tests.unit.test_register_subtraction import assert_source
from tests.unit.test_subtraction_foreign_file import native_signature

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
_PHASING = "qualtran.bloqs.rotations.phasing_via_cost_function.PhasingViaCostFunction"


def _source(mode: str, shape: tuple[int, ...], evaluation: Any = None) -> Any:
    import attrs
    from qualtran._infra.data_types import QBit
    from qualtran._infra.registers import Register as SourceRegister
    from qualtran._infra.registers import Side
    from qualtran.bloqs.basic_gates.identity import Identity
    from qualtran.bloqs.basic_gates.rotation import ZPowGate
    from qualtran.bloqs.basic_gates.x_basis import XGate
    from qualtran.bloqs.rotations.phasing_via_cost_function import (
        PhasingViaCostFunction,
    )
    from qualtran.bloqs.rotations.quantum_variable_rotation import QvrInterface

    @attrs.frozen
    class NoCostPhase(QvrInterface):
        @property
        def cost_registers(self) -> tuple[Any, ...]:
            return ()

        @property
        def extra_registers(self) -> tuple[Any, ...]:
            return (SourceRegister("phase", QBit(), shape=shape, side=Side[mode]),)

        def build_composite_bloq(self, bb: Any, **soqs: Any) -> dict[str, Any]:
            values = (
                np.empty(shape, dtype=object)
                if mode == "RIGHT"
                else np.asarray(soqs.pop("phase"), dtype=object)
            )
            for index in np.ndindex(shape):
                value = bb.allocate(dtype=QBit()) if mode == "RIGHT" else values[index]
                if mode == "RIGHT":
                    value = bb.add(XGate(), q=value)
                value = bb.add(ZPowGate(exponent=0.375), q=value)
                if mode == "LEFT":
                    value = bb.add(XGate(), q=value)
                    bb.free(value)
                else:
                    values[index] = value
            return {} if mode == "LEFT" else {"phase": values if shape else values[()]}

    return PhasingViaCostFunction(
        Identity() if evaluation is None else evaluation, NoCostPhase()
    )


def _programs(
    mode: str, shape: tuple[int, ...]
) -> tuple[ProgramRecord, ProgramRecord, ProgramRecord]:
    bit = quantum_bit_type()
    evaluate = ProgramRecord(
        (bit_gate_function("evaluation", (Register("q", bit),), CircuitIR(1, ())),)
    )
    erase = replace(
        evaluate, functions=(replace(evaluate.functions[0], name="erasure"),)
    )
    x = bit_gate_function(
        "extra_x", (Register("q", bit),), CircuitIR(1, (Instruction("x", (0,)),))
    )
    phase = bit_gate_function(
        "extra_z",
        (Register("q", bit),),
        CircuitIR(1, (Instruction("phase", (0,), params={"theta": pi * 0.375}),)),
    )
    port = Register("phase", bit, shape, side=RegisterSide[mode])
    builder = ComponentProgramBuilder(Signature((port,)), (x, phase))
    values = (
        np.empty(shape, dtype=object)
        if mode == "RIGHT"
        else np.asarray(builder.inputs["phase"], dtype=object)
    )
    for index in np.ndindex(shape):
        value = builder.allocate(dtype=bit) if mode == "RIGHT" else values[index]
        assert isinstance(value, RegisterBinding)
        if mode == "RIGHT":
            value = builder.call_dict(x.name, {"q": value})["q"]
        value = builder.call_dict(phase.name, {"q": value})["q"]
        assert isinstance(value, RegisterBinding)
        if mode == "LEFT":
            value = builder.call_dict(x.name, {"q": value})["q"]
            assert isinstance(value, RegisterBinding)
            builder.free(value)
        else:
            values[index] = value
    result = builder.finalize(
        {} if mode == "LEFT" else {"phase": values if shape else values[()]}
    )
    result = replace(
        result,
        functions=(replace(result.functions[0], name="phase"), *result.functions[1:]),
    )
    return evaluate, result, erase


def _source_gates(
    source: Any,
    inputs: dict[str, tuple[int, ...]],
    outputs: dict[str, tuple[int, ...]],
    private: tuple[int, ...] = (),
) -> tuple[Instruction, ...]:
    import cirq

    signature = source.signature
    manager = cirq.SimpleQubitManager()
    input_wires = tuple(wire for wires in inputs.values() for wire in wires)
    source_qubits = dict(
        zip(input_wires, manager.qalloc(len(input_wires)), strict=True)
    )
    quregs = {
        name: np.asarray([source_qubits[wire] for wire in wires], dtype=object).reshape(
            (*signature.get_left(name).shape, 1)
        )
        for name, wires in inputs.items()
    }
    circuit, returned = source.decompose_bloq().to_cirq_circuit_and_quregs(
        qubit_manager=manager, **quregs
    )
    mapping = {qubit: wire for wire, qubit in source_qubits.items()}
    for name, wires in outputs.items():
        mapping.update(zip(np.asarray(returned[name]).flat, wires, strict=True))
    raw = cirq.decompose(
        circuit,
        context=cirq.DecompositionContext(manager),
        keep=lambda op: isinstance(
            op.gate, (cirq.ZPowGate, cirq.XPowGate, cirq.CXPowGate, cirq.IdentityGate)
        ),
    )
    scratch = tuple(
        dict.fromkeys(
            q for operation in raw for q in operation.qubits if q not in mapping
        )
    )
    assert len(scratch) == len(private)
    mapping.update(zip(scratch, private, strict=True))
    gates = []
    for operation in raw:
        gate = operation.gate
        if isinstance(gate, cirq.IdentityGate):
            continue
        wires = tuple(mapping[q] for q in operation.qubits)
        if isinstance(gate, cirq.CXPowGate):
            assert gate.exponent == 1 and gate.global_shift == 0
            gates.append(Instruction("cx", wires))
        elif isinstance(gate, cirq.XPowGate):
            assert gate.exponent == 1 and gate.global_shift == 0
            gates.append(Instruction("x", wires))
        else:
            assert isinstance(gate, cirq.ZPowGate) and gate.global_shift == 0
            gates.append(
                Instruction("phase", wires, params={"theta": pi * gate.exponent})
            )
    return tuple(gates)


@pytest.mark.parametrize("mode", ("THRU", "RIGHT", "LEFT"))
@pytest.mark.parametrize("shape", ((), (2,)))
def test_source_empty_cost_root_extra_sides_arrays_complex_and_ownership(
    mode: str, shape: tuple[int, ...]
) -> None:
    source = _source(mode, shape)
    evaluate, phase, erase = _programs(mode, shape)
    for record in (evaluate, phase, erase):
        assert verify_program_record(record) == ()
    direct = cost_function_phase_program(evaluate, phase, erase, cost_names=())
    factories = cost_phasing_factories(
        exact_adjoint=lambda child: erase, phase_cost_registers=lambda child: ()
    )
    resolved = factories[_PHASING](
        "main",
        direct.functions[0].registers,
        {"cost_eval_oracle": evaluate.functions[0], "phase_oracle": phase.functions[0]},
    )
    assert isinstance(resolved, ProgramRecord)
    resolved = replace(
        resolved, functions=(*resolved.functions, *evaluate.functions, *phase.functions)
    )
    assert verify_program_record(resolved) == ()
    assert resolved.functions[0].registers == tuple(native_signature(source.signature))
    assert _native_trace(resolved) == _source_trace(
        SourceCalls, source.decompose_bloq()
    )
    count = int(np.prod(shape)) if shape else 1
    extra = tuple(range(1, count + 1))
    extra_words = tuple((wire,) for wire in extra)
    inputs = ((0,),) if mode == "RIGHT" else ((0,), *extra_words)
    clean = extra if mode == "RIGHT" else ()
    ownership = tuple((wire, False) for wire in extra) if mode == "LEFT" else ()
    bound = bind_register_program(
        resolved,
        inputs,
        clean=clean,
        input_ownership=ownership,
        allow_workspace_outputs=mode == "RIGHT",
    )
    assert bound == bind_register_program(
        direct,
        inputs,
        clean=clean,
        input_ownership=ownership,
        allow_workspace_outputs=mode == "RIGHT",
    )
    source_inputs: dict[str, tuple[int, ...]] = (
        {"q": (0,)} if mode == "RIGHT" else {"q": (0,), "phase": extra}
    )
    source_outputs: dict[str, tuple[int, ...]] = (
        {"q": (0,)} if mode == "LEFT" else {"q": (0,), "phase": extra}
    )
    expected = _source_gates(source, source_inputs, source_outputs)
    assert_source(bound.circuit.instructions, expected)
    assert CircuitIR.from_json(bound.circuit.to_json()) == bound.circuit
    whole = logical_operator(bound.circuit, count + 1)
    if mode == "RIGHT":
        source_block = whole[:, (0, 1 << count)]
    elif mode == "LEFT":
        # A source LEFT effect is the clean-output block, not a runtime
        # projection on arbitrary invalid inputs. The legal image runs below.
        source_block = whole[(0, 1 << count), :]
    else:
        source_block = whole
    torch.testing.assert_close(
        source_block,
        torch.tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=3e-12,
        rtol=0,
    )
    total = count + 2
    amplitudes = torch.arange(
        1, (1 << (count + 2 if mode == "THRU" else 2)) + 1, dtype=torch.float64
    )
    amplitudes = torch.complex(amplitudes, 2 - 3 * amplitudes)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << total, dtype=torch.complex128)
    view = initial.reshape(2, 1 << count, 2)
    if mode == "THRU":
        view[:] = amplitudes.reshape(2, 1 << count, 2)
        source_initial = amplitudes.reshape(2 << count, 2)
    else:
        view[:, -1 if mode == "LEFT" else 0, :] = amplitudes.reshape(2, 2)
        source_initial = (
            view.reshape(2 << count, 2) if mode == "LEFT" else amplitudes.reshape(2, 2)
        )
    result = (
        Circuit(total, inputs=initial, dtype=torch.complex128)
        .append(FieldOperation(total, bound.circuit.instructions))
        .state()
        .flatten()
    )
    target = (
        torch.tensor(source.tensor_contract(), dtype=torch.complex128) @ source_initial
    )
    if mode == "LEFT":
        expected_state = torch.zeros_like(view)
        expected_state[:, 0, :] = target
        assert torch.count_nonzero(result.reshape(2, 1 << count, 2)[:, 1:, :]) == 0
    else:
        expected_state = target.reshape(2, 1 << count, 2)
    torch.testing.assert_close(result, expected_state.flatten(), atol=3e-12, rtol=0)
    assert bound.outputs == (((0,),) if mode == "LEFT" else ((0,), *extra_words))
    assert sum(g.name == "phase" for g in bound.circuit.instructions) == count
    assert not any(g.name in {"measure", "reset"} for g in bound.circuit.instructions)


def test_duplicate_cost_names_still_refuse() -> None:
    evaluate, phase, erase = _programs("THRU", ())
    with pytest.raises(ValueError, match="distinct"):
        cost_function_phase_program(evaluate, phase, erase, cost_names=("cost", "cost"))


def test_cost_free_phase_still_evaluates_and_cleans_unused_right_results() -> None:
    import attrs
    from qualtran._infra.bloq import Bloq
    from qualtran._infra.data_types import QBit
    from qualtran._infra.registers import Register as SourceRegister
    from qualtran._infra.registers import Side
    from qualtran._infra.registers import Signature as SourceSignature
    from qualtran.bloqs.basic_gates.cnot import CNOT
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.resource_counting._qubit_counts import QubitCount

    @attrs.frozen
    class CopyEvaluation(Bloq):
        @property
        def signature(self) -> Any:
            return SourceSignature(
                (
                    SourceRegister("q", QBit()),
                    SourceRegister("r", QBit(), side=Side.RIGHT),
                )
            )

        def build_composite_bloq(self, bb: Any, **soqs: Any) -> dict[str, Any]:
            r = bb.allocate(dtype=QBit())
            q, r = bb.add(CNOT(), ctrl=soqs["q"], target=r)
            return {"q": q, "r": r}

    source = _source("THRU", (), CopyEvaluation())
    bit = quantum_bit_type()
    cx = bit_gate_function(
        "copy",
        (Register("ctrl", bit), Register("target", bit)),
        CircuitIR(2, (Instruction("cx", (0, 1)),)),
    )
    records = []
    for inverse in (False, True):
        builder = ComponentProgramBuilder(
            Signature(
                (
                    Register("q", bit),
                    Register(
                        "r",
                        bit,
                        side=RegisterSide.LEFT if inverse else RegisterSide.RIGHT,
                    ),
                )
            ),
            (cx,),
        )
        r = builder.inputs["r"] if inverse else builder.allocate(dtype=bit)
        result = builder.call_dict(cx.name, {"ctrl": builder.inputs["q"], "target": r})
        target = result["target"]
        assert isinstance(target, RegisterBinding)
        if inverse:
            builder.free(target)
        records.append(
            builder.finalize(
                {"q": result["ctrl"]} if inverse else {"q": result["ctrl"], "r": target}
            )
        )
    _, phase, _ = _programs("THRU", ())
    program = cost_function_phase_program(records[0], phase, records[1], cost_names=())
    assert verify_program_record(program) == ()
    assert _native_trace(program) == _source_trace(SourceCalls, source.decompose_bloq())
    bound = bind_register_program(program, ((0,), (1,)), clean=(2,))
    expected = _source_gates(
        source, {"q": (0,), "phase": (1,)}, {"q": (0,), "phase": (1,)}, private=(2,)
    )
    assert_source(bound.circuit.instructions, expected)
    raw = torch.arange(1, 9, dtype=torch.float64)
    amplitudes = torch.complex(raw, 2 - 3 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(16, dtype=torch.complex128)
    initial.reshape(4, 2, 2)[:, 0, :] = amplitudes.reshape(4, 2)
    actual = (
        Circuit(4, inputs=initial, dtype=torch.complex128)
        .append(FieldOperation(4, bound.circuit.instructions))
        .state()
        .reshape(4, 2, 2)
    )
    target_state = torch.tensor(
        source.tensor_contract(), dtype=torch.complex128
    ) @ amplitudes.reshape(4, 2)
    torch.testing.assert_close(actual[:, 0, :], target_state, atol=3e-12, rtol=0)
    assert torch.count_nonzero(actual[:, 1, :]) == 0
    assert [event.kind for event in bound.events] == ["allocate", "free"]
    assert all(event.wires == (2,) for event in bound.events)
    assert Counter(gate.name for gate in bound.circuit.instructions) == {
        "cx": 2,
        "phase": 1,
    }
    model = get_cost_value(source, QECGatesCost())
    assert model.clifford == 2 and model.rotation == 1 and model.measurement == 0
    assert get_cost_value(source, QubitCount()) == 3
    torch.testing.assert_close(
        logical_operator(bound.circuit, 2),
        torch.tensor(source.tensor_contract(), dtype=torch.complex128),
        atol=3e-12,
        rtol=0,
    )


def test_native_empty_cost_constructor_and_append_need_no_source_sdks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    evaluate, phase, erase = _programs("THRU", ())
    original = builtins.__import__

    def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq"}:
            raise AssertionError("Source SDK imported during native cost composition")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    factory = cost_phasing_factories(
        exact_adjoint=lambda child: erase, phase_cost_registers=lambda child: ()
    )[_PHASING]
    record = factory(
        "main",
        None,
        {"cost_eval_oracle": evaluate.functions[0], "phase_oracle": phase.functions[0]},
    )
    assert isinstance(record, ProgramRecord)
    record = replace(
        record, functions=(*record.functions, *evaluate.functions, *phase.functions)
    )
    bound = bind_register_program(record, ((0,), (1,)))
    assert (
        Circuit(2)
        .append(FieldOperation(2, bound.circuit.instructions))
        .to_ir()
        .instructions
        == bound.circuit.instructions
    )

"""External owned CBit values initialize immutable feedback slots at entry."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_classical import (
    discard_word_function,
    parity_word_function,
)
from flagquantum.compiler.registers.entry_bindings import entry_input_layout
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.classical_dataflow import parse_classical_xor
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import Operation, ProgramRecord
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.dynamic import DynamicCircuit
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_program_classical_registers import identity
from tests.unit.test_program_owned_measurement import measured_function
from tests.unit.test_register_classical_parity import feedback_function
from tests.unit.test_register_program_lowering import packed_circuit

pytestmark = pytest.mark.unit


def external_feedback_program(
    count: int, polarity: bool, measured: bool
) -> ProgramRecord:
    feedback = feedback_function(polarity)
    block = feedback.blocks[0]
    read = block.operations[0]
    assert block.return_values is not None and feedback.registers is not None
    feedback = replace(
        feedback,
        blocks=(
            replace(
                block,
                operations=(
                    read,
                    Operation("discard_classical_register", (read.results[0],)),
                    *block.operations[1:],
                ),
                return_values=block.return_values[1:],
            ),
        ),
        registers=(
            Register("c", classical_bit_type(), side=RegisterSide.LEFT),
            *feedback.registers[1:],
        ),
    )
    registers = [
        Register("x", classical_bit_type(), (count,)),
        Register("targets", quantum_bit_type(), (2,)),
        Register("spectator", quantum_bit_type()),
    ]
    if measured:
        registers.append(Register("q", quantum_bit_type(), side=RegisterSide.LEFT))
    builder = ComponentProgramBuilder(
        Signature(registers),
        [
            identity("cast", (Register("reg", classical_bit_type()),)),
            parity_word_function("parity", count + int(measured)),
            discard_word_function("discard"),
            measured_function("measurement"),
            feedback,
        ],
    )
    data = []
    for value in np.asarray(builder.inputs["x"]).flat:
        for _ in range(2):
            value = builder.call_dict("cast", {"reg": value})["reg"]
        data.append(value)
    if measured:
        data.append(builder.call_dict("measurement", {"q": builder.inputs["q"]})["c"])
    result = builder.call_dict("parity", {"x": np.array(data, dtype=object)})
    successors = np.asarray(result["x"])
    if measured:
        builder.call_dict("discard", {"c": successors[-1]})
    targets = builder.call_dict(
        "feedback", {"c": result["parity"], "targets": builder.inputs["targets"]}
    )["targets"]
    work = builder.allocate(dtype=quantum_bit_type())
    builder.free(work)
    return builder.finalize(
        {
            "x": successors[:count][::-1],
            "targets": targets,
            "spectator": builder.inputs["spectator"],
        }
    )


def physical_inputs(count: int, measured: bool) -> tuple[tuple[int, ...], ...]:
    return ((),) * count + tuple((i,) for i in range(3 + int(measured)))


@pytest.mark.parametrize("count", [0, 1, 2, 3])
@pytest.mark.parametrize("measured", [False, True])
def test_entry_initializations_are_fresh_before_measurement_and_parity(
    count: int, measured: bool
) -> None:
    record = external_feedback_program(count, True, measured)
    bits = tuple(i % 2 for i in range(count))
    bound = bind_register_program(
        record,
        physical_inputs(count, measured),
        clean=(3 + int(measured),),
        classical_inputs=(*bits, *((None,) * (3 + int(measured)))),
    )
    assert bound.outputs == ((),) * count + ((0,), (1,), (2,))
    assert bound.classical_outputs == (
        tuple((i,) for i in reversed(range(count))) + ((),) * 3 if count else ()
    )
    gates = bound.circuit.instructions
    assert tuple(parse_classical_xor(g).constant for g in gates[:count]) == tuple(
        map(bool, bits)
    )
    assert tuple(parse_classical_xor(g).target for g in gates[:count]) == tuple(
        range(count)
    )
    assert Counter(g.name for g in gates) == Counter(
        {"classical_xor": count + 1, "measure": int(measured), "z": 1, "x": 1}
    )
    parity = next(
        parse_classical_xor(g) for g in gates[count:] if g.name == "classical_xor"
    )
    assert parity.sources == tuple(range(count + int(measured)))
    assert parity.target == count + int(measured)
    assert all(
        g.metadata["conditions"] == ((parity.target, 1),)
        for g in gates
        if g.name in {"z", "x"}
    )


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("count", [0, 1, 2, 3])
@pytest.mark.parametrize("polarity", [False, True])
@pytest.mark.parametrize("measured", [False, True])
def test_every_classical_input_and_measurement_branch_matches_original_complex_tensors(
    count: int, polarity: bool, measured: bool
) -> None:
    from qualtran import CBit, CtrlSpec
    from qualtran.bloqs.basic_gates import MeasureZ, XGate, ZGate
    from qualtran.bloqs.bookkeeping.cast import Cast
    from qualtran.bloqs.gf_arithmetic.gf2_multiplication import Parity

    record = external_feedback_program(count, polarity, measured)
    width = 4 + int(measured)
    state = torch.zeros(1 << width, dtype=torch.complex128)
    state[::2] = torch.randn(
        1 << (width - 1),
        dtype=torch.complex128,
        generator=torch.Generator().manual_seed(108432 + count),
    )
    state /= state.norm()
    for bits in product((0, 1), repeat=count):
        bound = bind_register_program(
            record,
            physical_inputs(count, measured),
            clean=(width - 1,),
            classical_inputs=(*bits, *((None,) * (width - 1))),
        )
        gates = (
            Circuit(width)
            .append(FieldOperation(width, bound.circuit.instructions))
            .to_ir()
            .instructions
        )
        assert gates == bound.circuit.instructions
        for history in product((0, 1), repeat=int(measured)):
            expected = state.numpy().reshape((2,) * width)
            original = np.array(bits, dtype=np.uint8)
            for _ in range(2):
                original = np.array(
                    [
                        Cast(CBit(), CBit()).call_classically(reg=int(bit))[0]
                        for bit in original
                    ],
                    dtype=np.uint8,
                )
            if measured:
                copy, _ = MeasureZ().my_tensors({"q": "q"}, {"c": "c"})
                expected = np.moveaxis(
                    np.tensordot(
                        copy.data[:, :, history[0]].T, expected, axes=([1], [3])
                    ),
                    0,
                    3,
                )
            source_inputs = np.array([*original, *history], dtype=np.uint8)
            _, parity = Parity(count + int(measured)).call_classically(x=source_inputs)
            for gate, wire in ((ZGate(), 0), (XGate(), 1)):
                tensor = gate.controlled(
                    CtrlSpec(qdtypes=[CBit()], cvs=[int(polarity)])
                )._tensor_data()
                matrix = tensor[int(parity), :, int(parity), :]
                expected = np.moveaxis(
                    np.tensordot(matrix, expected, axes=([1], [wire])), 0, wire
                )
            actual = forced_branch(state, gates, history).reshape(expected.shape)
            np.testing.assert_allclose(actual.numpy(), expected, rtol=0, atol=3e-15)
            assert torch.count_nonzero(actual[..., 1]) == 0


@pytest.mark.parametrize("strategy", ["trajectory", "batched"])
@pytest.mark.parametrize("polarity", [False, True])
def test_runtime_initialization_measurement_and_discarded_snapshot(
    strategy: str, polarity: bool
) -> None:
    record = external_feedback_program(2, polarity, True)
    state = torch.zeros(32, dtype=torch.complex128)
    state[::2] = torch.randn(
        16, dtype=torch.complex128, generator=torch.Generator().manual_seed(108432)
    )
    state /= state.norm()
    for bits in product((0, 1), repeat=2):
        bound = bind_register_program(
            record,
            physical_inputs(2, True),
            clean=(4,),
            classical_inputs=(*bits, None, None, None, None),
        )
        circuit = DynamicCircuit(5, inputs=state, dtype=torch.complex128)
        circuit.append(FieldOperation(5, bound.circuit.instructions))
        result = run_dynamic(circuit, shots=8, seed=1084, strategy=strategy)
        for values, actual in zip(
            result.classical_bits, result.final_states.reshape(8, 32), strict=True
        ):
            assert tuple(map(int, values[:2])) == bits
            history = (int(values[2]),)
            assert int(values[3]) == (bits[0] ^ bits[1] ^ history[0])
            expected = forced_branch(state, bound.circuit.instructions, history)
            expected /= expected.norm()
            torch.testing.assert_close(actual, expected, rtol=0, atol=3e-15)
            assert torch.count_nonzero(actual[1::2]) == 0


@pytest.mark.parametrize("shape", [(), (2,), (2, 2), (0, 2)])
def test_shaped_classical_and_zero_width_quantum_ports_remain_distinct(
    shape: tuple[int, ...],
) -> None:
    registers = (
        Register("c", classical_bit_type(), shape),
        Register("empty", RegisterType("opaque", 0)),
        Register("q", quantum_bit_type()),
    )
    leaf = identity("leaf", registers)
    builder = ComponentProgramBuilder(Signature(registers), [leaf])
    inputs = builder.inputs
    for _ in range(2):
        inputs = builder.call_dict("leaf", inputs)
    record = builder.finalize(inputs)
    count = len(tuple(registers[0].all_idxs()))
    bits = tuple(bool(i % 2) for i in range(count))
    bound = bind_register_program(
        record, ((),) * (count + 1) + ((0,),), classical_inputs=(*bits, None, None)
    )
    assert bound.outputs == ((),) * (count + 1) + ((0,),)
    assert bound.classical_outputs == (
        (tuple((i,) for i in range(count)) + ((), ())) if count else ()
    )
    assert bound.events == ()
    assert len(bound.circuit.instructions) == count


@pytest.mark.parametrize(
    "value", [None, -1, 2, 1.0, np.int64(1), np.bool_(True), "1", ()]
)
def test_invalid_classical_values_fail_before_binding(value: Any) -> None:
    function = identity(
        "main", (Register("c", classical_bit_type()), Register("q", quantum_bit_type()))
    )
    with pytest.raises(ValueError, match="built-in zero/one"):
        bind_register_program(
            ProgramRecord((function,)), ((), (0,)), classical_inputs=(value, None)
        )


@pytest.mark.parametrize("values", [[], (1,), (1, None, None), (1, 0)])
def test_invalid_initialization_layout_rejects_atomically(values: Any) -> None:
    function = identity(
        "main", (Register("c", classical_bit_type()), Register("q", quantum_bit_type()))
    )
    with pytest.raises(ValueError, match="immutable|all entry|require None"):
        bind_register_program(
            ProgramRecord((function,)), ((), (0,)), classical_inputs=values
        )
    assert bind_register_program(
        ProgramRecord((function,)), ((), (0,)), classical_inputs=(1, None)
    ).classical_outputs == ((0,), ())


def test_classical_only_graph_uses_explicit_carrier_and_no_quantum_initialization() -> (
    None
):
    record = ProgramRecord((parity_word_function("main", 2),))
    bound = bind_register_program(record, ((), ()), classical_inputs=(0, 1), clean=(7,))
    assert bound.circuit.n_wires == 8 and bound.events == ()
    assert bound.outputs == ((), (), ()) and bound.classical_outputs == (
        (0,),
        (1,),
        (2,),
    )
    assert all(
        g.name == "classical_xor" and g.wires == () for g in bound.circuit.instructions
    )
    with pytest.raises(ValueError, match="at least one physical wire"):
        bind_register_program(record, ((), ()), classical_inputs=(0, 1))


def test_classical_position_cannot_claim_physical_wire_or_pool_owner() -> None:
    record = ProgramRecord(
        (
            identity(
                "main",
                (
                    Register("c", classical_bit_type()),
                    Register("q", quantum_bit_type()),
                ),
            ),
        )
    )
    with pytest.raises(ValueError, match="physical input widths"):
        bind_register_program(record, ((1,), (0,)), classical_inputs=(1, None))
    with pytest.raises(ValueError, match="input ownership"):
        bind_register_program(
            record,
            ((), (0,)),
            classical_inputs=(1, None),
            input_ownership=((1, False),),
        )


def test_gate_only_entry_layout_preserves_existing_quantum_callers() -> None:
    record = packed_circuit(CircuitIR(1, ()), (1,))
    leaf = next(f for f in record.functions if f.name == "gate_leaf")
    assert entry_input_layout(leaf, ()) == ((1,), (None,))
    bound = bind_register_program(record, ((0,),))
    assert bound.circuit == CircuitIR(1, ()) and bound.outputs == ((0,),)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("shape", [(), (2,), (2, 2), (0, 2)])
def test_original_entry_assignment_matches_native_row_major_slots(
    shape: tuple[int, ...],
) -> None:
    from qualtran import BloqBuilder, CBit, LeftDangle, RightDangle, Soquet
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.classical_sim import ClassicalSimState

    source_register = SourceRegister("c", CBit(), shape=shape)
    source_builder, source_inputs = BloqBuilder.from_signature(
        SourceSignature([source_register])
    )
    source = source_builder.finalize(**source_inputs)
    count = len(tuple(source_register.all_idxs()))
    native = ProgramRecord(
        (identity("main", (Register("c", classical_bit_type(), shape),)),)
    )
    for bits in product((0, 1), repeat=count):
        data = np.array(bits, dtype=np.uint8).reshape(shape)
        values = {"c": data if shape else int(data.item())}
        original = ClassicalSimState.from_cbloq(source, values)
        assigned = tuple(
            int(original.soq_assign[Soquet(LeftDangle, source_register, idx)])
            for idx in source_register.all_idxs()
        )
        assert assigned == bits
        bound = bind_register_program(
            native, ((),) * count, clean=(0,), classical_inputs=bits
        )
        slots: dict[int, int] = {}
        for gate in bound.circuit.instructions:
            assignment = parse_classical_xor(gate)
            slots[assignment.target] = assignment.evaluate(slots)
        returned = tuple(slots[item[0]] for item in bound.classical_outputs)
        assert returned == assigned
        if count == 0:
            with pytest.raises(KeyError) as failure:
                source.call_classically(**values)
            assert failure.value.args == (RightDangle,)
        else:
            (source_returned,) = source.call_classically(**values)
            np.testing.assert_array_equal(
                np.asarray(source_returned).reshape(-1), returned
            )


def test_initialized_slots_survive_independent_binding_values() -> None:
    record = ProgramRecord((identity("main", (Register("c", classical_bit_type()),)),))
    first = bind_register_program(record, ((),), clean=(0,), classical_inputs=(0,))
    second = bind_register_program(record, ((),), clean=(0,), classical_inputs=(1,))
    assert not parse_classical_xor(first.circuit.instructions[0]).constant
    assert parse_classical_xor(second.circuit.instructions[0]).constant
    assert first.classical_outputs == second.classical_outputs == ((0,),)

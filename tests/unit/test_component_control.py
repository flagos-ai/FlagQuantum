"""Pinned total-control calls retain specialized ports, phases and measured cleanup."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import product
from math import pi
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler._component_trace.program_call_graph import (
    program_component_call_graph,
)
from flagquantum.compiler.registers.component_builder import (
    BindingCollection,
    ComponentProgramBuilder,
)
from flagquantum.compiler.registers.component_control import control_component_program
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.primitives.controlled_rotation import (
    controlled_z_power_ir,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.core.ir.program.register_bindings import RegisterBinding
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_register_phase_gradient import source_gradient_gates
from tests.unit.test_register_physical_binding import allocated_circuit
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def body(
    name: str, registers: tuple[Register, ...], circuit: CircuitIR
) -> tuple[FunctionRecord, ...]:
    """Expose actual source ports around an existing measured primitive emitter."""
    width = sum(int(register.total_qubits()) for register in registers)
    private = allocated_circuit(circuit, width)
    definitions = tuple(
        replace(
            function,
            name=name + "_" + function.name,
            blocks=tuple(
                replace(
                    block,
                    operations=tuple(
                        replace(op, callee=name + "_" + op.callee) if op.callee else op
                        for op in block.operations
                    ),
                )
                for block in function.blocks
            ),
        )
        for function in private.functions
    )
    builder = ComponentProgramBuilder(Signature(registers), definitions, name=name)
    bits: list[RegisterBinding] = []
    for value in builder.inputs.values():
        for binding in np.asarray(value).flat:
            bits.extend(builder.split(binding))
    packed = builder.join(bits)
    result = builder.call_dict(name + "_main", {"data": packed})["data"]
    assert isinstance(result, RegisterBinding)
    returned = iter(builder.split(result))
    outputs: dict[str, BindingCollection] = {}
    for register in registers:
        array = np.empty(register.shape, dtype=object)
        for index in register.all_idxs():
            array[index] = builder.join([next(returned)], dtype=register.dtype)
        outputs[register.name] = array if register.shape else array[()]
    functions = builder.finalize(outputs).functions
    renamed = {
        f.name: f.name if f.name == name else name + "_" + f.name for f in functions
    }
    return tuple(
        replace(
            f,
            name=renamed[f.name],
            blocks=tuple(
                replace(
                    block,
                    operations=tuple(
                        replace(op, callee=renamed[op.callee]) if op.callee else op
                        for op in block.operations
                    ),
                )
                for block in f.blocks
            ),
        )
        for f in functions
    )


def add_h(
    builder: ComponentProgramBuilder,
    control: BindingCollection,
    values: dict[str, BindingCollection],
) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
    ctrl, q = builder.call_tuple("ch", {"ctrl": control, "target": values["q"]})
    return ctrl, (q,)


def add_phase(
    builder: ComponentProgramBuilder,
    control: BindingCollection,
    values: dict[str, BindingCollection],
) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
    output = builder.call_dict(
        "cp", {"q": np.array([control, values["q"]], dtype=object)}
    )["q"]
    assert isinstance(output, np.ndarray)
    return output[0], (output[1],)


def fixture_program(
    repeats: int, spectator: str = "spectator"
) -> tuple[ProgramRecord, Any]:
    from qualtran import Controlled, CtrlSpec
    from qualtran.bloqs.basic_gates import Hadamard, ZPowGate

    bit = quantum_bit_type()
    definitions = (
        *body("h", (Register("q", bit),), CircuitIR(1, (Instruction("h", (0,)),))),
        *body(
            "p",
            (Register("q", bit),),
            CircuitIR(1, (Instruction("phase", (0,), {"theta": pi / 4}),)),
        ),
        *body(
            "ch",
            (Register("ctrl", bit), Register("target", bit)),
            CircuitIR(2, (Instruction("ch", (0, 1)),)),
        ),
        *body(
            "cp",
            (Register("q", bit, (2,)),),
            CircuitIR(3, controlled_z_power_ir(0, 1, 0.25, 2, 0)),
        ),
    )
    signature = Signature([Register("q", bit), Register(spectator, bit)])
    native = ComponentProgramBuilder(signature, definitions)
    source = _source(signature)
    values, expected = native.inputs["q"], source.inputs["q"]
    for _ in range(repeats):
        for name, bloq in (("h", Hadamard()), ("p", ZPowGate(0.25))):
            values = native.call_dict(name, {"q": values})["q"]
            expected = source.builder.add(bloq, q=expected)
    record = native.finalize({"q": values, spectator: native.inputs[spectator]})
    composite = source.builder.finalize(
        **{"q": expected, spectator: source.inputs[spectator]}
    )
    return record, Controlled(composite, CtrlSpec())


@pytest.mark.parametrize("repeats", [1, 2, 3])
@pytest.mark.parametrize("spectator", ["spectator", "ctrl", "ctrl2"])
def test_source_specialization_renaming_packing_graph_counts_and_all_complex_branches(
    repeats: int, spectator: str
) -> None:
    from qualtran.bloqs.basic_gates import CHadamard, CZPowGate
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    record, source = fixture_program(repeats, spectator)
    controlled = control_component_program(record, {"h": add_h, "p": add_phase})
    expected = source.decompose_bloq()
    assert _native_trace(controlled) == _source_trace(_source(Signature([])), expected)
    assert tuple(
        _source(Signature([])).signature(
            Signature(controlled.functions[0].registers or ())
        )
    ) == tuple(source.signature)
    assert all(
        a is b
        for a, b in zip(record.functions[1:], controlled.functions[1:], strict=True)
    )
    assert verify_program_record(controlled) == ()
    assert controlled.canonical_payload is None
    targets = {CHadamard(): "ch", CZPowGate(0.25): "cp"}
    assert {targets[bloq]: count for bloq, count in source.bloq_counts().items()} == {
        "ch": repeats,
        "cp": repeats,
    }
    actual_counts = program_component_call_graph(
        controlled, keep={"ch", "cp"}.__contains__
    )
    assert actual_counts.leaf_counts == {"ch": repeats, "cp": repeats}
    bound = bind_register_program(controlled, ((0,), (1,), (2,)), clean=(3,))
    assert bound.outputs == ((0,), (1,), (2,))
    gates = bound.circuit.instructions
    assert Counter(g.name for g in gates) == {
        "ch": repeats,
        "cx": 6 * repeats,
        "h": 3 * repeats,
        "t": 2 * repeats,
        "tdg": 2 * repeats,
        "s": repeats,
        "phase": repeats,
        "measure": repeats,
        "cz": repeats,
        "reset": repeats,
    }
    source_gates = []
    for gate in gates:
        if gate.name == "ch":
            source_gates.append(Instruction("ch", (0, 1)))
        elif gate.name == "measure":
            reference, _ = source_gradient_gates(
                CZPowGate(0.25),
                {"q": (0, 1)},
                {"q": (0, 1)},
                work=3,
                bit=gate.metadata["classical_bit"],
            )
            source_gates.extend(reference)
    assert_source(gates, tuple(source_gates))
    model = get_cost_value(source, QECGatesCost())
    assert model.measurement == repeats and model.and_bloq == repeats
    leaf_models = {
        name: get_cost_value(bloq, QECGatesCost()) for bloq, name in targets.items()
    }
    for category in (
        "t",
        "toffoli",
        "cswap",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
    ):

        def cost(name: str, field: str = category) -> int:
            return int(getattr(leaf_models[name], field))

        assert actual_counts.estimate_leaf_cost(cost) == getattr(model, category)
    assert get_cost_value(source, QubitCount()) == 4
    initial = torch.zeros(16, dtype=torch.complex128)
    initial[::2] = torch.tensor([complex(i + 1, 3 - 2 * i) for i in range(8)])
    initial /= initial.norm()
    expected_state = torch.zeros_like(initial)
    expected_state[::2] = torch.as_tensor(
        np.einsum(
            "ij,j->i", source.tensor_contract(), initial[::2].numpy(), optimize=False
        )
    )
    for history in product((0, 1), repeat=repeats):
        state = forced_branch(initial, gates, history)
        torch.testing.assert_close(
            state, expected_state / 2 ** (repeats / 2), rtol=0, atol=2e-12
        )
        assert torch.count_nonzero(state[1::2]).item() == 0
    flattened, _ = expand_component_calls(controlled)
    assert (
        bind_register_program(flattened, ((0,), (1,), (2,)), clean=(3,)).circuit
        == bound.circuit
    )


@pytest.mark.parametrize(
    "names,expected",
    [
        ((), "ctrl"),
        (("ctrl",), "ctrl2"),
        (("ctrl", "ctrl2"), "ctrl3"),
        (("ctrl", "ctrl3"), "ctrl2"),
    ],
)
def test_empty_entry_threads_control_and_matches_source_name_collisions(
    names: tuple[str, ...], expected: str
) -> None:
    from qualtran import Controlled, CtrlSpec

    signature = Signature([Register(name, quantum_bit_type()) for name in names])
    native = ComponentProgramBuilder(signature)
    source = _source(signature)
    record = native.finalize(native.inputs)
    controlled = control_component_program(record, {})
    upstream = Controlled(source.builder.finalize(**source.inputs), CtrlSpec())
    assert controlled.functions[0].registers is not None
    assert (
        controlled.functions[0].registers[0].name
        == upstream.ctrl_reg_names[0]
        == expected
    )
    assert (
        _native_trace(controlled)
        == _source_trace(source, upstream.decompose_bloq())
        == []
    )
    assert program_component_call_graph(controlled).leaf_counts == {"main": 1}


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "unknown",
        "invalid",
        "directional",
        "bad_arity",
        "bad_target",
        "callback",
    ],
)
def test_failures_preserve_input_and_propagate_adder_errors(failure: str) -> None:
    record, _ = fixture_program(1)
    entry = "absent" if failure == "unknown" else "main"
    adders = {"h": add_h, "p": add_phase}
    if failure == "missing":
        adders.pop("p")
    elif failure == "invalid":
        record = replace(record, functions=(*record.functions, record.functions[0]))
    elif failure == "directional":
        from flagquantum.compiler.registers.layout.component_workspace import (
            allocate_word_function,
        )

        builder = ComponentProgramBuilder(
            Signature([Register("q", quantum_bit_type(), side=RegisterSide.RIGHT)]),
            [allocate_word_function("alloc", quantum_bit_type(), False)],
        )
        record = builder.finalize({"q": builder.call_dict("alloc", {})["reg"]})
    elif failure in {"bad_arity", "bad_target", "callback"}:

        def broken(
            builder: ComponentProgramBuilder,
            control: BindingCollection,
            values: dict[str, BindingCollection],
        ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
            if failure == "bad_target":
                builder.call_dict("absent", values)
            if failure == "callback":
                raise RuntimeError("adder failure")
            return control, ()

        adders["h"] = broken
    before = repr(record)
    with pytest.raises(RuntimeError if failure == "callback" else ValueError):
        control_component_program(record, adders, entry)
    assert repr(record) == before


def test_adder_replacement_preserves_consumers_and_exact_adjoint_restores_all_histories() -> (
    None
):
    record, source = fixture_program(2)
    controlled = control_component_program(record, {"h": add_h, "p": add_phase})
    forward = bind_register_program(controlled, ((0,), (1,), (2,)), clean=(3,))
    # The inverse uses the same immutable specialized bodies with exact source
    # adjoints supplied at the existing explicit-adjoint boundary.
    from flagquantum.compiler.registers.component_adjoint import (
        adjoint_component_program,
    )

    inverse_phase = body(
        "cp_inverse",
        (Register("q", quantum_bit_type(), (2,)),),
        CircuitIR(3, controlled_z_power_ir(0, 1, -0.25, 2, 0)),
    )
    module = replace(controlled, functions=(*controlled.functions, *inverse_phase))
    inverse = adjoint_component_program(module, {"ch": "ch", "cp": "cp_inverse"})
    backward = bind_register_program(inverse, ((0,), (1,), (2,)), clean=(3,))
    initial = torch.zeros(16, dtype=torch.complex128)
    initial[::2] = torch.tensor([complex(3 - i, 2 * i + 1) for i in range(8)])
    initial /= initial.norm()
    for first in product((0, 1), repeat=2):
        state = forced_branch(initial, forward.circuit.instructions, first)
        for second in product((0, 1), repeat=2):
            restored = forced_branch(state, backward.circuit.instructions, second)
            torch.testing.assert_close(restored, initial / 4, rtol=0, atol=2e-12)
            assert torch.count_nonzero(restored[1::2]).item() == 0

    def replacement(
        builder: ComponentProgramBuilder,
        control: BindingCollection,
        values: dict[str, BindingCollection],
    ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
        # Replace the specialized adder through its existing compiler boundary.
        return add_phase(builder, control, values)

    replaced = control_component_program(record, {"h": add_h, "p": replacement})
    assert bind_register_program(replaced, ((0,), (1,), (2,)), clean=(3,)) == forward
    source_inverse = source.adjoint()
    expected = torch.as_tensor(
        np.einsum(
            "ij,jk->ik",
            source_inverse.tensor_contract(),
            source.tensor_contract(),
            optimize=False,
        )
    )
    torch.testing.assert_close(
        expected, torch.eye(8, dtype=torch.complex128), rtol=0, atol=2e-12
    )


def test_custom_entry_block_clears_payload_and_keeps_forward_definitions() -> None:
    record, _ = fixture_program(1)
    function = record.functions[0]
    block = function.blocks[0]
    values = {
        v: replace(v, scope="body")
        for v in (*block.arguments, *(v for op in block.operations for v in op.results))
    }
    renamed = replace(
        block,
        name="body",
        arguments=tuple(values[v] for v in block.arguments),
        operations=tuple(
            replace(
                op,
                operands=tuple(values[v] for v in op.operands),
                results=tuple(values[v] for v in op.results),
            )
            for op in block.operations
        ),
        return_values=tuple(values[v] for v in block.return_values or ()),
    )
    record = replace(
        record,
        canonical_payload={"forward": "only"},
        functions=(
            replace(function, name="run", entry="body", blocks=(renamed,)),
            *record.functions[1:],
        ),
    )
    result = control_component_program(record, {"h": add_h, "p": add_phase}, "run")
    assert result.functions[0].entry == result.functions[0].blocks[0].name == "body"
    assert result.canonical_payload is None and record.canonical_payload == {
        "forward": "only"
    }
    assert verify_program_record(result) == ()
    assert (
        len(
            bind_register_program(
                result, ((0,), (1,), (2,)), clean=(3,), entry="run"
            ).circuit.instructions
        )
        > 0
    )


def test_retained_callers_are_rejected_when_entry_gains_control_port() -> None:
    record, _ = fixture_program(1)
    function = record.functions[0]
    caller = ComponentProgramBuilder(
        Signature(function.registers or ()), record.functions, name="caller"
    )
    module = caller.finalize(caller.call_dict("main", caller.inputs))
    with pytest.raises(ValueError, match="invalid constructed register Program"):
        control_component_program(module, {"h": add_h, "p": add_phase})


@pytest.mark.parametrize("empty_shape", [False, True])
def test_array_ports_are_carried_through_without_flattening_word_identity(
    empty_shape: bool,
) -> None:
    from qualtran import Controlled, CtrlSpec

    signature = Signature(
        [Register("words", quantum_bit_type(), (0, 2) if empty_shape else (2, 2))]
    )
    native = ComponentProgramBuilder(signature)
    source = _source(signature)
    record = native.finalize(native.inputs)
    result = control_component_program(record, {})
    expected = Controlled(source.builder.finalize(**source.inputs), CtrlSpec())
    if empty_shape:
        # The original final_soqs drops empty arrays. Keep the failure visible.
        with pytest.raises(ValueError):
            expected.decompose_bloq()
    else:
        assert _native_trace(result) == _source_trace(source, expected.decompose_bloq())
    assert result.functions[0].registers is not None
    assert result.functions[0].registers[1:] == record.functions[0].registers

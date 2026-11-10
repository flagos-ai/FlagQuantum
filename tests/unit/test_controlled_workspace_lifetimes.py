"""Controlled computations retain unconditional lifetime callbacks and pool ownership."""

from __future__ import annotations

from collections import Counter
from itertools import product
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
from flagquantum.compiler.registers.component_control import (
    ControlAdder,
    control_component_hierarchy,
    control_component_via_predicate,
)
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.control.component_bookkeeping import (
    bookkeeping_component_control_system,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.layout.component_workspace import (
    allocate_word_function,
    free_word_function,
)
from flagquantum.compiler.registers.primitives.controlled_not import (
    cnot_with_controls_ir,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    source_control_predicate_leaf,
    walk_source_graph,
)
from tests.unit.test_component_control import add_h, add_phase, body
from tests.unit.test_component_control_selection import assert_gate_parameters, fixtures
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_component_structured_control import normalize_trace, original_gates
from tests.unit.test_register_constant_addition import reference_arithmetic_gates

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
PATTERNS = ((0,), (1,), *tuple(product((0, 1), repeat=2)))


def scalar_bookkeeping(record: ProgramRecord, name: str) -> ControlAdder:
    _, adder = bookkeeping_component_control_system(record, name)

    def add(
        builder: ComponentProgramBuilder,
        control: BindingCollection,
        values: dict[str, BindingCollection],
    ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
        controls, outputs = adder(builder, (control,), values)
        return controls[0], outputs

    return add


def add_cnot(
    builder: ComponentProgramBuilder,
    control: BindingCollection,
    values: dict[str, BindingCollection],
) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
    outputs = builder.call_dict(
        "ccnot",
        {
            "ctrl": np.array([control, values["ctrl"]], dtype=object),
            "target": values["target"],
        },
    )
    ctrls = outputs["ctrl"]
    assert isinstance(ctrls, np.ndarray)
    return ctrls[0], (ctrls[1], outputs["target"])


def lifetime_fixture(dirty: bool, free_dirty: bool) -> tuple[ProgramRecord, Any]:
    from qualtran import QBit
    from qualtran.bloqs.basic_gates import CNOT, Hadamard, ZPowGate
    from qualtran.bloqs.bookkeeping import Allocate, Free

    record, _, _, _ = fixtures("spectator", True)
    bit = quantum_bit_type()
    definitions = (
        *(f for f in record.functions if f.name != "main"),
        allocate_word_function("allocate", bit, dirty),
        free_word_function("free", bit, free_dirty),
        free_word_function("allocate_adjoint", bit, dirty),
        allocate_word_function("free_adjoint", bit, free_dirty),
        *body(
            "cnot",
            (Register("ctrl", bit), Register("target", bit)),
            CircuitIR(2, (Instruction("cx", (0, 1)),)),
        ),
        *body(
            "ccnot",
            (Register("ctrl", bit, (2,)), Register("target", bit)),
            CircuitIR(3, cnot_with_controls_ir(1, 2, controls=(0,), values=(1,))),
        ),
    )
    signature = Signature((Register("q", bit), Register("spectator", bit)))
    builder = ComponentProgramBuilder(signature, definitions)
    observer = _source(signature)
    q = builder.call_dict("h", {"q": builder.inputs["q"]})["q"]
    sq = observer.builder.add(Hadamard(), q=observer.inputs["q"])
    work = builder.call_dict("allocate", {})["reg"]
    sw = observer.builder.add(Allocate(QBit(), dirty))
    q, work = builder.call_tuple("cnot", {"ctrl": q, "target": work})
    sq, sw = observer.builder.add(CNOT(), ctrl=sq, target=sw)
    q = builder.call_dict("p", {"q": q})["q"]
    sq = observer.builder.add(ZPowGate(0.25), q=sq)
    q, work = builder.call_tuple("cnot", {"ctrl": q, "target": work})
    sq, sw = observer.builder.add(CNOT(), ctrl=sq, target=sw)
    builder.call_tuple("free", {"reg": work})
    observer.builder.add(Free(QBit(), free_dirty), reg=sw)
    native = builder.finalize({"q": q, "spectator": builder.inputs["spectator"]})
    source = observer.builder.finalize(q=sq, spectator=observer.inputs["spectator"])
    return native, source


def controlled_fixture(
    values: tuple[int, ...], dirty: bool, free_dirty: bool
) -> tuple[ProgramRecord, Any]:
    from qualtran import Controlled, CtrlSpec
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd

    record, source = lifetime_fixture(dirty, free_dirty)
    mapping = {
        "h": add_h,
        "p": add_phase,
        "cnot": add_cnot,
        "allocate": scalar_bookkeeping(record, "allocate"),
        "free": scalar_bookkeeping(record, "free"),
    }
    if values == (1,):
        return control_component_hierarchy(record, mapping), Controlled(
            source, CtrlSpec()
        )
    controls = (
        Register("ctrl", quantum_bit_type(), (len(values),) if len(values) > 1 else ()),
    )
    return control_component_via_predicate(
        record, mapping, controls, (ControlRegister(1, values),)
    ), ControlledViaAnd(source, CtrlSpec(cvs=values if len(values) > 1 else values[0]))


def lifetime_source_gates(
    source: Any, n: int, bits: dict[int, int], output: bool = False
) -> tuple[tuple[Instruction, ...], tuple[tuple[str, bool], ...]]:
    from qualtran.bloqs.basic_gates import CHadamard, CZPowGate, Toffoli, XGate
    from qualtran.bloqs.bookkeeping import Allocate, Free

    events: list[tuple[str, bool]] = []
    life, phase = 2 * n + 1, 2 * n + 2
    ports: SourcePorts = {"ctrl": tuple(range(n)), "q": (n,), "spectator": (n + 1,)}

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        predicate = source_control_predicate_leaf(
            bloq,
            inputs,
            tuple(range(n + 2, 2 * n + 1)),
            tuple(bits[q] for q in range(n + 2, 2 * n + 1)),
        )
        if predicate is not None:
            return predicate
        if isinstance(bloq, Allocate):
            events.append(("allocate", bloq.dirty))
            return (), {"reg": (life,)}
        if isinstance(bloq, Free):
            assert inputs == {"reg": (life,)}
            events.append(("free", bloq.dirty))
            return (), {}
        if isinstance(bloq, Toffoli):
            return (
                reference_arithmetic_gates(
                    bloq.as_composite_bloq(), inputs, (), {}, keep_exchange_leaves=True
                ),
                inputs,
            )
        if isinstance(bloq, (CHadamard, CZPowGate, XGate)):
            return original_gates(bloq, inputs, (phase,), bits), inputs
        return None

    desired = {**ports, **({"out": (life,)} if output else {})}
    gates, returned = walk_source_graph(source, ports, desired, leaf)
    assert returned == desired
    return gates, tuple(events)


def initial_and_expected(
    n: int, dirty: bool, values: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    width = 2 * n + 3
    initial = torch.zeros(1 << width, dtype=torch.complex128)
    for ctrl in product((0, 1), repeat=n):
        prefix = sum(v << i for i, v in enumerate(reversed(ctrl)))
        for q, spare, borrowed in product((0, 1), repeat=3):
            if not dirty and borrowed:
                continue
            index = (prefix * 4 + q * 2 + spare) << (n + 1) | borrowed << 1
            initial[index] = torch.tensor(
                complex(index + 1, 2 - index), dtype=torch.complex128
            )
    initial /= initial.norm()
    expected = torch.zeros_like(initial)
    for index in range(initial.numel()):
        ctrl = tuple((index >> (width - 1 - i)) & 1 for i in range(n))
        if ctrl != values:
            expected[index] += initial[index]
            continue
        old = (index >> (width - 1 - n)) & 1
        for out in (0, 1):
            target = index & ~(1 << (width - 1 - n)) | out << (width - 1 - n)
            expected[target] += (
                initial[index]
                * (-1 if old and out else 1)
                / 2**0.5
                * np.exp(1j * np.pi * out / 4)
            )
    return initial, expected


@pytest.mark.parametrize("values", PATTERNS)
@pytest.mark.parametrize("dirty,free_dirty", tuple(product((False, True), repeat=2)))
@pytest.mark.parametrize("expanded", (False, True))
def test_controlled_body_keeps_lifetimes_unconditional_and_restores_joint_state(
    values: tuple[int, ...], dirty: bool, free_dirty: bool, expanded: bool
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    module, source = controlled_fixture(values, dirty, free_dirty)
    observer = _source(Signature([]))
    composite = source.decompose_bloq()
    assert normalize_trace(_native_trace(module)) == normalize_trace(
        _source_trace(observer, composite)
    )
    root = module.functions[0]
    terminals = {
        op.callee: instance.bloq
        for op, (instance, _, _) in zip(
            root.blocks[0].operations, composite.iter_bloqsoqs(), strict=True
        )
        if op.callee is not None
    }
    graph = program_component_call_graph(module, keep=terminals.__contains__)
    assert (
        Counter({terminals[name]: count for name, count in graph.leaf_counts.items()})
        == composite.bloq_counts()
    )
    if expanded:
        module, _ = expand_component_calls(module)
    n = len(values)
    life, phase = 2 * n + 1, 2 * n + 2
    pool = tuple(q for q in range(n + 2, phase + 1) if not dirty or q != life)
    bound = bind_register_program(
        module,
        tuple((i,) for i in range(n + 2)),
        clean=pool,
        borrowed=(life,) if dirty else (),
    )
    bits = {
        g.wires[0]: g.metadata["classical_bit"]
        for g in bound.circuit.instructions
        if g.name == "measure"
    }
    gates, events = lifetime_source_gates(source, n, bits)
    assert_gate_parameters(bound.circuit.instructions, gates)
    assert (
        tuple((e.kind, e.dirty) for e in bound.events if e.wires == (life,))
        == events
        == (("allocate", dirty), ("free", free_dirty))
    )
    assert bound.outputs == tuple((i,) for i in range(n + 2))
    initial, expected = initial_and_expected(n, dirty and free_dirty, values)
    for history in product((0, 1), repeat=n):
        actual = forced_branch(initial, bound.circuit.instructions, history)
        torch.testing.assert_close(actual, expected / 2 ** (n / 2), atol=3e-12, rtol=0)
        assert torch.count_nonzero(actual[1::2]) == 0
    counts = Counter(g.name for g in bound.circuit.instructions)
    assert counts["ccx"] == 2 and counts["t"] + counts["tdg"] == 4 * n
    assert counts["measure"] == counts["reset"] == n
    model = get_cost_value(source, QECGatesCost())
    for category in (
        "t",
        "toffoli",
        "cswap",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
    ):
        assert getattr(model, category) == sum(
            count * getattr(get_cost_value(terminals[name], QECGatesCost()), category)
            for name, count in graph.leaf_counts.items()
        )
    assert model.toffoli == 2 and model.total_t_count() == 35 + 4 * (n - 1)
    assert get_cost_value(source, QubitCount()) == 2 * n + 3 == bound.circuit.n_wires


def output_fixture(
    dirty: bool, consume: bool, free_dirty: bool
) -> tuple[ProgramRecord, Any]:
    from qualtran import CtrlSpec, QBit
    from qualtran.bloqs.basic_gates import CNOT, Hadamard, ZPowGate
    from qualtran.bloqs.bookkeeping import Allocate, Free

    record, _ = lifetime_fixture(dirty, free_dirty)
    bit = quantum_bit_type()
    signature = Signature(
        (
            Register("ctrl", bit),
            Register("q", bit),
            Register("spectator", bit),
            *((Register("out", bit, side=RegisterSide.RIGHT),) if not consume else ()),
        )
    )
    builder = ComponentProgramBuilder(signature, record.functions, name="output")
    observer = _source(signature)
    ctrl, (q,) = add_h(builder, builder.inputs["ctrl"], {"q": builder.inputs["q"]})
    _, callback = Hadamard().get_ctrl_system(CtrlSpec())
    sc, sq = callback(
        observer.builder, (observer.inputs["ctrl"],), {"q": observer.inputs["q"]}
    )
    (sctrl,), (s_q,) = tuple(sc), tuple(sq)
    ctrl, (work,) = scalar_bookkeeping(record, "allocate")(builder, ctrl, {})
    _, callback = Allocate(QBit(), dirty).get_ctrl_system(CtrlSpec())
    sc, sw = callback(observer.builder, (sctrl,), {})
    (sctrl,), (s_work,) = tuple(sc), tuple(sw)
    ctrl, (q, work) = add_cnot(builder, ctrl, {"ctrl": q, "target": work})
    _, callback = CNOT().get_ctrl_system(CtrlSpec())
    sc, sq = callback(observer.builder, (sctrl,), {"ctrl": s_q, "target": s_work})
    (sctrl,), (s_q, s_work) = tuple(sc), tuple(sq)
    ctrl, (q,) = add_phase(builder, ctrl, {"q": q})
    _, callback = ZPowGate(0.25).get_ctrl_system(CtrlSpec())
    sc, sq = callback(observer.builder, (sctrl,), {"q": s_q})
    (sctrl,), (s_q,) = tuple(sc), tuple(sq)
    if consume:
        ctrl, (q, work) = add_cnot(builder, ctrl, {"ctrl": q, "target": work})
        _, callback = CNOT().get_ctrl_system(CtrlSpec())
        sc, sq = callback(observer.builder, (sctrl,), {"ctrl": s_q, "target": s_work})
        (sctrl,), (s_q, s_work) = tuple(sc), tuple(sq)
        ctrl, data = scalar_bookkeeping(record, "free")(builder, ctrl, {"reg": work})
        assert data == ()
        _, callback = Free(QBit(), free_dirty).get_ctrl_system(CtrlSpec())
        sc, sq = callback(observer.builder, (sctrl,), {"reg": s_work})
        (sctrl,) = tuple(sc)
        assert tuple(sq) == ()
    native_outputs = {"ctrl": ctrl, "q": q, "spectator": builder.inputs["spectator"]}
    source_outputs = {
        "ctrl": sctrl,
        "q": s_q,
        "spectator": observer.inputs["spectator"],
    }
    if not consume:
        native_outputs["out"] = work
        source_outputs["out"] = s_work
    return builder.finalize(native_outputs), observer.builder.finalize(**source_outputs)


OUTPUT_CASES = tuple(
    (dirty, consume, free_dirty)
    for dirty in (False, True)
    for consume in (False, True)
    for free_dirty in ((False, True) if consume else (False,))
)


@pytest.mark.parametrize("dirty,consume,free_dirty", OUTPUT_CASES)
@pytest.mark.parametrize("expanded", (False, True))
def test_returned_work_transfer_and_caller_consumption_preserve_entanglement(
    dirty: bool, consume: bool, free_dirty: bool, expanded: bool
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    module, source = output_fixture(dirty, consume, free_dirty)
    observer = _source(Signature([]))
    assert normalize_trace(_native_trace(module, "output")) == normalize_trace(
        _source_trace(observer, source)
    )
    if expanded:
        module, _ = expand_component_calls(module, "output")
    bound = bind_register_program(
        module,
        ((0,), (1,), (2,)),
        clean=(4,) if dirty else (3, 4),
        borrowed=(3,) if dirty else (),
        entry="output",
        allow_workspace_outputs=not consume,
    )
    assert bound.outputs == ((0,), (1,), (2,)) + (() if consume else ((3,),))
    if not consume:
        with pytest.raises(ValueError, match="ownership transfer"):
            bind_register_program(
                module,
                ((0,), (1,), (2,)),
                clean=(4,) if dirty else (3, 4),
                borrowed=(3,) if dirty else (),
                entry="output",
            )
    bits = {
        g.wires[0]: g.metadata["classical_bit"]
        for g in bound.circuit.instructions
        if g.name == "measure"
    }
    gates, events = lifetime_source_gates(source, 1, bits, output=not consume)
    assert_gate_parameters(bound.circuit.instructions, gates)
    assert tuple((e.kind, e.dirty) for e in bound.events if e.wires == (3,)) == events
    initial, expected = initial_and_expected(
        1, dirty and (not consume or free_dirty), (1,)
    )
    if not consume:
        entangled = torch.zeros_like(expected)
        for index in range(expected.numel()):
            target = index ^ 2 if index & 16 and index & 8 else index
            entangled[target] = expected[index]
        expected = entangled
    for bit in (0, 1):
        actual = forced_branch(initial, bound.circuit.instructions, (bit,))
        torch.testing.assert_close(actual, expected / 2**0.5, atol=3e-12, rtol=0)
        assert torch.count_nonzero(actual[1::2]) == 0
    circuit = DynamicCircuit(5, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(5, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=8, seed=41, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(8, -1), expected.expand(8, -1), atol=3e-12, rtol=0
    )
    counts = Counter(g.name for g in bound.circuit.instructions)
    assert counts["ccx"] == (2 if consume else 1)
    model = get_cost_value(source, QECGatesCost())
    assert model.toffoli == counts["ccx"] and model.total_t_count() == (
        35 if consume else 31
    )
    assert get_cost_value(source, QubitCount()) == bound.circuit.n_wires == 5


@pytest.mark.parametrize("value", (0, 1))
@pytest.mark.parametrize("dirty,free_dirty", tuple(product((False, True), repeat=2)))
def test_controlled_lifetime_exact_adjoint_restores_all_history_pairs(
    value: int, dirty: bool, free_dirty: bool
) -> None:
    from flagquantum.compiler.registers.component_adjoint import (
        adjoint_component_hierarchy,
    )
    from tests.unit.test_component_hierarchy_adjoint import source_adjoint

    module, source = controlled_fixture((value,), dirty, free_dirty)
    adjoints = {
        "ch": "ch",
        "cp": "cp_inverse",
        "ccnot": "ccnot",
        "allocate": "allocate_adjoint",
        "free": "free_adjoint",
    }
    for f in module.functions:
        if f.name.endswith("_x"):
            adjoints[f.name] = f.name
    inverse = adjoint_component_hierarchy(module, adjoints)
    inverted = source_adjoint(source.decompose_bloq())
    observer = _source(Signature([]))
    assert normalize_trace(_native_trace(inverse)) == normalize_trace(
        _source_trace(observer, inverted.decompose_bloq())
    )
    bound = bind_register_program(
        module,
        ((0,), (1,), (2,)),
        clean=(4,) if dirty else (3, 4),
        borrowed=(3,) if dirty else (),
    )
    reverse = bind_register_program(
        inverse,
        ((0,), (1,), (2,)),
        clean=(4,) if free_dirty else (3, 4),
        borrowed=(3,) if free_dirty else (),
    )
    bits = {
        g.wires[0]: g.metadata["classical_bit"]
        for g in reverse.circuit.instructions
        if g.name == "measure"
    }
    gates, events = lifetime_source_gates(inverted, 1, bits)
    assert_gate_parameters(reverse.circuit.instructions, gates)
    assert (
        tuple((e.kind, e.dirty) for e in reverse.events if e.wires == (3,))
        == events
        == (("allocate", free_dirty), ("free", dirty))
    )
    initial, _ = initial_and_expected(1, dirty and free_dirty, (value,))
    for first, second in product((0, 1), repeat=2):
        computed = forced_branch(initial, bound.circuit.instructions, (first,))
        actual = forced_branch(computed, reverse.circuit.instructions, (second,))
        torch.testing.assert_close(actual, initial / 2, atol=3e-12, rtol=0)

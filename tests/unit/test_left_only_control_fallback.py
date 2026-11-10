"""Default predicate fallback consumes the real specialized positive callback."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import product
from typing import Any

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
    control_component_via_predicate,
)
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.control.component_control_selection import (
    select_component_control_system,
)
from flagquantum.compiler.registers.control.component_extension import (
    StructuredControlAdder,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.layout.auto_partition import AutoPartition
from flagquantum.compiler.registers.layout.component_auto_partition import (
    auto_partition_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    source_control_predicate_leaf,
    source_partition_leaf,
    walk_source_graph,
)
from tests.unit.test_component_auto_partition import source_partition
from tests.unit.test_component_control_selection import assert_gate_parameters, fixtures
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_component_structured_control import normalize_trace, original_gates
from tests.unit.test_directional_control_callbacks import directional_fixture
from tests.unit.test_left_only_control_selection import (
    coherent_input,
    expected_controlled,
    unexpected_fallback,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def fallback_fixture() -> tuple[ProgramRecord, Any, Any]:
    from qualtran import Bloq
    from qualtran.bloqs.mcmt.specialized_ctrl import _get_ctrl_system_1bit_cv

    record, _, native, originals = fixtures("spectator", True)
    record = replace(
        record,
        functions=tuple(
            replace(f, name="original") if f.name == "main" else f
            for f in record.functions
        ),
    )
    data = AutoPartition((("data", (("q", 1), ("spectator", 1))),))
    identity = AutoPartition((("data", (("data", 2),)),), left_only=True)
    record = auto_partition_component_program(
        record, data, "original", "original_packed"
    )
    record = auto_partition_component_program(
        record, identity, "original_packed", "original_directional"
    )
    original = source_partition(source_partition(originals[1].subbloq, data), identity)
    positive, _ = native[1]
    definitions = {f.name: f for f in positive.functions}
    for f in record.functions:
        assert f.name not in definitions or definitions[f.name] == f
        definitions[f.name] = f
    positive = replace(positive, functions=tuple(definitions.values()))
    layout = AutoPartition(
        (("ctrl", (("ctrl", 1),)), ("data", (("q", 1), ("spectator", 1))))
    )
    identity = AutoPartition(
        (("ctrl", (("ctrl", 1),)), ("data", (("data", 2),))), left_only=True
    )
    positive = auto_partition_component_program(
        positive, layout, "main", "positive_packed"
    )
    positive = auto_partition_component_program(
        positive, identity, "positive_packed", "positive_directional"
    )
    packed = source_partition(source_partition(originals[1], layout), identity)

    def system(component: Any, ctrl_spec: Any) -> Any:
        return _get_ctrl_system_1bit_cv(
            component,
            ctrl_spec,
            current_ctrl_bit=None,
            get_ctrl_bloq_and_ctrl_reg_name=lambda cv: (
                (packed, "ctrl") if cv == 1 else None
            ),
        )

    source = type(
        "StoredDirectionalBody",
        (Bloq,),
        {
            "signature": original.signature,
            "decompose_bloq": lambda self: original.decompose_bloq(),
            "get_ctrl_system": system,
        },
    )()
    return positive, source, originals[1]


def positive_adder(
    builder: ComponentProgramBuilder,
    control: BindingCollection,
    values: dict[str, BindingCollection],
) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
    outputs = builder.call_dict("positive_directional", {"ctrl": control, **values})
    return outputs.pop("ctrl"), tuple(outputs.values())


def fallback_module(record: ProgramRecord, values: tuple[int, ...]) -> ProgramRecord:
    controls = (Register("ctrl", quantum_bit_type(), (len(values),)),)
    patterns = (ControlRegister(1, values),)
    calls: list[bool] = []

    def fallback(
        requested: tuple[Register, ...], activation: tuple[ControlRegister, ...]
    ) -> tuple[ProgramRecord, StructuredControlAdder]:
        assert requested is controls and activation is patterns
        calls.append(True)
        module = control_component_via_predicate(
            record,
            {"original_directional": positive_adder},
            requested,
            activation,
            "original_directional",
        )
        return module, unexpected_fallback

    module, _ = select_component_control_system(
        controls, patterns, unexpected_fallback, fallback
    )
    assert calls == [True]
    return module


def fallback_gates(
    source: Any,
    child: Any,
    ports: SourcePorts,
    work: tuple[int, ...],
    bits: dict[int, int],
) -> tuple[Instruction, ...]:
    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        from qualtran.bloqs.basic_gates import XGate

        if isinstance(bloq, XGate):
            return original_gates(bloq, inputs, work[-1:], bits), inputs
        partition = source_partition_leaf(bloq, inputs, desired)
        if partition is not None:
            return partition
        predicate = source_control_predicate_leaf(
            bloq, inputs, work[:-1], tuple(bits[q] for q in work[:-1])
        )
        if predicate is not None:
            return predicate
        if bloq == child:
            return original_gates(bloq, inputs, work[-1:], bits), inputs
        return None

    gates, returned = walk_source_graph(source, ports, ports, leaf)
    assert returned == ports
    return gates


@pytest.mark.parametrize(
    "values",
    ((0,), (1,), *tuple(product((0, 1), repeat=2)), *tuple(product((0, 1), repeat=3))),
)
@pytest.mark.parametrize("expanded", (False, True))
def test_directional_default_fallback_calls_actual_positive_system(
    values: tuple[int, ...], expanded: bool
) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    record, source, child = fallback_fixture()
    n = len(values)
    if n == 1:
        original = ControlledViaAnd(source, CtrlSpec(cvs=values))
        module = control_component_via_predicate(
            record,
            {"original_directional": positive_adder},
            (Register("ctrl", quantum_bit_type(), (1,)),),
            (ControlRegister(1, values),),
            "original_directional",
        )
    else:
        original, _ = source.get_ctrl_system(CtrlSpec(cvs=values))
        assert isinstance(original, ControlledViaAnd)
        module = fallback_module(record, values)
    assert verify_program_record(module) == ()
    observer = _source(Signature([]))
    assert normalize_trace(
        _native_trace(module, "original_directional")
    ) == normalize_trace(_source_trace(observer, original.decompose_bloq()))
    root = next(f for f in module.functions if f.name == "original_directional")
    terminals = {
        op.callee: instance.bloq
        for op, (instance, _, _) in zip(
            root.blocks[0].operations,
            original.decompose_bloq().iter_bloqsoqs(),
            strict=True,
        )
        if op.callee is not None
    }
    graph = program_component_call_graph(
        module, entry="original_directional", keep=terminals.__contains__
    )
    assert (
        Counter({terminals[name]: count for name, count in graph.leaf_counts.items()})
        == original.decompose_bloq().bloq_counts()
    )
    if expanded:
        module, _ = expand_component_calls(module, "original_directional")
    bound = bind_register_program(
        module,
        (*((i,) for i in range(n)), (n, n + 1)),
        clean=tuple(range(n + 2, 2 * n + 2)),
        entry="original_directional",
    )
    assert bound.outputs == (*((i,) for i in range(n)), (n, n + 1))
    assert not any(
        e.kind == "free" and set(e.wires) & set(range(n + 2)) for e in bound.events
    )
    bits = {
        g.wires[0]: g.metadata["classical_bit"]
        for g in bound.circuit.instructions
        if g.name == "measure"
    }
    assert_gate_parameters(
        bound.circuit.instructions,
        fallback_gates(
            original,
            child,
            {"ctrl": tuple(range(n)), "data": (n, n + 1)},
            tuple(range(n + 2, 2 * n + 2)),
            bits,
        ),
    )
    initial = coherent_input(n + 2, n)
    expected = expected_controlled(initial, values, n)
    for history in product((0, 1), repeat=n):
        state = forced_branch(initial, bound.circuit.instructions, history)
        torch.testing.assert_close(state, expected / 2 ** (n / 2), atol=3e-12, rtol=0)
        assert torch.count_nonzero(state.reshape(-1, 1 << n)[:, 1:]) == 0
    counts = {
        name: sum(g.name == name for g in bound.circuit.instructions)
        for name in ("t", "tdg", "measure", "reset")
    }
    assert counts["t"] + counts["tdg"] == 4 * n
    assert counts["measure"] == counts["reset"] == n
    model = get_cost_value(original, QECGatesCost())
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
    assert model.total_t_count() == 27 + 4 * (n - 1) and model.measurement == n
    assert get_cost_value(original, QubitCount()) == bound.circuit.n_wires == 2 * n + 2


def test_default_generic_control_still_rejects_directional_decomposition() -> None:
    from qualtran import Controlled, CtrlSpec, DecomposeTypeError

    record, source, _ = fallback_fixture()
    with pytest.raises(DecomposeTypeError, match="non-thru"):
        Controlled(source, CtrlSpec()).decompose_bloq()
    with pytest.raises(ValueError, match="only THRU"):
        control_component_via_predicate(
            record,
            {
                f.name: positive_adder
                for f in record.functions
                if f.name != "original_directional"
            },
            (Register("added", quantum_bit_type(), (2,)),),
            (ControlRegister(1, (1, 1)),),
            "original_directional",
        )


def test_original_fallback_cannot_zip_renamed_directional_outputs() -> None:
    from qualtran import Bloq, BloqError, CtrlSpec
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd
    from qualtran.bloqs.mcmt.specialized_ctrl import _get_ctrl_system_1bit_cv

    record, packed, _ = directional_fixture(1, True, False, 1)
    signature = type(packed.signature)(
        [r for r in packed.signature if r.name != "ctrl"]
    )

    def system(component: Any, ctrl_spec: Any) -> Any:
        return _get_ctrl_system_1bit_cv(
            component,
            ctrl_spec,
            current_ctrl_bit=None,
            get_ctrl_bloq_and_ctrl_reg_name=lambda cv: (packed, "ctrl"),
        )

    source = type(
        "RenamedDirectionalBody",
        (Bloq,),
        {"signature": signature, "get_ctrl_system": system},
    )()
    with pytest.raises(BloqError, match="spectator") as failure:
        ControlledViaAnd(source, CtrlSpec(cvs=(1, 0))).decompose_bloq()
    assert type(failure.value).__name__ == "BloqError"
    # Native rejects the source's all-declarations/output zip before invoking the adder.
    with pytest.raises(ValueError, match="cannot pair directional output names"):
        control_component_via_predicate(
            record,
            {"packed0": positive_adder},
            (Register("added", quantum_bit_type(), (2,)),),
            (ControlRegister(1, (1, 0)),),
            "packed0",
        )


@pytest.mark.parametrize("values", tuple(product((0, 1), repeat=2)))
def test_default_directional_fallback_remains_appendable(
    values: tuple[int, ...],
) -> None:
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    record, _, _ = fallback_fixture()
    module = fallback_module(record, values)
    bound = bind_register_program(
        module, ((0,), (1,), (2, 3)), clean=(4, 5), entry="original_directional"
    )
    initial = coherent_input(4, 2)
    circuit = DynamicCircuit(6, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(6, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=8, seed=37, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(8, -1),
        expected_controlled(initial, values, 2).expand(8, -1),
        atol=3e-12,
        rtol=0,
    )


def inverse_module(module: ProgramRecord) -> ProgramRecord:
    import re

    from flagquantum.compiler.registers.component_adjoint import (
        adjoint_component_hierarchy,
    )
    from flagquantum.compiler.registers.layout.component_auto_partition import (
        _partition_program,
    )

    adjoints = {"ch": "ch", "cp": "cp_inverse"}
    functions = {f.name: f for f in module.functions}
    for f in module.functions:
        if re.fullmatch(r"__fq_partition_[0-9]+_split[0-9]+", f.name):
            name = f.name.replace("_split", "_join")
            assert f.registers is not None
            if name not in functions:
                for inverse in _partition_program(
                    tuple(Signature(f.registers).rights()), name, True
                ).functions:
                    functions[inverse.name] = inverse
            adjoints[f.name] = name
            adjoints[name] = f.name
        elif re.fullmatch(r"__fq_predicate_[0-9]+_(compute|erase)", f.name):
            adjoints[f.name] = (
                f.name.removesuffix("compute") + "erase"
                if f.name.endswith("compute")
                else f.name.removesuffix("erase") + "compute"
            )
    return adjoint_component_hierarchy(
        replace(module, functions=tuple(functions.values())),
        adjoints,
        "original_directional",
    )


@pytest.mark.parametrize("values", tuple(product((0, 1), repeat=2)))
def test_default_fallback_exact_adjoint_all_history_pairs(
    values: tuple[int, ...],
) -> None:
    from qualtran import CtrlSpec

    from tests.unit.test_component_hierarchy_adjoint import source_adjoint

    record, source, child = fallback_fixture()
    module = fallback_module(record, values)
    inverse = inverse_module(module)
    bound = bind_register_program(
        module, ((0,), (1,), (2, 3)), clean=(4, 5), entry="original_directional"
    )
    reverse = bind_register_program(
        inverse, ((0,), (1,), (2, 3)), clean=(4, 5), entry="original_directional"
    )
    assert reverse.outputs == bound.outputs
    original, _ = source.get_ctrl_system(CtrlSpec(cvs=values))
    inverted = source_adjoint(original.decompose_bloq())
    observer = _source(Signature([]))
    assert normalize_trace(
        _native_trace(inverse, "original_directional")
    ) == normalize_trace(_source_trace(observer, inverted.decompose_bloq()))
    bits = {
        g.wires[0]: g.metadata["classical_bit"]
        for g in reverse.circuit.instructions
        if g.name == "measure"
    }
    assert_gate_parameters(
        reverse.circuit.instructions,
        fallback_gates(
            inverted, child.adjoint(), {"ctrl": (0, 1), "data": (2, 3)}, (4, 5), bits
        ),
    )
    initial = coherent_input(4, 2)
    for first in product((0, 1), repeat=2):
        computed = forced_branch(initial, bound.circuit.instructions, first)
        for second in product((0, 1), repeat=2):
            actual = forced_branch(computed, reverse.circuit.instructions, second)
            torch.testing.assert_close(actual, initial / 4, atol=3e-12, rtol=0)
            assert torch.count_nonzero(actual.reshape(-1, 4)[:, 1:]) == 0

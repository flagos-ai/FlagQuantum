"""Original nested-control callbacks merge clauses before typed predicate binding."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler._component_trace.program_call_graph import (
    program_component_call_graph,
)
from flagquantum.compiler.registers.component_adjoint import adjoint_component_hierarchy
from flagquantum.compiler.registers.component_builder import (
    BindingCollection,
    ComponentProgramBuilder,
)
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.control.component_extension import (
    extend_component_control_system,
)
from flagquantum.compiler.registers.primitives.controlled_rotation import (
    controlled_z_power_ir,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_control import add_h, add_phase, body, fixture_program
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_component_hierarchy_adjoint import source_adjoint
from tests.unit.test_component_structured_control import (
    MULTI,
    SINGLE,
    native_controls,
    normalize_trace,
    original_gates,
)
from tests.unit.test_register_control_predicate import ControlCase, control_fixture
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

PAIRS = (
    (SINGLE[0], SINGLE[1]),
    (SINGLE[2], SINGLE[3]),
    (MULTI[0], SINGLE[0]),
    (SINGLE[1], MULTI[0]),
    (MULTI[0], MULTI[0]),
    (SINGLE[0], MULTI[2]),
    (SINGLE[0], MULTI[1]),
    (
        SINGLE[0],
        (("QBit", 1, (), (0,), "unsigned"), ("QUInt", 2, (), (2,), "unsigned")),
    ),
)


def extended_fixture(
    old_case: ControlCase,
    new_case: ControlCase,
    spectator: str,
) -> tuple[
    ProgramRecord,
    ProgramRecord,
    Any,
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
]:
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd

    record, positive = fixture_program(1, spectator)
    old_spec, old_patterns, _ = control_fixture(old_case)
    new_spec, new_patterns, _ = control_fixture(new_case)
    old_source = ControlledViaAnd(positive.subbloq, old_spec)
    source, source_adder = old_source.get_ctrl_system(new_spec)
    old_controls = native_controls(old_case, tuple(old_source.ctrl_reg_names))
    new_controls = native_controls(
        new_case, tuple(f"added_{i}" for i in range(len(new_case)))
    )
    module, adder = extend_component_control_system(
        record,
        {"h": add_h, "p": add_phase},
        old_controls,
        old_patterns,
        new_controls,
        new_patterns,
    )
    data = tuple(record.functions[0].registers or ())
    signature = Signature((*new_controls, *old_controls, *data))
    caller = ComponentProgramBuilder(signature, module.functions, name="caller")
    observer = _source(signature)
    source_new = tuple(observer.inputs[r.name] for r in new_controls)
    native_new = tuple(caller.inputs[r.name] for r in new_controls)
    old_names = tuple(r.name for r in (*old_controls, *data))
    native_values = {name: caller.inputs[name] for name in old_names}
    source_values = {name: observer.inputs[name] for name in old_names}
    native_returns = adder(caller, native_new, native_values)
    source_returns = source_adder(observer.builder, source_new, source_values)
    assert tuple(native_values) == tuple(source_values) == tuple(r.name for r in data)
    native_outputs = dict(
        zip(
            (r.name for r in signature),
            (*native_returns[0], *native_returns[1]),
            strict=True,
        )
    )
    source_outputs = dict(
        zip(
            (r.name for r in signature),
            (*source_returns[0], *source_returns[1]),
            strict=True,
        )
    )
    parent = caller.finalize(native_outputs)
    source_parent = observer.builder.finalize(**source_outputs)
    assert normalize_trace(_native_trace(parent, "caller")) == normalize_trace(
        _source_trace(observer, source_parent)
    )
    assert normalize_trace(_native_trace(module)) == normalize_trace(
        _source_trace(observer, source.decompose_bloq())
    )
    assert tuple(
        observer.signature(Signature(module.functions[0].registers or ()))
    ) == tuple(source.signature)
    assert verify_program_record(parent) == ()
    count = sum(len(p.bits()) for p in (*new_patterns, *old_patterns))
    offset = 0
    groups = []
    inputs: list[tuple[int, ...]] = []
    for register, pattern in zip(
        (*new_controls, *old_controls), (*new_patterns, *old_patterns), strict=True
    ):
        size = len(pattern.bits())
        group = tuple(range(offset, offset + size))
        groups.append(group)
        inputs.extend(
            group[i : i + pattern.width] for i in range(0, size, pattern.width)
        )
        offset += size
    return parent, module, source, tuple(groups), (*inputs, (count,), (count + 1,))


@pytest.mark.parametrize("old_case,new_case", PAIRS)
@pytest.mark.parametrize("spectator", ("spectator", "ctrl", "ctrl2"))
@pytest.mark.parametrize("expanded", (False, True))
def test_original_extension_graph_gates_all_branches_and_resources(
    old_case: ControlCase,
    new_case: ControlCase,
    spectator: str,
    expanded: bool,
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    parent, module, source, groups, inputs = extended_fixture(
        old_case, new_case, spectator
    )
    n = sum(map(len, groups))
    work = tuple(range(n + 2, 2 * n + 2))
    if expanded:
        parent, _ = expand_component_calls(parent, "caller")
    bound = bind_register_program(parent, inputs, clean=work, entry="caller")
    assert bound.outputs == inputs
    gates = bound.circuit.instructions
    measurements = {
        g.wires[0]: g.metadata["classical_bit"] for g in gates if g.name == "measure"
    }
    ports = dict(zip(source.ctrl_reg_names, groups, strict=True))
    ports.update({"q": (n,), spectator: (n + 1,)})
    reference = original_gates(source, ports, work, measurements)
    assert_source(gates, reference)
    assert [(g.name, g.wires, dict(g.params)) for g in gates if g.params] == [
        (g.name, g.wires, dict(g.params)) for g in reference if g.params
    ]
    counts = Counter(g.name for g in gates)
    assert counts["measure"] == counts["reset"] == n
    assert counts["t"] + counts["tdg"] == 4 * n
    assert get_cost_value(source, QubitCount()) == bound.circuit.n_wires == 2 * n + 2
    terminals = {
        op.callee: instance.bloq
        for op, (instance, _, _) in zip(
            module.functions[0].blocks[0].operations,
            source.decompose_bloq().iter_bloqsoqs(),
            strict=True,
        )
    }
    graph = program_component_call_graph(module, keep=terminals.__contains__)
    assert (
        Counter({terminals[name]: count for name, count in graph.leaf_counts.items()})
        == source.bloq_counts()
    )
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
    _, old_patterns, _ = control_fixture(old_case)
    _, new_patterns, _ = control_fixture(new_case)
    activation = int(
        "".join(str(b) for p in (*new_patterns, *old_patterns) for b in p.bits()), 2
    )
    initial = torch.zeros(1 << (2 * n + 2), dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for control, q, spare in product(range(1 << n), (0, 1), (0, 1)):
        index = ((control << 2) | (q << 1) | spare) << n
        amplitude = torch.tensor(
            complex(control + q + 1, spare - control), dtype=torch.complex128
        ) / (1 << (n + 2))
        initial[index] = amplitude
        if control != activation:
            expected[index] = amplitude
        else:
            for out in (0, 1):
                target = ((control << 2) | (out << 1) | spare) << n
                expected[target] += (
                    amplitude
                    * (-1 if q and out else 1)
                    / 2**0.5
                    * np.exp(1j * np.pi * out / 4)
                )
    for history in product((0, 1), repeat=n):
        torch.testing.assert_close(
            forced_branch(initial, gates, history),
            expected * 2 ** (-n / 2),
            atol=3e-12,
            rtol=0,
        )


def test_missing_old_controls_preserve_source_partial_pop_timing() -> None:
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd

    record, positive = fixture_program(1)
    old_case: ControlCase = (("QBit", 1, (), (0,), "unsigned"),) * 2
    old_spec, old_patterns, _ = control_fixture(old_case)
    new_spec, new_patterns, _ = control_fixture(SINGLE[0])
    old_source = ControlledViaAnd(positive.subbloq, old_spec)
    source, source_adder = old_source.get_ctrl_system(new_spec)
    old = native_controls(old_case, tuple(old_source.ctrl_reg_names))
    new = native_controls(SINGLE[0], ("added",))
    module, adder = extend_component_control_system(
        record, {"h": add_h, "p": add_phase}, old, old_patterns, new, new_patterns
    )
    signature = Signature((*new, *old, *(record.functions[0].registers or ())))
    native = ComponentProgramBuilder(signature, module.functions, name="caller")
    observer = _source(signature)
    dictionaries = []
    for builder, callback in ((native, adder), (observer, source_adder)):
        values = {old[0].name: builder.inputs[old[0].name], "q": builder.inputs["q"]}
        dictionaries.append(values)
        with pytest.raises(KeyError, match=old[1].name):
            callback(
                builder if builder is native else observer.builder,
                (builder.inputs["added"],),
                values,
            )
    assert tuple(dictionaries[0]) == tuple(dictionaries[1]) == ("q",)


@pytest.mark.parametrize(
    "failure", ("old_collision", "old_empty", "new_empty", "new_dtype")
)
def test_invalid_extension_preserves_original_record(failure: str) -> None:
    record, _ = fixture_program(1)
    old: tuple[Register, ...] = (Register("old", quantum_bit_type()),)
    new: tuple[Register, ...] = (Register("new", quantum_bit_type()),)
    _, patterns, _ = control_fixture(SINGLE[0])
    if failure == "old_collision":
        old = (Register("q", quantum_bit_type()),)
    elif failure == "old_empty":
        old = ()
    elif failure == "new_empty":
        new = ()
    elif failure == "new_dtype":
        new = (Register("new", RegisterType("signed", 1)),)
    before = repr(record)
    with pytest.raises(ValueError):
        extend_component_control_system(
            record, {"h": add_h, "p": add_phase}, old, patterns, new, patterns
        )
    assert repr(record) == before


@pytest.mark.parametrize("old_case", (SINGLE[0], MULTI[0]))
@pytest.mark.parametrize("expanded", (False, True))
def test_extended_exact_adjoint_restores_every_pair_of_measurement_histories(
    old_case: ControlCase, expanded: bool
) -> None:
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd

    record, positive = fixture_program(1)
    bit = quantum_bit_type()
    record = replace(
        record,
        functions=(
            *record.functions,
            *body(
                "p_inverse",
                (Register("q", bit),),
                CircuitIR(1, (Instruction("phase", (0,), {"theta": -np.pi / 4}),)),
            ),
            *body(
                "cp_inverse",
                (Register("q", bit, (2,)),),
                CircuitIR(3, controlled_z_power_ir(0, 1, -0.25, 2, 0)),
            ),
        ),
    )
    inverse = adjoint_component_hierarchy(record, {"h": "h", "p": "p_inverse"})
    old_spec, old_patterns, _ = control_fixture(old_case)
    new_spec, new_patterns, _ = control_fixture(SINGLE[1])
    old_source = ControlledViaAnd(positive.subbloq, old_spec)
    old = native_controls(old_case, tuple(old_source.ctrl_reg_names))
    new = native_controls(SINGLE[1], ("added",))

    def add_inverse(
        builder: ComponentProgramBuilder,
        control: BindingCollection,
        values: dict[str, BindingCollection],
    ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
        result = np.asarray(
            builder.call_dict(
                "cp_inverse", {"q": np.array([control, values["q"]], dtype=object)}
            )["q"]
        )
        return result[0], (result[1],)

    n = 1 + sum(len(p.bits()) for p in old_patterns)
    inputs = tuple((i,) for i in range(n + 2))
    work = tuple(range(n + 2, 2 * n + 2))
    circuits = []
    sources = (positive.subbloq, source_adjoint(positive.subbloq))
    for (module, mapping), source_body in zip(
        (
            (record, {"h": add_h, "p": add_phase}),
            (inverse, {"h": add_h, "p_inverse": add_inverse}),
        ),
        sources,
        strict=True,
    ):
        controlled, _ = extend_component_control_system(
            module, mapping, old, old_patterns, new, new_patterns
        )
        original, _ = ControlledViaAnd(source_body, old_spec).get_ctrl_system(new_spec)
        if expanded:
            controlled, _ = expand_component_calls(controlled)
        bound = bind_register_program(controlled, inputs, clean=work)
        gates = bound.circuit.instructions
        measurements = {
            g.wires[0]: g.metadata["classical_bit"]
            for g in gates
            if g.name == "measure"
        }
        ports = {
            original.ctrl_reg_names[0]: (0,),
            original.ctrl_reg_names[1]: tuple(range(1, n)),
            "q": (n,),
            "spectator": (n + 1,),
        }
        assert_source(gates, original_gates(original, ports, work, measurements))
        circuits.append(gates)
    initial = torch.zeros(1 << (2 * n + 2), dtype=torch.complex128)
    initial[:: 1 << n] = torch.tensor(
        [complex(i + 1, 2 * i - 3) for i in range(1 << (n + 2))], dtype=torch.complex128
    )
    initial /= initial.norm()
    for first, second in product(product((0, 1), repeat=n), repeat=2):
        state = forced_branch(initial, circuits[0], first)
        torch.testing.assert_close(
            forced_branch(state, circuits[1], second),
            initial / 2**n,
            atol=3e-12,
            rtol=0,
        )


def test_repeated_extensions_keep_one_combined_predicate_and_existing_append() -> None:
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    record, positive = fixture_program(1, "ctrl2")
    spec, patterns, _ = control_fixture(SINGLE[0])
    source = ControlledViaAnd(positive.subbloq, spec)
    old = native_controls(SINGLE[0], tuple(source.ctrl_reg_names))
    combined_patterns = patterns
    module = record
    for _ in range(2):
        new = native_controls(SINGLE[0], ("added",))
        module, _ = extend_component_control_system(
            record, {"h": add_h, "p": add_phase}, old, combined_patterns, new, patterns
        )
        source, _ = source.get_ctrl_system(spec)
        combined_patterns = (*patterns, *combined_patterns)
        old = tuple((module.functions[0].registers or ())[: len(combined_patterns)])
        assert tuple(
            _source(Signature([])).signature(
                Signature(module.functions[0].registers or ())
            )
        ) == tuple(source.signature)
        assert len(module.functions[0].blocks[0].operations) == 3
    inputs = tuple((i,) for i in range(5))
    bound = bind_register_program(module, inputs, clean=(5, 6, 7))
    gates = bound.circuit.instructions
    assert sum(g.name == "measure" for g in gates) == 3
    ports = dict(zip(source.ctrl_reg_names, ((0,), (1,), (2,)), strict=True))
    ports.update({"q": (3,), "ctrl2": (4,)})
    bits = {
        g.wires[0]: g.metadata["classical_bit"] for g in gates if g.name == "measure"
    }
    assert_source(gates, original_gates(source, ports, (5, 6, 7), bits))
    initial = torch.zeros(256, dtype=torch.complex128)
    initial[::8] = torch.tensor(
        [complex(i + 1, 5 - i) for i in range(32)], dtype=torch.complex128
    )
    initial /= initial.norm()
    expected = forced_branch(initial, gates, (0, 0, 0)) * 2**1.5
    circuit = DynamicCircuit(8, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(8, gates))
    result = run_dynamic(circuit, shots=8, seed=29, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(8, -1), expected.expand(8, -1), atol=3e-12, rtol=0
    )


@pytest.mark.parametrize("extra_count", (0, 2))
def test_original_zip_arity_and_failure_consumption(extra_count: int) -> None:
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd

    record, positive = fixture_program(1)
    spec, patterns, _ = control_fixture(SINGLE[0])
    old_source = ControlledViaAnd(positive.subbloq, spec)
    _, source_adder = old_source.get_ctrl_system(spec)
    old = native_controls(SINGLE[0], tuple(old_source.ctrl_reg_names))
    new = native_controls(SINGLE[0], ("added",))
    module, adder = extend_component_control_system(
        record, {"h": add_h, "p": add_phase}, old, patterns, new, patterns
    )
    signature = Signature((*new, *old, *(record.functions[0].registers or ())))
    native = ComponentProgramBuilder(signature, module.functions, name="caller")
    observer = _source(signature)
    for builder, callback in ((native, adder), (observer, source_adder)):
        values = {
            r.name: builder.inputs[r.name]
            for r in (*old, *(record.functions[0].registers or ()))
        }
        with pytest.raises(ValueError):
            callback(
                builder if builder is native else observer.builder,
                (builder.inputs["added"],) * extra_count,
                values,
            )
        assert tuple(values) == ("q", "spectator")

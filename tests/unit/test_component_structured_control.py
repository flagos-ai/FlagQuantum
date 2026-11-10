"""Pinned structured controls retain typed lifetimes, phases and measured cleanup."""

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
from flagquantum.compiler.registers.component_control import (
    control_component_via_predicate,
)
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.control.component_predicate import (
    control_predicate_program,
)
from flagquantum.compiler.registers.control.control_predicate import ControlRegister
from flagquantum.compiler.registers.primitives.controlled_rotation import (
    controlled_z_power_ir,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import verify_program_record
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    source_control_predicate_leaf,
    walk_source_graph,
)
from tests.unit.test_component_control import add_h, add_phase, body, fixture_program
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_register_control_predicate import (
    CASES,
    ControlCase,
    control_fixture,
)
from tests.unit.test_register_phase_gradient import source_gradient_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

SINGLE: tuple[ControlCase, ...] = tuple(
    (("QBit", 1, shape, (value,), "unsigned"),)
    for shape, value in product(((), (1,)), (0, 1))
)
MULTI: tuple[ControlCase, ...] = (
    *CASES[:3],
    (("QUInt", 3, (), (5,), "unsigned"), ("QAny", 2, (), (2,), "unsigned")),
)


def native_controls(case: ControlCase, names: tuple[str, ...]) -> tuple[Register, ...]:
    encodings = {
        "QBit": "bit",
        "QAny": "opaque",
        "QUInt": "uint",
        "BQUInt": "uint",
        "QInt": "signed",
        "QIntOnesComp": "ones_complement",
        "QMontgomeryUInt": "montgomery",
    }
    return tuple(
        Register(
            name,
            RegisterType(
                encodings[kind],
                width,
                parameters=(
                    (("bound", 3),)
                    if kind == "BQUInt"
                    else (("modulus", 5),) if kind == "QMontgomeryUInt" else ()
                ),
            ),
            shape,
        )
        for name, (kind, width, shape, _, _) in zip(names, case, strict=True)
    )


def normalize_trace(trace: list[Any]) -> list[Any]:
    # Named connections and declared output order carry semantics. Source adder
    # input dictionaries insert the control after the data ports.
    return [(position, dict(inputs), outputs) for position, inputs, outputs in trace]


def original_gates(
    source: Any, ports: SourcePorts, work: tuple[int, ...], measurements: dict[int, int]
) -> tuple[Instruction, ...]:
    from qualtran.bloqs.basic_gates import CHadamard, CZPowGate, XGate

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        predicate = source_control_predicate_leaf(
            bloq, inputs, work[:-1], tuple(measurements[w] for w in work[:-1])
        )
        if predicate is not None:
            return predicate
        if isinstance(bloq, XGate):
            return (Instruction("x", inputs["q"]),), inputs
        if isinstance(bloq, CHadamard):
            return (Instruction("ch", (*inputs["ctrl"], *inputs["target"])),), inputs
        if isinstance(bloq, CZPowGate):
            gates, _ = source_gradient_gates(
                bloq, inputs, inputs, work=work[-1], bit=measurements[work[-1]]
            )
            return gates, inputs
        return None

    gates, outputs = walk_source_graph(source, ports, ports, leaf)
    assert outputs == ports
    return gates


@pytest.mark.parametrize("case", (*SINGLE, *MULTI))
@pytest.mark.parametrize("spectator", ("spectator", "ctrl"))
@pytest.mark.parametrize("expanded", (False, True))
def test_source_graph_gates_all_histories_phases_and_private_work(
    case: ControlCase, spectator: str, expanded: bool
) -> None:
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    record, positive = fixture_program(1, spectator)
    spec, patterns, groups = control_fixture(case)
    source = ControlledViaAnd(positive.subbloq, spec)
    controls = native_controls(case, tuple(source.ctrl_reg_names))
    module = control_component_via_predicate(
        record, {"h": add_h, "p": add_phase}, controls, patterns
    )
    assert verify_program_record(module) == ()
    assert tuple(
        _source(Signature([])).signature(Signature(module.functions[0].registers or ()))
    ) == tuple(source.signature)
    assert normalize_trace(_native_trace(module)) == normalize_trace(
        _source_trace(_source(Signature([])), source.decompose_bloq())
    )
    terminals = {
        op.callee: instance.bloq
        for op, (instance, _, _) in zip(
            module.functions[0].blocks[0].operations,
            source.decompose_bloq().iter_bloqsoqs(),
            strict=True,
        )
    }
    graph = program_component_call_graph(module, keep=terminals.__contains__)
    weighted = Counter(
        {terminals[name]: count for name, count in graph.leaf_counts.items()}
    )
    assert weighted == source.bloq_counts()
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
    if expanded:
        module, _ = expand_component_calls(module)
    n = sum(map(len, groups))
    data = n + 2
    work = tuple(range(data, data + max(0, n - 1) + 1))
    inputs = tuple(
        group[i : i + pattern.width]
        for group, pattern in zip(groups, patterns, strict=True)
        for i in range(0, len(group), pattern.width)
    ) + ((n,), (n + 1,))
    bound = bind_register_program(module, inputs, clean=work)
    assert bound.outputs == inputs
    gates = bound.circuit.instructions
    measurements = {
        g.wires[0]: g.metadata["classical_bit"] for g in gates if g.name == "measure"
    }
    ports = dict(zip(source.ctrl_reg_names, groups, strict=True))
    ports.update({"q": (n,), spectator: (n + 1,)})
    reference = original_gates(source, ports, work, measurements)
    assert_source(gates, reference)
    counts = Counter(g.name for g in gates)
    assert counts["measure"] == n
    assert counts["t"] + counts["tdg"] == 4 * n
    model = get_cost_value(source, QECGatesCost())
    assert model.measurement == model.and_bloq == n
    # Independent source models count a CZPow rotation through its own abstract
    # categories. Primitive T gates are retained separately above.
    assert model.t == 1 and model.rotation == 2
    assert model.total_t_count() == 4 * (n - 1) + 27
    state = torch.zeros(1 << (data + len(work)), dtype=torch.complex128)
    expected = torch.zeros_like(state)
    activation = int("".join(str(bit) for p in patterns for bit in p.bits()), 2)
    for control, q, spectator_bit in product(range(1 << n), (0, 1), (0, 1)):
        index = ((control << 2) | (q << 1) | spectator_bit) << len(work)
        amplitude = complex(control + q + 1, spectator_bit - control - 1) / (
            1 << (n + 2)
        )
        state[index] = torch.tensor(amplitude, dtype=torch.complex128)
        if control != activation:
            expected[index] = torch.tensor(amplitude, dtype=torch.complex128)
            continue
        for out in (0, 1):
            target = ((control << 2) | (out << 1) | spectator_bit) << len(work)
            expected[target] += (
                amplitude
                * (-1 if q and out else 1)
                / 2**0.5
                * np.exp(1j * np.pi * out / 4)
            )
    for history in product((0, 1), repeat=n):
        torch.testing.assert_close(
            forced_branch(state, gates, history),
            expected * 2 ** (-n / 2),
            atol=3e-12,
            rtol=0,
        )


@pytest.mark.parametrize("case", MULTI)
def test_directional_predicate_ports_match_original(case: ControlCase) -> None:
    from qualtran.bloqs.mcmt.ctrl_spec_and import CtrlSpecAnd

    spec, patterns, _ = control_fixture(case)
    source = CtrlSpecAnd(spec)
    controls = native_controls(case, tuple(r.name for r in source.control_registers))
    for erase in (False, True):
        module = control_predicate_program(controls, patterns, uncompute=erase)
        registers = Signature(module.functions[0].registers or ())
        expected = source.adjoint().signature if erase else source.signature
        assert tuple(_source(Signature([])).signature(registers)) == tuple(expected)
        assert verify_program_record(module) == ()


@pytest.mark.parametrize(
    "failure",
    (
        "empty",
        "arity",
        "shape",
        "width",
        "encoding",
        "classical",
        "side",
        "names",
        "single_word",
        "single_matrix",
    ),
)
def test_invalid_control_descriptors_fail_without_modifying_program(
    failure: str,
) -> None:
    record, _ = fixture_program(1)
    controls: tuple[Register, ...] = (Register("ctrl", quantum_bit_type()),)
    patterns: tuple[ControlRegister, ...] = (ControlRegister(1, (0,)),)
    if failure == "empty":
        controls, patterns = (), ()
    elif failure == "arity":
        patterns = ()
    elif failure == "shape":
        patterns = (ControlRegister(1, (0, 1)),)
    elif failure == "width":
        patterns = (ControlRegister(2, (0,)),)
    elif failure == "encoding":
        controls = (Register("ctrl", RegisterType("fixed", 1)),)
    elif failure == "classical":
        controls = (Register("ctrl", RegisterType("bit", 1, 1)),)
    elif failure == "side":
        controls = (Register("ctrl", quantum_bit_type(), side=RegisterSide.LEFT),)
    elif failure == "names":
        controls = (Register("q", quantum_bit_type()),)
    elif failure == "single_word":
        controls = (Register("ctrl", RegisterType("uint", 1)),)
    elif failure == "single_matrix":
        controls = (Register("ctrl", quantum_bit_type(), (1, 1)),)
    before = repr(record)
    with pytest.raises(ValueError):
        control_component_via_predicate(
            record, {"h": add_h, "p": add_phase}, controls, patterns
        )
    assert repr(record) == before


def test_generated_prefix_collision_and_canonical_payload() -> None:
    record, _ = fixture_program(1)
    record = replace(
        record,
        canonical_payload={"old": "forward"},
        functions=(
            *record.functions,
            replace(record.functions[-1], name="__fq_predicate_0_reserved"),
        ),
    )
    module = control_component_via_predicate(
        record,
        {"h": add_h, "p": add_phase},
        (Register("ctrl", quantum_bit_type()),),
        (ControlRegister(1, (0,)),),
    )
    assert module.canonical_payload is None
    assert any(f.name == "__fq_predicate_1_positive" for f in module.functions)


@pytest.mark.parametrize("case", CASES[3:])
def test_typed_boundary_rejects_descriptors_outside_core_admission(
    case: ControlCase,
) -> None:
    _, patterns, _ = control_fixture(case)
    controls = native_controls(case, tuple(f"c{i}" for i in range(len(case))))
    with pytest.raises(ValueError):
        control_predicate_program(controls, patterns)


@pytest.mark.parametrize("case", (SINGLE[0], MULTI[0], MULTI[2]))
@pytest.mark.parametrize("expanded", (False, True))
def test_exact_adjoint_body_restores_all_forward_inverse_histories(
    case: ControlCase, expanded: bool
) -> None:
    bit = quantum_bit_type()
    record, _ = fixture_program(1)
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
    _, patterns, groups = control_fixture(case)
    controls = native_controls(case, ("ctrl",))

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

    forward = control_component_via_predicate(
        record, {"h": add_h, "p": add_phase}, controls, patterns
    )
    backwards = control_component_via_predicate(
        inverse, {"h": add_h, "p_inverse": add_inverse}, controls, patterns
    )
    n = sum(map(len, groups))
    work = tuple(range(n + 2, 2 * n + 2))
    inputs = tuple(
        group[i : i + pattern.width]
        for group, pattern in zip(groups, patterns, strict=True)
        for i in range(0, len(group), pattern.width)
    ) + ((n,), (n + 1,))
    circuits = []
    for module in (forward, backwards):
        if expanded:
            module, _ = expand_component_calls(module)
        circuits.append(
            bind_register_program(module, inputs, clean=work).circuit.instructions
        )
    initial = torch.zeros(1 << (2 * n + 2), dtype=torch.complex128)
    initial[:: 1 << n] = torch.tensor(
        [complex(i + 1, 3 - i) for i in range(1 << (n + 2))], dtype=torch.complex128
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


@pytest.mark.parametrize("case", (SINGLE[0], MULTI[0]))
def test_existing_append_protocol_and_specialization_replacement(
    case: ControlCase,
) -> None:
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    record, _ = fixture_program(1)
    _, patterns, groups = control_fixture(case)
    controls = native_controls(case, ("ctrl",))
    calls = []

    def replacement(
        builder: ComponentProgramBuilder,
        control: BindingCollection,
        values: dict[str, BindingCollection],
    ) -> tuple[BindingCollection, tuple[BindingCollection, ...]]:
        calls.append("specialized_h")
        return add_h(builder, control, values)

    modules = [
        control_component_via_predicate(
            record, {"h": adder, "p": add_phase}, controls, patterns
        )
        for adder in (add_h, replacement)
    ]
    n = sum(map(len, groups))
    inputs = tuple((i,) for i in range(n + 2))
    work = tuple(range(n + 2, 2 * n + 2))
    bound = [bind_register_program(m, inputs, clean=work) for m in modules]
    assert calls == ["specialized_h"] and bound[0].circuit == bound[1].circuit
    initial = torch.zeros(1 << (2 * n + 2), dtype=torch.complex128)
    initial[:: 1 << n] = torch.tensor(
        [complex(i + 1, i - 2) for i in range(1 << (n + 2))], dtype=torch.complex128
    )
    initial /= initial.norm()
    expected = forced_branch(initial, bound[0].circuit.instructions, (0,) * n) * 2 ** (
        n / 2
    )
    circuit = DynamicCircuit(2 * n + 2, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(2 * n + 2, bound[0].circuit.instructions))
    result = run_dynamic(circuit, shots=8, seed=29, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(8, -1), expected.expand(8, -1), atol=3e-12, rtol=0
    )


@pytest.mark.parametrize(
    "failure", ("one_bit", "direction", "reserved", "no_bits", "duplicate")
)
def test_predicate_rejects_invalid_direction_and_ports(failure: str) -> None:
    controls: tuple[Register, ...] = (Register("ctrl", quantum_bit_type(), (2,)),)
    patterns: tuple[ControlRegister, ...] = (ControlRegister(1, (0, 1)),)
    direction: bool = False
    if failure == "one_bit":
        controls, patterns = (Register("ctrl", quantum_bit_type()),), (
            ControlRegister(1, (0,)),
        )
    elif failure == "direction":
        # Runtime callers can supply values outside the annotation.
        kwargs: Any = {"uncompute": 1}
        with pytest.raises(ValueError, match="boolean direction"):
            control_predicate_program(controls, patterns, **kwargs)
        return
    elif failure == "reserved":
        controls = (Register("target", quantum_bit_type(), (2,)),)
    elif failure == "no_bits":
        controls, patterns = (Register("ctrl", quantum_bit_type(), (0,)),), (
            ControlRegister(1, ()),
        )
    elif failure == "duplicate":
        controls = (Register("ctrl", quantum_bit_type(), (2,)),) * 2
        patterns = patterns * 2
    with pytest.raises(ValueError):
        control_predicate_program(controls, patterns, uncompute=direction)

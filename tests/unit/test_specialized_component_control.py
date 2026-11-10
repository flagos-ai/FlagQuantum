"""Stored specialized helpers match original packed controls and measured channels."""

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
from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_control import control_component_hierarchy
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.control.specialized_component_control import (
    multi_controlled_component_program,
    specialized_component_control_system,
)
from flagquantum.compiler.registers.primitives.controlled_rotation import (
    controlled_z_power_ir,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_source_graph import (
    SourcePorts,
    SourceResult,
    walk_source_graph,
)
from tests.unit.test_component_control import add_h, add_phase, body, fixture_program
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_component_hierarchy_adjoint import source_adjoint
from tests.unit.test_component_structured_control import normalize_trace, original_gates
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

PATTERNS = (
    *tuple(product((0, 1), repeat=2)),
    *tuple(product((0, 1), repeat=3)),
    (0, 1, 0, 1),
    (1, 1, 1, 1),
    (0, 0, 0, 0, 0),
    (1, 0, 1, 0, 1),
)


def positive_fixture(spectator: str) -> tuple[ProgramRecord, Any]:
    record, source = fixture_program(1, spectator)
    return control_component_hierarchy(record, {"h": add_h, "p": add_phase}), source


def reference_gates(
    source: Any, ports: SourcePorts, work: tuple[int, ...], measurements: dict[int, int]
) -> tuple[Instruction, ...]:
    from qualtran.bloqs.mcmt.and_bloq import And, MultiAnd

    def leaf(
        bloq: Any, inputs: SourcePorts, desired: SourcePorts
    ) -> SourceResult | None:
        if bloq == source.ctrl_bloq:
            return original_gates(bloq, inputs, work[-1:], measurements), inputs
        base = getattr(bloq, "subbloq", bloq)
        if not isinstance(base, (And, MultiAnd)):
            return None
        erase = (isinstance(base, And) and base.uncompute) or bloq is not base
        outputs = {"ctrl": inputs["ctrl"]}
        if not erase:
            outputs["target"] = work[-2:-1]
            if len(work) > 2:
                outputs["junk"] = work[:-2]
        gates = reference_arithmetic_gates(
            bloq,
            {**inputs, **outputs} if isinstance(base, And) else inputs,
            () if erase else work[:-1],
            measurements,
            expected_outputs=outputs,
            cirq_style=isinstance(base, And),
        )
        return gates, outputs

    gates, returned = walk_source_graph(source, ports, ports, leaf)
    assert returned == ports
    return gates


@pytest.mark.parametrize("values", PATTERNS)
@pytest.mark.parametrize("spectator", ("spectator", "ctrl"))
@pytest.mark.parametrize("expanded", (False, True))
def test_helper_graph_gate_parameters_all_histories_and_resources(
    values: tuple[int, ...], spectator: str, expanded: bool
) -> None:
    from qualtran.bloqs.mcmt.specialized_ctrl import (
        _MultiControlledFromSinglyControlled,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    record, positive = positive_fixture(spectator)
    control_name = positive.ctrl_reg_names[0]
    source = _MultiControlledFromSinglyControlled(values, positive, control_name)
    module = multi_controlled_component_program(record, control_name, values)
    assert verify_program_record(module) == ()
    observer = _source(Signature([]))
    assert tuple(
        observer.signature(Signature(module.functions[0].registers or ()))
    ) == tuple(source.signature)
    assert normalize_trace(_native_trace(module)) == normalize_trace(
        _source_trace(observer, source.decompose_bloq())
    )
    terminals = {
        op.callee: instance.bloq
        for op, (instance, _, _) in zip(
            module.functions[0].blocks[0].operations,
            source.decompose_bloq().iter_bloqsoqs(),
            strict=True,
        )
        if op.callee is not None
    }
    functions = {f.name: f for f in module.functions}
    for name, bloq in terminals.items():
        assert tuple(
            observer.signature(Signature(functions[name].registers or ()))
        ) == tuple(bloq.signature)
    counts = program_component_call_graph(
        module, keep=terminals.__contains__
    ).leaf_counts
    assert (
        Counter({terminals[name]: count for name, count in counts.items()})
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
            for name, count in counts.items()
        )
    if expanded:
        module, _ = expand_component_calls(module)
    n = len(values)
    inputs = tuple((i,) for i in range(n + 2))
    work = tuple(range(n + 2, 2 * n + 2))
    bound = bind_register_program(module, inputs, clean=work)
    assert bound.outputs == inputs
    gates = bound.circuit.instructions
    slots = {
        g.wires[0]: g.metadata["classical_bit"] for g in gates if g.name == "measure"
    }
    ports = {control_name: tuple(range(n)), "q": (n,), spectator: (n + 1,)}
    reference = reference_gates(source, ports, work, slots)
    assert_source(gates, reference)
    assert [(g.name, g.wires, dict(g.params)) for g in gates if g.params] == [
        (g.name, g.wires, dict(g.params)) for g in reference if g.params
    ]
    emitted = Counter(g.name for g in gates)
    assert emitted["measure"] == emitted["reset"] == model.measurement == n
    assert emitted["t"] + emitted["tdg"] == 4 * n
    assert get_cost_value(source, QubitCount()) == bound.circuit.n_wires == 2 * n + 2
    activation = int("".join(map(str, values)), 2)
    initial = torch.zeros(1 << (2 * n + 2), dtype=torch.complex128)
    expected = torch.zeros_like(initial)
    for control, q, spare in product(range(1 << n), (0, 1), (0, 1)):
        index = ((control << 2) | (q << 1) | spare) << n
        amplitude = torch.tensor(
            complex(control + q + 1, spare - control - 1), dtype=torch.complex128
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


@pytest.mark.parametrize("old_value,new_value", tuple(product((0, 1), repeat=2)))
@pytest.mark.parametrize("spectator", ("spectator", "ctrl"))
def test_two_control_factory_callback_matches_original(
    old_value: int, new_value: int, spectator: str
) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.mcmt.specialized_ctrl import _get_ctrl_system_1bit_cv

    record, positive = positive_fixture(spectator)
    control_name = positive.ctrl_reg_names[0]
    source, original_adder = _get_ctrl_system_1bit_cv(
        positive,
        CtrlSpec(cvs=new_value),
        current_ctrl_bit=old_value,
        get_ctrl_bloq_and_ctrl_reg_name=lambda value: (positive, control_name),
    )
    module, adder = specialized_component_control_system(
        record, control_name, old_value, new_value
    )
    signature = Signature(
        (Register("added", quantum_bit_type()), *(record.functions[0].registers or ()))
    )
    native = ComponentProgramBuilder(signature, module.functions, name="caller")
    observer = _source(signature)
    values = {
        r.name: native.inputs[r.name] for r in (record.functions[0].registers or ())
    }
    source_values = {
        r.name: observer.inputs[r.name] for r in (record.functions[0].registers or ())
    }
    added, old_data = adder(native, native.inputs["added"], values)
    added_values, source_data = original_adder(
        observer.builder, (observer.inputs["added"],), source_values
    )
    assert (
        np.asarray(values[control_name]).shape
        == np.asarray(source_values[control_name]).shape
        == (2,)
    )
    caller = native.finalize(
        dict(zip((r.name for r in signature), (added, *old_data), strict=True))
    )
    composite = observer.builder.finalize(
        **dict(
            zip((r.name for r in signature), (*added_values, *source_data), strict=True)
        )
    )
    assert normalize_trace(_native_trace(caller, "caller")) == normalize_trace(
        _source_trace(observer, composite)
    )
    bound = bind_register_program(
        caller, tuple((i,) for i in range(4)), clean=(4, 5), entry="caller"
    )
    direct = bind_register_program(module, tuple((i,) for i in range(4)), clean=(4, 5))
    assert bound.circuit == direct.circuit
    assert tuple(
        observer.signature(Signature(module.functions[0].registers or ()))
    ) == tuple(source.signature)


@pytest.mark.parametrize("value", (0, 1))
def test_original_further_control_callback_defect_is_retained(value: int) -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.mcmt.specialized_ctrl import (
        _MultiControlledFromSinglyControlled,
    )

    _, positive = positive_fixture("spectator")
    source = _MultiControlledFromSinglyControlled((0, 1), positive, "ctrl")
    extended, callback = source.get_ctrl_system(CtrlSpec(cvs=value))
    assert extended.cvs == (value, 0, 1)
    observer = _source(
        Signature(
            (
                Register("added", quantum_bit_type()),
                Register("ctrl", quantum_bit_type(), (2,)),
                Register("q", quantum_bit_type()),
                Register("spectator", quantum_bit_type()),
            )
        )
    )
    values = {name: observer.inputs[name] for name in ("ctrl", "q", "spectator")}
    with pytest.raises(TypeError):
        callback(observer.builder, (observer.inputs["added"],), values)


@pytest.mark.parametrize(
    "failure",
    ("name", "shape", "side", "few", "bad", "boolean", "unknown", "collision", "width"),
)
def test_invalid_helper_rejects_before_modifying_original(failure: str) -> None:
    record, _ = positive_fixture("spectator")
    name = "absent" if failure == "name" else "ctrl"
    values: tuple[int, ...] = (0, 1)
    entry = "absent" if failure == "unknown" else "main"
    if failure in ("shape", "side", "width"):
        root = record.functions[0]
        registers = list(root.registers or ())
        registers[0] = (
            Register("ctrl", quantum_bit_type(), (1,))
            if failure == "shape"
            else (
                Register("ctrl", RegisterType("opaque", 2))
                if failure == "width"
                else Register("ctrl", quantum_bit_type(), side=RegisterSide.LEFT)
            )
        )
        record = replace(
            record,
            functions=(
                replace(root, registers=tuple(registers)),
                *record.functions[1:],
            ),
        )
    elif failure == "few":
        values = (1,)
    elif failure == "bad":
        values = (0, 2)
    elif failure == "boolean":
        values = (True, 1)
    elif failure == "collision":
        record = replace(
            record,
            functions=(
                *record.functions,
                replace(record.functions[-1], name="__fq_specialized_0_reserved"),
            ),
        )
        module = multi_controlled_component_program(record, name, values)
        assert any(f.name == "__fq_specialized_1_positive" for f in module.functions)
        return
    before = repr(record)
    with pytest.raises(ValueError):
        multi_controlled_component_program(record, name, values, entry)
    assert repr(record) == before


@pytest.mark.parametrize("values", ((0, 1), (1, 0), (0, 1, 0), (1, 0, 1)))
@pytest.mark.parametrize("expanded", (False, True))
def test_exact_adjoint_restores_all_measurement_history_pairs(
    values: tuple[int, ...], expanded: bool
) -> None:
    from qualtran.bloqs.mcmt.specialized_ctrl import (
        _MultiControlledFromSinglyControlled,
    )

    record, positive = positive_fixture("spectator")
    record = replace(
        record,
        functions=(
            *record.functions,
            *body(
                "cp_inverse",
                (Register("q", quantum_bit_type(), (2,)),),
                CircuitIR(3, controlled_z_power_ir(0, 1, -0.25, 2, 0)),
            ),
        ),
    )
    inverse = adjoint_component_hierarchy(record, {"ch": "ch", "cp": "cp_inverse"})
    n = len(values)
    inputs = tuple((i,) for i in range(n + 2))
    work = tuple(range(n + 2, 2 * n + 2))
    circuits = []
    for module, child in (
        (record, positive),
        (inverse, source_adjoint(positive.decompose_bloq())),
    ):
        controlled = multi_controlled_component_program(module, "ctrl", values)
        source = _MultiControlledFromSinglyControlled(values, child, "ctrl")
        if expanded:
            controlled, _ = expand_component_calls(controlled)
        gates = bind_register_program(
            controlled, inputs, clean=work
        ).circuit.instructions
        bits = {
            g.wires[0]: g.metadata["classical_bit"]
            for g in gates
            if g.name == "measure"
        }
        reference = reference_gates(
            source,
            {"ctrl": tuple(range(n)), "q": (n,), "spectator": (n + 1,)},
            work,
            bits,
        )
        assert_source(gates, reference)
        assert [(g.name, g.wires, dict(g.params)) for g in gates if g.params] == [
            (g.name, g.wires, dict(g.params)) for g in reference if g.params
        ]
        circuits.append(gates)
    initial = torch.zeros(1 << (2 * n + 2), dtype=torch.complex128)
    initial[:: 1 << n] = torch.tensor(
        [complex(i + 1, 3 - 2 * i) for i in range(1 << (n + 2))], dtype=torch.complex128
    )
    initial /= initial.norm()
    for first, second in product(product((0, 1), repeat=n), repeat=2):
        state = forced_branch(initial, circuits[0], first)
        restored = forced_branch(state, circuits[1], second)
        torch.testing.assert_close(restored, initial / 2**n, atol=3e-12, rtol=0)
        assert torch.count_nonzero(restored.reshape(-1, 1 << n)[:, 1:]) == 0


def test_specialized_helper_executes_through_existing_append() -> None:
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    record, _ = positive_fixture("spectator")
    module = multi_controlled_component_program(record, "ctrl", (1, 0, 1))
    bound = bind_register_program(
        module, tuple((i,) for i in range(5)), clean=(5, 6, 7)
    )
    initial = torch.zeros(256, dtype=torch.complex128)
    initial[::8] = torch.tensor(
        [complex(i + 1, 5 - i) for i in range(32)], dtype=torch.complex128
    )
    initial /= initial.norm()
    expected = forced_branch(initial, bound.circuit.instructions, (0, 0, 0)) * 2**1.5
    circuit = DynamicCircuit(8, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(8, bound.circuit.instructions))
    result = run_dynamic(circuit, shots=8, seed=29, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(8, -1), expected.expand(8, -1), atol=3e-12, rtol=0
    )

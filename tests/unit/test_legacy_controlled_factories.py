"""Structured control constructors reuse specialized child owners and exact erasure."""

from collections import Counter
from importlib import import_module
from itertools import product
from typing import Any

import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.compiler.formats.legacy_component_arguments import ControlSpecification
from flagquantum.compiler.formats.legacy_component_resolution import (
    resolve_component_program,
)
from flagquantum.compiler.formats.legacy_controlled_factories import (
    controlled_via_and_factories,
)
from flagquantum.compiler.formats.legacy_word_factories import _rename_program
from flagquantum.compiler.registers.component_control import (
    ControlAdder,
    _stored_control_adder,
    control_component_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import (
    FunctionRecord,
    ProgramRecord,
    verify_program_record,
)
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import SerializationError
from flagquantum.runtime.dynamic.execution import run_dynamic
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_control import add_h, add_phase, fixture_program
from tests.unit.test_component_structured_control import (
    MULTI,
    SIGNED,
    SINGLE,
    native_controls,
    original_gates,
)
from tests.unit.test_legacy_arithmetic_factories import _factories, _library
from tests.unit.test_multi_control_component import MeasurementSlots
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_control_predicate import ControlCase, control_fixture
from tests.unit.test_register_subtraction import assert_source

np = pytest.importorskip("numpy")

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
KEY = "qualtran.bloqs.mcmt.controlled_via_and.ControlledViaAnd"


def _check_state(
    source: Any, gates: tuple[Instruction, ...], logical: int, work: int
) -> None:
    coefficients = torch.arange(1, (1 << logical) + 1, dtype=torch.float64)
    amplitudes = torch.complex(coefficients, 3 - 7 * coefficients)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (logical + work), dtype=torch.complex128)
    initial[:: 1 << work] = amplitudes
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        1 << logical, 1 << logical
    )
    expected = torch.zeros_like(initial)
    expected[:: 1 << work] = matrix @ amplitudes
    measured = sum(g.name == "measure" for g in gates)
    for history in product((0, 1), repeat=measured):
        torch.testing.assert_close(
            forced_branch(initial, gates, history),
            expected / 2 ** (measured / 2),
            rtol=0,
            atol=4e-12,
        )
    for precision in (torch.complex64, torch.complex128):
        circuit = DynamicCircuit(
            logical + work, inputs=initial.to(precision), dtype=precision
        )
        circuit.append(FieldOperation(circuit.n_qubits, gates))
        result = run_dynamic(circuit, shots=2, seed=53, strategy="trajectory")
        torch.testing.assert_close(
            result.final_states.reshape(2, -1),
            expected.to(precision).expand(2, -1),
            rtol=0,
            atol=2e-6 if precision == torch.complex64 else 4e-12,
        )


@pytest.mark.parametrize("axis", ("XGate", "YGate", "ZGate", "Hadamard"))
@pytest.mark.parametrize("case", (*SINGLE, *SIGNED, MULTI[0], MULTI[1]))
def test_fixed_children_grouped_controls_source_gates_complex_branches_and_resources(
    axis: str, case: ControlCase
) -> None:
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    spec, patterns, groups = control_fixture(case)
    source = ControlledViaAnd(
        getattr(import_module("qualtran.bloqs.basic_gates"), axis)(), spec
    )
    program = resolve_component_program(
        _library(source, 0), 0, {**_factories(), **controlled_via_and_factories()}
    )
    n = int(spec.num_qubits)
    inputs = tuple(
        group[i : i + pattern.width]
        for group, pattern in zip(groups, patterns, strict=True)
        for i in range(0, len(group), pattern.width)
    ) + ((n,),)
    work = tuple(range(n + 1, 2 * n))
    bound = bind_register_program(program, inputs, clean=work)
    assert bound.outputs == inputs
    ports = dict(zip(source.ctrl_reg_names, groups, strict=True)) | {"q": (n,)}
    assert_source(
        bound.circuit.instructions,
        reference_arithmetic_gates(
            source,
            ports,
            work,
            MeasurementSlots(bound.circuit.instructions),
            keep_pauli_leaves=True,
        ),
    )
    actual = Counter(g.name for g in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert actual["t"] + actual["tdg"] == 4 * (n - 1)
    assert model.and_bloq == n - 1
    assert model.rotation == (2 if axis == "Hadamard" else 0)
    assert actual["ch"] == (1 if axis == "Hadamard" else 0)
    assert actual["measure"] == actual["reset"] == model.measurement == n - 1
    _check_state(source, bound.circuit.instructions, n + 1, len(work))


@pytest.mark.parametrize("spectator", ("spectator", "ctrl", "ctrl2"))
@pytest.mark.parametrize("case", (SINGLE[0], MULTI[0]))
def test_injected_multitarget_child_preserves_names_phases_and_private_work(
    spectator: str, case: ControlCase
) -> None:
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd

    record, positive_source = fixture_program(1, spectator)
    child = record.functions[0]
    calls: list[FunctionRecord] = []

    def provider(function: FunctionRecord) -> tuple[ProgramRecord, ControlAdder]:
        calls.append(function)
        assert function == child
        module = _rename_program(
            control_component_program(record, {"h": add_h, "p": add_phase}),
            "positive_body",
        )
        return module, _stored_control_adder(module.functions[0])

    spec, _, groups = control_fixture(case)
    native_spec = ControlSpecification(
        tuple(
            zip(
                (
                    r.dtype
                    for r in native_controls(
                        case, tuple(f"c{i}" for i in range(len(case)))
                    )
                ),
                spec.cvs,
                strict=True,
            )
        )
    )
    program = controlled_via_and_factories(control_system=provider)[KEY](
        "controlled", None, {"subbloq": child, "ctrl_spec": native_spec}
    )
    assert isinstance(program, ProgramRecord)
    assert verify_program_record(program) == ()
    assert calls == [child]
    source = ControlledViaAnd(positive_source.subbloq, spec)
    n = int(spec.num_qubits)
    work = tuple(range(n + 2, 2 * n + 2))
    bound = bind_register_program(
        program,
        (*((q,) for q in range(n)), (n,), (n + 1,)),
        clean=work,
        entry="controlled",
    )
    ports = dict(zip(source.ctrl_reg_names, groups, strict=True)) | {
        "q": (n,),
        spectator: (n + 1,),
    }
    slots = {
        g.wires[0]: g.metadata["classical_bit"]
        for g in bound.circuit.instructions
        if g.name == "measure"
    }
    assert_source(
        bound.circuit.instructions, original_gates(source, ports, work, slots)
    )
    _check_state(source, bound.circuit.instructions, n + 2, len(work))


def test_nested_recipe_combines_controls_before_old_clauses() -> None:
    from qualtran import CtrlSpec
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.mcmt.controlled_via_and import ControlledViaAnd

    from flagquantum.core.ir.program.register_types import quantum_bit_type

    child = _factories()["qualtran.bloqs.basic_gates.x_basis.XGate"]("leaf", None, {})
    assert isinstance(child, FunctionRecord)
    factory = controlled_via_and_factories()[KEY]
    bit = quantum_bit_type()
    inner = factory(
        "inner",
        None,
        {
            "subbloq": child,
            "ctrl_spec": ControlSpecification(((bit, np.array([0, 1])),)),
        },
    )
    assert isinstance(inner, ProgramRecord)
    outer = factory(
        "outer",
        None,
        {
            "subbloq": inner.functions[0],
            "ctrl_spec": ControlSpecification(((bit, np.array(0)),)),
        },
    )
    assert isinstance(outer, ProgramRecord)
    source = ControlledViaAnd(
        ControlledViaAnd(XGate(), CtrlSpec(cvs=[0, 1])), CtrlSpec(cvs=0)
    )
    work = (4, 5)
    bound = bind_register_program(
        outer, ((0,), (1,), (2,), (3,)), clean=work, entry="outer"
    )
    assert_source(
        bound.circuit.instructions,
        reference_arithmetic_gates(
            source,
            {source.ctrl_reg_names[0]: (0,), "ctrl": (1, 2), "q": (3,)},
            work,
            MeasurementSlots(bound.circuit.instructions),
            keep_pauli_leaves=True,
        ),
    )
    _check_state(source, bound.circuit.instructions, 4, len(work))


def test_unregistered_child_fails_without_synthesizing_a_substitute() -> None:
    child, _ = fixture_program(1)
    with pytest.raises(SerializationError, match="positive-control provider"):
        controlled_via_and_factories()[KEY](
            "bad",
            None,
            {
                "subbloq": child.functions[0],
                "ctrl_spec": ControlSpecification(
                    (((child.functions[0].registers or ())[0].dtype, np.array(1)),),
                ),
            },
        )


def test_symbolic_child_retains_declaration_without_calling_a_provider() -> None:
    import sympy

    from flagquantum.core.ir.program.register_types import (
        opaque_register_type,
        quantum_bit_type,
    )
    from flagquantum.core.ir.program.registers import Register

    child = FunctionRecord(
        "child",
        (),
        registers=(
            Register(
                "q",
                opaque_register_type(sympy.Symbol("n", integer=True, positive=True)),
            ),
        ),
    )
    spec = ControlSpecification(((quantum_bit_type(), np.array(1)),))
    result = controlled_via_and_factories()[KEY](
        "symbolic", None, {"subbloq": child, "ctrl_spec": spec}
    )
    assert isinstance(result, FunctionRecord)
    assert result.blocks == ()
    assert (
        result.registers is not None
        and result.registers[-1] == (child.registers or ())[0]
    )
    assert verify_program_record(ProgramRecord((result,))) == ()
    with pytest.raises(SerializationError, match="signature"):
        controlled_via_and_factories()[KEY](
            "wrong", (), {"subbloq": child, "ctrl_spec": spec}
        )


def test_provider_collision_is_reported_without_retaining_partial_state() -> None:
    from flagquantum.core.ir.program.register_types import quantum_bit_type

    record, _ = fixture_program(1)
    child = record.functions[0]
    spec = ControlSpecification(((quantum_bit_type(), np.array(1)),))

    def collision(function: FunctionRecord) -> tuple[ProgramRecord, ControlAdder]:
        module = control_component_program(record, {"h": add_h, "p": add_phase})
        return module, _stored_control_adder(module.functions[0])

    with pytest.raises(SerializationError, match="conflict"):
        controlled_via_and_factories(control_system=collision)[KEY](
            "bad", None, {"subbloq": child, "ctrl_spec": spec}
        )


@pytest.mark.parametrize(
    "example", ("_controlled_via_and_qbits", "_controlled_via_and_ints")
)
def test_complete_source_examples_preserve_gate_operands_and_cleanup(
    example: str,
) -> None:
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source = getattr(import_module("qualtran.bloqs.mcmt.controlled_via_and"), example)()
    program = resolve_component_program(
        _library(source, 0), 0, {**_factories(), **controlled_via_and_factories()}
    )
    cursor = 0
    inputs: list[tuple[int, ...]] = []
    ports: dict[str, tuple[int, ...]] = {}
    for register in source.signature:
        width = int(register.dtype.num_qubits)
        group = tuple(range(cursor, cursor + int(register.total_bits())))
        inputs.extend(group[i : i + width] for i in range(0, len(group), width))
        ports[register.name] = group
        cursor += len(group)
    n = int(source.ctrl_spec.num_qubits)
    work = tuple(range(cursor, cursor + n - 1))
    bound = bind_register_program(program, tuple(inputs), clean=work)
    assert bound.outputs == tuple(inputs)
    assert_source(
        bound.circuit.instructions,
        reference_arithmetic_gates(
            source,
            ports,
            work,
            MeasurementSlots(bound.circuit.instructions),
            keep_pauli_leaves=True,
        ),
    )
    actual = Counter(g.name for g in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert actual["t"] + actual["tdg"] == 4 * (n - 1)
    assert (
        actual["measure"]
        == actual["reset"]
        == model.measurement
        == model.and_bloq
        == n - 1
    )

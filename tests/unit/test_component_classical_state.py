"""Selected SSA classical stepping matches the pinned numerical call chain."""

from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np
import pytest

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.core.ir.program import (
    FunctionRecord,
    Operation,
    ProgramRecord,
    ValueRef,
)
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.classical_addition import register_add_classical_values
from flagquantum.simulation.classical_distributions import ClassicalDistribution
from flagquantum.simulation.classical_measurement import (
    measure_x_basis_phase,
    measure_x_classical_values,
)
from flagquantum.simulation.component_classical_state import ComponentClassicalState
from tests.unit.test_component_hierarchy_adjoint import wrapped_add
from tests.unit.test_program_owned_measurement import measured_function

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def order(record: ProgramRecord, entry: str = "main") -> tuple[int, ...]:
    return tuple(position for position, _, _ in iter_component_calls(record, entry))


def add_values(function: FunctionRecord, values: dict[str, object]) -> object:
    if function.name != "add":
        return NotImplemented
    a, b = values["a"], values["b"]
    assert isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer))
    left, right = register_add_classical_values(a, b, 3)
    return {"a": left, "b": right}


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_complete_add_graph_all_inputs_and_explicit_recursive_boundary(
    depth: int,
) -> None:
    from qualtran.simulation.classical_sim import ClassicalSimState, _get_in_vals

    record, source = wrapped_add(depth)
    original = record

    def recurse(
        function: FunctionRecord, values: dict[str, object]
    ) -> tuple[dict[str, object], None]:
        state = ComponentClassicalState(
            record,
            order(record, function.name),
            values,
            add_values,
            entry=function.name,
            recurse=recurse,
        )
        return state.simulate(), None

    for a, b, spectator in product(range(4), range(8), range(2)):
        values = {"a": a, "b": b, "spectator": spectator}
        state = ComponentClassicalState(
            record, order(record), values, add_values, recurse=recurse
        )
        with pytest.raises(KeyError):
            state.finalize()
        reference = ClassicalSimState.from_cbloq(source, values)
        instances = [instance for instance, _, _ in source.iter_bloqsoqs()]
        functions = {function.name: function for function in record.functions}
        for position, instance in zip(order(record), instances, strict=True):
            assert state.step() is state
            while reference.last_binst != instance:
                reference.step()
            operation = record.functions[0].blocks[0].operations[position]
            child = Signature(functions[operation.callee or ""].registers or ())
            observed = {
                register.name: state.assignments[value]
                for register, value in zip(
                    child.rights(), operation.results, strict=True
                )
            }
            expected_step = {
                register.name: _get_in_vals(instance, register, reference.soq_assign)
                for register in instance.bloq.signature.rights()
            }
            assert observed == expected_step
        actual = state.simulate()
        expected = reference.simulate()
        assert (
            actual == expected == {"a": a, "b": (b + 2 * a) % 8, "spectator": spectator}
        )
        assert state.finalize() == actual
        assert state.last_position == order(record)[-1]
        with pytest.raises(StopIteration):
            state.step()
        assert values == {"a": a, "b": b, "spectator": spectator}
    assert record is original


@pytest.mark.parametrize("shape", [(), (2,), (1, 2), (0,)])
def test_shaped_passthrough_and_array_storage_match_real_source(
    shape: tuple[int, ...],
) -> None:
    from qualtran import Bloq, QUInt, Register
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.classical_sim import ClassicalSimState

    from tests.unit.test_classical_action_tensor import signature_from_source

    source_signature = SourceSignature([Register("x", QUInt(9), shape=shape)])
    source_type: Any = type(
        "Identity",
        (Bloq,),
        {
            "signature": property(lambda self: source_signature),
            "on_classical_vals": lambda self, **values: values,
        },
    )
    signature = signature_from_source(source_signature)
    leaf_builder = ComponentProgramBuilder(signature, name="identity")
    leaf = leaf_builder.finalize(leaf_builder.inputs).functions[0]
    builder = ComponentProgramBuilder(signature, (leaf,))
    record = builder.finalize(builder.call_dict("identity", builder.inputs))
    value: object = np.full(shape, 257, dtype=np.int64) if shape else 257
    state = ComponentClassicalState(record, (0,), {"x": value}, lambda f, v: v)
    actual = state.simulate()["x"]
    source = ClassicalSimState.from_cbloq(
        source_type().as_composite_bloq(), {"x": value}
    )
    if shape == (0,):
        # Source lacks RightDangle for an empty external array. Native declared
        # return metadata retains it; preserve the original failure as a gap.
        with pytest.raises(KeyError, match="RightDangle"):
            source.simulate()
        assert (
            isinstance(actual, np.ndarray)
            and actual.shape == shape
            and actual.dtype == np.uint16
        )
    else:
        expected = source.simulate()["x"]
        np.testing.assert_array_equal(actual, expected)
        assert type(actual) is type(expected)
        if isinstance(actual, np.ndarray):
            assert isinstance(expected, np.ndarray)
            assert actual.dtype == expected.dtype == np.uint16


def measurement_record() -> ProgramRecord:
    child = measured_function("measure_x")
    block = child.blocks[0]
    wire, rotated, word = (
        ValueRef("entry", index, kind)
        for index, kind in ((2, "qubit"), (3, "qubit"), (4, "register"))
    )
    child = replace(
        child,
        blocks=(
            replace(
                block,
                operations=(
                    Operation("split_register", block.arguments, (wire,)),
                    Operation("h", (wire,), (rotated,)),
                    Operation(
                        "join_register",
                        (rotated,),
                        (word,),
                        register_type=quantum_bit_type(),
                    ),
                    replace(block.operations[0], operands=(word,)),
                ),
            ),
        ),
    )
    builder = ComponentProgramBuilder(Signature(child.registers or ()), (child,))
    return builder.finalize(builder.call_dict("measure_x", builder.inputs))


def measurement_values(function: FunctionRecord, values: dict[str, object]) -> object:
    return measure_x_classical_values(values["q"])


@pytest.mark.parametrize("input_bit,outcome", list(product((0, 1), repeat=2)))
def test_all_measurement_branches_and_occurrence_ids_match_source(
    input_bit: int, outcome: int
) -> None:
    from qualtran.bloqs.basic_gates import MeasureX
    from qualtran.simulation.classical_sim import PhasedClassicalSimState

    record = measurement_record()
    state = ComponentClassicalState(
        record,
        (0,),
        {"q": input_bit},
        measurement_values,
        lambda f, v: measure_x_basis_phase(v["q"]),
        phased=True,
        instance_ids=(73,),
        fixed_values={73: outcome},
    )
    source = PhasedClassicalSimState.from_cbloq(
        MeasureX().as_composite_bloq(), {"q": input_bit}, fixed_random_vals={0: outcome}
    )
    assert state.simulate() == source.simulate() == {"c": outcome}
    assert state.phase == source.phase == (-1 if input_bit == outcome == 1 else 1)
    assert (
        state.assignments[record.functions[0].blocks[0].operations[0].results[0]]
        == outcome
    )
    import torch

    from flagquantum.compiler.registers.register_binding import bind_register_program
    from tests.unit.measurement_checks import forced_branch

    bound = bind_register_program(record, ((0,),), clean=(1,))
    initial = torch.zeros(4, dtype=torch.complex128)
    initial[2 * input_bit] = 1
    branch = forced_branch(initial, bound.circuit.instructions, (outcome,))
    expected = torch.zeros_like(initial)
    expected[2 * outcome] = torch.tensor(state.phase / 2**0.5, dtype=expected.dtype)
    torch.testing.assert_close(branch, expected, atol=2e-14, rtol=0)


def test_rng_precedence_and_generator_state_match_source() -> None:
    from qualtran.bloqs.basic_gates import MeasureX
    from qualtran.simulation.classical_sim import PhasedClassicalSimState

    for seed in range(8):
        rng, original_rng = np.random.default_rng(seed), np.random.default_rng(seed)
        state = ComponentClassicalState(
            measurement_record(),
            (0,),
            {"q": 1},
            measurement_values,
            lambda f, v: measure_x_basis_phase(v["q"]),
            phased=True,
            rng=rng,
            fixed_values={0: 99},
        )
        source = PhasedClassicalSimState.from_cbloq(
            MeasureX().as_composite_bloq(),
            {"q": 1},
            rng=original_rng,
            fixed_random_vals={0: 99},
        )
        assert state.simulate() == source.simulate()
        assert state.phase == source.phase
        assert rng.bit_generator.state == original_rng.bit_generator.state


@pytest.mark.parametrize("contribution", [None, 1, 1j, 2])
@pytest.mark.parametrize("phased", [False, True])
def test_output_commit_precedes_phase_errors_and_callback_order(
    contribution: complex | None, phased: bool
) -> None:
    from qualtran import Bloq
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.classical_sim import (
        ClassicalSimState,
        PhasedClassicalSimState,
    )

    from tests.unit.test_component_expansion import _hierarchy

    native_trace: list[str] = []
    source_trace: list[str] = []

    class Probe:
        @property
        def signature(self) -> Any:
            return SourceSignature.build(q=1)

        def on_classical_vals(self, **values: Any) -> dict[str, int]:
            source_trace.append("values")
            return {"q": 1}

        def basis_state_phase(self, **values: Any) -> complex | None:
            source_trace.append("phase")
            return contribution

    source_type: Any = type(
        "Probe",
        (Bloq,),
        {
            "signature": Probe.signature,
            "on_classical_vals": Probe.on_classical_vals,
            "basis_state_phase": Probe.basis_state_phase,
        },
    )
    record, _ = _hierarchy(1, 1)

    def values(f: FunctionRecord, v: dict[str, object]) -> object:
        native_trace.append("values")
        return {"q": 1}

    def phase(f: FunctionRecord, v: dict[str, object]) -> complex | None:
        native_trace.append("phase")
        return contribution

    state = ComponentClassicalState(
        record, (0,), {"q": 0}, values, phase, phased=phased
    )
    source = (PhasedClassicalSimState if phased else ClassicalSimState).from_cbloq(
        source_type().as_composite_bloq(), {"q": 0}
    )
    if contribution is not None and (not phased or abs(contribution) != 1):
        for simulation in (source, state):
            with pytest.raises(ValueError):
                simulation.simulate()
        assert (
            state.assignments[record.functions[0].blocks[0].operations[0].results[0]]
            == 1
        )
        assert 1 in source.soq_assign.values()
        assert state.phase == 1
    else:
        assert state.simulate() == source.simulate() == {"q": 1}
        assert state.phase == (source.phase if phased else 1)
    assert native_trace == source_trace == ["values", "phase"]


@pytest.mark.parametrize(
    "kind",
    [
        "no_recurse",
        "value_type",
        "phase_without_values",
        "missing_output",
        "out_of_range",
        "distribution",
    ],
)
def test_callback_failure_boundaries_keep_step_and_assignment_prefix(kind: str) -> None:
    record, _ = wrapped_add(1)
    outputs: dict[str, object] = {"a": 1, "b": 2}
    if kind == "missing_output":
        outputs.pop("b")
    if kind == "out_of_range":
        outputs["b"] = 8
    if kind == "distribution":
        outputs["b"] = ClassicalDistribution(2)
    result = (
        NotImplemented
        if kind in {"no_recurse", "phase_without_values"}
        else (17 if kind == "value_type" else outputs)
    )
    state = ComponentClassicalState(
        record,
        (0, 1),
        {"a": 0, "b": 0, "spectator": 0},
        lambda f, v: result,
        lambda f, v: 1 if kind == "phase_without_values" else None,
    )
    expected = (
        NotImplementedError
        if kind == "no_recurse"
        else (TypeError if kind == "value_type" else ValueError)
    )
    with pytest.raises(expected):
        state.step()
    assert state.last_position == 0
    block = record.functions[0].blocks[0]
    if kind in {"missing_output", "out_of_range", "distribution"}:
        assert state.assignments[block.operations[0].results[0]] == 1
        assert block.operations[0].results[1] not in state.assignments


@pytest.mark.parametrize("order_value", [(0,), (0, 0), (1, 0), (False, 1)])
def test_invalid_orders_fail_before_callbacks(order_value: tuple[int, ...]) -> None:
    record, _ = wrapped_add(1)
    with pytest.raises(ValueError, match="order"):
        ComponentClassicalState(
            record, order_value, {}, lambda f, v: pytest.fail("callback")
        )


@pytest.mark.parametrize("ids", [(0,), (0, 0), (-1, 1), (False, 1)])
def test_invalid_occurrence_ids_fail_before_input_binding(ids: tuple[int, ...]) -> None:
    record, _ = wrapped_add(1)
    with pytest.raises(ValueError, match="instance IDs"):
        ComponentClassicalState(record, (0, 1), {}, add_values, instance_ids=ids)


def test_unknown_invalid_and_primitive_entries_are_refused() -> None:
    from tests.unit.test_component_expansion import _hierarchy

    record, _ = wrapped_add(1)
    primitive, _ = _hierarchy(1, 1)
    for module, entry, match in (
        (record, "missing", "unknown"),
        (primitive, "h", "unconditional typed"),
        (replace(record, functions=(record.functions[0],)), "main", "invalid"),
    ):
        with pytest.raises(ValueError, match=match):
            positions = tuple(
                range(
                    len(
                        next(
                            (f for f in module.functions if f.name == entry),
                            module.functions[0],
                        )
                        .blocks[0]
                        .operations
                    )
                )
            )
            ComponentClassicalState(module, positions, {}, add_values, entry=entry)


def test_phased_graph_reconstructs_full_complex_matrix_and_append() -> None:
    from collections import Counter

    import torch
    from qualtran import BloqBuilder, QBit
    from qualtran.bloqs.basic_gates import SGate, XGate, ZGate
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    import flagquantum as fq
    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.compiler.registers.register_binding import bind_register_program
    from tests.unit.test_component_expansion import _hierarchy

    base, _ = _hierarchy(1, 1)
    template = base.functions[1]
    leaves = tuple(
        replace(
            template,
            name=name,
            blocks=tuple(
                replace(
                    block,
                    operations=tuple(
                        replace(op, opcode=name) if op.opcode == "h" else op
                        for op in block.operations
                    ),
                )
                for block in template.blocks
            ),
        )
        for name in ("x", "s", "z")
    )
    builder = ComponentProgramBuilder(Signature(template.registers or ()), leaves)
    values = builder.inputs
    bb = BloqBuilder()
    q = bb.add_register("q", QBit())
    for name, source_gate in (("x", XGate()), ("s", SGate()), ("z", ZGate())):
        values = builder.call_dict(name, values)
        q = bb.add(source_gate, q=q)
    record, source = builder.finalize(values), bb.finalize(q=q)

    def action(function: FunctionRecord, values: dict[str, object]) -> object:
        value = values["q"]
        assert isinstance(value, (int, np.integer))
        return {"q": value ^ 1 if function.name == "x" else value}

    def phase(function: FunctionRecord, values: dict[str, object]) -> complex | None:
        value = values["q"]
        assert isinstance(value, (int, np.integer))
        if function.name == "x":
            return None
        if function.name == "s":
            return 1j ** int(value)
        return complex((-1) ** int(value))

    matrix = torch.zeros((2, 2), dtype=torch.complex128)
    for bit in (0, 1):
        state = ComponentClassicalState(
            record, order(record), {"q": bit}, action, phase, phased=True
        )
        output = state.simulate()["q"]
        assert isinstance(output, (int, np.integer))
        matrix[int(output), bit] = torch.tensor(state.phase, dtype=matrix.dtype)
    torch.testing.assert_close(
        matrix, torch.from_numpy(source.tensor_contract()), atol=2e-14, rtol=0
    )
    bound = bind_register_program(record, ((0,),), clean=(1,))
    counts = Counter(g.name for g in bound.circuit.instructions)
    assert counts == {"x": 1, "s": 1, "z": 1}
    assert get_cost_value(source, QECGatesCost()).clifford == 3
    initial = torch.zeros(4, dtype=torch.complex128)
    initial[::2] = torch.tensor([1 + 2j, -3 + 4j], dtype=torch.complex128) / 30**0.5
    result = (
        fq.Circuit(2, inputs=initial, dtype=torch.complex128)
        .append(FieldOperation(2, bound.circuit.instructions))
        .state()
        .flatten()
    )
    expected = torch.zeros_like(initial)
    expected[::2] = matrix @ initial[::2]
    torch.testing.assert_close(result, expected, atol=2e-14, rtol=0)
    assert torch.count_nonzero(result[1::2]) == 0


@pytest.mark.parametrize("phased,phase", [(False, 1j), (True, 2), (1, 1)])
def test_invalid_phase_configuration_is_refused(phased: bool, phase: complex) -> None:
    record, _ = wrapped_add(1)
    with pytest.raises(ValueError):
        ComponentClassicalState(
            record,
            (0, 1),
            {"a": 0, "b": 0, "spectator": 0},
            add_values,
            phased=phased,
            phase=phase,
        )


def test_resolved_field_metadata_is_not_treated_as_integer_values() -> None:
    from flagquantum.core.ir.program.registers import Register, RegisterType

    dtype = RegisterType(
        "finite_field",
        1,
        parameters=(("characteristic", 2), ("degree", 1), ("order", 2)),
    )
    signature = Signature([Register("x", dtype)])
    leaf_builder = ComponentProgramBuilder(signature, name="identity")
    leaf = leaf_builder.finalize(leaf_builder.inputs).functions[0]
    builder = ComponentProgramBuilder(signature, (leaf,))
    record = builder.finalize(builder.call_dict("identity", builder.inputs))
    with pytest.raises(ValueError, match="integer or bit"):
        ComponentClassicalState(record, (0,), {}, lambda f, v: v)

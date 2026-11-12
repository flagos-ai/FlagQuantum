# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Fixtures adapted from Qualtran v0.7.0 simulation/classical_sim_test.py.

"""Original five-bit array workflows use real native component bodies."""

from collections import Counter
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    BlockRecord,
    FunctionRecord,
    Operation,
    ProgramRecord,
    ValueRef,
)
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.component_classical_state import ComponentClassicalState

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def array_record(*, twice: bool = False) -> ProgramRecord:
    """Copy five inputs into fresh outputs, then XOR the original constant."""
    bit = quantum_bit_type()
    signature = Signature(
        [
            Register("x", bit, (5,)),
            Register("z", bit, (5,), RegisterSide.RIGHT),
        ]
    )
    arguments = tuple(ValueRef("entry", i, "register") for i in range(5))
    operations: list[Operation] = []
    xs, zs = [], []
    index = 5

    def fresh(kind: str) -> ValueRef:
        nonlocal index
        value = ValueRef("entry", index, kind)
        index += 1
        return value

    for i, argument in enumerate(arguments):
        allocated, xwire, zwire, control, target = (
            fresh(kind) for kind in ("register", "qubit", "qubit", "qubit", "qubit")
        )
        operations.extend(
            (
                Operation(
                    "allocate_register",
                    results=(allocated,),
                    register_type=bit,
                    workspace_dirty=False,
                ),
                Operation("split_register", (argument,), (xwire,)),
                Operation("split_register", (allocated,), (zwire,)),
                Operation("cx", (xwire, zwire), (control, target)),
            )
        )
        if i % 2 == 0:
            flipped = fresh("qubit")
            operations.append(Operation("x", (target,), (flipped,)))
            target = flipped
        xout, zout = fresh("register"), fresh("register")
        operations.extend(
            (
                Operation("join_register", (control,), (xout,), register_type=bit),
                Operation("join_register", (target,), (zout,), register_type=bit),
            )
        )
        xs.append(xout)
        zs.append(zout)
    leaf = FunctionRecord(
        "array_xor",
        (
            BlockRecord(
                "entry",
                arguments,
                operations=tuple(operations),
                return_values=(*xs, *zs),
            ),
        ),
        registers=tuple(signature),
    )
    if twice:
        signature = Signature(
            [
                Register("x", bit, (5,)),
                Register("y", bit, (5,), RegisterSide.RIGHT),
                Register("z", bit, (5,), RegisterSide.RIGHT),
            ]
        )
    builder = ComponentProgramBuilder(signature, (leaf,))
    first = builder.call_dict(leaf.name, builder.inputs)
    if not twice:
        return builder.finalize(first)
    second = builder.call_dict(leaf.name, {"x": first["z"]})
    return builder.finalize({"x": first["x"], "y": second["x"], "z": second["z"]})


def array_values(function: FunctionRecord, values: dict[str, object]) -> object:
    x = np.asarray(values["x"])
    return {"x": x, "z": np.logical_xor(x, [1, 0, 1, 0, 1]).astype(np.uint8)}


def array_phase(function: FunctionRecord, values: dict[str, object]) -> complex:
    return complex(np.prod(np.exp(np.asarray(values["x"]) * 1j / np.pi)))


@pytest.fixture(autouse=True)
def source_marker(pytestconfig: pytest.Config) -> None:
    pytestconfig.addinivalue_line("markers", "notebook: upstream notebook execution")


@pytest.mark.parametrize(
    "name",
    [
        "test_apply_classical",
        "test_apply_phased_classical",
        "test_phased_classical_on_normal_classical",
    ],
)
def test_original_array_assertions_with_native_execution(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import qualtran.simulation.classical_sim_test as source

    visits: list[str] = []

    def execute(
        bloq: Any, values: dict[str, object], *, phased: bool
    ) -> tuple[dict[str, object], complex]:
        record = array_record()
        state = ComponentClassicalState(
            record,
            (0,),
            values,
            array_values,
            array_phase if isinstance(bloq, source.ApplyPhasedClassicalTest) else None,
            phased=phased,
        )
        visits.append("native")
        return state.simulate(), state.phase

    def pure(bloq: Any, **values: object) -> tuple[object, ...]:
        return tuple(execute(bloq, values, phased=False)[0].values())

    monkeypatch.setattr(source.ApplyClassicalTest, "call_classically", pure)
    monkeypatch.setattr(
        source,
        "do_phased_classical_simulation",
        lambda bloq, vals: execute(bloq, vals, phased=True),
    )
    getattr(source, name)()
    assert len(visits) == (2 if name == "test_apply_classical" else 1)


def test_pure_array_phase_rejection_retains_native_diagnostic_and_outputs() -> None:
    from qualtran.simulation.classical_sim_test import ApplyPhasedClassicalTest

    values = {"x": np.zeros(5, dtype=np.uint8)}
    with pytest.raises(ValueError, match="do_phased_classical_simulation"):
        ApplyPhasedClassicalTest().call_classically(**values)
    state = ComponentClassicalState(
        array_record(), (0,), values, array_values, array_phase
    )
    with pytest.raises(ValueError, match="phased numerical evaluation"):
        state.step()
    np.testing.assert_array_equal(state.finalize()["z"], [1, 0, 1, 0, 1])
    assert state.phase == 1


def test_original_two_call_array_graph_and_step_assignments() -> None:
    from qualtran import BloqBuilder, LeftDangle, QBit
    from qualtran import Register as SourceRegister
    from qualtran.simulation.classical_sim import ClassicalSimState, _get_in_vals
    from qualtran.simulation.classical_sim_test import ApplyClassicalTest

    bb = BloqBuilder()
    x = bb.add_register(SourceRegister("x", QBit(), shape=(5,)))
    x, y = bb.add(ApplyClassicalTest(), x=x)
    y, z = bb.add(ApplyClassicalTest(), x=y)
    source = bb.finalize(x=x, y=y, z=z)
    record = array_record(twice=True)
    values = {"x": np.zeros(5, dtype=np.intc)}
    state = ComponentClassicalState(record, (0, 1), values, array_values)
    original = ClassicalSimState.from_cbloq(source, values)
    original.step()
    assert original.last_binst is LeftDangle and state.last_position is None
    for position in range(2):
        original.step()
        state.step()
        assert state.last_position == position
        assert original.last_binst.i == position
        operation = record.functions[0].blocks[0].operations[position]
        for name, offset in (("x", 0), ("z", 5)):
            reg = original.last_binst.bloq.signature.get_right(name)
            expected = _get_in_vals(original.last_binst, reg, original.soq_assign)
            actual = [
                state.assignments[r] for r in operation.results[offset : offset + 5]
            ]
            np.testing.assert_array_equal(actual, expected)
    for name, expected in original.finalize().items():
        actual = state.finalize()[name]
        np.testing.assert_array_equal(actual, expected)
        assert (
            isinstance(actual, np.ndarray)
            and actual.dtype == expected.dtype == np.uint8
        )
    with pytest.raises(StopIteration):
        state.step()


def test_full_array_isometry_complex_append_clean_spectator_and_resources() -> None:
    from qualtran.simulation.classical_sim_test import ApplyClassicalTest

    import flagquantum as fq
    from examples.algorithms.binary_field_arithmetic import FieldOperation

    record = array_record()
    bound = bind_register_program(
        record,
        tuple((i,) for i in range(5)),
        clean=tuple(range(5, 11)),
        allow_workspace_outputs=True,
    )
    counts = Counter(g.name for g in bound.circuit.instructions)
    assert counts == {"cx": 5, "x": 3}
    matrix = torch.zeros((1024, 32), dtype=torch.complex128)
    for column, bits in enumerate(product((0, 1), repeat=5)):
        values = {"x": np.asarray(bits, dtype=np.uint8)}
        state = ComponentClassicalState(record, (0,), values, array_values)
        actual = state.simulate()
        expected = ApplyClassicalTest().call_classically(**values)
        for name, value in zip(("x", "z"), expected, strict=True):
            np.testing.assert_array_equal(actual[name], value)
        row = 0
        for name in ("x", "z"):
            for bit in np.asarray(actual[name]):
                row = (row << 1) | int(bit)
        matrix[row, column] = 1
    torch.testing.assert_close(
        matrix.T.conj() @ matrix, torch.eye(32, dtype=matrix.dtype)
    )
    amplitudes = torch.complex(
        torch.arange(1, 33, dtype=torch.float64),
        torch.arange(32, dtype=torch.float64) - 7,
    )
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(2048, dtype=torch.complex128)
    initial[torch.arange(32) << 6] = amplitudes
    actual = (
        fq.Circuit(11, inputs=initial, dtype=torch.complex128)
        .append(FieldOperation(11, bound.circuit.instructions))
        .state()
        .flatten()
    )
    expected = torch.zeros_like(initial)
    expected[::2] = matrix @ amplitudes
    torch.testing.assert_close(actual, expected, atol=2e-14, rtol=0)
    assert torch.count_nonzero(actual[1::2]) == 0
    # The source fixture specifies only numerical action, with no decomposition
    # or call graph. Its resource model is unavailable, rather than zero gates.
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    with pytest.raises(NotImplementedError):
        get_cost_value(ApplyClassicalTest(), QECGatesCost())


@pytest.mark.parametrize(
    "kind", ["bit", "signed", "uint", "ones", "bounded", "sfxp", "large", "field"]
)
def test_original_multidimensional_assertions_with_native_numerical_codecs(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import qualtran as q
    import qualtran.simulation.classical_sim_test as source

    from flagquantum.core.ir.program import register_types as types
    from flagquantum.simulation.classical_typed_values import (
        bind_typed_register_values,
        collect_typed_register_value,
    )
    from tests.unit.test_finite_field_metadata import field_pair

    field, qfield = field_pair(2, 2, "default", False)
    native_type, source_type = {
        "bit": (types.quantum_bit_type(), q.QBit()),
        "signed": (types.signed_register_type(5), q.QInt(5)),
        "uint": (types.unsigned_register_type(5), q.QUInt(5)),
        "ones": (types.ones_complement_register_type(5), q.QIntOnesComp(5)),
        "bounded": (types.bounded_unsigned_register_type(5, 20), q.BQUInt(5, 20)),
        "sfxp": (types.fixed_point_register_type(5, 3, True), q.QFxp(5, 3, True)),
        "large": (types.signed_register_type(100), q.QInt(100)),
        "field": (field.register_type(), qfield),
    }[kind]
    visits: list[str] = []

    def numerical_call(bloq: Any, **values: object) -> tuple[object, ...]:
        # This is a numerical callback boundary, not wider executable IR
        # admission. Original fixture type assertions still execute unchanged.
        left = Register("x", native_type, (bloq.n,), RegisterSide.LEFT)
        right = Register("y", native_type, (bloq.n,), RegisterSide.RIGHT)
        inputs = collect_typed_register_value(
            left, bind_typed_register_values(left, values)
        )
        outputs = bloq.on_classical_vals(x=inputs)
        result = collect_typed_register_value(
            right, bind_typed_register_values(right, outputs)
        )
        visits.append("native-codecs")
        return (result,)

    monkeypatch.setattr(
        source.TestMultiDimensionalReg, "call_classically", numerical_call
    )
    if kind == "large":
        source.test_multidimensional_classical_sim_for_large_int()
    elif kind == "field":
        source.test_multidimensional_classical_sim_for_gqf()
    else:
        source.test_multidimensional_classical_sim_for_dtypes(source_type)
    assert visits == ["native-codecs"]


@pytest.mark.parametrize(
    "signed,width", [(False, n) for n in range(1, 5)] + [(True, n) for n in range(2, 5)]
)
def test_original_exhaustive_add_assertions_with_native_arithmetic(
    signed: bool, width: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    import qualtran.simulation.classical_sim_test as source

    from flagquantum.simulation.classical_addition import add_integer_values

    visits: list[tuple[int, int]] = []

    def addition(a: int, b: int, *, num_bits: int, is_signed: bool) -> int:
        visits.append((a, b))
        result = add_integer_values(a, b, num_bits=num_bits, signed=is_signed)
        assert isinstance(result, int)
        return result

    monkeypatch.setattr(source, "add_ints", addition)
    (source.test_add_ints_signed if signed else source.test_add_ints_unsigned)(width)
    assert len(visits) == 1 << (2 * width)


@pytest.mark.parametrize(
    "name",
    [
        "test_classical_distribution",
        "test_fixed_distribution",
        "test_phased_classical_distribution",
    ],
)
def test_original_distribution_assertions_with_native_state(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import qualtran.simulation.classical_sim_test as source

    from flagquantum.simulation.classical_measurement import measure_x_basis_phase
    from tests.unit.test_component_classical_state import (
        measurement_record,
        measurement_values,
    )

    record = measurement_record()
    visits: list[str] = []
    rng = np.random.default_rng(197)

    def execute(
        values: dict[str, object],
        *,
        phased: bool = False,
        generator: Any = None,
        fixed: Any = None,
    ) -> tuple[dict[str, object], complex]:
        state = ComponentClassicalState(
            record,
            (0,),
            values,
            measurement_values,
            (lambda f, v: measure_x_basis_phase(v["q"])) if phased else None,
            phased=phased,
            rng=generator,
            fixed_values=fixed,
        )
        result = state.simulate()
        visits.append("native")
        return result, state.phase

    def pure(bloq: Any, **values: object) -> tuple[object, ...]:
        return tuple(execute(values, generator=rng)[0].values())

    def fixed_call(
        signature: Any, values: dict[str, object], graph: Any, *, random_handler: Any
    ) -> tuple[dict[str, object], dict[object, object]]:
        assert isinstance(random_handler, source._FixedClassicalValHandler)
        return execute(values, fixed=random_handler._binst_i_to_val)[0], {}

    def phased_call(
        bloq: Any,
        vals: dict[str, object],
        rng: Any = None,
        fixed_random_vals: Any = None,
    ) -> tuple[dict[str, object], complex]:
        return execute(vals, phased=True, generator=rng, fixed=fixed_random_vals)

    monkeypatch.setattr(source.ClassicalDistributionTest, "call_classically", pure)
    monkeypatch.setattr(source, "call_cbloq_classically", fixed_call)
    monkeypatch.setattr(source, "do_phased_classical_simulation", phased_call)
    getattr(source, name)()
    assert len(visits) == (52 if name == "test_phased_classical_distribution" else 100)


def test_banned_distribution_rejects_before_output_assignment() -> None:
    from qualtran.simulation.classical_sim_test import test_banned_distribution

    from tests.unit.test_component_classical_state import (
        measurement_record,
        measurement_values,
    )

    test_banned_distribution()
    record = measurement_record()
    state = ComponentClassicalState(record, (0,), {"q": 0}, measurement_values)
    with pytest.raises(ValueError, match="non-deterministic"):
        state.step()
    assert len(state.assignments) == 1
    assert state.phase == 1 and state.last_position == 0


@pytest.mark.parametrize("case", ["valid", "bad_bit", "numpy_int", "wrong_shape"])
def test_original_validation_input_cases_through_native_binding(case: str) -> None:
    import networkx as nx
    from qualtran import QAny, QBit
    from qualtran import Register as SourceRegister
    from qualtran import Signature as SourceSignature
    from qualtran.simulation.classical_sim import ClassicalSimState

    from flagquantum.core.ir.program.register_types import opaque_register_type
    from flagquantum.simulation.classical_typed_values import bind_typed_register_values

    pairs = [
        (
            SourceRegister("one_bit_int", QBit()),
            Register("one_bit_int", quantum_bit_type()),
        ),
        (SourceRegister("int", QAny(5)), Register("int", opaque_register_type(5))),
        (
            SourceRegister("bit_arr", QBit(), shape=(5,)),
            Register("bit_arr", quantum_bit_type(), (5,)),
        ),
        (
            SourceRegister("int_arr", QAny(32), shape=(5,)),
            Register("int_arr", opaque_register_type(32), (5,)),
        ),
    ]
    values: dict[str, object] = {
        "one_bit_int": 1,
        "int": 5,
        "bit_arr": np.array([1, 0, 1, 0, 1], dtype=np.uint8),
        "int_arr": np.arange(5),
    }
    if case == "bad_bit":
        values["one_bit_int"] = 2
    elif case == "numpy_int":
        values["int"] = np.arange(5, dtype=np.uint8)[4]
    elif case == "wrong_shape":
        values["int_arr"] = np.arange(6)
    original = ClassicalSimState(SourceSignature([]), nx.DiGraph(), {})
    native_assignments: dict[tuple[str, tuple[int, ...]], object] = {}

    def native() -> None:
        for _, register in pairs:
            for index, value in bind_typed_register_values(register, values).items():
                native_assignments[register.name, index] = value

    if case in {"bad_bit", "wrong_shape"}:
        with pytest.raises(ValueError):
            original._update_assign_from_vals([p[0] for p in pairs], "MyBinst", values)
        with pytest.raises(ValueError):
            native()
    else:
        original._update_assign_from_vals([p[0] for p in pairs], "MyBinst", values)
        native()
    expected = {
        (soquet.reg.name, soquet.idx): value
        for soquet, value in original.soq_assign.items()
    }
    assert native_assignments.keys() == expected.keys()
    for key, value in expected.items():
        assert native_assignments[key] is value or (
            type(native_assignments[key]) is type(value)
            and native_assignments[key] == value
        )


def test_original_composed_phasing_assertions_with_stored_native_bodies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import qualtran.simulation.classical_sim_test as source

    from flagquantum.simulation.component_classical_state import (
        simulate_phased_component_calls,
    )
    from tests.unit.test_component_phased_evaluation import (
        orders,
        phase_action,
        phase_fixture,
        phase_values,
    )

    visits: list[int] = []

    def execute(
        bloq: Any, values: dict[str, object]
    ) -> tuple[dict[str, object], complex]:
        record, _ = phase_fixture(bloq.n, 1)
        visits.append(bloq.n)
        return simulate_phased_component_calls(
            record, orders(record), values, phase_values, phase_action
        )

    monkeypatch.setattr(source, "do_phased_classical_simulation", execute)
    source.test_derive_phasing_from_composed_bloq()
    assert visits == [0, 1, 2, 3]


@pytest.mark.parametrize("control,target", list(product((0, 1), repeat=2)))
def test_cnot_assignment_storage_has_explicit_native_identity_mapping(
    control: int, target: int
) -> None:
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.simulation.classical_sim import _get_in_vals, call_cbloq_classically

    from flagquantum.simulation.classical_controlled_bits import cnot_classical_values

    signature = Signature(
        [Register("ctrl", quantum_bit_type()), Register("target", quantum_bit_type())]
    )
    words = tuple(ValueRef("entry", i, "register") for i in range(4))
    wires = tuple(ValueRef("entry", i, "qubit") for i in range(4, 8))
    operations = (
        Operation("split_register", (words[0],), (wires[0],)),
        Operation("split_register", (words[1],), (wires[1],)),
        Operation("cx", wires[:2], wires[2:]),
        Operation(
            "join_register", (wires[2],), (words[2],), register_type=quantum_bit_type()
        ),
        Operation(
            "join_register", (wires[3],), (words[3],), register_type=quantum_bit_type()
        ),
    )
    leaf = FunctionRecord(
        "cnot",
        (
            BlockRecord(
                "entry", words[:2], operations=operations, return_values=words[2:]
            ),
        ),
        registers=tuple(signature),
    )
    builder = ComponentProgramBuilder(signature, (leaf,))
    record = builder.finalize(builder.call_dict(leaf.name, builder.inputs))

    def values(function: FunctionRecord, inputs: dict[str, object]) -> object:
        control, target = inputs["ctrl"], inputs["target"]
        assert isinstance(control, (int, np.integer))
        assert isinstance(target, (int, np.integer))
        c, t = cnot_classical_values(control, target)
        return {"ctrl": c, "target": t}

    state = ComponentClassicalState(
        record, (0,), {"ctrl": control, "target": target}, values
    )
    source = CNOT().as_composite_bloq()
    outputs, assignments = call_cbloq_classically(
        source.signature, {"ctrl": control, "target": target}, source._binst_graph
    )
    assert state.simulate() == outputs == {"ctrl": control, "target": control ^ target}
    assert len(assignments) == 6 and set(assignments) == set(source.all_soquets)
    assert len(state.assignments) == 4
    operation = record.functions[0].blocks[0].operations[0]
    instance = next(iter(source.bloq_instances))
    for register, result in zip(
        source.signature.rights(), operation.results, strict=True
    ):
        assert state.assignments[result] == _get_in_vals(
            instance, register, assignments
        )
    assert state.last_position == 0

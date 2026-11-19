"""Complete comparison graphs retain source calls, phases, history and work."""

from collections import Counter
from collections.abc import Callable
from itertools import product
from typing import Any, Literal

import pytest
import torch

from flagquantum.compiler.registers.comparison.constant_component import (
    less_than_constant_program,
)
from flagquantum.compiler.registers.comparison.controlled_component import (
    controlled_greater_program,
)
from flagquantum.compiler.registers.comparison.equality_component import (
    equality_program,
)
from flagquantum.compiler.registers.comparison.half_component import (
    half_comparison_program,
    half_greater_program,
)
from flagquantum.compiler.registers.comparison.linear_component import (
    linear_greater_program,
)
from flagquantum.compiler.registers.comparison.ordering_component import (
    less_equal_program,
)
from flagquantum.compiler.registers.register_binding import (
    BoundRegisterProgram,
    bind_register_program,
)
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.register_types import (
    signed_register_type,
    unsigned_register_type,
)
from flagquantum.core.ir.program.registers import RegisterType
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_multi_control_component import MeasurementSlots
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
Relation = Literal["gt", "ge", "lt", "le"]


def integer_types(width: int, encoding: str) -> tuple[RegisterType, Any]:
    from qualtran import QInt, QMontgomeryUInt, QUInt

    if encoding == "signed":
        return signed_register_type(width), QInt(width)
    if encoding == "montgomery":
        return RegisterType("montgomery", width), QMontgomeryUInt(width)
    return unsigned_register_type(width), QUInt(width)


def allocation_order(bound: BoundRegisterProgram) -> tuple[int, ...]:
    return tuple(
        w for event in bound.events if event.kind == "allocate" for w in event.wires
    )


def check_source(
    graph: ProgramRecord,
    source: Any,
    ports: dict[str, tuple[int, ...]],
    *,
    tree: bool = False,
) -> BoundRegisterProgram:
    logical = max(w for wires in ports.values() for w in wires) + 1
    bound = bind_register_program(
        graph, tuple(ports.values()), clean=tuple(range(logical, logical + 50))
    )
    assert verify_program_record(graph) == ()
    plan = allocation_order(bound)
    if tree and "y" in ports:
        difference = abs(len(ports["x"]) - len(ports["y"]))
        if difference > 2:
            plan = (
                plan[difference - 2],
                *plan[: difference - 2],
                *plan[difference - 1 :],
            )
    reference = reference_arithmetic_gates(
        source,
        ports,
        plan,
        MeasurementSlots(bound.circuit.instructions),
        reuse_allocations=False,
        cirq_style=tree,
    )
    assert_source(bound.circuit.instructions, reference)
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    actual = Counter(g.name for g in bound.circuit.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert actual["t"] + actual["tdg"] == model.total_t_count()
    assert actual["measure"] == model.measurement
    assert actual["reset"] == actual["measure"]
    return bound


@pytest.mark.parametrize(
    "left,right",
    (
        (1, 1),
        (1, 2),
        (2, 1),
        (1, 3),
        (3, 1),
        (1, 4),
        (4, 1),
        (2, 5),
        (5, 2),
        (3, 3),
        (5, 8),
        (8, 5),
    ),
)
def test_unequal_tree_source_generator_and_feedback(left: int, right: int) -> None:
    from qualtran.bloqs.arithmetic.comparison import LessThanEqual

    check_source(
        less_equal_program(left, right),
        LessThanEqual(left, right),
        {
            "x": tuple(range(left)),
            "y": tuple(range(left, left + right)),
            "target": (left + right,),
        },
        tree=True,
    )


@pytest.mark.parametrize(
    "width,threshold",
    ((1, 0), (1, 1), (2, 2), (3, 5), (4, 0), (4, 15), (4, 16), (4, 17)),
)
def test_constant_source_seed_scan_and_erasure(width: int, threshold: int) -> None:
    from qualtran.bloqs.arithmetic.comparison import LessThanConstant

    check_source(
        less_than_constant_program(width, threshold),
        LessThanConstant(width, threshold),
        {"x": tuple(range(width)), "target": (width,)},
        tree=True,
    )


@pytest.mark.parametrize("width", (1, 2, 3, 5, 8))
@pytest.mark.parametrize("encoding", ("uint", "signed", "montgomery"))
def test_word_equality_source_calls_and_gate_stream(width: int, encoding: str) -> None:
    from qualtran.bloqs.arithmetic.comparison import Equals

    dtype, upstream = integer_types(width, encoding)
    graph, source = equality_program(dtype), Equals(upstream)
    compare_layers(graph, source, {f.name for f in graph.functions if f.name != "main"})
    check_source(
        graph,
        source,
        {
            "x": tuple(range(width)),
            "y": tuple(range(width, 2 * width)),
            "target": (2 * width,),
        },
    )


@pytest.mark.parametrize("width", (1, 2, 3, 5, 8))
@pytest.mark.parametrize("signed", (False, True))
def test_linear_source_calls_and_measured_carry_chain(width: int, signed: bool) -> None:
    from qualtran.bloqs.arithmetic.comparison import LinearDepthGreaterThan

    graph, source = linear_greater_program(
        width, signed=signed
    ), LinearDepthGreaterThan(width, signed=signed)
    compare_layers(graph, source, {f.name for f in graph.functions if f.name != "main"})
    check_source(
        graph,
        source,
        {
            "a": tuple(range(width)),
            "b": tuple(range(width, 2 * width)),
            "target": (2 * width,),
        },
    )


@pytest.mark.parametrize("width", (1, 2, 3, 5, 8))
@pytest.mark.parametrize("encoding", ("uint", "signed", "montgomery"))
@pytest.mark.parametrize("cv", (0, 1))
def test_controlled_source_calls_both_polarities(
    width: int, encoding: str, cv: int
) -> None:
    from qualtran.bloqs.arithmetic.comparison import CLinearDepthGreaterThan

    dtype, upstream = integer_types(width, encoding)
    graph, source = controlled_greater_program(
        dtype, control_value=cv
    ), CLinearDepthGreaterThan(upstream, cv=cv)
    compare_layers(graph, source, {f.name for f in graph.functions if f.name != "main"})
    check_source(
        graph,
        source,
        {
            "ctrl": (0,),
            "a": tuple(range(1, width + 1)),
            "b": tuple(range(width + 1, 2 * width + 1)),
            "target": (2 * width + 1,),
        },
    )


def half_source(dtype: Any, relation: Relation) -> Any:
    from qualtran.bloqs.arithmetic.comparison import (
        LinearDepthHalfGreaterThan,
        LinearDepthHalfGreaterThanEqual,
        LinearDepthHalfLessThan,
        LinearDepthHalfLessThanEqual,
    )

    return {
        "gt": LinearDepthHalfGreaterThan,
        "ge": LinearDepthHalfGreaterThanEqual,
        "lt": LinearDepthHalfLessThan,
        "le": LinearDepthHalfLessThanEqual,
    }[relation](dtype)


@pytest.mark.parametrize("width", (1, 2, 3, 5, 8))
@pytest.mark.parametrize("encoding", ("uint", "signed", "montgomery"))
@pytest.mark.parametrize("relation", ("gt", "ge", "lt", "le"))
def test_half_source_graphs_output_aliases_and_resources(
    width: int, encoding: str, relation: Relation
) -> None:
    dtype, upstream = integer_types(width, encoding)
    source = half_source(upstream, relation)
    inputs = {"a": tuple(range(width)), "b": tuple(range(width, 2 * width))}
    graph = half_comparison_program(dtype, relation)
    bound = bind_register_program(
        graph,
        tuple(inputs.values()),
        clean=tuple(range(2 * width, 5 * width + 10)),
        allow_workspace_outputs=True,
    )
    outputs = dict(zip(("a", "b", "c", "target"), bound.outputs, strict=True))
    inverse = half_comparison_program(dtype, relation, adjoint=True)
    back = bind_register_program(
        inverse,
        bound.outputs,
        clean=tuple(range(4 * width + 5, 4 * width + 7)),
        input_ownership=tuple((w, False) for word in bound.outputs[2:] for w in word),
    )
    returned = dict(zip(("a", "b"), back.outputs, strict=True))
    assert returned == (
        inputs if relation in ("gt", "le") else {"a": inputs["b"], "b": inputs["a"]}
    )
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    for program, original, actual, before, after in (
        (graph, source, bound, inputs, outputs),
        (inverse, source.adjoint(), back, outputs, returned),
    ):
        assert verify_program_record(program) == ()
        compare_layers(
            program, original, {f.name for f in program.functions if f.name != "main"}
        )
        reference = reference_arithmetic_gates(
            original,
            before,
            allocation_order(actual),
            MeasurementSlots(actual.circuit.instructions),
            expected_outputs=after,
            reuse_allocations=False,
        )
        assert_source(actual.circuit.instructions, reference)
        counts = Counter(g.name for g in actual.circuit.instructions)
        model = get_cost_value(original, QECGatesCost())
        assert counts["t"] + counts["tdg"] == model.total_t_count()
        assert counts["measure"] == model.measurement


def assert_source_amplitudes(
    graph: ProgramRecord, source: Any, ports: tuple[tuple[int, ...], ...]
) -> None:
    logical = sum(map(len, ports))
    # A live entangled spectator precedes clean work in the native state.
    bound = bind_register_program(
        graph, ports, clean=tuple(range(logical + 1, logical + 30))
    )
    width = (
        max((w for g in bound.circuit.instructions for w in g.wires), default=logical)
        + 1
    )
    work = width - logical - 1
    coefficients = torch.arange(1, (1 << (logical + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(coefficients, 7 - 3 * coefficients)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << width, dtype=torch.complex128)
    initial[:: 1 << work] = amplitudes
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        1 << logical, 1 << logical
    )
    expected = torch.zeros_like(initial)
    expected[:: 1 << work] = (matrix @ amplitudes.reshape(-1, 2)).flatten()
    measured = sum(g.name == "measure" for g in bound.circuit.instructions)
    for history in product((0, 1), repeat=measured):
        torch.testing.assert_close(
            forced_branch(initial, bound.circuit.instructions, history),
            expected / 2 ** (measured / 2),
            rtol=0,
            atol=3e-12,
        )


@pytest.mark.parametrize(
    "kind", ("constant", "equality", "tree", "linear", "controlled0", "controlled1")
)
def test_complete_source_tensors_dirty_targets_spectator_and_all_branches(
    kind: str,
) -> None:
    from qualtran import QInt, QUInt
    from qualtran.bloqs.arithmetic.comparison import (
        CLinearDepthGreaterThan,
        Equals,
        LessThanConstant,
        LessThanEqual,
        LinearDepthGreaterThan,
    )

    ports: tuple[tuple[int, ...], ...]
    if kind == "constant":
        graph, source, ports = (
            less_than_constant_program(2, 2),
            LessThanConstant(2, 2),
            ((0, 1), (2,)),
        )
    elif kind == "equality":
        graph, source, ports = (
            equality_program(unsigned_register_type(2)),
            Equals(QUInt(2)),
            ((0, 1), (2, 3), (4,)),
        )
    elif kind == "tree":
        graph, source, ports = (
            less_equal_program(1, 3),
            LessThanEqual(1, 3),
            ((0,), (1, 2, 3), (4,)),
        )
    elif kind == "linear":
        graph, source, ports = (
            linear_greater_program(2),
            LinearDepthGreaterThan(2),
            ((0, 1), (2, 3), (4,)),
        )
    else:
        cv = int(kind[-1])
        graph, source, ports = (
            controlled_greater_program(signed_register_type(2), control_value=cv),
            CLinearDepthGreaterThan(QInt(2), cv=cv),
            ((0,), (1, 2), (3, 4), (5,)),
        )
    assert_source_amplitudes(graph, source, ports)


@pytest.mark.parametrize("signed", (False, True))
@pytest.mark.parametrize("relation", ("gt", "ge", "lt", "le"))
def test_half_full_complex_outputs_and_all_erase_branches(
    signed: bool, relation: Relation
) -> None:
    dtype, upstream = integer_types(2, "signed" if signed else "uint")
    ports = ((0, 1), (2, 3))
    forward = bind_register_program(
        half_comparison_program(dtype, relation),
        ports,
        clean=tuple(range(5, 11)),
        allow_workspace_outputs=True,
    )
    inverse = bind_register_program(
        half_comparison_program(dtype, relation, adjoint=True),
        forward.outputs,
        clean=tuple(
            w for w in range(5, 11) if all(w not in word for word in forward.outputs)
        ),
        input_ownership=tuple((w, False) for word in forward.outputs[2:] for w in word),
    )
    width = 11
    raw = torch.arange(1, 33, dtype=torch.float64)
    amplitudes = torch.complex(raw, 7 - 3 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << width, dtype=torch.complex128)
    initial[::64] = amplitudes
    source = half_source(upstream, relation)
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        256, 16
    )
    output = (matrix @ amplitudes.reshape(16, 2)).flatten()
    expected = torch.zeros_like(initial)
    wires = (
        *forward.outputs[0],
        *forward.outputs[1],
        *forward.outputs[2],
        *forward.outputs[3],
        4,
    )
    indices = torch.zeros(512, dtype=torch.int64)
    for i, wire in enumerate(wires):
        indices |= ((torch.arange(512) >> (len(wires) - 1 - i)) & 1) << (
            width - 1 - wire
        )
    expected[indices] = output
    actual = forced_branch(initial, forward.circuit.instructions, ())
    torch.testing.assert_close(actual, expected, rtol=0, atol=3e-12)
    for history in product((0, 1), repeat=2):
        torch.testing.assert_close(
            forced_branch(actual, inverse.circuit.instructions, history),
            initial / 2,
            rtol=0,
            atol=3e-12,
        )


def test_comparison_admission_failures_and_base_half() -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.comparison import _HalfLinearDepthGreaterThan

    graph = half_greater_program(unsigned_register_type(2))
    compare_layers(
        graph,
        _HalfLinearDepthGreaterThan(QUInt(2)),
        {f.name for f in graph.functions if f.name != "main"},
    )
    invalid_flag: Any = 1
    invalid_relation: Any = "invalid"
    factories: tuple[Callable[[], ProgramRecord], ...] = (
        lambda: linear_greater_program(0),
        lambda: linear_greater_program(2, signed=invalid_flag),
        lambda: controlled_greater_program(
            unsigned_register_type(2), control_value=True
        ),
        lambda: half_greater_program(unsigned_register_type(2), adjoint=invalid_flag),
        lambda: half_comparison_program(unsigned_register_type(2), invalid_relation),
    )
    for factory in factories:
        with pytest.raises(ValueError):
            factory()


@pytest.mark.integration
@pytest.mark.parametrize(
    "module",
    (
        "register_comparison",
        "register_equality",
        "register_less_equal",
        "register_greater",
        "register_controlled_comparison",
        "register_half_comparison",
    ),
)
def test_normal_append_consumers_execute_stored_graphs(
    module: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    def unavailable(*args: object, **kwargs: object) -> None:
        raise AssertionError("The replaced handwritten comparison path was called")

    for owner, name in (
        ("less_than", "less_than_constant_ir"),
        ("equality", "equality_ir"),
        ("equality", "equality_constant_ir"),
        ("less_equal", "less_equal_ir"),
        ("ordering", "greater_ir"),
        ("linear_comparison", "linear_greater_ir"),
        ("controlled_comparison", "controlled_greater_ir"),
        ("half_ordering", "half_compare_ir"),
    ):
        target = importlib.import_module(
            "flagquantum.compiler.registers.comparison." + owner
        )
        monkeypatch.setattr(target, name, unavailable)
    importlib.import_module("examples.algorithms." + module).main()


@pytest.mark.parametrize(
    "factory",
    (
        "_lt_k",
        "_leq",
        "_greater_than",
        "_equals",
        "_eq_k",
        "_clineardepthgreaterthan_example",
        "_lineardepthhalfgreaterthan_small",
        "_lineardepthhalflessthan_small",
        "_lineardepthhalfgreaterthanequal_small",
        "_lineardepthhalflessthanequal_small",
    ),
)
def test_actual_concrete_factories_existing_native_checks(factory: str) -> None:
    from qualtran.bloqs.arithmetic import comparison

    from flagquantum.compiler.registers.comparison.equality_component import (
        equality_constant_program,
    )
    from flagquantum.compiler.registers.comparison.ordering_component import (
        greater_than_program,
    )
    from flagquantum.core.ir import CircuitIR
    from tests.unit.register_example_checks import (
        CHECKS,
        CheckStatus,
        RegisterExample,
        check_example,
    )

    source = getattr(comparison, factory).make()
    tree = False
    if factory == "_lt_k":
        graph = less_than_constant_program(
            int(source.bitsize), int(source.less_than_val)
        )
        tree = True
    elif factory == "_leq":
        graph = less_equal_program(int(source.x_bitsize), int(source.y_bitsize))
        tree = True
    elif factory == "_greater_than":
        graph = greater_than_program(int(source.a_bitsize), int(source.b_bitsize))
    elif factory == "_eq_k":
        graph = equality_constant_program(int(source.bitsize), int(source.val))
    else:
        encoding = "signed" if type(source.dtype).__name__ == "QInt" else "uint"
        dtype, _ = integer_types(int(source.dtype.bitsize), encoding)
        if factory == "_equals":
            graph = equality_program(dtype)
        elif factory == "_clineardepthgreaterthan_example":
            graph = controlled_greater_program(dtype, control_value=int(source.cv))
        else:
            relations: dict[str, Relation] = {
                "_lineardepthhalfgreaterthan_small": "gt",
                "_lineardepthhalflessthan_small": "lt",
                "_lineardepthhalfgreaterthanequal_small": "ge",
                "_lineardepthhalflessthanequal_small": "le",
            }
            graph = half_comparison_program(dtype, relations[factory])
    cursor = 0
    ports: dict[str, tuple[int, ...]] = {}
    for register in source.signature.lefts():
        count = int(register.total_bits())
        ports[register.name] = tuple(range(cursor, cursor + count))
        cursor += count
    pool = tuple(range(cursor, cursor + 100))

    def lower(record: ProgramRecord) -> CircuitIR:
        return bind_register_program(
            record, tuple(ports.values()), clean=pool, allow_workspace_outputs=True
        ).circuit

    bound = bind_register_program(
        graph, tuple(ports.values()), clean=pool, allow_workspace_outputs=True
    )
    outputs = dict(
        zip((r.name for r in source.signature.rights()), bound.outputs, strict=True)
    )
    plan = allocation_order(bound)
    if factory == "_leq":
        difference = abs(int(source.x_bitsize) - int(source.y_bitsize))
        if difference > 2:
            plan = (
                plan[difference - 2],
                *plan[: difference - 2],
                *plan[difference - 1 :],
            )
    reference = reference_arithmetic_gates(
        source,
        ports,
        plan,
        MeasurementSlots(bound.circuit.instructions),
        expected_outputs=outputs,
        reuse_allocations=False,
        cirq_style=tree,
        preserve_components=factory == "_greater_than",
    )
    assert_source(bound.circuit.instructions, reference)
    for kind in CHECKS:
        result = check_example(
            RegisterExample(
                lambda: graph, lower, lambda: Counter(g.name for g in reference)
            ),
            kind,
        )
        assert result.status is CheckStatus.PASS, result.message

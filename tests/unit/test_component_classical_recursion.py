"""Recursive classical failures retain source context and assignment boundaries."""

from collections.abc import Callable
from typing import Any

import pytest

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.core.ir.program import FunctionRecord
from flagquantum.simulation.classical_measurement import MeasurementOutcomePhase
from flagquantum.simulation.component_classical_state import ComponentClassicalState
from tests.unit.test_component_bit_values import identity_record
from tests.unit.test_component_classical_state import order

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def source_leaf(
    values: Callable[..., object],
    phase: Callable[..., object],
    decompose: Callable[..., object],
) -> Any:
    from qualtran import Bloq, QBit
    from qualtran import Signature as SourceSignature

    factory: Any = type(
        "identity",
        (Bloq,),
        {
            "signature": SourceSignature.build_from_dtypes(x=QBit()),
            "on_classical_vals": values,
            "basis_state_phase": phase,
            "decompose_bloq": decompose,
        },
    )
    return factory()


def failure_chain(error: BaseException) -> list[tuple[type[BaseException], str]]:
    result = []
    current: BaseException | None = error
    while current is not None:
        result.append((type(current), str(current)))
        current = current.__cause__
    return result


@pytest.mark.parametrize("phased", [False, True])
@pytest.mark.parametrize("stage", ["values", "phase", "recurse"])
@pytest.mark.parametrize(
    "error_type", [NotImplementedError, ValueError, TypeError, KeyError]
)
def test_only_recursive_not_implemented_is_wrapped(
    phased: bool, stage: str, error_type: type[Exception]
) -> None:
    from qualtran.simulation.classical_sim import (
        ClassicalSimState,
        PhasedClassicalSimState,
    )

    error = error_type("leaf failure")
    trace: list[str] = []

    def action(at: str) -> object:
        trace.append(at)
        if at == stage:
            raise error
        return NotImplemented if at == "values" else None

    leaf = source_leaf(
        lambda self, **values: action("values"),
        lambda self, **values: action("phase"),
        lambda self: action("recurse"),
    )
    record = identity_record(())

    def recurse(
        function: FunctionRecord, values: dict[str, object]
    ) -> tuple[dict[str, object], None]:
        action("recurse")
        raise AssertionError("the failing decomposition must not return")

    def phase(function: FunctionRecord, values: dict[str, object]) -> None:
        action("phase")

    native = ComponentClassicalState(
        record,
        (0,),
        {"x": 1},
        lambda f, v: action("values"),
        phase,
        recurse=recurse,
        phased=phased,
    )
    source = (PhasedClassicalSimState if phased else ClassicalSimState).from_cbloq(
        leaf.as_composite_bloq(), {"x": 1}
    )
    with pytest.raises(error_type) as actual:
        native.simulate()
    native_trace = trace[:]
    trace.clear()
    with pytest.raises(error_type) as expected:
        source.simulate()
    assert native_trace == trace == ["values", "phase", "recurse"][: len(trace)]
    assert failure_chain(actual.value) == failure_chain(expected.value)
    if stage == "recurse" and error_type is NotImplementedError:
        assert actual.value.__cause__ is expected.value.__cause__ is error
    else:
        assert actual.value is expected.value is error
    block = record.functions[0].blocks[0]
    assert native.assignments == {block.arguments[0]: 1}
    assert native.last_position == 0 and native.phase == 1
    with pytest.raises(KeyError):
        native.finalize()


@pytest.mark.parametrize("depth", [2, 3, 5])
@pytest.mark.parametrize("phased", [False, True])
@pytest.mark.parametrize("value", [0, 1])
def test_nested_real_graphs_preserve_each_cause_and_uncommitted_parent(
    depth: int, phased: bool, value: int
) -> None:
    from qualtran import Bloq
    from qualtran.simulation.classical_sim import (
        ClassicalSimState,
        PhasedClassicalSimState,
    )

    error = NotImplementedError("terminal classical action unavailable")

    def fail(*args: object, **kwargs: object) -> object:
        raise error

    source = source_leaf(fail, lambda self, **v: None, fail)
    leaf = identity_record(()).functions[1]
    definitions: tuple[FunctionRecord, ...] = (leaf,)
    previous = leaf.name
    assert leaf.registers is not None
    from flagquantum.core.ir.program.signatures import Signature

    for level in range(depth):
        name = "main" if level == depth - 1 else f"layer{level}"
        builder = ComponentProgramBuilder(
            Signature(leaf.registers), definitions, name=name
        )
        record = builder.finalize(builder.call_dict(previous, builder.inputs))
        body = source.as_composite_bloq()
        factory: Any = type(
            name,
            (Bloq,),
            {
                "signature": source.signature,
                "decompose_bloq": lambda self, body=body: body,
            },
        )
        source = factory()
        definitions, previous = record.functions, name
    states: list[ComponentClassicalState] = []

    def values(function: FunctionRecord, inputs: dict[str, object]) -> object:
        if function.name == "identity":
            raise error
        return NotImplemented

    def recurse(
        function: FunctionRecord, inputs: dict[str, object]
    ) -> tuple[dict[str, object], complex | MeasurementOutcomePhase | None]:
        state = ComponentClassicalState(
            record,
            order(record, function.name),
            inputs,
            values,
            entry=function.name,
            recurse=recurse,
            phased=phased,
        )
        states.append(state)
        return state.simulate(), state.phase if phased else None

    native = ComponentClassicalState(
        record, order(record), {"x": value}, values, recurse=recurse, phased=phased
    )
    source_state = (
        PhasedClassicalSimState if phased else ClassicalSimState
    ).from_cbloq(source.decompose_bloq(), {"x": value})
    with pytest.raises(NotImplementedError) as actual:
        native.simulate()
    with pytest.raises(NotImplementedError) as expected:
        source_state.simulate()
    assert failure_chain(actual.value) == failure_chain(expected.value)
    assert len(failure_chain(actual.value)) == depth
    assert len(states) == depth - 1
    for state in [native, *states]:
        assert state.phase == 1 and state.last_position == 0
        assert len(state.assignments) == 1
        with pytest.raises(KeyError):
            state.finalize()

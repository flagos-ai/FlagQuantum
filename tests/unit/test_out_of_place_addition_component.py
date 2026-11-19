"""Directional sum outputs retain frozen source gates and measured consumption."""

from collections import Counter
from itertools import product
from typing import Any

import pytest
import torch

from flagquantum.compiler.registers.arithmetic.out_of_place_add_resource_model import (
    source_out_of_place_add_resource_formula,
)
from flagquantum.compiler.registers.arithmetic.out_of_place_addition_component import (
    out_of_place_addition_program,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.core.ir.program import ProgramRecord, verify_program_record
from flagquantum.core.ir.program.signatures import Signature
from tests.unit.measurement_checks import forced_branch
from tests.unit.register_example_checks import (
    CHECKS,
    CheckKind,
    CheckStatus,
    RegisterExample,
    check_example,
)
from tests.unit.test_component_graph import _native_trace, _source, _source_trace
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", (1, 2, 3, 5, 8))
@pytest.mark.parametrize("include_msb", (False, True))
@pytest.mark.parametrize("adjoint", (False, True))
def test_source_physical_dependencies_outputs_and_resource_model(
    width: int,
    include_msb: bool,
    adjoint: bool,
) -> None:
    from qualtran.bloqs.arithmetic.addition import OutOfPlaceAdder
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    graph = out_of_place_addition_program(
        width, include_msb=include_msb, adjoint=adjoint
    )
    assert verify_program_record(graph) == ()
    c = tuple(range(2 * width, 3 * width + int(include_msb)))
    ports = {"a": tuple(range(width)), "b": tuple(range(width, 2 * width))}
    outputs = dict(ports)
    if adjoint:
        ports["c"] = c
    else:
        outputs["c"] = c
    bound = bind_register_program(
        graph,
        tuple(ports.values()),
        clean=() if adjoint else c[::-1],
        allow_workspace_outputs=not adjoint,
        input_ownership=tuple((wire, False) for wire in c) if adjoint else (),
    )
    source = OutOfPlaceAdder(
        width, include_most_significant_bit=include_msb, is_adjoint=adjoint
    )
    # Cirq conversion schedules independent XORs before Allocate. Canonical
    # dependency order preserves their port edges without requiring instance IDs.
    # Frozen Cirq omits one-bit Split/Join; native integer ports keep those
    # zero-cost type views. Multi-bit words have identical comparable edges.
    if width > 1:
        native_trace = _native_trace(graph)
        source_trace = _source_trace(_source(Signature([])), source.decompose_bloq())
        identities: dict[tuple[object, ...], int] = {}

        def canonical(trace: list[object]) -> Counter[int]:
            predecessors: dict[str, int] = {}
            counts: Counter[int] = Counter()

            def references(value: Any) -> Any:
                if isinstance(value, tuple):
                    if (
                        len(value) == 3
                        and isinstance(value[0], str)
                        and value[0] in predecessors
                    ):
                        return (predecessors[value[0]], value[1], value[2])
                    return tuple(references(item) for item in value)
                return value

            for row in trace:
                assert isinstance(row, tuple)
                name, inputs, outputs = row
                key = (
                    tuple(sorted(references(inputs))),
                    tuple(output[0] for output in outputs),
                )
                identity = identities.setdefault(key, len(identities))
                predecessors[name] = identity
                counts[identity] += 1
            return counts

        assert canonical(native_trace) == canonical(source_trace)
    calls = Counter(op.callee for op in graph.functions[0].blocks[0].operations)
    assert calls["sum_and"] == width - 1 + int(include_msb)
    assert calls["sum_cx"] == sum(
        str(instance.bloq) == "CNOT"
        for instance, _, _ in source.decompose_bloq().iter_bloqsoqs()
    )
    slots = {
        g.wires[0]: g.metadata["classical_bit"]
        for g in bound.circuit.instructions
        if g.name == "measure"
    }
    original = reference_arithmetic_gates(
        source, ports, () if adjoint else c[::-1], slots, expected_outputs=outputs
    )
    assert_source(bound.circuit.instructions, original)
    assert bound.outputs == tuple(outputs.values())
    counts = Counter(g.name for g in bound.circuit.instructions)
    ands = width - 1 + int(include_msb)
    model = get_cost_value(source, QECGatesCost())
    assert (
        counts["t"] + counts["tdg"]
        == model.total_t_count()
        == (0 if adjoint else 4 * ands)
    )
    assert (
        counts["measure"]
        == counts["reset"]
        == model.measurement
        == (ands if adjoint else 0)
    )
    formula = source_out_of_place_add_resource_formula(
        width, include_msb=include_msb, adjoint=adjoint
    )
    for category in (
        "t",
        "toffoli",
        "cswap",
        "and_bloq",
        "clifford",
        "rotation",
        "measurement",
    ):
        assert getattr(formula, category) == getattr(model, category)
    assert (
        formula.logical_qubits
        == get_cost_value(source, QubitCount())
        == 2 * width + len(c)
    )
    assert Counter(event.kind for event in bound.events) == {
        "free" if adjoint else "allocate": len(c)
    }


@pytest.mark.parametrize("width", (1, 2, 3))
@pytest.mark.parametrize("include_msb", (False, True))
def test_source_full_isometry_entangled_spectator_and_all_erasure_histories(
    width: int,
    include_msb: bool,
) -> None:
    from qualtran.bloqs.arithmetic.addition import OutOfPlaceAdder

    count = width + int(include_msb)
    c = tuple(range(2 * width + 1, 2 * width + 1 + count))
    inputs = (tuple(range(width)), tuple(range(width, 2 * width)))
    forward = bind_register_program(
        out_of_place_addition_program(width, include_msb=include_msb),
        inputs,
        clean=c[::-1],
        allow_workspace_outputs=True,
    )
    backward = bind_register_program(
        out_of_place_addition_program(width, include_msb=include_msb, adjoint=True),
        (*inputs, c),
        input_ownership=tuple((wire, False) for wire in c),
    )
    raw = torch.arange(1, (1 << (2 * width + 1)) + 1, dtype=torch.float64)
    amplitudes = torch.complex(raw, 5 - 2 * raw)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(1 << (2 * width + 1 + count), dtype=torch.complex128)
    initial[:: 1 << count] = amplitudes
    source = OutOfPlaceAdder(width, include_most_significant_bit=include_msb)
    matrix = torch.as_tensor(source.tensor_contract(), dtype=torch.complex128).reshape(
        1 << (2 * width + count), 1 << (2 * width)
    )
    expected = (
        (matrix @ amplitudes.reshape(-1, 2))
        .reshape(1 << (2 * width), 1 << count, 2)
        .permute(0, 2, 1)
        .flatten()
    )
    actual = forced_branch(
        initial, CircuitIR.from_json(forward.circuit.to_json()).instructions, ()
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=3e-12)
    for history in product((0, 1), repeat=count - 1):
        recovered = forced_branch(actual, backward.circuit.instructions, history)
        torch.testing.assert_close(
            recovered, initial / 2 ** ((count - 1) / 2), rtol=0, atol=3e-12
        )
        assert torch.count_nonzero(recovered.reshape(-1, 1 << count)[:, 1:]) == 0


@pytest.mark.parametrize("large", (False, True))
@pytest.mark.parametrize("kind", CHECKS)
def test_actual_concrete_factories_all_existing_checks(
    large: bool, kind: CheckKind
) -> None:
    from qualtran.bloqs.arithmetic.addition import _add_oop_large, _add_oop_small

    source = (_add_oop_large if large else _add_oop_small).make()
    width = int(source.bitsize)
    inputs = (tuple(range(width)), tuple(range(width, 2 * width)))
    c = tuple(range(2 * width, 3 * width + 1))
    original = reference_arithmetic_gates(
        source,
        dict(zip(("a", "b"), inputs, strict=True)),
        c[::-1],
        {},
        expected_outputs={"a": inputs[0], "b": inputs[1], "c": c},
    )

    def lower(record: ProgramRecord) -> CircuitIR:
        return bind_register_program(
            record, inputs, clean=c[::-1], allow_workspace_outputs=True
        ).circuit

    result = check_example(
        RegisterExample(
            lambda: out_of_place_addition_program(width),
            lower,
            lambda: Counter(g.name for g in original),
        ),
        kind,
    )
    assert result.status is CheckStatus.PASS, result.message


def test_inverse_requires_explicit_input_pool_ownership() -> None:
    with pytest.raises(ValueError, match="owned|ownership|pool"):
        bind_register_program(
            out_of_place_addition_program(1, adjoint=True), ((0,), (1,), (2, 3))
        )

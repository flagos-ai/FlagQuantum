"""Stored recursive phase graphs reuse the existing numerical state owner."""

from collections import Counter
from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.layout.component_workspace import (
    allocate_word_function,
)
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import (
    FunctionRecord,
    Operation,
    ProgramRecord,
    ValueRef,
)
from flagquantum.core.ir.program.register_types import quantum_bit_type
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.classical_measurement import measure_x_basis_phase
from flagquantum.simulation.component_classical_state import (
    simulate_phased_component_calls,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_classical_state import (
    add_values,
    measurement_record,
    measurement_values,
    order,
)
from tests.unit.test_component_expansion import _hierarchy
from tests.unit.test_component_hierarchy_adjoint import wrapped_add

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(autouse=True)
def source_notebook_marker(pytestconfig: pytest.Config) -> None:
    # Import the original test's ComposedPhasing class without collecting its
    # notebook test or changing the frozen source module.
    pytestconfig.addinivalue_line("markers", "notebook: upstream notebook test")


def orders(record: ProgramRecord) -> dict[str, tuple[int, ...]]:
    return {
        f.name: order(record, f.name)
        for f in record.functions
        if all(op.opcode == "call" for op in f.blocks[0].operations)
    }


def wrap(record: ProgramRecord, source: Any, depth: int) -> tuple[ProgramRecord, Any]:
    from qualtran import Bloq

    for level in range(depth):
        old = next(f for f in record.functions if f.name == "main")
        name = f"outer{level}"
        definitions = tuple(
            replace(f, name=name) if f is old else f for f in record.functions
        )
        builder = ComponentProgramBuilder(Signature(old.registers or ()), definitions)
        record = builder.finalize(builder.call_dict(name, builder.inputs))
        factory: Any = type(
            name,
            (Bloq,),
            {
                "signature": source.signature,
                "decompose_bloq": lambda self, body=source: body,
            },
        )
        source = factory().as_composite_bloq()
    return record, source


def phase_fixture(n: int, depth: int) -> tuple[ProgramRecord, Any]:
    from qualtran.simulation.classical_sim_test import ComposedPhasing

    dtype = quantum_bit_type()
    one = allocate_word_function("one", dtype, False)
    block = one.blocks[0]
    q, wire, flipped, result = (
        ValueRef("entry", i, "register" if i in (0, 3) else "qubit") for i in range(4)
    )
    one = replace(
        one,
        blocks=(
            replace(
                block,
                operations=(
                    *block.operations,
                    Operation("split_register", (q,), (wire,)),
                    Operation("x", (wire,), (flipped,)),
                    Operation(
                        "join_register", (flipped,), (result,), register_type=dtype
                    ),
                ),
                return_values=(result,),
            ),
        ),
    )
    template = _hierarchy(1, 1)[0].functions[1]
    z = replace(
        template,
        name="z",
        blocks=tuple(
            replace(
                b,
                operations=tuple(
                    replace(op, opcode="z") if op.opcode == "h" else op
                    for op in b.operations
                ),
            )
            for b in template.blocks
        ),
    )
    builder = ComponentProgramBuilder(
        Signature([Register("x", dtype, side=RegisterSide.RIGHT)]), (one, z)
    )
    x = builder.call_dict("one", {})["reg"]
    for _ in range(n):
        x = builder.call_dict("z", {"q": x})["q"]
    return wrap(builder.finalize({"x": x}), ComposedPhasing(n).decompose_bloq(), depth)


def phase_values(function: FunctionRecord, values: dict[str, object]) -> object:
    if function.name == "one":
        return {"reg": 1}
    return values if function.name == "z" else NotImplemented


def phase_action(function: FunctionRecord, values: dict[str, object]) -> complex | None:
    return complex(-1 if values["q"] == 1 else 1) if function.name == "z" else None


@pytest.mark.parametrize("n", [0, 1, 2, 3, 7])
@pytest.mark.parametrize("depth", [1, 2, 4])
def test_original_composed_phasing_and_physical_append(n: int, depth: int) -> None:
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.simulation.classical_sim import do_phased_classical_simulation

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.circuit import Circuit
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates

    record, source = phase_fixture(n, depth)
    actual = simulate_phased_component_calls(
        record, orders(record), {}, phase_values, phase_action
    )
    expected = do_phased_classical_simulation(source, {})
    assert actual == expected == ({"x": 1}, (-1) ** n)
    bound = bind_register_program(
        record, (), clean=(0, 1), allow_workspace_outputs=True
    )
    assert bound.outputs == ((0,),)
    assert Counter(g.name for g in bound.circuit.instructions) == (
        {"x": 1, "z": n} if n else {"x": 1}
    )
    expected_stream = reference_arithmetic_gates(
        source, {}, (0, 1), {}, expected_outputs={"x": (0,)}, keep_pauli_leaves=True
    )
    assert [(g.name, g.wires) for g in bound.circuit.instructions] == [
        (g.name, g.wires) for g in expected_stream
    ]
    model = get_cost_value(source, QECGatesCost())
    assert model.total_t_count() == 0 and model.measurement == 0
    initial = torch.tensor([1, 0, 0, 0], dtype=torch.complex128)
    state = (
        Circuit(2, inputs=initial, dtype=torch.complex128)
        .append(FieldOperation(2, bound.circuit.instructions))
        .state()
        .flatten()
    )
    target = torch.zeros_like(initial)
    target[2] = torch.tensor(actual[1], dtype=target.dtype)
    torch.testing.assert_close(state, target, rtol=0, atol=1e-12)
    np.testing.assert_allclose(state[::2].numpy(), source.tensor_contract(), atol=1e-12)
    assert torch.count_nonzero(state[1::2]) == 0


@pytest.mark.parametrize("depth", [1, 2, 4])
@pytest.mark.parametrize("bit,outcome", list(product((0, 1), repeat=2)))
def test_nested_fixed_measurement_all_complex_branches(
    depth: int, bit: int, outcome: int
) -> None:
    from qualtran.bloqs.basic_gates import MeasureX
    from qualtran.simulation.classical_sim import do_phased_classical_simulation

    record, source = wrap(measurement_record(), MeasureX().as_composite_bloq(), depth)
    actual = simulate_phased_component_calls(
        record,
        orders(record),
        {"q": bit},
        lambda f, v: (
            measurement_values(f, v) if f.name == "measure_x" else NotImplemented
        ),
        lambda f, v: measure_x_basis_phase(v["q"]) if f.name == "measure_x" else None,
        fixed_values={0: outcome},
    )
    assert actual == do_phased_classical_simulation(
        source, {"q": bit}, fixed_random_vals={0: outcome}
    )
    bound = bind_register_program(record, ((0,),), clean=(1,))
    initial = torch.zeros(4, dtype=torch.complex128)
    initial[2 * bit] = 1
    branch = forced_branch(initial, bound.circuit.instructions, (outcome,))
    expected = torch.zeros_like(branch)
    expected[2 * outcome] = torch.tensor(actual[1] / 2**0.5, dtype=expected.dtype)
    torch.testing.assert_close(branch, expected, rtol=0, atol=1e-12)
    assert torch.count_nonzero(branch[1::2]) == 0
    assert Counter(g.name for g in bound.circuit.instructions) == {"h": 1, "measure": 1}


@pytest.mark.parametrize("depth", [1, 2, 4])
@pytest.mark.parametrize("seed", [0, 19, 201])
def test_nested_rng_shares_handler_and_preserves_precedence(
    depth: int, seed: int
) -> None:
    from qualtran.bloqs.basic_gates import MeasureX
    from qualtran.simulation.classical_sim import do_phased_classical_simulation

    record, source = wrap(measurement_record(), MeasureX().as_composite_bloq(), depth)
    rng, source_rng = np.random.default_rng(seed), np.random.default_rng(seed)
    for _ in range(16):
        actual = simulate_phased_component_calls(
            record,
            orders(record),
            {"q": 1},
            lambda f, v: (
                measurement_values(f, v) if f.name == "measure_x" else NotImplemented
            ),
            lambda f, v: (
                measure_x_basis_phase(v["q"]) if f.name == "measure_x" else None
            ),
            rng=rng,
            fixed_values={},
        )
        assert actual == do_phased_classical_simulation(
            source, {"q": 1}, rng=source_rng, fixed_random_vals={}
        )
        assert rng.bit_generator.state == source_rng.bit_generator.state


@pytest.mark.parametrize("depth", [1, 2, 4])
@pytest.mark.parametrize("outcome", [0, 1])
def test_nested_measurement_reconstructs_complex_superposition_branches(
    depth: int, outcome: int
) -> None:
    from qualtran.bloqs.basic_gates import MeasureX

    record, _ = wrap(measurement_record(), MeasureX().as_composite_bloq(), depth)
    coefficients = torch.tensor([1 + 2j, -3 + 4j], dtype=torch.complex128) / 30**0.5
    expected = torch.zeros(4, dtype=torch.complex128)
    for bit in (0, 1):
        values, phase = simulate_phased_component_calls(
            record,
            orders(record),
            {"q": bit},
            lambda f, v: (
                measurement_values(f, v) if f.name == "measure_x" else NotImplemented
            ),
            lambda f, v: (
                measure_x_basis_phase(v["q"]) if f.name == "measure_x" else None
            ),
            fixed_values={0: outcome},
        )
        assert values["c"] == outcome
        expected[2 * outcome] += coefficients[bit] * phase / 2**0.5
    initial = torch.zeros_like(expected)
    initial[::2] = coefficients
    bound = bind_register_program(record, ((0,),), clean=(1,))
    actual = forced_branch(initial, bound.circuit.instructions, (outcome,))
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
    assert torch.count_nonzero(actual[1::2]) == 0


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_recursive_add_all_basis_values_without_caller_recursion(depth: int) -> None:
    from qualtran.simulation.classical_sim import do_phased_classical_simulation

    record, source = wrapped_add(depth)
    for a, b, spectator in product(range(4), range(8), range(2)):
        values = {"a": a, "b": b, "spectator": spectator}
        actual = simulate_phased_component_calls(
            record, orders(record), values, add_values
        )
        assert actual == do_phased_classical_simulation(source, values)
        assert actual == ({"a": a, "b": (b + 2 * a) % 8, "spectator": spectator}, 1)


@pytest.mark.parametrize(
    "kind",
    [
        "missing_root",
        "missing_child",
        "bad_order",
        "no_handler",
        "fixed_missing",
        "bad_phase",
    ],
)
def test_recursive_evaluation_refuses_missing_policies_and_bad_actions(
    kind: str,
) -> None:
    record, _ = phase_fixture(1, 2)
    schedules = orders(record)
    if kind == "missing_root":
        schedules.pop("main")
    elif kind == "missing_child":
        schedules.pop("outer1")
    elif kind == "bad_order":
        schedules["outer1"] = ()
    if kind in {"no_handler", "fixed_missing"}:
        from qualtran.bloqs.basic_gates import MeasureX

        record, _ = wrap(measurement_record(), MeasureX().as_composite_bloq(), 2)
        with pytest.raises(KeyError if kind == "fixed_missing" else ValueError):
            simulate_phased_component_calls(
                record,
                orders(record),
                {"q": 1},
                lambda f, v: (
                    measurement_values(f, v)
                    if f.name == "measure_x"
                    else NotImplemented
                ),
                fixed_values={} if kind == "fixed_missing" else None,
            )
        return
    expected_error = NotImplementedError if kind.startswith("missing") else ValueError
    with pytest.raises(expected_error):
        simulate_phased_component_calls(
            record,
            schedules,
            {},
            phase_values,
            (
                (lambda f, v: 2 if f.name == "z" else None)
                if kind == "bad_phase"
                else phase_action
            ),
        )

"""Compiler occurrence labels survive recursive numerical measurement binding."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
from flagquantum.compiler.registers.component_expansion import expand_component_calls
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord, ProgramRecord
from flagquantum.core.ir.program.register_types import (
    classical_bit_type,
    quantum_bit_type,
)
from flagquantum.core.ir.program.registers import Register, RegisterSide
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.simulation.classical_measurement import measure_x_basis_phase
from flagquantum.simulation.component_classical_state import (
    simulate_phased_component_calls,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_classical_state import (
    measurement_record,
    measurement_values,
)
from tests.unit.test_component_phased_evaluation import orders, wrap

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def pair_fixture(
    depth: int, nested: bool = False
) -> tuple[ProgramRecord, Any, dict[str, tuple[int, ...]]]:
    from qualtran import Bloq, BloqBuilder, CBit, QBit
    from qualtran import Register as SourceRegister
    from qualtran import Side as SourceSide
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import MeasureX

    record, source = wrap(measurement_record(), MeasureX().as_composite_bloq(), depth)
    record = replace(
        record,
        functions=tuple(
            replace(f, name="branch") if f.name == "main" else f
            for f in record.functions
        ),
    )
    ids: dict[str, tuple[int, ...]] = {}
    if nested:
        record, labels = expand_component_calls(record, "branch")
        source = source.flatten(lambda i: not isinstance(i.bloq, MeasureX))
        assert labels == tuple(i.i for i, _, _ in source.iter_bloqsoqs())
        ids["branch"] = labels
        # A declared body enters the source phased recursion protocol. Passing
        # CompositeBloq as a direct numerical leaf instead invokes its pure
        # on_classical_vals action, a separate source limitation.
        factory: Any = type(
            "branch",
            (Bloq,),
            {
                "signature": source.signature,
                "decompose_bloq": lambda self, body=source: body,
            },
        )
        source = factory()
    signature = Signature(
        [
            Register("q", quantum_bit_type(), (2,), RegisterSide.LEFT),
            Register("c", classical_bit_type(), (2,), RegisterSide.RIGHT),
        ]
    )
    builder = ComponentProgramBuilder(signature, record.functions)
    inputs = builder.inputs["q"]
    assert isinstance(inputs, np.ndarray)
    outputs = np.empty((2,), dtype=object)
    bb, source_inputs = BloqBuilder.from_signature(
        SourceSignature(
            [
                SourceRegister("q", QBit(), (2,), SourceSide.LEFT),
                SourceRegister("c", CBit(), (2,), SourceSide.RIGHT),
            ]
        )
    )
    source_outputs = np.empty((2,), dtype=object)
    for index in range(2):
        outputs[index] = builder.call_dict("branch", {"q": inputs[index]})["c"]
        source_outputs[index] = bb.add_d(source, q=source_inputs["q"][index])["c"]
    record, source = builder.finalize({"c": outputs}), bb.finalize(c=source_outputs)
    if not nested:
        record, labels = expand_component_calls(record)
        source = source.flatten(lambda i: not isinstance(i.bloq, MeasureX))
        assert labels == tuple(i.i for i, _, _ in source.iter_bloqsoqs())
        ids["main"] = labels
    return record, source, ids


def evaluate(
    record: ProgramRecord,
    ids: Mapping[str, tuple[int, ...]] | None,
    values: Mapping[str, object],
    *,
    rng: np.random.Generator | None = None,
    fixed: Mapping[int, object] | None = None,
    schedules: Mapping[str, tuple[int, ...]] | None = None,
) -> tuple[dict[str, object], complex]:
    return simulate_phased_component_calls(
        record,
        orders(record) if schedules is None else schedules,
        values,
        lambda f, v: (
            measurement_values(f, v) if f.name == "measure_x" else NotImplemented
        ),
        lambda f, v: measure_x_basis_phase(v["q"]) if f.name == "measure_x" else None,
        instance_ids=ids,
        rng=rng,
        fixed_values=fixed,
    )


def reverse_root_storage(record: ProgramRecord) -> ProgramRecord:
    return replace(
        record,
        functions=tuple(
            (
                replace(
                    f,
                    blocks=(
                        replace(
                            f.blocks[0],
                            operations=tuple(reversed(f.blocks[0].operations)),
                        ),
                    ),
                )
                if f.name == "main"
                else f
            )
            for f in record.functions
        ),
    )


@pytest.mark.parametrize("depth", [0, 1, 3])
@pytest.mark.parametrize("outcomes", list(product((0, 1), repeat=2)))
@pytest.mark.parametrize("reverse", [False, True])
def test_expanded_fixed_labels_values_all_complex_branches_and_work_zero(
    depth: int, outcomes: tuple[int, int], reverse: bool
) -> None:
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.simulation.classical_sim import do_phased_classical_simulation

    record, source, ids = pair_fixture(depth)
    labels = ids["main"]
    assert labels != (0, 1) and len(set(labels)) == 2
    fixed = dict(zip(labels, outcomes, strict=True))
    if reverse:
        record = reverse_root_storage(record)
        ids = {"main": tuple(reversed(labels))}
    # Supplying only a visitation schedule cannot recover persistent IDs.
    with pytest.raises(KeyError):
        evaluate(record, None, {"q": np.array([0, 1])}, fixed=fixed)
    actual, phase = evaluate(record, ids, {"q": np.array([0, 1])}, fixed=fixed)
    expected, expected_phase = do_phased_classical_simulation(
        source, {"q": np.array([0, 1])}, fixed_random_vals=fixed
    )
    np.testing.assert_equal(actual["c"], expected["c"])
    np.testing.assert_equal(actual["c"], outcomes)
    assert phase == expected_phase == (-1) ** outcomes[1]
    coefficients = torch.tensor([1 + 2j, -3j, 4, -2 + 1j], dtype=torch.complex128)
    coefficients /= coefficients.norm()
    branch_target = torch.zeros(8, dtype=torch.complex128)
    for number, bits in enumerate(product((0, 1), repeat=2)):
        _, contribution = evaluate(record, ids, {"q": np.array(bits)}, fixed=fixed)
        branch_target[2 * (2 * outcomes[0] + outcomes[1])] += (
            coefficients[number] * contribution / 2
        )
    initial = torch.zeros_like(branch_target)
    initial[::2] = coefficients
    bound = bind_register_program(record, ((0,), (1,)), clean=(2,))
    history = tuple(
        outcomes[g.wires[0]] for g in bound.circuit.instructions if g.name == "measure"
    )
    branch = forced_branch(initial, bound.circuit.instructions, history)
    torch.testing.assert_close(branch, branch_target, rtol=0, atol=1e-12)
    assert torch.count_nonzero(branch[1::2]) == 0
    assert Counter(g.name for g in bound.circuit.instructions) == {"h": 2, "measure": 2}
    assert get_cost_value(source, QECGatesCost()).measurement == 2


@pytest.mark.parametrize("depth", [0, 1, 3])
@pytest.mark.parametrize("outcome", [0, 1])
def test_nested_body_ids_retain_source_local_identity_reuse(
    depth: int, outcome: int
) -> None:
    from qualtran.simulation.classical_sim import do_phased_classical_simulation

    record, source, ids = pair_fixture(depth, nested=True)
    assert "main" not in ids and len(ids["branch"]) == 1
    fixed = {ids["branch"][0]: outcome}
    values = {"q": np.array([1, 1])}
    actual, phase = evaluate(record, ids, values, fixed=fixed)
    expected, expected_phase = do_phased_classical_simulation(
        source, values, fixed_random_vals=fixed
    )
    np.testing.assert_equal(actual["c"], expected["c"])
    np.testing.assert_equal(actual["c"], [outcome, outcome])
    assert phase == expected_phase == 1


@pytest.mark.parametrize("seed", [0, 19, 201])
@pytest.mark.parametrize("reverse", [False, True])
def test_rng_sequence_obeys_visitation_not_stored_position(
    seed: int, reverse: bool
) -> None:
    from qualtran.simulation.classical_sim import do_phased_classical_simulation

    record, source, ids = pair_fixture(2)
    schedules = orders(record)
    if reverse:
        record = reverse_root_storage(record)
        ids = {"main": tuple(reversed(ids["main"]))}
        schedules["main"] = (1, 0)
    rng, source_rng = np.random.default_rng(seed), np.random.default_rng(seed)
    for _ in range(16):
        values = {"q": np.array([0, 1])}
        actual, phase = evaluate(
            record, ids, values, rng=rng, fixed={}, schedules=schedules
        )
        expected, expected_phase = do_phased_classical_simulation(
            source, values, rng=source_rng, fixed_random_vals={}
        )
        np.testing.assert_equal(actual["c"], expected["c"])
        assert phase == expected_phase
        assert rng.bit_generator.state == source_rng.bit_generator.state


@pytest.mark.parametrize("bad_ids", [(), (1,), (1, 1), (-1, 2), (False, 2), (1, 2, 3)])
def test_invalid_ids_reuse_state_refusal_before_numerical_actions(
    bad_ids: tuple[int, ...],
) -> None:
    record, _, _ = pair_fixture(1)
    with pytest.raises(ValueError, match="instance IDs"):
        evaluate(record, {"main": bad_ids}, {"q": np.array([0, 1])}, fixed={})


def test_nested_instance_mapping_is_snapshotted_before_actions() -> None:
    record, _, ids = pair_fixture(2, nested=True)
    label = ids["branch"][0]

    def values(function: FunctionRecord, inputs: dict[str, object]) -> object:
        if function.name == "branch":
            ids["branch"] = (999,)
        return (
            measurement_values(function, inputs)
            if function.name == "measure_x"
            else NotImplemented
        )

    outputs, phase = simulate_phased_component_calls(
        record,
        orders(record),
        {"q": np.array([0, 1])},
        values,
        lambda f, v: measure_x_basis_phase(v["q"]) if f.name == "measure_x" else None,
        instance_ids=ids,
        fixed_values={label: 1},
    )
    np.testing.assert_equal(outputs["c"], [1, 1])
    assert phase == -1 and ids["branch"] == (999,)


def test_source_direct_composite_action_pure_recursion_boundary_is_retained() -> None:
    from qualtran import BloqBuilder
    from qualtran.bloqs.basic_gates import MeasureX
    from qualtran.simulation.classical_sim import do_phased_classical_simulation

    bb, inputs = BloqBuilder.from_signature(MeasureX().signature)
    outputs = bb.add_d(MeasureX().as_composite_bloq(), **inputs)
    source = bb.finalize(**outputs)
    with pytest.raises(ValueError, match="can't be simulated purely classically"):
        do_phased_classical_simulation(source, {"q": 1}, fixed_random_vals={0: 1})

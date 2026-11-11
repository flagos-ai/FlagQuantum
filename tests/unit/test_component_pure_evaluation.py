"""Pure stored recursion preserves the source's separate default random handler."""

from collections import Counter
from itertools import product
from typing import Any

import numpy as np
import pytest
import torch

from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir.program import FunctionRecord
from flagquantum.simulation import component_pure_evaluation as native
from flagquantum.simulation.classical_distributions import ClassicalDistribution
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_bit_values import identity_record
from tests.unit.test_component_classical_recursion import source_leaf
from tests.unit.test_component_classical_state import add_values
from tests.unit.test_component_hierarchy_adjoint import wrapped_add
from tests.unit.test_component_phased_evaluation import orders, wrap
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("depth", [0, 1, 3])
@pytest.mark.parametrize("policy", ["default", "seeded", "fixed", "both"])
@pytest.mark.parametrize("seed", [0, 19, 201])
def test_pure_rng_matches_original_root_and_recursive_streams(
    depth: int, policy: str, seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qualtran.simulation.classical_sim import (
        ClassicalValDistribution,
        _FixedClassicalValHandler,
        _RandomClassicalValHandler,
        call_cbloq_classically,
    )

    leaf = source_leaf(
        lambda self, **v: {"x": ClassicalValDistribution([0, 1], [0.3, 0.7])},
        lambda self, **v: None,
        lambda self: NotImplemented,
    )
    record, source = wrap(identity_record(()), leaf.as_composite_bloq(), depth)
    implicit, expected_implicit = np.random.default_rng(seed), np.random.default_rng(
        seed
    )
    root, expected_root = np.random.default_rng(seed + 1), np.random.default_rng(
        seed + 1
    )
    monkeypatch.setattr(native, "_DEFAULT_RNG", implicit)
    # Seed the existing default handler temporarily, without replacing source code.
    defaults: Any = call_cbloq_classically.__defaults__
    monkeypatch.setattr(defaults[0], "_gen", expected_implicit)
    rng = root if policy in {"seeded", "both"} else None
    fixed = {0: 0} if policy in {"fixed", "both"} else None
    handler = (
        _RandomClassicalValHandler(expected_root)
        if rng is not None
        else _FixedClassicalValHandler(fixed) if fixed is not None else defaults[0]
    )

    def values(function: FunctionRecord, inputs: dict[str, object]) -> object:
        return (
            {"x": ClassicalDistribution([0, 1], [0.3, 0.7])}
            if function.name == "identity"
            else NotImplemented
        )

    observed = []
    for _ in range(24):
        actual = native.simulate_pure_component_calls(
            record, orders(record), {"x": 0}, values, rng=rng, fixed_values=fixed
        )
        expected, _ = call_cbloq_classically(
            source.signature, {"x": 0}, source._binst_graph, random_handler=handler
        )
        assert actual == expected
        observed.append(actual["x"])
        assert implicit.bit_generator.state == expected_implicit.bit_generator.state
        assert root.bit_generator.state == expected_root.bit_generator.state
    if depth:
        assert set(observed) == {0, 1}  # Root fixed values do not reach nested bodies.
        assert (
            root.bit_generator.state
            == np.random.default_rng(seed + 1).bit_generator.state
        )
    elif policy == "fixed":
        assert set(observed) == {0}


@pytest.mark.parametrize("depth", [0, 1, 3])
@pytest.mark.parametrize("phase", [1, -1, 1j])
def test_pure_recursion_rejects_even_identity_phase(depth: int, phase: complex) -> None:
    from qualtran.simulation.classical_sim import call_cbloq_classically

    leaf = source_leaf(lambda self, **v: v, lambda self, **v: phase, lambda self: None)
    record, source = wrap(identity_record(()), leaf.as_composite_bloq(), depth)
    with pytest.raises(ValueError, match="phase"):
        native.simulate_pure_component_calls(
            record,
            orders(record),
            {"x": 1},
            lambda f, v: v if f.name == "identity" else NotImplemented,
            lambda f, v: phase if f.name == "identity" else None,
        )
    with pytest.raises(ValueError, match="phase"):
        call_cbloq_classically(source.signature, {"x": 1}, source._binst_graph)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_pure_add_complete_complex_state_all_branches_and_resource_accounts(
    depth: int,
) -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic.addition import Add
    from qualtran.resource_counting import QECGatesCost, get_cost_value
    from qualtran.simulation.classical_sim import call_cbloq_classically

    record, source = wrapped_add(depth)
    bound = bind_register_program(record, ((0, 1), (2, 3, 4), (5,)), clean=(6, 7))
    measurements = [g for g in bound.circuit.instructions if g.name == "measure"]
    expected_gates = tuple(
        g
        for offset in (0, 2)
        for g in reference_arithmetic_gates(
            Add(QUInt(2), QUInt(3)),
            {"a": (0, 1), "b": (2, 3, 4)},
            (6, 7),
            {
                g.wires[0]: g.metadata["classical_bit"]
                for g in measurements[offset : offset + 2]
            },
        )
    )
    assert_source(bound.circuit.instructions, expected_gates)
    initial = torch.zeros(256, dtype=torch.complex128)
    coefficients = torch.complex(
        torch.arange(1, 65, dtype=torch.float64),
        torch.arange(64, 0, -1, dtype=torch.float64),
    )
    initial[::4] = coefficients / coefficients.norm()
    target = torch.zeros_like(initial)
    for a, b, spectator in product(range(4), range(8), range(2)):
        inputs = {"a": a, "b": b, "spectator": spectator}
        result = native.simulate_pure_component_calls(
            record, orders(record), inputs, add_values
        )
        expected, _ = call_cbloq_classically(
            source.signature, inputs, source._binst_graph
        )
        assert (
            result == expected == {"a": a, "b": (b + 2 * a) % 8, "spectator": spectator}
        )
        index = (a << 4) | (b << 1) | spectator
        value = result["b"]
        assert isinstance(value, int)
        output = (a << 4) | (value << 1) | spectator
        target[output << 2] = initial[index << 2]
    for branch in product((0, 1), repeat=4):
        actual = forced_branch(initial, bound.circuit.instructions, branch)
        torch.testing.assert_close(actual, target / 4, rtol=0, atol=1e-12)
        assert torch.count_nonzero(actual.reshape(-1, 4)[:, 1:]) == 0
    # Count actual primitives independently of the original resource model.
    assert Counter(g.name for g in bound.circuit.instructions) == Counter(
        g.name for g in expected_gates
    )
    model = get_cost_value(source, QECGatesCost())
    assert model.total_t_count() == 16 and model.measurement == 4


@pytest.mark.parametrize("missing", ["main", "outer0"])
def test_missing_schedule_retains_recursive_cause(missing: str) -> None:
    record = identity_record(())
    leaf = source_leaf(lambda self, **v: v, lambda self, **v: None, lambda self: None)
    record, _ = wrap(record, leaf.as_composite_bloq(), 1)
    schedules = orders(record)
    schedules.pop(missing)
    with pytest.raises(NotImplementedError, match="no classical call order") as failure:
        native.simulate_pure_component_calls(
            record, schedules, {"x": 1}, lambda f, v: NotImplemented
        )
    assert failure.value.__cause__ is not None

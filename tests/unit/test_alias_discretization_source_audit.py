"""Whole-file source preprocessing parity and unchanged coherent consumers."""

import random
import unittest
import warnings
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import asdict
from itertools import product
from typing import Any

import numpy as np
import pytest
import sympy
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.algorithms.primitives.alias_discretization import (
    _differences,
    _partial_sums,
    alias_precision_bits,
    discretize_probabilities,
    integer_alias_rows,
    reversible_sampling_rows,
)
from flagquantum.algorithms.primitives.alias_preparation import (
    alias_preparation_ir,
    source_dense_alias_preparation_cost,
)
from flagquantum.algorithms.primitives.alias_sampling import (
    AliasSamplingTable,
    alias_numerators,
    alias_sampling_table,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.dynamic import DynamicCircuit
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_alias_preparation import expected_state, layout

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def outcome(function: Callable[..., Any], *args: Any, **kwargs: Any) -> object:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            value = function(*args, **kwargs)
            result = ("value", type(value), value)
        except (
            ValueError,
            TypeError,
            ZeroDivisionError,
            OverflowError,
            IndexError,
            AssertionError,
        ) as error:
            result = ("error", type(error), str(error))
    return result, tuple((type(w.message), str(w.message)) for w in caught)


def test_cumulative_rounding_and_donation_exhaust_small_inputs() -> None:
    from qualtran.linalg import lcu_util as source

    for count in range(1, 5):
        for weights in product(range(4), repeat=count):
            if not sum(weights):
                continue
            for mu in range(1, 5):
                rounded = discretize_probabilities(weights, mu)
                assert rounded == source._discretize_probability_distribution(
                    weights, mu
                )
                assert integer_alias_rows(
                    rounded[0]
                ) == source._preprocess_for_efficient_roulette_selection(rounded[0])
                result = reversible_sampling_rows(weights, sub_bit_precision=mu)
                assert (
                    result
                    == source.preprocess_probabilities_for_reversible_sampling(
                        weights, sub_bit_precision=mu
                    )
                )
                assert alias_numerators(weights, mu=mu) == (
                    tuple(result[0]),
                    tuple(result[1]),
                )


def test_raw_integer_donors_preserve_errors_input_copy_and_negative_values() -> None:
    from qualtran.linalg.lcu_util import (
        _preprocess_for_efficient_roulette_selection as source,
    )

    for count in range(5):
        for values in product(range(-2, 4), repeat=count):
            weights = list(values)
            assert outcome(integer_alias_rows, weights) == outcome(source, weights)
            assert weights == list(values)


@pytest.mark.parametrize(
    "values",
    [
        [],
        [0, 0],
        [-1, 2],
        [1, -1],
        [float("nan"), 1],
        [float("inf"), 1],
        [np.float32(0), np.float32(0)],
        [2**80 + 1, 2**80 - 1],
    ],
)
@pytest.mark.parametrize("mu", [-1, 0, 1, 3])
def test_discretization_boundary_values_and_warnings(values: Any, mu: int) -> None:
    from qualtran.linalg.lcu_util import _discretize_probability_distribution as source

    assert outcome(discretize_probabilities, values, mu) == outcome(source, values, mu)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"epsilon": 0.1, "sub_bit_precision": 1},
        {"epsilon": 0},
        {"epsilon": -1},
        {"epsilon": float("nan")},
        {"epsilon": float("inf")},
        {"sub_bit_precision": 0},
        {"sub_bit_precision": -2},
        {"sub_bit_precision": True},
        {"sub_bit_precision": 1.5},
        {"epsilon": 0.125},
        {"sub_bit_precision": 2},
    ],
)
@pytest.mark.parametrize("weights", [[], [0, 0], [1, 2], [-1, 3]])
def test_preprocessor_optional_precision_failure_order(
    weights: list[float], kwargs: dict[str, Any]
) -> None:
    from qualtran.linalg.lcu_util import (
        preprocess_probabilities_for_reversible_sampling as source,
    )

    assert outcome(reversible_sampling_rows, weights, **kwargs) == outcome(
        source, weights, **kwargs
    )


@pytest.mark.parametrize(
    "rows",
    [
        0,
        -1,
        1,
        3,
        np.int64(4),
        sympy.Integer(4),
        sympy.Symbol("L", positive=True, integer=True),
    ],
)
@pytest.mark.parametrize(
    "precision",
    [
        0,
        -0.1,
        0.125,
        1,
        float("inf"),
        float("nan"),
        sympy.Rational(1, 4),
        sympy.Symbol("e", positive=True),
    ],
)
def test_concrete_and_symbolic_width_without_native_factory_validation(
    rows: Any, precision: Any
) -> None:
    from qualtran.linalg.lcu_util import sub_bit_prec_from_epsilon as source

    assert outcome(alias_precision_bits, rows, precision) == outcome(
        source, rows, precision
    )


def test_partial_sum_and_difference_generators_preserve_lazy_consumption() -> None:
    from qualtran.linalg import lcu_util as source

    events: list[int] = []
    for native, original in (
        (_partial_sums, source._partial_sums),
        (_differences, source._differences),
    ):

        def values() -> Iterator[int]:
            for value in (2**80 + 1, 7, -2):
                events.append(value)
                yield value

        traces = []
        for function in (original, native):
            events.clear()
            iterator = function(values())
            assert events == []
            trace = []
            for value in iterator:
                trace.append((type(value), value, tuple(events)))
            traces.append(trace)
        assert traces[0] == traces[1]
    assert tuple(_differences([])) == ()
    assert tuple(_partial_sums([])) == (0,)


def test_raw_stage_replacement_preserves_preprocessor_call_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qualtran.linalg import lcu_util as source

    from flagquantum.algorithms.primitives import alias_discretization as native

    traces = []
    calls: list[tuple[object, ...]] = []
    for module, names in (
        (
            source,
            (
                "sub_bit_prec_from_epsilon",
                "_discretize_probability_distribution",
                "_preprocess_for_efficient_roulette_selection",
            ),
        ),
        (
            native,
            ("alias_precision_bits", "discretize_probabilities", "integer_alias_rows"),
        ),
    ):
        calls.clear()

        def width(rows: int, precision: float) -> int:
            calls.append(("width", rows, precision))
            return 2

        def rounding(weights: list[float], bits: int) -> tuple[list[int], int]:
            calls.append(("round", tuple(weights), bits))
            return [1, 7], 8

        def donors(weights: list[int]) -> tuple[list[int], list[int]]:
            calls.append(("donors", tuple(weights)))
            return [1, 1], [1, 0]

        with monkeypatch.context() as context:
            for name, function in zip(names, (width, rounding, donors), strict=True):
                context.setattr(module, name, function)
            consumer = (
                source.preprocess_probabilities_for_reversible_sampling
                if module is source
                else native.reversible_sampling_rows
            )
            assert consumer([1, 7], epsilon=0.1) == ([1, 1], [1, 0], 2)
        traces.append(calls.copy())
    assert traces[0] == traces[1]


def test_all_original_test_cases_with_only_native_helper_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qualtran.linalg import lcu_util_test as original

    for name, function in (
        ("_discretize_probability_distribution", discretize_probabilities),
        ("_preprocess_for_efficient_roulette_selection", integer_alias_rows),
        ("preprocess_probabilities_for_reversible_sampling", reversible_sampling_rows),
        ("sub_bit_prec_from_epsilon", alias_precision_bits),
    ):
        monkeypatch.setattr(original, name, function)
    state = random.getstate()
    try:
        suite = unittest.defaultTestLoader.loadTestsFromModule(original)
        result = unittest.TestResult()
        suite.run(result)
        assert result.testsRun == 10
        assert not result.errors and not result.failures, (
            result.errors,
            result.failures,
        )
        original.test_raises_on_mu_zero()
    finally:
        random.setstate(state)


@pytest.mark.parametrize("weights", [(1, 3), (3, 1), (1, 1)])
def test_raw_rows_keep_coherent_append_branches_and_resource_models(
    weights: tuple[int, int],
) -> None:
    from qualtran.bloqs.state_preparation.state_preparation_alias_sampling import (
        StatePreparationAliasSampling,
    )
    from qualtran.linalg.lcu_util import (
        preprocess_probabilities_for_reversible_sampling as source,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    native = reversible_sampling_rows(weights, sub_bit_precision=1)
    reference = source(weights, sub_bit_precision=1)
    assert native == reference
    table = AliasSamplingTable(
        2, tuple(native[0]), tuple(native[1]), native[2], sum(weights)
    )
    # Factory uses absolute coefficient precision, raw helper normalized precision.
    assert table == alias_sampling_table(weights, precision=sum(weights) / 4)
    bound, sparse = layout(table)
    desired = expected_state(table, bound, sparse)
    initial = torch.zeros_like(desired)
    initial[0] = 1
    size = initial.numel().bit_length() - 1
    stream = alias_preparation_ir(table, *bound)
    other = AliasSamplingTable(
        2, tuple(reference[0]), tuple(reference[1]), reference[2], sum(weights)
    )
    assert stream == alias_preparation_ir(other, *bound)
    for inverse, state, expected in (
        (False, initial, desired),
        (True, desired, initial),
    ):
        gates = alias_preparation_ir(table, *bound, adjoint=inverse)
        circuit = DynamicCircuit(size, inputs=state, dtype=torch.complex128).append(
            FieldOperation(size, gates)
        )
        assert tuple(circuit.to_ir().instructions) == gates
        events = sum(g.name == "measure" for g in gates)
        for branch in product((0, 1), repeat=events):
            actual = forced_branch(state, gates, branch)
            torch.testing.assert_close(
                actual, expected / 2 ** (events / 2), rtol=0, atol=3e-12
            )
            for wire in bound[5]:
                mask = (torch.arange(actual.numel()) >> (size - wire - 1)) & 1 == 1
                torch.testing.assert_close(
                    actual[mask], torch.zeros_like(actual[mask]), rtol=0, atol=3e-12
                )
    estimate = estimate_component_resources(source_dense_alias_preparation_cost(table))
    model = StatePreparationAliasSampling.from_probabilities(
        weights, precision=sum(weights) / 4
    )
    assert {k: v for k, v in asdict(estimate.gates).items() if v} == get_cost_value(
        model, QECGatesCost()
    ).asdict()
    assert estimate.peak_logical_qubits == get_cost_value(model, QubitCount())
    counts = Counter(g.name for g in stream)
    assert counts["measure"] == estimate.gates.measurement

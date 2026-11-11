"""Source cycle order, lazy failure and unchanged register-emission consumers."""

import builtins
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import asdict
from itertools import combinations, permutations, product
from types import ModuleType
from typing import Any

import networkx as nx
import pytest
import torch

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum.circuit import Circuit
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from flagquantum.compiler.registers.sorting.cycle_decomposition import (
    partial_permutation_cycles,
    permutation_cycles,
)
from flagquantum.compiler.registers.sorting.permutation import (
    permutation_ir,
    source_partial_permutation_cycles,
)
from flagquantum.compiler.registers.sorting.permutation_resource_model import (
    source_permutation_cost,
)
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_register_constant_addition import reference_arithmetic_gates
from tests.unit.test_register_subtraction import assert_source

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("size", range(7))
def test_every_dense_permutation_through_six_preserves_source_order(size: int) -> None:
    from qualtran.linalg.permutation import decompose_permutation_into_cycles as source

    for values in permutations(range(size)):
        actual = tuple(permutation_cycles(values))
        expected = tuple(source(values))
        assert actual == expected  # No sorting or cycle-rotation normalization.
        reconstructed = list(range(size))
        for cycle in actual:
            for i, node in enumerate(cycle):
                reconstructed[node] = cycle[(i + 1) % len(cycle)]
        assert tuple(reconstructed) == values


def test_all_four_node_partial_maps_preserve_source_prefix_behavior() -> None:
    from qualtran.linalg.permutation import (
        decompose_permutation_map_into_cycles as source,
    )

    for count in range(5):
        for keys in combinations(range(4), count):
            for ordered in (keys, keys[::-1]):
                for values in product(range(4), repeat=count):
                    mapping = dict(zip(ordered, values, strict=True))
                    assert tuple(partial_permutation_cycles(mapping)) == tuple(
                        source(mapping)
                    )


@pytest.mark.parametrize(
    "values", [(1, 1, 0), (8, 0), (True, False), (None,), ([1],), ()]
)
def test_dense_nonpermutation_values_and_native_errors_match_source(
    values: Any,
) -> None:
    from qualtran.linalg.permutation import decompose_permutation_into_cycles as source

    outcomes = []
    for evaluator in (source, permutation_cycles):
        value: object
        try:
            value = tuple(evaluator(values))
        except (TypeError, ValueError, nx.NetworkXError) as error:
            value = type(error), str(error)
        outcomes.append(value)
    assert outcomes[0] == outcomes[1]


def test_dense_import_and_input_consumption_are_delayed_until_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qualtran.linalg.permutation import decompose_permutation_into_cycles as source

    events: list[str] = []
    saved_import: Callable[..., ModuleType] = builtins.__import__

    def guarded(name: str, *args: Any, **kwargs: Any) -> ModuleType:
        if name == "networkx":
            events.append("import")
            raise ModuleNotFoundError("test missing NetworkX")
        return saved_import(name, *args, **kwargs)

    class Values(list[int]):
        def __iter__(self) -> Iterator[int]:
            events.append("iterate")
            yield 1
            yield 0

    monkeypatch.setattr(builtins, "__import__", guarded)
    for evaluator in (source, permutation_cycles):
        events.clear()
        iterator = evaluator(Values())
        assert events == []
        with pytest.raises(ModuleNotFoundError, match="test missing NetworkX"):
            next(iterator)
        assert events == ["import"]
    # Partial traversal has no graph dependency.
    assert tuple(partial_permutation_cycles({0: 1})) == ((0, 1),)
    assert events == ["import"]


def test_dense_graph_provider_replacement_retains_filter_and_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qualtran.linalg.permutation import decompose_permutation_into_cycles as source

    calls: list[tuple[tuple[int, int], ...]] = []

    def recorded(graph: nx.DiGraph) -> Iterator[list[int]]:
        calls.append(tuple(graph.edges))
        yield [3]
        yield [4, 2, 1]
        yield [7, 6]

    monkeypatch.setattr(nx, "simple_cycles", recorded)
    for evaluator in (source, permutation_cycles):
        assert tuple(evaluator([1, 2, 0])) == ((4, 2, 1), (7, 6))
    assert calls[0] == calls[1] == ((0, 1), (1, 2), (2, 0))


def test_partial_mapping_mutation_failure_occurs_after_first_yield() -> None:
    from qualtran.linalg.permutation import (
        decompose_permutation_map_into_cycles as source,
    )

    for evaluator in (source, partial_permutation_cycles):
        mapping = {0: 1, 1: 0, 2: 3}
        iterator = evaluator(mapping)
        assert next(iterator) == (0, 1)
        mapping[4] = 5
        with pytest.raises(RuntimeError, match="dictionary changed size"):
            next(iterator)


def test_inherited_injective_map_prefix_loss_is_not_repaired() -> None:
    from qualtran.linalg.permutation import (
        decompose_permutation_map_into_cycles as source,
    )

    mapping = {0: 1, 2: 0}
    assert (
        tuple(source(mapping))
        == tuple(partial_permutation_cycles(mapping))
        == ((0, 1),)
    )
    assert source_partial_permutation_cycles(3, mapping) == ((0, 1),)
    # The declared 2 -> 0 edge is lost, so this is not a valid completion.
    realized = {0: 1, 1: 0, 2: 2}
    assert realized[2] != mapping[2]
    assert tuple(
        partial_permutation_cycles(dict(reversed(tuple(mapping.items()))))
    ) == (
        (2, 0, 1),
    )


@pytest.mark.parametrize("partial", [False, True])
def test_original_permutation_tests_with_native_bindings(
    monkeypatch: pytest.MonkeyPatch, partial: bool
) -> None:
    from qualtran.linalg import permutation_test

    monkeypatch.setattr(
        permutation_test, "decompose_permutation_into_cycles", permutation_cycles
    )
    monkeypatch.setattr(
        permutation_test,
        "decompose_permutation_map_into_cycles",
        partial_permutation_cycles,
    )
    consumer: Callable[[], None] = (
        permutation_test.test_decompose_sparse_prefix_permutation_into_cycles
        if partial
        else permutation_test.test_decompose_permutation_into_cycles
    )
    consumer()


@pytest.mark.parametrize("kind", ["three_cycle", "two_cycles", "partial"])
def test_discovered_cycles_emit_original_gates_all_branches_and_costs(
    kind: str,
) -> None:
    from qualtran.bloqs.arithmetic.permutation import Permutation
    from qualtran.linalg.permutation import (
        decompose_permutation_into_cycles as source_dense,
    )
    from qualtran.linalg.permutation import (
        decompose_permutation_map_into_cycles as source_partial,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    dense = [1, 2, 0, 3] if kind == "three_cycle" else [1, 0, 3, 2]
    cycles = (
        tuple(partial_permutation_cycles({0: 1, 1: 2}))
        if kind == "partial"
        else tuple(permutation_cycles(dense))
    )
    expected_cycles = (
        tuple(source_partial({0: 1, 1: 2}))
        if kind == "partial"
        else tuple(source_dense(dense))
    )
    assert cycles == expected_cycles
    gates = permutation_ir((0, 1), cycles, 2, (3,), (40,), size=4)
    source = Permutation(4, expected_cycles, bitsize=2)
    allocation_plan = tuple(
        wire
        for cycle in cycles
        for i in range(len(cycle) + 1)
        for wire in (3, *((2,) if i == 0 else ()))
    )
    expected_gates = reference_arithmetic_gates(
        source, {"x": (0, 1)}, allocation_plan, {3: 40}, reuse_allocations=False
    )
    assert_source(gates, expected_gates)
    circuit = Circuit(5, dtype=torch.complex128).append(FieldOperation(5, gates))
    assert circuit.to_ir().instructions == gates

    indices = torch.arange(8, dtype=torch.int64)
    data, spectator = indices >> 1, indices & 1
    amplitudes = torch.complex(
        indices.to(torch.float64) + 1, 3 * indices.to(torch.float64) - 5
    )
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(32, dtype=torch.complex128)
    initial[(data << 3) | spectator] = amplitudes
    mapped = data.clone()
    for cycle in cycles:
        before = mapped.clone()
        for i, node in enumerate(cycle):
            mapped[before == node] = cycle[(i + 1) % len(cycle)]
    expected = torch.zeros_like(initial)
    expected[(mapped << 3) | spectator] = amplitudes
    events = sum(gate.name == "measure" for gate in gates)
    inverse = permutation_ir(
        (0, 1), tuple(c[::-1] for c in reversed(cycles)), 2, (3,), (40,), size=4
    )
    for outcomes in product((0, 1), repeat=events):
        actual = forced_branch(initial, gates, outcomes)
        torch.testing.assert_close(
            actual, expected / 2 ** (events / 2), rtol=0, atol=1e-12
        )
        restored = forced_branch(expected, inverse, outcomes)
        torch.testing.assert_close(
            restored, initial / 2 ** (events / 2), rtol=0, atol=1e-12
        )
        # Explicitly inspect both work bits for each unnormalized branch.
        work = ((torch.arange(32) >> 1) & 3) != 0
        torch.testing.assert_close(
            actual[work], torch.zeros_like(actual[work]), rtol=0, atol=1e-12
        )
        torch.testing.assert_close(
            restored[work], torch.zeros_like(restored[work]), rtol=0, atol=1e-12
        )

    estimate = estimate_component_resources(source_permutation_cost(4, 2, cycles))
    model = get_cost_value(source, QECGatesCost())
    assert {k: v for k, v in asdict(estimate.gates).items() if v} == model.asdict()
    assert estimate.peak_logical_qubits == get_cost_value(source, QubitCount())
    counts = Counter(gate.name for gate in gates)
    assert counts["t"] + counts["tdg"] == model.total_t_count() == 4 * events
    assert counts["measure"] == model.measurement == events

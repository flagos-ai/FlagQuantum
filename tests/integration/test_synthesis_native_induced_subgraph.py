"""Preserve the complete induced-subgraph graph square and all eight layers."""

from pathlib import Path
from typing import Literal

import pytest

from examples.synthesis.applications.optimization.max_induced_k_color_subgraph.comparisons.circuit import (
    BINDINGS,
    cost_factors,
    graph_cost,
    reference_circuit,
)
from examples.synthesis.applications.optimization.max_induced_k_color_subgraph.main import (
    main,
)
from examples.synthesis.compare_states import compare_states
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate

pytestmark = pytest.mark.integration
SOURCE = (
    Path(__file__).resolve().parents[1]
    / "fixtures/classiq_qmod/corpus/original/applications/optimization/max_induced_k_color_subgraph/max_induced_k_color_subgraph.qmod"
)


def test_independent_factor_tables_match_classical_graph_for_every_input() -> None:
    factors = cost_factors()
    assert max(len(bits) for bits, _ in factors) == 4
    for basis in range(4096):
        bits = tuple((basis >> i) & 1 for i in range(12))
        factored = sum(
            table[sum(bits[logical] << index for index, logical in enumerate(support))]
            for support, table in factors
        )
        assert factored == graph_cost(bits)
    assert graph_cost((0,) * 12) == 0.0
    assert graph_cost((1,) * 12) == -1936.0


@pytest.mark.parametrize("dataset", [0, 1])
@pytest.mark.parametrize("objective", ["width", "depth"])
def test_full_original_and_native_match_independent_raw_states(
    dataset: int,
    objective: Literal["width", "depth"],
) -> None:
    bindings = {"params": BINDINGS[dataset]}
    original = synth.synthesize(
        SOURCE, classical_inputs=bindings, objective=objective, max_workspace=0
    )
    native = synth.synthesize(
        main, classical_inputs=bindings, objective=objective, max_workspace=0
    )
    reference = reference_circuit(BINDINGS[dataset])
    compare_states(native, reference, ("a", "b"))
    compare_states(original, reference, ("v.x[0]", "v.x[1]"))
    for program in (native, original):
        resource = program.report["resources"]
        assert isinstance(resource, ResourceEstimate)
        assert resource.n_qubits == 12
        assert program.report["workspace_qubits"] == 0


def test_actual_resource_budgets_refuse_full_program() -> None:
    bindings = {"params": BINDINGS[0]}
    for width in (1, 11):
        with pytest.raises(ValueError):
            synth.synthesize(main, classical_inputs=bindings, max_qubits=width)
    with pytest.raises(ValueError):
        synth.synthesize(main, classical_inputs=bindings, max_operations=10)
    with pytest.raises(ValueError):
        synth.synthesize(main, classical_inputs=bindings, memory_limit_bytes=1)

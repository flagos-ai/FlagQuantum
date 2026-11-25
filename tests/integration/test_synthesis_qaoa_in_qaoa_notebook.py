"""Check original-size hierarchical QAOA against independent Circuit states."""

import itertools
import json
import random
from pathlib import Path

import networkx as nx
import pytest
import torch

import flagquantum as fq
from examples.synthesis.applications.optimization.qaoa_in_qaoa.overview import (
    model,
    quantum,
    reference,
    workflow,
)
from flagquantum import synth

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def observed() -> workflow.Workflow:
    before = random.getstate()
    tensor_rng_before = torch.random.get_rng_state()
    np = pytest.importorskip("numpy")
    numpy_before = np.random.get_state()
    result = workflow.run_workflow()
    assert random.getstate() == before
    assert torch.equal(torch.random.get_rng_state(), tensor_rng_before)
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert (numpy_before[1] == numpy_after[1]).all()
    assert numpy_before[2:] == numpy_after[2:]
    return result


def test_complete_fresh_hierarchy(observed: workflow.Workflow) -> None:
    assert (len(observed.graph), observed.graph.number_of_edges()) == (40, 158)
    assert [len(group) for group in observed.initial_groups] == [
        6,
        6,
        2,
        4,
        4,
        3,
        4,
        4,
        4,
        3,
    ]
    assert observed.contraction_sizes == (10, 2)
    assert len(observed.runs) == 13
    assert observed.runs[0].nodes == (36, 8, 17, 23, 24, 30)
    assert len(observed.assignment) == 40
    assert observed.cut == sum(
        observed.assignment[a] != observed.assignment[b]
        for a, b in observed.graph.edges
    )
    assert (observed.cut, observed.greedy_cut, observed.random_cut) == (107, 111, 78)
    assert any(weight < 0 for run in observed.runs for _, _, weight in run.edges)
    for run in observed.runs:
        assert len(run.trace) == 20
        assert run.budget_reached and not run.converged
        assert run.parameters == min(run.trace, key=lambda row: row.cost).parameters
        assert sum(row.count for row in run.observations) == 1000
        assert run.resources is not None
        assert run.resources.n_qubits == len(run.nodes) <= 6
        assert run.assignment == tuple(
            (run.observations[0].word >> bit) & 1 for bit in range(len(run.nodes))
        )
        for row in run.observations:
            bits = tuple(int(v) for v in f"{row.word:0{len(run.nodes)}b}"[::-1])
            assert row.cut == sum(w for a, b, w in run.edges if bits[a] != bits[b])
        assert run.observations[0] == max(
            run.observations, key=lambda row: (row.cut, row.count)
        )


@pytest.mark.parametrize("run_index", range(13))
@pytest.mark.parametrize(
    "angles", [(0.17, -0.28, 0.39, 0.61), (-0.45, 0.23, -0.71, 0.52)]
)
def test_absolute_full_state_for_every_observed_subproblem(
    observed: workflow.Workflow, run_index: int, angles: tuple[float, ...]
) -> None:
    run = observed.runs[run_index]
    count = len(run.nodes)
    native = quantum.build_program(count, run.edges, angles)
    circuit = reference.reference_circuit(count, run.edges, angles)
    expected = fq.run(circuit).statevector()
    assert native.report["workspace_qubits"] == 0
    torch.testing.assert_close(
        fq.run(native).statevector(), expected, rtol=0, atol=1e-12
    )


def test_signed_contraction_identity_on_every_flip() -> None:
    graph = nx.Graph()
    graph.add_weighted_edges_from([(9, 3, 1), (9, 4, 3), (3, 7, -2), (4, 7, 4)])
    groups = [{9, 3}, {4, 7}]
    initial = {9: 0, 3: 1, 4: 1, 7: 1}
    compressed = model.contract(graph, groups, initial)
    assert compressed[0][1]["weight"] == -5
    assert model.local_edges(graph) == (
        (0, 1, 1.0),
        (0, 2, 3.0),
        (1, 3, -2.0),
        (2, 3, 4.0),
    )
    for flips in itertools.product((0, 1), repeat=2):
        full = {
            node: initial[node] ^ flips[i]
            for i, group in enumerate(groups)
            for node in group
        }
        assert model.cut_value(graph, full) == model.cut_value(
            graph, initial
        ) + model.cut_value(compressed, dict(enumerate(flips)))


def test_edgeless_community_does_not_synthesize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*args: object) -> None:
        pytest.fail("edgeless path must not synthesize")

    monkeypatch.setattr(workflow, "build_program", refuse)
    run = workflow.solve_community(nx.empty_graph(3), random.Random(246))
    assert len(run.assignment) == 3 and run.resources is None
    assert not run.trace and not run.observations


@pytest.mark.parametrize(
    "width,edges",
    [
        (0, ((0, 1, 1.0),)),
        (7, ((0, 1, 1.0),)),
        (2, ((0, 2, 1.0),)),
        (2, ((0, 1, 0.0),)),
        (2, ((0, 1, 1.0), (0, 1, -1.0))),
    ],
)
def test_invalid_community(
    width: int, edges: tuple[tuple[int, int, float], ...]
) -> None:
    with pytest.raises(ValueError):
        quantum.build_program(width, edges, (0.1, 0.2, 0.3, 0.4))


@pytest.mark.parametrize("budget", [{"max_qubits": 1}, {"max_operations": 1}])
def test_synthesis_budget_rejection(budget: dict[str, int]) -> None:
    with pytest.raises(
        ValueError, match=rf"synthesis\.constraints\.{next(iter(budget))}"
    ):
        synth.synthesize(
            quantum.community_program,
            classical_inputs={
                "node_count": 2,
                "first": (0,),
                "second": (1,),
                "weights": (-1.0,),
                "params": (0.1, 0.2, 0.3, 0.4),
            },
            max_qubits=budget.get("max_qubits"),
            max_operations=budget.get("max_operations"),
        )


def test_notebook_preserves_both_source_stages() -> None:
    root = Path(quantum.__file__).parent
    book = json.loads((root / "walkthrough.ipynb").read_text())
    proof = json.loads((root / "provenance.json").read_text())
    cells = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    assert len(cells) == 2
    assert all(
        cell["execution_count"] is None and not cell["outputs"] for cell in cells
    )
    assert [row["index"] for row in proof["reviewed_code_cells"]] == [1, 2]

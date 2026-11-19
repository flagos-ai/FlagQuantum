"""Accept graph generation, entangled walk readout and honest archived figures."""

import json
from collections import Counter
from pathlib import Path
from typing import cast

import networkx as nx
import pytest
import torch

import flagquantum as fq
from examples.synthesis.tutorials.basics.quantumwalk_complex_network.comparisons.circuit import (
    reference_circuit,
)
from examples.synthesis.tutorials.basics.quantumwalk_complex_network.main import (
    build_program,
)
from examples.synthesis.tutorials.basics.quantumwalk_complex_network.overview import (
    main as walk,
)
from tests.integration.synthesis_notebook_support import execute_code_cells

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = (
    ROOT / "examples/synthesis/tutorials/basics/quantumwalk_complex_network/overview"
)


def test_original_graph_and_neighbor_probabilities() -> None:
    graph = walk.make_graph()
    assert sorted(graph.edges) == [(0, 1), (0, 2), (0, 3), (1, 2)]
    torch.testing.assert_close(
        walk.neighbor_probabilities(graph),
        torch.tensor(
            (
                (0, 1 / 3, 1 / 3, 1 / 3),
                (0.5, 0, 0.5, 0),
                (0.5, 0.5, 0, 0),
                (1, 0, 0, 0),
            ),
            dtype=torch.float64,
        ),
        atol=0,
        rtol=0,
    )
    eight = walk.make_graph(8)
    assert len(eight) == 8 and len(eight.edges) == 16
    torch.testing.assert_close(
        walk.neighbor_probabilities(eight).sum(dim=1),
        torch.ones(8, dtype=torch.float64),
        atol=1e-15,
        rtol=0,
    )


@pytest.mark.parametrize("steps", [0, 1, 2, 3])
def test_full_position_coin_state_phase_work_zero_and_independent_marginal(
    steps: int,
) -> None:
    program = build_program(steps)
    actual = fq.run(program)
    state = actual.statevector().reshape(16, -1)
    reference = fq.run(reference_circuit(steps)).statevector().flatten()
    torch.testing.assert_close(state[:, 0], reference, atol=1e-12, rtol=0)
    assert float(state[:, 1:].abs().square().sum()) < 1e-22
    assert tuple(program.registers) == ("x",)
    assert len(program.registers["x"]) == 2
    assert program.report["workspace_qubits"] == 1
    # The other two retained bits belong to the dropped coin, not zero work.
    assert state.shape == (16, 2)
    density = reference.reshape(4, 4) @ reference.reshape(4, 4).conj().T
    assert float(torch.trace(density @ density).real) < 0.99
    probabilities = dict(program.probabilities(actual, registers=("x",)))
    for node in range(4):
        assert probabilities.get((node,), 0) == pytest.approx(
            float(density[node, node].real), abs=1e-12
        )
    if steps:
        torch.testing.assert_close(
            density.diag().real,
            torch.tensor(
                walk.archived_results(4)["ideal"][steps - 1], dtype=torch.float64
            ),
            atol=1e-8,
            rtol=0,
        )


def test_actual_three_step_execution_sampling_and_archived_tables() -> None:
    program, counts = walk.run_workflow()
    assert sum(counts.values()) == 1000 and set(counts) == set(range(4))
    assert counts == walk.run_workflow()[1]
    probabilities = dict(program.probabilities(fq.run(program), registers=("x",)))
    assert all(
        abs(counts[node] / 1000 - probabilities[(node,)]) < 0.06 for node in range(4)
    )
    for nodes in (4, 8):
        data = walk.archived_results(nodes)
        for field in ("hardware_agnostic_counts", "hardware_aware_counts"):
            rows = data[field]
            assert len(rows) == 4
            assert all(len(row) == nodes and sum(row) == 10000 for row in rows)
            assert all(
                type(value) is int and value >= 0 for row in rows for value in row
            )
        assert len(data["ideal"]) == 4
        assert all(
            len(row) == nodes and abs(sum(row) - 1) < 1e-7 for row in data["ideal"]
        )
    assert walk.archived_results(8)["hardware_agnostic_counts"][0] == [
        1330,
        1238,
        1391,
        1302,
        1283,
        1144,
        1158,
        1154,
    ]
    assert walk.archived_results(8)["hardware_aware_counts"][0] == [
        1350,
        1352,
        1275,
        1342,
        1205,
        1214,
        1104,
        1158,
    ]


@pytest.mark.parametrize("nodes", [0, 3, 5, True])
def test_unsupported_graph_and_archive_size_refuse(nodes: int) -> None:
    with pytest.raises(ValueError):
        walk.make_graph(nodes)
    with pytest.raises(ValueError):
        walk.archived_results(nodes)


def test_nonconsecutive_and_isolated_nodes_refuse() -> None:
    graph = nx.empty_graph(4)
    with pytest.raises(ValueError, match="nonempty"):
        walk.neighbor_probabilities(graph)
    graph = nx.path_graph([1, 2, 3, 4])
    with pytest.raises(ValueError, match="consecutive"):
        walk.neighbor_probabilities(graph)


@pytest.mark.parametrize("preexisting_figure", (False, True))
def test_plots_notebook_and_source_provenance(
    preexisting_figure: bool, request: pytest.FixtureRequest
) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.container import BarContainer

    original_numbers = set(plt.get_fignums())

    def close_created_figures() -> None:
        for number in set(plt.get_fignums()) - original_numbers:
            plt.close(number)

    request.addfinalizer(close_created_figures)
    if preexisting_figure:
        plt.subplots()
    before = set(plt.get_fignums())
    namespace, executed = execute_code_cells(DIRECTORY / "walkthrough.ipynb")
    assert executed == 4
    counts = cast(Counter[int], namespace["counts"])
    assert sum(counts.values()) == 1000
    figures = [
        plt.figure(number) for number in plt.get_fignums() if number not in before
    ]
    assert [len(figure.axes) for figure in figures] == [1, 1, 5]
    assert list(figures[0].get_size_inches()) == [3, 3]
    assert list(figures[2].get_size_inches()) == [18, 4]
    source = walk.archived_results(8)
    for index, axes in enumerate(figures[2].axes[1:]):
        assert axes.get_title() == f"t={index+1}"
        bars = [item for item in axes.containers if isinstance(item, BarContainer)]
        assert len(bars) == 3
        expected = (
            source["ideal"][index],
            [x / 10000 for x in source["hardware_agnostic_counts"][index]],
            [x / 10000 for x in source["hardware_aware_counts"][index]],
        )
        assert [item.get_label() for item in bars] == [
            "Archived ideal",
            "Torino: HW agnostic",
            "Torino: HW aware",
        ]
        for series, values in zip(bars, expected, strict=True):
            assert [bar.get_height() for bar in series.patches] == pytest.approx(values)
    assert before <= set(plt.get_fignums())
    notebook = json.loads((DIRECTORY / "walkthrough.ipynb").read_text())
    assert all(
        not cell["outputs"] and cell["execution_count"] is None
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    proof = json.loads((DIRECTORY / "provenance.json").read_text())
    assert [row["index"] for row in proof["reviewed_code_cells"]] == [
        3,
        5,
        7,
        9,
        11,
        13,
        15,
        17,
        20,
        22,
        25,
        28,
        30,
    ]
    assert proof["acceptance_test"] == str(Path(__file__).relative_to(ROOT))


@pytest.mark.parametrize(
    "counts", [Counter(), Counter({4: 1}), Counter({0: -1}), Counter({0: 0})]
)
def test_invalid_sample_plot_refuses(counts: Counter[int]) -> None:
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError):
        walk.plot_samples(counts)

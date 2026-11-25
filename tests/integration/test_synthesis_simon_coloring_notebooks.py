"""Accept complete Simon recoveries and the sampled graph-coloring workflow."""

import json
from collections import Counter
from collections.abc import Callable
from itertools import product
from pathlib import Path
from typing import cast

import pytest
import torch

import flagquantum as fq
from examples.synthesis.algorithms.foundational.simon.overview import main as simon
from examples.synthesis.compare_states import compare_states
from examples.synthesis.tutorials.basics.grover_graph_coloring.comparisons.circuit import (
    marked,
    reference_circuit,
)
from examples.synthesis.tutorials.basics.grover_graph_coloring.overview import (
    main as coloring,
)
from flagquantum import synth
from tests.integration.synthesis_notebook_support import execute_code_cells

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = (
    (
        "algorithms/foundational/simon",
        3,
        (6, 8, 9, 11, 13, 19, 21, 23, 25, 27, 28, 29, 33, 35, 37, 39, 40, 41),
    ),
    ("tutorials/basics/grover_graph_coloring", 4, (2, 4, 7, 9, 11, 12, 13, 14, 16)),
)


def representative(value: int) -> int:
    other = value ^ 6
    return value if value < other else other


def bipartite(value: int) -> int:
    pivot = (value >> 2) & 1
    return (value & 3) | sum(
        (((value >> bit) & 1) ^ pivot) << bit for bit in range(3, 6)
    )


def simon_reference(
    width: int, oracle: Callable[[int], int], interference: bool
) -> fq.Circuit:
    """Build independent per-output-bit permutations with bounded dense matrices."""
    circuit = fq.Circuit(2 * width, dtype=torch.complex128)
    for bit in range(width):
        circuit.h(bit)
    size = 1 << (width + 1)
    for bit in range(width):
        matrix = torch.zeros((size, size), dtype=torch.complex128)
        for value in range(1 << width):
            for target in (0, 1):
                before = 2 * value + target
                after = 2 * value + (target ^ ((oracle(value) >> bit) & 1))
                matrix[after, before] = 1
        circuit.any(*range(width), 2 * width - 1 - bit, matrix=matrix)
    if interference:
        for bit in range(width):
            circuit.h(bit)
    return circuit


@pytest.fixture(scope="module")
def simon_workflow() -> (
    tuple[dict[str, synth.Program], dict[str, Counter[tuple[int, ...]]], dict[str, int]]
):
    return simon.run_workflow()


def test_both_simon_mappings_and_algorithms_keep_complete_absolute_states(
    simon_workflow: tuple[
        dict[str, synth.Program], dict[str, Counter[tuple[int, ...]]], dict[str, int]
    ],
) -> None:
    programs, counts, secrets = simon_workflow
    assert secrets == {"simple_algorithm": 6, "bipartite_algorithm": 60}
    for name, width, oracle, shots in (
        ("simple_mapping", 5, representative, 1000),
        ("simple_algorithm", 5, representative, 250),
        ("bipartite_mapping", 6, bipartite, 1000),
        ("bipartite_algorithm", 6, bipartite, 300),
    ):
        program = programs[name]
        compare_states(
            program,
            simon_reference(width, oracle, name.endswith("algorithm")),
            ("x", "res"),
        )
        assert program.report["workspace_qubits"] == 0
        assert sum(counts[name].values()) == shots
        assert tuple(program.registers) == ("x", "res")
        if name.endswith("mapping"):
            assert all(res == oracle(x) for x, res in counts[name])
        else:
            secret = secrets[name]
            assert all((x & secret).bit_count() % 2 == 0 for x, _ in counts[name])


def test_gf2_recovery_for_every_nonzero_secret_through_six_bits() -> None:
    checked = 0
    for width in range(1, 7):
        for secret in range(1, 1 << width):
            hyperplane = [
                row for row in range(1 << width) if (row & secret).bit_count() % 2 == 0
            ]
            assert simon.recover_secret(hyperplane, width) == secret
            checked += 1
    assert checked == 120


@pytest.mark.parametrize("samples,width", [([], 5), ([0, 0], 6), ([1, 2], 2)])
def test_gf2_rejects_nonunique_nonzero_secret(samples: list[int], width: int) -> None:
    with pytest.raises(ValueError, match="independent"):
        simon.recover_secret(samples, width)


@pytest.mark.parametrize("samples,width", [([-1], 5), ([32], 5), ([True], 5)])
def test_gf2_rejects_invalid_decoded_samples(samples: list[int], width: int) -> None:
    with pytest.raises(ValueError, match="decoded integers"):
        simon.recover_secret(samples, width)


@pytest.mark.parametrize("width", [0, 7, True])
def test_gf2_rejects_unsupported_width(width: int) -> None:
    with pytest.raises(ValueError, match="width"):
        simon.recover_secret([0], width)


@pytest.fixture(scope="module")
def coloring_workflow() -> tuple[synth.Program, Counter[tuple[int, ...]]]:
    return coloring.run_workflow()


def test_graph_constraints_one_grover_iteration_and_both_clean_workspaces(
    coloring_workflow: tuple[synth.Program, Counter[tuple[int, ...]]],
) -> None:
    program, counts = coloring_workflow
    independent_marked = set(marked())
    actual_marked = set()
    for values in product(range(4), repeat=5):
        word = sum(value << (2 * (4 - index)) for index, value in enumerate(values))
        if coloring.valid(values):
            actual_marked.add(word)
    assert actual_marked == independent_marked and len(actual_marked) == 16
    expected = fq.run(reference_circuit(1)).statevector().reshape(1024, -1)
    state = fq.run(program).statevector().flatten()
    width = state.numel().bit_length() - 1
    assert width <= 24
    logical = tuple(
        wire for name in ("a", "b", "c", "d", "e") for wire in program.registers[name]
    )
    hidden = tuple(wire for wire in range(width) if wire not in logical)
    ordered = state.reshape((2,) * width).permute((*logical, *hidden)).reshape(1024, -1)
    assert float(expected[:, 1:].abs().square().sum()) < 1e-24
    assert float(ordered[:, 1:].abs().square().sum()) < 1e-24
    torch.testing.assert_close(ordered[:, 0], expected[:, 0], atol=1e-12, rtol=0)
    assert sum(counts.values()) == 1000 and len(counts.most_common(16)) == 16
    # Preserve the sampled ranking, including the invalid row in this fixed draw.
    assert any(not coloring.valid(values) for values, _ in counts.most_common(16))


def test_complete_notebooks_preserve_mapping_plots_and_all_sampled_graphs() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    try:
        for relative, cells, _ in WORKFLOWS:
            namespace, executed = execute_code_cells(
                ROOT / "examples/synthesis" / relative / "overview/walkthrough.ipynb"
            )
            assert executed == cells
            if relative.endswith("simon"):
                figures = cast(list[Figure], namespace["figures"])
                counts = cast(dict[str, Counter[tuple[int, ...]]], namespace["counts"])
                assert namespace["secrets"] == {
                    "simple_algorithm": 6,
                    "bipartite_algorithm": 60,
                }
                assert len(figures) == 2
                for figure, name in zip(
                    figures, ("simple_mapping", "bipartite_mapping"), strict=True
                ):
                    axes = figure.axes[0]
                    assert (axes.get_xlabel(), axes.get_ylabel()) == ("$x$", "$f(x)$")
                    actual_pairs = torch.stack(
                        (
                            torch.as_tensor(
                                axes.lines[0].get_xdata(), dtype=torch.float64
                            ),
                            torch.as_tensor(
                                axes.lines[0].get_ydata(), dtype=torch.float64
                            ),
                        ),
                        dim=1,
                    )
                    torch.testing.assert_close(
                        actual_pairs,
                        torch.tensor(list(counts[name]), dtype=torch.float64),
                        atol=0,
                        rtol=0,
                    )
            else:
                graph, grid = namespace["graph"], namespace["grid"]
                assert isinstance(graph, Figure) and isinstance(grid, Figure)
                assert len(graph.axes) == 1 and len(graph.axes[0].lines) == 13
                assert len(graph.axes[0].collections) == 9 and len(grid.axes) == 16
                ranked = cast(
                    Counter[tuple[int, ...]], namespace["counts"]
                ).most_common(16)
                for index, (axes, (values, count)) in enumerate(
                    zip(grid.axes, ranked, strict=True)
                ):
                    assert len(axes.lines) == 13 and len(axes.collections) == 9
                    assert (
                        axes.get_title()
                        == f"Row {index+1}: count={count}, valid={coloring.valid(values)}"
                    )
    finally:
        plt.close("all")


def test_graph_plot_rejects_missing_sampled_rows() -> None:
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="sixteen distinct"):
        coloring.plot_samples(Counter({(0, 0, 0, 0, 0): 1000}))


@pytest.mark.parametrize("relative,cells,source_cells", WORKFLOWS)
def test_source_provenance_and_clean_native_notebooks(
    relative: str, cells: int, source_cells: tuple[int, ...]
) -> None:
    path = ROOT / "examples/synthesis" / relative / "overview/walkthrough.ipynb"
    proof = json.loads(path.with_name("provenance.json").read_text())
    assert tuple(cell["index"] for cell in proof["reviewed_code_cells"]) == source_cells
    document = json.loads(path.read_text())
    assert sum(cell["cell_type"] == "code" for cell in document["cells"]) == cells
    assert len({cell["id"] for cell in document["cells"]}) == len(document["cells"])
    for cell in document["cells"]:
        source = "".join(cell["source"])
        assert "from classiq" not in source and "source_format" not in source
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None

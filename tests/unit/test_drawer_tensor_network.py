"""Native tensor views preserve pinned topology without touching numerical data."""

from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.algorithms.factorized_controlled_not import (
    controlled_not_chain,
    factorized_cnot_supply,
)
from examples.algorithms.tensor_network_inspection import inspection_example, main
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.core.ir.program.signatures import Signature
from flagquantum.drawer.tensor_network import (
    draw_tensor_network,
    export_tensor_network_dot,
    tensor_index_occurrences,
)
from flagquantum.simulation.tensor_network.component_tensor_nodes import (
    component_tensor_nodes_with_labels,
)
from flagquantum.simulation.tensor_network.models import TensorNetworkNode
from flagquantum.simulation.tensor_network.register_tensor_binding import (
    contract_register_tensors,
)
from flagquantum.simulation.tensor_network.register_tensor_indices import (
    register_tensor_outer_indices,
)

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("count", [1, 2, 3, 7])
@pytest.mark.parametrize("friendly", [False, True])
def test_complete_source_axis_incidence_and_friendly_outer_indices(
    count: int, friendly: bool
) -> None:
    from qualtran import BloqBuilder
    from qualtran import Signature as SourceSignature
    from qualtran.bloqs.basic_gates import CNOT
    from qualtran.simulation.tensor import cbloq_to_quimb

    graph = controlled_not_chain(count)
    like = torch.empty((), dtype=torch.complex128)
    order = tuple(i for i, _, _ in iter_component_calls(graph, "cnot_chain"))
    nodes, incoming, outgoing = component_tensor_nodes_with_labels(
        graph, order, partial(factorized_cnot_supply, like=like), like, "cnot_chain"
    )
    signature = Signature(graph.functions[0].registers or ())
    descriptors = register_tensor_outer_indices(
        signature, incoming, outgoing, friendly=friendly
    )
    bb, values = BloqBuilder.from_signature(SourceSignature.build(ctrl=1, target=1))
    for _ in range(count):
        values = bb.add_d(CNOT(), **values)
    source = bb.finalize(**values)
    original = cbloq_to_quimb(source, friendly_indices=friendly)
    source_occurrences: dict[object, list[tuple[int, int]]] = {}
    for ordinal, node in enumerate(original.tensors):
        for axis, label in enumerate(node.inds):
            source_occurrences.setdefault(label, []).append((ordinal, axis))
    actual = tensor_index_occurrences(nodes)
    assert tuple(actual.values()) == tuple(
        tuple(v) for v in source_occurrences.values()
    )
    assert Counter(len(v) for v in actual.values()) == Counter(
        len(v) for v in source_occurrences.values()
    )
    source_by_axes = {tuple(v): label for label, v in source_occurrences.items()}
    for label, descriptor in descriptors.items():
        assert descriptor == source_by_axes[actual[label]]
    assert sum(len(v) for v in actual.values()) == sum(n.tensor.ndim for n in nodes)
    assert len([v for v in actual.values() if len(v) == 1]) == 4
    before = contract_register_tensors(nodes, signature, incoming, outgoing)
    dot = export_tensor_network_dot(
        nodes, index_names={label: str(value) for label, value in descriptors.items()}
    )
    assert dot.count(" -- ") == 6 * count
    after = contract_register_tensors(nodes, signature, incoming, outgoing)
    torch.testing.assert_close(after, before, atol=0, rtol=0)
    np.testing.assert_array_equal(
        after.numpy(),
        (
            original.to_dense(("ctrl_0r", "target_0r"), ("ctrl_0l", "target_0l"))
            if friendly
            else source.tensor_contract()
        ),
    )


@pytest.mark.parametrize(
    "labels,shape,expected",
    [
        ((2, 2), (3, 3), {2: ((0, 0), (0, 1))}),
        ((4, 7), (0, 1), {4: ((0, 0),), 7: ((0, 1),)}),
        ((), (), {}),
    ],
)
def test_trace_empty_dimension_and_scalar_metadata(
    labels: tuple[int, ...],
    shape: tuple[int, ...],
    expected: dict[int, tuple[tuple[int, int], ...]],
) -> None:
    node = TensorNetworkNode(
        torch.empty(shape, device="meta"), labels, name="metadata_only"
    )
    assert tensor_index_occurrences((node,)) == expected
    dot = export_tensor_network_dot((node,))
    assert dot.count(" -- ") == len(labels)
    assert "metadata_only" in dot


def test_hyperedge_retains_all_tensor_axes_and_names_do_not_define_identity() -> None:
    nodes = tuple(
        TensorNetworkNode(torch.empty((2,), device="meta"), (5,), name=f"T{i}")
        for i in range(3)
    )
    assert tensor_index_occurrences(nodes) == {5: ((0, 0), (1, 0), (2, 0))}
    assert export_tensor_network_dot(nodes).count(" -- i0") == 3
    nodes, _ = inspection_example()
    names = dict.fromkeys(tensor_index_occurrences(nodes), 'same "name"\nnext')
    dot = export_tensor_network_dot(nodes, index_names=names)
    assert 'same \\"name\\"\\nnext' in dot
    assert dot.count("shape=diamond") == 5


@pytest.mark.parametrize(
    "nodes",
    [
        (TensorNetworkNode(torch.empty(2), ()),),
        (TensorNetworkNode(torch.empty(2, 3), (1, 1)),),
        (TensorNetworkNode(torch.empty(2), (-1,)),),
        (TensorNetworkNode(torch.empty(2), (True,)),),
    ],
)
def test_invalid_shapes_and_labels_are_rejected(
    nodes: tuple[TensorNetworkNode, ...],
) -> None:
    with pytest.raises(ValueError):
        tensor_index_occurrences(nodes)
    with pytest.raises(ValueError):
        export_tensor_network_dot(nodes)


def test_names_require_existing_edges_before_matplotlib_allocates() -> None:
    nodes, _ = inspection_example()
    with pytest.raises(ValueError, match="existing indices"):
        export_tensor_network_dot(nodes, index_names={999: "missing"})
    with pytest.raises(ValueError, match="existing indices"):
        draw_tensor_network(nodes, index_names={999: "missing"})


@pytest.mark.parametrize("mode", ["cnot", "scalar", "empty", "trace", "hyperedge"])
def test_render_metadata_only_network_and_preserve_autograd(
    mode: str, tmp_path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    nodes, names = inspection_example()
    if mode == "scalar":
        nodes, names = (
            TensorNetworkNode(torch.empty((), device="meta"), (), "scalar"),
        ), {}
    elif mode == "empty":
        nodes, names = (), {}
    elif mode == "trace":
        nodes, names = (
            TensorNetworkNode(torch.empty(2, 2, device="meta"), (3, 3)),
        ), {}
    elif mode == "hyperedge":
        nodes = tuple(
            TensorNetworkNode(torch.empty(2, device="meta"), (7,)) for _ in range(3)
        )
        names = {7: "hyperedge"}
    identities = tuple(id(node.tensor) for node in nodes)
    figure, axes = draw_tensor_network(
        nodes,
        index_names=names,
        node_colors={"COPY": "blue", "XOR": "orange"} if mode == "cnot" else None,
    )
    try:
        figure.canvas.draw()
        assert len(axes.lines) == sum(len(node.labels) for node in nodes)
        assert tuple(id(node.tensor) for node in nodes) == identities
        figure.savefig(tmp_path / f"{mode}.svg")
        assert (tmp_path / f"{mode}.svg").stat().st_size > 1000
    finally:
        plt.close(figure)


def test_view_does_not_break_complex_gradient() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    phase = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    data = torch.exp(1j * phase) * torch.ones(2, dtype=torch.complex128)
    nodes = (TensorNetworkNode(data, (4,), name="phase"),)
    tensor_index_occurrences(nodes)
    export_tensor_network_dot(nodes)
    figure, _ = draw_tensor_network(nodes)
    plt.close(figure)
    gradient = torch.autograd.grad(nodes[0].tensor.imag.sum(), phase)[0]
    torch.testing.assert_close(gradient, 2 * torch.cos(phase))


def test_example_exports_real_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main()
    main(tmp_path)
    assert "2 tensors, 4 outer indices, 1 inner index" in capsys.readouterr().out
    assert (tmp_path / "controlled-not.dot").read_text().count(" -- ") == 6
    assert (tmp_path / "controlled-not.png").stat().st_size > 1000

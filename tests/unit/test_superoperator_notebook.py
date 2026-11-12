"""Execute every frozen supertensor cell and the independent native workflow."""

import ast
import builtins
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
import torch

from examples.algorithms.superoperator_workflow import (
    ZMeasurementDilation,
    superoperator_example,
)
from flagquantum.drawer.tensor_network import tensor_index_occurrences

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]
CODE_CELLS = (2, 3, 5, 8, 9, 12, 13, 15, 17, 19, 21, 23, 25)


def execute(notebook: dict[str, Any]) -> tuple[dict[str, Any], dict[int, Any]]:
    """Execute untouched code strings and capture notebook display expressions."""
    namespace: dict[str, Any] = {"display": lambda *args: None}
    results: dict[int, Any] = {}
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        last = tree.body[-1]
        expression = None
        if isinstance(last, ast.Expr):
            tree.body.pop()
            expression = last.value
        exec(compile(tree, f"supertensor:{i}", "exec"), namespace)
        results[i] = (
            eval(
                compile(ast.Expression(expression), f"supertensor:{i}", "eval"),
                namespace,
            )
            if expression is not None
            else None
        )
    assert tuple(results) == CODE_CELLS
    return namespace, results


@pytest.fixture
def reference(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], dict[int, Any]]:
    import qualtran
    import qualtran.drawing
    import quimb.tensor as qtn

    raw = (Path(qualtran.__file__).parent / "simulation/supertensor.ipynb").read_bytes()
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "6feb17250e1ed1cdd271e65dcb8f5036808cdae3"
    )
    calls: list[object] = []
    with monkeypatch.context() as patch:
        patch.setattr(
            qualtran.drawing, "show_bloq", lambda *args, **kwargs: calls.append(args)
        )
        patch.setattr(
            qtn.TensorNetwork, "draw", lambda *args, **kwargs: calls.append(args)
        )
        result = execute(json.loads(raw))
    assert len(calls) == 6
    return result


def test_complete_notebook_asset_and_independent_execution(
    reference: tuple[dict[str, Any], dict[int, Any]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    source, original_results = reference
    notebook = json.loads(
        (ROOT / "examples/notebooks/superoperator_simulation.ipynb").read_text()
    )
    assert len(notebook["cells"]) == 28
    assert sum(c["cell_type"] == "markdown" for c in notebook["cells"]) == 15
    assert len({c["id"] for c in notebook["cells"]}) == 28
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    for cell in notebook["cells"]:
        assert (
            cell["metadata"] == {}
            and not cell.get("outputs")
            and not cell.get("attachments")
        )
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
    imported = builtins.__import__

    def native_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"qualtran", "cirq", "quimb"}:
            raise AssertionError(f"Native notebook imported reference package {name}")
        return imported(name, *args, **kwargs)

    plt.close("all")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(builtins, "__import__", native_import)
            actual, results = execute(notebook)
        for index in (2, 3, 12, 13):
            np.testing.assert_allclose(
                results[index].numpy(), original_results[index], atol=2e-14, rtol=0
            )
        for name in ("super_h", "rho_coin_flip", "rho_coherent"):
            np.testing.assert_allclose(
                actual[name].numpy(), source[name], atol=2e-14, rtol=0
            )
        assert results[8] == original_results[8] == (2, 2, 2, 2)
        assert results[9] == original_results[9] == (2, 2)
        np.testing.assert_allclose(
            actual["dilation_isometry"].numpy(),
            source["meas_cbloq"].tensor_contract(),
            atol=2e-14,
            rtol=0,
        )
        np.testing.assert_allclose(
            actual["measurement_channel"].numpy(),
            source["meas2_cbloq"].tensor_contract(superoperator=True),
            atol=2e-14,
            rtol=0,
        )
        assert len(plt.get_fignums()) == 6
        for i, number in enumerate(plt.get_fignums()):
            figure = plt.figure(number)
            figure.canvas.draw()
            figure.savefig(tmp_path / f"superoperator-{i}.png")
            assert (tmp_path / f"superoperator-{i}.png").stat().st_size > 5000
    finally:
        plt.close("all")


@pytest.mark.parametrize("kind", ["coin", "dilation", "measurement"])
def test_original_every_leaf_axis_and_trace_topology(
    kind: Literal["coin", "dilation", "measurement"],
    reference: tuple[dict[str, Any], dict[int, Any]],
) -> None:
    from qualtran.simulation.tensor import cbloq_to_quimb, cbloq_to_superquimb
    from qualtran.simulation.tensor._dense import _group_outer_inds

    source = reference[0][
        {"coin": "coin_flip", "dilation": "meas_cbloq", "measurement": "meas2_cbloq"}[
            kind
        ]
    ]
    native = superoperator_example(kind)
    tn = cbloq_to_superquimb(source)
    original = list(tn.tensors)
    nodes = native.doubled_nodes()
    assert len(nodes) == len(original)
    mapping: dict[object, int] = {}
    reverse: dict[int, object] = {}
    for tensor, node in zip(original, nodes, strict=True):
        np.testing.assert_allclose(
            node.tensor.resolve_conj().numpy(), tensor.data, atol=0, rtol=0
        )
        assert len(node.labels) == len(tensor.inds)
        for index, label in zip(tensor.inds, node.labels, strict=True):
            assert mapping.setdefault(index, label) == label
            assert reverse.setdefault(label, index) == index
    groups = _group_outer_inds(tn, source.signature, superoperator=True)
    from flagquantum.simulation.tensor_network.register_superoperator_binding import (
        register_superoperator_label_groups,
    )

    assert tuple(
        tuple(mapping[i] for i in group) for group in groups
    ) == register_superoperator_label_groups(
        native.signature, native.incoming, native.outgoing
    )
    assert sum(len(v) for v in tensor_index_occurrences(nodes).values()) == sum(
        t.ndim for t in original
    )
    if kind == "dilation":
        pure = cbloq_to_quimb(source)
        for tensor, node in zip(pure.tensors, native.nodes, strict=True):
            np.testing.assert_array_equal(node.tensor.numpy(), tensor.data)
    else:
        with pytest.raises(ValueError):
            cbloq_to_quimb(source)
        with pytest.raises(ValueError, match="discard"):
            native.pure()


def test_physical_dilation_exact_isometry_branches_clean_slots_and_append(
    reference: tuple[dict[str, Any], dict[int, Any]],
) -> None:
    from flagquantum.circuit import Circuit
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic
    from tests.unit.measurement_checks import forced_branch

    source = reference[0]["meas_cbloq"]
    # The actual source calls are Zero, H, controlled-Z, H, Cast. Cast is an
    # identity, not a sampled measurement or clearing operation.
    assert [str(b.bloq) for b, _, _ in source.iter_bloqnections()] == [
        "|0>",
        "H",
        "CZ",
        "H",
        "Cast",
    ]
    gates = ZMeasurementDilation().instructions()
    assert [(g.name, g.wires, g.params) for g in gates] == [
        ("h", (1,), {}),
        ("cz", (1, 0), {}),
        ("h", (1,), {}),
    ]
    iso = torch.as_tensor(source.tensor_contract())
    from flagquantum.core.ir import CircuitIR
    from flagquantum.simulation.unitary import get_unitary

    # Check every column of the clean-record subspace, including spectator
    # entanglement, rather than inferring isometry equality from one vector.
    unitary = get_unitary(CircuitIR(3, gates), dtype=torch.complex128)
    torch.testing.assert_close(
        unitary[:, [0, 1, 4, 5]],
        torch.kron(iso, torch.eye(2, dtype=torch.complex128)),
        atol=2e-14,
        rtol=0,
    )
    amplitudes = torch.tensor([1 + 2j, 3 - 4j, -2 + 3j, 4 + 1j], dtype=torch.complex128)
    amplitudes /= amplitudes.norm()
    initial = torch.zeros(8, dtype=torch.complex128)
    initial[[0, 1, 4, 5]] = amplitudes
    circuit = Circuit(3, inputs=initial, dtype=torch.complex128)
    circuit.append(ZMeasurementDilation())
    expected = torch.einsum("ab,bs->as", iso, amplitudes.reshape(2, 2)).reshape(-1)
    torch.testing.assert_close(circuit.state().flatten(), expected, atol=2e-14, rtol=0)
    op = ZMeasurementDilation(measured=True)
    counts = Counter(g.name for g in op.instructions())
    assert counts == {"h": 2, "cz": 1, "measure": 1, "x": 1, "reset": 1}
    assert op.instructions()[4].metadata == {"conditions": ((0, 1),)}
    for bit in (0, 1):
        branch = forced_branch(initial, op.instructions(), (bit,))
        expected_branch = torch.zeros_like(initial)
        expected_branch[:2] = amplitudes.reshape(2, 2)[bit]
        torch.testing.assert_close(branch, expected_branch, atol=2e-14, rtol=0)
        assert torch.count_nonzero(branch[2:]) == 0
        # Discard q after the coherent isometry gives precisely this record
        # sector, including its original phase and unnormalized probability.
        assert torch.allclose(
            branch.norm() ** 2, amplitudes.reshape(2, 2)[bit].abs().square().sum()
        )
    dynamic = DynamicCircuit(3, inputs=initial, dtype=torch.complex128)
    dynamic.append(op)
    result = run_dynamic(dynamic, shots=8, seed=41, strategy="trajectory")
    for state in result.final_states.reshape(-1, 8):
        assert torch.count_nonzero(state[2:]) == 0
        assert any(
            torch.allclose(state[:2], row / row.norm(), atol=2e-14, rtol=0)
            for row in amplitudes.reshape(2, 2)
        )


def test_complex_density_phase_gradient_and_source_resource_models(
    reference: tuple[dict[str, Any], dict[int, Any]],
) -> None:
    from qualtran.bloqs.basic_gates import MeasureZ
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    theta = torch.tensor(0.31, dtype=torch.float64, requires_grad=True)
    psi = torch.stack((torch.cos(theta).to(torch.complex128), 1j * torch.sin(theta)))
    rho = psi[:, None] * psi.conj()[None, :]
    channel = superoperator_example("measurement").channel()
    measured = torch.einsum("abij,ij->ab", channel, rho)
    torch.testing.assert_close(measured, torch.diag(rho.diag()), atol=2e-14, rtol=0)
    (gradient,) = torch.autograd.grad(measured[1, 1].real, (theta,))
    torch.testing.assert_close(gradient, torch.sin(2 * theta), atol=2e-14, rtol=0)
    model = get_cost_value(MeasureZ(), QECGatesCost())
    assert model.measurement == 1 and model.clifford == 0
    dilation_model = get_cost_value(reference[0]["meas_cbloq"], QECGatesCost())
    assert dilation_model.clifford == 3 and dilation_model.measurement == 0

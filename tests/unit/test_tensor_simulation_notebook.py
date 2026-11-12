"""Execute all original numerical cells and an independent native notebook.

Only the original presentation calls are intercepted: Graphviz is optional.
No reference constructor, tensor supplier, flattening or contraction is replaced.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]
CODE_CELLS = (1, 3, 5, 8, 10, 12, 14, 16, 18, 19, 21, 22, 24, 26, 28, 31, 33, 35)


def execute_cells(notebook: dict[str, Any], label: str) -> dict[str, Any]:
    """Execute cell bytes in one namespace, just as a fresh notebook kernel."""
    namespace: dict[str, Any] = {}
    executed = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            exec(
                compile("".join(cell["source"]), f"{label}:{index}", "exec"), namespace
            )
            executed.append(index)
    assert tuple(executed) == CODE_CELLS
    return namespace


def test_complete_source_and_native_workflows(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import qualtran
    import qualtran.drawing as drawing
    import qualtran.simulation.tensor as tensor
    import quimb.tensor as qtn
    from matplotlib import pyplot as plt

    raw = (Path(qualtran.__file__).parent / "simulation/tensor.ipynb").read_bytes()
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "d91373e7d8eb21c48a0b870d99d1bb2973971861"
    )
    source_notebook = json.loads(raw)
    assert len(source_notebook["cells"]) == 36
    assert sum(c["cell_type"] == "markdown" for c in source_notebook["cells"]) == 18
    assert not any(
        c.get("outputs") or c.get("attachments") for c in source_notebook["cells"]
    )
    assert source_notebook["metadata"]["kernelspec"]["name"] == "python3"
    presentations: list[str] = []

    # Test-only optional-renderer boundary. Real source numerical methods remain
    # installed and all 18 original cell strings are compiled unchanged.
    def source_score(*args: object, **kwargs: object) -> None:
        presentations.append("score")

    def source_network(*args: object, **kwargs: object) -> None:
        presentations.append("network")

    with monkeypatch.context() as original_patch:
        original_patch.setattr(drawing, "show_bloq", source_score)
        original_patch.setattr(qtn.TensorNetwork, "draw", source_network)
        source = execute_cells(source_notebook, "original-tensor")
    assert presentations == ["score", "network", "network", "score", "score"]
    assert "Expected error because we didn't flatten first" in capsys.readouterr().out

    native_notebook = json.loads(
        (ROOT / "examples/notebooks/tensor_simulation.ipynb").read_text()
    )
    assert len(native_notebook["cells"]) == 36
    assert sum(c["cell_type"] == "markdown" for c in native_notebook["cells"]) == 18
    assert not any(
        c.get("outputs") or c.get("attachments") for c in native_notebook["cells"]
    )

    def refuse_reference(*args: object, **kwargs: object) -> None:
        raise AssertionError("native notebook must not call reference numerics")

    plt.close("all")
    try:
        with monkeypatch.context() as native_patch:
            native_patch.setattr(qualtran.Bloq, "tensor_contract", refuse_reference)
            native_patch.setattr(tensor, "bloq_to_dense", refuse_reference)
            native_patch.setattr(tensor, "cbloq_to_quimb", refuse_reference)
            native = execute_cells(native_notebook, "native-tensor")
        numbers = plt.get_fignums()
        assert len(numbers) == 5
        for index, number in enumerate(numbers):
            figure = plt.figure(number)
            figure.canvas.draw()
            assert figure.axes and all(axis.has_data() for axis in figure.axes)
            path = tmp_path / f"tensor-workflow-{index}.png"
            figure.savefig(path, dpi=100)
            pixels = plt.imread(path)
            assert pixels.shape[0] >= 100 and pixels.shape[1] >= 100
            assert np.std(pixels[..., :3]) > 0.01
    finally:
        plt.close("all")

    # Full complex arrays preserve phase, including Bell's negative unitary entries.
    for native_name, expected in (
        ("cnot_matrix", source["cnot"].tensor_contract()),
        ("plus_state", source["PlusState"]().tensor_contract()),
        ("zero_effect", source["ZeroEffect"]().tensor_contract()),
        ("and_matrix", source["And"]().tensor_contract()),
        ("multi_matrix", source["bloq"].tensor_contract()),
        ("custom_result", source["MyCNOT"]().tensor_contract()),
        ("bell_state", source["BellState"]().tensor_contract()),
        ("bell_unitary", source["bell_circuit"].tensor_contract()),
        ("bell_from_zero", source["bell_state_cbloq"].tensor_contract()),
    ):
        np.testing.assert_allclose(
            native[native_name].numpy(), expected, rtol=0, atol=2e-14
        )
    assert native["unflattened_refused"] and native["cnot_equal"]
    assert native["selected_instances"] == (1, 2, 3)
    assert len(native["custom_nodes"]) == 3
    for actual, expected in zip(
        native["cnot_nodes"], source["tn"].tensors, strict=True
    ):
        np.testing.assert_array_equal(actual.tensor.numpy(), expected.data)
        assert actual.name in expected.tags
    from qualtran.simulation.tensor import cbloq_to_quimb

    source_multi = cbloq_to_quimb(source["bloq"].decompose_bloq())
    for actual, expected in zip(
        native["multi_nodes"], source_multi.tensors, strict=True
    ):
        np.testing.assert_array_equal(actual.tensor.numpy(), expected.data)
    info = source_multi.contraction_info(optimize="greedy")
    profile = native["contraction_info"]
    assert int(info.opt_cost) == 2 * profile.estimated_cost == 9216
    assert int(info.largest_intermediate) == profile.peak_size == 2048
    assert len(info.path) == profile.n_steps == 2
    label = native["example_inner_index"]
    assert native["occurrences"][label] == ((0, 4), (1, 0))
    torch.testing.assert_close(
        native["bell_from_zero"], native["bell_state"], rtol=0, atol=0
    )

"""Execute every frozen notebook cell through explicit native boundary adapters.

Foreign constructors exist only in this test boundary. The notebook bytes and
assertions stay unchanged; numerical data and drawing come from native owners.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from examples.algorithms.classical_tensor_crosscheck import prepare_classical_crosscheck
from examples.algorithms.classical_tensor_networks import (
    closed_crosscheck_network,
    select_crosscheck_graph,
)
from examples.algorithms.classical_tensor_scores import crosscheck_score_data
from flagquantum.core.ir.program import ProgramRecord
from flagquantum.drawer.register_score_data import MusicalScoreData
from flagquantum.drawer.register_score_mpl import draw_register_score
from flagquantum.simulation.tensor_network.contraction import (
    _contract_nodes_with_slicing_plan,
)
from flagquantum.simulation.tensor_network.models import (
    TensorNetworkNode,
    TensorNetworkSlicingPlan,
)
from flagquantum.simulation.unitary import get_unitary

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@dataclass(frozen=True)
class NativeGate:
    """Translate the notebook's four constructor calls at the reference boundary."""

    kind: str
    width: int = 5

    @property
    def signature(self) -> "NativeSignature":
        return NativeSignature(2 * self.width)


@dataclass(frozen=True)
class NativeSignature:
    width: int

    def n_qubits(self) -> int:
        return self.width


@dataclass(frozen=True)
class NativeClosed:
    kind: str
    record: ProgramRecord

    def as_composite_bloq(self) -> "NativeClosed":
        return self

    def flatten(self) -> "NativeClosed":
        return NativeClosed(self.kind, select_crosscheck_graph(self.record, self.kind))

    def tensor_contract(self) -> np.ndarray[Any, Any]:
        return network(self).contract()


@dataclass(frozen=True)
class NativeNetwork:
    nodes: tuple[TensorNetworkNode, ...]
    plan: TensorNetworkSlicingPlan

    def contraction_width(self) -> float:
        return math.log2(self.plan.peak_size)

    def contract(self) -> np.ndarray[Any, Any]:
        return (
            _contract_nodes_with_slicing_plan(self.nodes, (), self.plan)
            .detach()
            .numpy()
        )


def network(closed: NativeClosed) -> NativeNetwork:
    return NativeNetwork(*closed_crosscheck_network(closed.record, closed.kind))


def flank(
    gate: NativeGate,
    in_vals: dict[str, object],
    out_vals: dict[str, object] | None = None,
) -> NativeClosed:
    return NativeClosed(
        gate.kind,
        prepare_classical_crosscheck(gate.kind, in_vals, out_vals, width=gate.width),
    )


def score(closed: NativeClosed) -> MusicalScoreData:
    return crosscheck_score_data(closed.record, closed.kind)


def test_every_original_notebook_cell_uses_native_numerics_and_drawing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    pytestconfig: pytest.Config,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import qualtran
    import qualtran.bloqs.arithmetic as arithmetic
    import qualtran.bloqs.basic_gates as basic
    import qualtran.drawing as drawing
    import qualtran.simulation.tensor as tensor
    import qualtran.simulation.xcheck_classical_quimb as crosscheck
    from matplotlib import pyplot as plt

    import examples.algorithms.classical_tensor_crosscheck as native

    pytestconfig.addinivalue_line("markers", "notebook: upstream notebook execution")
    import qualtran.simulation.xcheck_classical_quimb_test as original

    path = Path(crosscheck.__file__).with_suffix(".ipynb")
    raw = path.read_bytes()
    assert (
        hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        == "9d20a4bf4468b6013b8c3c26263cf58f8c78854e"
    )
    notebook = json.loads(raw)
    draws: list[MusicalScoreData] = []
    networks: list[NativeNetwork] = []

    def draw(data: MusicalScoreData, **options: float) -> object:
        figure, axes = draw_register_score(data, **options)
        figure.canvas.draw()
        assert axes.has_data()
        draws.append(data)
        plt.close(figure)
        return figure, axes

    def show(closed: NativeClosed, *, type: str) -> object:
        assert type == "musical_score"
        return draw(score(closed))

    def to_network(closed: NativeClosed) -> NativeNetwork:
        result = network(closed)
        networks.append(result)
        return result

    def bounded_unitary(circuit: Any, **kwargs: Any) -> torch.Tensor:
        assert circuit.n_wires <= 3
        return get_unitary(circuit, **kwargs)

    monkeypatch.setattr(native, "get_unitary", bounded_unitary)
    monkeypatch.setattr(basic, "XGate", lambda: NativeGate("xgate"))
    monkeypatch.setattr(basic, "CNOT", lambda: NativeGate("cnot"))
    monkeypatch.setattr(basic, "Toffoli", lambda: NativeGate("toffoli"))
    monkeypatch.setattr(qualtran, "QUInt", lambda bitsize: bitsize)
    monkeypatch.setattr(arithmetic, "Add", lambda bitsize: NativeGate("add", bitsize))
    monkeypatch.setattr(crosscheck, "flank_with_classical_vectors", flank)
    monkeypatch.setattr(drawing, "show_bloq", show)
    monkeypatch.setattr(drawing, "get_musical_score_data", score)
    monkeypatch.setattr(drawing, "draw_musical_score", draw)
    monkeypatch.setattr(tensor, "cbloq_to_quimb", to_network)
    namespace: dict[str, Any] = {}
    cells = []

    def execute_notebook(name: str) -> None:
        assert name == "xcheck_classical_quimb"
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                exec(
                    compile(
                        "".join(cell["source"]), f"{path.name}:cell{index}", "exec"
                    ),
                    namespace,
                )
                cells.append(index)

    monkeypatch.setattr(original.qlt_testing, "execute_notebook", execute_notebook)
    try:
        original.test_notebook()
    finally:
        plt.close("all")
    assert len(cells) == 13
    assert len(draws) == 5
    assert len(networks) == 1
    large = networks[0]
    assert max(n.tensor.numel() for n in large.nodes) <= 32
    assert large.plan.peak_size <= 1024
    assert large.plan.n_slices == 1
    np.testing.assert_allclose(large.contract(), 1, atol=2e-12, rtol=0)
    assert namespace["add_large"].signature.n_qubits() == 64
    assert namespace["result"] == 0
    assert namespace["add_flat"].record == select_crosscheck_graph(
        namespace["add_tt"].record, "add"
    )
    output = capsys.readouterr().out
    assert "universe has been annihilated" in output
    assert "64-qubit gate" in output
    assert path.read_bytes() == raw


@pytest.mark.parametrize(
    "width,valid", [(5, True), (5, False), (32, True), (32, False)]
)
def test_bounded_network_preserves_correct_and_wrong_effects(
    width: int, valid: bool
) -> None:
    outputs = {"a": 2, "b": 5 + int(not valid)}
    graph = prepare_classical_crosscheck("add", {"a": 2, "b": 3}, outputs, width=width)
    selected = select_crosscheck_graph(graph, "add")
    assert selected == select_crosscheck_graph(selected, "add")
    for record in (graph, selected):
        nodes, plan = closed_crosscheck_network(record, "add")
        assert plan.peak_size <= 1024 and plan.n_slices == 1
        assert max(node.tensor.numel() for node in nodes) <= 32
        torch.testing.assert_close(
            _contract_nodes_with_slicing_plan(nodes, (), plan),
            torch.tensor(int(valid), dtype=torch.complex128),
            atol=2e-12,
            rtol=0,
        )


def test_network_refuses_open_boundary() -> None:
    from examples.algorithms.classical_tensor_crosscheck import crosscheck_program
    from flagquantum.compiler.registers.component_builder import ComponentProgramBuilder
    from flagquantum.core.ir.program.signatures import Signature

    gate = crosscheck_program("xgate")
    root = gate.functions[0]
    builder = ComponentProgramBuilder(
        Signature(root.registers or ()), gate.functions, name="classical_vectors"
    )
    record = builder.finalize(builder.call_dict("main", builder.inputs))
    with pytest.raises(ValueError, match="requires closed ports"):
        closed_crosscheck_network(record, "xgate")


def test_bounded_example_main(capsys: pytest.CaptureFixture[str]) -> None:
    from examples.algorithms.classical_tensor_networks import main

    main()
    output = capsys.readouterr().out
    assert "Add(32): (1+0j), planned peak" in output
    assert "1 slices" in output


def test_native_notebook_runs_without_reference_boundary_adapters(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from qualtran import Bloq

    def refuse_reference(*args: object, **kwargs: object) -> None:
        raise AssertionError("native notebook must not execute reference numerics")

    monkeypatch.setattr(Bloq, "tensor_contract", refuse_reference)
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/notebooks/classical_tensor_crosscheck.ipynb"
    )
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    namespace: dict[str, Any] = {}
    count = 0
    try:
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                exec(
                    compile(
                        "".join(cell["source"]), f"{path.name}:cell{index}", "exec"
                    ),
                    namespace,
                )
                count += 1
        assert count == 13
        assert len(plt.get_fignums()) == 5
        for number in plt.get_fignums():
            plt.figure(number).canvas.draw()
        assert namespace["large_result"].item() == 1
        assert namespace["plan"].peak_size <= 1024
        assert namespace["plan"].n_slices == 1
    finally:
        plt.close("all")
    assert "64 logical gate bits; 501 nodes;" in capsys.readouterr().out

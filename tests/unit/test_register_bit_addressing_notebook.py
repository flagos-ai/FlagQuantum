"""Close all frozen addressing tutorial cells through the executable typed graph."""

import builtins
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from hashlib import sha1
from itertools import product
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

from tests.unit.test_register_wiring_notebook import execute

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
pytest.importorskip("numpy")
pytest.importorskip("qualtran")
pytest.importorskip("cirq")
pytest.importorskip("IPython")
pytest.importorskip("graphviz")
ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT.parent
    / "ftqc-reference-env/tutorials/quantum-programming/3-addressing-bits.ipynb"
)
NATIVE = ROOT / "examples/notebooks/register_bit_addressing.ipynb"


@pytest.fixture(scope="module")
def original() -> dict[str, Any]:
    raw = SOURCE.read_bytes()
    assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "a0f1acfbe5dd8c0bc05ced68b3255bcbcabad717"
    )
    return execute(SOURCE)


@pytest.fixture(scope="module")
def native() -> dict[str, Any]:
    return execute(NATIVE)


def test_complete_cells_use_native_owners(
    original: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    source, actual = json.loads(SOURCE.read_text()), json.loads(NATIVE.read_text())
    assert len(source["cells"]) == 11 and len(actual["cells"]) == 12
    assert [c["cell_type"] for c in actual["cells"][:-1]] == [
        c["cell_type"] for c in source["cells"]
    ]
    assert [c["metadata"]["frozen_source_cell"] for c in actual["cells"][:-1]] == list(
        range(11)
    )
    original_import = builtins.__import__

    def native_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name.split(".")[0] in {"qualtran", "cirq", "kickmix"}:
            raise AssertionError("Native addressing must use native owners")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", native_import)
    state = execute(NATIVE)
    assert state["n"] == original["n"] == 4
    assert state["result"] == {"x": 6, "y": 5}


def test_every_signed_input_pair_follows_stored_calls(
    original: dict[str, Any], native: dict[str, Any]
) -> None:
    for x, y in product(range(-8, 8), repeat=2):
        result = native["classical_swap"](x, y)
        assert (
            (result["x"], result["y"])
            == original["swap_routine"].call_classically(x=x, y=y)
            == (y, x)
        )


def test_exact_primitive_stream_and_separate_model(
    original: dict[str, Any], native: dict[str, Any]
) -> None:
    import cirq
    import numpy as np
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from flagquantum.compiler.registers.register_binding import bind_register_program
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates

    gates = native["circuit"].to_ir().instructions
    source_circuit, _ = original["swap_routine"].to_cirq_circuit_and_quregs(
        x=np.asarray(cirq.LineQubit.range(4)),
        y=np.asarray(cirq.LineQubit.range(4, 8)),
    )
    source_swaps = list(source_circuit.all_operations())
    assert all(op.gate == cirq.SWAP for op in source_swaps)
    assert [(g.name, g.wires) for g in gates] == [
        ("swap", tuple(q.x for q in op.qubits)) for op in source_swaps
    ]
    source = reference_arithmetic_gates(
        original["swap_routine"],
        {"x": (0, 1, 2, 3), "y": (4, 5, 6, 7)},
        (),
        {},
        keep_exchange_leaves=True,
    )
    from flagquantum.compiler.basis_translation import translate
    from flagquantum.core.ir import Instruction

    expanded: list[Instruction] = []
    for gate in gates:
        lowered = translate(gate, can_run=lambda instruction: instruction.name == "cx")
        assert lowered is not None
        expanded.extend(lowered)
    assert tuple(expanded) == source
    assert Counter(g.name for g in source) == {"cx": 12}
    assert Counter(g.name for g in gates) == native["primitive_counts"] == {"swap": 4}
    assert [g.wires for g in gates] == [(i, i + 4) for i in range(4)]
    bound = bind_register_program(native["swap_routine"], ((0, 1, 2, 3), (4, 5, 6, 7)))
    assert bound.outputs == ((0, 1, 2, 3), (4, 5, 6, 7)) and bound.events == ()
    cost = get_cost_value(original["swap_routine"], QECGatesCost())
    model = native["model"]
    assert cost.clifford == model.clifford == 4
    assert cost.total_t_count() == model.t == 0
    assert cost.measurement == model.measurement == 0
    assert (
        get_cost_value(original["swap_routine"], QubitCount())
        == model.logical_qubits
        == 8
    )


def test_full_complex_state_with_entangled_spectator(
    original: dict[str, Any], native: dict[str, Any]
) -> None:
    import numpy as np

    import flagquantum as fq

    coefficients = torch.arange(512, dtype=torch.float64)
    state = torch.complex(coefficients + 1, 3 * coefficients - 7)
    state /= state.norm()
    circuit = fq.Circuit(9, inputs=state, dtype=torch.complex128).append(
        native["WordSwap"]()
    )
    result = circuit.state().flatten()
    source_tensor = torch.from_numpy(
        np.asarray(original["swap_routine"].tensor_contract())
    ).to(torch.complex128)
    expected = (source_tensor @ state.reshape(256, 2)).flatten()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert circuit.n_qubits == 9
    assert all(g.name == "swap" for g in circuit.to_ir().instructions)


def test_complete_graph_and_musical_geometry(
    original: dict[str, Any], native: dict[str, Any]
) -> None:
    from qualtran.drawing import get_musical_score_data

    from tests.unit.test_drawer_component_score import assert_source_geometry

    calls = list(original["swap_routine"].iter_bloqsoqs())
    assert Counter(type(call[0].bloq).__name__ for call in calls) == {
        "Split": 2,
        "TwoBitSwap": 4,
        "Join": 2,
    }
    score = native["score"]
    assert len(score.columns) == 8 and score.output_order == ("call6:reg", "call7:reg")
    assert_source_geometry(
        native["score_data"], get_musical_score_data(original["swap_routine"])
    )
    assert native["graph_dot"].count("->") == len(original["swap_routine"].connections)

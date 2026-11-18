"""Close every frozen reflection notebook cell through native owners."""

import builtins
import json
from collections.abc import Mapping, Sequence
from hashlib import sha1
from math import sqrt
from pathlib import Path
from types import ModuleType

import pytest
import torch

from tests.unit.test_addition_file_audit import execute
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_complete_reference_and_sdk_free_native_notebook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import IPython.display
    from qualtran import Adjoint
    from qualtran.bloqs.basic_gates import XGate
    from qualtran.bloqs.mcmt import MultiControlZ
    from qualtran.bloqs.state_preparation import StatePreparationAliasSampling
    from qualtran.resource_counting.generalizers import ignore_split_join

    root = Path(__file__).resolve().parents[2]
    raw = (
        root.parent / "ftqc-reference-env/qualtran/bloqs/reflections/reflections.ipynb"
    ).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "e4d3316f906b697634149f4e61903333d947e7af"
    )
    source_notebook = json.loads(raw)
    native_notebook = json.loads(
        (root / "examples/notebooks/prepared_state_reflections.ipynb").read_text()
    )
    assert len(source_notebook["cells"]) == 19
    assert len(native_notebook["cells"]) == 21
    assert [c["cell_type"] for c in source_notebook["cells"]] == [
        c["cell_type"] for c in native_notebook["cells"][:19]
    ]
    displays: list[object] = []
    monkeypatch.setattr(IPython.display, "display", displays.append)
    reference = execute(source_notebook)
    original_import = builtins.__import__

    def native_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name.split(".")[0] in {"qualtran", "cirq"}:
            raise AssertionError(
                "Native notebook imports must stop at owned boundaries"
            )
        return original_import(name, globals, locals, fromlist, level)

    displays.clear()
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", native_import)
        native = execute(native_notebook)
    assert len(displays) == 10
    for name in ("refl_using_prep", "refl_around_zero", "prepare_identity"):
        assert [(r.name, r.bitsize, r.side.name) for r in native[name].signature] == [
            (r.name, r.bitsize, r.side.name) for r in reference[name].signature
        ]
    source = reference["refl_using_prep"]
    assert (
        native["prepare_gate"].l1_norm_of_coeffs
        == source.prepare_gate.l1_norm_of_coeffs
    )
    assert_source_model(
        native["refl_using_prep"].resource_model(
            native["prepare_cost"], native["prepare_cost"]
        ),
        source,
    )
    _, counts = source.call_graph(max_depth=1, generalizer=ignore_split_join)
    expected = {}
    for call, multiplicity in counts.items():
        if isinstance(call, StatePreparationAliasSampling):
            name = "Alias PREPARE"
        elif isinstance(call, Adjoint):
            assert isinstance(call.subbloq, StatePreparationAliasSampling)
            name = "Alias PREPARE adjoint"
        elif isinstance(call, MultiControlZ):
            assert call.concrete_cvs == (0, 0, 0)
            name = "MultiControlZ(0,0,0)"
        else:
            assert isinstance(call, XGate)
            name = "X"
        expected[name] = multiplicity
    assert native["refl_using_prep_graph"].leaf_counts == expected
    _, counts = reference["prepare_identity"].call_graph(
        max_depth=1, generalizer=ignore_split_join
    )
    assert native["prepare_identity_graph"].leaf_counts == {
        f"Identity({call.bitsize})": count for call, count in counts.items()
    }
    expected_state = torch.tensor([1, 0, -1, 0], dtype=torch.complex128) / sqrt(2)
    torch.testing.assert_close(
        native["circuit"].state().flatten(), expected_state, rtol=0, atol=2e-12
    )
    assert native["measured_primitives"] == {"i": 2, "x": 4, "cz": 1, "phase": 1}
    assert native["small_model"].gates.total_t_count() == 0
    assert native["small_model"].peak_logical_qubits == 2
    svg = [item for item in displays if isinstance(item, IPython.display.SVG)]
    assert len(svg) == 5
    for item in svg:
        assert "<svg" in item.data and "<text" in item.data

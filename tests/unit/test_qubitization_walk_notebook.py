"""Execute all walk notebook cells with frozen factory data and owned bindings."""

import builtins
import json
from collections.abc import Mapping, Sequence
from hashlib import sha1
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from tests.unit.test_addition_file_audit import execute
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_complete_source_and_native_walk_notebook(
    monkeypatch: pytest.MonkeyPatch,
    pytestconfig: pytest.Config,
) -> None:
    import IPython.display
    from qualtran.bloqs.block_encoding.lcu_block_encoding import SelectBlockEncoding
    from qualtran.bloqs.reflections.reflection_using_prepare import (
        ReflectionUsingPrepare,
    )
    from qualtran.resource_counting.generalizers import ignore_split_join

    root = Path(__file__).resolve().parents[2]
    raw = (
        root.parent
        / "ftqc-reference-env/qualtran/bloqs/qubitization/qubitization_walk_operator.ipynb"
    ).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "a31dafeb427e4d7e13bfb0eb9bc903b38a15a316"
    )
    original = json.loads(raw)
    migrated = json.loads(
        (root / "examples/notebooks/qubitization_walk.ipynb").read_text()
    )
    assert len(original["cells"]) == 16 and len(migrated["cells"]) == 18
    assert [c["cell_type"] for c in original["cells"]] == [
        c["cell_type"] for c in migrated["cells"][:16]
    ]
    displays: list[object] = []
    monkeypatch.setattr(IPython.display, "display", displays.append)
    # Original integral fixtures declare their upstream notebook test marker.
    pytestconfig.addinivalue_line("markers", "notebook: upstream notebook execution")
    source = execute(original)
    original_import = builtins.__import__

    def native_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name.split(".")[0] in {"qualtran", "cirq", "openfermion"}:
            raise AssertionError("Native notebook must use owned implementations")
        return original_import(name, globals, locals, fromlist, level)

    displays.clear()
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", native_import)
        native = execute(migrated)
    assert len(displays) == 8
    for native_name, source_name in (
        ("walk_signature", "walk_op"),
        ("thc_signature", "thc_walk_op"),
        ("sparse_signature", "walk_op_chem_sparse"),
    ):
        assert {
            r.name: (r.bitsize, r.shape, r.side.name) for r in native[native_name]
        } == {
            r.name: (r.bitsize, r.shape, r.side.name)
            for r in source[source_name].signature
        }
    assert native["walk_op"].alpha == source["walk_op"].sum_of_lcu_coefficients
    assert native["qroam_blocking_factor"] == source["qroam_blocking_factor"] == 32
    for name in ("t_l", "eta", "zeta", "tpq", "eris"):
        np.testing.assert_array_equal(native[name], source[name])
    original_prepare = source["thc_walk_op"].prepare
    assert native["thc_data"].source_datasets == tuple(
        tuple(map(int, column))
        for column in (
            original_prepare.theta,
            original_prepare.alt_theta,
            original_prepare.alt_mu,
            original_prepare.alt_nu,
            original_prepare.keep,
        )
    )
    assert native["thc_data"].normalization == original_prepare.l1_norm_of_coeffs
    assert native["walk_op_chem_sparse"].alpha == pytest.approx(
        source["walk_op_chem_sparse"].sum_of_lcu_coefficients
    )
    assert_source_model(native["ising_example"].resource_model(), source["walk_op"])
    _, original_calls = source["walk_op"].call_graph(
        max_depth=1, generalizer=ignore_split_join
    )
    expected = {}
    for call, count in original_calls.items():
        if isinstance(call, SelectBlockEncoding):
            name = "SELECT block"
        else:
            assert isinstance(call, ReflectionUsingPrepare)
            name = "Prepared-state reflection"
        expected[name] = count
    assert native["walk_graph"].leaf_counts == expected
    target = torch.tensor([1, 0, 1, 0, -1, 0, 1, 0], dtype=torch.complex128) / 2
    torch.testing.assert_close(
        native["circuit"].state().flatten(), target, rtol=0, atol=2e-12
    )
    assert native["measured_primitives"] == {
        "i": 2,
        "x": 6,
        "cx": 1,
        "cz": 2,
        "phase": 1,
    }
    assert native["small_model"].gates.total_t_count() == 0
    assert native["small_model"].peak_logical_qubits == 3
    svg = [item for item in displays if isinstance(item, IPython.display.SVG)]
    assert len(svg) == 4
    assert all("<svg" in item.data and "<text" in item.data for item in svg)

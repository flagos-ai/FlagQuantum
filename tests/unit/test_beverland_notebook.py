"""Compare every cell of the three original Beverland literature scenarios."""

import builtins
import contextlib
import io
import json
from collections.abc import Mapping, Sequence
from hashlib import sha1
from pathlib import Path
from types import ModuleType
from typing import Any

import black
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_complete_notebook_intermediates_and_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[2]
    raw = (
        root.parent
        / "ftqc-reference-env/qualtran/surface_code/beverland_et_al_model.ipynb"
    ).read_bytes()
    assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "1dca78647d0fd0f8d0a8cd8e37c64861b066fbc0"
    )
    source = json.loads(raw)
    native = json.loads(
        (root / "examples/notebooks/beverland_resource_model.ipynb").read_text()
    )
    assert len(source["cells"]) == 57
    assert len(native["cells"]) == 58
    original_import = builtins.__import__

    def native_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name.split(".")[0] in {"qualtran", "cirq", "openfermion"}:
            raise AssertionError("Native execution must use FlagQuantum owners")
        return original_import(name, globals, locals, fromlist, level)

    source_state: dict[str, Any] = {}
    native_state: dict[str, Any] = {}
    compared = 0
    for i, (expected_cell, actual_cell) in enumerate(
        zip(source["cells"], native["cells"], strict=False)
    ):
        assert actual_cell["metadata"]["frozen_source_cell"] == i
        assert expected_cell["cell_type"] == actual_cell["cell_type"]
        if expected_cell["cell_type"] != "code":
            continue
        source_code = "".join(expected_cell["source"])
        native_code = "".join(actual_cell["source"])
        assert black.format_str(native_code, mode=black.Mode()) == native_code
        expected_output, actual_output = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(expected_output):
            exec(compile(source_code, f"source-cell-{i}", "exec"), source_state)
        with monkeypatch.context() as patch:
            patch.setattr(builtins, "__import__", native_import)
            with contextlib.redirect_stdout(actual_output):
                exec(compile(native_code, f"native-cell-{i}", "exec"), native_state)
        assert actual_output.getvalue() == expected_output.getvalue(), i
        for key in (
            "logical_qubits",
            "error_budget",
            "c_min",
            "t_operations",
            "d",
            "t_s",
            "num_factories",
            "factory_qubits",
            "distillation_qubits",
            "q",
            "total_seconds",
            "total_days",
            "total_hours",
            "h",
            "m",
        ):
            if key in source_state:
                assert native_state[key] == source_state[key], (i, key)
        for name, layers in (("qd", 501), ("chem", 205000000), ("shor", 12)):
            key = f"{name}_alg"
            if key in source_state:
                actual, expected = native_state[key], source_state[key]
                assert actual.peak_logical_qubits == expected.n_algo_qubits
                assert expected.n_rotation_layers == layers
                for gate, count in expected.n_logical_gates.asdict().items():
                    assert getattr(actual.gates, gate) == count
        compared += 1
    assert compared == sum(c["cell_type"] == "code" for c in source["cells"])
    assert native_state["qd_alg"].gates.measurement == 140000
    assert native_state["shor_alg"].gates.toffoli == 3730000000
    assert native_state["BEVERLAND_ROTATION_COST"].overhead == 5.3

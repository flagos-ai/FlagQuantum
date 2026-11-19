"""Run the entire frozen THC physical-cost tutorial through native owners."""

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


def test_complete_thc_notebook_and_optimal_physical_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[2]
    raw = (
        root.parent / "ftqc-reference-env/qualtran/surface_code/thc_compilation.ipynb"
    ).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "e82d0fee44ceb5f9cf2216ba929358e140003869"
    )
    original = json.loads(raw)
    native = json.loads(
        (root / "examples/notebooks/thc_physical_compilation.ipynb").read_text()
    )
    assert len(original["cells"]) == 7
    assert len(native["cells"]) == 8
    source_state: dict[str, Any] = {}
    native_state: dict[str, Any] = {}
    original_import = builtins.__import__

    def native_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name.split(".")[0] in {"qualtran", "cirq", "openfermion"}:
            raise AssertionError("Native tutorial must use FlagQuantum owners")
        return original_import(name, globals, locals, fromlist, level)

    for i, (source_cell, native_cell) in enumerate(
        zip(original["cells"], native["cells"], strict=False)
    ):
        assert native_cell["metadata"]["frozen_source_cell"] == i
        assert native_cell["cell_type"] == source_cell["cell_type"]
        if source_cell["cell_type"] != "code":
            continue
        expected_output, actual_output = io.StringIO(), io.StringIO()
        source_code, native_code = "".join(source_cell["source"]), "".join(
            native_cell["source"]
        )
        assert black.format_str(native_code, mode=black.Mode()) == native_code
        with contextlib.redirect_stdout(expected_output):
            exec(compile(source_code, f"source-cell-{i}", "exec"), source_state)
        with monkeypatch.context() as patch:
            patch.setattr(builtins, "__import__", native_import)
            with contextlib.redirect_stdout(actual_output):
                exec(compile(native_code, f"native-cell-{i}", "exec"), native_state)
        assert actual_output.getvalue() == expected_output.getvalue(), i
        for name in ("cost", "best_cost"):
            if name not in source_state:
                continue
            actual, expected = native_state[name], source_state[name]
            assert actual.failure_probability == expected.failure_prob
            assert actual.duration_hours == expected.duration_hr
            assert actual.physical_qubits == expected.footprint
            assert actual.qubit_hours == expected.qubit_hours
        for name in ("n_algo_qubits", "distillation_error", "data_error"):
            if name in source_state:
                assert native_state[name] == source_state[name]
        for name in ("factory", "best_factory"):
            if name in source_state:
                actual, expected = native_state[name], source_state[name]
                assert actual.n_factories == expected.n_factories == 4
                assert (
                    actual.base_factory.level_one_distance
                    == expected.base_factory.distillation_l1_d
                )
                assert (
                    actual.base_factory.level_two_distance
                    == expected.base_factory.distillation_l2_d
                )
        for name in ("data_block", "best_data_block"):
            if name in source_state:
                actual, expected = native_state[name], source_state[name]
                assert actual.distance == expected.data_d
                assert actual.routing_overhead == expected.routing_overhead == 0.5
    assert native_state["n_logical_gates"].toffoli == 6665400000
    assert native_state["best_cost"].duration_hours == 89.1034375
    assert native_state["best_cost"].physical_qubits == 2933032
    assert native_state["best_cost"].failure_probability == 0.007725395132201774
    assert native_state["best_factory"].base_factory.level_one_distance == 17
    assert native_state["best_factory"].base_factory.level_two_distance == 29
    assert native_state["best_data_block"].distance == 33

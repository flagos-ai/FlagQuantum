"""Execute the complete local estimator notebook with a fixed optional SDK."""

import builtins
import json
from collections.abc import Mapping, Sequence
from hashlib import sha1
from importlib.metadata import version
from pathlib import Path
from types import ModuleType

import black
import pytest

from flagquantum.algorithms.primitives.modular_exponentiation import (
    ModularExponentiation,
)
from flagquantum.compiler._component_trace.resource_model import (
    GateCost,
    ModeledResources,
)
from flagquantum.ecosystem.resource_estimator import microsoft_logical_counts
from tests.unit.test_addition_file_audit import execute

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]

SpaceChart = pytest.importorskip("qsharp_widgets").SpaceChart
ModExp = pytest.importorskip("qualtran.bloqs.cryptography.rsa").ModExp
_source_resources = pytest.importorskip("qualtran.resource_counting")
QubitCount = _source_resources.QubitCount
get_cost_value = _source_resources.get_cost_value


def test_complete_original_and_native_local_estimator_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[2]
    raw = (
        root.parent
        / "ftqc-reference-env/qualtran/surface_code/msft_resource_estimator_interop.ipynb"
    ).read_bytes()
    assert (
        sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == "25b2ba4cb6b9073444016e3c89e72f9de4a0c2e8"
    )
    assert version("qsharp") == version("qsharp-widgets") == "1.9.0"
    source, native = json.loads(raw), json.loads(
        (root / "examples/notebooks/physical_resource_interop.ipynb").read_text()
    )
    assert len(source["cells"]) == 8 and len(native["cells"]) == 9
    for i, (left, right) in enumerate(
        zip(source["cells"], native["cells"], strict=False)
    ):
        assert left["cell_type"] == right["cell_type"]
        assert right["metadata"]["frozen_source_cell"] == i
        if right["cell_type"] == "code":
            code = "".join(right["source"])
            assert black.format_str(code, mode=black.Mode()) == code
    original_state = execute(source)
    original_import = builtins.__import__

    def native_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name.split(".")[0] in {"qualtran", "cirq", "openfermion"}:
            raise AssertionError(
                "Native interoperability must use owned workload models"
            )
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", native_import)
        native_state = execute(native)
    assert native_state["estimates"].data() == original_state["estimates"].data()
    assert len(native_state["estimates"].data()) == 4
    assert (
        SpaceChart(native_state["estimates"]).estimates
        == SpaceChart(original_state["estimates"]).estimates
    )
    get_counts = native_state["get_modular_exponentiation_counts"]
    for base, modulus, exponent, width in (
        (2, 3, 2, 2),
        (4, 15, 3, 4),
        (8, 221, 32, 32),
    ):
        original = ModExp(base, modulus, exponent, width)
        actual = get_counts(ModularExponentiation(base, modulus, exponent, width))
        reference = original_state["get_bloq_logical_counts"](original)
        assert all(actual[key] == reference[key] for key in actual)
    assert get_counts(native_state["mod_exp"]) == {
        "numQubits": 64,
        "tCount": 1334272,
        "rotationCount": 0,
        "rotationDepth": 0,
    }
    assert (
        native_state["mod_exp"].source_signature.n_qubits()
        == original_state["mod_exp"].signature.n_qubits()
        == 64
    )
    assert get_cost_value(original_state["mod_exp"], QubitCount()) == 195
    workspace, _ = native_state["mod_exp"].workspace_requirements()
    assert 64 + workspace == 195
    assert (
        microsoft_logical_counts(ModeledResources(GateCost(t=1334272), 195))[
            "numQubits"
        ]
        == 195
    )
    graph = native_state["component_graph"]
    assert "exponent" in graph and "x" in graph and "32" in graph

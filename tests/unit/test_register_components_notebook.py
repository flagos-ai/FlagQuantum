"""Close the complete frozen component tutorial through native domain owners."""

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
pytest.importorskip("IPython")
pytest.importorskip("graphviz")
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / "ftqc-reference-env/tutorials/quantum-programming/2-bloqs.ipynb"
NATIVE = ROOT / "examples/notebooks/register_components.ipynb"


@pytest.fixture(scope="module")
def original() -> dict[str, Any]:
    raw = SOURCE.read_bytes()
    assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "8ab4e2b48c82b0104d6b89340128d21386a5fa8f"
    )
    return execute(SOURCE)


@pytest.fixture(scope="module")
def native() -> dict[str, Any]:
    return execute(NATIVE)


def test_complete_cells_execute_with_native_owners(
    original: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    source, actual = json.loads(SOURCE.read_text()), json.loads(NATIVE.read_text())
    assert len(source["cells"]) == 38 and len(actual["cells"]) == 39
    assert [cell["cell_type"] for cell in actual["cells"][:-1]] == [
        cell["cell_type"] for cell in source["cells"]
    ]
    assert [
        cell["metadata"]["frozen_source_cell"] for cell in actual["cells"][:-1]
    ] == list(range(38))
    original_import = builtins.__import__

    def native_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name.split(".")[0] in {"qualtran", "cirq", "kickmix"}:
            raise AssertionError("Native component tutorial must use native owners")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", native_import)
    state = execute(NATIVE)
    state["circuit"].to_ir().validate()
    assert state["signature_only_width"] == 8
    assert state["peak_width"] == 23
    assert state["negated_value"] == -5
    assert state["decomposition_dot"].count("->") == 3
    assert state["call_dot"].count("->") == 2
    assert "2048" in state["large_dot"]
    assert "n" in state["symbolic_dot"]
    assert len(original["negate"].signature) == len(state["symbolic"].signature) == 1


def test_all_values_and_full_eight_bit_tensor(
    original: dict[str, Any], native: dict[str, Any]
) -> None:
    import numpy as np

    source, actual = original["Negate"](8), native["negate"]
    for value in range(-128, 128):
        assert actual.classical(value) == source.call_classically(x=value)[0]
    np.testing.assert_array_equal(native["unitary"], original["unitary"])
    assert native["unitary"].shape == (256, 256)
    np.testing.assert_array_equal(native["unitary"].T @ native["unitary"], np.eye(256))


def test_concrete_large_and_symbolic_hierarchy_and_models(
    original: dict[str, Any], native: dict[str, Any]
) -> None:
    from qualtran.bloqs.arithmetic import AddK, BitwiseNot
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    for width in (2, 3, 8, 2048, native["n"]):
        source, actual = original["Negate"](width), native["Negation"](width)
        assert [(r.name, r.bitsize) for r in actual.signature] == [
            (r.name, r.bitsize) for r in source.signature
        ]
        children = sorted(
            source.decompose_bloq().iter_bloqsoqs(), key=lambda item: item[0].i
        )
        assert [type(item[0].bloq) for item in children] == [BitwiseNot, AddK]
        assert children[1][1] == {"x": children[0][2][0]}
        assert [call.operation for call in actual.calls] == [
            "bitwise_not",
            "constant_add",
        ]
        assert [call.widths for call in actual.calls] == [(width,), (width,)]
        assert actual.calls[1].constant == 1
        assert (
            actual.gate_cost().asdict()
            == get_cost_value(source, QECGatesCost()).asdict()
        )
        if width == 2048:
            # The original tutorial draws this width but never queries its peak.
            with pytest.raises(RuntimeError, match="maximum recursion depth"):
                get_cost_value(source, QubitCount())
            assert actual.logical_width() == 6143
        else:
            assert actual.logical_width() == get_cost_value(source, QubitCount())
    with pytest.raises(ValueError, match="positive integer"):
        native["symbolic"].instructions()


def test_actual_eight_bit_primitives_operands_and_feedback(
    original: dict[str, Any], native: dict[str, Any]
) -> None:
    from tests.unit.test_register_constant_addition import reference_arithmetic_gates
    from tests.unit.test_register_subtraction import assert_source

    expected = reference_arithmetic_gates(
        original["Negate"](8),
        {"x": native["target"]},
        (*native["constant"], *native["carries"]),
        dict(zip(native["carries"], native["classical_bits"], strict=True)),
    )
    gates = native["circuit"].to_ir().instructions
    assert_source(gates, expected)
    assert Counter(g.name for g in gates) == native["primitive_counts"]
    assert len(gates) == 168
    assert native["gate_model"].and_bloq == 7
    assert native["gate_model"].clifford == 56
    assert native["gate_model"].measurement == 7


def test_appended_component_all_coherent_histories_and_zero_work(
    native: dict[str, Any],
) -> None:
    import flagquantum as fq
    from tests.unit.measurement_checks import forced_branch

    width, work = 3, 5
    data = torch.arange(16, dtype=torch.int64)
    x, spectator = data >> 1, data & 1
    values = data.to(torch.float64)
    amplitudes = torch.complex(values + 1, 3 * values - 5)
    amplitudes /= amplitudes.norm()
    state = torch.zeros(512, dtype=torch.complex128)
    state[(x << (work + 1)) | spectator] = amplitudes
    expected = torch.zeros_like(state)
    expected[((-x % 8) << (work + 1)) | spectator] = amplitudes
    circuit = fq.Circuit(9, inputs=state, dtype=torch.complex128).append(
        native["Negation"](width)
    )
    for outcomes in product((0, 1), repeat=2):
        branch = forced_branch(state, circuit.to_ir().instructions, outcomes)
        torch.testing.assert_close(branch, expected / 2, rtol=0, atol=2e-12)
        assert torch.count_nonzero(branch.reshape(8, 32, 2)[:, 1:]) == 0

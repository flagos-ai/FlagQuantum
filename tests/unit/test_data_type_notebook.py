# Copyright 2024 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""The complete dtype tutorial composes native metadata, codecs and programs."""

import builtins
import json
from collections.abc import Mapping, Sequence
from hashlib import sha1
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

pytest.importorskip("qualtran")
pytest.importorskip("numpy")
pytest.importorskip("graphviz")

from flagquantum.circuit import Circuit
from flagquantum.core.ir import Instruction
from flagquantum.core.ir.program import verify_program_record
from tests.unit.test_register_wiring_notebook import execute

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / "ftqc-reference-env/qualtran/DataTypes.ipynb"
NATIVE = ROOT / "examples/notebooks/register_data_types.ipynb"


@pytest.fixture(scope="module")
def original() -> dict[str, Any]:
    raw = SOURCE.read_bytes()
    assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == (
        "edba7005d69b811369ec869e13b3ce77ea33880c"
    )
    namespace = execute(SOURCE)
    cells = json.loads(raw)["cells"]
    for index, key in ((1, "prepared_source"), (10, "complement_source")):
        checkpoint = dict(namespace)
        exec("".join(cells[index]["source"]), checkpoint)
        namespace[key] = checkpoint["cbloq"]
    return namespace


@pytest.fixture(scope="module")
def native() -> dict[str, Any]:
    return execute(NATIVE)


def test_all_source_cells_have_native_owners(
    original: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    source = json.loads(SOURCE.read_text())
    native = json.loads(NATIVE.read_text())
    assert len(source["cells"]) == len(native["cells"]) == 24
    assert [cell["cell_type"] for cell in source["cells"]] == [
        cell["cell_type"] for cell in native["cells"]
    ]
    assert [cell["metadata"]["frozen_source_cell"] for cell in native["cells"]] == list(
        range(24)
    )
    real_import = builtins.__import__

    def native_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name.split(".")[0] in {"qualtran", "cirq", "kickmix"}:
            raise AssertionError("Native dtype tutorial must use native owners")
        return real_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", native_import)
        actual = execute(NATIVE)
    assert len(original["dtypes"]) == len(actual["dtypes"]) == 19


def test_all_declared_widths_and_domain_endpoint_encodings(
    original: dict[str, Any], native: dict[str, Any]
) -> None:
    import qualtran

    for source, target, encoded in zip(
        original["dtypes"], native["dtypes"], native["encoding_rows"], strict=True
    ):
        assert (source.num_bits, source.num_qubits, source.num_cbits) == (
            target.num_bits,
            target.num_qubits,
            target.num_cbits,
        )
        if isinstance(source, qualtran.QAny):
            assert encoded is None
            continue
        domain = list(source.get_classical_domain())
        expected = tuple(source.to_bits(value) for value in (domain[0], domain[-1]))
        assert encoded == expected
    assert native["compatibility"] == (True, True, True, False, True, False, False)
    assert native["integer_to_bits"](0x30, 8) == original["QUInt"](8).to_bits(0x30)


def test_large_negation_is_a_compact_declared_graph(
    original: dict[str, Any], native: dict[str, Any]
) -> None:
    from qualtran.bloqs.arithmetic import AddK, BitwiseNot

    source = original["negate"].decompose_bloq()
    steps = sorted(source.iter_bloqsoqs(), key=lambda row: row[0].i)
    assert [type(row[0].bloq) for row in steps] == [BitwiseNot, AddK]
    assert source.signature.n_qubits() == native["large_signature"].n_qubits() == 2048
    graph = native["large_graph"]
    assert {call.operation for call in graph.leaf_counts} == {
        "bitwise_not",
        "constant_add",
    }
    assert list(graph.leaf_counts.values()) == [1, 1]
    assert all(call.widths == (2048,) for call in graph.leaf_counts)


def test_preparation_cast_and_bitwise_body_preserve_exact_states(
    original: dict[str, Any], native: dict[str, Any]
) -> None:
    for name in ("prepared", "complement", "cast_program"):
        assert verify_program_record(native[name]) == ()
    prepared = native["prepared_bound"]
    assert prepared.outputs == ((1,), (0,))
    assert prepared.circuit.instructions == (Instruction("h", (0,)),)
    expected = torch.tensor([1, 0, 1, 0], dtype=torch.complex128) / 2**0.5
    torch.testing.assert_close(
        native["prepared_circuit"].state().flatten().to(torch.complex128),
        expected,
        rtol=0,
        atol=1e-7,
    )
    cast = native["cast_bound"]
    assert set(cast.outputs[0]) == {0, 1, 2} and not cast.circuit.instructions
    # The final source graph contains three clean allocations, Join and Cast.
    assert len(original["cbloq"].bloq_instances) == 5
    from qualtran.resource_counting import QECGatesCost, get_cost_value

    source_complement = original["complement_source"]
    source_steps = sorted(source_complement.iter_bloqsoqs(), key=lambda row: row[0].i)
    assert [type(row[0].bloq).__name__ for row in source_steps] == [
        "Split",
        "XGate",
        "XGate",
        "XGate",
        "Join",
    ]
    model = get_cost_value(source_complement, QECGatesCost())
    assert model.clifford == 3 and model.total_t_count() == model.measurement == 0
    body = native["complement_bound"].circuit
    assert body.instructions == tuple(Instruction("x", (i,)) for i in range(3))
    initial = torch.arange(1, 17, dtype=torch.float64).to(torch.complex128)
    initial *= torch.exp(0.13j * torch.arange(16, dtype=torch.float64))
    initial /= initial.norm()
    circuit = Circuit(4, inputs=initial, dtype=torch.complex128)
    circuit.append(native["RegisterOperation"](body))
    source_matrix = torch.as_tensor(
        source_complement.tensor_contract(), dtype=torch.complex128
    )
    expected = torch.einsum("ij,jk->ik", source_matrix, initial.reshape(8, 2)).flatten()
    torch.testing.assert_close(circuit.state().flatten(), expected, rtol=0, atol=2e-14)
    circuit.append(native["RegisterOperation"](body))
    torch.testing.assert_close(circuit.state().flatten(), initial, rtol=0, atol=2e-14)
    assert native["observed_counts"] == {"x": 3}

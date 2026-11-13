# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Assertions adapted from frozen Qualtran v0.7.0 addition tests and notebook.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/arithmetic/addition_test.py

"""Complete addition assets reuse native arithmetic, metadata and drawing owners."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pytest
import torch

from examples.algorithms.addition_catalog import (
    addition_example_graph,
    addition_examples,
    addition_signature_data,
    addition_signature_svg,
)
from examples.algorithms.factorized_addition_tensors import addition_tensor_providers
from flagquantum.compiler.registers.arithmetic.addition_component import (
    addition_component_program,
)
from flagquantum.compiler.registers.component_graph import iter_component_calls
from flagquantum.compiler.registers.component_tensor_expansion import (
    expand_components_for_tensor,
)
from flagquantum.drawer.addition_symbols import (
    addition_wire_symbol,
)
from flagquantum.drawer.register_score import RegisterStep
from flagquantum.simulation.tensor_network.component_dense import (
    contract_component_tensors,
)
from tests.unit.test_constant_addition_resource_width import source_key as add_key
from tests.unit.test_out_of_place_resource_width import source_key as oop_key

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]
ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT.parent / "ftqc-reference-env"
ORIGINAL_TESTS = {
    "test_examples",
    "test_add_decomposition",
    "test_add_diff_size_registers",
    "test_add_truncated",
    "test_subtract",
    "test_addition_gate_counts",
    "test_add_tensor_contract",
    "test_add_call_classically",
    "test_add_call_classically_overflow",
    "test_add_truth_table",
    "test_add_in_cbloq",
    "test_add_classical",
    "test_add_symb",
    "test_add_mixed_signed_result_reg",
    "test_add_mixed_signed_op_reg",
    "test_out_of_place_adder",
    "test_controlled_add_k",
    "test_add_k_decomp_unsigned",
    "test_add_k_decomp_signed",
    "test_classical_add_k_unsigned",
    "test_classical_add_signed_overflow",
    "test_classical_add_k_signed",
    "test_notebook",
    "test_outofplaceadder_classical_action",
    "test_controlled_add_from_add",
}


@pytest.fixture(scope="module", autouse=True)
def frozen_complete_source_assets() -> None:
    with (ROOT.parent / "FlagQuantum-Qualtran-Kickmix-移植进度.csv").open(
        encoding="utf-8-sig"
    ) as stream:
        blobs = {row["源文件"]: row["源对象"] for row in csv.DictReader(stream)}
    for filename in ("addition.py", "addition_test.py", "addition.ipynb"):
        path = "qualtran/bloqs/arithmetic/" + filename
        raw = (REFERENCE / path).read_bytes()
        assert (
            hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blobs[path]
        )
    tree = ast.parse(
        (REFERENCE / "qualtran/bloqs/arithmetic/addition_test.py").read_text()
    )
    assert {node.name for node in tree.body if isinstance(node, ast.FunctionDef)} == (
        ORIGINAL_TESTS | {"add_reference_t_complexity"}
    )


@pytest.mark.parametrize("index", range(10))
def test_all_original_example_signatures_and_one_level_graphs(index: int) -> None:
    from qualtran.bloqs.arithmetic import addition as original
    from qualtran.bloqs.arithmetic.addition import OutOfPlaceAdder
    from qualtran.resource_counting.generalizers import ignore_split_join

    name, call, signature = addition_examples()[index]
    source = getattr(original, "_" + name)()
    assert tuple((r.name, r.bitsize, r.shape, r.side.name) for r in signature) == tuple(
        (r.name, r.bitsize, r.shape, r.side.name) for r in source.signature
    )
    expected_encoding = tuple(
        "signed" if type(r.dtype).__name__ == "QInt" else "uint"
        for r in source.signature
    )
    assert tuple(r.dtype.encoding for r in signature) == expected_encoding
    graph, sigma = source.call_graph(max_depth=1, generalizer=ignore_split_join)

    def key(node: Any) -> Any:
        return oop_key(node) if isinstance(node, OutOfPlaceAdder) else add_key(node)

    native = addition_example_graph(call)
    assert native.root == key(source)
    assert native.direct_calls == {
        (key(a), key(b)): edge["n"] for a, b, edge in graph.edges(data=True)
    }
    assert native.leaf_counts == {key(node): count for node, count in sigma.items()}
    svg = addition_signature_svg([(name, call, signature)])
    element = ElementTree.fromstring(svg)
    labels = [node.text for node in element.iter() if node.tag.endswith("}text")]
    assert name in labels
    assert any(
        all(r.name + "=" in label for r in signature) for label in labels if label
    )


def test_native_glyph_admission_and_score_consumer() -> None:
    with pytest.raises(ValueError, match="addition display"):
        addition_wire_symbol(RegisterStep("X", gate="x"), "x", "THRU")
    with pytest.raises(ValueError, match="a or b"):
        addition_wire_symbol(RegisterStep("Add", gate="add"), "c", "THRU")
    assert (
        addition_wire_symbol(
            RegisterStep("A long constant title", gate="constant_add"), None, None
        )[1]["text"]
        == "A long c.."
    )
    for index in (1, 5, 8):
        data = addition_signature_data(addition_examples()[index])
        assert data["vlines"]
        texts = [port["symb_attributes"].get("text") for port in data["soqs"]]
        assert {1: "a+b", 5: "c", 8: "x"}[index] in texts


def execute(notebook: dict[str, Any]) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            exec(
                compile("".join(cell["source"]), f"addition-cell-{index}", "exec"),
                namespace,
            )
    return namespace


def test_complete_frozen_and_native_notebooks(monkeypatch: pytest.MonkeyPatch) -> None:
    import IPython.display
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    original = json.loads(
        (REFERENCE / "qualtran/bloqs/arithmetic/addition.ipynb").read_text()
    )
    native = json.loads(
        (ROOT / "examples/notebooks/register_addition.ipynb").read_text()
    )
    assert len(original["cells"]) == len(native["cells"]) == 33
    assert [c["cell_type"] for c in original["cells"]] == [
        c["cell_type"] for c in native["cells"]
    ]
    renders: list[object] = []
    monkeypatch.setattr(IPython.display, "display", lambda value: renders.append(value))
    try:
        source_namespace = execute(original)
        renders.clear()
        native_namespace = execute(native)
        assert len(renders) == 6
        for prefix in ("add_symb", "add_oop_symb", "add_k"):
            assert prefix + "_g" in source_namespace
            assert (
                native_namespace[prefix + "_g"].leaf_counts
                == native_namespace[prefix + "_sigma"]
            )
        assert len(native_namespace["catalog"]) == 10
    finally:
        plt.close("all")


def test_all_64_original_five_bit_tensor_assertions() -> None:
    from qualtran import QUInt
    from qualtran.bloqs.arithmetic import Add

    graph = addition_component_program(5)
    like = torch.empty((), dtype=torch.complex128)
    providers = addition_tensor_providers(graph, {}, like)
    selected, _ = expand_components_for_tensor(
        graph, providers.has_custom_tensor, full_flatten=False
    )
    order = tuple(i for i, _, _ in iter_component_calls(selected))
    actual = contract_component_tensors(selected, order, providers.supply, like)
    expected = torch.as_tensor(Add(QUInt(5)).tensor_contract(), dtype=torch.complex128)
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-12)
    for a, b in product(range(8), repeat=2):
        assert actual[(a << 5) | (a + b), (a << 5) | b].item() == pytest.approx(
            1, abs=2e-12
        )


@pytest.mark.parametrize("signed_left,signed_right", [(False, True), (True, False)])
def test_original_four_bit_mixed_classical_assertions(
    signed_left: bool, signed_right: bool
) -> None:
    from qualtran import QInt, QUInt
    from qualtran.bloqs.arithmetic import Add

    from flagquantum.simulation.classical_addition import register_add_classical_values

    source = Add(
        (QInt if signed_left else QUInt)(4), (QInt if signed_right else QUInt)(4)
    )
    left_domain = range(-8, 8) if signed_left else range(16)
    right_domain = range(-8, 8) if signed_right else range(16)
    for left, right in product(left_domain, right_domain):
        assert register_add_classical_values(
            left, right, 4, signed=signed_right
        ) == source.call_classically(a=left, b=right)


def test_original_32_bit_classical_and_control_conversion_assertions() -> None:
    from qualtran import CtrlSpec, QInt, QUInt
    from qualtran.bloqs.arithmetic import Add

    from flagquantum.simulation.classical_addition import register_add_classical_values
    from flagquantum.simulation.classical_controlled_addition import (
        controlled_add_classical_values,
    )

    assert (
        register_add_classical_values(10, 3, 32, signed=True)
        == Add(QInt(32)).decompose_bloq().call_classically(a=10, b=3)
        == (10, 13)
    )
    source = Add(QUInt(32)).controlled(CtrlSpec(cvs=0))
    for control in (0, 1):
        assert controlled_add_classical_values(
            control, 5, 10, 32, control_value=0
        ) == source.call_classically(ctrl=control, a=5, b=10)


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_original_out_of_place_classical_assertions(width: int) -> None:
    from qualtran.bloqs.arithmetic import OutOfPlaceAdder

    from flagquantum.simulation.classical_addition import (
        out_of_place_sum_classical_values,
    )

    for adjoint in (False, True):
        source = OutOfPlaceAdder(width, is_adjoint=adjoint)
        for left, right in product(range(1 << width), repeat=2):
            output = left + right if adjoint else None
            assert out_of_place_sum_classical_values(
                left, right, width, adjoint=adjoint, output=output
            ) == source.on_classical_vals(a=left, b=right, c=output)

"""Check the pinned first-quantization tutorial's complete resource figures."""

import ast
import inspect
import json
from collections import Counter
from pathlib import Path

import pytest

from examples.algorithms.first_quantization_catalog import notebook_cases
from flagquantum.algorithms.first_quantization_resource_model import (
    FirstQuantizationPrepareDeclaration,
    FirstQuantizationSelectDeclaration,
)
from flagquantum.compiler._component_trace.call_graph import modeled_call_graph
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from tests.unit.test_first_quantization_resource_model import (
    assert_first_quantization_signature,
    assert_source_model,
)

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
def test_first_quantization_package_contract() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry import pbc
    from qualtran.bloqs.chemistry.pbc import first_quantization
    from qualtran.bloqs.chemistry.pbc.first_quantization import select_and_prepare

    assert not ast.parse(inspect.getsource(pbc)).body
    definitions = ast.parse(inspect.getsource(first_quantization)).body
    assert len(definitions) == 2
    assert isinstance(definitions[1], ast.ImportFrom)
    assert definitions[1].module == "select_and_prepare"
    assert [item.name for item in definitions[1].names] == [
        "PrepareFirstQuantization",
        "SelectFirstQuantization",
    ]
    for item in definitions[1].names:
        assert getattr(first_quantization, item.name) is getattr(
            select_and_prepare, item.name
        )

    notebook = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "notebooks"
        / "register_first_quantization.ipynb"
    )
    introduction = json.loads(notebook.read_text(encoding="utf-8"))["cells"][0]
    assert introduction["cell_type"] == "markdown"
    assert inspect.getdoc(first_quantization) in "".join(introduction["source"])


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("normalization", [None, 0, 7.5, "L"])
def test_first_quantization_normalization_and_grouped_ports(normalization) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.pbc.first_quantization import (
        PrepareFirstQuantization,
        SelectFirstQuantization,
    )
    from sympy import Symbol

    norm = (
        Symbol(normalization, positive=True)
        if isinstance(normalization, str)
        else normalization
    )
    options = dict(m_param=32, num_bits_nuc_pos=7, num_bits_t=13, num_bits_rot_aa=6)
    source = PrepareFirstQuantization(4, 5, 3, 5, sum_of_l1_coeffs=norm, **options)
    native = FirstQuantizationPrepareDeclaration(
        4, 5, 3, 5, sum_of_l1_coeffs=norm, **options
    )
    assert_first_quantization_signature(
        native.selection_registers, source.selection_registers
    )
    assert_first_quantization_signature(native.junk_registers, source.junk_registers)
    if norm is None:
        with pytest.raises(ValueError, match="sum_of_l1_coeffs"):
            _ = source.l1_norm_coeffs
        with pytest.raises(ValueError, match="sum_of_l1_coeffs"):
            _ = native.l1_norm_of_coeffs
    else:
        assert native.l1_norm_of_coeffs == source.l1_norm_coeffs == norm
    assert_source_model(native.resource_model(), source)
    assert_source_model(native.resource_model(adjoint=True), source.adjoint())

    source_select = SelectFirstQuantization(4, 5, 3, 5, **options)
    native_select = FirstQuantizationSelectDeclaration(4, 5, 3, 5, **options)
    assert_first_quantization_signature(
        native_select.control_registers, source_select.control_registers
    )
    assert_first_quantization_signature(
        native_select.selection_registers, source_select.selection_registers
    )
    assert_first_quantization_signature(
        native_select.target_registers, source_select.target_registers
    )
    assert_source_model(native_select.resource_model(), source_select)


@pytest.mark.ftqc_reference
def test_first_quantization_tutorial_examples_match_pinned_source() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.pbc.first_quantization import (
        PrepareFirstQuantization,
        SelectFirstQuantization,
    )
    from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_t import (
        PrepareTFirstQuantization,
    )
    from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_uv import (
        PrepareUVFirstQuantization,
    )
    from qualtran.bloqs.chemistry.pbc.first_quantization.prepare_zeta import (
        PrepareZetaState,
    )
    from qualtran.bloqs.chemistry.pbc.first_quantization.select_t import (
        SelectTFirstQuantization,
    )
    from qualtran.bloqs.chemistry.pbc.first_quantization.select_uv import (
        SelectUVFirstQuantization,
    )
    from qualtran.resource_counting import QubitCount, get_cost_value
    from qualtran.resource_counting.generalizers import ignore_split_join

    originals = (
        PrepareFirstQuantization(6, 10, 10, 10),
        SelectFirstQuantization(6, 10, 10, 10),
        PrepareTFirstQuantization(5, 10),
        PrepareUVFirstQuantization(5, 10, 10, 256, 10, 16),
        SelectTFirstQuantization(5, 10),
        SelectUVFirstQuantization(5, 10, 10, 16),
        PrepareZetaState(10, 10, 8),
    )

    for (signature, model), source in zip(
        notebook_cases().values(), originals, strict=True
    ):
        assert_first_quantization_signature(signature, source.signature)
        graph = modeled_call_graph(model, max_depth=1)
        _, expected = source.call_graph(max_depth=1, generalizer=ignore_split_join)
        assert Counter(
            (
                child.name,
                estimate_ftqc_resources(child).logical.peak_logical_qubits,
                count,
            )
            for child, count in graph.leaf_counts.items()
        ) == Counter(
            (type(child).__name__, get_cost_value(child, QubitCount()), count)
            for child, count in expected.items()
        )
        assert_source_model(model, source)

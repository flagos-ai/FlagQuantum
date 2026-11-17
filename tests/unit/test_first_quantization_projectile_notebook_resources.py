"""Check both frozen projectile notebook examples."""

import inspect
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest

from examples.algorithms.first_quantization_projectile_catalog import notebook_cases
from flagquantum.algorithms.first_quantization_projectile_resource_model import (
    ProjectilePrepareDeclaration,
    ProjectileSelectDeclaration,
)
from flagquantum.compiler._component_trace.call_graph import modeled_call_graph
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from tests.unit.test_first_quantization_projectile_resource_model import (
    assert_projectile_signature,
)
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_projectile_package_exports_and_introduction() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.pbc.first_quantization import projectile
    from qualtran.bloqs.chemistry.pbc.first_quantization.projectile import (
        select_and_prepare,
    )

    assert projectile.PrepareFirstQuantizationWithProj is (
        select_and_prepare.PrepareFirstQuantizationWithProj
    )
    assert projectile.SelectFirstQuantizationWithProj is (
        select_and_prepare.SelectFirstQuantizationWithProj
    )
    notebook = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "notebooks"
        / "register_first_quantization_projectile.ipynb"
    )
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    markdown = "\n".join(
        "".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown"
    )
    assert inspect.getdoc(projectile) in markdown


@pytest.mark.parametrize("custom_precision", [False, True])
def test_projectile_declaration_grouped_ports_and_resources(
    custom_precision: bool,
) -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.pbc.first_quantization.projectile import (
        PrepareFirstQuantizationWithProj,
        SelectFirstQuantizationWithProj,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    args = (3, 5, 5, 3, 6) if custom_precision else (6, 8, 10, 10, 10)
    options = (
        dict(m_param=32, num_bits_nuc_pos=7, num_bits_t=13, num_bits_rot_aa=6)
        if custom_precision
        else {}
    )
    source = PrepareFirstQuantizationWithProj(*args, **options)
    prepare = ProjectilePrepareDeclaration(*args, **options)
    assert_projectile_signature(prepare.selection_registers, source.selection_registers)
    assert_projectile_signature(prepare.junk_registers, source.junk_registers)
    assert_source_model(prepare.resource_model(), source)
    assert not hasattr(source, "l1_norm_coeffs")
    with pytest.raises(NotImplementedError):
        _ = prepare.l1_norm_of_coeffs

    reverse = source.adjoint()
    with pytest.raises(RuntimeError, match="does not accept Soquets.*flag"):
        get_cost_value(reverse, QubitCount())
    native_reverse = estimate_ftqc_resources(prepare.resource_model(adjoint=True))
    assert {k: v for k, v in asdict(native_reverse.logical.gates).items() if v} == (
        get_cost_value(reverse, QECGatesCost()).asdict()
    )
    assert native_reverse.emitted is None

    source_select = SelectFirstQuantizationWithProj(*args, **options)
    select = ProjectileSelectDeclaration(*args, **options)
    assert_projectile_signature(
        select.control_registers, source_select.control_registers
    )
    assert_projectile_signature(
        select.selection_registers, source_select.selection_registers
    )
    assert_projectile_signature(select.target_registers, source_select.target_registers)
    assert_source_model(select.resource_model(), source_select)


def test_projectile_notebook_graphs_match_original() -> None:
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.pbc.first_quantization.projectile import (
        PrepareFirstQuantizationWithProj,
        SelectFirstQuantizationWithProj,
    )
    from qualtran.resource_counting import QubitCount, get_cost_value
    from qualtran.resource_counting.generalizers import ignore_split_join

    originals = (
        PrepareFirstQuantizationWithProj(6, 8, 10, 10, 10),
        SelectFirstQuantizationWithProj(6, 8, 10, 10, 10),
    )
    for (signature, model), source in zip(
        notebook_cases().values(), originals, strict=True
    ):
        assert_projectile_signature(signature, source.signature)
        actual = modeled_call_graph(model, max_depth=1)
        _, expected = source.call_graph(max_depth=1, generalizer=ignore_split_join)
        assert Counter(
            (
                child.name,
                estimate_ftqc_resources(child).logical.peak_logical_qubits,
                count,
            )
            for child, count in actual.leaf_counts.items()
        ) == Counter(
            (type(child).__name__, get_cost_value(child, QubitCount()), count)
            for child, count in expected.items()
        )
        assert_source_model(model, source)

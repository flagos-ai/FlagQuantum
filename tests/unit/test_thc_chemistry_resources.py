"""Verify all original THC teaching fixtures and isolated graph views."""

from xml.etree.ElementTree import fromstring

import pytest

from examples.algorithms.chemistry_resource_tutorial import thc_prepare_t_breakdown
from examples.algorithms.thc_chemistry_resources import (
    folded_thc_costs,
    generalize_thc_resource_call,
    lookup_graph_label,
    prepare_register_score,
    thc_flame_svg,
    thc_tutorial_integrals,
    tutorial_call_model,
    uniform_register_score,
    uniform_t_breakdown,
)
from flagquantum.algorithms.primitives.multiplexers.thc_resource_model import (
    select_thc_cost,
)
from flagquantum.algorithms.primitives.thc_chemistry_data import thc_chemistry_data
from flagquantum.algorithms.primitives.thc_chemistry_prepare import PrepareTHCChemistry
from flagquantum.algorithms.primitives.thc_uniform_preparation import (
    THCUniformPreparation,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


def test_original_examples_and_rendered_costs():
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.thc import (
        PrepareTHC,
        SelectTHC,
        UniformSuperpositionTHC,
    )
    from qualtran.resource_counting.classify_bloqs import classify_t_count_by_bloq_type

    fixture = thc_tutorial_integrals()
    uniform = THCUniformPreparation(10, 4)
    prep = PrepareTHCChemistry(
        thc_chemistry_data(*fixture, probability_bits=8, log_block_size=2)
    )
    uniform_model = uniform.source_resource_model()
    prep_model = prep.source_resource_model()
    select_model = select_thc_cost(10, 8, 12, 10)
    source_uniform = UniformSuperpositionTHC(10, 4)
    source_prep = PrepareTHC.from_hamiltonian_coeffs(
        *fixture, num_bits_state_prep=8, log_block_size=2
    )

    uniform_bins = uniform_t_breakdown(uniform_model)
    assert uniform_bins == classify_t_count_by_bloq_type(source_uniform)
    assert uniform_bins["arithmetic"] == 276
    prep_bins = thc_prepare_t_breakdown(prep_model)
    assert prep_bins == classify_t_count_by_bloq_type(source_prep.decompose_bloq())
    assert prep_bins["data_loading"] == 248

    for model, source, width in (
        (uniform_model, source_uniform, 4),
        (prep_model, source_prep, 8),
        (select_model, SelectTHC(10, 8, 12, 10), 12),
    ):
        diagram = tutorial_call_model(model, width=width)
        assert_source_model(diagram, source)
        expected = estimate_component_resources(model).gates.total_t_count()
        assert sum(folded_thc_costs(diagram).values()) == expected
        assert len(fromstring(thc_flame_svg(diagram))) > 1

    from qualtran.bloqs.bookkeeping import Join, Split

    uniform_calls = source_uniform.decompose_bloq().bloq_instances
    assert len(uniform_register_score(uniform).columns) == sum(
        not isinstance(node.bloq, (Join, Split)) for node in uniform_calls
    )
    assert len(prepare_register_score(prep).columns) == len(
        source_prep.decompose_bloq().bloq_instances
    )


def test_tutorial_resource_generalization_is_local():
    from flagquantum.compiler._component_trace.resource_generalization import (
        BookkeepingCall,
        CliffordCall,
        ConjunctionCall,
        PhaseCall,
        RotationCall,
    )

    h = CliffordCall("h", 1)
    assert generalize_thc_resource_call(h) == CliffordCall("arbitrary", 1)
    assert h.opcode == "h"
    assert generalize_thc_resource_call(CliffordCall("cz", 2)) == CliffordCall(
        "arbitrary", 2
    )
    assert generalize_thc_resource_call(PhaseCall("s", adjoint=True)) == CliffordCall(
        "arbitrary", 1
    )
    assert generalize_thc_resource_call(ConjunctionCall((0, 1))) == ConjunctionCall(
        (None, None)
    )
    assert generalize_thc_resource_call(RotationCall("ry", 0.25)) == RotationCall(
        "ry", None
    )
    for kind in ("allocate", "free", "split", "join"):
        assert generalize_thc_resource_call(BookkeepingCall(kind, (1,))) is None
    assert lookup_graph_label((6,), (1, 4)) == (
        "QROM(selection_bitsizes=(6,), target_bitsizes=(1, 4))"
    )
    assert lookup_graph_label((6,), (1, 4), block_sizes=(4,)) == (
        "SelectSwapQROM(target_bitsizes=(1, 4), block_sizes=(4,))"
    )

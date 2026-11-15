"""Check the pinned first-quantization tutorial's complete resource figures."""

from collections import Counter

import pytest

from examples.algorithms.first_quantization_catalog import notebook_cases
from flagquantum.compiler._component_trace.call_graph import modeled_call_graph
from flagquantum.qec.resources.estimation import estimate_ftqc_resources
from tests.unit.test_first_quantization_resource_model import assert_source_model

pytestmark = pytest.mark.unit


@pytest.mark.ftqc_reference
def test_first_quantization_tutorial_examples_match_pinned_source() -> None:
    pytest.importorskip("qualtran")
    from qualtran import BQUInt, QAny, QBit
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

    def source_dtype(port):
        if port.dtype.encoding == "uint":
            return BQUInt(port.bitsize, dict(port.dtype.parameters)["bound"])
        if port.dtype.encoding == "bit":
            return QBit()
        assert port.dtype.encoding == "opaque"
        return QAny(port.bitsize)

    for (signature, model), source in zip(
        notebook_cases().values(), originals, strict=True
    ):
        assert [
            (reg.name, reg.shape, reg.side.value, source_dtype(reg))
            for reg in signature
        ] == [
            (reg.name, reg.shape, reg.side.value, reg.dtype) for reg in source.signature
        ]
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

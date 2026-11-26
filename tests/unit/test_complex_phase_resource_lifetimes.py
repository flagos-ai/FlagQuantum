"""Phase model peaks follow original free-before-decoder source scheduling."""

from dataclasses import fields

import pytest
import sympy

from flagquantum.algorithms.primitives.complex_phase_oracle import ComplexPhaseOracle
from flagquantum.compiler._component_trace.resource_model import GateCost
from flagquantum.compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from flagquantum.core.ir.program.register_types import (
    bounded_unsigned_register_type,
    opaque_register_type,
)
from flagquantum.core.ir.program.registers import Register
from tests.unit.mean_cost_reference import source_encoder

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize(
    "forward_peak,inverse_peak,expected", [(4, 7, 7), (11, 4, 11), (2, 2, 5)]
)
def test_asymmetric_source_encoder_costs_and_every_category(
    forward_peak: int, inverse_peak: int, expected: int
) -> None:
    from qualtran.bloqs.mean_estimation.complex_phase_oracle import (
        ComplexPhaseOracle as SourcePhase,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    encoder = source_encoder(forward_peak, inverse_peak)
    models = []
    for child in (encoder, encoder.adjoint()):
        cost = get_cost_value(child, QECGatesCost())
        models.append(
            SymbolicResourceFormula(
                logical_qubits=get_cost_value(child, QubitCount()),
                **{f.name: getattr(cost, f.name) for f in fields(GateCost)},
            )
        )
    q = Register("q", bounded_unsigned_register_type(1, 2))
    t = Register("t", opaque_register_type(1))
    native = ComplexPhaseOracle((), (q,), (t,), 1)
    actual = native.source_resource_model(*models)
    source = SourcePhase(encoder, 1)
    for original in (source, source.adjoint()):
        assert (
            actual.logical_qubits == get_cost_value(original, QubitCount()) == expected
        )
        cost = get_cost_value(original, QECGatesCost())
        for field in fields(GateCost):
            assert (
                sympy.simplify(getattr(actual, field.name) - getattr(cost, field.name))
                == 0
            ), field.name

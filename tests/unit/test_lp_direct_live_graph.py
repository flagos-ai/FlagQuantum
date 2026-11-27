"""Execute unchanged LP graphs whose constructor serializer rejects scalar controls."""

from collections import Counter
from math import sqrt

import pytest

from tests._reference_dependencies import unavailable_reference

try:
    import kickmix
    import numpy as np
    from qualtran.bloqs.phase_estimation.lp_resource_state import LPResourceState
    from qualtran.resource_counting._bloq_counts import QECGatesCost
    from qualtran.resource_counting._costing import get_cost_value
    from qualtran.serialization.bloq import bloqs_to_proto
except ModuleNotFoundError as error:
    unavailable_reference(error)

from examples.algorithms.binary_field_arithmetic import FieldOperation
from flagquantum import Circuit
from flagquantum.compiler.registers.register_binding import bind_register_program
from flagquantum.core.ir import CircuitIR
from flagquantum.ecosystem.cirq._register_graph import import_register_graph
from tests.unit.measurement_certificate import certify_erasure_histories
from tests.unit.sparse_register_execution import execute_sparse_register
from tests.unit.test_live_scalar_register_graph import assert_live_calls

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("width", (1, 3, 7))
def test_original_lp_graph_all_histories_complex_state_and_clean_work(
    width: int,
) -> None:
    assert getattr(kickmix, "__version__") == "0.1" and hasattr(kickmix, "Simulator")
    source = LPResourceState(width)
    graph = source.decompose_bloq().flatten()
    # This original constructor-codec defect remains observable and unmodified.
    with pytest.raises(ValueError, match="Cannot serialize.*numpy.uint8"):
        bloqs_to_proto(graph, max_depth=1, pred=lambda _: False)
    expected = np.asarray(source.tensor_contract()).reshape(-1)
    program = import_register_graph(graph)
    assert_live_calls(program, graph)
    clean = tuple(range(1, 3 * width + 13))
    bound = bind_register_program(
        program, (), clean=clean, allow_workspace_outputs=True
    )
    assert len(bound.outputs) == 1 and len(bound.outputs[0]) == width
    total = bound.circuit.n_wires
    ir = (
        Circuit(total).append(FieldOperation(total, bound.circuit.instructions)).to_ir()
    )
    ir = CircuitIR.from_json(ir.to_json())
    events = certify_erasure_histories(
        ir.instructions, n_wires=total, zero_wires=clean, check_clean_output=False
    )
    counts = Counter(g.name for g in ir.instructions)
    model = get_cost_value(source, QECGatesCost())
    assert events == counts["measure"] == width + 1
    assert counts["t"] + counts["tdg"] == 4 * (width + 1)
    assert model.and_bloq == width + 1 and model.rotation == 6 * (width + 1)
    spectator = (sqrt(0.3), 1j * sqrt(0.7))
    actual = execute_sparse_register(
        ir, {0: spectator[0], 1 << (total - 1): spectator[1]}, seed=width
    )
    target = {}
    output_mask = sum(1 << (total - wire - 1) for wire in bound.outputs[0])
    for external, coefficient in enumerate(spectator):
        for word, amplitude in enumerate(expected):
            address = external << (total - 1)
            address |= sum(
                ((word >> (width - bit - 1)) & 1) << (total - wire - 1)
                for bit, wire in enumerate(bound.outputs[0])
            )
            target[address] = complex(coefficient * amplitude)
    assert (
        max(
            abs(actual.get(k, 0) - target.get(k, 0))
            for k in actual.keys() | target.keys()
        )
        < 3e-12
    )
    work_mask = ((1 << (total - 1)) - 1) ^ output_mask
    assert sum(abs(a) ** 2 for k, a in actual.items() if k & work_mask) < 1e-27
    assert sum(abs(a) ** 2 for a in actual.values()) == pytest.approx(1, abs=3e-12)

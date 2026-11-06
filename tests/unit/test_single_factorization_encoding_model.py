"""Verify both locked SF encoding resource graphs and source boundaries."""

from dataclasses import asdict

import pytest

from flagquantum.algorithms.primitives.single_factorization_encoding_model import (
    source_single_factorization_block_encoding_cost,
    source_single_factorization_one_body_cost,
)
from flagquantum.compiler._component_trace.resource_model import (
    estimate_component_resources,
)

pytestmark = pytest.mark.unit


def _assert_source_graph(model, source):
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    assert model.qubits == source.signature.n_qubits()
    instructions = tuple(source.decompose_bloq().iter_bloqsoqs())
    calls = tuple(edge for edge in model.calls for _ in range(edge.repetitions))
    assert [edge.component.name for edge in calls] == [
        type(instance.bloq).__name__ for instance, _, _ in instructions
    ]
    for candidate, reference in (
        (model, source),
        *(
            (edge.component, instruction[0].bloq)
            for edge, instruction in zip(calls, instructions, strict=True)
        ),
    ):
        native = estimate_component_resources(candidate)
        expected = get_cost_value(reference, QECGatesCost())
        assert {
            key: value for key, value in asdict(native.gates).items() if value
        } == expected.asdict()
        assert native.peak_logical_qubits == get_cost_value(reference, QubitCount())
    return instructions


@pytest.mark.ftqc_reference
@pytest.mark.parametrize("forward,inverse", [((1, 1), (1, 1)), ((4, 8), (2, 4))])
@pytest.mark.parametrize("adjoint", [False, True])
def test_locked_one_body_graph(forward, inverse, adjoint):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.sf.single_factorization import (
        SingleFactorizationOneBody,
    )

    source = SingleFactorizationOneBody(
        50,
        10,
        12,
        7,
        is_adjoint=adjoint,
        kp1=forward[0],
        kp2=forward[1],
        kp1_inv=inverse[0],
        kp2_inv=inverse[1],
    )
    model = source_single_factorization_one_body_cost(
        50, 10, 12, 7, forward_blocks=forward, inverse_blocks=inverse
    )
    _assert_source_graph(model, source)


@pytest.mark.ftqc_reference
@pytest.mark.parametrize(
    "orbitals,aux,alias,rotations,forward,inverse",
    [
        (10, 50, 12, (7, 7), (1, 1), (1, 1)),
        (10, 50, 12, (8, 8), (1, 1), (1, 1)),
        (108, 200, 10, (5, 7), (4, 32), (2, 256)),
    ],
)
def test_locked_full_encoding_graph(orbitals, aux, alias, rotations, forward, inverse):
    pytest.importorskip("qualtran")
    from qualtran.bloqs.chemistry.sf.single_factorization import (
        SingleFactorizationBlockEncoding,
    )

    source = SingleFactorizationBlockEncoding(
        orbitals,
        aux,
        alias,
        *rotations,
        kp1=forward[0],
        kp2=forward[1],
        kp1_inv=inverse[0],
        kp2_inv=inverse[1],
    )
    model = source_single_factorization_block_encoding_cost(
        aux,
        orbitals,
        alias,
        rotations[1],
        forward_blocks=forward,
        inverse_blocks=inverse,
    )
    instructions = _assert_source_graph(model, source)
    assert set(instructions[2][1]) == {"control", "reg0_", "reg1_", "reg2_", "reg3_"}
    assert instructions[2][1]["control"].reg.name == "succ_l"
    with pytest.raises(NotImplementedError):
        _ = source.alpha
    with pytest.raises(NotImplementedError):
        _ = source.epsilon

"""Compare the source controlled q-array CY calls and every raw history."""

from collections import Counter
from hashlib import sha1
from importlib import import_module
from pathlib import Path

import pytest
import torch

from tests._reference_dependencies import unavailable_reference

try:
    import_module("numpy")
    import cirq

    import_module("qualtran")
    import_module("kickmix")
except ModuleNotFoundError as _reference_import_error:
    unavailable_reference(_reference_import_error)

from flagquantum.compiler.registers.primitives.controlled_y_component import (
    controlled_cirq_y_program,
    controlled_scalar_y_program,
)
from tests.unit.test_chebyshev_component import assert_all_small_raw_histories
from tests.unit.test_component_hierarchy_control import compare_layers
from tests.unit.test_default_linear_combination_oracles import bind
from tests.unit.test_generic_controlled_word_components import (
    frozen_generic_controls as frozen_generic_controls,
)
from tests.unit.test_momentum_source_declarations import frozen_source as frozen_source
from tests.unit.test_multi_control_component import atomic_leaves
from tests.unit.test_multiplexed_source_binding import logical_operator

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.fixture(scope="module", autouse=True)
def frozen_y_and_rotation_sources() -> None:
    for name, blob in (
        ("y_gate", "a3b305679692809a00407b630e2cc9f4227caebe"),
        ("rotation", "69d2f19d636db2de0105919fb6b53421ba5b94b4"),
    ):
        module = import_module("qualtran.bloqs.basic_gates." + name)
        assert module.__file__ is not None
        raw = Path(module.__file__).read_bytes()
        assert sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == blob


@pytest.mark.parametrize("inverse", (False, True))
def test_controlled_cirq_y_source_calls_operator_and_work(
    frozen_generic_controls: None, inverse: bool
) -> None:
    from qualtran import Controlled, CtrlSpec
    from qualtran.cirq_interop import CirqGateAsBloq

    cy = cirq.Y.controlled()
    source = Controlled(CirqGateAsBloq(cy ** (-1 if inverse else 1)), CtrlSpec())
    record = controlled_cirq_y_program()
    source_calls = tuple(source.decompose_bloq().iter_bloqsoqs())
    if hasattr(cirq, "CY"):
        assert len(source_calls) == 3
        compare_layers(record, source, atomic_leaves(record))
    else:
        # Cirq 1.6's generic ControlledGate(Y) is a different six-call source
        # profile. Check its boundary explicitly, without claiming this owner
        # imports that graph or equating its analytical cost with our recipe.
        assert [type(row[0].bloq).__name__ for row in source_calls] == [
            "Controlled",
            "Controlled",
            "Toffoli",
            "Controlled",
            "Toffoli",
            "Controlled",
        ]
    bound, logical = bind(record)
    expected = torch.as_tensor(cirq.unitary(cy.controlled()), dtype=torch.complex128)
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical), expected, atol=8e-11, rtol=0
    )
    assert_all_small_raw_histories(bound.circuit, logical, expected)
    assert logical == 3 and bound.circuit.n_wires == 4
    # This is the native realization of two atomic half-X leaves, separate
    # from the upstream analytical rotation categories.
    assert Counter(g.name for g in bound.circuit.instructions) == {
        "h": 7,
        "cphase": 2,
        "t": 2,
        "tdg": 2,
        "cx": 6,
        "s": 1,
        "cz": 2,
        "measure": 1,
        "reset": 1,
    }
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    if hasattr(cirq, "CY"):
        model = get_cost_value(source, QECGatesCost())
        assert (model.and_bloq, model.clifford, model.rotation, model.measurement) == (
            1,
            2,
            2,
            1,
        )
        assert get_cost_value(source, QubitCount()) == 4


def test_scalar_cy_specialization_source_graph_all_histories_and_resources(
    frozen_generic_controls: None,
) -> None:
    from qualtran.bloqs.basic_gates import CYGate
    from qualtran.bloqs.mcmt.specialized_ctrl import (
        _MultiControlledFromSinglyControlled,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    source = _MultiControlledFromSinglyControlled(
        cvs=(1, 1), ctrl_bloq=CYGate(), ctrl_reg_name="ctrl"
    )
    record = controlled_scalar_y_program()
    compare_layers(record, source, atomic_leaves(record))
    bound, logical = bind(record)
    expected = torch.as_tensor(
        cirq.unitary(cirq.Y.controlled(num_controls=2)), dtype=torch.complex128
    )
    torch.testing.assert_close(
        logical_operator(bound.circuit, logical), expected, atol=1e-12, rtol=0
    )
    assert_all_small_raw_histories(bound.circuit, logical, expected)
    assert logical == 3 and bound.circuit.n_wires == 4
    model = get_cost_value(source, QECGatesCost())
    assert (model.and_bloq, model.clifford, model.rotation, model.measurement) == (
        1,
        2,
        0,
        1,
    )
    assert get_cost_value(source, QubitCount()) == 4

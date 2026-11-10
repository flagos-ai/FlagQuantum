"""Different allocation/free annotations do not change a wire's original pool."""

from __future__ import annotations

from collections import Counter
from itertools import product

import pytest
import torch

from flagquantum.compiler.registers.component_expansion import expand_component_calls
from tests.unit.measurement_checks import forced_branch
from tests.unit.test_component_control_selection import assert_gate_parameters
from tests.unit.test_controlled_workspace_lifetimes import PATTERNS
from tests.unit.test_serial_shaped_erasure import (
    bind_serial,
    expected_sequence,
    reference_serial,
    serial_fixture,
)
from tests.unit.test_shaped_controlled_erasure import initial_and_expected

pytestmark = [pytest.mark.unit, pytest.mark.ftqc_reference]


@pytest.mark.parametrize("values", PATTERNS)
@pytest.mark.parametrize("dirty,free_dirty", ((False, True), (True, False)))
@pytest.mark.parametrize("expanded", (False, True))
def test_mixed_lifetime_flags_keep_pool_identity_and_restore_all_history_pairs(
    values: tuple[int, ...],
    dirty: bool,
    free_dirty: bool,
    expanded: bool,
) -> None:
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value

    from examples.algorithms.binary_field_arithmetic import FieldOperation
    from flagquantum.dynamic import DynamicCircuit
    from flagquantum.runtime.dynamic.execution import run_dynamic

    n = len(values)
    forward, source_forward = serial_fixture(
        values, dirty, (0, 1), free_dirty=free_dirty
    )
    inverse, source_inverse = serial_fixture(
        values, dirty, (1, 0), inverse=True, free_dirty=free_dirty
    )
    if expanded:
        forward, _ = expand_component_calls(forward, "serial")
        inverse, _ = expand_component_calls(inverse, "serial")
    computed = bind_serial(forward, n, dirty)
    backwards = bind_serial(inverse, n, free_dirty)
    original_forward, source_forward_pool = reference_serial(source_forward, n, dirty)
    original_inverse, source_inverse_pool = reference_serial(
        source_inverse, n, free_dirty
    )
    assert_gate_parameters(computed.circuit.instructions, original_forward)
    assert_gate_parameters(backwards.circuit.instructions, original_inverse)
    life = 2 * n + 3
    for bound, events, allocate_flag, free_flag in (
        (computed, source_forward_pool, dirty, free_dirty),
        (backwards, source_inverse_pool, free_dirty, dirty),
    ):
        # Source qfree receives no dirty argument: pool provenance and the Free
        # operation's annotation are deliberately distinct evidence tracks.
        native = tuple((e.kind, e.wires, e.dirty) for e in bound.events if e.wires)
        assert [(kind, wires) for kind, wires, _ in native] == [
            (kind, wires) for kind, wires, _ in events
        ]
        assert [(kind, flag) for kind, wires, flag in native if wires == (life,)] == [
            ("allocate", allocate_flag),
            ("free", free_flag),
        ] * 2
        assert [(kind, flag) for kind, wires, flag in events if wires == (life,)] == [
            ("allocate", allocate_flag),
            ("free", allocate_flag),
        ] * 2
        assert [item for item in native if item[1] != (life,)] == [
            item for item in events if item[1] != (life,)
        ]
        measured = [g for g in bound.circuit.instructions if g.name == "measure"]
        assert [g.metadata["classical_bit"] for g in measured] == list(range(2 * n))
    assert computed.outputs == backwards.outputs
    # Either a clean Allocate or a clean Free constrains the private lifetime
    # to zero here. Arbitrary borrowed state is covered only for dirty/dirty.
    initial, _ = initial_and_expected(n, False, values, 0)
    expected = expected_sequence(initial, values, (0, 1))
    histories = tuple(product((0, 1), repeat=2 * n))
    for first in histories:
        intermediate = forced_branch(initial, computed.circuit.instructions, first)
        upstream_forward = forced_branch(initial, original_forward, first)
        torch.testing.assert_close(intermediate, upstream_forward, atol=4e-12, rtol=0)
        torch.testing.assert_close(intermediate, expected / 2**n, atol=4e-12, rtol=0)
        assert (
            torch.count_nonzero(
                intermediate.reshape((2,) * (2 * n + 5)).select(life, 1)
            )
            == 0
        )
        for second in histories:
            actual = forced_branch(intermediate, backwards.circuit.instructions, second)
            upstream = forced_branch(upstream_forward, original_inverse, second)
            torch.testing.assert_close(actual, upstream, atol=4e-12, rtol=0)
            torch.testing.assert_close(
                actual, initial / 2 ** (2 * n), atol=4e-12, rtol=0
            )
    circuit = DynamicCircuit(2 * n + 5, inputs=initial, dtype=torch.complex128)
    circuit.append(FieldOperation(2 * n + 5, computed.circuit.instructions))
    circuit.append(FieldOperation(2 * n + 5, backwards.circuit.instructions))
    result = run_dynamic(circuit, shots=8, seed=79, strategy="trajectory")
    torch.testing.assert_close(
        result.final_states.reshape(8, -1), initial.expand(8, -1), atol=4e-12, rtol=0
    )
    for source, bound in ((source_forward, computed), (source_inverse, backwards)):
        counts = Counter(g.name for g in bound.circuit.instructions)
        model = get_cost_value(source, QECGatesCost())
        assert counts["t"] + counts["tdg"] == 8 * n
        assert counts["ccx"] == 4 and counts["measure"] == counts["reset"] == 2 * n
        assert model.total_t_count() == 2 * (35 + 4 * (n - 1))
        assert (
            get_cost_value(source, QubitCount()) == bound.circuit.n_wires == 2 * n + 5
        )


@pytest.mark.parametrize("values", ((0,), (1,)))
def test_clean_free_is_an_obligation_not_a_runtime_reset(
    values: tuple[int, ...],
) -> None:
    from qualtran import QBit
    from qualtran.bloqs.bookkeeping import Free

    module, source = serial_fixture(values, True, (0, 1), free_dirty=False)
    bound = bind_serial(module, 1, True)
    original, events = reference_serial(source, 1, True)
    assert_gate_parameters(bound.circuit.instructions, original)
    initial, _ = initial_and_expected(1, True, values, 0)
    expected = expected_sequence(initial, values, (0, 1))
    for history in product((0, 1), repeat=2):
        actual = forced_branch(initial, bound.circuit.instructions, history)
        torch.testing.assert_close(actual, expected / 2, atol=4e-12, rtol=0)
        # This deliberately invalid clean-Free precondition remains invalid.
        assert torch.count_nonzero(actual.reshape((2,) * 7).select(5, 1)) > 0
    assert [(kind, flag) for kind, wires, flag in events if wires == (5,)] == [
        ("allocate", True),
        ("free", True),
    ] * 2
    with pytest.raises(ValueError, match="non-zero"):
        Free(QBit(), dirty=False).on_classical_vals(reg=1)
    assert Free(QBit(), dirty=True).on_classical_vals(reg=1) == {}

"""The locked generator context's ordered clean and borrowed views."""

from dataclasses import replace

import pytest

from flagquantum.compiler.registers.layout.workspace import (
    GenerationWorkspace,
    RegisterWorkspace,
)

pytestmark = pytest.mark.unit


def test_nested_source_context_preserves_parent_and_strided_lane_order() -> None:
    # Locked circuit_gen_ctx.h, blob ab6cc825738f96922e1a447c0ea7693aaa861ed8.
    # The source strided span visits 10, 9, 8. The newer segment visits 5, 6.
    root = GenerationWorkspace((1, 2, 3))
    parent = root.with_borrowed((10, 9, 8))
    child = parent.with_borrowed((5, 6))
    assert child.with_borrowed(()) is child
    assert parent.borrowed_count == 3
    assert child.borrowed_count == 5
    assert child.borrowed_back() == 6
    assert child.borrowed_prefix(4) == (5, 6, 10, 9)
    _, offset_view = child.take_clean(1)
    minimized = replace(offset_view, minimize_qubits=True)
    assert minimized.preferred_clean == ()
    assert offset_view.preferred_clean == (2, 3)
    taken, remainder = minimized.take_clean(1)
    assert taken == (2,)
    assert remainder.clean == (3,)
    assert remainder.borrowed_prefix(5) == (5, 6, 10, 9, 8)
    assert parent.clean == (1, 2, 3)
    assert parent.borrowed_prefix(3) == (10, 9, 8)


def test_context_rejects_short_requests_before_touching_caller_leases() -> None:
    view = GenerationWorkspace((), True).with_borrowed((6,))
    assert not view.has_no_workspace
    with pytest.raises(ValueError, match="Insufficient borrowed"):
        view.borrowed_prefix(2)
    with pytest.raises(ValueError, match="Insufficient clean"):
        view.take_clean(1)
    empty = GenerationWorkspace(())
    assert empty.has_no_workspace and empty.borrowed_count == 0
    assert empty.borrowed_prefix(0) == ()
    with pytest.raises(ValueError, match="Empty borrowed"):
        empty.borrowed_back()
    pool = RegisterWorkspace(view.clean, borrowed=view.borrowed_prefix(1))
    lease = pool.allocate(1, dirty=True)
    assert lease.wires == (6,)
    pool.free(lease)
    assert pool.allocate(1, dirty=True).wires == lease.wires

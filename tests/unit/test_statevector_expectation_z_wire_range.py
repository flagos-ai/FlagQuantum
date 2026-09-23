"""`Circuit.expectation_z` must refuse a wire outside the statevector.

The Z sign of a wire is read from bit ``n_wires - 1 - wire``. An out-of-range
wire makes that shift negative, and torch returns all ones for a negative shift,
so the wire reported ``+1`` at every step instead of being refused. A negative
wire index behaved the same way rather than denoting a wire from the end.
"""

from __future__ import annotations

import pytest
import torch

import flagquantum as fq

pytestmark = pytest.mark.unit

N_WIRES = 3
OUT_OF_RANGE_WIRES: tuple[int, ...] = (N_WIRES, N_WIRES + 1, 9, -1, -2, -(N_WIRES + 1))


def _prepared_circuit() -> fq.Circuit:
    """A circuit whose three Z expectations are -1, +1, +1 and never all equal.

    ``x(0)`` makes wire 0 disagree with the other two, so a wire that silently
    reported the wrong bit cannot be mistaken for a correct answer.
    """

    circuit = fq.Circuit(N_WIRES)
    circuit.x(0)
    return circuit


def test_the_reference_values_disagree_on_every_wire() -> None:
    """Guard the test's own assumption before it is used to judge the fix."""

    circuit = _prepared_circuit()
    assert circuit.expectation_z().tolist() == [[-1.0, 1.0, 1.0]]
    for wire in range(N_WIRES):
        assert circuit.expectation_z(wire).tolist() == [[(-1.0, 1.0, 1.0)[wire]]]


@pytest.mark.parametrize("wire", OUT_OF_RANGE_WIRES)
def test_an_out_of_range_wire_is_refused(wire: int) -> None:
    with pytest.raises(ValueError, match="wire is outside the statevector"):
        _prepared_circuit().expectation_z(wire)


@pytest.mark.parametrize("wire", OUT_OF_RANGE_WIRES)
def test_an_out_of_range_wire_used_to_report_a_positive_expectation(wire: int) -> None:
    """The defect this guards against, stated as the value that was returned.

    On the unfixed code every one of these wires produced ``+1.0`` on all three
    wires -- a plausible number that no caller could distinguish from a real
    measurement.
    """

    with pytest.raises(ValueError):
        _prepared_circuit().expectation_z(wire)

    # The unfixed answer, kept explicit so the regression is legible.
    silently_positive = torch.ones(1, 1)
    assert silently_positive.tolist() == [[1.0]]


def test_an_out_of_range_wire_inside_a_selection_is_refused() -> None:
    circuit = _prepared_circuit()
    for wires in ([0, N_WIRES], [N_WIRES, 0], [-1], [0, 1, -1], (2, 9)):
        with pytest.raises(ValueError, match="wire is outside the statevector"):
            circuit.expectation_z(wires)


def test_every_in_range_wire_still_evaluates() -> None:
    """The guard must not reject a wire the statevector does have."""

    circuit = _prepared_circuit()
    assert circuit.expectation_z(0).tolist() == [[-1.0]]
    assert circuit.expectation_z(1).tolist() == [[1.0]]
    assert circuit.expectation_z(2).tolist() == [[1.0]]
    assert circuit.expectation_z().tolist() == [[-1.0, 1.0, 1.0]]
    assert circuit.expectation_z([0, 2]).tolist() == [[-1.0, 1.0]]
    assert circuit.expectation_z((2, 2)).tolist() == [[1.0, 1.0]]


def test_a_single_wire_circuit_still_works() -> None:
    """The boundary case: the only valid wire is 0."""

    circuit = fq.Circuit(1)
    assert circuit.expectation_z().tolist() == [[1.0]]
    assert circuit.expectation_z(0).tolist() == [[1.0]]
    with pytest.raises(ValueError, match="wire is outside the statevector"):
        circuit.expectation_z(1)
    with pytest.raises(ValueError, match="wire is outside the statevector"):
        circuit.expectation_z(-1)

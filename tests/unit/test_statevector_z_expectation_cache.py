"""Unit tests for the Z-expectation table and its ceiling.

``Circuit.expectation_z`` used to reach every per-wire Z expectation through a
``(2 ** n_wires, k)`` float32 sign table cached on the circuit, one table per
requested wire set. That table is ten times the state at twenty wires, and the
requests that fill the cache are narrow ones - nineteen single-wire keys measure
76 MiB - so the change here is a ceiling on what stays resident plus a narrower
path that needs no table at all.

Five things are pinned, and the second is what makes the rest worth anything:

* the table path is unchanged, including the two edges it has always had: an
  empty request raises, and a wire the circuit does not have returns one rather
  than raising;
* the ceiling is a guard rail, not a tuning knob, so the request pattern the
  cache was measured on stays entirely resident and no reachable request changes
  behaviour;
* the narrow path is exact where it claims to be, returns the order it was asked
  for, repeats a repeated wire, and carries the same gradient;
* the narrow path is the path actually taken, observed through the cache being
  left empty rather than by reading the output, since both paths agree to within
  rounding;
* a request wider than the narrow path's limit still uses the table even with the
  switch on.
"""

from __future__ import annotations

import pytest
import torch

import flagquantum as fq
import flagquantum.simulation.statevector.local as local
from flagquantum.simulation.statevector.operations import (
    _DEFAULT_Z_SIGN_CACHE_BYTES,
    _Z_MARGINAL_MAXIMUM_WIRES,
    _z_sign_cache_byte_limit,
)

pytestmark = pytest.mark.unit

N_WIRES = 6


def _circuit(*, wires: int = N_WIRES, bsz: int = 1) -> fq.Circuit:
    """A circuit with correlations across every wire, so no wire is trivial."""

    circuit = fq.Circuit(wires, bsz=bsz)
    for wire in range(wires):
        circuit.h(wire)
        circuit.ry(wire, 0.13 * (wire + 1))
    for wire in range(wires - 1):
        circuit.cx(wire, wire + 1)
    circuit.state(refresh=True)
    return circuit


def _table_expectations(circuit: fq.Circuit, wires: tuple[int, ...]) -> torch.Tensor:
    """The pre-change arithmetic, spelled out rather than called."""

    probabilities = torch.abs(circuit.state()) ** 2
    basis = torch.arange(probabilities.shape[-1], dtype=torch.int64)
    signs = torch.stack(
        tuple(1 - 2 * ((basis >> (circuit.n_wires - 1 - wire)) & 1) for wire in wires),
        dim=-1,
    ).to(dtype=probabilities.dtype)
    return probabilities @ signs


def _resident_bytes(circuit: fq.Circuit) -> int:
    return sum(
        tensor.element_size() * tensor.nelement()
        for tensor in circuit._statevector_z_signs.values()
    )


# --- the table path is unchanged -------------------------------------------------


@pytest.mark.parametrize(
    "wires",
    [None, (0,), (2, 0), (1, 1, 0), (0, 1, 2, 3), tuple(range(N_WIRES))],
)
def test_the_table_path_is_bitwise_identical_to_the_pre_switch_arithmetic(
    monkeypatch: pytest.MonkeyPatch, wires: tuple[int, ...] | None
) -> None:
    monkeypatch.delenv("FQ_CPU_Z_MARGINAL", raising=False)
    circuit = _circuit()
    requested = tuple(range(N_WIRES)) if wires is None else wires
    result = circuit.expectation_z(wires)
    assert torch.equal(result, _table_expectations(circuit, requested))


def test_the_table_path_keeps_the_empty_request_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FQ_CPU_Z_MARGINAL", raising=False)
    with pytest.raises(RuntimeError):
        _circuit().expectation_z(())


def test_the_table_path_keeps_an_out_of_range_wire_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wire the circuit does not have has always returned one, not raised."""

    monkeypatch.delenv("FQ_CPU_Z_MARGINAL", raising=False)
    circuit = _circuit()
    result = circuit.expectation_z((N_WIRES,))
    assert result.shape == (1, 1)
    assert float(result[0, 0]) == pytest.approx(1.0, abs=1e-6)
    assert float(_table_expectations(circuit, (N_WIRES,))[0, 0]) == float(result[0, 0])


def test_a_repeated_wire_repeats_its_column(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FQ_CPU_Z_MARGINAL", raising=False)
    circuit = _circuit()
    repeated = circuit.expectation_z((1, 1, 0))
    assert repeated.shape == (1, 3)
    assert float(repeated[0, 0]) == float(repeated[0, 1])
    # A three-column matmul and a one-column one do not accumulate in the same
    # order even in the table path, so the third column is compared to its own
    # single-wire request within rounding rather than exactly.
    assert float(repeated[0, 2]) == pytest.approx(
        float(circuit.expectation_z((0,))[0, 0]), abs=1e-6
    )


# --- the ceiling is a guard rail -------------------------------------------------


def test_the_measured_request_pattern_stays_entirely_resident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nineteen single-wire keys at twenty wires are 76 MiB, well under it."""

    monkeypatch.delenv("FQ_CPU_Z_SIGN_CACHE_BYTES", raising=False)
    monkeypatch.delenv("FQ_CPU_Z_MARGINAL", raising=False)
    circuit = _circuit(wires=12)
    for wire in range(1, 12):
        circuit.expectation_z((wire,))
    assert len(circuit._statevector_z_signs) == 11
    assert _resident_bytes(circuit) == 11 * (1 << 12) * 4


def test_the_cache_evicts_the_least_recently_used_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FQ_CPU_Z_MARGINAL", raising=False)
    entry_bytes = (1 << N_WIRES) * 2 * 4
    monkeypatch.setenv("FQ_CPU_Z_SIGN_CACHE_BYTES", str(entry_bytes * 3))
    circuit = _circuit()
    for first, second in ((0, 1), (1, 2), (2, 3)):
        circuit.expectation_z((first, second))
    circuit.expectation_z((0, 1))  # touch the oldest entry
    circuit.expectation_z((3, 4))
    keys = [key[0] for key in circuit._statevector_z_signs]
    assert keys == [(2, 3), (0, 1), (3, 4)]
    assert _resident_bytes(circuit) <= entry_bytes * 3


def test_the_cache_never_exceeds_its_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FQ_CPU_Z_MARGINAL", raising=False)
    entry_bytes = (1 << N_WIRES) * 2 * 4
    monkeypatch.setenv("FQ_CPU_Z_SIGN_CACHE_BYTES", str(entry_bytes * 2))
    circuit = _circuit()
    for first in range(N_WIRES - 1):
        circuit.expectation_z((first, first + 1))
        assert _resident_bytes(circuit) <= entry_bytes * 2


def test_a_table_larger_than_the_ceiling_is_used_but_not_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FQ_CPU_Z_MARGINAL", raising=False)
    monkeypatch.setenv("FQ_CPU_Z_SIGN_CACHE_BYTES", "0")
    circuit = _circuit()
    first = circuit.expectation_z((0, 1))
    assert circuit._statevector_z_signs == {}
    assert torch.equal(first, circuit.expectation_z((0, 1)))


def test_a_malformed_ceiling_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_CPU_Z_SIGN_CACHE_BYTES", "not a number")
    assert _z_sign_cache_byte_limit() == _DEFAULT_Z_SIGN_CACHE_BYTES
    monkeypatch.delenv("FQ_CPU_Z_SIGN_CACHE_BYTES", raising=False)
    assert _z_sign_cache_byte_limit() == _DEFAULT_Z_SIGN_CACHE_BYTES


# --- the narrow path -------------------------------------------------------------


@pytest.mark.parametrize("width", range(1, _Z_MARGINAL_MAXIMUM_WIRES + 1))
def test_a_narrow_request_reads_the_marginal_and_leaves_the_cache_empty(
    monkeypatch: pytest.MonkeyPatch, width: int
) -> None:
    monkeypatch.delenv("FQ_CPU_Z_SIGN_CACHE_BYTES", raising=False)
    wires = tuple(range(width))
    # Two wires wider than the request, so the request never asks for the whole
    # state and the reduction has a complement to sum over.
    off = _circuit(wires=width + 2).expectation_z(wires)
    monkeypatch.setenv("FQ_CPU_Z_MARGINAL", "1")
    circuit = _circuit(wires=width + 2)
    on = circuit.expectation_z(wires)
    assert circuit._statevector_z_signs == {}
    assert on.shape == off.shape
    torch.testing.assert_close(on, off, rtol=0, atol=1e-5)


def test_a_request_over_the_whole_state_is_still_reduced_without_a_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The complement is empty here, so the reduction is the whole tensor."""

    monkeypatch.setenv("FQ_CPU_Z_MARGINAL", "1")
    circuit = _circuit(wires=4)
    torch.testing.assert_close(
        circuit.expectation_z(),
        _table_expectations(circuit, tuple(range(4))),
        rtol=0,
        atol=1e-5,
    )
    assert circuit._statevector_z_signs == {}


def test_a_request_wider_than_the_limit_still_uses_the_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_CPU_Z_MARGINAL", "1")
    monkeypatch.delenv("FQ_CPU_Z_SIGN_CACHE_BYTES", raising=False)
    circuit = _circuit(wires=_Z_MARGINAL_MAXIMUM_WIRES + 2)
    circuit.expectation_z(tuple(range(_Z_MARGINAL_MAXIMUM_WIRES + 1)))
    assert len(circuit._statevector_z_signs) == 1


def test_the_marginal_path_returns_the_request_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_CPU_Z_MARGINAL", "1")
    circuit = _circuit()
    ascending = circuit.expectation_z((0, 2))
    descending = circuit.expectation_z((2, 0))
    assert torch.equal(descending, ascending.flip(-1))
    assert not torch.equal(descending, ascending)


def test_the_marginal_path_repeats_a_repeated_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_CPU_Z_MARGINAL", "1")
    circuit = _circuit()
    repeated = circuit.expectation_z((1, 1, 0))
    assert repeated.shape == (1, 3)
    assert float(repeated[0, 0]) == float(repeated[0, 1])
    torch.testing.assert_close(
        repeated, _table_expectations(circuit, (1, 1, 0)), rtol=0, atol=1e-5
    )


def test_the_marginal_path_reads_each_wire_off_its_own_slice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A superposition wire is zero, a flipped wire is minus one, a bare one is one."""

    monkeypatch.setenv("FQ_CPU_Z_MARGINAL", "1")
    circuit = fq.Circuit(4).h(0).x(1).cx(1, 2)
    circuit.state(refresh=True)
    result = circuit.expectation_z((0, 1, 3))
    assert float(result[0, 0]) == pytest.approx(0.0, abs=1e-6)
    assert float(result[0, 1]) == pytest.approx(-1.0, abs=1e-6)
    assert float(result[0, 2]) == pytest.approx(1.0, abs=1e-6)


def test_a_batch_of_states_is_reduced_per_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FQ_CPU_Z_MARGINAL", "1")
    circuit = _circuit(bsz=2)
    result = circuit.expectation_z((0, 1))
    assert result.shape == (2, 2)
    torch.testing.assert_close(
        result, _table_expectations(circuit, (0, 1)), rtol=0, atol=1e-5
    )


def test_the_marginal_path_keeps_the_empty_request_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_CPU_Z_MARGINAL", "1")
    with pytest.raises(RuntimeError):
        _circuit().expectation_z(())


def test_the_marginal_path_keeps_an_out_of_range_wire_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FQ_CPU_Z_MARGINAL", "1")
    circuit = _circuit()
    result = circuit.expectation_z((N_WIRES,))
    assert result.shape == (1, 1)
    assert float(result[0, 0]) == pytest.approx(1.0, abs=1e-6)


def test_the_marginal_path_carries_the_same_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The table path is differentiated in training loops, so this is not optional."""

    def gradient(*, switch_on: bool) -> torch.Tensor:
        if switch_on:
            monkeypatch.setenv("FQ_CPU_Z_MARGINAL", "1")
        else:
            monkeypatch.delenv("FQ_CPU_Z_MARGINAL", raising=False)
        angles = torch.tensor([0.3, 0.9, 1.4], requires_grad=True)
        circuit = fq.Circuit(3)
        for wire in range(3):
            circuit.ry(wire, angles[wire])
        circuit.expectation_z((2, 0)).sum().backward()
        return angles.grad.detach().clone()

    torch.testing.assert_close(gradient(switch_on=True), gradient(switch_on=False))


def test_the_marginal_helper_agrees_with_a_hand_computed_distribution() -> None:
    """Two wires, four amplitudes, expectations read off by hand."""

    probabilities = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    wire_one = local._z_expectations_from_marginal(probabilities, (1,), n_wires=2)
    wire_zero = local._z_expectations_from_marginal(probabilities, (0,), n_wires=2)
    both = local._z_expectations_from_marginal(probabilities, (1, 0), n_wires=2)
    assert float(wire_one[0, 0]) == pytest.approx((0.1 + 0.3) - (0.2 + 0.4))
    assert float(wire_zero[0, 0]) == pytest.approx((0.1 + 0.2) - (0.3 + 0.4))
    # The differences are summed in float32, so the hand computation is a
    # reference to within rounding rather than to the last bit.
    torch.testing.assert_close(both, torch.tensor([[-0.2, -0.4]]), rtol=0, atol=1e-7)

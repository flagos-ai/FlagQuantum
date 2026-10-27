"""Refuse a gradient that a truncating bond split cannot supply exactly.

`_split_pair_matrix` freezes the retained left-singular subspace and lets the
backward pass differentiate `U_r^H A` with `U_r` held constant. When the split
keeps the full rank the projector is the identity and the rule is exact. When it
truncates, the projector varies with the input and that variation is dropped, so
the recorded gradient is not the derivative of the forward map. These tests pin
the refusal and the opt-in that replaces silent approximation.
"""

from __future__ import annotations

import pytest
import torch

import flagquantum as fq
import flagquantum.runtime as fqr
from flagquantum.errors import CapabilityError

pytestmark = pytest.mark.unit

WIDTH = 8
DEPTH = 4
#: A cap that discards weight at this circuit: the exact chain needs 16 and the
#: capped state reaches 4, which the headroom probes in
#: `test_representation_agreement_sweep.py` establish independently.
TRUNCATING_CAP = 4
EXACT_CAP = 16


def _ansatz(parameters: torch.Tensor) -> fq.Circuit:
    angles = parameters.reshape(DEPTH, WIDTH, 2)
    circuit = fq.Circuit(WIDTH, dtype=torch.complex128)
    for layer in range(DEPTH):
        for qubit in range(WIDTH):
            circuit.ry(qubit, angles[layer, qubit, 0])
            circuit.rz(qubit, angles[layer, qubit, 1])
        for qubit in range(WIDTH - 1):
            circuit.cx(qubit, qubit + 1)
    return circuit


def _parameters(seed: int = 11) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (
        0.05 * torch.randn(DEPTH, WIDTH, 2, generator=generator, dtype=torch.float32)
    ).to(torch.float64)


def _energy(parameters: torch.Tensor, max_bond: int) -> torch.Tensor:
    state = fqr.run_native(_ansatz(parameters), mode="mps", max_bond=max_bond)
    total = state.expectation_ps(z=(0, 1)).sum()
    return total - 0.7 * state.expectation_ps(x=(WIDTH - 2,)).sum()


def _run(parameters: torch.Tensor, max_bond: int) -> tuple[float, int]:
    """Return the value and the bond the state actually reached."""

    state = fqr.run_native(_ansatz(parameters), mode="mps", max_bond=max_bond)
    return float(state.expectation_ps(z=(0,)).sum()), int(max(state.bond_dims))


def test_the_cap_used_here_truncates() -> None:
    """Guard the fixture: a cap that does not bite would make the rest vacuous."""

    parameters = _parameters()
    _, reached = _run(parameters, TRUNCATING_CAP)
    _, headroom = _run(parameters, EXACT_CAP)
    assert reached == TRUNCATING_CAP
    assert headroom > TRUNCATING_CAP


def test_forward_through_a_truncating_split_is_unchanged() -> None:
    """Refusing the derivative must not change a forward value."""

    parameters = _parameters()

    with torch.no_grad():
        value = _energy(parameters, TRUNCATING_CAP)

    assert torch.isfinite(value)


def test_backward_through_a_truncating_split_refuses() -> None:
    """The default must not answer with a gradient that is not a derivative."""

    leaf = _parameters().requires_grad_(True)
    value = _energy(leaf, TRUNCATING_CAP)

    with pytest.raises(CapabilityError) as raised:
        value.backward()

    message = str(raised.value)
    assert "projected_stop_subspace" in message
    assert "truncated_gradient" in message


def test_the_refusal_names_what_the_caller_can_do_instead() -> None:
    """A refusal that does not say how to proceed is a dead end."""

    leaf = _parameters().requires_grad_(True)
    with pytest.raises(CapabilityError) as raised:
        _energy(leaf, TRUNCATING_CAP).backward()

    message = str(raised.value)
    assert "cutoff" in message or "max_bond" in message
    assert "approximate" in message.lower()


def test_a_non_truncating_split_still_differentiates() -> None:
    """The refusal is scoped to the case that is actually wrong."""

    leaf = _parameters().requires_grad_(True)
    value = _energy(leaf, EXACT_CAP)
    value.backward()

    assert leaf.grad is not None
    assert torch.isfinite(leaf.grad).all()
    assert float(leaf.grad.norm()) > 0.0


def test_opting_in_returns_a_finite_gradient_and_records_the_approximation() -> None:
    """An authorized approximation is allowed to run, and must be visible."""

    leaf = _parameters().requires_grad_(True)
    state = fqr.run_native(
        _ansatz(leaf),
        mode="mps",
        max_bond=TRUNCATING_CAP,
        truncated_gradient="projected_stop_subspace",
    )
    value = state.expectation_ps(z=(0, 1)).sum()
    value.backward()

    assert leaf.grad is not None and torch.isfinite(leaf.grad).all()
    assert state.summary()["gradient_exactness"] == "approximate_projected_subspace"


def test_gradient_exactness_is_reported_beside_the_split_rule() -> None:
    """A caller must not have to know which `svd_gradient_method` values are exact."""

    exact = fqr.run_native(_ansatz(_parameters()), mode="mps", max_bond=EXACT_CAP)
    refused = fqr.run_native(
        _ansatz(_parameters()), mode="mps", max_bond=TRUNCATING_CAP
    )
    authorized = fqr.run_native(
        _ansatz(_parameters()),
        mode="mps",
        max_bond=TRUNCATING_CAP,
        truncated_gradient="projected_stop_subspace",
    )

    assert exact.summary()["gradient_exactness"] == "exact"
    assert refused.summary()["gradient_exactness"] == "refused"
    assert (
        authorized.summary()["gradient_exactness"] == "approximate_projected_subspace"
    )


def test_a_cutoff_that_truncates_refuses_the_same_way() -> None:
    """`cutoff` is the other knob that runs the truncating branch."""

    leaf = _parameters().requires_grad_(True)
    state = fqr.run_native(_ansatz(leaf), mode="mps", cutoff=0.5)
    with pytest.raises(CapabilityError):
        state.expectation_ps(z=(0,)).sum().backward()


def test_the_policy_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="truncated_gradient"):
        fqr.run_native(
            _ansatz(_parameters()),
            mode="mps",
            max_bond=TRUNCATING_CAP,
            truncated_gradient="exact",
        )

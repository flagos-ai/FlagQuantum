"""Gate parameters must resolve to exactly one row per batch entry.

A batched circuit carries ``bsz`` amplitudes, so the kernel multiplies ``bsz`` gate
matrices and needs one parameter row per entry. A value that cannot become that row
used to reach the kernel and fail there, as a shape complaint about a tensor the
caller never wrote. These tests pin the accepted layouts and the public error.
"""

import pytest
import torch

import flagquantum as fq
from flagquantum.errors import ValidationError

pytestmark = pytest.mark.unit

BSZ = 3
ANGLES = [0.1, 0.2, 0.3]
MODES = ("statevector", "mps", "tensor_network")


def _z_expectation(circuit, mode: str = "statevector") -> torch.Tensor:
    result = fq.run(
        circuit,
        outputs=fq.expectation(fq.Z(0)),
        options=fq.ExecutionOptions(mode=mode),
    )
    return result.expectation().reshape(-1)


def _ry_circuit(theta, bsz: int = BSZ):
    return fq.Circuit(2, bsz=bsz).ry(0, theta).cx(0, 1)


@pytest.mark.parametrize("mode", MODES)
def test_batched_parameter_vector_is_used_row_by_row(mode: str) -> None:
    """One vector of ``bsz`` angles, one angle per batch entry."""

    theta = torch.tensor(ANGLES)
    batched = _z_expectation(_ry_circuit(theta), mode=mode)

    torch.testing.assert_close(
        batched, torch.cos(theta), rtol=1e-6, atol=1e-6, check_dtype=False
    )


def test_batched_parameter_vector_agrees_with_independent_single_sample_runs() -> None:
    """Cross-check against the unbatched path instead of only against the formula."""

    theta = torch.tensor(ANGLES)
    batched = _z_expectation(_ry_circuit(theta))

    per_sample = torch.tensor(
        [_z_expectation(_ry_circuit(angle, bsz=1))[0] for angle in ANGLES]
    )
    torch.testing.assert_close(batched, per_sample, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("value", [0.1, torch.tensor(0.1), torch.tensor([0.1])])
def test_scalar_parameter_broadcasts_over_the_batch(value) -> None:
    """A scalar angle applies to every batch entry."""

    batched = _z_expectation(_ry_circuit(value))

    assert batched.shape == (BSZ,)
    torch.testing.assert_close(
        batched, torch.full((BSZ,), float(torch.cos(torch.tensor(0.1))))
    )


@pytest.mark.parametrize("length", [2, 4])
def test_parameter_length_must_be_one_or_the_batch_size(length: int) -> None:
    """A short or long vector is refused as a validation error, not an execution one."""

    with pytest.raises(ValidationError) as captured:
        _z_expectation(_ry_circuit(torch.arange(length, dtype=torch.float32)))

    assert captured.value.category == "validation"
    assert "Gate 'ry' parameter 'theta' has length" in str(captured.value)
    assert f"expected 1 or the batch size {BSZ}" in str(captured.value)


@pytest.mark.parametrize("shape", [(3, 1), (1, 1), (2, 1), (1, 3), (3, 2)])
def test_two_dimensional_parameter_is_refused(shape: tuple[int, ...]) -> None:
    """A matrix of angles has no single reading for a flat batch, so it is refused."""

    with pytest.raises(ValidationError) as captured:
        _z_expectation(_ry_circuit(torch.ones(*shape)))

    assert captured.value.category == "validation"
    assert f"Gate 'ry' parameter 'theta' has shape ({shape[0]}, {shape[1]})" in str(
        captured.value
    )
    assert f"vector of length {BSZ} for a batch of {BSZ}" in str(captured.value)


def test_multi_parameter_gate_accepts_vectors_and_scalars() -> None:
    """Each parameter of a multi-parameter gate follows the same layout rule."""

    vectors = _z_expectation(
        fq.Circuit(2, bsz=BSZ).u3(
            0,
            theta=torch.tensor(ANGLES),
            phi=torch.tensor([0.4, 0.5, 0.6]),
            lbd=torch.tensor([0.7, 0.8, 0.9]),
        )
    )
    torch.testing.assert_close(
        vectors,
        torch.cos(torch.tensor(ANGLES)),
        rtol=1e-6,
        atol=1e-6,
        check_dtype=False,
    )

    mixed = _z_expectation(
        fq.Circuit(2, bsz=BSZ).u3(
            0, theta=0.1, phi=torch.tensor([0.4, 0.5, 0.6]), lbd=0.3
        )
    )
    torch.testing.assert_close(
        mixed, torch.full((BSZ,), float(torch.cos(torch.tensor(0.1))))
    )


def test_multi_parameter_gate_refuses_mixed_lengths() -> None:
    """Two parameters cannot disagree about the batch they describe."""

    with pytest.raises(ValidationError) as captured:
        _z_expectation(
            fq.Circuit(2, bsz=BSZ).u3(
                0, theta=torch.tensor(ANGLES), phi=torch.tensor([0.4, 0.5]), lbd=0.3
            )
        )

    assert captured.value.category == "validation"
    assert "Gate 'u3' parameter 'phi' has length 2" in str(captured.value)


def test_batched_parameter_gradient_matches_the_analytic_derivative() -> None:
    """Batching must not disturb autograd: d<Z>/dtheta is -sin(theta) per entry."""

    theta = torch.tensor(ANGLES, requires_grad=True)
    _z_expectation(_ry_circuit(theta)).sum().backward()

    assert theta.grad is not None
    torch.testing.assert_close(
        theta.grad, -torch.sin(torch.tensor(ANGLES)), rtol=1e-6, atol=1e-6
    )


def test_broadcast_parameter_gradient_sums_over_the_batch() -> None:
    """A broadcast angle is one variable feeding every entry, so its gradient sums."""

    theta = torch.tensor(0.1, requires_grad=True)
    _z_expectation(_ry_circuit(theta)).sum().backward()

    assert theta.grad is not None
    torch.testing.assert_close(
        theta.grad, torch.tensor(-BSZ * float(torch.sin(torch.tensor(0.1))))
    )

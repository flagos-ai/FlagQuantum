import pytest
import torch

from flagquantum.simulation.mps.observables import (
    mps_heisenberg_local_scan,
    mps_z_zz_local_scan,
    transfer_mps_operator_environment,
    transfer_mps_operator_right_environment,
)

pytestmark = pytest.mark.unit


def _site(bit: int) -> torch.Tensor:
    tensor = torch.zeros((1, 1, 2, 1), dtype=torch.complex128)
    tensor[0, 0, bit, 0] = 1
    return tensor


def test_local_z_zz_scan_tracks_single_and_adjacent_channels() -> None:
    base = torch.ones((1, 1, 1), dtype=torch.complex128)
    outputs = mps_z_zz_local_scan(
        (base, torch.zeros_like(base), torch.zeros_like(base), torch.zeros_like(base)),
        (_site(0), _site(1)),
        (0, 1),
        {0: (0,)},
        {1: (1,)},
        compiled=False,
    )

    torch.testing.assert_close(outputs[2].real, torch.ones_like(base.real))
    torch.testing.assert_close(outputs[3].real, -torch.ones_like(base.real))


def test_local_heisenberg_scan_matches_product_state_energy() -> None:
    base = torch.ones((1, 1, 1), dtype=torch.complex128)
    zeros = tuple(torch.zeros_like(base) for _ in range(4))
    coefficients = (
        torch.tensor([1.0, 2.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([3.0]),
    )

    outputs = mps_heisenberg_local_scan(
        (base, *zeros),
        (_site(0), _site(0)),
        (0, 1),
        coefficients,
    )

    torch.testing.assert_close(
        outputs[-1].real,
        torch.full_like(base.real, 6.0),
    )


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_heisenberg_scan_preserves_bell_correlations_and_gradients(
    dtype: torch.dtype,
) -> None:
    left = torch.zeros((1, 1, 2, 2), dtype=dtype)
    right = torch.zeros((1, 2, 2, 1), dtype=dtype)
    left[0, 0, 0, 0] = left[0, 0, 1, 1] = 2**-0.5
    right[0, 0, 0, 0] = right[0, 1, 1, 0] = 1
    base = torch.ones((1, 1, 1), dtype=dtype)
    real_dtype = base.real.dtype
    field = torch.tensor([0.3, -0.7], dtype=real_dtype, requires_grad=True)
    x = torch.tensor([2.0], dtype=real_dtype, requires_grad=True)
    y = torch.tensor([-3.0], dtype=real_dtype, requires_grad=True)
    z = torch.tensor([4.0], dtype=real_dtype, requires_grad=True)

    outputs = mps_heisenberg_local_scan(
        (base, *(torch.zeros_like(base) for _ in range(4))),
        (left, right),
        (0, 1),
        (field, x, y, z),
    )
    energy = outputs[-1].real.sum()
    energy.backward()

    torch.testing.assert_close(energy, torch.tensor(9.0, dtype=real_dtype))
    torch.testing.assert_close(field.grad, torch.zeros_like(field))
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    torch.testing.assert_close(y.grad, -torch.ones_like(y))
    torch.testing.assert_close(z.grad, torch.ones_like(z))


@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(torch.complex64, 1e-4), (torch.complex128, 1e-10)],
)
@pytest.mark.parametrize(
    ("left", "physical", "right"),
    [
        (2, 2, 1),
        (7, 4, 1),
        (3, 2, 5),
        (64, 2, 1),
        (1, 2, 4),
        (9, 3, 4),
    ],
)
def test_right_transfer_folds_the_leg_it_names_when_the_bonds_differ(
    dtype: torch.dtype,
    tolerance: float,
    left: int,
    physical: int,
    right: int,
) -> None:
    """The two bond legs of a site tensor are different legs.

    ``T[i, p, r]`` contracts the left bond ``i`` with the site to its left and
    the right bond ``r`` with the site to its right, and the reverse scan hands
    this function an environment on ``r``. The folded form reshapes the bra
    tensor by fusing one bond leg with the physical leg, so a reshaped extent
    taken from the wrong leg is only reusable while the two are equal -- and a
    chain is bond-uniform only in its interior. The rightmost tensor of the scan
    is ``(batch, chi, physical, 1)``, so the square case is the one shape the
    scan never presents first.
    """

    generator = torch.Generator().manual_seed(17)
    environment = torch.randn((1, right, right), generator=generator, dtype=dtype)
    tensor = torch.randn((1, left, physical, right), generator=generator, dtype=dtype)
    operator = torch.randn((physical, physical), generator=generator, dtype=dtype)

    expected = _reference_right_transfer(environment, tensor, operator)
    actual = transfer_mps_operator_right_environment(tensor, operator, environment)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def _reference_right_transfer(
    environment: torch.Tensor,
    tensor: torch.Tensor,
    operator: torch.Tensor,
) -> torch.Tensor:
    """The four-operand contraction the right transfer is a reassociation of."""

    return torch.einsum(
        "bipr,pq,bjqs,brs->bij",
        tensor.conj(),
        operator,
        tensor,
        environment,
    )


@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(torch.complex64, 1e-4), (torch.complex128, 1e-10)],
)
@pytest.mark.parametrize(
    ("batch", "bond", "physical"),
    [(1, 64, 2), (1, 4, 2), (3, 8, 2), (2, 16, 3)],
)
def test_right_transfer_reassociation_matches_the_four_operand_form(
    dtype: torch.dtype,
    tolerance: float,
    batch: int,
    bond: int,
    physical: int,
) -> None:
    """The folded right transfer is the same map, not a similar one.

    The tolerance is the arithmetic's own: a complex64 contraction of this shape
    accumulates ``bond`` terms per output element, so float32 rounding alone
    reaches ~1e-5. A wrong reassociation is not near that scale -- the recorded
    mis-derived form differs by O(1) relative -- so the bound still discriminates.
    """

    generator = torch.Generator().manual_seed(11)
    environment = torch.randn((batch, bond, bond), generator=generator, dtype=dtype)
    tensor = torch.randn(
        (batch, bond, physical, bond), generator=generator, dtype=dtype
    )
    operator = torch.randn((physical, physical), generator=generator, dtype=dtype)

    expected = _reference_right_transfer(environment, tensor, operator)
    actual = transfer_mps_operator_right_environment(tensor, operator, environment)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def test_right_transfer_reassociation_keeps_gradients_through_both_operands() -> None:
    """The reassociation must preserve the differentiated path, not just values."""

    batch, bond, physical = 1, 8, 2
    environment = torch.randn((batch, bond, bond), dtype=torch.complex128)

    def _loss(folded: bool) -> tuple[torch.Tensor, ...]:
        generator = torch.Generator().manual_seed(23)
        tensor = torch.randn(
            (batch, bond, physical, bond),
            generator=generator,
            dtype=torch.complex128,
            requires_grad=True,
        )
        operator = torch.randn(
            (physical, physical),
            generator=generator,
            dtype=torch.complex128,
            requires_grad=True,
        )
        if folded:
            value = transfer_mps_operator_right_environment(
                tensor, operator, environment
            )
        else:
            value = _reference_right_transfer(environment, tensor, operator)
        torch.real(value.square().sum()).backward()
        return (
            value.detach(),
            tensor.grad.detach(),
            operator.grad.detach(),
        )

    expected_value, expected_tensor_grad, expected_operator_grad = _loss(False)
    actual_value, actual_tensor_grad, actual_operator_grad = _loss(True)

    torch.testing.assert_close(actual_value, expected_value, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(
        actual_tensor_grad, expected_tensor_grad, rtol=1e-10, atol=1e-10
    )
    torch.testing.assert_close(
        actual_operator_grad, expected_operator_grad, rtol=1e-10, atol=1e-10
    )


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_left_and_right_transfers_are_adjoint_on_the_same_site(
    dtype: torch.dtype,
) -> None:
    """The two directions act as a map and its transpose on the environment.

    Both scans contract the same site operator into
    ``B[(i, j), (r, s)] = sum_{p, q} conj(T[i, p, r]) O[p, q] T[j, q, s]``; the
    left scan applies that map to the environment and the right scan applies its
    transpose. The inner-product identity below holds only if both directions
    still build the same ``B``, so it fails if an edit changes one and not the
    other -- which a value comparison against a re-typed einsum cannot catch.
    """

    generator = torch.Generator().manual_seed(5)
    tensor = torch.randn((1, 8, 2, 8), generator=generator, dtype=dtype)
    operator = torch.randn((2, 2), generator=generator, dtype=dtype)
    left_environment = torch.randn((1, 8, 8), generator=generator, dtype=dtype)
    right_environment = torch.randn((1, 8, 8), generator=generator, dtype=dtype)

    mapped = transfer_mps_operator_environment(left_environment, tensor, operator)
    transposed = transfer_mps_operator_right_environment(
        tensor, operator, right_environment
    )

    left_inner = torch.sum(mapped * right_environment)
    right_inner = torch.sum(left_environment * transposed)

    # Measured: the identity holds to 1.5e-5 in complex64 (float32 accumulation
    # over this many terms) while a mismatched ``B`` differs by 1.6e1, so this
    # bound sits four orders of magnitude below the failure signal.
    torch.testing.assert_close(left_inner, right_inner, rtol=1e-4, atol=1e-3)

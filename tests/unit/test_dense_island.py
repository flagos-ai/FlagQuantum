import torch

from flagquantum.simulation.dense_island import DenseIslandPlan, DenseIslandState
from flagquantum.simulation.statevector_ops import _apply_matrix


def test_equal_width_plan_and_zero_state():
    plan = DenseIslandPlan.equal_width(10, island_width=4, max_bond=16)
    state = DenseIslandState.zero(plan, batch_size=2)
    assert plan.intervals == ((0, 4), (4, 8), (8, 10))
    assert state.to_statevector().shape == (2, 2**10)
    assert torch.count_nonzero(state.to_statevector()[:, 0]) == 2


def test_local_gates_match_dense_statevector_and_gradient():
    plan = DenseIslandPlan.equal_width(6, island_width=3, max_bond=8)
    state = DenseIslandState.zero(plan)
    angle = torch.tensor(0.37, requires_grad=True)
    cosine, sine = torch.cos(angle / 2), torch.sin(angle / 2)
    ry = torch.stack((torch.stack((cosine, -sine)), torch.stack((sine, cosine)))).to(
        torch.complex64
    )
    cx = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=torch.complex64,
    )
    state.apply_local(ry, (1,))
    state.apply_local(cx, (1, 2))
    actual = state.to_statevector()
    reference = torch.zeros(1, 2**6, dtype=torch.complex64)
    reference[:, 0] = 1
    reference = _apply_matrix(reference, ry, (1,), 6)
    reference = _apply_matrix(reference, cx, (1, 2), 6)
    torch.testing.assert_close(actual, reference)
    torch.real(actual.sum()).backward()
    assert angle.grad is not None and torch.isfinite(angle.grad)


def test_cross_island_gate_matches_exact_statevector_and_gradient():
    plan = DenseIslandPlan.equal_width(6, island_width=3, max_bond=8)
    state = DenseIslandState.zero(plan)
    angle = torch.tensor(0.41, requires_grad=True)
    reference_angle = angle.detach().clone().requires_grad_(True)
    cosine, sine = torch.cos(angle / 2), torch.sin(angle / 2)
    ry = torch.stack((torch.stack((cosine, -sine)), torch.stack((sine, cosine)))).to(
        torch.complex64
    )
    cx = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=torch.complex64,
    )
    state.apply_local(ry, (2,))
    state.apply_local(cx, (2, 3))
    actual = state.to_statevector()
    reference = torch.zeros(1, 2**6, dtype=torch.complex64)
    reference[:, 0] = 1
    reference_cosine = torch.cos(reference_angle / 2)
    reference_sine = torch.sin(reference_angle / 2)
    reference_ry = torch.stack(
        (
            torch.stack((reference_cosine, -reference_sine)),
            torch.stack((reference_sine, reference_cosine)),
        )
    ).to(torch.complex64)
    reference = _apply_matrix(reference, reference_ry, (2,), 6)
    reference = _apply_matrix(reference, cx, (2, 3), 6)
    torch.testing.assert_close(actual, reference)
    actual_gradient = torch.autograd.grad(torch.real(actual.sum()), angle)[0]
    reference_gradient = torch.autograd.grad(
        torch.real(reference.sum()), reference_angle
    )[0]
    torch.testing.assert_close(actual_gradient, reference_gradient)
    assert state.summary()["gradient_method"] == "exact_autograd"
    assert state.merge_count == state.split_count == 1


def test_truncated_cross_island_split_records_error_and_stable_gradient():
    plan = DenseIslandPlan.equal_width(4, island_width=2, max_bond=1)
    state = DenseIslandState.zero(plan)
    angle = torch.tensor(0.37, requires_grad=True)
    c, s = torch.cos(angle / 2), torch.sin(angle / 2)
    rxx = torch.stack(
        (
            torch.stack((c, c * 0, c * 0, -1j * s)),
            torch.stack((c * 0, c, -1j * s, c * 0)),
            torch.stack((c * 0, -1j * s, c, c * 0)),
            torch.stack((-1j * s, c * 0, c * 0, c)),
        )
    ).to(torch.complex64)
    state.apply_local(rxx, (1, 2))
    loss = torch.real(state.to_statevector().sum())
    loss.backward()
    summary = state.summary()
    assert summary["gradient_method"] == "retained_subspace_adjoint"
    assert summary["truncation_error"] > 0
    assert angle.grad is not None and torch.isfinite(angle.grad)


def test_block_environment_z_zz_matches_statevector_energy_and_gradient():
    plan = DenseIslandPlan.equal_width(6, island_width=2, max_bond=8)
    state = DenseIslandState.zero(plan)
    angle = torch.tensor(0.29, requires_grad=True)
    reference_angle = angle.detach().clone().requires_grad_(True)

    def ry(value):
        c, s = torch.cos(value / 2), torch.sin(value / 2)
        return torch.stack((torch.stack((c, -s)), torch.stack((s, c)))).to(
            torch.complex64
        )

    cx = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=torch.complex64,
    )
    state.apply_local(ry(angle), (1,))
    state.apply_local(cx, (1, 2))
    state.apply_local(ry(0.7 * angle), (4,))
    state.apply_local(cx, (3, 4))
    z_weights = torch.linspace(-0.3, 0.4, 6)
    zz_weights = torch.linspace(0.2, -0.5, 5)
    actual = state.expectation_z_zz_chain(
        z_weights=z_weights, zz_weights=zz_weights
    ).sum()

    reference = torch.zeros(1, 2**6, dtype=torch.complex64)
    reference[:, 0] = 1
    reference = _apply_matrix(reference, ry(reference_angle), (1,), 6)
    reference = _apply_matrix(reference, cx, (1, 2), 6)
    reference = _apply_matrix(reference, ry(0.7 * reference_angle), (4,), 6)
    reference = _apply_matrix(reference, cx, (3, 4), 6)
    probabilities = torch.abs(reference) ** 2
    indices = torch.arange(2**6)
    expected = torch.zeros(())
    for wire, weight in enumerate(z_weights):
        z = 1 - 2 * ((indices >> (5 - wire)) & 1)
        expected = expected + weight * torch.sum(probabilities * z)
    for wire, weight in enumerate(zz_weights):
        left_z = 1 - 2 * ((indices >> (5 - wire)) & 1)
        right_z = 1 - 2 * ((indices >> (4 - wire)) & 1)
        expected = expected + weight * torch.sum(probabilities * left_z * right_z)
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual, angle)[0]
    expected_gradient = torch.autograd.grad(expected, reference_angle)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-5, rtol=1e-5)

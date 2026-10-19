from __future__ import annotations

import itertools
import math

import pytest
import torch

from flagquantum import Circuit
from flagquantum.compiler import lower_noise_model
from flagquantum.core.ir import CircuitIR, Instruction
from flagquantum.noise import (
    KrausChannel,
    NoiseModel,
    bit_flip_channel,
    two_qubit_depolarizing_channel,
)
from flagquantum.simulation.statevector.noisy import (
    apply_amplitude_damping_batched,
    apply_kraus_batched,
    apply_matrix_batched,
    expectation_z,
    run_noisy_trajectory_batch,
)

pytestmark = pytest.mark.unit

_BASIS_ZERO = torch.tensor([[1.0, 0.0]], dtype=torch.complex64)


def _generators(count: int) -> list[torch.Generator]:
    return [torch.Generator().manual_seed(index) for index in range(count)]


def _rotation(angle: float) -> torch.Tensor:
    """Return the real rotation that takes ``|0>`` to ``cos(angle/2)|0> + ...``."""

    cosine = math.cos(angle / 2)
    sine = math.sin(angle / 2)
    return torch.tensor([[cosine, -sine], [sine, cosine]], dtype=torch.complex64)


def _channel_ir(
    name: str, operators: tuple[torch.Tensor, ...], n_wires: int
) -> CircuitIR:
    """Lower a channel alone, the way ``lower_noise_model`` writes one."""

    return CircuitIR(
        n_wires=n_wires,
        instructions=(
            Instruction(
                name,
                tuple(range(n_wires)),
                matrix=operators,
                metadata={"is_channel": True},
            ),
        ),
    )


def _hoeffding(tolerance: float, count: int, span: float = 2.0) -> float:
    """Return an upper bound on the chance a bounded mean misses by ``tolerance``."""

    return 2 * math.exp(-2 * count * tolerance**2 / span**2)


def test_trajectory_batch_rejects_noise_channel_without_kraus_matrices() -> None:
    ir = CircuitIR(
        n_wires=1,
        instructions=(
            Instruction(
                "bit_flip",
                (0,),
                params={"probability": 0.5},
                metadata={"is_channel": True},
            ),
        ),
    )
    initial = torch.tensor([[1.0, 0.0]], dtype=torch.complex64)

    with pytest.raises(ValueError, match="bit_flip.*requires Kraus matrices"):
        run_noisy_trajectory_batch(initial, ir, _generators(1))


def test_generic_kraus_kernel_samples_and_normalizes_a_certain_branch() -> None:
    state = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]], dtype=torch.complex64)
    zero = torch.zeros((2, 2), dtype=torch.complex64)
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex64)

    evolved = apply_kraus_batched(state, (zero, x), (0,), 1, _generators(2))

    torch.testing.assert_close(
        evolved, torch.tensor([[[0, 1]], [[0, 1]]], dtype=torch.complex64)
    )
    torch.testing.assert_close(expectation_z(evolved, 1), -torch.ones((2, 1, 1)))


def test_amplitude_damping_fast_path_handles_a_certain_jump() -> None:
    state = torch.tensor([[[0.0, 1.0]], [[0.0, 1.0]]], dtype=torch.complex64)
    operators = (
        torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex64),
        torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.complex64),
    )

    evolved = apply_amplitude_damping_batched(state, operators, 0, 1, _generators(2))

    torch.testing.assert_close(
        evolved, torch.tensor([[[1, 0]], [[1, 0]]], dtype=torch.complex64)
    )
    torch.testing.assert_close(expectation_z(evolved, 1), torch.ones((2, 1, 1)))


def test_lowered_trajectory_batch_runs_without_runtime_lifecycle() -> None:
    circuit = Circuit(1).x(0)
    ir = lower_noise_model(circuit, NoiseModel().add("x", bit_flip_channel(1.0)))
    initial = torch.tensor([[1.0, 0.0]], dtype=torch.complex64)

    state, expectation, unitary, amplitude, generic = run_noisy_trajectory_batch(
        initial,
        ir,
        _generators(2),
    )

    torch.testing.assert_close(
        state, torch.tensor([[[1, 0]], [[1, 0]]], dtype=torch.complex64)
    )
    torch.testing.assert_close(expectation, torch.ones((2, 1, 1)))
    assert (unitary, amplitude, generic) == (2, 0, 0)


def test_branch_route_is_decided_by_the_operators_and_not_by_the_opcode() -> None:
    """A mixture reaches the branch route under any name, and only if it is one."""

    identity = torch.eye(2, dtype=torch.complex64)
    hadamard = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / math.sqrt(2)
    # Named after nothing in the opcode table, and its branches are not Pauli.
    named_nothing = KrausChannel(
        "check_error",
        (math.sqrt(0.7) * identity, math.sqrt(0.3) * hadamard),
    )
    # Named after a channel the opcode table does route, and not a mixture: both
    # operators are diagonal but neither is a scale times a unitary, while the
    # pair still satisfies K0^T K0 + K1^T K1 = I.
    named_bit_flip = KrausChannel(
        "bit_flip",
        (
            torch.tensor([[0.8, 0.0], [0.0, 0.6]], dtype=torch.complex64),
            torch.tensor([[0.6, 0.0], [0.0, 0.8]], dtype=torch.complex64),
        ),
    )
    # Every operator is a scale times a unitary and the scales do not square to
    # one, so this set is not a channel and has no branches to offer.
    not_trace_preserving = (
        torch.tensor([[0.9, 0.0], [0.0, 0.9]], dtype=torch.complex64),
        torch.tensor([[0.9, 0.0], [0.0, 0.9]], dtype=torch.complex64),
    )

    for channel, expected in ((named_nothing, (4, 0)), (named_bit_flip, (0, 4))):
        ir = _channel_ir(channel.name, channel.kraus, 1)

        _, _, branches, amplitude, generic = run_noisy_trajectory_batch(
            _BASIS_ZERO, ir, _generators(4)
        )

        assert channel.is_unitary_mixture is (expected[0] == 4)
        assert (branches, generic) == expected
        assert amplitude == 0

    # The kernel does not invent a mixture for a set the classifier refuses; on a
    # hand-built instruction such a set still reaches a kernel rather than being
    # refused here, which is where the refusal already happened for every
    # instruction the compiler lowers.
    with pytest.raises(ValueError, match="trace-preserving"):
        KrausChannel("bit_flip", not_trace_preserving)
    _, _, branches, _, generic = run_noisy_trajectory_batch(
        _BASIS_ZERO, _channel_ir("bit_flip", not_trace_preserving, 1), _generators(4)
    )
    assert (branches, generic) == (0, 4)


def test_branch_route_applies_a_branch_that_is_not_a_pauli_operator() -> None:
    """A Hadamard branch is applied as written, not replaced by a Pauli stand-in."""

    hadamard = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / math.sqrt(2)
    ir = _channel_ir("hadamard_error", (hadamard,), 1)

    state, expectation, branches, amplitude, generic = run_noisy_trajectory_batch(
        _BASIS_ZERO.expand(2, -1).clone(), ir, _generators(3)
    )

    # H|0> = |+>, whose Z expectation is zero; an identity stand-in would give 1.
    torch.testing.assert_close(
        state,
        torch.tensor([[[1, 1]]], dtype=torch.complex64).expand(3, 2, -1) / math.sqrt(2),
    )
    torch.testing.assert_close(expectation, torch.zeros((3, 2, 1)))
    assert (branches, amplitude, generic) == (6, 0, 0)


def test_branch_route_reaches_every_branch_of_a_four_branch_mixture() -> None:
    """The sampled mean matches the full branch set and no subset of it."""

    weights = tuple(
        item / 1.000001 for item in (0.220548, 0.212731, 0.346877, 0.219845)
    )
    angles = tuple(math.pi * item for item in (0.131588, 0.109634, 0.972849, 0.106959))
    operators = tuple(
        math.sqrt(weight) * _rotation(angle)
        for weight, angle in zip(weights, angles, strict=True)
    )
    channel = KrausChannel("four_branch", operators)
    mixture = channel.unitary_mixture
    assert mixture is not None
    assert mixture.probabilities.shape == (4,)
    torch.testing.assert_close(
        mixture.probabilities,
        torch.tensor(weights, dtype=mixture.probabilities.dtype),
        rtol=1e-5,
        atol=1e-8,
    )

    trajectories = 20000
    ir = _channel_ir(channel.name, channel.kraus, 1)
    _, expectation, branches, _, generic = run_noisy_trajectory_batch(
        _BASIS_ZERO, ir, _generators(trajectories)
    )
    assert (branches, generic) == (trajectories, 0)
    observed = float(expectation.mean())

    # Each branch is read from the classifier's own output, so a branch set the
    # classifier merged would not silently redefine the prediction.
    branch_z = [
        float(
            expectation_z(
                apply_matrix_batched(_BASIS_ZERO.unsqueeze(0), unitary, (0,), 1), 1
            )
        )
        for unitary in mixture.unitaries
    ]
    candidates = {}
    for size in range(1, len(branch_z) + 1):
        for subset in itertools.combinations(range(len(branch_z)), size):
            total = sum(weights[index] for index in subset)
            candidates[subset] = (
                sum(weights[index] * branch_z[index] for index in subset) / total
            )
    closest = min(candidates, key=lambda subset: abs(candidates[subset] - observed))

    tolerance = 0.05
    assert _hoeffding(tolerance, trajectories) < 1e-9
    assert closest == tuple(range(len(branch_z)))
    assert abs(observed - candidates[closest]) <= tolerance
    for subset, prediction in candidates.items():
        if subset != closest:
            # The nearest competitor is the three heaviest branches; the bound
            # above is well inside the gap this observation has to clear.
            assert abs(prediction - observed) > tolerance


def test_branch_route_carries_a_two_wire_channel_no_opcode_table_lists() -> None:
    """Two-qubit depolarizing is routed by its operators, not by its name."""

    channel = two_qubit_depolarizing_channel(0.0)
    ir = _channel_ir(channel.name, channel.kraus, 2)
    initial = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.complex64)

    state, _, branches, amplitude, generic = run_noisy_trajectory_batch(
        initial, ir, _generators(5)
    )

    # At probability zero the channel is the identity, so the state is the
    # evidence; the counters are the evidence for which kernel produced it.
    torch.testing.assert_close(state, initial.expand(5, -1, -1))
    assert (branches, amplitude, generic) == (5, 0, 0)

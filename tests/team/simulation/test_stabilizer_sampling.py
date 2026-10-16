"""The stabilizer engine's samples against the dense amplitude path.

`test_stabilizer_engine.py` checks the engine's own contract. This module checks
the physics, and it does it by differential comparison: the same circuit is run
through the local dense statevector path, whose amplitudes are exact, and the
sampled histogram has to agree with the distribution those amplitudes induce.
That is the only way to tell a fast sampler from a fast sampler of the wrong
circuit.

The capacity half is why the engine exists. A Clifford circuit's tableau grows
quadratically with the wire count, so the wire counts reached here are the ones
where a dense amplitude store is not merely slow but unrepresentable.
"""

from __future__ import annotations

import math
import random

import pytest
import torch

import flagquantum as fq
from flagquantum.simulation.stabilizer import sample_stabilizer

pytestmark = pytest.mark.integration

pytest.importorskip("stim")

# Wide enough that a distribution test is decisive at these shot counts, small
# enough that the dense reference is exact rather than itself statistical.
DIFFERENTIAL_SHOTS = 100_000
DIFFERENTIAL_WIRES = (3, 4, 5, 6)
# One seed for all four wire counts. It is chosen so that each circuit's exact
# law excludes at least one outcome, which is what makes the strict support check
# below non-vacuous; the test asserts that rather than assuming it.
DIFFERENTIAL_SEED = 1002
# The total-variation distance between an empirical histogram and the exact law
# has expectation near `sqrt(outcomes / (2 * pi * shots))`, which is 0.010 at
# six wires and this shot count. A 4x margin keeps the test decisive against a
# wrong distribution while leaving room for the sampling noise that a seed does
# not pin across machines.
TV_MARGIN = 0.04


def _random_clifford(n_wires: int, *, seed: int) -> fq.Circuit:
    """Return a shallow Clifford circuit on `n_wires` wires.

    Two-wire gates are drawn only between adjacent wires so that the dense
    reference stays cheap, and the layers alternate between entangling passes and
    single-wire rotations so the state is not a product state.
    """

    generator = random.Random(seed)
    circuit = fq.Circuit(n_wires)
    for _ in range(3):
        for wire in range(n_wires):
            getattr(circuit, generator.choice(("h", "s", "sdg", "x", "z", "sx")))(wire)
        for wire in range(n_wires - 1):
            getattr(circuit, generator.choice(("cx", "cz", "swap")))(wire, wire + 1)
    return circuit


def _ghz_chain(n_wires: int) -> fq.Circuit:
    """Return `H` on wire 0 followed by a CX ladder: the canonical GHZ circuit."""

    circuit = fq.Circuit(n_wires).h(0)
    for wire in range(n_wires - 1):
        circuit = circuit.cx(wire, wire + 1)
    return circuit


def _dense_distribution(circuit: fq.Circuit) -> torch.Tensor:
    """Return the exact outcome law over all wires, indexed by wire 0 as the top bit."""

    return circuit.probabilities().reshape(-1).to(torch.float64)


def _empirical_distribution(samples: torch.Tensor, n_wires: int) -> torch.Tensor:
    """Return the histogram of `samples`, in the same index order as `_dense_distribution`."""

    weights = 1 << torch.arange(
        n_wires - 1, -1, -1, dtype=torch.int64, device=samples.device
    )
    indices = (samples.to(torch.int64) * weights).sum(dim=1)
    histogram = torch.bincount(indices, minlength=2**n_wires).to(torch.float64)
    return histogram / samples.shape[0]


@pytest.mark.parametrize("n_wires", DIFFERENTIAL_WIRES)
def test_the_sampled_law_matches_the_dense_statevector_law(n_wires: int) -> None:
    """The differential test: samples have to come from the circuit's own law.

    Total variation alone could be passed by an engine that samples a state close
    to the right one, so the support is checked too: an outcome the exact law
    calls impossible must never appear. Together they rule out the two ways a
    translator goes wrong — a reversed control, which moves weight between
    outcomes, and a dropped gate, which spreads weight onto outcomes the circuit
    cannot reach.
    """

    circuit = _random_clifford(n_wires, seed=DIFFERENTIAL_SEED)
    exact = _dense_distribution(circuit)
    samples = sample_stabilizer(circuit, shots=DIFFERENTIAL_SHOTS, seed=4242)
    empirical = _empirical_distribution(samples, n_wires)

    total_variation = float((empirical - exact).abs().sum() / 2)
    assert total_variation < TV_MARGIN, total_variation

    # An exact zero is the strongest available statement, and a circuit whose law
    # excludes no outcome cannot make it. The seed is chosen so that each wire
    # count does, and the assertion says so rather than letting a full-support
    # circuit quietly make this half of the check vacuous.
    impossible = (exact == 0).nonzero().reshape(-1).tolist()
    assert (
        impossible
    ), "the circuit is not entangling enough to have impossible outcomes"
    assert float(empirical[impossible].sum()) == 0.0


def test_a_ghz_chain_samples_only_its_two_correlated_outcomes() -> None:
    """A CX ladder is where a reversed control shows up as the wrong correlation.

    The GHZ state is the sharpest small case: half the outcomes are impossible,
    and the two that remain differ in every bit, so a single reversed CX collapses
    both the support and the correlation at once.
    """

    n_wires = 12
    circuit = _ghz_chain(n_wires)
    exact = _dense_distribution(circuit)
    samples = sample_stabilizer(circuit, shots=DIFFERENTIAL_SHOTS, seed=11)
    empirical = _empirical_distribution(samples, n_wires)

    allowed = {0, 2**n_wires - 1}
    assert {int(index) for index in empirical.nonzero().reshape(-1)} == allowed
    assert float((empirical - exact).abs().sum() / 2) < TV_MARGIN

    # Every shot is uniform, which is the correlation the dense law also states.
    assert torch.equal(samples, samples[:, :1].expand_as(samples))


def test_the_engine_reaches_wire_counts_no_amplitude_store_can_hold() -> None:
    """The capacity claim, stated as a size rather than as a timing.

    A thousand-wire Clifford circuit is routine for a tableau and impossible for
    an amplitude store, and the gap is not one of degree: the assertion on the
    store size is what makes that concrete, and the correlation check is what says
    the answer is not merely fast but right. No speed is claimed here — the sample
    count is small on purpose, because the claim is capacity, not throughput.
    """

    n_wires = 1024
    circuit = _ghz_chain(n_wires)
    samples = sample_stabilizer(circuit, shots=8, seed=7)

    assert samples.shape == (8, n_wires)
    assert torch.equal(samples, samples[:, :1].expand_as(samples))
    assert {int(value) for value in samples[:, 0]} == {0, 1}

    # A `complex64` amplitude store for this many wires would need far more
    # memory than any machine has, which is why the tableau representation is not
    # an optimization of the dense path but the only representation available.
    tebibytes = 2**n_wires * 8 / 2**40
    assert tebibytes > 1_000_000


def test_the_sampled_law_is_independent_of_the_requested_wire_subset() -> None:
    """Sampling a subset must not change the law on the wires that were sampled.

    A translator that reorders or drops gates when it only needs some wires would
    keep the full-wire distribution intact and break this. The check projects the
    exact joint law onto the subset and compares, which is a different statistic
    from the full-wire comparison.
    """

    n_wires = 5
    wires = [0, 2, 4]
    circuit = _random_clifford(n_wires, seed=31337)
    exact = _dense_distribution(circuit)

    # Wire 0 is the most significant index bit, so summing out the wires that
    # were not requested is an axis reduction on the joint law.
    projected = exact.reshape([2] * n_wires)
    for axis in sorted(set(range(n_wires)) - set(wires), reverse=True):
        projected = projected.sum(dim=axis)
    projected = projected.reshape(-1)

    samples = sample_stabilizer(
        circuit, shots=DIFFERENTIAL_SHOTS, seed=808, qubits=wires
    )
    empirical = _empirical_distribution(samples, len(wires))

    assert float((empirical - projected).abs().sum() / 2) < TV_MARGIN


def test_a_stabilizer_state_samples_its_exact_deterministic_outcome() -> None:
    """A zero-variance case is the one place the exact law is a point mass.

    Total-variation thresholds cannot distinguish a sampler that is right from one
    that is nearly right when the answer is deterministic, because any deviation
    shows up immediately. This is the sharpest available check on bit order.
    """

    circuit = fq.Circuit(6).x(0).x(2).h(1).h(1).s(3).sdg(3)
    samples = sample_stabilizer(circuit, shots=16, seed=99)

    assert samples.tolist() == [[1, 0, 1, 0, 0, 0]] * 16


def test_the_sampled_shots_are_not_all_identical_on_an_entangled_state() -> None:
    """Guard against an engine that samples one outcome and repeats it.

    A sampler returning the modal outcome for every shot would pass a support
    check and a loose total-variation check on a narrow distribution. Both
    outcomes of a Bell pair have to appear, and roughly equally.
    """

    circuit = fq.Circuit(2).h(0).cx(0, 1)
    samples = sample_stabilizer(circuit, shots=4000, seed=5)
    rows = {tuple(row) for row in samples.tolist()}

    assert rows == {(0, 0), (1, 1)}
    zeros = int((samples[:, 0] == 0).sum())
    assert abs(zeros - 2000) < 8 * math.sqrt(4000 / 4)

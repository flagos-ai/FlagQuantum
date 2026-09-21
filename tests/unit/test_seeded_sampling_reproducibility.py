"""`ExecutionOptions(seed=...)` reaches every sampling request the caller can make.

`ExecutionOptions.seed` is documented as a "Reproducible stochastic execution seed"
(`docs/development/API_CHANGE_PROPOSAL_002_EXECUTION_OPTIONS.md`) with no qualification about
which output reads it, and `fq.run` passes it to the output lowering once, for every request.
Before this file existed, seeding reached sampling over circuit wires but not Pauli-basis
sampling of an observable, whose measurement carried no seed and fell back to a fresh
generator.

Three properties are asserted together, because any one of them alone can be satisfied by a
broken implementation:

* a seeded request replays exactly;
* an unseeded request does not, so the first property is not vacuously true;
* two distinct seeds do not agree, which is what shows the seed is *read* rather than that a
  fresh `torch.Generator` merely happens to be deterministic.
"""

from __future__ import annotations

import pytest
import torch

import flagquantum as fq

pytestmark = pytest.mark.unit

SAMPLING_REQUESTS = (
    pytest.param(lambda: fq.counts(), id="counts-over-wires"),
    pytest.param(lambda: fq.samples(qubits=(0,)), id="samples-over-wires"),
    pytest.param(lambda: fq.counts(fq.Z(0)), id="counts-over-observable"),
    pytest.param(lambda: fq.samples(fq.Z(0)), id="samples-over-observable"),
    pytest.param(
        lambda: fq.counts(fq.X(0) @ fq.Y(1)), id="counts-over-two-wire-observable"
    ),
)


def _sample(circuit: fq.Circuit, outputs: object, *, seed: int | None, shots: int = 64):
    """One sampling call, returning whichever payload the request asked for."""

    return fq.run(
        circuit,
        outputs=outputs,
        shots=shots,
        options=fq.ExecutionOptions(seed=seed),
    )


def _payload(result: object, request_kind: str):
    if request_kind == "counts":
        return result.counts
    return result.samples


def _request_kind(outputs: object) -> str:
    return "counts" if outputs.kind == "counts" else "samples"


@pytest.mark.parametrize("outputs_factory", SAMPLING_REQUESTS)
def test_a_seeded_sampling_request_replays_exactly(outputs_factory: object) -> None:
    circuit = fq.Circuit(2).h(0).h(1)
    outputs = outputs_factory()
    kind = _request_kind(outputs)

    first = _payload(_sample(circuit, outputs, seed=5), kind)
    second = _payload(_sample(circuit, outputs, seed=5), kind)

    if kind == "counts":
        assert first == second
    else:
        torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("outputs_factory", SAMPLING_REQUESTS)
def test_an_unseeded_sampling_request_does_not_replay_exactly(
    outputs_factory: object,
) -> None:
    circuit = fq.Circuit(2).h(0).h(1)
    outputs = outputs_factory()
    kind = _request_kind(outputs)

    draws = [_payload(_sample(circuit, outputs, seed=None), kind) for _ in range(6)]

    if kind == "counts":
        assert len({repr(draw) for draw in draws}) > 1
    else:
        assert any(not torch.equal(draws[0], draw) for draw in draws[1:])


@pytest.mark.parametrize("outputs_factory", SAMPLING_REQUESTS)
def test_distinct_seeds_do_not_agree(outputs_factory: object) -> None:
    circuit = fq.Circuit(2).h(0).h(1)
    outputs = outputs_factory()
    kind = _request_kind(outputs)

    draws = [
        _payload(_sample(circuit, outputs, seed=seed), kind) for seed in (1, 2, 3, 4)
    ]

    if kind == "counts":
        assert len({repr(draw) for draw in draws}) > 1
    else:
        assert any(not torch.equal(draws[0], draw) for draw in draws[1:])


def test_a_batched_seeded_sampling_request_replays_per_entry() -> None:
    circuit = fq.Circuit(2, bsz=3).h(0).h(1)

    first = _sample(circuit, fq.counts(fq.Z(0)), seed=9).counts
    second = _sample(circuit, fq.counts(fq.Z(0)), seed=9).counts

    assert first == second
    assert len(first) == 3


@pytest.mark.parametrize("outputs_factory", SAMPLING_REQUESTS)
def test_the_seed_survives_a_serialized_plan(outputs_factory: object) -> None:
    circuit = fq.Circuit(2).h(0).h(1)
    outputs = outputs_factory()
    kind = _request_kind(outputs)
    options = fq.ExecutionOptions(seed=11, shots=64)

    plan = fq.plan(circuit, outputs=outputs, options=options)
    restored = fq.ExecutionPlan.from_json(plan.to_json())

    first = _payload(fq.run(restored), kind)
    second = _payload(fq.run(restored), kind)

    if kind == "counts":
        assert first == second
    else:
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_seeding_pauli_basis_sampling_does_not_change_what_is_sampled() -> None:
    # A bit-flip-free check on the distribution rather than on the seed: `h(0)` makes Z(0)
    # a fair coin, so a seeded 4096-shot run must still split it near evenly. This is the
    # guard against "reproducible" being achieved by sampling something else.
    circuit = fq.Circuit(1).h(0)

    counts = _sample(circuit, fq.counts(fq.Z(0)), seed=3, shots=4096).counts[0]

    assert sum(counts.values()) == 4096
    zeroes = counts.get("0", 0)
    assert abs(zeroes - 2048) < 5 * (4096**0.5) / 2

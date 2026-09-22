"""Pinned properties of the optional vectorized trajectory row sampler.

``sample_rows`` draws one categorical choice per trajectory row, and the two
paths it can take are different estimators over the same distribution. That makes
"is the switch on?" the question every other test here is a special case of, so
the properties are pinned rather than the code:

1. Off is the shipped behaviour, and it is not merely equivalent to it: with the
   switch unset the output is bitwise the output of the arithmetic that was there
   before this switch existed.
2. On is a different draw for the same seed. That is the contract change, and a
   test asserts it so the documentation cannot drift into claiming otherwise.
3. Both paths are reproducible for a seed, equivariant under a permutation of the
   generators, and blind to the trajectory batch size - each row's choice comes
   from its own generator and from nothing else.
4. Both paths refuse the same malformed rows. The inverse-CDF path has to check,
   because inverting a cumulative sum returns an index for input that
   ``torch.multinomial`` rejects, and a silently chosen branch is worse than the
   error it replaces.
5. Both paths sample the law they were given.

What is *not* pinned is the relative cost: a wall-clock assertion on a host this
noisy would fail for reasons unrelated to the code.
"""

from __future__ import annotations

import pytest
import torch

from flagquantum.simulation.statevector import noisy

pytestmark = pytest.mark.unit

SWITCH = "FQ_CPU_VECTORIZED_ROW_SAMPLING"


@pytest.fixture(autouse=True)
def _switch_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test states its own switch; none inherits another's."""

    monkeypatch.delenv(SWITCH, raising=False)


def _generators(count: int, *, base: int = 1000) -> list[torch.Generator]:
    return [torch.Generator().manual_seed(base + index) for index in range(count)]


def _probabilities(
    trajectories: int, branches: int, *, circuit_batch: int = 1, seed: int = 5
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.rand(trajectories, circuit_batch, branches, generator=generator)
    return raw / raw.sum(dim=-1, keepdim=True)


def _shipped_arithmetic(
    probabilities: torch.Tensor, generators: list[torch.Generator]
) -> torch.Tensor:
    """The per-row loop verbatim, as the pre-switch code spelled it."""

    choices = [
        torch.multinomial(row, 1, replacement=True, generator=generator).squeeze(-1)
        for row, generator in zip(probabilities, generators, strict=True)
    ]
    return torch.stack(choices)


# ------------------------------------------------------- 1. off is the old answer


@pytest.mark.parametrize(
    ("trajectories", "circuit_batch", "branches"),
    [(4, 1, 2), (4, 1, 4), (16, 1, 4), (32, 3, 4), (8, 2, 3)],
)
def test_the_default_path_is_bitwise_the_arithmetic_it_replaced(
    trajectories: int, circuit_batch: int, branches: int
) -> None:
    probabilities = _probabilities(trajectories, branches, circuit_batch=circuit_batch)
    got = noisy.sample_rows(probabilities, _generators(trajectories))
    expected = _shipped_arithmetic(probabilities, _generators(trajectories))
    assert torch.equal(got, expected)


@pytest.mark.parametrize(
    "value", [None, "", "0", "false", "FALSE", "off", "no", " 0 ", "nonsense"]
)
def test_the_switch_stays_off_unless_it_says_otherwise(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv(SWITCH, raising=False)
    else:
        monkeypatch.setenv(SWITCH, value)
    assert noisy._vectorized_row_sampling_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "on", "yes", " 1 "])
def test_the_switch_turns_on_when_asked(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(SWITCH, value)
    assert noisy._vectorized_row_sampling_enabled() is True


def test_the_switch_is_read_at_call_time_not_at_import_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process that sets the variable after importing gets the new kernel."""

    probabilities = _probabilities(8, 4)
    off = noisy.sample_rows(probabilities, _generators(8))
    monkeypatch.setenv(SWITCH, "1")
    on = noisy.sample_rows(probabilities, _generators(8))
    assert not torch.equal(off, on)
    monkeypatch.delenv(SWITCH)
    assert torch.equal(noisy.sample_rows(probabilities, _generators(8)), off)


# ------------------------------------------------- 2. on is a different draw


def test_the_switch_changes_the_seed_to_choice_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reason the switch exists and the reason its default is off."""

    probabilities = _probabilities(16, 4)
    off = noisy.sample_rows(probabilities, _generators(16))
    monkeypatch.setenv(SWITCH, "1")
    on = noisy.sample_rows(probabilities, _generators(16))
    assert off.shape == on.shape
    assert not torch.equal(off, on)
    # Not merely a relabelling: the two are allowed to agree on some rows, and
    # preferring one branch systematically would be the failure this catches.
    assert 0 < int((off != on).sum()) < off.numel()


# ----------------------------------------------------- 3. row identity and shape


@pytest.mark.parametrize("switch_on", [False, True])
@pytest.mark.parametrize(
    ("trajectories", "circuit_batch", "branches"),
    [(1, 1, 1), (1, 1, 4), (5, 1, 2), (5, 4, 3), (9, 2, 5)],
)
def test_the_shape_is_the_batch_shape(
    monkeypatch: pytest.MonkeyPatch,
    switch_on: bool,
    trajectories: int,
    circuit_batch: int,
    branches: int,
) -> None:
    if switch_on:
        monkeypatch.setenv(SWITCH, "1")
    probabilities = _probabilities(trajectories, branches, circuit_batch=circuit_batch)
    got = noisy.sample_rows(probabilities, _generators(trajectories))
    assert got.shape == (trajectories, circuit_batch)
    assert got.dtype == torch.int64
    assert int(got.min()) >= 0 and int(got.max()) < branches


@pytest.mark.parametrize("switch_on", [False, True])
def test_a_row_is_drawn_from_its_own_generator(
    monkeypatch: pytest.MonkeyPatch, switch_on: bool
) -> None:
    """Permuting the generators permutes the rows and changes nothing else.

    Every row is given the same distribution, because with different
    distributions a permuted draw differs for a reason that has nothing to do
    with which generator fed which row.
    """

    if switch_on:
        monkeypatch.setenv(SWITCH, "1")
    probabilities = torch.full((6, 1, 4), 0.25)
    order = [3, 0, 5, 1, 4, 2]
    straight = noisy.sample_rows(probabilities, _generators(6))
    permuted = noisy.sample_rows(
        probabilities, [torch.Generator().manual_seed(1000 + index) for index in order]
    )
    assert torch.equal(permuted, straight[order])


@pytest.mark.parametrize("switch_on", [False, True])
def test_a_seed_reproduces_on_both_paths(
    monkeypatch: pytest.MonkeyPatch, switch_on: bool
) -> None:
    if switch_on:
        monkeypatch.setenv(SWITCH, "1")
    probabilities = _probabilities(8, 4)
    first = noisy.sample_rows(probabilities, _generators(8))
    second = noisy.sample_rows(probabilities, _generators(8))
    assert torch.equal(first, second)


@pytest.mark.parametrize("switch_on", [False, True])
def test_a_row_does_not_depend_on_how_many_rows_there_are(
    monkeypatch: pytest.MonkeyPatch, switch_on: bool
) -> None:
    """Trajectory batch size is not part of a trajectory's stream."""

    if switch_on:
        monkeypatch.setenv(SWITCH, "1")
    every = _probabilities(12, 4)
    first_six = noisy.sample_rows(every[:6], _generators(6))
    split = noisy.sample_rows(every, _generators(12))
    assert torch.equal(split[:6], first_six)


@pytest.mark.parametrize("switch_on", [False, True])
def test_a_row_is_sampled_by_relative_weight_not_by_total(
    monkeypatch: pytest.MonkeyPatch, switch_on: bool
) -> None:
    """An unnormalised row is the distribution it normalises to.

    The inverse-CDF path scales its uniform by the row total, so a row that does
    not sum to one is the one place where this could silently differ from
    ``multinomial``, which normalises internally.
    """

    if switch_on:
        monkeypatch.setenv(SWITCH, "1")
    branches = 4
    draws = 20_000
    # Counts rather than probabilities: the row sums to ten, not one.
    unnormalised = torch.tensor([1.0, 2.0, 3.0, 4.0]).repeat(draws, 1, 1)
    frequencies = (
        torch.bincount(
            noisy.sample_rows(unnormalised, _generators(draws, base=5000)).reshape(-1),
            minlength=branches,
        )
        / draws
    )
    target = unnormalised[0, 0] / unnormalised[0, 0].sum()
    assert torch.allclose(frequencies, target, atol=0.02)


@pytest.mark.parametrize("switch_on", [False, True])
def test_the_sampled_law_is_the_law_it_was_given(
    monkeypatch: pytest.MonkeyPatch, switch_on: bool
) -> None:
    if switch_on:
        monkeypatch.setenv(SWITCH, "1")
    branches = 4
    target = torch.tensor([0.1, 0.2, 0.3, 0.4])
    draws = 40_000
    probabilities = target.repeat(draws, 1, 1)
    counts = torch.bincount(
        noisy.sample_rows(probabilities, _generators(draws, base=7000)).reshape(-1),
        minlength=branches,
    )
    assert torch.allclose(counts / draws, target, atol=0.01)


@pytest.mark.parametrize("switch_on", [False, True])
def test_a_concentrated_row_is_always_the_concentrated_branch(
    monkeypatch: pytest.MonkeyPatch, switch_on: bool
) -> None:
    if switch_on:
        monkeypatch.setenv(SWITCH, "1")
    probabilities = torch.tensor([[0.0, 0.0, 1.0]]).repeat(32, 1).unsqueeze(1)
    assert torch.equal(
        noisy.sample_rows(probabilities, _generators(32)), torch.full((32, 1), 2)
    )


# -------------------------------------------------------- 4. refusing bad input


@pytest.mark.parametrize("switch_on", [False, True])
def test_a_mismatched_generator_list_is_still_a_value_error(
    monkeypatch: pytest.MonkeyPatch, switch_on: bool
) -> None:
    if switch_on:
        monkeypatch.setenv(SWITCH, "1")
    with pytest.raises(ValueError, match="shorter than argument 1"):
        noisy.sample_rows(_probabilities(4, 4), _generators(3))


@pytest.mark.parametrize("switch_on", [False, True])
@pytest.mark.parametrize(
    "row",
    [
        [0.0, 0.0, 0.0],
        [0.5, 0.6, -0.1],
        [0.5, float("nan"), 0.5],
        [0.5, float("inf"), 0.5],
    ],
    ids=["zero-total", "negative", "nan", "inf"],
)
def test_a_row_multinomial_refuses_is_refused_on_both_paths(
    monkeypatch: pytest.MonkeyPatch, switch_on: bool, row: list[float]
) -> None:
    """The inverse-CDF path must not answer what multinomial rejects."""

    if switch_on:
        monkeypatch.setenv(SWITCH, "1")
    probabilities = torch.tensor([row, row])
    with pytest.raises(RuntimeError, match="multinomial|probability tensor"):
        noisy.sample_rows(probabilities, _generators(2))


@pytest.mark.parametrize("switch_on", [False, True])
def test_a_refused_row_is_refused_even_beside_a_good_one(
    monkeypatch: pytest.MonkeyPatch, switch_on: bool
) -> None:
    if switch_on:
        monkeypatch.setenv(SWITCH, "1")
    probabilities = torch.tensor([[0.25, 0.25, 0.5], [0.0, 0.0, 0.0]])
    with pytest.raises(RuntimeError, match="multinomial|probability tensor"):
        noisy.sample_rows(probabilities, _generators(2))


# ------------------------------------------------- 5. through the public runtime


def _noisy_circuit():
    import flagquantum as fq

    circuit = fq.Circuit(3)
    for wire in range(3):
        circuit.h(wire)
    for _ in range(4):
        for wire in range(2):
            circuit.cx(wire, wire + 1)
        for wire in range(3):
            circuit.ry(wire, 0.23 * (wire + 1))
    return circuit


@pytest.mark.parametrize("switch_on", [False, True])
def test_the_trajectory_batch_size_still_does_not_change_a_seeded_result(
    monkeypatch: pytest.MonkeyPatch, switch_on: bool
) -> None:
    """The invariant the runtime documents, held on both paths."""

    import flagquantum.noise as fqn
    import flagquantum.runtime as fqr

    if switch_on:
        monkeypatch.setenv(SWITCH, "1")
    circuit = _noisy_circuit()
    noise_model = fqn.NoiseModel().add("cx", fqn.bit_flip_channel(0.05))
    results = [
        fqr.run_noisy_statevector(
            circuit,
            noise_model,
            trajectories=32,
            trajectory_batch_size=batch,
            seed=7,
        )
        for batch in (32, 16, 8)
    ]
    assert all(
        torch.equal(results[0].expectation_z, result.expectation_z)
        for result in results[1:]
    )
    assert results[0].trajectory_ids == tuple(range(32))


@pytest.mark.parametrize("switch_on", [False, True])
def test_the_runtime_reaches_the_path_the_switch_selects(
    monkeypatch: pytest.MonkeyPatch, switch_on: bool
) -> None:
    """The switch is wired to the runtime, not merely readable next to it."""

    import flagquantum.noise as fqn
    import flagquantum.runtime as fqr

    if switch_on:
        monkeypatch.setenv(SWITCH, "1")
    calls: list[str] = []
    per_row, by_inverse_cdf = (
        noisy._sample_rows_per_row,
        noisy._sample_rows_by_inverse_cdf,
    )

    def record(name, inner):
        def wrapper(*args, **kwargs):
            calls.append(name)
            return inner(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(noisy, "_sample_rows_per_row", record("per_row", per_row))
    monkeypatch.setattr(
        noisy, "_sample_rows_by_inverse_cdf", record("inverse_cdf", by_inverse_cdf)
    )
    noise_model = fqn.NoiseModel().add("cx", fqn.bit_flip_channel(0.05))
    fqr.run_noisy_statevector(
        _noisy_circuit(),
        noise_model,
        trajectories=32,
        trajectory_batch_size=32,
        seed=7,
    )
    assert set(calls) == {"inverse_cdf" if switch_on else "per_row"}

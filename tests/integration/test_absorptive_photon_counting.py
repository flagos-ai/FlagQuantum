"""Independent scalar blocks distinguish absorption from QND count effects."""

from __future__ import annotations

import math
import random

import pytest
import torch

pytestmark = pytest.mark.integration


def reference_weight(observed: int, incoming: int, efficiency: float) -> float:
    if efficiency == 0:
        return float(observed == 0)
    if efficiency == 1:
        return float(observed == incoming)
    return (
        math.comb(incoming, observed)
        * efficiency**observed
        * (1 - efficiency) ** (incoming - observed)
    )


def reference_inputs(cutoff: int) -> tuple[torch.Tensor, ...]:
    vectors = list(torch.eye(cutoff**2, dtype=torch.complex128).unbind())
    for left, right, phase in ((0, cutoff + 1, 1), (cutoff, cutoff + 1, 1j)):
        vector = torch.zeros(cutoff**2, dtype=torch.complex128)
        vector[left] = 1 / math.sqrt(2)
        vector[right] = torch.tensor(phase / math.sqrt(2), dtype=torch.complex128)
        vectors.append(vector)
    return tuple(torch.outer(vector, vector.conj()) for vector in vectors)


def reference_branches(
    density: torch.Tensor, cutoff: int, efficiency: float, mode: int = 0
) -> tuple[torch.Tensor, ...]:
    """Fill surviving vacuum blocks directly, without constructing any Kraus map."""
    branches = []
    for observed in range(cutoff):
        output = torch.zeros_like(density)
        for row in range(cutoff):
            for column in range(cutoff):
                for incoming in range(observed, cutoff):
                    a, b = (
                        (incoming * cutoff + row, incoming * cutoff + column)
                        if mode == 0
                        else (row * cutoff + incoming, column * cutoff + incoming)
                    )
                    out_a, out_b = (
                        (row, column) if mode == 0 else (row * cutoff, column * cutoff)
                    )
                    output[out_a, out_b] += (
                        reference_weight(observed, incoming, efficiency) * density[a, b]
                    )
        branches.append(output)
    return tuple(branches)


def reference_unread(density: torch.Tensor, cutoff: int, mode: int = 0) -> torch.Tensor:
    output = torch.zeros_like(density)
    for row in range(cutoff):
        for column in range(cutoff):
            value = sum(
                (
                    density[n * cutoff + row, n * cutoff + column]
                    if mode == 0
                    else density[row * cutoff + n, column * cutoff + n]
                )
                for n in range(cutoff)
            )
            a, b = (row, column) if mode == 0 else (row * cutoff, column * cutoff)
            output[a, b] = value
    return output


def reference_hom(cutoff: int, transmission: float) -> torch.Tensor:
    density = torch.zeros((cutoff**2, cutoff**2), dtype=torch.complex128)
    density[0, 0] = (1 - transmission) ** 2
    density[1, 1] = density[cutoff, cutoff] = transmission * (1 - transmission)
    for row in (2, 2 * cutoff):
        for column in (2, 2 * cutoff):
            density[row, column] = transmission**2 / 2
    return density


def reference_probabilities(
    cutoff: int, transmission: float, efficiency: float
) -> torch.Tensor:
    registered = transmission * efficiency
    weights = [
        1 - registered + registered**2 / 2,
        registered * (1 - registered),
        registered**2 / 2,
    ]
    return torch.tensor(weights + [0.0] * (cutoff - 3), dtype=torch.float64)


@pytest.mark.parametrize("cutoff", [3, 4])
@pytest.mark.parametrize("efficiency", [0.0, 0.65, 1.0])
def test_scalar_reference_closure_and_closed_count_law(
    cutoff: int, efficiency: float
) -> None:
    for density in reference_inputs(cutoff):
        for mode in (0, 1):
            rows = reference_branches(density, cutoff, efficiency, mode)
            torch.testing.assert_close(
                sum(rows), reference_unread(density, cutoff, mode), atol=1e-12, rtol=0
            )
            assert abs(float(sum(row.trace().real for row in rows)) - 1) <= 1e-12
    for rate in (0.0, 0.2):
        transmission = math.exp(-rate * math.pi / (4 * 0.7))
        rows = reference_branches(
            reference_hom(cutoff, transmission), cutoff, efficiency
        )
        weights = torch.stack([row.trace().real for row in rows])
        torch.testing.assert_close(
            weights,
            reference_probabilities(cutoff, transmission, efficiency),
            atol=1e-12,
            rtol=0,
        )


@pytest.mark.parametrize("cutoff", [3, 4])
@pytest.mark.parametrize("efficiency", [0.0, 0.65, 1.0])
@pytest.mark.parametrize("mode", [0, 1])
def test_complete_instrument_and_all_complex_scalar_blocks(
    cutoff: int, efficiency: float, mode: int
) -> None:
    from examples.absorptive_photon_counting import absorptive_instrument
    from flagquantum.operators import SuperOperator

    instrument = absorptive_instrument(cutoff, efficiency, mode=mode)
    assert list(instrument) == [str(n) for n in range(cutoff)]
    effect = sum(k.mH @ k for operators in instrument.values() for k in operators)
    torch.testing.assert_close(
        effect, torch.eye(cutoff**2, dtype=torch.complex128), atol=1e-12, rtol=0
    )
    for density in reference_inputs(cutoff):
        expected = reference_branches(density, cutoff, efficiency, mode)
        weighted = torch.zeros_like(density)
        for operators, oracle in zip(instrument.values(), expected, strict=True):
            actual = SuperOperator.from_kraus(operators).apply(density)
            torch.testing.assert_close(actual, oracle, atol=1e-12, rtol=0)
            probability = float(oracle.trace().real)
            if probability > 0:
                assert probability > 1e-8
                original_trace = actual.trace().real
                conditional = actual / original_trace
                torch.testing.assert_close(
                    conditional, oracle / probability, atol=1e-10, rtol=0
                )
                assert abs(float(conditional.trace().real) - 1) <= 1e-10
                assert float(abs(conditional - conditional.mH).max()) <= 1e-10
                assert float(torch.linalg.eigvalsh(conditional).min()) >= -1e-10
                weighted += original_trace * conditional
            else:
                torch.testing.assert_close(
                    actual, torch.zeros_like(actual), atol=1e-12, rtol=0
                )
        torch.testing.assert_close(
            weighted, reference_unread(density, cutoff, mode), atol=1e-10, rtol=0
        )


def test_absorption_differs_from_qnd_and_unresolved_alternatives_are_incoherent() -> (
    None
):
    from examples.absorptive_photon_counting import absorptive_instrument
    from examples.boson_interference import counting_instrument
    from flagquantum.operators import SuperOperator

    density = torch.zeros((9, 9), dtype=torch.complex128)
    density[6, 6] = 1
    absorbed = SuperOperator.from_kraus(absorptive_instrument(3, 1.0)["2"]).apply(
        density
    )
    qnd = SuperOperator.from_kraus(counting_instrument()["2/0"]).apply(density)
    assert float(absorbed[0, 0].real) == 1 and float(qnd[6, 6].real) == 1
    missed = SuperOperator.from_kraus(absorptive_instrument(3, 0.0)["0"]).apply(density)
    assert torch.equal(absorbed, missed)
    bell, coherent = reference_inputs(3)[-2:]
    instrument = absorptive_instrument(3, 0.65)
    correct = SuperOperator.from_kraus(instrument["0"]).apply(bell)
    wrong_sum = torch.stack(instrument["0"]).sum(dim=0)
    wrong = wrong_sum @ bell @ wrong_sum.mH
    assert abs(complex(correct[0, 1])) <= 1e-12
    assert abs(complex(wrong[0, 1])) > 0.1
    for operators, oracle in zip(
        instrument.values(), reference_branches(coherent, 3, 0.65), strict=True
    ):
        torch.testing.assert_close(
            SuperOperator.from_kraus(operators).apply(coherent),
            oracle,
            atol=1e-12,
            rtol=0,
        )
    assert abs(complex(reference_branches(coherent, 3, 0.65)[1][0, 1]).imag) > 0.1


@pytest.mark.parametrize("cutoff", [3, 4])
@pytest.mark.parametrize("rate", [0.0, 0.2])
@pytest.mark.parametrize("efficiency", [0.0, 0.65, 1.0])
def test_evolved_hom_sampling_full_branches_and_same_environment_replay(
    cutoff: int, rate: float, efficiency: float
) -> None:
    import flagquantum.lindblad as fql
    from examples.absorptive_photon_counting import absorptive_instrument
    from examples.boson_interference import interference_plan
    from flagquantum.operators import SuperOperator
    from flagquantum.runtime import sample_terminal

    plan = interference_plan(rate, cutoff=cutoff)
    evolution = fql.run(plan)
    assert evolution.density_matrices is not None
    density = evolution.density_matrices[-1]
    transmission = math.exp(-rate * math.pi / (4 * 0.7))
    expected_density = reference_hom(cutoff, transmission)
    torch.testing.assert_close(density, expected_density, atol=1e-10, rtol=0)
    instrument = absorptive_instrument(cutoff, efficiency)
    torch_state, python_state = torch.get_rng_state(), random.getstate()
    observed = sample_terminal(density, instrument, shots=4096, seed=7)
    replay = fql.run(fql.LindbladPlan.from_json(plan.to_json()))
    assert replay.density_matrices is not None
    assert torch.equal(evolution.density_matrices, replay.density_matrices)
    reconstructed = absorptive_instrument(cutoff, efficiency)
    repeated = sample_terminal(
        replay.density_matrices[-1], reconstructed, shots=4096, seed=7
    )
    assert isinstance(observed.value, torch.Tensor) and isinstance(
        repeated.value, torch.Tensor
    )
    assert torch.equal(observed.value, repeated.value)
    assert (
        torch.equal(torch_state, torch.get_rng_state())
        and python_state == random.getstate()
    )
    assert observed.value.shape == (4096,) and observed.value.dtype == torch.int64
    assert observed.qubits == () and observed.shots == 4096
    assert observed.metadata["outcome_labels"] == [str(n) for n in range(cutoff)]
    expected_weights = reference_probabilities(cutoff, transmission, efficiency)
    torch.testing.assert_close(
        torch.tensor(observed.metadata["probabilities"], dtype=torch.float64),
        expected_weights,
        atol=1e-10,
        rtol=0,
    )
    counts = torch.bincount(observed.value, minlength=cutoff)
    assert int(counts.sum()) == 4096
    for count, probability in zip(
        counts.tolist(), expected_weights.tolist(), strict=True
    ):
        if probability == 0:
            assert count == 0
        elif probability == 1:
            assert count == 4096
        else:
            assert abs(count / 4096 - probability) < 6 * math.sqrt(
                probability * (1 - probability) / 4096
            )
    expected_branches = reference_branches(expected_density, cutoff, efficiency)
    actual_branches = []
    for operators, oracle in zip(instrument.values(), expected_branches, strict=True):
        actual = SuperOperator.from_kraus(operators).apply(density)
        actual_branches.append(actual)
        torch.testing.assert_close(actual, oracle, atol=1e-10, rtol=0)
        probability = float(oracle.trace().real)
        if probability > 0:
            assert probability > 1e-8
            conditional = actual / actual.trace().real
            torch.testing.assert_close(
                conditional, oracle / probability, atol=1e-10, rtol=0
            )
            torch.testing.assert_close(
                conditional[:cutoff, :cutoff].trace(),
                torch.tensor(1.0, dtype=torch.complex128),
                atol=1e-10,
                rtol=0,
            )
        else:
            torch.testing.assert_close(
                actual, torch.zeros_like(actual), atol=1e-10, rtol=0
            )
    torch.testing.assert_close(
        sum(actual_branches),
        reference_unread(expected_density, cutoff),
        atol=1e-10,
        rtol=0,
    )


@pytest.mark.parametrize("efficiency", [0.0, 0.65, 1.0])
def test_sector_closed_embedding_with_no_infinite_fock_claim(efficiency: float) -> None:
    import flagquantum.lindblad as fql
    from examples.absorptive_photon_counting import absorptive_instrument
    from examples.boson_interference import interference_plan
    from flagquantum.operators import SuperOperator

    embedding = torch.tensor([4 * a + b for a in range(3) for b in range(3)])
    inputs3 = reference_inputs(3)
    inputs4 = reference_inputs(4)
    paired = [
        (inputs3[3 * a + b], inputs4[4 * a + b])
        for a in range(3)
        for b in range(3)
        if a + b <= 2
    ]
    paired.extend(zip(inputs3[-2:], inputs4[-2:], strict=True))
    for rate in (0.0, 0.2):
        a = fql.run(interference_plan(rate, cutoff=3)).density_matrices
        b = fql.run(interference_plan(rate, cutoff=4)).density_matrices
        assert a is not None and b is not None
        torch.testing.assert_close(
            a,
            b.index_select(1, embedding).index_select(2, embedding),
            atol=1e-10,
            rtol=0,
        )
        paired.append((a[-1], b[-1]))
    instrument3, instrument4 = absorptive_instrument(
        3, efficiency
    ), absorptive_instrument(4, efficiency)
    for density3, density4 in paired:
        for m in range(4):
            branch4 = SuperOperator.from_kraus(instrument4[str(m)]).apply(density4)
            if m == 3:
                torch.testing.assert_close(
                    branch4, torch.zeros_like(branch4), atol=1e-10, rtol=0
                )
                continue
            branch3 = SuperOperator.from_kraus(instrument3[str(m)]).apply(density3)
            embedded = branch4.index_select(0, embedding).index_select(1, embedding)
            torch.testing.assert_close(branch3, embedded, atol=1e-10, rtol=0)
            reference_probability = float(
                reference_branches(density3, 3, efficiency)[m].trace().real
            )
            if reference_probability > 1e-8:
                torch.testing.assert_close(
                    branch3 / branch3.trace().real,
                    embedded / branch4.trace().real,
                    atol=1e-10,
                    rtol=0,
                )

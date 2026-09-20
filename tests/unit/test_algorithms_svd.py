"""What the singular-value unit reads, checked against the matrix's own decomposition.

Every expected value in this file was produced by a real run of the construction it
checks, at six counting wires and 20000 shots, where one standard error of a share near
one half is about a third of a percent. The reference path is
:func:`torch.linalg.svdvals`, which is the decomposition the readout estimates, and the
phase grid the distribution sits on is checked a second time against
:func:`torch.linalg.eigvalsh` of the embedding -- a different routine from the exponential
the circuit is built from, so the counters are not compared against a value the
construction was handed.

Two things here are instruments rather than comparisons, and both are written so that a
stub cannot satisfy them. The extraction reads the encoding's block out of the circuit's
own state, one basis state at a time, and asserts both what the block is (the embedding
over ``alpha``) and what it is not (the embedding, or the matrix): a function that
returned its argument would fail the second half, and so would one that encoded the
embedding without its subnormalisation. The post-selection test reads the probability the
ancilla survives at, which is the same number the module's docstring states the block
costs to read.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest
import torch

from flagquantum.algorithms.primitives.state_preparation import append_arbitrary_state
from flagquantum.algorithms.svd import (
    SingularValueResult,
    _append_block_encoding,
    _input_amplitudes,
    _PhaseFromEmbedding,
    _subnormalisation,
    estimate_singular_values,
)
from flagquantum.circuit import Circuit

pytestmark = pytest.mark.unit

# A diagonal matrix, whose singular vectors are the basis states themselves, so the
# post-selection probability of a basis state is the squared singular value over alpha.
_DIAGONAL = torch.tensor([[1.0, 0.0], [0.0, 2.0]])

# A matrix whose two off-diagonal blocks differ, so the embedding is not a symmetric
# matrix written twice and its singular vectors are not the basis states.
_NON_SYMMETRIC = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Four rows and four columns, which is the unit's bound: a symmetric tridiagonal matrix
# whose four singular values are all distinct, so a readout cannot be excused as landing
# between two of them.
_FOUR = torch.tensor(
    [
        [1.0, 0.5, 0.0, 0.0],
        [0.5, 2.0, 0.25, 0.0],
        [0.0, 0.25, 3.0, 0.5],
        [0.0, 0.0, 0.5, 4.0],
    ]
)

_MATRICES = (_DIAGONAL, _NON_SYMMETRIC, _FOUR)

# The width and the sample size every measured value in this file was taken at.
_COUNTING_WIRES = 6
_SHOTS = 20000
_SEED = 11

# Measured at the settings above, one readout per matrix in ``_MATRICES`` order. The
# readout is the counting register's mode, so it is a grid value of the register's own
# resolution and not the singular value it estimates; what ties it to that value is
# ``SingularValueResult.within``, which the tests below assert separately.
_MEASURED_READOUTS = (1.9764235376052373, 5.567413560173162, 4.191491800734554)
_MEASURED_SHARES = (0.6611, 0.53065, 0.55525)


def _embedding(matrix: torch.Tensor) -> torch.Tensor:
    """Return ``[[0, A], [A^T, 0]]`` for a real square matrix, as the module forms it."""
    rows = int(matrix.shape[0])
    data = matrix.to(torch.float64)
    embedding = torch.zeros((2 * rows, 2 * rows), dtype=torch.float64)
    embedding[:rows, rows:] = data
    embedding[rows:, :rows] = data.T
    return embedding


def _extracted_block(embedding: torch.Tensor, alpha: float) -> torch.Tensor:
    """Read the encoding's block out of the circuit, one basis state at a time.

    The block is ``<0| U |0>`` on the embedding's wires, so its column ``j`` is what the
    circuit leaves on those wires when it is entered with the basis state ``|j>`` and the
    ancilla in ``|0>``. Each column is therefore read from a circuit of its own, and no
    unitary is ever written out as a matrix: what is read is the state the composition
    leaves, which is the only way this package exposes a circuit's action.
    """
    n_wires = int(embedding.shape[0]).bit_length() - 1
    dimension = int(embedding.shape[0])
    columns = []
    for basis in range(dimension):
        circuit = Circuit(1 + n_wires)
        for index in range(n_wires):
            if (basis >> (n_wires - 1 - index)) & 1:
                circuit.gate("x", 1 + index)
        _append_block_encoding(
            circuit,
            embedding,
            alpha,
            ancilla=0,
            wires=list(range(1, 1 + n_wires)),
        )
        columns.append(circuit.state().reshape(-1)[:dimension])
    return torch.stack(columns, dim=1)


def _singular_vectors(
    matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the matrix's left vectors, singular values and right vectors."""
    left, spectrum, right = torch.linalg.svd(matrix.to(torch.float64))
    return left, spectrum, right


def _positive_eigenvector(matrix: torch.Tensor, index: int) -> torch.Tensor:
    """Return the embedding's eigenvector of ``+sigma_index`` as amplitudes.

    The eigenvector of the embedding for ``+sigma`` is ``(|0>|u> + |1>|v>)/sqrt(2)``, with
    ``u`` and ``v`` the singular vectors, and the block qubit leads the register, so the
    amplitudes are the left vector's column followed by the right vector's row.
    """
    left, _, right = _singular_vectors(matrix)
    return torch.cat((left[:, index], right[index, :])) / math.sqrt(2.0)


def test_the_extraction_is_the_embedding_over_alpha() -> None:
    """The composition's block is the embedding over the subnormalisation."""
    for matrix in _MATRICES:
        embedding = _embedding(matrix)
        alpha = _subnormalisation(embedding, None)
        block = _extracted_block(embedding, alpha)
        assert float((block - embedding / alpha).abs().max()) < 1e-6
        # The block of a real Hermitian matrix is real: the two branches' unitaries are
        # conjugates of one another, so their mean is real by construction.
        assert float(block.imag.abs().max()) < 1e-6


def test_the_extraction_is_not_the_embedding_or_the_matrix() -> None:
    """The block is subnormalised, so neither the embedding nor the matrix comes back.

    This is the other direction of the check above, and it is the one a stub fails: a
    function that returned the matrix it was given would agree with the matrix here.
    """
    for matrix in _MATRICES:
        embedding = _embedding(matrix)
        alpha = _subnormalisation(embedding, None)
        rows = int(matrix.shape[0])
        block = _extracted_block(embedding, alpha).real
        assert float((block - embedding).abs().max()) > 0.1
        # The encoding's own off-diagonal quarter is the matrix over alpha, not the
        # matrix: the subnormalisation is part of what an encoding of a matrix is.
        assert float((block[:rows, rows:] - matrix).abs().max()) > 0.1
        assert (
            float((block[:rows, rows:] - matrix.to(torch.float64) / alpha).abs().max())
            < 1e-6
        )
        # The block is what a contraction looks like: no unitary has an operator of norm
        # greater than one as a block, and this one is the embedding over alpha, whose
        # spectral norm the subnormalisation divides down to at most one.
        assert float(torch.linalg.matrix_norm(block, ord=2)) <= 1.0
        assert float(torch.linalg.matrix_norm(block, ord=2)) == pytest.approx(
            float(torch.linalg.matrix_norm(embedding, ord=2)) / alpha, rel=1e-5
        )


def test_a_subnormalisation_without_room_for_the_block_is_refused() -> None:
    """A factor below the embedding's spectral norm is refused, not encoded.

    Below that norm the block would be an operator of norm greater than one, and no
    unitary has such an operator as one of its blocks, so there is no encoding to build.
    """
    embedding = _embedding(_NON_SYMMETRIC)
    spectral = float(torch.linalg.matrix_norm(embedding, ord=2))
    _subnormalisation(embedding, spectral)
    with pytest.raises(ValueError, match="no unitary has an operator of norm"):
        _subnormalisation(embedding, spectral / 2.0)
    with pytest.raises(ValueError, match="positive and finite"):
        _subnormalisation(embedding, 0.0)
    with pytest.raises(ValueError, match="must be a real number"):
        _subnormalisation(embedding, True)
    with pytest.raises(ValueError, match="divided by nothing"):
        _subnormalisation(torch.zeros((4, 4), dtype=torch.float64), None)
    # The construction checks the same thing at its own entry, so a factor that never
    # passed through the builder cannot reach the selection's arccosine and be folded
    # into the unit circle silently.
    with pytest.raises(ValueError, match="no unitary has an operator of norm"):
        _append_block_encoding(
            Circuit(3), embedding, spectral / 2.0, ancilla=0, wires=[1, 2]
        )
    # And the one thing the selection itself refuses: an argument that is not Hermitian.
    spoiled = _embedding(_NON_SYMMETRIC)
    spoiled[0, 3] += 0.5
    with pytest.raises(ValueError, match="not Hermitian"):
        _append_block_encoding(
            Circuit(3),
            spoiled,
            _subnormalisation(spoiled, None),
            ancilla=0,
            wires=[1, 2],
        )


def test_the_subnormalisation_defaults_to_the_frobenius_norm() -> None:
    """The default factor is the embedding's Frobenius norm, which covers its spectrum."""
    for matrix in _MATRICES:
        embedding = _embedding(matrix)
        alpha = _subnormalisation(embedding, None)
        assert alpha == pytest.approx(
            float(torch.linalg.matrix_norm(embedding, ord="fro")), abs=1e-12
        )
        assert alpha >= float(torch.linalg.matrix_norm(embedding, ord=2))
        assert alpha >= float(torch.linalg.svdvals(matrix)[0])


@pytest.mark.parametrize(
    ("matrix", "readout", "share"),
    list(zip(_MATRICES, _MEASURED_READOUTS, _MEASURED_SHARES, strict=True)),
)
def test_the_readout_is_the_singular_value_at_the_mode(
    matrix: torch.Tensor, readout: float, share: float
) -> None:
    """The mode's readout is the measured one, and it lands on the largest singular value."""
    result = estimate_singular_values(
        matrix, n_counting_wires=_COUNTING_WIRES, shots=_SHOTS, seed=_SEED
    )
    assert result.dominant_singular_value == pytest.approx(readout, abs=1e-6)
    assert result.dominant_share == pytest.approx(share, abs=0.01)
    assert result.resolution == pytest.approx(
        2.0 * result.alpha / 2.0**_COUNTING_WIRES, abs=1e-12
    )
    largest = float(torch.linalg.svdvals(matrix)[0])
    assert result.within(largest)
    # Half a step is the whole of the accuracy contract; measured at these settings, the
    # readout sits strictly inside it rather than at its edge.
    assert abs(result.dominant_singular_value - largest) < result.resolution / 2.0


def test_the_mode_sits_at_the_counter_nearest_the_largest_eigenvalues_phase() -> None:
    """The mode's counter is the one the embedding's own largest eigenvalue implies.

    The eigenvalue is read with :func:`torch.linalg.eigvalsh` of the embedding, which is
    not the routine the circuit's exponential is built from, so this compares the
    distribution against a separately computed spectrum rather than against itself.
    """
    for matrix in _MATRICES:
        embedding = _embedding(matrix)
        alpha = _subnormalisation(embedding, None)
        eigenvalues = torch.linalg.eigvalsh(embedding)
        largest = float(eigenvalues.max())
        # The positive branch's phase, on the register's own grid.
        phase = 1.0 - largest / (2.0 * alpha)
        nearest = round(phase * 2.0**_COUNTING_WIRES)
        result = estimate_singular_values(
            matrix, n_counting_wires=_COUNTING_WIRES, shots=_SHOTS, seed=_SEED
        )
        mode = max(result.distribution, key=result.distribution.__getitem__)
        assert int(mode, 2) == nearest
        # And the counter value reads the eigenvalue back, to the register's resolution.
        assert result.within(abs(largest))


def test_the_lower_half_of_the_register_carries_almost_none_of_the_sample() -> None:
    """The input state has no overlap with the negative eigenvectors, so nothing peaks there.

    The test below asserts that zero overlap directly; measured at these settings, what the
    lower half carries is under a fiftieth of the sample while the mode's own share is over
    a half, and the shares there follow the dominant peak's phase-estimation tail rather
    than forming a peak of their own.
    """
    for matrix in _MATRICES:
        result = estimate_singular_values(
            matrix, n_counting_wires=_COUNTING_WIRES, shots=_SHOTS, seed=_SEED
        )
        lower = sum(
            value
            for key, value in result.distribution.items()
            if int(key, 2) < 2 ** (_COUNTING_WIRES - 1)
        )
        assert lower < 0.02
        mode = max(result.distribution, key=result.distribution.__getitem__)
        assert int(mode, 2) >= 2 ** (_COUNTING_WIRES - 1)
        assert result.dominant_share > 0.5


@pytest.mark.parametrize("matrix", _MATRICES)
def test_the_input_state_is_the_weighted_positive_eigenvector_state(
    matrix: torch.Tensor,
) -> None:
    """The state's overlap is the weighting the module's docstring states, and no more.

    Its overlap with the ``+sigma_i`` eigenvector is ``sigma_i / ||A||_F`` and its overlap
    with the ``-sigma_i`` eigenvector is zero. The negative half of that is the part that
    makes the readout a singular value rather than a pair of opposite eigenvalues.
    """
    amplitudes = _input_amplitudes(matrix).to(torch.complex128)
    left, spectrum, right = _singular_vectors(matrix)
    norm = float(torch.linalg.matrix_norm(matrix.to(torch.float64), ord="fro"))
    for index in range(int(matrix.shape[0])):
        negative = torch.cat((left[:, index], -right[index, :])) / math.sqrt(2.0)
        positive = torch.cat((left[:, index], right[index, :])) / math.sqrt(2.0)
        assert (
            abs(complex(torch.vdot(negative.to(torch.complex128), amplitudes))) < 1e-9
        )
        overlap = abs(complex(torch.vdot(positive.to(torch.complex128), amplitudes)))
        assert overlap == pytest.approx(float(spectrum[index]) / norm, abs=1e-12)
    assert float(torch.linalg.vector_norm(amplitudes)) == pytest.approx(1.0, abs=1e-12)


def test_the_extracted_columns_are_the_post_selection_probabilities() -> None:
    """A block column's squared norm is the basis state's post-selection probability.

    The probabilities are bounded by the ratio the module's docstring states, which is the
    largest value the probability takes over all inputs.
    """
    for matrix in _MATRICES:
        embedding = _embedding(matrix)
        alpha = _subnormalisation(embedding, None)
        block = _extracted_block(embedding, alpha).real
        largest = float(torch.linalg.svdvals(matrix)[0])
        bound = (largest / alpha) ** 2
        probabilities = [
            float(block[:, column].norm() ** 2) for column in range(block.shape[1])
        ]
        assert max(probabilities) <= bound + 1e-6
        # For a diagonal matrix the basis states are its singular vectors, so the bound is
        # attained and the columns are the squared singular values over alpha squared.
        if torch.equal(matrix, matrix.diag().diag()):
            assert max(probabilities) == pytest.approx(bound, abs=1e-6)


def test_the_post_selection_probability_of_the_dominant_direction_is_the_ratio() -> (
    None
):
    """Preparing the dominant eigenvector and post-selecting costs the squared ratio.

    This is the module docstring's statement read at the one input that attains it, and it
    is read from the circuit rather than from the block matrix.
    """
    for matrix in _MATRICES:
        embedding = _embedding(matrix)
        alpha = _subnormalisation(embedding, None)
        n_wires = int(embedding.shape[0]).bit_length() - 1
        circuit = Circuit(1 + n_wires)
        append_arbitrary_state(
            circuit, _positive_eigenvector(matrix, 0), list(range(1, 1 + n_wires))
        )
        _append_block_encoding(
            circuit, embedding, alpha, ancilla=0, wires=list(range(1, 1 + n_wires))
        )
        state = circuit.state().reshape(-1)
        dimension = int(embedding.shape[0])
        kept = float((state[:dimension].abs() ** 2).sum())
        largest = float(torch.linalg.svdvals(matrix)[0])
        assert kept == pytest.approx((largest / alpha) ** 2, abs=1e-6)


def test_every_form_of_the_phase_unitary_appends_a_gate() -> None:
    """Each of the operator's three forms appends one gate on the wires it is given.

    Phase estimation consumes the power form only, so the plain form and the
    single-controlled form are reached by nothing else in the package: this is where they
    are exercised, and it is the reason the class docstring says they are implemented
    rather than refused.
    """
    embedding = _embedding(_NON_SYMMETRIC)
    unitary = _PhaseFromEmbedding(embedding, _subnormalisation(embedding, None))
    wires = list(range(1, 1 + unitary.n_wires))
    plain = Circuit(1 + unitary.n_wires)
    unitary.apply(plain, wires)
    controlled = Circuit(2 + unitary.n_wires)
    unitary.apply_controlled(controlled, 0, wires)
    powered = Circuit(2 + unitary.n_wires)
    unitary.apply_power_controlled(powered, 0, wires, 3)
    for circuit in (plain, controlled, powered):
        assert len(circuit.to_qir()) == 1
    # The two controlled forms are unitaries on one more wire than the plain one, which is
    # what makes them the same operator with an identity prepended on the control branch.
    matrix = torch.as_tensor(plain.to_qir()[0]["gate"]).to(torch.complex128)
    product = matrix @ matrix.conj().T
    assert float((product - torch.eye(matrix.shape[0])).abs().max()) < 1e-6
    for circuit in (controlled, powered):
        emitted = torch.as_tensor(circuit.to_qir()[0]["gate"])
        assert emitted.shape[0] == 2 * matrix.shape[0]


@pytest.mark.parametrize("matrix", _MATRICES)
def test_the_distribution_is_the_counting_register_marginal(
    matrix: torch.Tensor,
) -> None:
    """The distribution is keyed by the counting register alone, and sums to one."""
    result = estimate_singular_values(
        matrix, n_counting_wires=_COUNTING_WIRES, shots=_SHOTS, seed=_SEED
    )
    assert set(len(key) for key in result.distribution) == {_COUNTING_WIRES}
    assert sum(result.distribution.values()) == pytest.approx(1.0, abs=1e-12)
    assert all(0.0 <= share <= 1.0 for share in result.distribution.values())
    assert result.n_counting_wires == _COUNTING_WIRES


def test_within_accepts_half_a_step_and_rejects_more() -> None:
    """The accuracy contract is half a counter step, at the boundary and beyond it.

    The step is a power of two and the readout is exact in binary, so the boundary itself
    is representable and the comparison at it is not decided by rounding.
    """
    result = SingularValueResult(
        dominant_singular_value=2.0,
        dominant_share=0.5,
        distribution={"100000": 0.5},
        n_counting_wires=6,
        resolution=0.5,
        alpha=4.0,
    )
    assert result.within(2.0)
    assert result.within(2.25)
    assert result.within(1.75)
    assert not result.within(2.2500000001)
    assert not result.within(1.7499999999)
    assert not result.within(0.0)


def test_the_result_rejects_fields_that_cannot_come_from_a_readout() -> None:
    """Every refusal the result class performs is reachable, and the message names it."""

    def build(**overrides: object) -> SingularValueResult:
        fields: dict[str, object] = {
            "dominant_singular_value": 2.0,
            "dominant_share": 0.5,
            "distribution": {"100000": 0.5},
            "n_counting_wires": 6,
            "resolution": 0.4,
            "alpha": 4.0,
        }
        fields.update(overrides)
        return SingularValueResult(**fields)  # type: ignore[arg-type]

    build()
    with pytest.raises(ValueError, match="positive and finite"):
        build(alpha=0.0)
    with pytest.raises(ValueError, match="must lie in \\(0, alpha\\]"):
        build(dominant_singular_value=0.0)
    with pytest.raises(ValueError, match="must lie in \\(0, alpha\\]"):
        build(dominant_singular_value=4.5)
    with pytest.raises(ValueError, match="sample share"):
        build(dominant_share=1.5)
    with pytest.raises(ValueError, match="must not be empty"):
        build(distribution={})
    with pytest.raises(ValueError, match="at least one wire"):
        build(n_counting_wires=0)
    with pytest.raises(ValueError, match="step must be positive"):
        build(resolution=0.0)


def test_the_result_is_frozen() -> None:
    """A result is a reading, not a register: its fields cannot be reassigned."""
    result = estimate_singular_values(
        _DIAGONAL, n_counting_wires=_COUNTING_WIRES, shots=_SHOTS, seed=_SEED
    )
    with pytest.raises(FrozenInstanceError):
        result.alpha = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        ("not a tensor", "must be a torch.Tensor"),
        (torch.zeros(4, dtype=torch.float64), "two-dimensional"),
        (torch.zeros(2, 3, dtype=torch.float64), "must be square"),
        (torch.zeros(3, 3, dtype=torch.float64), "power of two"),
        (torch.zeros(8, 8, dtype=torch.float64), "bounded at 2 index wires"),
        (torch.zeros(2, 2, dtype=torch.int64), "floating-point"),
        (torch.zeros(2, 2, dtype=torch.float64), "no embedding to normalise"),
        (torch.tensor([[1.0, float("nan")], [0.0, 1.0]]), "must be finite"),
    ],
)
def test_the_estimate_refuses_a_matrix_it_cannot_embed(
    matrix: object, message: str
) -> None:
    """Every matrix-level refusal is reachable, and each names its own condition."""
    with pytest.raises(ValueError, match=message):
        estimate_singular_values(matrix, n_counting_wires=2, shots=16)  # type: ignore[arg-type]


def test_the_estimate_refuses_a_width_or_a_sample_size_below_one() -> None:
    """A register with no wires and a sample with no shots are both refused."""
    with pytest.raises(ValueError, match="at least one counting wire"):
        estimate_singular_values(_DIAGONAL, n_counting_wires=0)
    with pytest.raises(ValueError, match="at least one shot"):
        estimate_singular_values(_DIAGONAL, n_counting_wires=2, shots=0)


def test_the_estimate_replays_with_a_seed() -> None:
    """The same seed replays the same distribution exactly, and another seed is a re-sample."""
    first = estimate_singular_values(
        _NON_SYMMETRIC, n_counting_wires=_COUNTING_WIRES, shots=512, seed=7
    )
    again = estimate_singular_values(
        _NON_SYMMETRIC, n_counting_wires=_COUNTING_WIRES, shots=512, seed=7
    )
    assert first.distribution == again.distribution
    assert first.dominant_singular_value == again.dominant_singular_value
    other = estimate_singular_values(
        _NON_SYMMETRIC, n_counting_wires=_COUNTING_WIRES, shots=512, seed=8
    )
    assert other.distribution != first.distribution

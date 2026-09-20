"""Singular values of a real matrix, read from the phase of its Hermitian embedding.

The singular values of a matrix are the magnitudes of the eigenvalues of the Hermitian
matrix that carries it as its off-diagonal block, and this unit reads them from a phase:
that embedding is exponentiated and phase-estimated, and the counting register's mode is
read back as a singular value. The singular-value-estimation subroutine this unit follows
is the one a recommendation algorithm is built on in Kerenidis and Prakash, "Quantum
Recommendation Systems", ITCS 2017, LIPIcs Vol. 67, 49:1-49:21, DOI
10.4230/LIPIcs.ITCS.2017.49; the singular value decomposition of non-sparse low-rank
matrices is the subject of Rebentrost, Steffens, Marvian and Lloyd, *Physical Review A*
**97**(1), 012327 (2018), DOI 10.1103/PhysRevA.97.012327, approached there by exponentiating
the matrix. The block encoding this module builds in private is the one of Gilyén, Su, Low
and Wiebe, "Quantum singular value transformation and beyond: exponential improvements for
quantum matrix arithmetics", STOC 2019, pp. 193-204, DOI 10.1145/3313276.3316366,
Definition 1, which is where the subnormalisation factor and the convention
``||A|| <= alpha + eps`` are fixed.

**The premise is the input model, and this unit does not meet it.** The algorithm's cost
is counted in queries to a structure that returns the matrix's entries, and against that
count the state the estimation is applied to is assumed to be preparable -- and that
state is built from the matrix's singular vectors. Neither is present here. This unit
holds the matrix as an ordinary torch tensor, forms its embedding, exponentiates the
embedding as a dense matrix, and builds the input state by calling
:func:`torch.linalg.svd`, whose decomposition is the very thing the readout estimates. The
reading of the matrix, the exponentiation and the state preparation are therefore paid
explicitly rather than assumed away, and **no end-to-end advantage follows**: the
properties the query model is what buys are absent, and nothing here is faster, or
smaller, than calling :func:`torch.linalg.svd` directly.

**Dequantization exists and is recorded here.** Tang, STOC 2019,
DOI 10.1145/3313276.3316310, gives a classical algorithm for the recommendation problem
that removes the *exponential* speed-up, and it is **"only polynomially slower"**: its
bound contains ``eps**-12``, and the author calls it "a large slowdown in some
exponents". It is not a classical algorithm that matches the quantum runtime, and no
sentence here says that it is. Arrazola et al., *Quantum* **4**, 307 (2020), record the
practical conditions the dequantized algorithms need, and Gharibian-Le Gall (STOC 2022 /
SICOMP **52**(4)) give the hardness result for singular-value estimation under the sparse
access model, which is what rules the dequantization out for that model.

**The block encoding, and what reading it costs.** The private construction below
composes a preparation, a selection and the adjoint of the preparation. Its
subnormalisation ``alpha`` is what the block is smaller than the embedding by: the
extracted block is the embedding over ``alpha``, and ``alpha`` is at least the embedding's
spectral norm, so ``||A|| <= alpha`` holds for the matrix whose embedding that is. **A
block encoding is not free to read.** A readout that post-selects the ancilla succeeds
with probability ``||(A / alpha) |psi>||**2`` on a normalised input ``|psi>``, whose
greatest value over inputs is ``(||A|| / alpha)**2``; where ``alpha`` is much larger than
``||A||`` that probability is exponentially small, and a caller who is not told so will
take the block for free. The count that succeeds is the same number the extraction here
reads: the block's column ``j`` is the post-selected amplitude vector of the basis state
``|j>``, so its squared norm is that state's post-selection probability.

**The phase is not the singular value.** At the exponent ``pi`` the embedding's eigenvalue
``mu`` is carried by the eigenphase ``exp(-i * pi * mu / alpha)``, which is the phase
``phi = (-mu / (2 * alpha)) mod 1``. A counting register of ``m`` wires resolves a phase
to ``1 / 2**m``, so a counter value ``k`` reads the singular value
``2 * alpha * (1 - k / 2**m)``, and the readout therefore lies in ``(0, alpha]``. The
inversion is part of the readout and not a cosmetic detail: dropping the ``2 * alpha``
factor reports a phase, which is a wrong singular value rather than an error.

**The input state is where the singular vectors enter, and it weights the eigenvalues.**
The circuit prepares the state whose overlap with the embedding's eigenvector of
``+sigma_i`` is ``sigma_i / ||A||_F`` and whose overlap with the eigenvector of
``-sigma_i`` is zero, which is the weighting the vectorised matrix carries. A larger
singular value therefore carries a larger share of the sample before the counting
register's resolution spreads it. The state is prepared by
:func:`~flagquantum.algorithms.primitives.state_preparation.append_arbitrary_state`,
which solves for its rotation angles with a classical pass over every amplitude, so the
preparation is paid for in the classical work the premise paragraph describes.

**Which singular value the mode reports is not something this module predicts.** The mode
is the counting register's counter value with the largest share of the sample, and the
readout is that counter value's singular value. The module does not claim that is the
largest singular value: that depends on how each eigenvalue's share is spread across the
counter values the register has, which the module does not compute.
:meth:`SingularValueResult.within` states the accuracy contract and nothing more, so a
caller who needs the largest singular value specifically has to check the readout.

**Wire layout.** The counting register is wires ``0 .. m - 1``; the embedding's block
qubit follows it, and the embedding's data register follows that. The counting register's
bits lead every sample key, as :meth:`flagquantum.circuit.Circuit.counts` returns them,
one character per wire with wire 0 the most significant.

**The unit is bounded at four rows and four columns.** The embedding of an ``n``-wire
matrix is ``2 ** (n + 1)``-dimensional, and every form of the phase unitary is a dense
gate on it, so that is the demonstration scale the unit is written at.

This unit is demonstration scale. It makes no performance, capacity, convergence, or
hardware claim, and it does not select a runtime.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from ..circuit import Circuit
from .primitives.phase_estimation import append_phase_estimation
from .primitives.state_preparation import append_arbitrary_state

__all__ = ["SingularValueResult", "estimate_singular_values"]

# The default sample size, matching the other sampling units in this package.
_DEFAULT_SHOTS = 4096

# The embedding of a matrix with ``n`` index wires is ``2 ** (n + 1)``-dimensional and
# every form of the phase unitary is a dense gate on it, so the unit is bounded at two
# index wires: matrices of at most four rows and four columns.
_MAX_DATA_WIRES = 2


@dataclass(frozen=True, kw_only=True)
class SingularValueResult:
    """The outcome of one sampled :func:`estimate_singular_values` run.

    ``kw_only`` is not optional here, and the reason is how a call site reads rather than
    what it would catch: four of the fields are bare floats -- a singular value, a share
    of a sample, a step and a factor -- so a positional spelling would read as a row of
    unrelated numbers. A transposition among the floats is not what ``kw_only`` protects
    against either: the singular value's bound and the step's bound overlap, so a pair
    written the wrong way round can satisfy both and be accepted. Measured, the readout of
    the two-row matrix ``[[1, 0], [0, 2]]`` -- ``dominant_singular_value`` near two and
    ``resolution`` near a tenth of one -- constructs with those two fields transposed,
    because the step the readout was resolved at is a positive number inside the singular
    value's own bound, and the two fields then hold each other's values.

    Attributes:
        dominant_singular_value: The singular value read out at the mode ``k`` of
            ``distribution``, which is ``2 * alpha * (1 - k / 2**n_counting_wires)``. The
            module does not claim this is the largest singular value of the matrix and
            does not predict which singular value it will be; the mode is simply the
            counter value with the largest share. :meth:`within` is how a caller checks a
            specific singular value against it. The value lies in ``(0, alpha]``, which is
            the range the register's upper half resolves to.
        dominant_share: The share of the sample that landed on that mode. It is not the
            singular value's weight in the matrix, and the counting register's resolution
            spreads each eigenvalue's share over neighbouring counter values: see the
            module docstring.
        distribution: The counting register's marginal distribution, keyed by its
            big-endian bit string, one character per counting wire. Shares of the sample,
            so the values sum to one. The embedding's wires are folded away rather than
            ignored: a key is truncated to its leading ``n_counting_wires`` characters,
            because reading the full-register keys instead would follow the embedding's
            wires wherever the counting register is correlated with them.
        n_counting_wires: The width of the counting register the estimate came from.
        resolution: The phase step the counting register resolves, expressed in
            singular-value units as ``2 * alpha / 2**n_counting_wires``. A resolved
            singular value is accurate to about half of it, which is the accuracy
            :meth:`within` checks and the whole of the accuracy contract.
        alpha: The subnormalisation the embedding was read at, which is the factor the
            readout is scaled by and what :attr:`resolution` is stated in units of. It is
            at least the matrix's spectral norm and, by the module's default, the
            embedding's Frobenius norm.
    """

    dominant_singular_value: float
    dominant_share: float
    distribution: Mapping[str, float]
    n_counting_wires: int
    resolution: float
    alpha: float

    def __post_init__(self) -> None:
        """Reject a result that cannot be a singular-value readout.

        Raises:
            ValueError: If ``alpha`` is not positive and finite; if the dominant singular
                value is outside ``(0, alpha]``; if the dominant share is outside
                ``[0, 1]``; if the distribution is empty; if the register width is less
                than one; or if the resolution is not positive.
        """
        if not math.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError(
                "the subnormalisation a block is read at must be positive and finite, "
                f"got alpha={self.alpha}; the readout is the block it extracts, and a "
                "non-positive factor is not one"
            )
        if not 0.0 < self.dominant_singular_value <= self.alpha:
            raise ValueError(
                "a singular-value readout must lie in (0, alpha], got "
                f"dominant_singular_value={self.dominant_singular_value} against "
                f"alpha={self.alpha}; the register's upper half resolves the phases of "
                "the embedding's non-negative eigenvalues, and the lower half is not a "
                "singular value at this subnormalisation"
            )
        if not 0.0 <= self.dominant_share <= 1.0:
            raise ValueError(
                "a sample share must lie in [0, 1], got "
                f"dominant_share={self.dominant_share}"
            )
        if not self.distribution:
            raise ValueError(
                "the counting distribution must not be empty; with no counter value "
                "there is no singular value to read"
            )
        if self.n_counting_wires < 1:
            raise ValueError(
                "the counting register needs at least one wire, got "
                f"n_counting_wires={self.n_counting_wires}"
            )
        if self.resolution <= 0.0:
            raise ValueError(
                f"the counter step must be positive, got resolution={self.resolution}"
            )

    def within(self, singular_value: float) -> bool:
        """Return whether ``singular_value`` is within half a counter step of the readout.

        This is the accuracy contract the module makes, expressed once here rather than
        re-derived by every caller. Half a step is the whole of it: ``within`` accepts
        ``singular_value`` when the mode's counter value lies within half a step of that
        singular value's phase, and rejects it otherwise. A full step would accept a value
        the neighbouring counter value could equally claim, which is why the bound is half
        and not whole.

        **It does not claim that the readout resolves the largest singular value.** The
        mode is the counter value with the largest share, and the module does not predict
        which singular value that will be: :meth:`within` compares a value the caller
        names against what the mode reported, and answers only whether the two are that
        close. A caller who needs the largest singular value has to ask this for the
        largest singular value and read the answer.

        It is a resolution statement and not a distribution statement: a value this
        rejects is not excluded at any stated confidence level, and a singular value whose
        weighting in the input state is small is not disqualified by the readout at all.

        Args:
            singular_value: The singular value to compare against, normally a known one.

        Returns:
            ``True`` if
            ``abs(self.dominant_singular_value - singular_value) <= resolution / 2``.
        """
        return abs(self.dominant_singular_value - singular_value) <= self.resolution / 2


def estimate_singular_values(
    A: torch.Tensor,  # noqa: N803
    *,
    n_counting_wires: int,
    shots: int = _DEFAULT_SHOTS,
    seed: int | None = None,
) -> SingularValueResult:
    """Estimate the singular values of ``A`` from the phase of its embedding.

    ``A`` is spelled as the construction reads it, which is the one place this package
    keeps an upper-case parameter name and the reason the naming rule is suppressed above:
    every formula in the module docstring is stated in terms of ``A`` and of the embedding
    it is carried in.

    The readout is the singular value at the counting register's mode, read back as
    described in the module docstring: the embedding ``[[0, A], [A^T, 0]]`` is
    exponentiated as a dense matrix, the state whose overlap with each ``+sigma_i``
    eigenvector is ``sigma_i / ||A||_F`` and whose overlap with each ``-sigma_i``
    eigenvector is zero is prepared on the embedding's wires, and phase estimation reads
    the counting register. The embedding, its exponential and the input state's amplitudes
    are all formed classically; see the module docstring for what that costs and what it
    gives up.

    Args:
        A: The data matrix, a real floating-point tensor, square, of at least two rows,
            its dimension a power of two so that the embedding's wires are whole. It must
            not be the zero matrix, whose embedding has no positive subnormalisation to
            normalise by.
        n_counting_wires: The width of the counting register, at least one. The resolution
            is ``2 * alpha / 2**n_counting_wires`` in singular-value units.
        shots: The number of samples to draw, at least one. Every share in the returned
            distribution is a count over exactly this many samples.
        seed: The sampler's seed, or ``None`` to draw from the ambient generator. The same
            seed and the same arguments replay the same result exactly.

    Returns:
        The singular value read out at the counting register's mode, with that value's
        measured share, the counting register's distribution, the resolution they were
        read at and the subnormalisation they were read under. The module does not claim
        the readout is the largest singular value and does not predict which singular
        value it will be; use :meth:`SingularValueResult.within` to check a specific one.

    Raises:
        ValueError: If ``A`` is not a square two-dimensional real floating-point tensor
            whose dimension is a power of two of at least two and at most four; if ``A``
            holds a non-finite entry; if ``A`` is the zero matrix, whose embedding has no
            positive subnormalisation; if ``n_counting_wires`` is less than one; or if
            ``shots`` is less than one.
    """
    _validated_counting_width(n_counting_wires)
    _validated_shots(shots)
    embedding = _embedding_of(A)
    alpha = _subnormalisation(embedding, None)
    amplitudes = _input_amplitudes(A)
    circuit = _estimation_circuit(
        embedding,
        amplitudes,
        alpha=alpha,
        n_counting_wires=n_counting_wires,
    )
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    observed = circuit.counts(shots, generator=generator)[0]
    # ``counts`` is typed ``dict[str | int, int]`` because its ``format`` can be "int";
    # this call takes the default "bin", so every key is already the bit string this
    # mapping is typed as, and the coercion below is a no-op at run time.
    counts = {str(key): count for key, count in observed.items()}
    distribution = _counting_distribution(counts, n_counting_wires)
    mode = max(distribution, key=distribution.__getitem__)
    # The base is a float because the stubs type ``int ** int`` as ``Any``, since a
    # negative exponent yields a float, and an ``Any`` expression fails the type gate.
    resolution = 2.0 * alpha / 2.0**n_counting_wires
    return SingularValueResult(
        dominant_singular_value=2.0
        * alpha
        * (1.0 - int(mode, 2) / 2.0**n_counting_wires),
        dominant_share=distribution[mode],
        distribution=distribution,
        n_counting_wires=n_counting_wires,
        resolution=resolution,
        alpha=alpha,
    )


class _PhaseFromEmbedding:
    """``exp(-i * pi * H / alpha)`` in the controlled forms phase estimation needs.

    The unitary is one dense matrix, so every form is that matrix or a power of it with
    the identity prepended on the control branch: a controlled ``U`` is ``diag(I,
    U**power)`` in the control wire's block order, which is the block diagonal of the two.
    The unconditional ``apply`` embeds the matrix on the embedding's own wires.

    The exponential is taken of the block the private construction extracts, which is the
    embedding over ``alpha`` rather than the embedding itself. The readout's inversion is
    stated against ``alpha``, so a run taken against the unnormalised embedding would not
    be inverted by it.

    Phase estimation consumes the power form only; the other two are what the protocol
    asks of an operator that can also be applied plainly or under one control, and they
    are implemented rather than refused because a dense matrix is the case where both are
    well defined.
    """

    def __init__(self, embedding: torch.Tensor, alpha: float) -> None:
        block = embedding.to(device="cpu", dtype=torch.float64) / alpha
        self._matrix = torch.matrix_exp(-1j * math.pi * block.to(torch.complex128)).to(
            torch.complex64
        )
        self._n_wires = int(self._matrix.shape[0]).bit_length() - 1

    @property
    def n_wires(self) -> int:
        """The number of wires the embedding occupies, block qubit and data register."""
        return self._n_wires

    def apply(self, circuit: Circuit, wires: Sequence[int]) -> None:
        """Append ``U`` to ``circuit`` on the embedding's wires.

        Args:
            circuit: The circuit to extend.
            wires: The embedding's wires.
        """
        circuit.any(*wires, unitary=self._matrix, name="embedding_phase")

    def apply_controlled(
        self, circuit: Circuit, control: int, wires: Sequence[int]
    ) -> None:
        """Append ``U`` controlled on ``control``, its power form at exponent one.

        Args:
            circuit: The circuit to extend.
            control: The wire ``U`` is controlled on.
            wires: The embedding's wires.
        """
        self.apply_power_controlled(circuit, control, wires, 1)

    def apply_power_controlled(
        self, circuit: Circuit, control: int, wires: Sequence[int], power: int
    ) -> None:
        """Append ``U`` raised to ``power``, controlled on ``control``.

        The exponent is the counting wire's significance, so it reaches
        ``2 ** (n_counting_wires - 1)``; the power is taken of the unitary itself rather
        than of its exponent, which keeps the emitted gate one dense matrix on the
        embedding's wires.

        Args:
            circuit: The circuit to extend.
            control: The wire ``U`` is controlled on.
            wires: The embedding's wires.
            power: The exponent, at least zero.
        """
        block = torch.linalg.matrix_power(self._matrix, power)
        identity = torch.eye(block.shape[0], dtype=block.dtype, device=block.device)
        zeros = torch.zeros_like(block)
        controlled = torch.cat(
            (
                torch.cat((identity, zeros), dim=1),
                torch.cat((zeros, block), dim=1),
            ),
            dim=0,
        )
        circuit.any(control, *wires, unitary=controlled, name="embedding_phase_power")


# A block encoding is a construction the primitive layer would admit only if a second
# algorithm module needed one, and this is its only consumer: ``pca`` was measured not to
# need it, and no other Phase 2 unit encodes a matrix at all. Keep it here, private, until
# that changes; promote it to ``primitives/block_encoding.py`` if a second algorithm
# module ever needs a block encoding, since one consumer does not earn a module.
def _append_block_encoding(
    circuit: Circuit,
    embedding: torch.Tensor,
    alpha: float,
    *,
    ancilla: int,
    wires: Sequence[int],
) -> None:
    """Append the block encoding of ``embedding`` over ``alpha`` to ``circuit``.

    The construction is ``(PREP^dagger (x) I) . SELECT . (PREP (x) I)``. ``SELECT`` is
    diagonal in the ancilla, its ``|0>`` branch carrying the unitary ``U_0`` and its
    ``|1>`` branch the unitary ``U_1``, and the two are chosen so that
    ``(U_0 + U_1) / 2`` is the embedding over ``alpha``: each is diagonal in the
    embedding's own eigenbasis, and on an eigenvector of eigenvalue ``mu`` they act as the
    unit-modulus pair whose mean is ``mu / alpha``. ``PREP`` is then the preparation whose
    two amplitudes are equal, which is the Hadamard on the ancilla; the ``h`` gates below
    are that preparation and its adjoint, and they are the general construction's special
    case of one ancilla wire with equal coefficients rather than a decoration. The block
    the composition extracts is ``<0| U |0> = (U_0 + U_1) / 2`` on the embedding's wires.

    ``embedding`` must be Hermitian, which is what makes the eigenvalues real and the
    pair's mean their quotient by ``alpha``: the decomposition is the spectral one, and
    ``eigh`` reads one triangle of its argument, so a non-Hermitian matrix would be
    encoded from the wrong triangle rather than refused. Every call site of this module
    passes the Hermitian embedding that :func:`_embedding_of` forms.

    Args:
        circuit: The circuit to extend.
        embedding: The Hermitian matrix to encode, of power-of-two dimension.
        alpha: The subnormalisation, at least the embedding's spectral norm.
        ancilla: The wire the preparation and the selection are controlled on, outside
            ``wires``.
        wires: The embedding's wires, block qubit first.

    Raises:
        ValueError: If ``embedding`` is not Hermitian; or if ``alpha`` fails the
            validation :func:`_subnormalisation` applies to it, which is checked here
            because a factor below the embedding's spectral norm would otherwise reach the
            selection's arccosine and be folded into the unit circle silently.
    """
    alpha = _subnormalisation(embedding, alpha)
    circuit.gate("h", ancilla)
    circuit.any(
        ancilla,
        *wires,
        unitary=_select_unitary(embedding, alpha),
        name="block_encoding_select",
    )
    circuit.gate("h", ancilla)


def _select_unitary(embedding: torch.Tensor, alpha: float) -> torch.Tensor:
    """Return the ``SELECT`` matrix of the block encoding on the ancilla and ``wires``.

    The matrix is ``diag(U_0, U_1)`` in the ancilla's block order, which is the two
    branches' unitaries stacked along the diagonal the prepared ancilla selects between.
    Each branch is diagonal in the embedding's eigenbasis with eigenvalues of modulus one,
    and the two eigenvalues on an eigenvector of the embedding are the conjugate pair whose
    mean is that eigenvector's eigenvalue over ``alpha``.

    Args:
        embedding: The Hermitian matrix to encode.
        alpha: The subnormalisation, at least the embedding's spectral norm.

    Returns:
        The dense ``SELECT`` gate, ``2 ** (n + 1)`` by ``2 ** (n + 1)`` for an ``n``-wire
        embedding.

    Raises:
        ValueError: If ``embedding`` is not Hermitian.
    """
    if not bool(torch.allclose(embedding, embedding.T.conj())):
        raise ValueError(
            "the block encoding is built from the matrix's own eigenbasis, and eigh "
            "reads one triangle of its argument, so a matrix that is not Hermitian "
            "would be encoded from the wrong triangle rather than refused"
        )
    eigenvalues, eigenvectors = torch.linalg.eigh(embedding)
    # The eigenvalues of a contraction, and the pair of phases whose mean they are: the
    # cosine is the eigenvalue over alpha, and the two phases are its two arccosines.
    phases = torch.arccos((eigenvalues / alpha).clamp(-1.0, 1.0))
    basis = eigenvectors.to(torch.complex128)
    branch = []
    for sign in (1.0, -1.0):
        branch.append(
            basis @ torch.diag(torch.exp(1j * sign * phases)) @ basis.conj().T
        )
    zeros = torch.zeros_like(branch[0])
    return torch.cat(
        (
            torch.cat((branch[0], zeros), dim=1),
            torch.cat((zeros, branch[1]), dim=1),
        ),
        dim=0,
    )


def _embedding_of(A: torch.Tensor) -> torch.Tensor:  # noqa: N803
    """Validate ``A`` and return the Hermitian embedding ``[[0, A], [A^T, 0]]``.

    Args:
        A: The candidate data matrix.

    Returns:
        The embedding in double precision on the CPU, of shape ``(2 * n, 2 * n)`` for an
        ``n`` by ``n`` matrix.

    Raises:
        ValueError: If ``A`` is not a torch.Tensor; if it is not a square
            two-dimensional tensor; if it is not a real floating-point tensor; if its
            dimension is not a power of two of at least two, or exceeds the unit's bound;
            or if it holds a non-finite entry.
    """
    if not isinstance(A, torch.Tensor):
        raise ValueError(
            f"A must be a torch.Tensor, got {type(A).__name__}; the embedding is built "
            "from a data matrix, not from a description of one"
        )
    if A.dim() != 2:
        raise ValueError(
            "A must be a two-dimensional data matrix, got shape "
            f"{tuple(int(size) for size in A.shape)}; its two dimensions index the rows "
            "and the columns the embedding carries as one block"
        )
    rows, columns = (int(size) for size in A.shape)
    if rows != columns:
        raise ValueError(
            f"A must be square, got {rows} row(s) and {columns} column(s); the "
            "embedding carries the matrix and its transpose as the two off-diagonal "
            "blocks of one square matrix, which needs both sides to be the same width"
        )
    if not A.is_floating_point():
        raise ValueError(
            f"A must be a real floating-point tensor, got dtype {A.dtype}; the "
            "embedding is exponentiated as a dense matrix here, and an integer matrix "
            "would be silently promoted"
        )
    if rows < 2 or rows & (rows - 1):
        raise ValueError(
            f"A's dimension must be a power of two of at least two, got {rows}; a "
            "register of whole wires carries a power-of-two number of amplitudes"
        )
    if rows.bit_length() - 1 > _MAX_DATA_WIRES:
        raise ValueError(
            f"this unit is bounded at {_MAX_DATA_WIRES} index wires, so at most "
            f"{2**_MAX_DATA_WIRES} rows and columns, got {rows}: the embedding is twice "
            "as wide as the matrix and every form of the phase unitary is a dense gate "
            "on it, which is the demonstration scale the unit is written at"
        )
    if not bool(torch.isfinite(A).all()):
        raise ValueError(
            "A must be finite; a single non-finite entry makes every singular value "
            "undefined rather than merely inaccurate"
        )
    data = A.detach().to(device="cpu", dtype=torch.float64)
    embedding = torch.zeros((2 * rows, 2 * rows), dtype=torch.float64)
    embedding[:rows, rows:] = data
    embedding[rows:, :rows] = data.T
    return embedding


def _subnormalisation(embedding: torch.Tensor, alpha: float | None) -> float:
    """Return the subnormalisation a block encoding of ``embedding`` is read at.

    ``None`` is the embedding's Frobenius norm, which is at least its spectral norm and is
    available without diagonalising anything. A value is accepted when it is at least that
    spectral norm, and **refused when it is below it**: the extracted block would then be
    an operator of norm greater than one, and no unitary has such an operator as one of
    its blocks, so the construction has no encoding to build and would be building
    something else under the name of one.

    Args:
        embedding: The Hermitian matrix to encode.
        alpha: The candidate subnormalisation, or ``None`` for the embedding's Frobenius
            norm.

    Returns:
        The subnormalisation.

    Raises:
        ValueError: If the matrix is the zero matrix, whose Frobenius norm is not a
            positive factor; if ``alpha`` is not a positive finite real number; or if it
            is below the embedding's spectral norm.
    """
    if alpha is None:
        factor = float(torch.linalg.matrix_norm(embedding, ord="fro"))
        if factor == 0.0:
            raise ValueError(
                "the zero matrix has no embedding to normalise; its Frobenius norm is "
                "zero, so the block would be divided by nothing"
            )
        return factor
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError(
            f"the subnormalisation must be a real number, got {alpha!r}; it is the "
            "factor the block is smaller than the matrix by, and a flag or a tensor is "
            "not one"
        )
    factor = float(alpha)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError(
            f"the subnormalisation must be positive and finite, got {factor}; a "
            "non-positive factor is not what the block is smaller than the matrix by"
        )
    spectral = float(torch.linalg.matrix_norm(embedding, ord=2))
    if factor < spectral:
        raise ValueError(
            "the subnormalisation must be at least the embedding's spectral norm "
            f"{spectral}, got {factor}; the block would be that matrix over {factor}, an "
            "operator of norm greater than one, and no unitary has an operator of norm "
            "greater than one as one of its blocks, so this is not a matrix that can be "
            "encoded at this factor at all"
        )
    return factor


def _input_amplitudes(A: torch.Tensor) -> torch.Tensor:  # noqa: N803
    """Return the amplitudes of the state phase estimation is applied to.

    The state is the one whose overlap with the embedding's ``+sigma_i`` eigenvector is
    ``sigma_i / ||A||_F`` and whose overlap with the ``-sigma_i`` eigenvector is zero, so
    its squared overlap with that eigenvector is the singular value's squared share of
    ``||A||_F``, which is the weighting a vectorised matrix carries. In the embedding's
    basis the state is the sum of the ``+sigma`` eigenvectors weighted that way, and the
    amplitudes are the classical factors it is spelled from: the left singular vectors
    scaled by those weights on the block qubit's ``|0>`` branch, and the right ones on its
    ``|1>`` branch.

    The singular vectors are computed here classically, which is the premise the module
    docstring states: what the readout is an estimate of is the decomposition this
    function already holds.

    Args:
        A: The data matrix, validated as in :func:`_embedding_of`.

    Returns:
        The amplitudes, ``2 * n`` real values in double precision, the block qubit's
        ``|0>`` branch first.
    """
    data = A.detach().to(device="cpu", dtype=torch.float64)
    left, spectrum, right = torch.linalg.svd(data)
    weights = spectrum / float(torch.linalg.matrix_norm(data, ord="fro"))
    return torch.cat((left @ weights, right.T @ weights)) / math.sqrt(2.0)


def _estimation_circuit(
    embedding: torch.Tensor,
    amplitudes: torch.Tensor,
    *,
    alpha: float,
    n_counting_wires: int,
) -> Circuit:
    """Build the phase estimation circuit of ``embedding`` over ``alpha``.

    The circuit prepares the input state on the embedding's wires, then hands the counting
    register and the embedding's wires to
    :func:`~flagquantum.algorithms.primitives.phase_estimation.append_phase_estimation`,
    which emits the counting register's Hadamards, the controlled powers of the
    exponential, and the inverse Fourier transform itself.

    Args:
        embedding: The Hermitian embedding of the data matrix.
        amplitudes: The input state's amplitudes, the block qubit's ``|0>`` branch first.
        alpha: The subnormalisation the exponential is taken over.
        n_counting_wires: The width of the counting register, at least one.

    Returns:
        A circuit over ``n_counting_wires`` plus the embedding's own wires, the counting
        register leading.

    Raises:
        ValueError: If the amplitudes, the widths or the subnormalisation fail the
            validation :func:`~flagquantum.algorithms.primitives.state_preparation.append_arbitrary_state`,
            :func:`~flagquantum.algorithms.primitives.phase_estimation.append_phase_estimation`
            or :func:`_subnormalisation` applies to them.
    """
    n_embedding_wires = int(embedding.shape[0]).bit_length() - 1
    evaluation = list(range(n_counting_wires, n_counting_wires + n_embedding_wires))
    circuit = Circuit(n_counting_wires + n_embedding_wires)
    append_arbitrary_state(circuit, amplitudes, evaluation)
    append_phase_estimation(
        circuit,
        unitary=_PhaseFromEmbedding(embedding, alpha),
        counting_wires=list(range(n_counting_wires)),
        evaluation_wires=evaluation,
    )
    return circuit


def _counting_distribution(
    counts: Mapping[str, int], n_counting_wires: int
) -> dict[str, float]:
    """Return the counting register's marginal distribution as shares of the sample.

    Each count key carries every wire, so the embedding's wires are folded away before the
    distribution is formed: a key is truncated to its leading ``n_counting_wires``
    characters and its count accumulated onto that counter value. Reading the full-register
    keys instead would follow the embedding's wires wherever the counting register is
    correlated with them and return a distribution that is not the counting register's,
    without raising.

    Args:
        counts: The sample counts, keyed by the big-endian bit string of the whole
            register.
        n_counting_wires: The width of the counting register, at least one.

    Returns:
        One share per observed counter value, summing to one.
    """
    folded: dict[str, int] = {}
    for key, count in counts.items():
        counter = key[:n_counting_wires]
        folded[counter] = folded.get(counter, 0) + count
    total = sum(folded.values())
    return {key: count / total for key, count in folded.items()}


def _validated_counting_width(n_counting_wires: int) -> None:
    """Check the width of the counting register.

    Args:
        n_counting_wires: The register width as given.

    Raises:
        ValueError: If ``n_counting_wires`` is less than one.
    """
    if n_counting_wires < 1:
        raise ValueError(
            "singular-value estimation needs at least one counting wire, got "
            f"n_counting_wires={n_counting_wires}; a register with no wires resolves no "
            "phase and reads no singular value"
        )


def _validated_shots(shots: int) -> None:
    """Check the number of samples a run draws.

    Args:
        shots: The sample size as given.

    Raises:
        ValueError: If ``shots`` is less than one.
    """
    if shots < 1:
        raise ValueError(
            f"a run needs at least one shot, got shots={shots}; with no samples there is "
            "no counter value to read a singular value from"
        )

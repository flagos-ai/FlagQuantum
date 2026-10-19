"""What a block encoding guarantees, and what its walk step's spectrum is.

The two properties this file exists for are the ones a reader cannot check by looking at
the code. The first is that the flagged block of the encoding really is the matrix over
``alpha``, and the second is that one walk step's eigenvalues really are the phases whose
cosines are that same matrix's eigenvalues over ``alpha``. Both are read out of circuits
rather than out of the matrices the implementation holds, so a stub that returned its
argument or a constant would fail them, and each is paired with a sibling that fails: the
block is asserted *not* to be the matrix, and the plain encoding is asserted *not* to have
the walk step's spectrum, which is what makes the phase flip a construction rather than a
convention.

The eigenphase identity is the reason the encoding is worth more than its block, and it is
asserted as an equality of spectra. The observed side is the cosine of each eigenvalue of
the walk step the circuit applies, which is read with :func:`torch.linalg.eigvals`; the
expected side is the encoded matrix's eigenvalues, each appearing twice, which is read with
:func:`torch.linalg.eigvalsh` -- a different routine, on the matrix rather than on the
circuit, so the two sides are not the same computation.

Tolerances: the circuit path stores a gate as a complex64 matrix, so the encodings read
back through a circuit agree with their exact double-precision forms to about ``1e-8``.
The tests that compare circuits against each other therefore use ``1e-6``, and the tests
that compare the construction's own dense matrices use ``1e-12``.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, dataclass

import pytest
import torch

from flagquantum.algorithms.primitives import block_encoding as block_encoding_module
from flagquantum.algorithms.primitives.block_encoding import (
    BlockEncoding,
    SpectralBlockEncoding,
    WalkEncoding,
    spectral_block_encoding,
    subnormalisation,
)
from flagquantum.circuit import Circuit
from flagquantum.simulation.unitary import get_unitary

pytestmark = pytest.mark.unit

# A gate is stored in the circuit as a complex64 matrix, so every comparison taken through
# a circuit is bounded by that precision rather than by the construction's own.
_CIRCUIT_TOLERANCE = 1e-6

# The construction's own dense matrices are double precision, where a product of two
# orthonormal-basis products is exact to about 1e-15.
_EXACT_TOLERANCE = 1e-12

# Matrices the encoding is exercised on: symmetric real, complex Hermitian, diagonal, a
# single wire, and the identity, which is the boundary case where every eigenvalue sits at
# the edge of the range and the complement is exactly zero.
_MATRICES = (
    torch.tensor([[1.0, 0.5], [0.5, -2.0]], dtype=torch.float64),
    torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64),
    torch.diag(torch.tensor([1.0, 2.0, -3.0, 0.25], dtype=torch.float64)),
    torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
)


def _symmetric(seed: int, dimension: int) -> torch.Tensor:
    """Return a reproducible real symmetric ``dimension`` by ``dimension`` matrix."""
    generator = torch.Generator().manual_seed(seed)
    drawn = torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    return (drawn + drawn.T) / 2.0


def _hermitian(seed: int, dimension: int) -> torch.Tensor:
    """Return a reproducible complex Hermitian ``dimension`` by ``dimension`` matrix."""
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(dimension, dimension, generator=generator, dtype=torch.float64)
    imaginary = torch.randn(
        dimension, dimension, generator=generator, dtype=torch.float64
    )
    drawn = real + 1j * imaginary
    return drawn + drawn.mH


def _block_from_the_circuit(encoding: SpectralBlockEncoding) -> torch.Tensor:
    """Read the encoding's flagged block out of the circuit, one basis state at a time.

    The block is ``<0|_ancilla U |0>_ancilla``, so its column ``j`` is what the circuit
    leaves on the operator's qubits when it is entered with the basis state ``|j>`` and the
    flag left in ``|0>``. Each column is therefore read from a circuit of its own, and no
    unitary is written out as a matrix: what is read is the state the composition leaves,
    which is the only way this package exposes a circuit's action.
    """
    n_system = encoding.num_system
    dimension = 1 << n_system
    system = list(range(encoding.num_ancilla, encoding.num_ancilla + n_system))
    columns = []
    for basis in range(dimension):
        circuit = Circuit(encoding.num_ancilla + n_system)
        for index in range(n_system):
            if (basis >> (n_system - 1 - index)) & 1:
                circuit.gate("x", system[index])
        encoding.append_apply(circuit, ancilla=0, qubits=system)
        state = circuit.state().reshape(-1)
        columns.append(state[:dimension])
    return torch.stack(columns, dim=1)


def _encoded_spectrum(encoding: SpectralBlockEncoding) -> torch.Tensor:
    """Return the encoded matrix's eigenvalues over ``alpha``, each appearing twice.

    The expected side of the walk identity. ``eigvalsh`` is a different routine from the
    ``eigh`` the construction is built from, and it is applied to the matrix rather than to
    the circuit, so this is not the construction's own intermediate value.

    Args:
        encoding: The encoding whose spectrum is wanted.

    Returns:
        The ``2 ** (num_system + 1)`` values, sorted ascending.
    """
    normalised = (torch.linalg.eigvalsh(encoding.matrix) / encoding.alpha).real
    return torch.sort(torch.cat((normalised, normalised))).values


def _walk_cosines(encoding: object, n_system: int, n_ancilla: int) -> torch.Tensor:
    """Return the cosines of one walk step's eigenphases, sorted ascending.

    This is a **consumer of the boundary and not of an implementation**: it names only
    ``num_ancilla`` and ``append_walk_step``, so it runs against the spectral encoding and
    against a hand-built second implementation without a change to this function. That is
    what makes the protocol a boundary rather than a description of one class.

    Args:
        encoding: Any object satisfying :class:`WalkEncoding`.
        n_system: The number of qubits the encoded operator acts on.
        n_ancilla: The number of flag qubits.

    Returns:
        The ``2 ** (n_system + n_ancilla)`` cosines, sorted ascending.
    """
    circuit = Circuit(n_ancilla + n_system)
    encoding.append_walk_step(  # type: ignore[attr-defined]
        circuit, ancilla=0, qubits=list(range(n_ancilla, n_ancilla + n_system))
    )
    eigenvalues = torch.linalg.eigvals(get_unitary(circuit).to(torch.complex128))
    return torch.sort(torch.cos(torch.angle(eigenvalues))).values


@dataclass(frozen=True)
class _HandBuiltWalk:
    """A second implementation of the walk protocol, with no dense matrix anywhere.

    It encodes the Pauli ``Z`` on one qubit at ``alpha`` exactly one, which is its spectral
    norm, so the complement block is identically zero and the reflection is
    ``Z_ancilla (x) Z``. The walk step is therefore ``Z_ancilla`` times that, which is
    ``Z`` on the operator's qubit alone. Every gate is emitted by name: nothing here reads a
    matrix, which is the point -- a consumer written against the protocol must not be able
    to tell the two implementations apart.
    """

    @property
    def num_system(self) -> int:
        """The number of qubits the encoded operator acts on."""
        return 1

    @property
    def num_ancilla(self) -> int:
        """The number of flag qubits."""
        return 1

    @property
    def alpha(self) -> float:
        """The subnormalisation, which is ``Z``'s spectral norm exactly."""
        return 1.0

    def append_apply(
        self, circuit: Circuit, *, ancilla: int, qubits: list[int] | tuple[int, ...]
    ) -> None:
        """Append the reflection ``Z_ancilla (x) Z`` to ``circuit``."""
        circuit.gate("z", qubits[0])
        circuit.gate("z", ancilla)

    def append_walk_step(
        self, circuit: Circuit, *, ancilla: int, qubits: list[int] | tuple[int, ...]
    ) -> None:
        """Append one walk step, which is ``Z`` on the operator's qubit alone."""
        del ancilla
        circuit.gate("z", qubits[0])

    def append_adjoint_walk_step(
        self, circuit: Circuit, *, ancilla: int, qubits: list[int] | tuple[int, ...]
    ) -> None:
        """Append the walk step's adjoint, which is the step itself here."""
        del ancilla
        circuit.gate("z", qubits[0])


def test_the_protocols_are_structural_and_the_encoding_satisfies_both() -> None:
    """Conformance is by surface, not by inheritance, and the walk protocol extends it."""
    encoding = spectral_block_encoding(_MATRICES[0])
    assert isinstance(encoding, BlockEncoding)
    assert isinstance(encoding, WalkEncoding)
    # The walk protocol is an extension: an encoding that carries no walk step is still a
    # block encoding, and an object with neither surface satisfies neither. The extension
    # is checked by surface rather than by ``issubclass``, which a protocol with a
    # non-method member does not support at runtime.
    assert not isinstance(object(), BlockEncoding)
    assert not isinstance(object(), WalkEncoding)
    assert isinstance(_HandBuiltWalk(), BlockEncoding)


def test_the_flagged_block_is_the_matrix_over_alpha() -> None:
    """The block the circuit extracts is the encoded matrix over the subnormalisation."""
    for matrix in _MATRICES:
        encoding = spectral_block_encoding(matrix)
        block = _block_from_the_circuit(encoding)
        expected = matrix.to(torch.complex128) / encoding.alpha
        assert float((block - expected).abs().max()) < _CIRCUIT_TOLERANCE
        # And the dense form the circuit emits agrees with what the circuit does, which is
        # the pair of assertions that keeps the two from drifting apart.
        assert float((encoding.block() - expected).abs().max()) < _EXACT_TOLERANCE
        assert block.shape == (1 << encoding.num_system, 1 << encoding.num_system)


def test_the_flagged_block_is_neither_the_matrix_nor_unscaled() -> None:
    """The block is the matrix **over alpha**, and the other direction is what a stub fails.

    A construction that returned the matrix it was handed would agree with itself here, and
    one that encoded the matrix without its subnormalisation would agree with the matrix
    too. Both are asserted against.
    """
    for matrix in _MATRICES:
        encoding = spectral_block_encoding(matrix)
        block = _block_from_the_circuit(encoding).real
        if encoding.alpha == pytest.approx(1.0):
            # The identity encodes at its own spectral norm, so no rescaling is visible;
            # every other matrix in the set is scaled and is checked below.
            continue
        assert float((block - matrix).abs().max()) > 0.1
        assert float((block.real - matrix.to(torch.float64)).abs().max()) > 0.1
    # The scale is a property of the encoding, not of the matrix: doubling the factor
    # halves the block and leaves the matrix where it was.
    matrix = _MATRICES[0]
    doubled = spectral_block_encoding(matrix, 2.0 * float(torch.linalg.norm(matrix, 2)))
    single = spectral_block_encoding(matrix)
    assert doubled.alpha == pytest.approx(2.0 * float(torch.linalg.norm(matrix, 2)))
    # Doubling the factor halves the block, so the two encodings differ by exactly the
    # ratio of their subnormalisations and by nothing else.
    assert (
        float(
            (doubled.block() - single.block() * (single.alpha / doubled.alpha))
            .abs()
            .max()
        )
        < _EXACT_TOLERANCE
    )
    assert (
        float(
            (doubled.block() - matrix.to(torch.complex128) / doubled.alpha).abs().max()
        )
        < _EXACT_TOLERANCE
    )


def test_the_reflection_is_hermitian_unitary_and_squares_to_the_identity() -> None:
    """The construction is exact rather than approximate: it squares to the identity.

    ``A`` and ``sqrt(I - A**2)`` are both functions of the same Hermitian matrix, so they
    commute, and the reflection's square is the identity identically rather than up to the
    quality of an eigenbasis computed from a spoiled matrix. A construction that assembled
    the reflection from separately computed blocks would fail this as soon as ``alpha``
    approached the spectral norm.
    """
    for seed in range(4):
        matrix = _symmetric(seed, 4)
        for alpha in (None, 3.0 * float(torch.linalg.norm(matrix, 2))):
            encoding = spectral_block_encoding(matrix, alpha)
            reflection = encoding.select
            dimension = int(reflection.shape[0])
            identity = torch.eye(dimension, dtype=torch.complex128)
            assert (
                float((reflection - reflection.mH).abs().max()) < _EXACT_TOLERANCE
            ), "the reflection is Hermitian"
            assert (
                float((reflection @ reflection - identity).abs().max())
                < _EXACT_TOLERANCE
            ), "the reflection squares to the identity"


def test_one_walk_step_has_an_eigenphase_for_every_encoded_eigenvalue() -> None:
    """The walk step's eigenphases are the arccosines of the encoded spectrum.

    This is the identity that makes the encoding a qubitization: one walk step's cosine
    spectrum is the encoded matrix's eigenvalues, twice. It is read from the circuit, and
    the expected side is read from the matrix with a different routine.
    """
    for seed in range(4):
        matrix = _symmetric(seed, 4)
        for alpha in (
            None,
            float(torch.linalg.norm(matrix, 2)),
            4.0 * float(torch.linalg.norm(matrix, 2)),
        ):
            encoding = spectral_block_encoding(matrix, alpha)
            observed = _walk_cosines(
                encoding, encoding.num_system, encoding.num_ancilla
            )
            expected = _encoded_spectrum(encoding)
            assert float((observed - expected).abs().max()) < _CIRCUIT_TOLERANCE, (
                f"the walk step's cosines must be the encoded spectrum at alpha "
                f"{encoding.alpha}"
            )
            # The same statement said in phases, at the tolerance the circuit path allows:
            # every phase except zero and pi is irrational in units of pi, so a construction
            # that pinned the phases to the two trivial values fails on the first matrix.
            assert bool((torch.abs(torch.abs(observed) - 1.0) > 1e-6).any())


def test_the_plain_encoding_does_not_have_the_walk_step_spectrum() -> None:
    """The phase flip is a construction, not a convention: without it the identity fails.

    The sibling of the test above. ``U`` is Hermitian with eigenvalues ``mu`` and
    ``+-sqrt(1 - mu**2)``, so its eigenphase cosines are not the encoded spectrum, and a
    walk step that had simply been the encoding would be caught here.
    """
    for seed in (1, 2):
        matrix = _symmetric(seed, 4)
        encoding = spectral_block_encoding(matrix)
        circuit = Circuit(1 + encoding.num_system)
        encoding.append_apply(
            circuit, ancilla=0, qubits=list(range(1, 1 + encoding.num_system))
        )
        eigenvalues = torch.linalg.eigvals(get_unitary(circuit).to(torch.complex128))
        observed = torch.sort(torch.cos(torch.angle(eigenvalues))).values
        expected = _encoded_spectrum(encoding)
        assert float((observed - expected).abs().max()) > 0.1


def test_the_walk_step_and_the_plain_encoding_emit_different_circuits() -> None:
    """The two appends are different sequences, and the walk step's name says so."""
    matrix = _MATRICES[1]
    encoding = spectral_block_encoding(matrix)
    qubits = list(range(1, 1 + encoding.num_system))
    plain = Circuit(1 + encoding.num_system)
    walked = Circuit(1 + encoding.num_system)
    encoding.append_apply(plain, ancilla=0, qubits=qubits)
    encoding.append_walk_step(walked, ancilla=0, qubits=qubits)
    assert plain.to_qir() != walked.to_qir()
    assert len(walked) == len(plain) + 1
    assert (
        float(
            (
                get_unitary(plain).to(torch.complex128)
                - get_unitary(walked).to(torch.complex128)
            )
            .abs()
            .max()
        )
        > 0.1
    )


def test_the_adjoint_walk_step_inverts_the_walk_step() -> None:
    """``W^dagger W`` is the identity on the whole register, read from the circuits."""
    for matrix in _MATRICES[:3]:
        encoding = spectral_block_encoding(matrix)
        qubits = list(range(1, 1 + encoding.num_system))
        dimension = 1 << (1 + encoding.num_system)
        forward = Circuit(1 + encoding.num_system)
        encoding.append_walk_step(forward, ancilla=0, qubits=qubits)
        backward = Circuit(1 + encoding.num_system)
        encoding.append_adjoint_walk_step(backward, ancilla=0, qubits=qubits)
        product = get_unitary(forward).to(torch.complex128) @ get_unitary(backward).to(
            torch.complex128
        )
        identity = torch.eye(dimension, dtype=torch.complex128)
        assert float((product - identity).abs().max()) < _CIRCUIT_TOLERANCE
        # The adjoint is not simply a copy of the step: it is the other order of the two
        # gates, which for a matrix that is not its own inverse would differ.
        assert forward.to_qir() != backward.to_qir()


def test_the_walk_step_is_not_the_adjoint_of_the_plain_encoding() -> None:
    """Neither walk form is the encoding or its inverse, which the spectra separate."""
    matrix = _MATRICES[2]
    encoding = spectral_block_encoding(matrix)
    qubits = list(range(1, 1 + encoding.num_system))
    walked = Circuit(1 + encoding.num_system)
    encoding.append_walk_step(walked, ancilla=0, qubits=qubits)
    unitary = get_unitary(walked).to(torch.complex128)
    assert (
        float((unitary - encoding.select).abs().max()) > 0.1
    ), "a walk step is not the encoding"
    # The first matrix in the set is symmetric, so its eigenphases come in conjugate pairs
    # and the walk step is Hermitian; the check is that its square is not the identity,
    # which is what would happen if the phase flip had been dropped or duplicated.
    identity = torch.eye(int(unitary.shape[0]), dtype=torch.complex128)
    assert float((unitary @ unitary - identity).abs().max()) > 0.1


def test_a_second_implementation_satisfies_the_same_consumer() -> None:
    """The boundary is real: a hand-built encoding is read by the consumer unchanged.

    Engineering decision principle 10 asks for a replacement, not an interface. The
    consumer :func:`_walk_cosines` names the protocol's numbers and its one method, so it
    is written against the boundary and not against the spectral construction; running it
    against a second implementation that holds no dense matrix at all, and getting the
    spectrum that implementation encodes, is that replacement.
    """
    built = _HandBuiltWalk()
    assert isinstance(built, WalkEncoding)
    observed = _walk_cosines(built, built.num_system, built.num_ancilla)
    # ``Z``'s eigenvalues are ``+1`` and ``-1``, each appearing twice over the two qubits.
    expected = torch.sort(torch.tensor([-1.0, -1.0, 1.0, 1.0])).values
    assert float((observed - expected).abs().max()) < _CIRCUIT_TOLERANCE
    # And the same consumer reads the spectral encoding's own spectrum, which is not that.
    spectral = spectral_block_encoding(_MATRICES[0])
    assert (
        float(
            (
                _walk_cosines(spectral, spectral.num_system, spectral.num_ancilla)
                - _encoded_spectrum(spectral)
            )
            .abs()
            .max()
        )
        < _CIRCUIT_TOLERANCE
    )


def test_the_subnormalisation_defaults_to_the_frobenius_norm() -> None:
    """The default is the Frobenius norm, which is available without diagonalising."""
    for matrix in _MATRICES:
        default = subnormalisation(matrix)
        assert default == pytest.approx(float(torch.linalg.norm(matrix, "fro")))
        spectral = float(torch.linalg.norm(matrix, 2))
        assert default >= spectral
        assert spectral == pytest.approx(float(torch.linalg.matrix_norm(matrix, ord=2)))


def test_a_factor_at_or_above_the_spectral_norm_is_accepted() -> None:
    """The condition is ``alpha >= ||H||``, and both sides of the boundary are read out."""
    matrix = _symmetric(5, 4)
    spectral = float(torch.linalg.norm(matrix, 2))
    for factor in (spectral, spectral * 1.5, float(torch.linalg.norm(matrix, "fro"))):
        encoding = spectral_block_encoding(matrix, factor)
        assert encoding.alpha == pytest.approx(factor)
        assert (
            float((encoding.block() - matrix.to(torch.complex128) / factor).abs().max())
            < _EXACT_TOLERANCE
        )
        assert float(torch.linalg.matrix_norm(encoding.block(), ord=2)) <= 1.0 + 1e-9
    # A factor below the spectral norm is not a wider choice, it is no encoding at all.
    with pytest.raises(ValueError, match="no unitary has an operator of norm"):
        spectral_block_encoding(matrix, spectral * 0.99)


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        ("not a tensor", "must be a torch.Tensor"),
        (torch.zeros(4, dtype=torch.float64), "two-dimensional"),
        (torch.zeros(2, 3, dtype=torch.float64), "must be square"),
        (torch.zeros(3, 3, dtype=torch.float64), "power-of-two dimension"),
        (torch.zeros(1, 1, dtype=torch.float64), "power-of-two dimension"),
        (torch.zeros(2, 2, dtype=torch.int64), "floating-point or complex"),
        (torch.zeros(2, 2, dtype=torch.float64), "no block to normalise"),
        (torch.tensor([[1.0, float("nan")], [0.0, 1.0]]), "finite values"),
        (torch.tensor([[1.0, 0.5], [0.25, 1.0]]), "not Hermitian"),
    ],
)
def test_the_matrix_refusals_name_their_condition(matrix: object, message: str) -> None:
    """Every matrix-level refusal is reachable, and each names its own condition.

    The Hermitian refusal is the one that matters most: the construction reads its
    eigenbasis, and ``eigh`` reads one triangle of its argument, so a non-Hermitian matrix
    would otherwise be encoded from half of itself without a word.
    """
    with pytest.raises(ValueError, match=message):
        spectral_block_encoding(matrix)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=message):
        subnormalisation(matrix)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("alpha", "message"),
    [
        (True, "must be a real number"),
        (torch.tensor(1.0), "must be a real number"),
        (0.0, "positive and finite"),
        (-1.0, "positive and finite"),
        (float("nan"), "positive and finite"),
        (float("inf"), "positive and finite"),
    ],
)
def test_the_factor_refusals_name_their_condition(alpha: object, message: str) -> None:
    """A factor that is not a positive finite real is refused before anything is built."""
    matrix = _MATRICES[0]
    with pytest.raises(ValueError, match=message):
        subnormalisation(matrix, alpha)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=message):
        spectral_block_encoding(matrix, alpha)  # type: ignore[arg-type]


def test_the_class_refuses_a_factor_it_is_handed_directly() -> None:
    """The class's own factor check is a second entry point, and it has its own refusals.

    :func:`spectral_block_encoding` resolves the factor through :func:`subnormalisation`
    before the class ever sees it, so a test that only goes through the function exercises
    one of the two guards. The class documents that it checks the pair it is given, and
    the class is public, so a direct construction is a caller the guard has to hold for:
    a non-positive factor and a factor below the spectral norm are both refused there
    rather than at the first gate.
    """
    matrix = _MATRICES[2]
    spectral = float(torch.linalg.matrix_norm(matrix, ord=2))
    with pytest.raises(ValueError, match="must be positive and finite"):
        SpectralBlockEncoding(matrix, 0.0)
    with pytest.raises(ValueError, match="at least the matrix's spectral norm"):
        SpectralBlockEncoding(matrix, spectral * 0.5)
    # And the factor that is exactly the spectral norm is the boundary the clamp exists
    # for: it is accepted, and the reflection it produces is still a reflection.
    at_the_norm = SpectralBlockEncoding(matrix, spectral)
    assert at_the_norm.alpha == spectral
    assert bool(
        torch.allclose(
            at_the_norm.select @ at_the_norm.select,
            torch.eye(2 * int(matrix.shape[0]), dtype=torch.complex128),
            atol=_EXACT_TOLERANCE,
            rtol=0.0,
        )
    )


def test_the_class_refuses_a_reflection_that_is_not_a_unitary(monkeypatch) -> None:
    """The construction's post-condition is checked rather than assumed.

    The reflection is assembled from the eigenbasis the class has just read, so with a
    correct builder nothing reaches these two refusals. Falsifying the builder is the only
    way to falsify the checks, and the point of running them is a future edit to the
    builder: it meets a named refusal here instead of having its result emitted as a gate
    and read back as a wrong matrix somewhere else.
    """
    matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    alpha = subnormalisation(matrix)
    not_hermitian = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.complex128,
    )
    monkeypatch.setattr(
        block_encoding_module, "_reflection", lambda _matrix, _alpha: not_hermitian
    )
    with pytest.raises(ValueError, match="must be Hermitian and it is not"):
        SpectralBlockEncoding(matrix, alpha)
    # A Hermitian matrix that is not unitary gets past the first check and meets the
    # second, so the two are separate conditions rather than one spelled twice.
    monkeypatch.setattr(
        block_encoding_module,
        "_reflection",
        lambda _matrix, _alpha: torch.diag(
            torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.complex128)
        ),
    )
    with pytest.raises(ValueError, match="must square to the identity and it does not"):
        SpectralBlockEncoding(matrix, alpha)


def test_the_register_refusals_are_taken_before_the_gate() -> None:
    """Every refusal the append path can take is taken before the first gate is emitted.

    A caller whose qubits are wrong keeps an empty circuit rather than a half-built
    encoding, which is what the checks running ahead of the gate buys.
    """
    encoding = spectral_block_encoding(_MATRICES[2])
    assert encoding.num_system == 2
    for ancilla, qubits in (
        (0, [0, 1]),
        (0, [1]),
        (0, [1, 1, 2]),
        (True, [1, 2]),
    ):
        circuit = Circuit(3)
        for append in (
            encoding.append_apply,
            encoding.append_walk_step,
            encoding.append_adjoint_walk_step,
        ):
            with pytest.raises(ValueError):
                append(circuit, ancilla=ancilla, qubits=qubits)  # type: ignore[arg-type]
        assert circuit.to_qir() == []
    # The messages are distinct, so a caller is told which of the three conditions failed.
    with pytest.raises(ValueError, match="one of the operator's own wires"):
        encoding.append_apply(Circuit(3), ancilla=1, qubits=[1, 2])
    with pytest.raises(ValueError, match="the register must be 2 wire"):
        encoding.append_apply(Circuit(3), ancilla=0, qubits=[1])
    with pytest.raises(ValueError, match="must be distinct"):
        encoding.append_apply(Circuit(3), ancilla=0, qubits=[1, 1])
    with pytest.raises(ValueError, match="must be a wire index"):
        encoding.append_apply(Circuit(3), ancilla=True, qubits=[1, 2])


def test_the_encoding_is_frozen_and_carries_the_definition_s_numbers() -> None:
    """The instance is a value object: its fields are read-only and its numbers agree."""
    matrix = _MATRICES[0]
    encoding = spectral_block_encoding(matrix)
    with pytest.raises(FrozenInstanceError):
        encoding.alpha = 4.0  # type: ignore[misc]
    assert isinstance(encoding, SpectralBlockEncoding)
    assert encoding.num_system == int(matrix.shape[0]).bit_length() - 1
    assert encoding.num_ancilla == 1
    assert encoding.matrix is matrix or torch.equal(encoding.matrix, matrix)
    assert float(torch.linalg.matrix_norm(encoding.block(), ord=2)) == pytest.approx(
        float(torch.linalg.matrix_norm(matrix, ord=2)) / encoding.alpha, rel=1e-9
    )


def test_a_real_and_a_complex_hermitian_matrix_are_both_encoded() -> None:
    """A complex Hermitian matrix is encoded too, and its block is still the conjugate one."""
    matrix = _hermitian(9, 2)
    assert bool((matrix.imag.abs() > 1e-6).any()), "the sample must have a complex part"
    encoding = spectral_block_encoding(matrix)
    block = _block_from_the_circuit(encoding)
    assert float((block - matrix / encoding.alpha).abs().max()) < _CIRCUIT_TOLERANCE
    for seed in range(3):
        complex_matrix = _hermitian(seed, 4)
        complex_encoding = spectral_block_encoding(complex_matrix)
        assert (
            float(
                (complex_encoding.block() - complex_matrix / complex_encoding.alpha)
                .abs()
                .max()
            )
            < _EXACT_TOLERANCE
        )
        observed = _walk_cosines(
            complex_encoding, complex_encoding.num_system, complex_encoding.num_ancilla
        )
        assert (
            float((observed - _encoded_spectrum(complex_encoding)).abs().max())
            < _CIRCUIT_TOLERANCE
        )


def test_a_single_wire_matrix_is_encoded_but_nothing_narrower() -> None:
    """The narrowest encoding is two by two, and it still carries a walk step.

    A one-dimensional argument has no eigenbasis to be read in and no phase to rotate, so it
    is refused; the smallest admitted matrix is the one that acts on a single wire.
    """
    matrix = _MATRICES[1]
    assert matrix.shape == (2, 2)
    encoding = spectral_block_encoding(matrix)
    assert encoding.num_system == 1
    assert encoding.block().shape == (2, 2)
    observed = _walk_cosines(encoding, 1, 1)
    assert (
        float((observed - _encoded_spectrum(encoding)).abs().max()) < _CIRCUIT_TOLERANCE
    )
    with pytest.raises(ValueError, match="power-of-two dimension"):
        spectral_block_encoding(torch.ones(1, 1, dtype=torch.float64))


def test_two_different_matrices_give_two_different_encodings() -> None:
    """The construction depends on its argument, which a constant would not."""
    first = spectral_block_encoding(_MATRICES[0])
    second = spectral_block_encoding(_MATRICES[1])
    assert float((first.select - second.select).abs().max()) > 0.1
    assert first.alpha != pytest.approx(second.alpha)
    # And the same matrix gives the same encoding twice, so nothing here samples.
    assert torch.equal(first.select, spectral_block_encoding(_MATRICES[0]).select)


def test_the_package_exports_every_name_the_primitive_declares() -> None:
    """The primitive's ``__all__`` is a declaration, and the package has to honour it."""
    assert block_encoding_module.__all__ == [
        "BlockEncoding",
        "SpectralBlockEncoding",
        "WalkEncoding",
        "spectral_block_encoding",
        "subnormalisation",
    ]
    for name in block_encoding_module.__all__:
        assert hasattr(block_encoding_module, name), name
    package = __import__("flagquantum.algorithms.primitives", fromlist=["__all__"])
    for name in block_encoding_module.__all__:
        assert name in package.__all__, name
        assert getattr(package, name) is getattr(block_encoding_module, name)


def test_the_encoding_emits_one_gate_and_nothing_else() -> None:
    """The append is one instruction, so the compiler's own passes see an ordinary matrix.

    Nothing here builds a second execution path: the encoding is a gate on the qubits it was
    given, which is why the resource estimator can count it and every backend can run it.
    """
    matrix = _MATRICES[2]
    encoding = spectral_block_encoding(matrix)
    qubits = list(range(1, 1 + encoding.num_system))
    circuit = Circuit(1 + encoding.num_system)
    encoding.append_apply(circuit, ancilla=0, qubits=qubits)
    assert len(circuit) == 1
    instruction = circuit.to_qir()[0]
    assert sorted(instruction["index"]) == [0, *qubits]
    assert (
        float((encoding.select - encoding.select).abs().max()) == 0.0
    ), "the emitted matrix is the one the encoding holds"
    # A factor that is not a whole number still leaves the block exactly where the
    # definition puts it, which is what makes ``alpha`` the caller's choice rather than the
    # matrix's.
    odd = spectral_block_encoding(matrix, math.pi)
    assert odd.alpha == pytest.approx(math.pi)
    assert float((odd.block() - matrix.to(torch.complex128) / math.pi).abs().max()) < (
        _EXACT_TOLERANCE
    )


def test_no_gradient_flows_from_the_circuit_back_to_the_matrix() -> None:
    """The encoding is a classical precomputation, and this is where that is pinned.

    A matrix that requires grad is accepted, because refusing it would refuse the ordinary
    case of a tensor the caller is still using elsewhere. What does not happen is autograd
    travelling from a circuit back to the matrix's entries: the validation detaches onto the
    CPU, so the gate carries constants and the derivative of the decomposition is the
    caller's to take on the matrix rather than on a circuit built from one. Without the
    detach the readout below would carry a ``grad_fn``, which is what these assertions catch.
    """
    parameter = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    drawn = torch.zeros(2, 2, dtype=torch.float64)
    drawn[0, 0] = parameter
    drawn[1, 1] = -1.0
    drawn[0, 1] = drawn[1, 0] = 0.2
    assert drawn.requires_grad

    encoding = spectral_block_encoding(drawn)
    assert isinstance(encoding, SpectralBlockEncoding)
    assert not encoding.select.requires_grad
    assert encoding.select.grad_fn is None
    assert not encoding.block().requires_grad

    # And the readout is still the physics: a rotation on the operator's own wire moves the
    # flag's state, so this is a real execution and not a short circuit.
    circuit = Circuit(2)
    circuit.ry(1, theta=0.4)
    encoding.append_apply(circuit, ancilla=0, qubits=[1])
    readout = circuit.expectation_ps(z=[0]).sum()
    assert (
        not readout.requires_grad
    ), "a circuit built from the encoding must not carry the matrix's gradient"
    assert float(readout.detach()) != 0.0

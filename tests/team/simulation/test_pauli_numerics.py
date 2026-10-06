from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from flagquantum.simulation.pauli import (
    exponential_pauli_operator,
    infer_n_wires_from_dense_state,
    normalized_pauli_basis,
    pauli_product_density_expectation,
    pauli_product_operator,
    pauli_product_statevector_expectation,
    pauli_words,
)

pytestmark = pytest.mark.unit

_COMPLEX128 = torch.complex128

# One case is (theta, word, targets, n_wires, the amplitudes a Pauli rotation leaves
# on the all-zero input).  The amplitudes are the analytic closed form
# cos(theta) on the states the word fixes and -i sin(theta) on the states it flips,
# and the first five cases are the call forms the reference implementation documents
# for this operation.
_REFERENCE_ROTATIONS = (
    (1.0, "XX", (0, 1), 2, {"00": 0.540302305868140, "11": -0.841470984807897j}),
    (
        math.pi / 2,
        "YX",
        (0, 2),
        3,
        {"101": 1.0 + 0.0j},
    ),
    (
        math.pi / 2,
        "X",
        (1,),
        3,
        {"010": -1.0j},
    ),
    (
        math.pi / 2,
        "XXX",
        (2, 0, 1),
        3,
        {"111": -1.0j},
    ),
    (
        math.pi / 2,
        "XZ",
        (0, 1),
        2,
        {"10": -1.0j},
    ),
)


def _oracle_operators(
    word: str, targets: tuple[int, ...]
) -> tuple[tuple[int, str], ...]:
    return tuple(zip(targets, word.lower(), strict=True))


def _diagonalization_oracle(
    theta: float,
    word: str,
    targets: tuple[int, ...],
    n_wires: int,
) -> torch.Tensor:
    """Exponentiate through an eigendecomposition of the Pauli product.

    A Pauli product is Hermitian with eigenvalues in ``{+1, -1}``, so this oracle
    reaches the answer by a different route than the closed form: it never assumes
    the product squares to the identity, it derives that from the spectrum.
    """

    product = pauli_product_operator(
        _oracle_operators(word, targets),
        n_wires,
        dtype=_COMPLEX128,
        device="cpu",
    )
    values, vectors = torch.linalg.eigh(product)
    phases = torch.exp((-1j * float(theta)) * values)
    return (vectors * phases) @ vectors.conj().T


def _power_series_oracle(
    theta: float,
    word: str,
    targets: tuple[int, ...],
    n_wires: int,
    terms: int = 90,
) -> torch.Tensor:
    """Sum the defining power series of the exponential term by term."""

    product = pauli_product_operator(
        _oracle_operators(word, targets),
        n_wires,
        dtype=_COMPLEX128,
        device="cpu",
    )
    generator = (-1j * float(theta)) * product
    eye = torch.eye(2**n_wires, dtype=_COMPLEX128)
    total, term = eye, eye
    for order in range(1, terms):
        term = term @ generator / order
        total = total + term
    return total


def _amplitudes(
    theta: float,
    word: str,
    targets: tuple[int, ...],
    n_wires: int,
) -> dict[str, complex]:
    """Rotate the all-zero state and report the amplitudes on the reachable basis states."""

    basis = torch.zeros(2**n_wires, dtype=_COMPLEX128)
    basis[0] = 1.0
    vector = (
        exponential_pauli_operator(
            theta, word, targets, n_wires, dtype=_COMPLEX128, device="cpu"
        )
        @ basis
    )
    return {
        format(index, f"0{n_wires}b"): complex(vector[index])
        for index in range(2**n_wires)
        if float(vector[index].abs()) > 1e-12
    }


@pytest.mark.parametrize(
    ("theta", "word", "targets", "n_wires"),
    (
        (0.0, "XX", (0, 1), 2),
        (0.3, "X", (0,), 1),
        (1.0, "ZZ", (0, 1), 2),
        (math.pi / 2, "YX", (0, 2), 3),
        (-2.7, "YZIX", (0, 1, 2, 3), 4),
        (6.9, "IIII", (0, 1, 2, 3), 4),
        (0.3, "", (), 2),
    ),
)
def test_the_closed_form_agrees_with_diagonalization_and_a_power_series(
    theta: float, word: str, targets: tuple[int, ...], n_wires: int
) -> None:
    actual = exponential_pauli_operator(
        theta, word, targets, n_wires, dtype=_COMPLEX128, device="cpu"
    )

    torch.testing.assert_close(
        actual,
        _diagonalization_oracle(theta, word, targets, n_wires),
        atol=1e-14,
        rtol=0.0,
    )
    torch.testing.assert_close(
        actual,
        _power_series_oracle(theta, word, targets, n_wires),
        atol=1e-14,
        rtol=0.0,
    )
    identity = torch.eye(2**n_wires, dtype=_COMPLEX128)
    torch.testing.assert_close(actual.conj().T @ actual, identity, atol=1e-14, rtol=0.0)


@pytest.mark.parametrize(
    ("theta", "word", "targets", "n_wires", "expected"),
    _REFERENCE_ROTATIONS,
)
def test_the_rotation_reproduces_the_reference_amplitudes(
    theta: float,
    word: str,
    targets: tuple[int, ...],
    n_wires: int,
    expected: dict[str, complex],
) -> None:
    measured = _amplitudes(theta, word, targets, n_wires)

    assert set(measured) == set(expected)
    for label, amplitude in expected.items():
        assert measured[label] == pytest.approx(amplitude, abs=1e-14)


def test_the_rotation_is_the_negative_exponential_of_the_product() -> None:
    """The reference operation applies ``exp(-i theta P)``, not its conjugate."""

    theta = 0.7
    actual = exponential_pauli_operator(
        theta, "X", (0,), 1, dtype=_COMPLEX128, device="cpu"
    )
    x = torch.tensor([[0, 1], [1, 0]], dtype=_COMPLEX128)
    angle = torch.tensor(theta, dtype=torch.float64)
    identity = torch.eye(2, dtype=_COMPLEX128)
    positive = torch.cos(angle) * identity + 1j * torch.sin(angle) * x
    negative = torch.cos(angle) * identity - 1j * torch.sin(angle) * x

    torch.testing.assert_close(actual, negative, atol=1e-14, rtol=0.0)
    # The two exponentials are numerically far apart, so the assertion above is a
    # real discrimination rather than a tolerance accident.
    assert float((actual - positive).abs().max()) > 1e-2


def test_an_all_identity_word_returns_the_global_phase_it_denotes() -> None:
    theta = 0.9
    phase = complex(math.cos(theta), -math.sin(theta))

    for n_wires in (1, 2, 3):
        actual = exponential_pauli_operator(
            theta,
            "I" * n_wires,
            tuple(range(n_wires)),
            n_wires,
            dtype=_COMPLEX128,
            device="cpu",
        )
        expected = phase * torch.eye(2**n_wires, dtype=_COMPLEX128)
        torch.testing.assert_close(actual, expected, atol=1e-14, rtol=0.0)
        # Reading an all-identity word as the identity would drop the phase, so the
        # reference is separated from the identity by more than half a unit.
        dropped = exponential_pauli_operator(
            0.0,
            "I" * n_wires,
            tuple(range(n_wires)),
            n_wires,
            dtype=_COMPLEX128,
            device="cpu",
        )
        assert float((actual - dropped).abs().max()) > 0.5


@pytest.mark.parametrize(
    ("word", "targets", "bare_targets"),
    (
        ("IX", (0, 1), (1,)),
        ("XI", (0, 1), (0,)),
        ("XYI", (0, 1, 2), (0, 1)),
        ("IIZ", (0, 1, 2), (2,)),
    ),
)
def test_an_identity_character_consumes_a_target_without_acting_on_it(
    word: str, targets: tuple[int, ...], bare_targets: tuple[int, ...]
) -> None:
    theta = 0.5
    n_wires = max(targets) + 1

    padded = exponential_pauli_operator(
        theta, word, targets, n_wires, dtype=_COMPLEX128, device="cpu"
    )
    bare = exponential_pauli_operator(
        theta,
        "".join(character for character in word if character != "I"),
        bare_targets,
        n_wires,
        dtype=_COMPLEX128,
        device="cpu",
    )

    assert torch.equal(padded, bare)


def test_the_word_places_its_characters_in_the_given_order() -> None:
    """``"XZ"`` on ``(0, 1)`` flips wire 0 while ``"ZX"`` flips wire 1."""

    forward = _amplitudes(math.pi / 2, "XZ", (0, 1), 2)
    reversed_word = _amplitudes(math.pi / 2, "ZX", (0, 1), 2)

    assert set(forward) == {"10"}
    assert set(reversed_word) == {"01"}


def test_the_operator_is_embedded_on_the_wires_the_circuit_declares() -> None:
    theta = 0.5
    identity = torch.eye(2, dtype=_COMPLEX128)
    embedded = exponential_pauli_operator(
        theta, "X", (2,), 3, dtype=_COMPLEX128, device="cpu"
    )
    x = torch.tensor([[0, 1], [1, 0]], dtype=_COMPLEX128)
    explicit = math.cos(theta) * torch.eye(8, dtype=_COMPLEX128) - 1j * math.sin(
        theta
    ) * torch.kron(torch.kron(identity, identity), x)

    torch.testing.assert_close(embedded, explicit, atol=1e-14, rtol=0.0)


@pytest.mark.parametrize("dtype", (torch.complex64, torch.complex128))
def test_the_dense_form_keeps_the_requested_dtype(dtype: torch.dtype) -> None:
    actual = exponential_pauli_operator(
        0.4, "XYZ", (0, 1, 2), 3, dtype=dtype, device="cpu"
    )

    assert actual.dtype == dtype
    assert actual.shape == (8, 8)


@pytest.mark.parametrize(
    ("word", "targets"),
    (("XY", (0,)), ("X", (0, 1)), ("", (0,))),
)
def test_a_word_and_its_targets_must_describe_the_same_product(
    word: str, targets: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match="acts on exactly one target"):
        exponential_pauli_operator(
            0.3, word, targets, 2, dtype=_COMPLEX128, device="cpu"
        )


# A lowercase alphabet letter is the sharp case here: this constructor accepts the
# reference implementation's own spelling, which is uppercase, and does not fold
# case, even though the product reader beside it does.  The rest place a character
# outside the alphabet at a different position each: the annihilation letter
# another operator algebra uses, a punctuation mark, a digit, whitespace, and an
# escaped character.
@pytest.mark.parametrize(
    "word", ("x", "yz", "i", "A", "xA", "XXZ?", "x x", "1", "ZZ\\n")
)
def test_a_character_outside_the_pauli_alphabet_is_refused(word: str) -> None:
    targets = tuple(range(len(word)))
    with pytest.raises(ValueError, match="X, Y, Z, or I"):
        exponential_pauli_operator(
            0.3, word, targets, len(word), dtype=_COMPLEX128, device="cpu"
        )


@pytest.mark.parametrize("targets", ((0, 0), (1, 2, 1), (0, 0, 0)))
def test_a_repeated_target_is_refused(targets: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="must be unique"):
        exponential_pauli_operator(
            0.3,
            "X" * len(targets),
            targets,
            3,
            dtype=_COMPLEX128,
            device="cpu",
        )


@pytest.mark.parametrize("targets", ((-1,), (0, -2)))
def test_a_negative_target_is_refused(targets: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        exponential_pauli_operator(
            0.3,
            "X" * len(targets),
            targets,
            2,
            dtype=_COMPLEX128,
            device="cpu",
        )


def test_a_target_beyond_the_declared_wires_is_refused() -> None:
    with pytest.raises(ValueError, match="wire index out of range"):
        exponential_pauli_operator(
            0.3, "XX", (0, 2), 2, dtype=_COMPLEX128, device="cpu"
        )


@pytest.mark.parametrize("target", (True, 0.5, "1", None, (0,)))
def test_a_target_that_is_not_an_integer_is_refused(target: object) -> None:
    with pytest.raises(ValueError, match="target must be an integer"):
        exponential_pauli_operator(
            0.3, "X", (target,), 2, dtype=_COMPLEX128, device="cpu"
        )


def test_integral_target_labels_from_other_libraries_are_accepted() -> None:
    theta = 0.5

    labels = exponential_pauli_operator(
        theta,
        "X",
        (np.int64(1),),
        2,
        dtype=_COMPLEX128,
        device="cpu",
    )
    tensors = exponential_pauli_operator(
        theta,
        "X",
        (torch.tensor(1, dtype=torch.int32),),
        2,
        dtype=_COMPLEX128,
        device="cpu",
    )

    assert torch.equal(labels, tensors)


def test_a_word_that_is_not_a_string_is_refused() -> None:
    with pytest.raises(ValueError, match="must be a string"):
        exponential_pauli_operator(
            0.3, ("X",), (0,), 1, dtype=_COMPLEX128, device="cpu"  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "theta", (float("inf"), float("-inf"), float("nan"), True, "0.7", None, [0.1])
)
def test_an_angle_that_does_not_denote_a_finite_number_is_refused(
    theta: object,
) -> None:
    with pytest.raises(ValueError, match="theta must be"):
        exponential_pauli_operator(
            theta,  # type: ignore[arg-type]
            "X",
            (0,),
            1,
            dtype=_COMPLEX128,
            device="cpu",
        )


def test_pauli_product_matches_for_statevector_density_and_dense_operator() -> None:
    state = torch.tensor(
        [[2**-0.5, 0.0, 0.0, 2**-0.5]],
        dtype=torch.complex128,
    )
    density = state.unsqueeze(-1) * state.conj().unsqueeze(-2)
    operators = ((0, "x"), (1, "x"))

    state_value = pauli_product_statevector_expectation(state, operators, 2)
    density_value = pauli_product_density_expectation(density, operators, 2)
    operator = pauli_product_operator(
        operators,
        2,
        dtype=state.dtype,
        device=state.device,
    )
    direct_value = torch.real(
        torch.einsum("bi,ij,bj->b", state.conj(), operator, state)
    )

    torch.testing.assert_close(state_value, torch.ones(1, dtype=torch.float64))
    torch.testing.assert_close(density_value, state_value)
    torch.testing.assert_close(direct_value, state_value)


def test_dense_state_dimension_must_be_a_power_of_two() -> None:
    assert infer_n_wires_from_dense_state(torch.zeros(2, 8)) == 3
    with pytest.raises(ValueError, match="power of two"):
        infer_n_wires_from_dense_state(torch.zeros(2, 6))


@pytest.mark.parametrize("name", ("rx", "ry", "rz"))
def test_pauli_products_reject_parameterized_gates(name: str) -> None:
    state = torch.tensor([1, 0], dtype=torch.complex128)
    operators = ((0, name),)
    with pytest.raises(ValueError, match="fixed gate matrices"):
        pauli_product_operator(operators, 1, dtype=state.dtype, device=state.device)
    with pytest.raises(ValueError, match="fixed gate matrices"):
        pauli_product_statevector_expectation(state, operators, 1)


@pytest.mark.parametrize("density_mode", (False, True))
def test_pauli_z_expectation_has_analytic_rotation_gradient(density_mode: bool) -> None:
    theta = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    state = torch.stack((torch.cos(theta / 2), torch.sin(theta / 2))).to(
        torch.complex128
    )
    operators = ((0, "z"),)
    if density_mode:
        density = state[:, None] * state.conj()[None, :]
        value = pauli_product_density_expectation(density, operators, 1)
    else:
        value = pauli_product_statevector_expectation(state, operators, 1)

    gradient = torch.autograd.grad(value.sum(), theta)[0]
    torch.testing.assert_close(value.squeeze(), theta.cos())
    torch.testing.assert_close(gradient, -theta.sin())


_PAULI_LITERALS = {
    # Written out here rather than read from the gate table, so this oracle cannot
    # inherit a convention error from the table the code under test reads. Each
    # character maps to (real part, imaginary part), each a 2x2 literal.
    "I": (((1.0, 0.0), (0.0, 1.0)), ((0.0, 0.0), (0.0, 0.0))),
    "X": (((0.0, 1.0), (1.0, 0.0)), ((0.0, 0.0), (0.0, 0.0))),
    "Y": (((0.0, 0.0), (0.0, 0.0)), ((0.0, -1.0), (1.0, 0.0))),
    "Z": (((1.0, 0.0), (0.0, -1.0)), ((0.0, 0.0), (0.0, 0.0))),
}


def _literal_pauli_matrix(word: str) -> torch.Tensor:
    """Build one word's matrix by an explicit Kronecker product of literals."""

    result = torch.ones(1, 1, dtype=_COMPLEX128)
    for character in word:
        real, imaginary = _PAULI_LITERALS[character]
        entry = torch.complex(
            torch.tensor(real, dtype=torch.float64),
            torch.tensor(imaginary, dtype=torch.float64),
        )
        result = torch.kron(result, entry)
    return result.to(_COMPLEX128)


def test_pauli_words_enumerates_every_word_once_identity_first() -> None:
    assert pauli_words(0) == ("",)
    assert pauli_words(1) == ("I", "X", "Y", "Z")
    # The first qubit is the most significant character, so the identity word
    # leads and the all-Z word closes the list.
    assert pauli_words(2) == (
        "II",
        "IX",
        "IY",
        "IZ",
        "XI",
        "XX",
        "XY",
        "XZ",
        "YI",
        "YX",
        "YY",
        "YZ",
        "ZI",
        "ZX",
        "ZY",
        "ZZ",
    )
    for n_qubits in (0, 1, 2, 3):
        words = pauli_words(n_qubits)
        assert len(words) == 4**n_qubits
        assert len(set(words)) == len(words)
        assert all(len(word) == n_qubits for word in words)


def test_pauli_words_refuses_a_count_that_is_not_a_non_negative_integer() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        pauli_words(1.0)
    with pytest.raises(ValueError, match="must be an integer"):
        pauli_words(True)
    with pytest.raises(ValueError, match="must be non-negative"):
        pauli_words(-1)


def test_the_normalized_pauli_basis_is_orthonormal_in_both_dtypes() -> None:
    for n_qubits in (1, 2, 3):
        dimension = 2**n_qubits
        for dtype in (torch.complex64, torch.complex128):
            entries = normalized_pauli_basis(n_qubits, dtype=dtype, device="cpu")
            expected = 4**n_qubits
            assert len(entries) == expected
            assert all(entry.shape == (dimension, dimension) for entry in entries)
            assert all(entry.dtype == dtype for entry in entries)

            stacked = torch.stack(list(entries)).reshape(expected, expected)
            gram = stacked.mH @ stacked
            deviation = float(
                torch.max(torch.abs(gram - torch.eye(expected, dtype=dtype)))
            )
            # The residual is bounded by the dtype's own epsilon rather than by a
            # number chosen here: a normalized Pauli family is orthonormal exactly,
            # so anything above a few epsilons would be a scale error.
            assert deviation <= 8.0 * float(torch.finfo(dtype).eps)


def test_each_normalized_basis_element_is_its_word_over_root_dimension() -> None:
    for n_qubits in (1, 2):
        dimension = 2**n_qubits
        entries = normalized_pauli_basis(n_qubits, dtype=_COMPLEX128, device="cpu")
        words = pauli_words(n_qubits)
        assert len(entries) == len(words)
        for word, entry in zip(words, entries, strict=True):
            expected = _literal_pauli_matrix(word) / math.sqrt(dimension)
            torch.testing.assert_close(entry, expected)


def test_the_normalized_basis_honours_its_declared_device() -> None:
    # ``meta`` records the requested placement without needing an accelerator, so
    # this checks that the argument reaches every element rather than being
    # accepted and dropped.
    entries = normalized_pauli_basis(1, dtype=_COMPLEX128, device="meta")
    assert all(entry.device.type == "meta" for entry in entries)


def test_the_normalized_pauli_basis_refuses_a_negative_count() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        normalized_pauli_basis(-1, dtype=_COMPLEX128, device="cpu")

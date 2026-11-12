"""Real optional upstream inputs checked against independent matrix action."""

from __future__ import annotations

import itertools
import pathlib
import typing
from collections.abc import Callable
from importlib import import_module

import pytest
import sympy
import torch

import flagquantum as fq
from flagquantum.algorithms.chemistry import uccsd_ansatz
from flagquantum.algorithms.fermion.encoding import Encoding, encode_fermion
from flagquantum.algorithms.fermion.hubbard import fermi_hubbard_chain
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.core.algebra.pauli import PauliOperator, PauliTerm
from flagquantum.ecosystem.openfermion import (
    from_openfermion_fermion,
    from_openfermion_interaction,
    from_openfermion_qubit,
    to_openfermion_fermion,
    to_openfermion_qubit,
)
from flagquantum.observables.fermion import FermionOperator, FermionTerm
from tests.unit.fermion.oracles import fock_matrix

if typing.TYPE_CHECKING:
    from openfermion import MolecularData as OpenMolecularData

of = pytest.importorskip("openfermion")
# NumPy is required by the installed upstream package; a broken install must fail.
np = import_module("numpy")
pytestmark = pytest.mark.openfermion


def _ordered_matrix(
    terms: dict[tuple[tuple[int, int], ...], complex], width: int
) -> torch.Tensor:
    result = torch.zeros((2**width, 2**width), dtype=torch.complex128)
    for column in range(2**width):
        for word, coefficient in terms.items():
            bits = [(column >> (width - 1 - q)) & 1 for q in range(width)]
            amplitude = coefficient
            for mode, action in reversed(word):
                if bits[mode] == action:
                    amplitude = 0j
                    break
                amplitude *= (-1) ** sum(bits[:mode])
                bits[mode] = action
            row = sum(bit << (width - 1 - q) for q, bit in enumerate(bits))
            result[row, column] += amplitude
    return result


def test_all_short_ordered_words_and_complex_round_trips() -> None:
    for size in range(5):
        for word in itertools.product(((0, 0), (0, 1), (1, 0), (1, 1)), repeat=size):
            source = of.FermionOperator(word, 0.7 - 0.3j)
            before = dict(source.terms)
            native = from_openfermion_fermion(source)
            torch.testing.assert_close(
                fock_matrix(native, 2), _ordered_matrix(before, 2)
            )
            exported = to_openfermion_fermion(native)
            assert from_openfermion_fermion(exported) == native
            # Compare actual upstream actions, not its different normal-order keys.
            torch.testing.assert_close(
                torch.tensor(
                    of.get_sparse_operator(exported, n_qubits=2).toarray(),
                    dtype=torch.complex128,
                ),
                torch.tensor(
                    of.get_sparse_operator(source, n_qubits=2).toarray(),
                    dtype=torch.complex128,
                ),
            )
            assert source.terms == before


@pytest.fixture
def molecular_data_source(tmp_path: pathlib.Path) -> OpenMolecularData:
    from flagquantum.algorithms.chemistry_domain.problem import from_native_molecule
    from flagquantum.algorithms.molecular import create_molecular_hamiltonian

    geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))]
    native = from_native_molecule(create_molecular_hamiltonian(geometry))
    source = of.MolecularData(
        geometry, "sto-3g", 1, filename=str(tmp_path / "molecule")
    )
    source.n_orbitals = native.n_orbitals
    source.n_qubits = native.n_modes
    source.one_body_integrals = native.one_body.numpy().copy()
    source.two_body_integrals = native.two_body.permute(0, 2, 3, 1).numpy().copy()
    source.nuclear_repulsion = native.nuclear_repulsion
    source.hf_energy = native.reference_energy
    return source


def test_molecular_data_matches_upstream_hamiltonian_and_file_input(
    molecular_data_source: OpenMolecularData,
) -> None:
    from flagquantum.algorithms.chemistry_domain.problem import fermion_hamiltonian
    from flagquantum.ecosystem.chemistry.molecular_data import (
        from_openfermion_molecular_data,
    )
    from flagquantum.ecosystem.chemistry.molecular_hdf5 import read_molecular_hdf5

    source = molecular_data_source
    problem = from_openfermion_molecular_data(source)
    source.save()
    file_problem = read_molecular_hdf5(source.filename + ".hdf5")
    torch.testing.assert_close(problem.one_body, file_problem.one_body)
    torch.testing.assert_close(problem.two_body, file_problem.two_body)
    assert problem.energy_offset == file_problem.energy_offset
    assert problem.reference_energy == file_problem.reference_energy
    assert problem.reference_occupation == file_problem.reference_occupation
    native = fermion_hamiltonian(problem)
    torch.testing.assert_close(
        fock_matrix(native, 4),
        torch.tensor(
            of.get_sparse_operator(source.get_molecular_hamiltonian()).toarray()
        ),
    )
    encodings: tuple[Encoding, ...] = ("jordan_wigner", "parity", "bravyi_kitaev")
    for encoding in encodings:
        encoded = encode_fermion(native, n_modes=4, encoding=encoding)
        circuit = fq.Circuit(4, dtype=torch.complex128, device="cpu")
        for q, bit in enumerate(encoded.reference_bits(problem.reference_occupation)):
            if bit:
                circuit.x(q)
        result = fq.run(
            circuit, outputs=[fq.expectation(to_observable(encoded.operator))]
        )
        assert float(
            torch.as_tensor(result.expectation()).reshape(-1)[0]
        ) == pytest.approx(-1.1166843871, abs=1e-8)
    source.one_body_integrals[:] = float("nan")
    source.two_body_integrals[:] = float("nan")
    torch.testing.assert_close(problem.one_body, file_problem.one_body)
    torch.testing.assert_close(problem.two_body, file_problem.two_body)


@pytest.mark.parametrize("dtype", ["float32", "int64", ">f8"])
def test_molecular_data_numeric_storage_and_owned_native_dtype(
    molecular_data_source: OpenMolecularData,
    dtype: str,
) -> None:
    from flagquantum.ecosystem.chemistry.molecular_data import (
        from_openfermion_molecular_data,
    )

    source = molecular_data_source
    source.one_body_integrals = source.one_body_integrals.astype(dtype)
    source.two_body_integrals = source.two_body_integrals.astype(dtype)
    source.hf_energy = None
    source.n_qubits = None
    problem = from_openfermion_molecular_data(source)
    assert problem.one_body.dtype == problem.two_body.dtype == torch.float64
    assert problem.one_body.device.type == "cpu"
    assert problem.reference_energy is None and problem.n_modes == 4
    assert float(problem.one_body[0, 0]) == float(source.one_body_integrals[0, 0])


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_orbitals", 0),
        ("n_orbitals", 33),
        ("n_orbitals", True),
        ("n_electrons", 1),
        ("n_electrons", 6),
        ("n_electrons", 2.0),
        ("multiplicity", 3),
        ("multiplicity", True),
        ("n_qubits", 3),
        ("nuclear_repulsion", float("inf")),
        ("hf_energy", float("nan")),
        ("nuclear_repulsion", 0.5j),
        ("hf_energy", False),
    ],
)
def test_molecular_data_corrupted_metadata(
    molecular_data_source: OpenMolecularData,
    field: str,
    value: object,
) -> None:
    from flagquantum.ecosystem.chemistry.molecular_data import (
        from_openfermion_molecular_data,
    )

    setattr(molecular_data_source, field, value)
    with pytest.raises((TypeError, ValueError)):
        from_openfermion_molecular_data(molecular_data_source)


@pytest.mark.parametrize("field", ["one_body_integrals", "two_body_integrals"])
@pytest.mark.parametrize("dtype", ["bool", "complex128", "object"])
def test_molecular_data_unsupported_integral_storage(
    molecular_data_source: OpenMolecularData,
    field: str,
    dtype: str,
) -> None:
    from flagquantum.ecosystem.chemistry.molecular_data import (
        from_openfermion_molecular_data,
    )

    array = getattr(molecular_data_source, field)
    setattr(molecular_data_source, field, array.astype(dtype))
    with pytest.raises(ValueError, match="real integral tensor"):
        from_openfermion_molecular_data(molecular_data_source)


def test_molecular_data_missing_cache_never_reads_a_file(
    molecular_data_source: OpenMolecularData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flagquantum.ecosystem.chemistry.molecular_data import (
        from_openfermion_molecular_data,
    )

    def forbidden(name: str) -> None:
        raise AssertionError("object conversion must never load integrals from a file")

    source = molecular_data_source
    monkeypatch.setattr(source, "get_from_file", forbidden)
    assert from_openfermion_molecular_data(source).n_modes == 4
    source.one_body_integrals = None
    with pytest.raises(TypeError, match="cached dense"):
        from_openfermion_molecular_data(source)


def test_molecular_data_rejects_invalid_tensor_values_and_symmetry(
    molecular_data_source: OpenMolecularData,
) -> None:
    from flagquantum.ecosystem.chemistry.molecular_data import (
        from_openfermion_molecular_data,
    )

    source = molecular_data_source
    source.one_body_integrals[0, 1] += 0.3
    with pytest.raises(ValueError, match="symmetric"):
        from_openfermion_molecular_data(source)
    source.one_body_integrals[0, 1] -= 0.3
    source.two_body_integrals[0, 0, 0, 1] += 0.3
    with pytest.raises(ValueError, match="symmetry"):
        from_openfermion_molecular_data(source)
    source.two_body_integrals[:] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        from_openfermion_molecular_data(source)


def test_molecular_data_numpy_integer_metadata_and_boolean_rejection(
    molecular_data_source: OpenMolecularData,
) -> None:
    from flagquantum.ecosystem.chemistry.molecular_data import (
        from_openfermion_molecular_data,
    )

    source = molecular_data_source
    integer = source.one_body_integrals.astype("int64")[0, 0]
    source.n_orbitals = type(integer)(2)
    source.n_electrons = type(integer)(2)
    source.n_qubits = type(integer)(4)
    source.multiplicity = type(integer)(1)
    assert from_openfermion_molecular_data(source).n_modes == 4
    source.multiplicity = source.one_body_integrals.astype("bool")[0, 0]
    with pytest.raises(TypeError, match="multiplicity"):
        from_openfermion_molecular_data(source)


def test_molecular_data_rejects_wrong_external_type() -> None:
    from flagquantum.ecosystem.chemistry.molecular_data import (
        from_openfermion_molecular_data,
    )

    with pytest.raises(TypeError, match="MolecularData"):
        from_openfermion_molecular_data(of.FermionOperator())


def test_pauli_complex_sum_against_upstream_matrix_and_independent_factors() -> None:
    source = of.QubitOperator("X0 Y2", 0.5j) + of.QubitOperator("Z1", -2)
    source += of.QubitOperator((), 0.3 - 0.2j)
    native = from_openfermion_qubit(source)
    matrix = pauli_matrix(native, n_qubits=3)
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    z = torch.diag(torch.tensor([1, -1], dtype=torch.complex128))
    eye = torch.eye(2, dtype=torch.complex128)
    expected = 0.5j * torch.kron(torch.kron(x, eye), y)
    expected -= 2 * torch.kron(torch.kron(eye, z), eye)
    expected += (0.3 - 0.2j) * torch.eye(8, dtype=torch.complex128)
    torch.testing.assert_close(matrix, expected)
    torch.testing.assert_close(
        matrix, torch.tensor(of.get_sparse_operator(source, n_qubits=3).toarray())
    )
    exported = to_openfermion_qubit(native)
    assert from_openfermion_qubit(exported) == native
    exported.terms.clear()
    assert native.terms


def test_tiny_coefficients_identity_zero_and_ownership() -> None:
    fermion = FermionOperator((FermionTerm(1e-30, (0,), ()), FermionTerm(2j)))
    pauli = PauliOperator((PauliTerm(1e-30, ((0, "x"),)), PauliTerm(2j)))
    assert from_openfermion_fermion(to_openfermion_fermion(fermion)) == fermion
    assert from_openfermion_qubit(to_openfermion_qubit(pauli)) == pauli
    assert from_openfermion_fermion(of.FermionOperator()) == FermionOperator()
    assert from_openfermion_qubit(of.QubitOperator()) == PauliOperator()
    assert to_openfermion_fermion(FermionOperator()).terms == {}
    assert to_openfermion_qubit(PauliOperator()).terms == {}
    source = of.FermionOperator("0 0^", 2)
    native = from_openfermion_fermion(source)
    source.terms.clear()
    assert fock_matrix(native, 1)[0, 0].item() == 2


def test_converted_operator_uses_unchanged_measurement_consumer() -> None:
    source = of.FermionOperator("0^ 0", 1.5) + of.FermionOperator("1^ 1", -0.5)
    source += of.FermionOperator((), 0.25)
    native = from_openfermion_fermion(source)
    encodings: tuple[Encoding, ...] = ("jordan_wigner", "parity", "bravyi_kitaev")
    for encoding in encodings:
        encoded = encode_fermion(native, n_modes=3, encoding=encoding)
        circuit = fq.Circuit(3, dtype=torch.complex128)
        for q, bit in enumerate(encoded.reference_bits((1, 1, 0))):
            if bit:
                circuit.x(q)
        value = fq.run(
            circuit, outputs=[fq.expectation(to_observable(encoded.operator))]
        )
        assert float(
            torch.as_tensor(value.expectation()).reshape(-1)[0]
        ) == pytest.approx(1.25)


@pytest.mark.parametrize("convert", [from_openfermion_fermion, from_openfermion_qubit])
@pytest.mark.parametrize("source", [None, {}, 1, FermionOperator(), PauliOperator()])
def test_wrong_external_type(
    convert: Callable[[object], object], source: object
) -> None:
    with pytest.raises(TypeError):
        convert(source)


@pytest.mark.parametrize("convert", [to_openfermion_fermion, to_openfermion_qubit])
def test_wrong_native_type(convert: Callable[[object], object]) -> None:
    with pytest.raises(TypeError):
        convert(of.FermionOperator())


@pytest.mark.parametrize("kind", ["fermion", "qubit"])
@pytest.mark.parametrize(
    "coefficient",
    [
        True,
        float("nan"),
        float("inf"),
        1j * float("inf"),
        "x",
        sympy.Symbol("x"),
        torch.tensor(1.0),
        torch.tensor(1.0, requires_grad=True),
    ],
)
def test_invalid_external_coefficient(kind: str, coefficient: object) -> None:
    source = of.FermionOperator() if kind == "fermion" else of.QubitOperator()
    source.terms = {(): coefficient}
    convert = from_openfermion_fermion if kind == "fermion" else from_openfermion_qubit
    with pytest.raises((TypeError, ValueError)):
        convert(source)


@pytest.mark.parametrize(
    "kind,word",
    [
        ("fermion", ((0, 2),)),
        ("fermion", ((0, True),)),
        ("fermion", ((-1, 0),)),
        ("fermion", ((True, 0),)),
        ("fermion", ((0.0, 1),)),
        ("fermion", ((0,),)),
        ("qubit", ((0, "A"),)),
        ("qubit", ((0, "x"),)),
        ("qubit", ((0, "X"), (0, "Y"))),
        ("qubit", ((-1, "Z"),)),
        ("qubit", ((True, "Z"),)),
        ("qubit", "X0"),
    ],
)
def test_corrupted_term_dictionary(kind: str, word: object) -> None:
    source = of.FermionOperator() if kind == "fermion" else of.QubitOperator()
    source.terms = {word: 1}
    convert = from_openfermion_fermion if kind == "fermion" else from_openfermion_qubit
    with pytest.raises((TypeError, ValueError)):
        convert(source)


def test_corrupted_mapping_and_wrong_upstream_operator_kind() -> None:
    source = of.FermionOperator()
    source.terms = []
    with pytest.raises(TypeError, match="mapping"):
        from_openfermion_fermion(source)
    with pytest.raises(TypeError):
        from_openfermion_fermion(of.QubitOperator())
    with pytest.raises(TypeError):
        from_openfermion_qubit(of.FermionOperator())


def test_interaction_complex_coefficients_against_two_matrix_oracles() -> None:
    rng = np.random.default_rng(427)
    width = 3
    one = rng.normal(size=(width, width)) + 1j * rng.normal(size=(width, width))
    two = rng.normal(size=(width,) * 4) + 1j * rng.normal(size=(width,) * 4)
    source = of.InteractionOperator(0.3 - 0.2j, one, two)
    words: dict[tuple[tuple[int, int], ...], complex] = {(): complex(source.constant)}
    for p, q in itertools.product(range(width), repeat=2):
        words[((p, 1), (q, 0))] = one[p, q]
    for p, q, r, s in itertools.product(range(width), repeat=4):
        words[((p, 1), (q, 1), (r, 0), (s, 0))] = two[p, q, r, s]
    native = from_openfermion_interaction(source)
    matrix = fock_matrix(native, width)
    torch.testing.assert_close(matrix, _ordered_matrix(words, width))
    torch.testing.assert_close(
        matrix, torch.tensor(of.get_sparse_operator(source).toarray())
    )
    assert native == from_openfermion_fermion(of.get_fermion_operator(source))
    one[:] = 0
    two[:] = 0
    source.constant = 0
    torch.testing.assert_close(fock_matrix(native, width), matrix)


@pytest.mark.parametrize("dtype", ["int64", "uint32", "float32", "complex128"])
def test_interaction_numeric_dtypes_and_quartic_order(dtype: str) -> None:
    one = np.zeros((2, 2), dtype=dtype)
    two = np.zeros((2,) * 4, dtype=dtype)
    two[1, 0, 0, 1] = 3
    native = from_openfermion_interaction(of.InteractionOperator(2, one, two))
    assert native == FermionOperator((FermionTerm(2), FermionTerm(-3, (0, 1), (0, 1))))


def test_interaction_zero_width_and_unpruned_coefficients() -> None:
    source = of.InteractionOperator(0, np.zeros((0, 0)), np.zeros((0,) * 4))
    assert from_openfermion_interaction(source) == FermionOperator()
    source.constant = 0.5j
    assert from_openfermion_interaction(source) == FermionOperator((FermionTerm(0.5j),))
    one = np.diag([1e-30, 0.0])
    two = np.zeros((2,) * 4)
    two[0, 1, 0, 1] = 2e-30
    source = of.InteractionOperator(3e-30, one, two)
    assert from_openfermion_interaction(source) == FermionOperator(
        (
            FermionTerm(3e-30),
            FermionTerm(1e-30, (0,), (0,)),
            FermionTerm(2e-30, (0, 1), (0, 1)),
        )
    )


@pytest.mark.parametrize("encoding", ["jordan_wigner", "parity", "bravyi_kitaev"])
def test_interaction_uses_existing_encoding_and_measurement(encoding: Encoding) -> None:
    source = of.InteractionOperator(0.25, np.diag([1.0, 2.0]), np.zeros((2,) * 4))
    encoded = encode_fermion(
        from_openfermion_interaction(source), n_modes=2, encoding=encoding
    )
    circuit = fq.Circuit(2, dtype=torch.complex128)
    for q, bit in enumerate(encoded.reference_bits((1, 0))):
        if bit:
            circuit.x(q)
    value = fq.run(circuit, outputs=[fq.expectation(to_observable(encoded.operator))])
    assert float(torch.as_tensor(value.expectation()).reshape(-1)[0]) == pytest.approx(
        1.25
    )


@pytest.mark.parametrize("location", ["constant", "one", "two"])
@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), complex(0, float("inf"))]
)
def test_interaction_nonfinite_values_even_in_zero_products(
    location: str, value: complex
) -> None:
    source = of.InteractionOperator(
        0, np.zeros((2, 2), dtype=complex), np.zeros((2,) * 4, dtype=complex)
    )
    if location == "constant":
        source.constant = value
    elif location == "one":
        source.one_body_tensor[0, 0] = value
    else:
        source.two_body_tensor[0, 0, 0, 0] = value
    with pytest.raises(ValueError):
        from_openfermion_interaction(source)


@pytest.mark.parametrize(
    "one,two",
    [
        (np.zeros((2, 3)), np.zeros((2,) * 4)),
        (np.zeros(2), np.zeros((2,) * 4)),
        (np.zeros((2, 2)), np.zeros((3,) * 4)),
        (np.zeros((2, 2), dtype=bool), np.zeros((2,) * 4)),
        (np.zeros((2, 2)), np.zeros((2,) * 4, dtype=object)),
        ([[0, 0], [0, 0]], np.zeros((2,) * 4)),
    ],
)
def test_interaction_corrupted_arrays_fail(one: object, two: object) -> None:
    source = of.InteractionOperator(0, np.zeros((2, 2)), np.zeros((2,) * 4))
    source.n_body_tensors[(1, 0)] = one
    source.n_body_tensors[(1, 1, 0, 0)] = two
    with pytest.raises((TypeError, ValueError)):
        from_openfermion_interaction(source)


@pytest.mark.parametrize(
    "constant", [True, sympy.Symbol("x"), torch.tensor(1.0), np.array(1.0)]
)
def test_interaction_invalid_constant(constant: object) -> None:
    source = of.InteractionOperator(constant, np.zeros((2, 2)), np.zeros((2,) * 4))
    with pytest.raises(TypeError):
        from_openfermion_interaction(source)


def test_interaction_rejects_wrong_type_and_corrupted_keys() -> None:
    with pytest.raises(TypeError):
        from_openfermion_interaction(of.FermionOperator())
    source = of.InteractionOperator(0, np.zeros((2, 2)), np.zeros((2,) * 4))
    source.n_body_tensors[(1,)] = np.zeros(2)
    with pytest.raises(ValueError):
        from_openfermion_interaction(source)
    source.n_body_tensors = []
    with pytest.raises(TypeError):
        from_openfermion_interaction(source)


def _uccsd_training_trace(
    operator: FermionOperator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    observable = to_observable(
        encode_fermion(operator, n_modes=4, encoding="jordan_wigner").operator
    )

    def energy(parameters: torch.Tensor) -> torch.Tensor:
        circuit = uccsd_ansatz(4, 2, parameters, dtype=torch.complex128, device="cpu")
        value = fq.run(circuit, outputs=[fq.expectation(observable)]).expectation()
        assert isinstance(value, torch.Tensor)
        return value.reshape(())

    parameters = torch.tensor(
        [0.1, -0.07, 0.2], dtype=torch.float32, requires_grad=True
    )
    initial = parameters.detach().clone()
    first = energy(parameters)
    gradient = torch.autograd.grad(first, parameters)[0]
    assert torch.all(torch.isfinite(gradient)) and float(gradient.norm()) > 0.1
    step = 1e-3
    for index in range(3):
        shift = torch.zeros_like(initial)
        shift[index] = step
        finite_difference = (energy(initial + shift) - energy(initial - shift)) / (
            2 * step
        )
        torch.testing.assert_close(
            gradient[index], finite_difference.to(gradient.dtype), atol=2e-4, rtol=2e-4
        )

    optimizer = torch.optim.SGD([parameters], lr=0.1)
    energies, gradients = [], []
    for _ in range(16):
        optimizer.zero_grad()
        value = energy(parameters)
        energies.append(value.detach())
        torch.autograd.backward(value)
        assert parameters.grad is not None and torch.all(
            torch.isfinite(parameters.grad)
        )
        gradients.append(parameters.grad.detach().clone())
        optimizer.step()
    final = energy(parameters).detach()
    energies.append(final)
    assert float(final) < float(first.detach()) - 0.1
    assert float((parameters.detach() - initial).norm()) > 0.1
    state = (
        uccsd_ansatz(4, 2, parameters.detach(), dtype=torch.complex128, device="cpu")
        .state()
        .reshape(-1)
    )
    outside = [index for index in range(16) if index.bit_count() != 2]
    assert float(state[outside].abs().square().sum()) < 1e-20
    torch.testing.assert_close(
        state.abs().square().sum(), torch.tensor(1.0, dtype=torch.float64)
    )
    reference = torch.vdot(state, fock_matrix(operator, 4) @ state)
    assert float(reference.imag.abs()) < 1e-12
    torch.testing.assert_close(final, reference.real)
    return torch.stack(energies), torch.stack(gradients), parameters.detach()


def test_interaction_replacement_preserves_native_uccsd_training() -> None:
    # Independently write the spin-interleaved two-site Hubbard coefficients.
    one = -0.2 * np.eye(4)
    for spin in (0, 1):
        one[spin, spin + 2] = one[spin + 2, spin] = -1.0
    two = np.zeros((4,) * 4)
    two[0, 1, 1, 0] = two[2, 3, 3, 2] = 2.0
    source = of.InteractionOperator(0.375, one, two)
    imported = from_openfermion_interaction(source)
    native = fermi_hubbard_chain(2, t=1.0, u=2.0, mu=0.2) + FermionOperator(
        (FermionTerm(0.375),)
    )
    assert imported == native
    # Destroy the external input before encoding, differentiation or optimization.
    one[:] = float("nan")
    two[:] = float("nan")
    source.n_body_tensors = {}
    expected = _uccsd_training_trace(native)
    actual = _uccsd_training_trace(imported)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, atol=1e-7, rtol=1e-7)

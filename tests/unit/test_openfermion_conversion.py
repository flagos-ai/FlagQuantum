"""Real optional upstream inputs checked against independent matrix action."""

from __future__ import annotations

import itertools
import pathlib
import typing
from collections.abc import Callable

import pytest
import sympy
import torch

import flagquantum as fq
from flagquantum.algorithms.fermion.encoding import Encoding, encode_fermion
from flagquantum.algorithms.fermion.numerics import pauli_matrix
from flagquantum.algorithms.fermion.observables import to_observable
from flagquantum.core.algebra.pauli import PauliOperator, PauliTerm
from flagquantum.ecosystem.openfermion import (
    from_openfermion_fermion,
    from_openfermion_qubit,
    to_openfermion_fermion,
    to_openfermion_qubit,
)
from flagquantum.observables.fermion import FermionOperator, FermionTerm
from tests.unit.fermion.oracles import fock_matrix

if typing.TYPE_CHECKING:
    from openfermion import MolecularData as OpenMolecularData

of = pytest.importorskip("openfermion")
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


def test_molecular_data_replacement_preserves_native_uccsd_training(
    molecular_data_source: OpenMolecularData,
) -> None:
    from flagquantum.algorithms.chemistry import uccsd_ansatz
    from flagquantum.algorithms.chemistry_domain.problem import (
        fermion_hamiltonian,
        from_native_molecule,
    )
    from flagquantum.algorithms.molecular import create_molecular_hamiltonian
    from flagquantum.ecosystem.chemistry.molecular_data import (
        from_openfermion_molecular_data,
    )
    from flagquantum.observables import Observable

    source = molecular_data_source
    imported = from_openfermion_molecular_data(source)
    native = from_native_molecule(
        create_molecular_hamiltonian(
            [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))]
        )
    )
    reference = torch.tensor(
        of.get_sparse_operator(source.get_molecular_hamiltonian()).toarray(),
        dtype=torch.complex128,
    )
    # Conversion owns the snapshot before any circuit or optimizer is built.
    source.one_body_integrals[:] = float("nan")
    source.two_body_integrals[:] = float("nan")
    source.nuclear_repulsion = float("nan")
    observables = [
        to_observable(
            encode_fermion(
                fermion_hamiltonian(problem), n_modes=4, encoding="jordan_wigner"
            ).operator
        )
        for problem in (native, imported)
    ]

    def energy(angles: torch.Tensor, observable: Observable) -> torch.Tensor:
        circuit = uccsd_ansatz(4, 2, angles, dtype=torch.complex128, device="cpu")
        value = fq.run(circuit, outputs=[fq.expectation(observable)]).expectation()
        assert isinstance(value, torch.Tensor)
        return value.reshape(())

    initial = torch.tensor([0.1, -0.07, 0.2], dtype=torch.float32)
    parameters = [initial.clone().requires_grad_() for _ in observables]
    optimizers = [torch.optim.SGD([angles], lr=0.2) for angles in parameters]
    first = energy(parameters[0], observables[0]).detach()
    gradient = torch.autograd.grad(
        energy(parameters[1], observables[1]), parameters[1]
    )[0]
    assert float(gradient.norm()) > 0.01
    # UCCSD normalizes angles to float32; resolve differences above that scale.
    epsilon = 1e-3
    for index in range(3):
        shift = torch.zeros_like(initial)
        shift[index] = epsilon
        difference = (
            energy(initial + shift, observables[1])
            - energy(initial - shift, observables[1])
        ) / (2 * epsilon)
        torch.testing.assert_close(
            gradient[index], difference.to(gradient.dtype), atol=2e-4, rtol=2e-4
        )

    for _ in range(24):
        values, gradients = [], []
        for angles, observable, optimizer in zip(
            parameters, observables, optimizers, strict=True
        ):
            optimizer.zero_grad()
            value = energy(angles, observable)
            torch.autograd.backward(value)
            assert angles.grad is not None and torch.all(torch.isfinite(angles.grad))
            values.append(value.detach())
            gradients.append(angles.grad.detach().clone())
            optimizer.step()
        torch.testing.assert_close(values[0], values[1], atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(gradients[0], gradients[1], atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(parameters[0], parameters[1], atol=1e-7, rtol=1e-7)

    final = energy(parameters[1], observables[1]).detach()
    assert float(final) < float(first) - 0.001
    assert float((parameters[1].detach() - initial).norm()) > 0.01
    state = (
        uccsd_ansatz(4, 2, parameters[1].detach(), dtype=torch.complex128, device="cpu")
        .state()
        .reshape(-1)
    )
    outside = [index for index in range(16) if index.bit_count() != 2]
    assert float(state[outside].abs().square().sum()) < 1e-20
    torch.testing.assert_close(
        state.abs().square().sum(), torch.tensor(1.0, dtype=torch.float64)
    )
    exact = torch.vdot(state, reference @ state)
    assert float(exact.imag.abs()) < 1e-12
    torch.testing.assert_close(final, exact.real, atol=1e-10, rtol=1e-10)
    sector = [index for index in range(16) if index.bit_count() == 2]
    assert (
        float(final)
        >= float(torch.linalg.eigvalsh(reference[sector][:, sector])[0]) - 1e-10
    )


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

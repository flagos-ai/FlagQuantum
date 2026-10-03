"""The chemistry ansatz builders, checked against a Fock-space oracle.

Every numeric assertion here compares a circuit's unitary against
``exp(i * theta / 2 * H)`` for ``H = jordan_wigner(1j * T)``, where ``T`` is the
excitation generator rebuilt in this file.  The oracle therefore runs through the
Jordan-Wigner image of :mod:`flagquantum.observables.fermion` -- which its own
suite checks against a Fock-space matrix -- and through a dense Pauli product,
and shares no code with the gate sequence under test.  An error in a basis
change, in a conditional ladder, in the sign an exchanged index costs, or in the
eighth-of-an-angle that the double excitation uses shows up as a disagreement
between two matrices rather than as a disagreement the circuit can absorb.

The generator is rebuilt here instead of being taken from the module for the same
reason: ``T = a_v^dagger a_o - a_o^dagger a_v`` written out with the ladder
operators is the definition, and importing it would make the comparison circular.

The two deliberate divergences from CUDA-Q recorded in the module docstring have
their own tests: the occupied-below-virtual requirement, and the placement of the
reference determinant for an odd electron count.
"""

import itertools

import numpy as np
import pytest
import torch

import flagquantum as fq
from flagquantum.algorithms import chemistry
from flagquantum.algorithms.chemistry import (
    UCCSDExcitations,
    coupler_hardware_efficient_ansatz,
    coupler_hardware_efficient_parameter_count,
    double_excitation,
    excitation_operator,
    single_excitation,
    uccsd_ansatz,
    uccsd_excitations,
    uccsd_factors,
)
from flagquantum.observables.fermion import annihilate, create, jordan_wigner
from flagquantum.simulation.pauli import pauli_product_operator
from flagquantum.simulation.unitary import get_unitary

pytestmark = pytest.mark.unit

_ATOL = 1e-12
_WRONG_CONVENTION = 1e-2


def _rounded(coefficient: complex) -> complex:
    """Return a coefficient with its real and imaginary parts rounded.

    The two generators are built by different fold orders, so their terms are
    compared on rounded coefficients rather than on the arithmetic that produced
    them. Python's ``round`` refuses a ``complex``, so the two parts are rounded
    separately and the check needs no third-party array library.
    """

    return complex(round(coefficient.real, 12), round(coefficient.imag, 12))


def _angles(parameters: object) -> torch.Tensor:
    """Return parameters as the float32 vector the module reads.

    The module normalizes every angle to float32 before it reaches a gate, so a
    reference built from the original Python floats would differ from the circuit
    by the float32 rounding of each angle -- about 1e-8 per rotation. Both sides
    of a comparison read this vector instead.
    """

    return torch.as_tensor(parameters, dtype=torch.float32).reshape(-1)


def _generator(occupied: tuple[int, ...], virtual: tuple[int, ...]) -> object:
    """Rebuild ``a_v^dagger a_o - a_o^dagger a_v`` from the ladder operators."""

    return _product(virtual, creating=True) * _product(occupied, creating=False) - (
        _product(occupied, creating=True) * _product(virtual, creating=False)
    )


def _product(degrees: tuple[int, ...], *, creating: bool) -> object:
    """Return one ladder-operator product, or the identity when it is empty."""

    from flagquantum.observables.fermion import FermionOperator, FermionTerm

    if not degrees:
        return FermionOperator((FermionTerm(1.0, (), ()),))
    operator = None
    for degree in degrees:
        factor = create(degree) if creating else annihilate(degree)
        operator = factor if operator is None else operator * factor
    return operator


def _dense(observable: object, n_modes: int) -> torch.Tensor:
    """Build the dense matrix of a Pauli observable, wire 0 leftmost."""

    matrix = torch.zeros(2**n_modes, 2**n_modes, dtype=torch.complex128)
    for term in observable.terms:
        matrix = matrix + term.coefficient * pauli_product_operator(
            term.factors, n_modes, dtype=torch.complex128, device="cpu"
        )
    return matrix


def _exponential(
    n_modes: int,
    occupied: tuple[int, ...],
    virtual: tuple[int, ...],
    theta: float,
) -> torch.Tensor:
    """Return ``exp(i * theta / 2 * H)`` for the excitation's ``H``."""

    generator = _generator(occupied, virtual)
    hermitian = _dense(jordan_wigner(1j * generator, n_modes=n_modes), n_modes)
    return torch.matrix_exp(0.5j * theta * hermitian)


def _circuit_unitary(circuit: fq.Circuit) -> torch.Tensor:
    """Return the exact unitary of a circuit as complex128."""

    return get_unitary(
        circuit.to_device("cpu") if circuit.device != "cpu" else circuit,
    ).to(torch.complex128)


_SINGLE_CASES = (
    (2, 0, 1, 0.7),
    (3, 0, 2, 0.7),
    (4, 0, 3, -1.1),
    (4, 1, 2, 2.5),
    (6, 2, 5, 0.3),
)


def test_the_single_excitation_is_the_exponential_of_its_generator() -> None:
    for n_qubits, occupied, virtual, theta in _SINGLE_CASES:
        circuit = single_excitation(
            n_qubits, occupied, virtual, theta, dtype=torch.complex128
        )
        target = _exponential(n_qubits, (occupied,), (virtual,), theta)
        deviation = float((_circuit_unitary(circuit) - target).abs().max())
        assert deviation < _ATOL, (n_qubits, occupied, virtual, theta, deviation)


def test_the_single_excitation_realizes_the_positive_exponential() -> None:
    n_qubits, occupied, virtual, theta = 3, 0, 2, 0.7
    circuit = single_excitation(
        n_qubits, occupied, virtual, theta, dtype=torch.complex128
    )
    generator = _generator((occupied,), (virtual,))
    hermitian = _dense(jordan_wigner(1j * generator, n_modes=n_qubits), n_qubits)
    opposite = torch.matrix_exp(-0.5j * theta * hermitian)
    deviation = float((_circuit_unitary(circuit) - opposite).abs().max())
    assert deviation > _WRONG_CONVENTION


_DOUBLE_ORDERINGS = (
    (4, (0, 1), (2, 3), 0.7),
    (4, (1, 0), (2, 3), 0.7),
    (4, (0, 1), (3, 2), 0.7),
    (4, (1, 0), (3, 2), 0.7),
    (6, (0, 1), (4, 5), -1.1),
    (4, (0, 2), (1, 3), 0.3),
)

def test_the_double_excitation_is_the_exponential_of_its_generator() -> None:
    for n_qubits, occupied, virtual, theta in _DOUBLE_ORDERINGS:
        circuit = double_excitation(
            n_qubits, occupied, virtual, theta, dtype=torch.complex128
        )
        target = _exponential(n_qubits, occupied, virtual, theta)
        deviation = float((_circuit_unitary(circuit) - target).abs().max())
        assert deviation < _ATOL, (n_qubits, occupied, virtual, theta, deviation)


def test_the_double_excitation_realizes_the_positive_exponential() -> None:
    n_qubits, occupied, virtual, theta = 4, (0, 1), (2, 3), 0.7
    circuit = double_excitation(
        n_qubits, occupied, virtual, theta, dtype=torch.complex128
    )
    generator = _generator(occupied, virtual)
    hermitian = _dense(jordan_wigner(1j * generator, n_modes=n_qubits), n_qubits)
    opposite = torch.matrix_exp(-0.5j * theta * hermitian)
    deviation = float((_circuit_unitary(circuit) - opposite).abs().max())
    assert deviation > _WRONG_CONVENTION


def test_an_exchanged_index_flips_the_operator_and_the_circuit_together() -> None:
    """The generator and the circuit must take the same sign from the same order.

    ``excitation_operator`` and ``double_excitation`` read the two collections of
    indices independently.  If only one of them noticed the exchange order, the
    circuit would evolve a generator whose sign disagrees with the operator the
    caller screens gradients against, and every energy built from the pair would
    be wrong while each half looked self-consistent.
    """

    theta = 0.7
    for occupied, virtual in itertools.product(
        ((0, 1), (1, 0)), ((2, 3), (3, 2))
    ):
        operator = excitation_operator(occupied, virtual)
        reference = excitation_operator((0, 1), (2, 3))
        assert operator.terms == reference.terms or operator.terms == tuple(
            type(term)(-term.coefficient, term.creations, term.annihilations)
            for term in reference.terms
        )
        circuit = double_excitation(
            4, occupied, virtual, theta, dtype=torch.complex128
        )
        target = _exponential(4, occupied, virtual, theta)
        deviation = float((_circuit_unitary(circuit) - target).abs().max())
        assert deviation < _ATOL, (occupied, virtual, deviation)


def test_the_two_exchange_orders_give_opposite_circuits() -> None:
    forward = double_excitation(4, (0, 1), (2, 3), 0.7, dtype=torch.complex128)
    backward = double_excitation(4, (0, 1), (3, 2), 0.7, dtype=torch.complex128)
    deviation = float((_circuit_unitary(forward) - _circuit_unitary(backward)).abs().max())
    assert deviation > _WRONG_CONVENTION


def test_the_excitation_operator_is_anti_hermitian() -> None:
    for occupied, virtual in (
        ((0,), (1,)),
        ((2,), (5,)),
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
    ):
        generator = excitation_operator(occupied, virtual)
        assert generator.dagger().terms == tuple(
            type(term)(-term.coefficient, term.creations, term.annihilations)
            for term in generator.terms
        )
        assert generator.is_hermitian() is False


def _independent_census(n_electrons: int, n_qubits: int) -> tuple[int, int, int]:
    """Return (singles, same-spin doubles, mixed-spin doubles) from closed forms."""

    occupied = n_electrons // 2
    virtual = (n_qubits - n_electrons) // 2
    singles = 2 * occupied * virtual
    same_spin = 2 * (occupied * (occupied - 1) // 2) * (virtual * (virtual - 1) // 2)
    mixed = occupied * occupied * virtual * virtual
    return singles, same_spin, mixed


def test_the_excitation_census_matches_the_closed_forms() -> None:
    for n_electrons, n_qubits in ((2, 4), (4, 8), (2, 6), (6, 8), (4, 6), (0, 4)):
        excitations = uccsd_excitations(n_electrons, n_qubits)
        singles, same_spin, mixed = _independent_census(n_electrons, n_qubits)
        assert len(excitations.singles_alpha) + len(excitations.singles_beta) == singles
        assert (
            len(excitations.doubles_alpha) + len(excitations.doubles_beta) == same_spin
        )
        assert len(excitations.doubles_mixed) == mixed
        assert excitations.parameter_count == singles + same_spin + mixed


def test_the_excitation_lists_are_in_the_order_uccsd_applies_them() -> None:
    excitations = uccsd_excitations(4, 8)
    assert excitations.factors == (
        *excitations.singles_alpha,
        *excitations.singles_beta,
        *excitations.doubles_mixed,
        *excitations.doubles_alpha,
        *excitations.doubles_beta,
    )
    assert all(len(factor) == 2 for factor in excitations.singles_alpha)
    assert all(len(factor) == 4 for factor in excitations.doubles_alpha)
    assert all(len(factor) == 4 for factor in excitations.doubles_mixed)


def test_every_enumerated_factor_is_a_generator_the_builders_accept() -> None:
    for n_electrons, n_qubits in ((2, 4), (4, 8), (2, 6)):
        for factor in uccsd_excitations(n_electrons, n_qubits).factors:
            occupied, virtual = factor[: len(factor) // 2], factor[len(factor) // 2 :]
            generator = excitation_operator(occupied, virtual)
            assert generator.terms
            assert generator.is_hermitian() is False


def test_the_excitation_spin_orbitals_are_the_lowest_occupied_and_the_rest_virtual() -> (
    None
):
    excitations = uccsd_excitations(2, 6)
    assert excitations.reference_occupation == (0, 1)
    assert excitations.occupied_alpha == (0,)
    assert excitations.occupied_beta == (1,)
    assert excitations.virtual_alpha == (2, 4)
    assert excitations.virtual_beta == (3, 5)


def test_the_reference_determinant_is_the_lowest_occupied_spin_orbitals() -> None:
    for n_electrons, n_qubits in ((2, 4), (4, 6), (2, 6)):
        excitations = uccsd_excitations(n_electrons, n_qubits)
        circuit = uccsd_ansatz(
            n_qubits,
            n_electrons,
            [0.0] * excitations.parameter_count,
            dtype=torch.complex128,
        )
        probabilities = circuit.probabilities().reshape(-1)
        expected = 0
        for spin_orbital in excitations.reference_occupation:
            expected |= 1 << (n_qubits - 1 - spin_orbital)
        assert float(probabilities[expected]) > 1.0 - 1e-12
        assert float(probabilities.sum()) == pytest.approx(1.0, abs=1e-12)


def test_the_uccsd_product_is_the_excitation_gates_in_the_declared_order() -> None:
    n_electrons, n_qubits = 2, 4
    excitations = uccsd_excitations(n_electrons, n_qubits)
    values = _angles([0.4, 0.4, 0.4])
    product = uccsd_factors(
        n_electrons, n_qubits, values, dtype=torch.complex128
    )

    expected = torch.eye(2**n_qubits, dtype=torch.complex128)
    for factor, angle in zip(excitations.factors, values, strict=True):
        if len(factor) == 2:
            step = single_excitation(
                n_qubits, factor[0], factor[1], angle, dtype=torch.complex128
            )
        else:
            step = double_excitation(
                n_qubits, factor[:2], factor[2:], angle, dtype=torch.complex128
            )
        expected = _circuit_unitary(step) @ expected

    deviation = float((_circuit_unitary(product) - expected).abs().max())
    assert deviation < _ATOL


def test_the_uccsd_product_applies_its_factors_in_order() -> None:
    """The parameter vector is read in factor order, not reversed or sorted."""

    n_electrons, n_qubits = 2, 4
    excitations = uccsd_excitations(n_electrons, n_qubits)
    values = _angles([0.4, -0.3, 0.9])
    forward = _circuit_unitary(
        uccsd_factors(n_electrons, n_qubits, values, dtype=torch.complex128)
    )

    expected = torch.eye(2**n_qubits, dtype=torch.complex128)
    for factor, angle in zip(excitations.factors, values, strict=True):
        if len(factor) == 2:
            step = single_excitation(
                n_qubits, factor[0], factor[1], angle, dtype=torch.complex128
            )
        else:
            step = double_excitation(
                n_qubits, factor[:2], factor[2:], angle, dtype=torch.complex128
            )
        expected = _circuit_unitary(step) @ expected
    assert float((forward - expected).abs().max()) < _ATOL

    backward = _circuit_unitary(
        uccsd_factors(
            n_electrons, n_qubits, _angles([0.9, -0.3, 0.4]), dtype=torch.complex128
        )
    )
    assert float((forward - backward).abs().max()) > _WRONG_CONVENTION


def test_the_uccsd_ansatz_adds_the_reference_occupation_to_the_product() -> None:
    n_electrons, n_qubits = 2, 4
    excitations = uccsd_excitations(n_electrons, n_qubits)
    parameters = [0.0] * excitations.parameter_count
    ansatz = uccsd_ansatz(
        n_qubits, n_electrons, parameters, dtype=torch.complex128
    )
    assert ansatz.analysis().n_instructions == (
        fq.Circuit(n_qubits).x(0).x(1).analysis().n_instructions
        + uccsd_factors(
            n_electrons, n_qubits, parameters, dtype=torch.complex128
        )
        .analysis()
        .n_instructions
    )


def test_the_coupler_parameter_count_is_the_declared_formula() -> None:
    for n_qubits, layers in ((1, 0), (2, 1), (4, 2), (6, 3)):
        assert coupler_hardware_efficient_parameter_count(
            n_qubits, layers
        ) == 2 * n_qubits * (1 + layers)


def test_the_coupler_ansatz_is_the_declared_rotation_and_coupler_sequence() -> None:
    n_qubits, layers = 3, 2
    values = _angles([0.1 * index for index in range(2 * n_qubits * (1 + layers))])
    circuit = coupler_hardware_efficient_ansatz(
        n_qubits, layers, values, dtype=torch.complex128
    )
    analysis = circuit.analysis()
    assert analysis.two_qubit_gates == layers * (n_qubits - 1)
    assert analysis.n_instructions == 2 * n_qubits * (1 + layers) + layers * (
        n_qubits - 1
    )

    expected = fq.Circuit(n_qubits, dtype=torch.complex128)
    cursor = 0
    for qubit in range(n_qubits):
        expected.ry(qubit, theta=values[cursor])
        expected.rz(qubit, theta=values[cursor + 1])
        cursor += 2
    for _ in range(layers):
        for qubit in range(n_qubits - 1):
            expected.cx(qubit, qubit + 1)
        for qubit in range(n_qubits):
            expected.ry(qubit, theta=values[cursor])
            expected.rz(qubit, theta=values[cursor + 1])
            cursor += 2

    deviation = float(
        (_circuit_unitary(circuit) - _circuit_unitary(expected)).abs().max()
    )
    assert deviation < _ATOL


def test_the_coupler_ansatz_honours_an_explicit_coupler_list() -> None:
    circuit = coupler_hardware_efficient_ansatz(
        4,
        1,
        [0.0] * 16,
        couplers=[(3, 0), (1, 2)],
        dtype=torch.complex128,
    )
    analysis = circuit.analysis()
    assert analysis.two_qubit_gates == 2
    assert analysis.max_gate_width == 2


def test_the_coupler_ansatz_defaults_to_the_nearest_neighbour_chain() -> None:
    chain = coupler_hardware_efficient_ansatz(
        4, 1, [0.0] * 16, dtype=torch.complex128
    )
    explicit = coupler_hardware_efficient_ansatz(
        4,
        1,
        [0.0] * 16,
        couplers=[(0, 1), (1, 2), (2, 3)],
        dtype=torch.complex128,
    )
    assert _circuit_unitary(chain).equal(_circuit_unitary(explicit))


def test_the_two_hardware_efficient_ansatze_are_different_circuits() -> None:
    """The coupler ansatz and ``algorithms.core``'s ansatz are not interchangeable."""

    from flagquantum.algorithms.core import (
        hardware_efficient_ansatz,
        hardware_efficient_parameter_count,
    )

    n_qubits, layers = 3, 2
    coupler_count = coupler_hardware_efficient_parameter_count(n_qubits, layers)
    core_count = hardware_efficient_parameter_count(n_qubits, layers)
    assert coupler_count == core_count + 2 * n_qubits

    coupler = _circuit_unitary(
        coupler_hardware_efficient_ansatz(
            n_qubits, layers, [0.5] * coupler_count, dtype=torch.complex128
        )
    )
    core = _circuit_unitary(
        hardware_efficient_ansatz(n_qubits, layers, [0.5] * core_count)
    )
    assert float((coupler - core).abs().max()) > _WRONG_CONVENTION


def test_the_chemistry_module_is_reachable_and_stays_out_of_the_root_namespace() -> None:
    assert chemistry.uccsd_ansatz is uccsd_ansatz
    assert chemistry.UCCSDExcitations is UCCSDExcitations
    for name in chemistry.__all__:
        assert hasattr(chemistry, name)
        assert not hasattr(fq, name), name

def test_an_excitation_refuses_an_occupied_index_that_is_not_below_its_virtuals() -> (
    None
):
    with pytest.raises(ValueError, match="below the virtual ones"):
        single_excitation(4, 2, 1, 0.5)
    with pytest.raises(ValueError, match="below the virtual ones"):
        double_excitation(6, (0, 4), (2, 3), 0.5)
    with pytest.raises(ValueError, match="below the virtual ones"):
        excitation_operator((3,), (1,))


@pytest.mark.parametrize(
    ("occupied", "virtual", "message"),
    (
        ((0,), (0,), "below the virtual ones"),
        ((0, 1), (1, 2), "below the virtual ones"),
        ((0,), (1, 2), "moves as many electrons as it fills"),
        ((0, 1, 2), (3, 4, 5), "one or two electrons"),
        ((), (1,), "moves as many electrons as it fills"),
        ((0,), (), "moves as many electrons as it fills"),
    ),
)
def test_an_excitation_operator_refuses_a_malformed_excitation(
    occupied: tuple[int, ...], virtual: tuple[int, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        excitation_operator(occupied, virtual)


@pytest.mark.parametrize(
    ("occupied", "virtual", "message"),
    (
        ((0, 1), (2, 2), "must be distinct"),
        ((0, 0), (2, 3), "must be distinct"),
        ((0,), (2, 3), "moves as many electrons as it fills"),
    ),
)
def test_a_circuit_refuses_a_malformed_excitation(
    occupied: tuple[int, ...], virtual: tuple[int, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        double_excitation(4, occupied, virtual, 0.5)


@pytest.mark.parametrize(
    ("n_electrons", "n_qubits", "message"),
    (
        (2, 5, "must be even"),
        (6, 4, "must be at most"),
        (True, 4, "must be an integer"),
        (-1, 4, "must be at least 0"),
        (2.0, 4, "must be an integer"),
    ),
)
def test_the_excitation_census_refuses_a_malformed_electron_count(
    n_electrons: object, n_qubits: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        uccsd_excitations(n_electrons, n_qubits)


@pytest.mark.parametrize(
    ("theta", "message"),
    (
        (float("inf"), "must be finite"),
        (float("nan"), "must be finite"),
        ("0.5", "must be a real number"),
        (True, "must be a real number"),
        (None, "must be a real number"),
    ),
)
def test_an_excitation_refuses_an_unusable_angle(theta: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        single_excitation(2, 0, 1, theta)


def test_a_circuit_refuses_an_index_outside_its_declared_width() -> None:
    with pytest.raises(ValueError, match="outside the 2 spin orbital"):
        single_excitation(2, 0, 2, 0.5)
    with pytest.raises(ValueError, match="must be at least 1"):
        single_excitation(0, 0, 1, 0.5)
    with pytest.raises(ValueError, match="must be at least 1"):
        double_excitation(0, (0,), (1,), 0.5)
    with pytest.raises(ValueError, match="outside the 4 spin orbital"):
        double_excitation(4, (0, 1), (2, 4), 0.5)


@pytest.mark.parametrize("parameters", ([], [0.1, 0.2], [0.1, 0.2, 0.3, 0.4]))
def test_the_uccsd_product_refuses_the_wrong_number_of_parameters(
    parameters: list[float],
) -> None:
    with pytest.raises(ValueError, match=f"Expected 3 parameters, got {len(parameters)}"):
        uccsd_factors(2, 4, parameters)


def test_the_uccsd_product_reads_a_tensor_parameter_vector() -> None:
    values = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64, requires_grad=True)
    circuit = uccsd_factors(2, 4, values, dtype=torch.complex128)
    assert circuit.is_parameterized() is False
    loss = circuit.probabilities().reshape(-1)[0b1100]
    loss.backward()
    assert values.grad is not None
    assert float(values.grad.abs().sum()) > 0.0


def test_every_angle_reaches_a_gate_at_the_runtime_real_dtype() -> None:
    """A caller's Python float is read once, at float32, and stays there.

    The module normalizes the whole parameter vector before any angle reaches a
    gate, so the angle the IR carries is the float32 rounding of the caller's
    number rather than the float64 number itself. The two differ in the eighth
    decimal, which is the difference between a circuit whose numbers match its
    own declared dtype and one that silently keeps a wider angle.
    """

    values = _angles([0.1, 0.2, 0.3])
    circuit = uccsd_factors(2, 4, values, dtype=torch.complex64)
    carried = [
        instruction.params["theta"]
        for instruction in circuit.to_ir()
        if instruction.name == "rz"
    ]
    assert carried, "the product must rotate at least one conditional phase"
    for value in carried:
        assert torch.as_tensor(value).dtype == torch.float32
    # The first factor is an alpha single, whose conditional phase is half its
    # angle: the float32 half of the float32 angle, not the float64 number.
    assert float(carried[0]) == float(values[0]) / 2.0
    assert float(carried[0]) != 0.05


def test_a_scalar_tensor_angle_stays_a_tensor_in_the_ir() -> None:
    """A tensor angle keeps its graph: it is carried, not read out to a float.

    ``single_excitation`` is the entry point a differentiating caller uses, so an
    angle that arrives as a tensor must reach the IR as the same tensor, with the
    graph the caller built. Collapsing it to a Python float would build the same
    circuit today and would silently drop the derivative tomorrow.
    """

    angle = torch.tensor(0.7, requires_grad=True)
    circuit = single_excitation(2, 0, 1, angle)
    carried = [
        instruction.params["theta"]
        for instruction in circuit.to_ir()
        if instruction.name == "rz"
    ]
    assert carried
    for value in carried:
        assert isinstance(value, torch.Tensor)
        assert value.requires_grad
        assert value.grad_fn is not None


@pytest.mark.parametrize(
    ("occupied", "virtual", "message"),
    (
        (False, 1, "occupied index must be an integer"),
        (0.0, 1, "occupied index must be an integer"),
        (True, 1, "occupied index must be an integer"),
        (-1, 1, "occupied index must be non-negative"),
        (5, 1, "occupied index 5 is outside the 2 spin orbital"),
    ),
)
def test_a_circuit_refuses_a_malformed_occupied_index(
    occupied: object, virtual: object, message: str
) -> None:
    """``False`` and ``0.0`` are both usable as wire 0 and must both be refused.

    The two are the reason the refusal cannot be delegated to the circuit's own
    range check: ``operator.index`` accepts a bool and rejects a float, and
    coercing either one turns a caller's mistake into a silently different
    excitation.
    """

    with pytest.raises(ValueError, match=message):
        single_excitation(2, occupied, virtual, 0.5)


@pytest.mark.parametrize(
    ("n_qubits", "layers", "message"),
    ((-1, 0, "must be at least 0"), (2, -1, "must be at least 0"), (2.0, 1, "integer")),
)
def test_the_coupler_parameter_count_refuses_a_malformed_count(
    n_qubits: object, layers: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        coupler_hardware_efficient_parameter_count(n_qubits, layers)


@pytest.mark.parametrize(
    ("couplers", "message"),
    (
        ([(0, 0)], "must join two qubits"),
        ([(0, 5)], "outside the 4 spin orbital"),
        ([(0, 1, 2)], "must be a pair of qubits"),
        ([(0, 1), (0, 1)], "must be distinct"),
        ([0], "must be a pair of qubits"),
        (["ab"], "must be a pair of qubits"),
    ),
)
def test_the_coupler_ansatz_refuses_a_malformed_coupler(
    couplers: list[object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        coupler_hardware_efficient_ansatz(4, 1, [0.0] * 16, couplers=couplers)


def test_the_coupler_ansatz_refuses_the_wrong_number_of_parameters() -> None:
    with pytest.raises(ValueError, match="Expected 8 parameters, got 7"):
        coupler_hardware_efficient_ansatz(2, 1, [0.0] * 7)
    with pytest.raises(ValueError, match="Expected 8 parameters, got 4"):
        coupler_hardware_efficient_ansatz(2, 1, [0.0] * 4, couplers=[])


def test_the_double_excitation_uses_eight_conditional_blocks() -> None:
    analysis = double_excitation(
        4, (0, 1), (2, 3), 0.7, dtype=torch.complex128
    ).analysis()
    assert analysis.two_qubit_gates == 40
    assert analysis.max_gate_width == 2
    assert analysis.n_instructions == 84


def test_the_single_excitation_uses_two_conditional_ladders() -> None:
    analysis = single_excitation(
        4, 0, 3, 0.7, dtype=torch.complex128
    ).analysis()
    assert analysis.two_qubit_gates == 12
    assert analysis.max_gate_width == 2


def test_a_zero_angle_excitation_is_the_identity() -> None:
    for circuit in (
        single_excitation(3, 0, 2, 0.0, dtype=torch.complex128),
        double_excitation(4, (0, 1), (2, 3), 0.0, dtype=torch.complex128),
    ):
        unitary = _circuit_unitary(circuit)
        identity = torch.eye(2 ** circuit.n_qubits, dtype=torch.complex128)
        assert float((unitary - identity).abs().max()) < _ATOL


def test_every_excitation_circuit_is_unitary() -> None:
    for circuit in (
        single_excitation(3, 0, 2, 0.7, dtype=torch.complex128),
        double_excitation(4, (0, 1), (2, 3), 0.7, dtype=torch.complex128),
        uccsd_factors(2, 4, [0.1, 0.2, 0.3], dtype=torch.complex128),
        coupler_hardware_efficient_ansatz(3, 1, [0.1] * 12, dtype=torch.complex128),
    ):
        unitary = _circuit_unitary(circuit)
        dimension = unitary.shape[0]
        identity = torch.eye(dimension, dtype=torch.complex128)
        deviation = float((unitary.conj().T @ unitary - identity).abs().max())
        assert deviation < _ATOL


def test_the_excitations_commute_with_the_particle_number_of_the_reference() -> None:
    """An excitation conserves the electron count, so it never leaves the sector.

    The reference determinant for two electrons in four spin orbitals has weight
    only on the two-excitation sector.  Applying one excitation circuit to it must
    keep all of the probability inside the spin orbitals' electron count of two,
    which a circuit with a wrong ladder length would not.
    """

    n_electrons, n_qubits = 2, 4
    for factor in uccsd_excitations(n_electrons, n_qubits).factors:
        if len(factor) == 2:
            circuit = single_excitation(
                n_qubits, factor[0], factor[1], 0.9, dtype=torch.complex128
            )
        else:
            circuit = double_excitation(
                n_qubits, factor[:2], factor[2:], 0.9, dtype=torch.complex128
            )
        for spin_orbital in (0, 1):
            circuit.x(spin_orbital)
        probabilities = circuit.probabilities().reshape(-1).real.numpy()
        for state, weight in enumerate(probabilities):
            if weight > 1e-12:
                occupations = sum(
                    (state >> (n_qubits - 1 - wire)) & 1 for wire in range(n_qubits)
                )
                assert occupations == n_electrons


def test_the_reference_occupation_a_caller_reads_is_the_lowest_electrons() -> None:
    for n_electrons, n_qubits in ((0, 4), (2, 4), (3, 6), (4, 8)):
        excitations = uccsd_excitations(n_electrons, n_qubits)
        assert excitations.reference_occupation == tuple(range(n_electrons))
        assert excitations.n_qubits == n_qubits
        assert excitations.n_electrons == n_electrons


def test_the_census_is_a_value_with_interned_collections() -> None:
    excitations = uccsd_excitations(2, 4)
    assert isinstance(excitations, UCCSDExcitations)
    assert not hasattr(excitations, "__dict__")
    with pytest.raises(Exception):
        excitations.n_electrons = 4


def test_each_same_spin_double_ascends_inside_its_collections() -> None:
    """A same-spin double is stored ascending, and the generator's sign follows.

    ``excitation_operator`` changes sign when the two indices of one collection
    are exchanged, so a census that stored a descending pair would hand every
    consumer a generator of the opposite sign while the circuit, which sorts its
    arguments, stayed the same. The mixed-spin doubles are deliberately not
    covered: their order is ``(occupied_alpha, occupied_beta, virtual_beta,
    virtual_alpha)``, which is a spin split rather than an ascent.
    """

    excitations = uccsd_excitations(4, 8)
    assert excitations.doubles_alpha
    assert excitations.doubles_beta
    for collection in (excitations.doubles_alpha, excitations.doubles_beta):
        for first_occupied, second_occupied, first_virtual, second_virtual in collection:
            assert first_occupied < second_occupied
            assert first_virtual < second_virtual
            assert second_occupied < first_virtual


def test_the_dense_oracle_agrees_with_the_generator_it_rebuilds() -> None:
    """Guard the oracle itself: the rebuilt and imported generators must agree."""

    for occupied, virtual in (((0,), (1,)), ((0, 1), (2, 3)), ((0, 2), (1, 3))):
        rebuilt = _generator(occupied, virtual)
        imported = excitation_operator(occupied, virtual)
        assert [
            (np.round(term.coefficient, 12), term.creations, term.annihilations)
            for term in rebuilt.terms
        ] == [
            (np.round(term.coefficient, 12), term.creations, term.annihilations)
            for term in imported.terms
        ]

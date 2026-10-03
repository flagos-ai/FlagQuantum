"""The Trotter product formula and the Pauli-exponential circuit it is built from.

The primitive is checked against the dense exponential the simulation layer already
owns, and the product formula is checked against ``torch.matrix_exp`` of the
Hamiltonian matrix.  Nothing here asserts an error *bound*: the two convergence
tests measure the observed order of the defect, which is a property of the
composition, not a bound on a caller's Hamiltonian.
"""

from __future__ import annotations

import math

import pytest
import torch

import flagquantum as fq
import flagquantum.algorithms as algorithms
from flagquantum.algorithms import (
    TROTTER_ORDERS,
    Hamiltonian,
    pauli_term,
    transverse_field_ising,
    trotter_circuit,
)
from flagquantum.algorithms import trotter as trotter_module
from flagquantum.algorithms.trotter import pauli_exponential_circuit
from flagquantum.compiler.resource_estimation import estimate_resources
from flagquantum.errors import CapabilityError, ValidationError
from flagquantum.simulation.pauli import (
    exponential_pauli_operator,
    pauli_product_operator,
    pauli_word_operators,
)
from flagquantum.simulation.unitary import get_unitary

pytestmark = pytest.mark.unit

_PHASE_FREE_TOLERANCE = 1e-12
# The residual left after one global phase is divided out.
#
# The primitive is exact up to a phase the gate set cannot express, so a comparison
# against a dense exponential must remove that phase before subtracting. The measured
# residual across every word below is at most 1.2e-16, four orders inside this value.


def _phase_free_residual(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return how far two unitaries are apart once one global phase is removed."""

    product = actual @ expected.conj().T
    phase = product[0, 0]
    if abs(phase) == 0.0:
        return float("inf")
    return float((product / phase - torch.eye(product.shape[0])).abs().max())


def _dense_exponential(theta: float, word: str, targets: tuple[int, ...], n: int):
    """Return ``exp(-i * theta * P)`` from the simulation layer's own operator."""

    return exponential_pauli_operator(
        theta,
        word,
        targets,
        n,
        dtype=torch.complex128,
        device="cpu",
    )


def _trotter_defect(order: int, steps: int, *, time: float = 0.4) -> float:
    """Return the largest entrywise gap between a Trotter circuit and exact evolution."""

    hamiltonian = transverse_field_ising(3, coupling=0.7, field=0.5)
    circuit = trotter_circuit(
        hamiltonian,
        time,
        steps=steps,
        order=order,
        dtype=torch.complex128,
    )
    exact = torch.matrix_exp(-1j * time * hamiltonian.matrix(dtype=torch.complex128))
    return float((get_unitary(circuit) - exact).abs().max())


# --- the Pauli exponential primitive ---


@pytest.mark.parametrize(
    ("word", "targets", "n_qubits"),
    [
        ("X", (0,), 1),
        ("Y", (0,), 1),
        ("Z", (0,), 1),
        ("XX", (0, 1), 2),
        ("YY", (0, 1), 2),
        ("ZZ", (0, 1), 2),
        ("XY", (0, 1), 2),
        ("XYZ", (0, 1, 2), 3),
        ("YI", (0, 1), 2),
        ("XZY", (2, 0, 1), 3),
        ("ZIZ", (0, 1, 2), 3),
    ],
)
def test_the_primitive_equals_the_dense_pauli_exponential(
    word: str, targets: tuple[int, ...], n_qubits: int
) -> None:
    theta = 0.37
    circuit = pauli_exponential_circuit(
        theta, word, targets, n_qubits, dtype=torch.complex128
    )
    residual = _phase_free_residual(
        get_unitary(circuit),
        _dense_exponential(theta, word, targets, n_qubits),
    )
    assert residual < _PHASE_FREE_TOLERANCE


def test_the_primitive_keeps_the_word_and_the_targets_positional() -> None:
    """``YX`` on ``(0, 1)`` is ``Y`` on wire 0 times ``X`` on wire 1, not the reverse."""

    circuit = pauli_exponential_circuit(0.4, "YX", (0, 1), 2, dtype=torch.complex128)
    expected = pauli_product_operator(
        pauli_word_operators("YX", (0, 1)),
        2,
        dtype=torch.complex128,
        device="cpu",
    )
    # exp(-i * theta * YX) has YX as its only non-identity component, so the dense
    # reference is reconstructible from the operator the module read.
    theta = 0.4
    reference = math.cos(theta) * torch.eye(4, dtype=torch.complex128) - (
        1j * math.sin(theta) * expected
    )
    assert _phase_free_residual(get_unitary(circuit), reference) < _PHASE_FREE_TOLERANCE


@pytest.mark.parametrize(
    ("word", "targets", "expected"),
    [
        ("ZZ", (0, 1), ["cx", "rz", "cx"]),
        ("X", (0,), ["h", "rz", "h"]),
        ("Y", (0,), ["sdg", "h", "rz", "h", "s"]),
        (
            "XYZ",
            (0, 1, 2),
            ["h", "sdg", "h", "cx", "cx", "rz", "cx", "cx", "h", "h", "s"],
        ),
    ],
)
def test_the_primitive_is_a_basis_change_a_ladder_and_one_rotation(
    word: str, targets: tuple[int, ...], expected: list[str]
) -> None:
    circuit = pauli_exponential_circuit(0.3, word, targets, 3, dtype=torch.complex128)
    assert [
        instruction.name for instruction in circuit.to_ir().instructions
    ] == expected


def test_a_diagonal_word_carries_no_basis_change() -> None:
    circuit = pauli_exponential_circuit(0.3, "ZZ", (0, 1), 2, dtype=torch.complex128)
    names = [instruction.name for instruction in circuit.to_ir().instructions]
    assert set(names) == {"cx", "rz"}


def test_the_rotation_is_double_the_requested_angle() -> None:
    """The ladder turns ``RZ(t)`` on the anchor into ``exp(-i * t * P)``."""

    circuit = pauli_exponential_circuit(0.25, "ZZ", (0, 1), 2, dtype=torch.complex128)
    rotations = [
        instruction
        for instruction in circuit.to_ir().instructions
        if instruction.name == "rz"
    ]
    assert len(rotations) == 1
    assert abs(float(rotations[0].params["theta"]) - 0.5) < 1e-15


def test_the_ladder_controls_on_the_remaining_wires_and_targets_the_first() -> None:
    circuit = pauli_exponential_circuit(
        0.3, "XYZ", (0, 1, 2), 3, dtype=torch.complex128
    )
    controls = [
        instruction.wires
        for instruction in circuit.to_ir().instructions
        if instruction.name == "cx"
    ]
    assert controls == [(1, 0), (2, 0), (1, 0), (2, 0)]


def test_the_primitive_honours_the_requested_dtype() -> None:
    single = pauli_exponential_circuit(0.3, "X", (0,), 1, dtype=torch.complex64)
    double = pauli_exponential_circuit(0.3, "X", (0,), 1, dtype=torch.complex128)
    assert single.state().dtype == torch.complex64
    assert double.state().dtype == torch.complex128


def test_the_primitive_defaults_to_the_runtime_dtype() -> None:
    circuit = pauli_exponential_circuit(0.3, "X", (0,), 1)
    assert circuit.state().dtype == fq.Circuit(1).state().dtype


def test_the_primitive_is_deterministic() -> None:
    first = pauli_exponential_circuit(0.3, "XYZ", (0, 1, 2), 3)
    second = pauli_exponential_circuit(0.3, "XYZ", (0, 1, 2), 3)
    assert first.to_ir().content_hash == second.to_ir().content_hash


@pytest.mark.parametrize("word", ["I", "II", "III"])
def test_an_identity_word_is_refused_because_its_exponential_is_a_global_phase(
    word: str,
) -> None:
    with pytest.raises(CapabilityError, match="global phase"):
        pauli_exponential_circuit(0.3, word, tuple(range(len(word))), len(word))


def test_an_identity_character_consumes_its_target_without_acting() -> None:
    """A word with support keeps its identity factors out of the ladder."""

    circuit = pauli_exponential_circuit(
        0.3, "ZIZ", (0, 1, 2), 3, dtype=torch.complex128
    )
    controls = [
        instruction.wires
        for instruction in circuit.to_ir().instructions
        if instruction.name == "cx"
    ]
    assert controls == [(2, 0), (2, 0)]


@pytest.mark.parametrize("word", ["A", "Zz", "X "])
def test_a_character_outside_the_pauli_labels_is_refused(word: str) -> None:
    with pytest.raises(ValidationError):
        pauli_exponential_circuit(0.3, word, tuple(range(len(word))), len(word))


def test_a_repeated_target_is_refused() -> None:
    with pytest.raises(ValidationError, match="unique"):
        pauli_exponential_circuit(0.3, "XX", (0, 0), 2)


def test_a_word_longer_than_its_targets_is_refused() -> None:
    with pytest.raises(ValidationError, match="one target"):
        pauli_exponential_circuit(0.3, "XX", (0,), 2)


def test_a_target_outside_the_register_is_refused() -> None:
    with pytest.raises(ValueError, match=r"outside a 2-wire circuit"):
        pauli_exponential_circuit(0.3, "Z", (3,), 2)


@pytest.mark.parametrize("width", [0, -1])
def test_a_register_that_cannot_hold_a_wire_is_refused(width: int) -> None:
    with pytest.raises(ValueError, match="n_qubits must be a positive integer"):
        pauli_exponential_circuit(0.3, "Z", (0,), width)


@pytest.mark.parametrize("width", [0, -1])
def test_an_unsupported_word_on_a_zero_width_register_is_refused_for_the_width(
    width: int,
) -> None:
    """The width is checked first, so the refusal names the width and not the word.

    An identity word has no support and would otherwise be refused as a global phase,
    which is a different and misleading reason for the same call.
    """

    with pytest.raises(ValueError, match="n_qubits must be a positive integer"):
        pauli_exponential_circuit(0.3, "I", (0,), width)


def test_a_negative_target_is_refused_by_the_word_reader() -> None:
    """The word reader owns target validation, so this module adds no second copy.

    The message is asserted in full because the circuit's own wire validation would
    otherwise catch the same call one layer later, with a message that also contains
    ``non-negative`` -- so a weaker match would not tell the two layers apart.
    """

    with pytest.raises(
        ValidationError, match=r"Pauli word targets must be non-negative, got \(-1,\)"
    ):
        pauli_exponential_circuit(0.3, "Z", (-1,), 2)


@pytest.mark.parametrize(
    "theta",
    [float("nan"), float("inf"), float("-inf"), True, False, "0.3", None, 1 + 0j],
)
def test_an_angle_that_is_not_a_finite_real_number_is_refused(theta: object) -> None:
    with pytest.raises((ValidationError, TypeError)):
        pauli_exponential_circuit(theta, "Z", (0,), 1)  # type: ignore[arg-type]


@pytest.mark.parametrize("theta", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_angle_is_refused_by_the_angle_reader(theta: float) -> None:
    """The reader owns finiteness, so a non-finite angle is refused before the circuit.

    The circuit's own angle validation would refuse it too, which is why the message
    is asserted: without that, a reader that passed the value through would still be
    caught one layer later and the reader's own check would be untested.
    """

    with pytest.raises(ValidationError, match="theta must be finite"):
        pauli_exponential_circuit(theta, "Z", (0,), 1)


def test_an_integer_angle_is_still_an_angle() -> None:
    circuit = pauli_exponential_circuit(1, "Z", (0,), 1, dtype=torch.complex128)
    rotations = [
        instruction
        for instruction in circuit.to_ir().instructions
        if instruction.name == "rz"
    ]
    assert abs(float(rotations[0].params["theta"]) - 2.0) < 1e-15


# --- the product formula ---


def test_the_product_formula_reaches_a_commuting_hamiltonian_exactly() -> None:
    """Two diagonal terms commute, so no order and no step count can introduce defect."""

    hamiltonian = Hamiltonian([pauli_term(1.0, "Z", 0), pauli_term(0.5, "Z", 1)])
    circuit = trotter_circuit(
        hamiltonian, 0.4, steps=3, order=1, dtype=torch.complex128
    )
    exact = torch.matrix_exp(-1j * 0.4 * hamiltonian.matrix(dtype=torch.complex128))
    assert _phase_free_residual(get_unitary(circuit), exact) < _PHASE_FREE_TOLERANCE


def test_one_term_is_the_primitive_again() -> None:
    hamiltonian = Hamiltonian([pauli_term(0.7, "XY", (0, 1))])
    circuit = trotter_circuit(
        hamiltonian, 0.3, steps=1, order=1, dtype=torch.complex128
    )
    reference = pauli_exponential_circuit(
        0.7 * 0.3, "XY", (0, 1), 2, dtype=torch.complex128
    )
    assert _phase_free_residual(get_unitary(circuit), get_unitary(reference)) < (
        _PHASE_FREE_TOLERANCE
    )


def test_the_symmetric_step_repeats_the_terms_in_reverse() -> None:
    """Order 2 is forward then backward, so the diagonal rotations read 0, 1, 1, 0."""

    hamiltonian = Hamiltonian([pauli_term(1.0, "Z", 0), pauli_term(0.5, "Z", 1)])
    circuit = trotter_circuit(
        hamiltonian, 0.4, steps=1, order=2, dtype=torch.complex128
    )
    rotations = [
        instruction.wires[0]
        for instruction in circuit.to_ir().instructions
        if instruction.name == "rz"
    ]
    assert rotations == [0, 1, 1, 0]


def test_the_first_order_step_applies_the_terms_forward_only() -> None:
    hamiltonian = Hamiltonian([pauli_term(1.0, "Z", 0), pauli_term(0.5, "Z", 1)])
    circuit = trotter_circuit(
        hamiltonian, 0.4, steps=1, order=1, dtype=torch.complex128
    )
    rotations = [
        instruction.wires[0]
        for instruction in circuit.to_ir().instructions
        if instruction.name == "rz"
    ]
    assert rotations == [0, 1]


def test_the_declared_term_order_is_preserved() -> None:
    forward = Hamiltonian([pauli_term(1.0, "Z", 0), pauli_term(1.0, "X", 1)])
    backward = Hamiltonian([pauli_term(1.0, "X", 1), pauli_term(1.0, "Z", 0)])
    first = trotter_circuit(forward, 0.4, order=1, dtype=torch.complex128)
    second = trotter_circuit(backward, 0.4, order=1, dtype=torch.complex128)
    assert _rotation_wires(first) == [0, 1]
    assert _rotation_wires(second) == [1, 0]
    assert first.to_ir().content_hash != second.to_ir().content_hash


def _rotation_wires(circuit: fq.Circuit) -> list[int]:
    """Return the wire of every rotation, in the order the circuit applies them."""

    return [
        instruction.wires[0]
        for instruction in circuit.to_ir().instructions
        if instruction.name == "rz"
    ]


def test_the_circuit_length_grows_with_the_step_count() -> None:
    hamiltonian = Hamiltonian([pauli_term(1.0, "Z", 0), pauli_term(0.5, "X", 1)])
    one = trotter_circuit(hamiltonian, 0.4, steps=1, order=1, dtype=torch.complex128)
    four = trotter_circuit(hamiltonian, 0.4, steps=4, order=1, dtype=torch.complex128)
    assert len(four) == 4 * len(one)


def test_a_wider_register_embeds_the_same_evolution() -> None:
    hamiltonian = Hamiltonian([pauli_term(0.8, "ZZ", (0, 1))])
    narrow = trotter_circuit(hamiltonian, 0.4, order=2, dtype=torch.complex128)
    wide = trotter_circuit(
        hamiltonian, 0.4, order=2, n_qubits=4, dtype=torch.complex128
    )
    assert narrow.n_wires == 2
    assert wide.n_wires == 4
    embedded = torch.kron(
        get_unitary(narrow).contiguous(),
        torch.eye(4, dtype=torch.complex128),
    )
    assert _phase_free_residual(get_unitary(wide), embedded) < _PHASE_FREE_TOLERANCE


def test_the_first_order_defect_is_second_order_in_the_step_length() -> None:
    coarse = _trotter_defect(1, 2)
    fine = _trotter_defect(1, 4)
    assert coarse > 0.0
    assert 1.9 < coarse / fine < 2.1


def test_the_second_order_defect_is_third_order_in_the_step_length() -> None:
    coarse = _trotter_defect(2, 2)
    fine = _trotter_defect(2, 4)
    assert coarse > 0.0
    assert 3.7 < coarse / fine < 4.3


def test_the_symmetric_step_is_more_accurate_than_the_unsymmetric_one() -> None:
    first = _trotter_defect(1, 4)
    second = _trotter_defect(2, 4)
    assert second < first / 10.0


def test_more_steps_bring_the_product_formula_closer_to_exact_evolution() -> None:
    assert _trotter_defect(2, 8) < 1e-3


def test_the_product_formula_is_deterministic() -> None:
    hamiltonian = transverse_field_ising(3)
    first = trotter_circuit(hamiltonian, 0.4, order=2)
    second = trotter_circuit(hamiltonian, 0.4, order=2)
    assert first.to_ir().content_hash == second.to_ir().content_hash


def _coefficient_value_and_gradient(
    value: float, *, order: int, steps: int
) -> tuple[float, float]:
    """Return the register value and its derivative from one Trotter circuit.

    A rotation about ``y`` after the evolution turns the evolved angle into a register
    value, which is what makes the coefficient observable at all: without it the
    register value would not depend on the angle the coefficient scales.
    """

    coefficient = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    hamiltonian = Hamiltonian([pauli_term(coefficient, "X", 0)])
    circuit = trotter_circuit(
        hamiltonian, 0.3, steps=steps, order=order, dtype=torch.complex128
    )
    circuit.ry(0, theta=0.7)
    observed = circuit.expectation_ps(z=[0]).sum()
    observed.backward()
    assert coefficient.grad is not None
    return float(observed.detach()), float(coefficient.grad)


def _exact_coefficient_value(value: float) -> float:
    """Return the same register value from the dense exponential, with no circuit.

    One term is its own product formula at every order and every step count, because
    the halves of a step commute with each other, so the Trotter circuit and this
    reference are the same evolution and their derivatives must agree.
    """

    operator = exponential_pauli_operator(
        value * 0.3,
        "X",
        (0,),
        1,
        dtype=torch.complex128,
        device="cpu",
    )
    state = torch.zeros(2, dtype=torch.complex128)
    state[0] = 1.0
    half = 0.7 / 2.0
    rotation = torch.tensor(
        [
            [math.cos(half), -math.sin(half)],
            [math.sin(half), math.cos(half)],
        ],
        dtype=torch.complex128,
    )
    evolved = rotation @ (operator @ state)
    return float((evolved[0].abs() ** 2 - evolved[1].abs() ** 2).real)


@pytest.mark.parametrize(("order", "steps"), [(1, 1), (2, 1), (2, 3)])
def test_a_tensor_coefficient_keeps_a_gradient_that_matches_the_exact_evolution(
    order: int, steps: int
) -> None:
    value, gradient = _coefficient_value_and_gradient(0.6, order=order, steps=steps)
    assert abs(value - _exact_coefficient_value(0.6)) < 1e-12

    step = 1e-6
    reference = (
        _exact_coefficient_value(0.6 + step) - _exact_coefficient_value(0.6 - step)
    ) / (2 * step)
    assert reference != 0.0
    assert abs(gradient - reference) < 1e-8


def test_a_tensor_coefficient_at_the_second_order_is_exact_for_one_term() -> None:
    """Two half steps of one term compose to the whole step, so the order cannot show."""

    first, _ = _coefficient_value_and_gradient(0.8, order=1, steps=2)
    second, _ = _coefficient_value_and_gradient(0.8, order=2, steps=2)
    assert abs(first - second) < 1e-14


# --- refusals ---


@pytest.mark.parametrize(
    ("pauli", "wires"),
    [("I", 0), ("II", (0, 1))],
)
def test_a_term_that_is_a_multiple_of_the_identity_is_refused(
    pauli: str, wires: int | tuple[int, int]
) -> None:
    hamiltonian = Hamiltonian([pauli_term(1.0, pauli, wires)])
    with pytest.raises(CapabilityError, match="term 0 is a multiple of the identity"):
        trotter_circuit(hamiltonian, 0.4)


def test_an_identity_term_is_refused_before_a_later_term_can_hide_it() -> None:
    hamiltonian = Hamiltonian([pauli_term(1.0, "II", (0, 1)), pauli_term(1.0, "Z", 0)])
    with pytest.raises(CapabilityError, match="term 0 is a multiple of the identity"):
        trotter_circuit(hamiltonian, 0.4)


def test_the_refusal_names_the_term_it_came_from() -> None:
    hamiltonian = Hamiltonian(
        [pauli_term(1.0, "Z", 0), pauli_term(1.0, "Z", 1), pauli_term(1.0, "I", 2)]
    )
    with pytest.raises(CapabilityError, match="term 2 is a multiple of the identity"):
        trotter_circuit(hamiltonian, 0.4)


@pytest.mark.parametrize("coefficient", [1.0 + 2.0j, 0.5j, -3j])
def test_a_coefficient_with_an_imaginary_part_is_refused(coefficient: complex) -> None:
    hamiltonian = Hamiltonian([pauli_term(coefficient, "Z", 0)])
    with pytest.raises(CapabilityError, match="non-Hermitian"):
        trotter_circuit(hamiltonian, 0.4)


def test_a_complex_coefficient_tensor_is_refused() -> None:
    coefficient = torch.tensor(1.0 + 1.0j, dtype=torch.complex128)
    hamiltonian = Hamiltonian([pauli_term(coefficient, "Z", 0)])
    with pytest.raises(CapabilityError, match="non-Hermitian"):
        trotter_circuit(hamiltonian, 0.4)


def test_a_coefficient_holding_several_values_is_refused() -> None:
    coefficient = torch.tensor([1.0, 2.0], dtype=torch.float64)
    hamiltonian = Hamiltonian([pauli_term(coefficient, "Z", 0)])
    with pytest.raises(ValueError, match="a Hamiltonian term carries one"):
        trotter_circuit(hamiltonian, 0.4)


@pytest.mark.parametrize("coefficient", [True, "1.0", b"1.0", None, [1.0]])
def test_a_coefficient_that_is_not_a_number_is_refused(coefficient: object) -> None:
    hamiltonian = Hamiltonian([pauli_term(coefficient, "Z", 0)])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="which is not a number"):
        trotter_circuit(hamiltonian, 0.4)


@pytest.mark.parametrize("order", [0, 3, -1, True, 1.0, "1", None])
def test_an_unsupported_order_is_refused(order: object) -> None:
    hamiltonian = Hamiltonian([pauli_term(1.0, "Z", 0)])
    with pytest.raises(ValueError, match="order must be one of 1, 2"):
        trotter_circuit(hamiltonian, 0.4, order=order)  # type: ignore[arg-type]


@pytest.mark.parametrize("steps", [0, -1, True, 1.0, "1", None])
def test_a_step_count_that_is_not_a_positive_integer_is_refused(steps: object) -> None:
    hamiltonian = Hamiltonian([pauli_term(1.0, "Z", 0)])
    with pytest.raises(ValueError, match="steps must be a positive integer"):
        trotter_circuit(hamiltonian, 0.4, steps=steps)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "time", [float("nan"), float("inf"), True, "0.4", None, 1 + 0j]
)
def test_a_time_that_is_not_a_finite_real_number_is_refused(time: object) -> None:
    hamiltonian = Hamiltonian([pauli_term(1.0, "Z", 0)])
    with pytest.raises((ValidationError, TypeError)):
        trotter_circuit(hamiltonian, time)  # type: ignore[arg-type]


def test_a_register_narrower_than_a_term_is_refused() -> None:
    hamiltonian = Hamiltonian([pauli_term(1.0, "Z", 2)])
    with pytest.raises(ValueError, match=r"outside a 2-wire circuit"):
        trotter_circuit(hamiltonian, 0.4, n_qubits=2)


def test_a_zero_width_register_cannot_hold_a_term() -> None:
    hamiltonian = Hamiltonian([pauli_term(1.0, "Z", 0)])
    with pytest.raises(ValueError, match=r"outside a 0-wire circuit"):
        trotter_circuit(hamiltonian, 0.4, n_qubits=0)


def test_a_negative_time_reverses_the_evolution() -> None:
    """A negative time is a legitimate input: it is the inverse evolution."""

    hamiltonian = Hamiltonian([pauli_term(0.9, "XZ", (0, 1))])
    forward = trotter_circuit(hamiltonian, 0.4, order=2, dtype=torch.complex128)
    backward = trotter_circuit(hamiltonian, -0.4, order=2, dtype=torch.complex128)
    product = get_unitary(forward) @ get_unitary(backward)
    phase = product[0, 0]
    assert float(
        (product / phase - torch.eye(4, dtype=torch.complex128)).abs().max()
    ) < (_PHASE_FREE_TOLERANCE)


# --- the module surface ---


def test_the_declared_orders_are_the_ones_the_module_builds() -> None:
    assert TROTTER_ORDERS == (1, 2)
    assert trotter_module.TROTTER_ORDERS is TROTTER_ORDERS


def test_every_exported_name_resolves() -> None:
    for name in trotter_module.__all__:
        assert hasattr(trotter_module, name), name


@pytest.mark.parametrize(
    "name", ["TROTTER_ORDERS", "pauli_exponential_circuit", "trotter_circuit"]
)
def test_the_package_exposes_the_module_objects_and_not_copies(name: str) -> None:
    """The subpackage surface is the module's own object, not a second definition."""

    assert getattr(algorithms, name) is getattr(trotter_module, name)


def test_the_package_declares_every_trotter_name_it_imports() -> None:
    """A name imported but left out of ``__all__`` is invisible to a star import."""

    for name in ("TROTTER_ORDERS", "pauli_exponential_circuit", "trotter_circuit"):
        assert name in algorithms.__all__, name


def test_the_result_is_an_ordinary_circuit_the_compiler_already_owns() -> None:
    """The module emits gates, not a new instruction, so the passes already read it."""

    hamiltonian = Hamiltonian([pauli_term(1.0, "ZZ", (0, 1)), pauli_term(0.5, "X", 0)])
    circuit = trotter_circuit(hamiltonian, 0.4, order=2, dtype=torch.complex128)
    estimate = estimate_resources(circuit)
    assert estimate.n_wires == 2
    assert estimate.used_wires == 2
    assert estimate.n_operations == len(circuit)
    assert estimate.operation_counts["rz"] == 4
    assert estimate.t_count == 0


def test_the_circuit_executes_to_the_exact_evolved_state() -> None:
    """The whole vertical path, from a Hamiltonian to a state, is the evolution asked for."""

    hamiltonian = transverse_field_ising(2, coupling=0.8, field=0.3)
    circuit = trotter_circuit(
        hamiltonian, 0.5, steps=4, order=2, dtype=torch.complex128
    )
    exact = torch.matrix_exp(-1j * 0.5 * hamiltonian.matrix(dtype=torch.complex128))
    gap = float((circuit.state()[0] - exact[:, 0]).abs().max())
    assert gap < 1e-3

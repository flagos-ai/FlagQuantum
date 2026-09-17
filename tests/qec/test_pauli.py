"""Unit coverage for phase-free Pauli operators."""

from __future__ import annotations

import pytest

from flagquantum.qec.pauli import Pauli

pytestmark = pytest.mark.unit


def test_identity_has_no_support_and_renders_as_i() -> None:
    identity = Pauli()

    assert identity.support == ()
    assert identity.weight == 0
    assert identity.is_identity
    assert identity.to_text() == "I"


@pytest.mark.parametrize(
    ("label", "expected"),
    (
        ("i", Pauli()),
        ("x", Pauli(x_wires=(2,))),
        ("Y", Pauli(x_wires=(2,), z_wires=(2,))),
        ("z", Pauli(z_wires=(2,))),
    ),
)
def test_from_label_maps_every_pauli_letter(label: str, expected: Pauli) -> None:
    assert Pauli.from_label(label, 2) == expected


def test_from_label_rejects_an_unknown_letter() -> None:
    with pytest.raises(ValueError, match="label"):
        Pauli.from_label("q", 0)


def test_from_label_rejects_a_negative_wire() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Pauli.from_label("x", -1)


def test_support_unions_both_components_in_wire_order() -> None:
    pauli = Pauli(x_wires=(1, 3), z_wires=(2,))

    assert pauli.support == (1, 2, 3)
    assert pauli.weight == 3


def test_weight_counts_a_y_wire_once() -> None:
    assert Pauli(x_wires=(0,), z_wires=(0,)).weight == 1


@pytest.mark.parametrize(
    "pauli",
    (
        Pauli(),
        Pauli(x_wires=(0,)),
        Pauli(z_wires=(2,)),
        Pauli(x_wires=(0,), z_wires=(0,)),
        Pauli(x_wires=(0, 4), z_wires=(2,)),
    ),
)
def test_text_round_trips(pauli: Pauli) -> None:
    assert Pauli.from_text(pauli.to_text()) == pauli


@pytest.mark.parametrize("text", ("", "   ", "Q0", "X", "X0**Z1", "I0"))
def test_from_text_rejects_malformed_text(text: str) -> None:
    with pytest.raises(ValueError):
        Pauli.from_text(text)


@pytest.mark.parametrize(
    "wires",
    ({"x_wires": (2, 0)}, {"x_wires": (1, 1)}, {"z_wires": (3, 1)}),
)
def test_unordered_or_repeated_wires_are_rejected(wires: dict) -> None:
    with pytest.raises(ValueError, match="increasing"):
        Pauli(**wires)


def test_negative_wires_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Pauli(z_wires=(-1,))


def test_non_integer_wires_are_rejected() -> None:
    with pytest.raises(TypeError, match="integer"):
        Pauli(x_wires=(1.5,))


def test_booleans_are_not_wire_indices() -> None:
    with pytest.raises(TypeError, match="integer"):
        Pauli(x_wires=(True,))


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    (
        (Pauli(x_wires=(0,)), Pauli(z_wires=(0,)), False),
        (Pauli(x_wires=(0,)), Pauli(z_wires=(1,)), True),
        (Pauli(x_wires=(0, 1)), Pauli(z_wires=(0, 1)), True),
        (Pauli(x_wires=(0,), z_wires=(0,)), Pauli(z_wires=(0,)), False),
        (Pauli(), Pauli(x_wires=(5,), z_wires=(5,)), True),
        (Pauli(x_wires=(0,)), Pauli(x_wires=(0,)), True),
        (Pauli(z_wires=(0, 1)), Pauli(x_wires=(1,), z_wires=(0,)), False),
    ),
)
def test_commutes_with_uses_the_symplectic_product(
    left: Pauli, right: Pauli, expected: bool
) -> None:
    assert left.commutes_with(right) is expected
    assert right.commutes_with(left) is expected


def test_composition_cancels_a_repeated_letter() -> None:
    assert Pauli(x_wires=(0,)) * Pauli(x_wires=(0,)) == Pauli()


def test_composition_of_x_and_z_on_one_wire_is_y() -> None:
    assert Pauli(x_wires=(0,)) * Pauli(z_wires=(0,)) == Pauli(
        x_wires=(0,), z_wires=(0,)
    )


def test_composition_is_commutative_and_associative() -> None:
    left = Pauli(x_wires=(0, 2))
    middle = Pauli(z_wires=(2, 3))
    right = Pauli(x_wires=(3,))

    assert left * middle == middle * left
    assert (left * middle) * right == left * (middle * right)


def test_equal_paulis_compare_equal_and_hash_equal() -> None:
    assert Pauli(x_wires=(1,)) == Pauli(x_wires=(1,))
    assert len({Pauli(x_wires=(1,)), Pauli(x_wires=(1,))}) == 1

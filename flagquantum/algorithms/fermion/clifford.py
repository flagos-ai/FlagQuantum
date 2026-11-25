"""Exact symbolic H, S dagger and CX conjugation of native Pauli values."""

from __future__ import annotations

from ...core.algebra.pauli import PauliOperator, PauliTerm, pauli
from ...core.ir import Instruction


def symplectic_masks(term: PauliTerm) -> tuple[int, int]:
    """Return logical-qubit X and Z bit masks; Y sets both bits."""
    x = sum(1 << q for q, axis in term.factors if axis in ("x", "y"))
    z = sum(1 << q for q, axis in term.factors if axis in ("y", "z"))
    return x, z


def pauli_words_commute(left: PauliTerm, right: PauliTerm) -> bool:
    """Check the binary symplectic product, independently of coefficients."""
    lx, lz = symplectic_masks(left)
    rx, rz = symplectic_masks(right)
    return ((lx & rz).bit_count() + (lz & rx).bit_count()) % 2 == 0


def conjugate_pauli(operator: PauliOperator, gate: Instruction) -> PauliOperator:
    """Return G operator G dagger for a native H, S dagger or CX instruction.

    No matrix is constructed. In particular, the CX image of Y is calculated
    as i times its X image times its Z image, retaining its exact phase.
    """
    if gate.name not in ("h", "sdg", "cx") or gate.matrix is not None or gate.params:
        raise ValueError(
            "symbolic stabilizer conjugation supports fixed h, sdg and cx only"
        )
    terms: list[PauliTerm] = []
    for term in operator.terms:
        image = pauli(term.coefficient)
        for qubit, axis in term.factors:
            if gate.name != "cx":
                coefficient = 1
                mapped = axis
                if qubit == gate.wires[0]:
                    if gate.name == "h":
                        mapped = {"x": "z", "y": "y", "z": "x"}[axis]
                        coefficient = -1 if axis == "y" else 1
                    else:
                        mapped = {"x": "y", "y": "x", "z": "z"}[axis]
                        coefficient = -1 if axis == "x" else 1
                image = image * pauli(coefficient, ((qubit, mapped),))
            else:
                control, target = gate.wires
                x = pauli(factors=((qubit, "x"),))
                z = pauli(factors=((qubit, "z"),))
                if qubit == control:
                    x = x * pauli(factors=((target, "x"),))
                if qubit == target:
                    z = z * pauli(factors=((control, "z"),))
                image = image * (x if axis == "x" else z if axis == "z" else 1j * x * z)
        terms.extend(image.terms)
    return PauliOperator(tuple(terms))

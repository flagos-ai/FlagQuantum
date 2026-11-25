# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from the call graphs in Qualtran v0.7.0 Hubbard hopping and interaction.
# https://github.com/quantumlib/Qualtran/tree/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/trotter/hubbard

"""Declared logical costs for the locked square-lattice Hubbard terms."""

from math import isfinite
from typing import cast

from ...compiler._component_trace.resource_classification import classify_resource_gate
from ...compiler._component_trace.resource_generalization import RotationCall
from ...compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
)
from ...compiler.registers.layout.fermionic_signatures import (
    four_mode_signature,
    spin_word_signature,
)
from ...compiler.registers.phase.hamming_weight_phasing import (
    source_hamming_weight_phasing_cost,
)
from ...compiler.registers.resources.symbolic_resource_model import (
    SymbolicResourceFormula,
)
from ...core.ir.program.signatures import Signature
from ...core.value_math import SymbolicInt, SymbolicReal, smax
from .fermionic_fourier import source_two_bit_fermionic_fourier_cost


def _source_rz_cost(angle: float) -> ComponentCost:
    if not isfinite(angle):
        raise ValueError("Hubbard rotation angle must be finite")
    family = classify_resource_gate(RotationCall("rz", angle))
    return ComponentCost(
        "Rz",
        1,
        GateCost(
            t=int(family == "t"),
            clifford=int(family == "clifford"),
            rotation=int(family == "rotation"),
        ),
    )


def source_hubbard_interaction_signature(length: SymbolicInt) -> Signature:
    """Declare the two side-width through words in both interaction variants."""
    return spin_word_signature(length)


def source_hubbard_interaction_cost(
    length: int, angle: float, hubb_u: float, *, hamming_weight: bool = False
) -> ComponentCost:
    """Count one Rz per site or two batches of Hamming-weight phasing."""
    if type(length) is not int or length < 1:
        raise ValueError("Hubbard lattice length must be a positive integer")
    width = int(source_hubbard_interaction_signature(length).n_qubits())
    exponent = angle * hubb_u
    if hamming_weight:
        phase = source_hamming_weight_phasing_cost(length**2 // 2, exponent)
        return ComponentCost("InteractionHWP", width, calls=(CostCall(phase, 2),))
    return ComponentCost(
        "Interaction", width, calls=(CostCall(_source_rz_cost(exponent), length**2),)
    )


def source_hubbard_hopping_plaquette_signature() -> Signature:
    """Declare the source four-qubit through array."""
    return four_mode_signature()


def source_hubbard_hopping_tile_signature(length: SymbolicInt) -> Signature:
    """Declare the source's two opaque spin words at the lattice side width."""
    if isinstance(length, int) and length % 2:
        raise ValueError("Only even length lattices are supported")
    return spin_word_signature(length)


def source_hubbard_hopping_plaquette_cost(kappa: float) -> ComponentCost:
    """Count four two-bit Fourier calls and two equal-angle Rz calls."""
    return ComponentCost(
        "HoppingPlaquette",
        int(source_hubbard_hopping_plaquette_signature().n_qubits()),
        calls=(
            CostCall(source_two_bit_fermionic_fourier_cost(0, 1), 4),
            CostCall(_source_rz_cost(kappa), 2),
        ),
    )


def source_hubbard_hopping_tile_cost(
    length: int,
    angle: float,
    tau: float = 1,
    *,
    pink: bool = True,
    hamming_weight: bool = False,
) -> ComponentCost:
    """Count the locked plaquette or batched phasing graph for a single color."""
    if type(length) is not int or length < 2 or length % 2:
        raise ValueError("Hubbard hopping requires a positive even lattice length")
    width = int(source_hubbard_hopping_tile_signature(length).n_qubits())
    color = "pink" if pink else "gold"
    kappa = tau * angle
    plaquettes = length**2 // 2
    if hamming_weight:
        return ComponentCost(
            f"HoppingTileHWP[{color}]",
            width,
            calls=(
                CostCall(source_two_bit_fermionic_fourier_cost(0, 1), 4 * plaquettes),
                CostCall(source_hamming_weight_phasing_cost(plaquettes, kappa), 2),
            ),
        )
    return ComponentCost(
        f"HoppingTile[{color}]",
        width,
        calls=(CostCall(source_hubbard_hopping_plaquette_cost(kappa), plaquettes),),
    )


def source_hubbard_hopping_plaquette_formula(
    kappa: SymbolicReal,
) -> SymbolicResourceFormula:
    """Count the plaquette declaration for a concrete or symbolic rotation.

    This is a logical estimate, not an emitted gate stream or synthesis bound.
    The four Fourier children reuse their independent concrete resource owner.
    """
    if isinstance(kappa, (int, float)) and not isfinite(kappa):
        raise ValueError("Hubbard rotation angle must be finite")
    fourier = source_two_bit_fermionic_fourier_cost(0, 1).gates
    family = classify_resource_gate(RotationCall("rz", kappa))
    return SymbolicResourceFormula(
        and_bloq=0,
        measurement=0,
        logical_qubits=4,
        t=4 * fourier.t + 2 * int(family == "t"),
        clifford=4 * fourier.clifford + 2 * int(family == "clifford"),
        rotation=4 * fourier.rotation + 2 * int(family == "rotation"),
    )


def source_hubbard_hopping_tile_formula(
    length: SymbolicInt, angle: SymbolicReal, tau: SymbolicReal = 1
) -> SymbolicResourceFormula:
    """Estimate an ordinary tile without requiring a concrete lattice size.

    The source declares floor(L**2/2) plaquettes and uses Max(4, 2*L) for
    symbolic width. HWP is a separate declaration whose unchanged symbolic
    source resource queries fail; it is not substituted here.
    """
    if isinstance(length, int) and (length < 2 or length % 2):
        raise ValueError("Hubbard hopping requires a positive even lattice length")
    source_hubbard_hopping_tile_signature(length)
    plaquette = source_hubbard_hopping_plaquette_formula(tau * angle)
    repetitions = length**2 // 2
    return SymbolicResourceFormula(
        and_bloq=0,
        measurement=0,
        logical_qubits=cast(SymbolicInt, smax(4, 2 * length)),
        t=repetitions * plaquette.t,
        clifford=repetitions * plaquette.clifford,
        rotation=repetitions * plaquette.rotation,
    )


def source_hubbard_interaction_formula(
    length: SymbolicInt, angle: SymbolicReal, hubb_u: SymbolicReal
) -> SymbolicResourceFormula:
    """Estimate the ordinary interaction's L**2 declared equal-angle Rz calls.

    Source angles and word widths belong to this model. They differ from
    the native shifted-ZZ physical binding described in the file audit.
    """
    if isinstance(length, int) and length < 1:
        raise ValueError("Hubbard lattice length must be a positive integer")
    exponent = angle * hubb_u
    if isinstance(exponent, (int, float)) and not isfinite(exponent):
        raise ValueError("Hubbard rotation angle must be finite")
    family = classify_resource_gate(RotationCall("rz", exponent))
    return SymbolicResourceFormula(
        and_bloq=0,
        measurement=0,
        logical_qubits=cast(SymbolicInt, smax(1, 2 * length)),
        t=length**2 * int(family == "t"),
        clifford=length**2 * int(family == "clifford"),
        rotation=length**2 * int(family == "rotation"),
    )

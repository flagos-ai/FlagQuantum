# Copyright 2023 Google LLC
# SPDX-License-Identifier: Apache-2.0
# Adapted from Qualtran v0.7.0's chemistry resource estimation tutorial.
# https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/chemistry/resource_estimation.ipynb

"""Build the original chemistry tutorial's three resource comparisons."""

from dataclasses import dataclass, replace
from math import pi
from typing import cast

import numpy as np
from numpy.typing import NDArray

from flagquantum.algorithms.first_quantization import first_quantization_signature
from flagquantum.algorithms.first_quantization_resource_model import (
    source_prepare_first_quantization_cost,
    source_select_first_quantization_cost,
)
from flagquantum.algorithms.primitives.double_factorization_encoding_model import (
    source_double_factorization_block_encoding_cost,
)
from flagquantum.algorithms.primitives.multiplexers.thc_resource_model import (
    select_thc_cost,
    thc_rotations_cost,
)
from flagquantum.algorithms.primitives.thc_chemistry_data import thc_chemistry_data
from flagquantum.algorithms.primitives.thc_chemistry_prepare import PrepareTHCChemistry
from flagquantum.compiler._component_trace.gate_budget import t_and_ccz_budget
from flagquantum.compiler._component_trace.resource_model import (
    ComponentCost,
    CostCall,
    GateCost,
    estimate_component_resources,
)
from flagquantum.drawer.register_score import (
    RegisterScore,
    RegisterStep,
    layout_register_score,
)

SPIN_ORBITALS = 108
THC_RANK = 350
ANGLE_BITS = 16
ALIAS_BITS = 10
ELECTRON_COUNTS = (10, 20, 40, 60, 80)
SCENARIOS = ("SELECT", "PREPARE", "PREPARE†")


@dataclass(frozen=True)
class THCTutorialModels:
    normalization: float
    select: ComponentCost
    prepare: ComponentCost
    unprepare: ComponentCost
    rotations: ComponentCost


def thc_fixture() -> (
    tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
):
    """Use the source contractions with a reproducible synthetic input."""
    spatial = SPIN_ORBITALS // 2
    rng = np.random.default_rng(398)
    tpq = rng.normal(size=(spatial, spatial))
    zeta = rng.normal(size=(THC_RANK, THC_RANK))
    zeta = 0.5 * (zeta + zeta.T)
    eta = rng.normal(size=(THC_RANK, spatial))
    eri = np.einsum("Pp,Pr,Qq,Qs,PQ->prqs", eta, eta, eta, eta, zeta, optimize=True)
    corrected = tpq - 0.5 * np.einsum("illj->ij", eri) + np.einsum("llij->ij", eri)
    return np.linalg.eigvalsh(corrected), eta, zeta


def thc_models(
    eigenvalues: NDArray[np.float64],
    eta: NDArray[np.float64],
    zeta: NDArray[np.float64],
) -> THCTutorialModels:
    """Build the source LCU components at the tutorial's precisions."""
    data = thc_chemistry_data(eigenvalues, eta, zeta, probability_bits=ALIAS_BITS)
    prep = PrepareTHCChemistry(data)
    rotation = thc_rotations_cost(
        data.num_mu, data.num_spin_orb, ANGLE_BITS, kr1=16, kr2=16
    )
    rotation_graph = replace(
        rotation,
        gates=GateCost(),
        calls=(
            CostCall(
                ComponentCost("Toffoli", 3, GateCost(toffoli=1)), rotation.gates.toffoli
            ),
        ),
    )
    return THCTutorialModels(
        data.normalization,
        select_thc_cost(
            data.num_mu,
            data.num_spin_orb,
            ANGLE_BITS,
            data.probability_bits,
            kr1=16,
            kr2=16,
        ),
        prep.source_resource_model(),
        prep.source_resource_model(adjoint=True),
        rotation_graph,
    )


def _generalize_swaps(model: ComponentCost) -> ComponentCost:
    if model.name == "SwapWithZero":
        # This model leaf consists exclusively of the QROAM CSwapApprox ladder.
        return replace(model, gates=GateCost(cswap=model.gates.t // 4))
    return replace(
        model,
        calls=tuple(
            replace(call, component=_generalize_swaps(call.component))
            for call in model.calls
        ),
    )


def modeled_ccz(component: ComponentCost) -> int:
    """Apply the tutorial's approximate-swap generalizer and CCZ accounting."""
    gates = estimate_component_resources(_generalize_swaps(component)).gates
    return cast(int, t_and_ccz_budget(gates, ts_per_rotation=0)["n_ccz"])


def thc_prepare_t_breakdown(prep: ComponentCost) -> dict[str, int]:
    """Classify PREPARE's original T equivalents at the same stopping nodes."""
    groups: dict[str, int] = {}
    families = {
        "ToContiguousIndex": "arithmetic",
        "QROAMClean": "data_loading",
        "LessThanEqual": "arithmetic",
        "CSwap": "swaps",
        "MultiControlX": "multi_control_pauli",
    }
    for call in prep.calls:
        node = call.component
        if node.name == "UniformSuperpositionTHC":
            groups["rotations"] = 11 * node.gates.rotation
            groups["toffoli"] = 4 * node.gates.toffoli
            for child in node.calls:
                family = {
                    "ReflectionUsingPrepare": "reflection",
                    "MultiControlX": "multi_control_pauli",
                }.get(child.component.name, "arithmetic")
                count = estimate_component_resources(
                    child.component
                ).gates.total_t_count()
                groups[family] = groups.get(family, 0) + child.repetitions * count
        else:
            family = families[node.name]
            count = estimate_component_resources(node).gates.total_t_count()
            groups[family] = groups.get(family, 0) + call.repetitions * count
    return groups


def df_model() -> ComponentCost:
    return source_double_factorization_block_encoding_cost(
        360, SPIN_ORBITALS, 13031, ALIAS_BITS, ANGLE_BITS
    )


def volume_from_rs_eta(rs: float, eta: float) -> float:
    return rs**3 * (4.0 * pi * eta / 3)


def approximate_alpha_first_quantization(
    volume: float, eta: float, momentum_bits: int
) -> float:
    """Use the tutorial's cubic-cell approximation for the LCU one-norm."""
    plane_waves = (2**momentum_bits - 1) ** 3
    kinetic = 6 * eta * pi**2 / volume ** (2 / 3) * (2 ** (momentum_bits - 1) - 1) ** 2
    potential = eta**2 * (plane_waves / volume) ** (1 / 3)
    return float(kinetic + 2 * potential)


def first_quantization_models(
    eta: int,
) -> tuple[ComponentCost, ComponentCost, ComponentCost]:
    return (
        source_select_first_quantization_cost(6, eta, eta),
        source_prepare_first_quantization_cost(6, eta, eta, eta),
        source_prepare_first_quantization_cost(6, eta, eta, eta, adjoint=True),
    )


def first_quantization_select_breakdown(select: ComponentCost) -> dict[str, int]:
    """Group the SELECT clauses by source child, omitting bookkeeping."""
    totals: dict[str, int] = {}
    for call in select.calls:
        name = call.component.name
        totals[name] = totals.get(name, 0) + call.repetitions * modeled_ccz(
            call.component
        )
    return dict(sorted(totals.items()))


def first_quantization_select_score() -> RegisterScore:
    """Lay out the two-particle SELECT tutorial with its scratch lifetimes."""
    signature = first_quantization_signature(6, 2, 80, 256, prepare=False)
    widths = {reg.name: int(reg.total_bits()) for reg in signature}
    inputs = tuple(widths)
    widths.update(p=18, q=18, rl=16)

    def step(name: str, *ports: str) -> RegisterStep:
        return RegisterStep(name, ports, ports)

    return layout_register_score(
        widths,
        inputs,
        (
            RegisterStep("Allocate p", outputs=("p",)),
            RegisterStep("Allocate q", outputs=("q",)),
            RegisterStep("Allocate Rl", outputs=("rl",)),
            step("MultiSwap", "i", "sys", "p"),
            step("MultiSwap", "j", "sys", "q"),
            step("SELECT T", "tuv", "plus_t", "w", "r", "s", "p"),
            step("SELECT UV", "tuv", "uv", "l", "rl", "nu_x", "nu_y", "nu_z", "p", "q"),
            step("MultiSwap", "i", "sys", "p"),
            step("MultiSwap", "j", "sys", "q"),
            RegisterStep("Free p", ("p",)),
            RegisterStep("Free q", ("q",)),
            RegisterStep("Free Rl", ("rl",)),
        ),
    )


def thc_select_score() -> RegisterScore:
    """Draw the locked SELECT port order with the shared angle register."""
    orbitals = SPIN_ORBITALS // 2
    address_bits = THC_RANK.bit_length()
    widths = dict(
        succ=1,
        nu_eq_mp1=1,
        mu=address_bits,
        nu=address_bits,
        plus_mn=1,
        plus_a=1,
        plus_b=1,
        sigma=ALIAS_BITS,
        rot=1,
        sys_a=orbitals,
        sys_b=orbitals,
    )
    inputs = tuple(widths)
    widths["data"] = ANGLE_BITS
    spin_swap = ("plus_b", "sys_a", "sys_b")
    rotation = ("nu_eq_mp1", "data", "mu", "sys_a")
    ordered = (
        ("CSwap", spin_swap),
        ("In_mu-R", rotation),
        ("Controlled Z₀", ("succ", "sys_b")),
        ("In_mu-R†", rotation),
        ("CSwap", spin_swap),
        ("X", ("plus_mn",)),
        ("CSwap flags", ("nu_eq_mp1", "plus_a", "plus_b")),
        ("CSwap addresses", ("nu_eq_mp1", "mu", "nu")),
        ("CSwap", spin_swap),
        ("In_mu-R", rotation),
        ("Controlled Z₀", ("succ", "nu_eq_mp1", "sys_b")),
        ("In_mu-R†", rotation),
        ("CSwap", spin_swap),
    )
    return layout_register_score(
        widths,
        inputs,
        (
            RegisterStep("Allocate data", outputs=("data",)),
            *(RegisterStep(name, ports, ports) for name, ports in ordered),
            RegisterStep("Free data", ("data",)),
        ),
    )

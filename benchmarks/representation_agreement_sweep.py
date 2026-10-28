"""Measure statevector/MPS/tensor-network agreement over widths and depths.

The architectural invariant is that backend selection may change execution but
not the meaning of the program or its result contract. For the approximate
representations that is a quantitative claim: a truncated MPS discards weight
and a contracted network accumulates error in an order the dense path never
takes. One circuit cannot characterize it, so this sweep drives one
parameterized workload through every available representation over a grid and
records the spread per cell.

Two properties of the measurement matter more than the grid's size.

First, the spread is reported **relative** to the dense energy as well as
absolutely. The energy of this Hamiltonian grows with width, so an absolute
spread grows with width by normalization alone; a plot of it would show a trend
that is arithmetic rather than physics.

Second, each cell records the **observed** maximum bond dimension beside the
cap it was given. A cell whose observed bond never reaches its cap was not
truncated, so its spread measures floating-point noise and says nothing about
agreement under approximation. Reading such a cell as evidence of agreement is
the specific error this sweep exists to prevent: at the defaults of
``examples/vqe_switch_sv_mps_tn.py`` (depth 2, ``max_bond`` 16) the bond reaches
only 4, so the spread it reports is complex64 epsilon.

Examples
--------
Measure the checked-in grid and seal it. The payload is a single-device result,
so it belongs under ``benchmarks/results/local/``::

    python benchmarks/representation_agreement_sweep.py \
        --output benchmarks/results/local/representation_agreement_cpu_grid.json

A fast subgrid, which is what the conformance test drives::

    python benchmarks/representation_agreement_sweep.py \
        --widths 4,6 --depths 2,6 --structures nearest_neighbour \
        --output /tmp/agreement-smoke.json

Repeat the identical seeded angles with double-precision amplitudes::

    python benchmarks/representation_agreement_sweep.py --dtype complex128 \
        --output /tmp/agreement-complex128.json

Measure state, observable and gradient conformance over several parameter
draws, which is the corpus #899 asks for::

    python benchmarks/representation_agreement_sweep.py --gradients \
        --widths 4,8 --depths 2,4 --max-bond 4,16 --seeds 260720,260721,260722 \
        --output /tmp/conformance.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

import flagquantum as fq
import flagquantum.runtime as fqr
from flagquantum.simulation.tensor_network import tensor_network_expectations
from flagquantum.simulation.tensor_network.state import TensorNetworkState

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The representations the invariant is claimed over. ``statevector`` is first
#: because it is the exact reference the other two are compared against.
MODES = ("statevector", "mps", "tensor_network")

STRUCTURES = ("nearest_neighbour", "all_to_all")

#: Widest circuit the independent oracle is run on. The parameter-shift rule
#: needs two exact executions per parameter, and this workload has
#: ``2 * depth * width`` of them, so its cost doubles with every wire. The bound
#: is a budget, not a claim: a wider cell records that the oracle was skipped
#: rather than borrowing an untested agreement.
ORACLE_MAX_WIDTH = 8

#: A cut whose two neighbouring Schmidt weights differ by less than this is
#: treated as degenerate in exact arithmetic. Where they tie, no SVD chooses the
#: retained subspace and no derivative of the truncating map exists, so a
#: gradient measured there is a property of the arithmetic rather than of the
#: backend. Weights are probabilities, so the threshold is absolute and small.
NEAR_DEGENERATE_GAP = 1e-9


def _integers(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def _modes(value: str) -> tuple[str, ...]:
    requested = tuple(item.strip() for item in value.split(","))
    unknown = [item for item in requested if item not in MODES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown representation(s) {unknown}; choose from {MODES}"
        )
    if "statevector" not in requested:
        raise argparse.ArgumentTypeError(
            "statevector is the exact reference the spread is measured against "
            "and cannot be dropped"
        )
    return requested


def _structures(value: str) -> tuple[str, ...]:
    requested = tuple(item.strip() for item in value.split(","))
    unknown = [item for item in requested if item not in STRUCTURES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown interaction structure(s) {unknown}; choose from {STRUCTURES}"
        )
    return requested


def build_ansatz(parameters: torch.Tensor, structure: str) -> fq.Circuit:
    """Build the workload at one width, depth, and interaction structure.

    The rotation layer is identical across structures so that only the
    entangling pattern differs: ``nearest_neighbour`` keeps the chain an MPS
    represents natively, while ``all_to_all`` couples every pair and is the
    regime where a chain representation and a contraction order would be
    expected to matter.
    """
    depth, n_qubits, _ = parameters.shape
    dtype = torch.complex128 if parameters.dtype == torch.float64 else torch.complex64
    circuit = fq.Circuit(n_qubits, device=parameters.device, dtype=dtype)
    for qubit in range(n_qubits):
        circuit.h(qubit)
    for layer in range(depth):
        for qubit in range(n_qubits):
            circuit.ry(qubit, parameters[layer, qubit, 0])
            circuit.rz(qubit, parameters[layer, qubit, 1])
        if structure == "nearest_neighbour":
            for qubit in range(n_qubits - 1):
                circuit.cx(qubit, qubit + 1)
        else:
            for control in range(n_qubits):
                for target in range(control + 1, n_qubits):
                    circuit.cx(control, target)
    return circuit


def exact_max_bond(n_qubits: int) -> int:
    """Return the bond dimension an exact open chain of this width needs.

    The widest cut of an ``n``-qubit chain splits it in half, so an untruncated
    MPS needs ``2 ** (n // 2)`` there. A cap at or above this value cannot
    truncate anything, however close the observed bond comes to it.
    """
    return 1 << (n_qubits // 2)


def _expectation(
    target: Any, n_qubits: int, *, x: tuple[int, ...] = (), z: tuple[int, ...] = ()
) -> torch.Tensor:
    if hasattr(target, "expectation_ps"):
        return target.expectation_ps(x=x, z=z).sum()
    state_circuit = fq.Circuit(
        n_qubits, device=target.device, dtype=target.dtype, inputs=target
    )
    return state_circuit.expectation_ps(x=x, z=z).sum()


def energy(target: Any, n_qubits: int) -> torch.Tensor:
    """Open-boundary transverse-field Ising energy.

    This is the observable ``examples/vqe_switch_sv_mps_tn.py`` trains, so a
    cell of this sweep measures the same quantity the single-point result
    reports.
    """
    if isinstance(target, TensorNetworkState) and target.max_intermediate_bytes is None:
        # The same bra/ket is shared by every Ising term. Contract the complete
        # observable batch once instead of rebuilding it for each Pauli product.
        observables: list[dict[str, tuple[int, ...]]] = [
            {"z": (qubit, qubit + 1)} for qubit in range(n_qubits - 1)
        ] + [{"x": (qubit,)} for qubit in range(n_qubits)]
        values = tensor_network_expectations(
            target.plan,
            observables,
            contraction_strategy=target.contraction_strategy,
        )
        coefficients = values.new_tensor([-1.0] * (n_qubits - 1) + [-0.7] * n_qubits)
        return (values * coefficients).sum()
    value = sum(
        -_expectation(target, n_qubits, z=(qubit, qubit + 1))
        for qubit in range(n_qubits - 1)
    )
    return value + sum(
        -0.7 * _expectation(target, n_qubits, x=(qubit,)) for qubit in range(n_qubits)
    )


#: Observables a state conformance check reports beside the Ising energy. One
#: expectation value is a projection, so a state comparison that used only the
#: energy could not separate two states that happen to share it.
CONFORMANCE_OBSERVABLES: dict[str, dict[str, tuple[int, ...]]] = {
    "z:0": {"z": (0,)},
    "x:1": {"x": (1,)},
    "zz:0,1": {"z": (0, 1)},
}


def _measure_observable(
    target: Any, n_qubits: int, observable: dict[str, tuple[int, ...]]
) -> float | None:
    """Return one Pauli-product expectation, or None when the width excludes it."""

    if any(qubit >= n_qubits for qubits in observable.values() for qubit in qubits):
        return None
    return float(_expectation(target, n_qubits, **observable))


def dense_state(target: Any) -> torch.Tensor:
    """Return a representation's state as a dense ``(batch, amplitude)`` tensor.

    The dense path hands back a tensor and the contracted paths hand back state
    objects, so one accessor keeps a conformance comparison from silently
    skipping the mode whose type it did not expect. Materializing here is the
    point: this is a conformance check against the exact path at widths where a
    dense state fits.
    """

    state = target if isinstance(target, torch.Tensor) else target.state()
    return state.reshape(-1, state.shape[-1])


def state_conformance(target: Any, reference: Any) -> dict[str, float]:
    """Compare two states by normalized overlap, with norm error kept apart.

    Fidelity is a *normalized* overlap, so a pure rescaling of a correct state
    reports fidelity 1 and a separate norm drift of 1 rather than one blended
    number. That separation is what lets a reader tell a norm bug from a state
    that genuinely differs.
    """

    psi = dense_state(target).to(torch.complex128)
    exact = dense_state(reference).to(torch.complex128)
    if psi.shape != exact.shape:
        raise ValueError(
            f"states have incompatible shapes {tuple(psi.shape)} and "
            f"{tuple(exact.shape)}"
        )
    overlap = (exact.conj() * psi).sum(dim=-1).abs().square()
    denominator = exact.abs().square().sum(dim=-1) * psi.abs().square().sum(dim=-1)
    fidelity = torch.where(
        denominator > 0, overlap / denominator, torch.zeros_like(overlap)
    )
    norms = (
        psi.abs().square().sum(dim=-1).sqrt() / exact.abs().square().sum(dim=-1).sqrt()
    )
    norm_drift = (norms - 1).abs()
    return {
        "fidelity_vs_statevector": float(fidelity.clamp(max=1.0).mean()),
        "norm_drift": float(norm_drift.mean()),
    }


def gradient_conformance(
    gradient: torch.Tensor, reference: torch.Tensor
) -> dict[str, float | None]:
    """Compare a gradient to a reference in scale and in direction.

    An optimizer follows direction, so the two are reported apart: cosine
    similarity isolates direction, and the relative L2 error is what a rescaled
    but correctly directed gradient shows up in.
    """

    measured = gradient.detach().to(torch.float64).reshape(-1)
    exact = reference.detach().to(torch.float64).reshape(-1)
    if measured.shape != exact.shape:
        raise ValueError(
            f"gradients have incompatible shapes {tuple(measured.shape)} and "
            f"{tuple(exact.shape)}"
        )
    scale = float(exact.norm())
    difference = float((measured - exact).norm())
    cosine = (
        None
        if scale == 0.0 or float(measured.norm()) == 0.0
        else float(torch.dot(measured, exact) / (measured.norm() * exact.norm()))
    )
    return {
        "relative_l2_error": None if scale == 0.0 else difference / scale,
        "cosine_similarity": cosine,
    }


def _energy_gradient(
    parameters: torch.Tensor,
    *,
    structure: str,
    n_qubits: int,
    mode: str,
    max_bond: int | None = None,
) -> torch.Tensor:
    """Return ``d energy / d parameters`` through one representation."""

    leaf = parameters.detach().clone().requires_grad_(True)
    options: dict[str, object] = (
        {"max_bond": max_bond} if mode == "mps" and max_bond is not None else {}
    )
    if options:
        # This sweep measures the retained-subspace rule itself: whether autograd
        # through a truncated split reproduces the derivative of that split. The
        # default refuses it precisely because the rule is not the derivative of
        # the forward map, so measuring the difference requires asking for it.
        options["truncated_gradient"] = "projected_stop_subspace"
    target = fqr.run_native(build_ansatz(leaf, structure), mode=mode, **options)
    value = energy(target, n_qubits)
    (gradient,) = torch.autograd.grad(value, leaf)
    return gradient


def ideal_gradient(
    parameters: torch.Tensor, *, structure: str, n_qubits: int
) -> torch.Tensor:
    """Return the gradient the exact statevector path reports."""

    return _energy_gradient(
        parameters, structure=structure, n_qubits=n_qubits, mode="statevector"
    )


def truncated_gradient(
    parameters: torch.Tensor, *, structure: str, n_qubits: int, max_bond: int
) -> torch.Tensor:
    """Return the gradient autograd reports for the *truncated* MPS computation."""

    return _energy_gradient(
        parameters,
        structure=structure,
        n_qubits=n_qubits,
        mode="mps",
        max_bond=max_bond,
    )


def parameter_shift_gradient(
    parameters: torch.Tensor, *, structure: str, n_qubits: int
) -> torch.Tensor:
    """Return the exact energy gradient from the parameter-shift rule.

    The rule is exact for a parameter that enters through one rotation whose
    generator has eigenvalues ``±1/2``, which is every gate this workload is
    built from, so it is an independent check on the exact path rather than a
    restatement of its autograd result. It differentiates the *exact* map; the
    truncated map's derivative is a different quantity and this oracle says
    nothing about it.
    """

    flat = parameters.detach().reshape(-1)
    gradient = torch.zeros(flat.numel(), dtype=torch.float64)
    for index in range(flat.numel()):
        shifted_energies: list[float] = []
        for offset in (math.pi / 2, -math.pi / 2):
            shifted = flat.clone()
            shifted[index] += offset
            with torch.no_grad():
                state = fqr.run_native(
                    build_ansatz(shifted.reshape(parameters.shape), structure),
                    mode="statevector",
                )
                shifted_energies.append(float(energy(state, n_qubits)))
        gradient[index] = 0.5 * (shifted_energies[0] - shifted_energies[1])
    return gradient


def schmidt_gap(state: Any, rank: int) -> dict[str, Any]:
    """Return the smallest exact Schmidt-weight gap at any cut a cap truncates.

    A truncated SVD keeps the ``rank`` largest weights. When the last kept and
    first discarded weight are equal, that choice is arbitrary, and a gradient
    through it depends on which side of the tie the arithmetic lands. Recording
    the gap turns that into a stated boundary of the measurement instead of an
    unexplained jump.

    The spectrum is read from the exact dense state, not from the MPS whose
    truncation is in question, so the gap is a property of the circuit rather
    than of the representation that discarded the weight.
    """

    psi = torch.as_tensor(state).detach().to(torch.complex128)
    if psi.dim() == 2:
        if psi.shape[0] != 1:
            raise ValueError(
                f"expected one state, got batch size {psi.shape[0]}; the gap is "
                "reported per state"
            )
        psi = psi[0]
    psi = psi.reshape(-1)
    norm = psi.abs().square().sum().sqrt()
    if float(norm) == 0.0:
        raise ValueError("cannot report a Schmidt gap of the zero state")
    psi = psi / norm
    n_qubits = int(round(math.log2(psi.numel())))
    gaps: list[float] = []
    for cut in range(1, n_qubits):
        matrix = psi.reshape(2**cut, 2 ** (n_qubits - cut))
        weights = torch.linalg.svdvals(matrix).square()
        if weights.numel() <= rank:
            # This cut carries no more weight than the cap keeps, so the cap
            # makes no decision here and there is no gap to report.
            continue
        gaps.append(float(weights[rank - 1] - weights[rank]))
    if not gaps:
        return {"rank": rank, "min_gap": None, "near_degenerate": None}
    min_gap = min(gaps)
    return {
        "rank": rank,
        "min_gap": min_gap,
        "near_degenerate": min_gap < NEAR_DEGENERATE_GAP,
    }


def oracle_section(
    parameters: torch.Tensor, *, structure: str, n_qubits: int
) -> dict[str, Any]:
    """Check the exact path's autograd gradient against the parameter-shift rule.

    Returns a record that says either how closely the two agree or why the
    oracle was not run. A skipped oracle is named rather than omitted, because an
    absent key reads as an agreement that was never measured.
    """

    if n_qubits > ORACLE_MAX_WIDTH:
        return {
            "method": "parameter_shift",
            "applies_to": "statevector",
            "skipped": (
                f"width {n_qubits} is above the oracle's bounded domain "
                f"({ORACLE_MAX_WIDTH})"
            ),
        }
    oracle = parameter_shift_gradient(
        parameters, structure=structure, n_qubits=n_qubits
    )
    exact = ideal_gradient(parameters, structure=structure, n_qubits=n_qubits)
    conformance = gradient_conformance(oracle, exact)
    return {
        "method": "parameter_shift",
        "applies_to": "statevector",
        "valid_domain": (
            "exact statevector map only: the rule is exact for the rotation "
            "layers, and it does not apply to a map whose bond truncation "
            "discarded weight"
        ),
        "compares": ["statevector"],
        **conformance,
    }


def truncated_derivative_error(
    parameters: torch.Tensor,
    *,
    structure: str,
    n_qubits: int,
    max_bond: int,
    step: float = 1e-5,
    indices: Sequence[int] | None = None,
    checks: int = 8,
) -> dict[str, Any]:
    """Compare autograd's truncated gradient with finite differences of the same map.

    This answers a different question from "does the gradient match the ideal
    circuit": it asks whether the derivative of the computation that actually
    ran is the derivative of the computation that actually ran. Central finite
    differences of that same truncated map settle it without consulting any
    other representation, so the answer does not presuppose that the ideal
    gradient is reachable.

    ``max_relative_error`` divides each component's absolute error by the norm of
    the autograd gradient, so a near-zero component cannot inflate it.
    """

    flat = parameters.detach().reshape(-1)
    selected = (
        tuple(indices)
        if indices is not None
        else tuple(range(0, flat.numel(), max(flat.numel() // checks, 1)))
    )
    approximate = truncated_gradient(
        parameters, structure=structure, n_qubits=n_qubits, max_bond=max_bond
    ).reshape(-1)
    scale = float(approximate.norm())
    errors: list[float] = []
    for index in selected:
        plus = flat.clone()
        minus = flat.clone()
        plus[index] += step
        minus[index] -= step
        with torch.no_grad():
            forward = float(
                energy(
                    fqr.run_native(
                        build_ansatz(plus.reshape(parameters.shape), structure),
                        mode="mps",
                        max_bond=max_bond,
                    ),
                    n_qubits,
                )
            )
            backward = float(
                energy(
                    fqr.run_native(
                        build_ansatz(minus.reshape(parameters.shape), structure),
                        mode="mps",
                        max_bond=max_bond,
                    ),
                    n_qubits,
                )
            )
        differences = (forward - backward) / (2 * step)
        errors.append(abs(differences - float(approximate[index])))
    return {
        "method": "central_finite_difference",
        "step": step,
        "indices": list(selected),
        "checked": len(selected),
        "max_abs_error": max(errors) if errors else 0.0,
        "max_relative_error": (max(errors) / scale) if errors and scale else 0.0,
    }


def smoothness_probe(
    parameters: torch.Tensor,
    *,
    structure: str,
    n_qubits: int,
    max_bond: int,
    index: int = 2,
    half_range: float = 0.05,
    points: int = 7,
) -> dict[str, Any]:
    """Measure whether the truncated energy is locally smooth in one parameter.

    A finite-difference comparison is only meaningful where the map is
    differentiable. Slicing a bond decides which subspace is kept, and that
    decision can flip inside a small interval, so the same measurement that says
    "autograd disagrees with the difference" has to say whether a derivative
    exists at all. ``largest_step_jump`` is the biggest change in energy between
    consecutive scan points; a smooth cell moves by well under 1e-3 per point
    here, while a cell whose truncation decision flips jumps by order 0.1.
    """

    flat = parameters.detach().reshape(-1)
    base = float(flat[index])
    energies: list[float] = []
    bonds: list[int] = []
    for offset in torch.linspace(-half_range, half_range, points).tolist():
        shifted = flat.clone()
        shifted[index] = base + offset
        circuit = build_ansatz(shifted.reshape(parameters.shape), structure)
        with torch.no_grad():
            state = fqr.run_native(circuit, mode="mps", max_bond=max_bond)
            energies.append(float(energy(state, n_qubits)))
            bonds.append(int(max(state.bond_dims)))
    jumps = [
        abs(later - earlier)
        for earlier, later in zip(energies, energies[1:], strict=False)
    ]
    return {
        "index": index,
        "half_range": half_range,
        "points": points,
        "largest_step_jump": max(jumps),
        "bond_changed": len(set(bonds)) > 1,
        "smooth": max(jumps) < 1e-2,
        "energies": [round(value, 12) for value in energies],
    }


def gradient_section(
    parameters: torch.Tensor,
    *,
    structure: str,
    n_qubits: int,
    max_bond: int,
    modes: tuple[str, ...],
    smoothness: bool = True,
) -> dict[str, Any]:
    """Measure state, observable and gradient conformance for one cell.

    A mode that cannot produce a state or a derivative is recorded as
    unsupported rather than dropped: substituting another representation for it
    would report agreement that was never measured.
    """

    reference_target = fqr.run_native(
        build_ansatz(parameters, structure), mode="statevector"
    )
    reference_state = dense_state(reference_target)
    reference_gradient = ideal_gradient(
        parameters, structure=structure, n_qubits=n_qubits
    )
    states: dict[str, dict[str, float]] = {}
    observables: dict[str, dict[str, float | None]] = {}
    per_mode: dict[str, dict[str, float | None]] = {}
    unsupported: dict[str, str] = {}
    for mode in modes:
        try:
            options = {"max_bond": max_bond} if mode == "mps" else {}
            target = fqr.run_native(
                build_ansatz(parameters, structure), mode=mode, **options
            )
            states[mode] = state_conformance(target, reference_state)
            observables[mode] = {
                label: _measure_observable(target, n_qubits, observable)
                for label, observable in CONFORMANCE_OBSERVABLES.items()
            }
            gradient = _energy_gradient(
                parameters,
                structure=structure,
                n_qubits=n_qubits,
                mode=mode,
                max_bond=max_bond,
            )
            per_mode[mode] = gradient_conformance(gradient, reference_gradient)
        except Exception as exc:  # a refusal is recorded, never substituted
            unsupported[mode] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    section: dict[str, Any] = {
        "reference": "statevector",
        "states": states,
        "observables": observables,
        "per_mode": per_mode,
        "unsupported": unsupported,
        "oracle": oracle_section(parameters, structure=structure, n_qubits=n_qubits),
        "schmidt": schmidt_gap(reference_state, max_bond),
    }
    if "mps" in per_mode:
        # A central difference of a float32 energy divides two nearly equal
        # values by 2e-5, which spends about six of the digits the difference
        # carried. At complex64 that noise, not the backward pass, is what the
        # check reports, so the cell says which precision the number is read at.
        double_precision = parameters.dtype == torch.float64
        section["mps_own_map"] = {
            "derivative": truncated_derivative_error(
                parameters,
                structure=structure,
                n_qubits=n_qubits,
                max_bond=max_bond,
            ),
            "finite_difference_domain": (
                "valid: double-precision energies and a 1e-5 step"
                if double_precision
                else (
                    "not a check on autograd below double precision: the "
                    "difference of two float32 energies with a 1e-5 step loses "
                    "about six digits, so this number is the probe's arithmetic "
                    "rather than a statement about the backward pass"
                )
            ),
            "smoothness": (
                smoothness_probe(
                    parameters,
                    structure=structure,
                    n_qubits=n_qubits,
                    max_bond=max_bond,
                )
                if smoothness
                else None
            ),
        }
    return section


def measure_cell(
    *,
    n_qubits: int,
    depth: int,
    structure: str,
    max_bond: int,
    seed: int,
    device: str,
    modes: tuple[str, ...] = MODES,
    dtype: torch.dtype = torch.complex64,
    gradients: bool = False,
) -> dict[str, Any]:
    """Measure one grid cell: the same parameters through every representation.

    The parameters are drawn once per cell from a seeded generator and shared by
    every run, so a disagreement is a property of the representation rather than
    of a different starting point.

    ``gradients=True`` adds the state, observable and parameter-gradient sections
    of #899. It is off by default so the published precision artifacts keep the
    shape they were measured in.
    """
    if dtype not in (torch.complex64, torch.complex128):
        raise ValueError("dtype must be torch.complex64 or torch.complex128")
    generator = torch.Generator(device=device).manual_seed(seed)
    # Draw once in float32, including the scale, then widen exactly. Both
    # precision runs therefore execute identical angles rather than different
    # random streams or differently rounded input programs.
    parameters = (
        0.05
        * torch.randn(
            depth, n_qubits, 2, device=device, generator=generator, dtype=torch.float32
        )
    ).to(dtype=torch.float64 if dtype == torch.complex128 else torch.float32)
    failures: dict[str, str] = {}

    # Whether the cap actually discarded weight. An observed bond equal to the
    # cap does not settle it: a shallow circuit stops growing on its own, and at
    # depth 2 and width 8 the bond is 4 whether the cap is 4, 16, or 256. So the
    # same circuit runs once more with one doubling of headroom -- a state that
    # would have grown past the cap exceeds it here, and one that would not
    # reports the same bond. Probing at the exact bond would answer the same
    # question at exponential cost, since an exact chain of 20 qubits needs 1024.
    headroom_cap = min(exact_max_bond(n_qubits), 2 * max_bond)
    headroom_bond: int | None = None
    if "mps" in modes:
        try:
            with torch.no_grad():
                headroom_state = fqr.run_native(
                    build_ansatz(parameters, structure),
                    mode="mps",
                    max_bond=headroom_cap,
                )
                headroom_bond = int(max(headroom_state.bond_dims))
        except Exception as exc:
            headroom_bond = None
            failures["headroom_probe"] = (
                f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
            )

    energies: dict[str, float | None] = {}
    observed_bond: int | None = None
    bond_dims: list[int] | None = None
    discarded_weight: float | None = None
    max_step_error: float | None = None
    truncation_steps: int | None = None
    for mode in modes:
        options = {"max_bond": max_bond} if mode == "mps" else {}
        try:
            with torch.no_grad():
                target = fqr.run_native(
                    build_ansatz(parameters, structure), mode=mode, **options
                )
                if mode == "mps":
                    observed_bond = int(max(target.bond_dims))
                    bond_dims = [int(item) for item in target.bond_dims]
                    report = target.summary()
                    # The weight the cap discarded, as the state recorded it, so
                    # "truncated" is not inferred from bonds alone. A bond that
                    # reaches the cap without this being positive was held
                    # exactly.
                    discarded_weight = float(report["truncation_error"])
                    max_step_error = float(report["max_truncation_error"])
                    truncation_steps = int(report["truncation_steps"])
                energies[mode] = float(energy(target, n_qubits))
        except Exception as exc:
            energies[mode] = None
            failures[mode] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"

    measured = [value for value in energies.values() if value is not None]
    reference = energies["statevector"]
    digits = 17 if dtype == torch.complex128 else 9
    # Preserve double precision in the record; nine digits suffice only for
    # complex64. Compute both spreads from the unrounded measurements.
    energies = {
        mode: None if value is None else float(f"{value:.{digits}g}")
        for mode, value in energies.items()
    }
    # The dense path is the exact reference the relative spread is divided by,
    # named explicitly rather than taken positionally.
    spread = max(measured) - min(measured) if len(measured) > 1 else None
    relative = (
        spread / abs(reference)
        if spread is not None and reference not in (None, 0.0)
        else None
    )
    spread = None if spread is None else float(f"{spread:.6g}")
    relative = None if relative is None else float(f"{relative:.6g}")
    cell: dict[str, Any] = {
        "n_qubits": n_qubits,
        "depth": depth,
        "structure": structure,
        "max_bond_cutoff": max_bond,
        "seed": seed,
        "observed_max_bond": observed_bond,
        # Every bond the run observed, not only the widest: a cap that bites at
        # one cut and not another is not described by its maximum.
        "mps_bond_dims": bond_dims,
        # What the cap discarded, per the state's own record. Zero means the
        # state was represented exactly at this cap.
        "mps_discarded_weight": discarded_weight,
        "mps_max_truncation_error": max_step_error,
        "mps_truncation_steps": truncation_steps,
        # The cap the headroom probe used, and the bond it reached. The cutoff
        # is `min(exact_max_bond(n_qubits), 2 * max_bond)` and the exact bond is
        # `2 ** (n_qubits // 2)`, so both are recomputable from the three fields
        # above; only the measured bond is recorded.
        "headroom_max_bond": headroom_bond,
        # The cap discarded weight only if the state grew past it once given
        # room. A cell where it did not was represented exactly at this cap, so
        # its spread is complex64 noise rather than evidence about agreement
        # under approximation.
        "mps_truncated": (
            None
            if observed_bond is None or headroom_bond is None
            else headroom_bond > max_bond
        ),
        "energies": energies,
        "spread_absolute": spread,
        "spread_relative": relative,
    }
    if gradients:
        section = gradient_section(
            parameters,
            structure=structure,
            n_qubits=n_qubits,
            max_bond=max_bond,
            modes=modes,
        )
        cell["states"] = section["states"]
        cell["observables"] = section["observables"]
        cell["gradients"] = {
            "reference": section["reference"],
            "per_mode": section["per_mode"],
            "unsupported": section["unsupported"],
            "oracle": section["oracle"],
        }
        cell["schmidt"] = section["schmidt"]
        if "mps_own_map" in section:
            cell["gradients"]["mps_own_map"] = section["mps_own_map"]
    # A mode that raised is named with its error; a cell that measured every
    # requested mode carries no `failures` key at all. Recording an empty one in
    # every cell spends the benchmark-results ratchet budget on the absence of
    # news, and `summarize` reads the key's presence rather than its emptiness.
    if failures:
        cell["failures"] = failures
    return cell


def package_record(path: Path, version: str | None) -> dict[str, Any]:
    """Describe where the imported package lives, relative to this checkout.

    A package inside this tree is named by its repository-relative path: the
    absolute one would carry the producing host's home directory into a public
    artifact, and the record only has to say which tree ran.
    """

    resolved = path.resolve()
    root = REPO_ROOT.resolve()
    inside = root in resolved.parents
    return {
        "path": str(resolved.relative_to(root)) if inside else str(resolved),
        "version": version,
        "from_repository": inside,
    }


def require_repository_package(module: Any = fq) -> dict[str, Any]:
    """Refuse to measure unless the imported ``flagquantum`` is this tree.

    ``source_commit`` is read from this repository's ``git`` state, not from the
    interpreter, so a run that imported an installed package would seal a
    measurement of one codebase under another's revision. That happened once
    while this corpus was being built, and nothing in the payload said so: every
    cell recorded a refused derivative, which reads as a property of the
    representation rather than as the wrong program being executed.

    Failing closed keeps the payload's own revision claim answerable.
    """

    location = getattr(module, "__file__", None)
    if location is None:
        raise RuntimeError(
            "the imported flagquantum has no source location, so this payload "
            "cannot claim this repository's revision"
        )
    record = package_record(Path(location), getattr(module, "__version__", None))
    if not record["from_repository"]:
        raise RuntimeError(
            f"the imported flagquantum is {record['path']}, which is not this "
            f"repository ({REPO_ROOT.resolve()}); a payload that names this "
            "repository's revision would then describe another tree -- run with "
            "this checkout on sys.path, for example PYTHONPATH=<checkout>"
        )
    return record


def tolerances_for(dtype: torch.dtype) -> dict[str, float | None]:
    """Return the bounds this precision's conformance claims are read against.

    The two precisions cannot share a bound. At complex64 the same arithmetic
    in two representations already separates by about ``4e-7`` through float32
    accumulation, so a double-precision bound would fail honest evidence, and a
    single-precision bound applied to doubles would pass a broken run.

    A truncated cell has no asserted bound: the distance from the exact gradient
    is the quantity being measured, and a threshold on it would turn a
    measurement of approximation into a pass/fail on an expected size of it.
    """

    # A truncated split is measured only through the retained-subspace rule,
    # whose agreement with the derivative of its own forward map is the defect
    # this harness reports; no bound is asserted on either quantity.
    truncated_gradient_relative = None
    if dtype == torch.complex128:
        return {
            "oracle_relative": 1e-9,
            "untruncated_state_fidelity_deficit": 1e-10,
            "untruncated_gradient_relative": 1e-10,
            "truncated_gradient_relative": truncated_gradient_relative,
        }
    if dtype == torch.complex64:
        return {
            "oracle_relative": 1e-5,
            "untruncated_state_fidelity_deficit": 1e-5,
            "untruncated_gradient_relative": 1e-4,
            "truncated_gradient_relative": truncated_gradient_relative,
        }
    raise ValueError(f"no conformance tolerance for {dtype}")


def measure_grid(
    *,
    widths: Sequence[int],
    depths: Sequence[int],
    structures: Sequence[str],
    max_bonds: Sequence[int],
    seeds: Sequence[int],
    all_to_all_max_width: int = 8,
    modes: tuple[str, ...] = MODES,
    device: str = "cpu",
    dtype: torch.dtype = torch.complex64,
    gradients: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Measure every configuration at every seed, returning cells and skips.

    The seed varies inside a configuration rather than outside it, so a reader
    comparing two rows of a payload is comparing parameter draws of the same
    program. A configuration that is too expensive to measure is returned in the
    skip list rather than dropped, so a payload can say what it does not cover.
    """

    cells: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for structure in structures:
        for n_qubits in widths:
            if (
                structure == "all_to_all"
                and n_qubits > all_to_all_max_width
                and "tensor_network" in modes
            ):
                # Recorded rather than dropped: a reader has to be able to see
                # which part of the grid was not measured and why.
                skipped.append(
                    {
                        "structure": structure,
                        "n_qubits": n_qubits,
                        "reason": (
                            "all-to-all tensor-network contraction exceeds the "
                            "CPU budget above width "
                            f"{all_to_all_max_width}"
                        ),
                    }
                )
                continue
            for depth in depths:
                for max_bond in max_bonds:
                    for seed in seeds:
                        cell = measure_cell(
                            n_qubits=n_qubits,
                            depth=depth,
                            structure=structure,
                            max_bond=max_bond,
                            seed=seed,
                            device=device,
                            modes=modes,
                            dtype=dtype,
                            gradients=gradients,
                        )
                        cells.append(cell)
                        print(
                            json.dumps(
                                {
                                    key: cell[key]
                                    for key in (
                                        "structure",
                                        "n_qubits",
                                        "depth",
                                        "max_bond_cutoff",
                                        "seed",
                                        "observed_max_bond",
                                        "mps_discarded_weight",
                                        "mps_truncated",
                                        "spread_relative",
                                    )
                                }
                            ),
                            flush=True,
                        )
    return cells, skipped


def summarize(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Report the bound separately over truncated and untruncated cells.

    Mixing the two would let the untruncated cells, whose spread sits at
    complex64 epsilon, dominate a bound that is supposed to describe behavior
    under approximation.
    """
    truncated = [
        cell
        for cell in cells
        if cell["mps_truncated"] and cell["spread_relative"] is not None
    ]
    untruncated = [
        cell
        for cell in cells
        if cell["mps_truncated"] is False and cell["spread_relative"] is not None
    ]

    def bound(group: list[dict[str, Any]]) -> dict[str, Any]:
        if not group:
            return {"cells": 0}
        worst = max(group, key=lambda cell: cell["spread_relative"])
        return {
            "cells": len(group),
            # Already rounded on the cell; carried through unchanged so the
            # summary and the cell it names cannot disagree.
            "max_spread_relative": worst["spread_relative"],
            "max_spread_absolute": worst["spread_absolute"],
            "worst_cell": {
                key: worst[key]
                for key in ("n_qubits", "depth", "structure", "max_bond_cutoff")
            },
        }

    incomplete = [
        {key: cell[key] for key in ("n_qubits", "depth", "structure", "failures")}
        for cell in cells
        if "failures" in cell
    ]
    return {
        "cells_total": len(cells),
        "truncated": bound(truncated),
        "untruncated": bound(untruncated),
        "incomplete_cells": incomplete,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # The default grid is bounded by the tensor-network path, not by the
    # physics. That path re-contracts the whole network once per Pauli term and
    # an energy needs 2n-1 of them, so its cost grows with depth and, on the
    # all-to-all structure, with width squared: one term measured 0.6s, 3.3s and
    # 10.7s at depths 2, 4 and 6 on the chain at width 6, and a single term at
    # width 12 all-to-all depth 4 did not finish inside two minutes.
    # Statevector and MPS finish every one of these cells in under a second.
    #
    # So the chain is swept to width 20 while all-to-all stops at 8, and the
    # deeper and wider all-to-all cells are left to a run that can afford them
    # -- `--structures all_to_all --widths 12,16 --modes statevector,mps` is
    # affordable today and says nothing about the tensor-network path, which is
    # the honest trade rather than a silent omission.
    parser.add_argument("--widths", type=_integers, default=(4, 8, 12, 16, 20))
    parser.add_argument("--depths", type=_integers, default=(2, 4))
    parser.add_argument(
        "--all-to-all-max-width",
        type=int,
        default=8,
        help=(
            "widest all-to-all cell to attempt; its tensor-network contraction "
            "grows with width squared and is the sweep's cost ceiling"
        ),
    )
    parser.add_argument("--structures", type=_structures, default=STRUCTURES)
    parser.add_argument("--max-bond", type=_integers, default=(4, 16))
    parser.add_argument(
        "--modes",
        type=_modes,
        default=MODES,
        help=(
            "representations to compare; dropping tensor_network makes a wide "
            "grid affordable, at the cost of saying nothing about that path"
        ),
    )
    parser.add_argument(
        "--seeds",
        type=_integers,
        default=(260720,),
        help=(
            "parameter draws per configuration; one draw per cell is a single "
            "sample of a random initialization, so a grid that states a bound "
            "over seeds varies them"
        ),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--dtype", choices=("complex64", "complex128"), default="complex64"
    )
    parser.add_argument(
        "--gradients",
        action="store_true",
        help=(
            "also record per-representation state fidelity, extra observables and "
            "parameter gradients beside the energy spread"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    started = time.perf_counter()
    package = require_repository_package()
    cells, skipped = measure_grid(
        widths=arguments.widths,
        depths=arguments.depths,
        structures=arguments.structures,
        max_bonds=arguments.max_bond,
        seeds=arguments.seeds,
        all_to_all_max_width=arguments.all_to_all_max_width,
        modes=arguments.modes,
        device=arguments.device,
        dtype=getattr(torch, arguments.dtype),
        gradients=arguments.gradients,
    )

    payload = {
        "schema": "flagquantum.representation_agreement_sweep.v1",
        # The benchmark's identity, which the contract in
        # tests/benchmark_contract/test_benchmark_contract.py requires of every
        # unsealed payload beside its scalability blockers.
        "benchmark": "representation_agreement_sweep",
        "artifact_class": "measured_local_run",
        "claim": (
            "backend selection changes execution but not the result contract; "
            "this payload states the regime that was measured and the bound the "
            "spread held to inside it"
        ),
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        # The layout contract every checked-in non-release payload answers to,
        # enforced by tests/test_distributed_scalability_audit.py: a payload
        # under results/local must declare its evidence class and name why it
        # cannot carry a scalability claim, so a reader cannot mistake a
        # single-device measurement for a sharded one.
        "non_release_evidence": True,
        "benchmark_evidence_class": "local_non_release",
        "scalability_blockers": [
            "single-process CPU agreement measurement across representations; "
            "it shards no workload and is not distributed scalability evidence"
        ],
        "modes": list(arguments.modes),
        "gradients_measured": bool(arguments.gradients),
        # Which installed package produced these numbers. `source_commit` comes
        # from git, so this is the field that ties the measurement to it.
        "imported_package": package,
        # The bounds a reader has to check this payload's quantities against, so
        # the artifact carries its own tolerances rather than requiring the
        # README beside it.
        "tolerances": tolerances_for(getattr(torch, arguments.dtype)),
        "oracle_max_width": ORACLE_MAX_WIDTH,
        "near_degenerate_gap": NEAR_DEGENERATE_GAP,
        "grid": {
            "widths": list(arguments.widths),
            "depths": list(arguments.depths),
            "structures": list(arguments.structures),
            "max_bond_cutoffs": list(arguments.max_bond),
        },
        "observable": "open-boundary transverse-field Ising, h=0.7",
        "parameter_initialization": (
            "0.05 * float32 randn, one seeded draw shared per cell; "
            "complex128 widens the same float32 angles exactly"
        ),
        "seed": arguments.seeds[0],
        # Every draw the grid used. A bound stated over one initialization is a
        # sample, so the corpus names all the draws it actually ran.
        "seeds": list(arguments.seeds),
        "device": arguments.device,
        "dtype": arguments.dtype,
        "summary": summarize(cells),
        "skipped_configurations": skipped,
        "cells": cells,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "command": " ".join(sys.argv),
        "python": platform.python_version(),
        "torch": torch.__version__,
        # The published revision this measurement ran against. Recording HEAD
        # instead would name the working commit, which is rewritten by a rebase
        # or an amend before the payload is sealed; the evidence-revision check
        # requires a revision a ref of this repository reaches, and an orphaned
        # hash is exactly what it rejects.
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "origin/main"], cwd=REPO_ROOT, text=True, timeout=15
        ).strip(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, timeout=15
        ).strip(),
        # Whether the measuring code itself was committed when it ran. A whole
        # working tree cannot be reported clean here -- writing this payload
        # dirties the tree it would describe -- but the script that produced the
        # numbers can be, and that is the part a reader needs to re-run.
        "measuring_source_committed": not subprocess.check_output(
            ["git", "status", "--porcelain", "--", str(Path(__file__).resolve())],
            cwd=REPO_ROOT,
            text=True,
            timeout=15,
        ).strip(),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(arguments.output), **payload["summary"]}, indent=2))
    return 2 if payload["summary"]["incomplete_cells"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

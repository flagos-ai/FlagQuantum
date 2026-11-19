"""Compare the source finite-width HHL and SWAP test with a direct solve."""

from dataclasses import dataclass

import torch

from flagquantum import synth
from flagquantum.compiler import ResourceEstimate

from .model import COMPLEX, Model
from .quantum import SwapReadout, build_native, logical_state


@dataclass(frozen=True)
class Solution:
    """Actual coherent readout of all 32 words and the eight population steps."""

    program: synth.Program
    resources: ResourceEstimate
    logical: torch.Tensor
    statevector: torch.Tensor
    reference: torch.Tensor
    fidelity: float
    probability: float


def solve(model: Model) -> Solution:
    """Run native HHL and read the indicator-one, phase-zero branch."""
    program = build_native(model.evolution, model.angles, model.padded)
    logical = logical_state(program)
    branch = logical[1, :, 0]
    unnormalized = branch / model.cutoff
    reference = torch.linalg.solve(model.hermitian, model.padded).to(COMPLEX)
    fidelity = float(
        torch.vdot(reference / reference.norm(), branch / branch.norm()).abs().square()
    )
    resources = program.report["resources"]
    if not isinstance(resources, ResourceEstimate):
        raise TypeError("Expected the native HHL resource estimate")
    return Solution(
        program,
        resources,
        logical,
        unnormalized,
        reference,
        fidelity,
        float(branch.abs().square().sum()),
    )


def show_result(solution: Solution, swap: SwapReadout) -> None:
    """Report both readouts with their distinct postselection conditions."""
    print("Native HHL physical resources:", solution.resources)
    print("SWAP test counts (test by indicator):", swap.counts.tolist())
    print("Actual SWAP test shots:", int(swap.counts.sum()))
    print("Sampled indicator-conditioned overlap:", swap.root_overlap)
    print("Indicator-one/phase-zero probability:", solution.probability)
    print("Statevector solution fidelity:", solution.fidelity)
    print("Signed HHL real solution:", solution.statevector.real.tolist())
    print("Direct Hermitian solve:", solution.reference.real.tolist())

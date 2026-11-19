"""Execute the two original-sized periodic Laplacian block encodings."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

import flagquantum as fq
from examples.synthesis.community.paper_implementation_project.block_encoding_nd_laplacian.periodic_laplacian_1d.comparisons.circuit import (
    reference_circuit as reference_1d,
)
from examples.synthesis.community.paper_implementation_project.block_encoding_nd_laplacian.periodic_laplacian_1d.main import (
    main as main_1d,
)
from examples.synthesis.community.paper_implementation_project.block_encoding_nd_laplacian.periodic_laplacian_nd.comparisons.circuit import (
    reference_circuit as reference_nd,
)
from examples.synthesis.community.paper_implementation_project.block_encoding_nd_laplacian.periodic_laplacian_nd.main import (
    main as main_nd,
)
from flagquantum import synth
from flagquantum.compiler import ResourceEstimate

from .model import laplacian_1d, laplacian_nd


@dataclass(frozen=True)
class Case:
    """A physical execution and its unnormalized projected block, with phase intact."""

    program: synth.Program
    resources: ResourceEstimate
    projected: npt.NDArray[np.complex128]
    expected: npt.NDArray[np.complex128]
    full_state_error: float
    block_error: float
    projection_probability: float
    fidelity: float

    @property
    def source_display(self) -> npt.NDArray[np.float64]:
        """Reproduce the source's real, first-amplitude-phase-aligned display only."""
        phase = np.angle(self.projected[0])
        return np.asarray((self.projected / np.exp(1j * phase)).real, dtype=np.float64)


def run_case(*, multidimensional: bool = False) -> Case:
    """Preserve depth optimization, exact widths seven/six, and both zero inputs."""
    width = 6 if multidimensional else 7
    program = synth.synthesize(
        main_nd if multidimensional else main_1d,
        objective="depth",
        max_qubits=width,
        max_workspace=0,
        max_operations=10000,
        memory_limit_bytes=1024**2,
    )
    resources = program.report["resources"]
    assert isinstance(resources, ResourceEstimate)
    raw = fq.run(program).statevector().flatten()
    logical = tuple(wire for wires in program.registers.values() for wire in wires)
    assert len(logical) == width and raw.numel() == 1 << width
    ordered = raw.reshape((2,) * width).permute(logical).flatten()
    expected_state = (
        fq.run(reference_nd() if multidimensional else reference_1d())
        .statevector()
        .flatten()
    )
    error = float(torch.max(torch.abs(ordered - expected_state)))
    if multidimensional:
        projected = ordered.reshape(8, 4, 2)[:, 0, 0].numpy().copy()
        matrix = laplacian_nd((2, 4))
    else:
        projected = ordered.reshape(32, 4)[:, 0].numpy().copy()
        matrix = laplacian_1d(32)
    expected = matrix[:, 0]
    block_error = float(np.max(np.abs(projected - expected)))
    projection_probability = float(np.vdot(projected, projected).real)
    fidelity = float(
        abs(np.vdot(projected, expected)) ** 2
        / (projection_probability * np.vdot(expected, expected).real)
    )
    if error >= 1e-11 or block_error >= 1e-11:
        raise ValueError("native full state or projected Laplacian block disagrees")
    return Case(
        program,
        resources,
        projected,
        expected,
        error,
        block_error,
        projection_probability,
        fidelity,
    )

"""Check both original QSVT degrees and the symmetric fit derivative."""

import json
from pathlib import Path

import pytest
import torch

from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview import (
    phases,
    workflow,
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("large,degree,data", [(False, 101, 3), (True, 255, 4)])
def test_source_inverse(large: bool, degree: int, data: int) -> None:
    result = workflow.solve(large)
    assert result.request_degree == degree and result.data_width == data
    assert result.phases.phases.numel() == degree + 1
    assert torch.equal(result.phases.phases, result.phases.phases.flip((0,)))
    assert result.phases.iterations <= 12
    assert result.phases.residual < 1e-10
    assert result.spectral_error < 1e-10
    assert result.full_state_error < 1e-10
    assert result.solution_l2 < 0.1
    assert result.resources.n_qubits == (12 if large else 13)
    assert result.workspace_probability < 1e-24
    assert result.resources.n_operations == (87012 if large else 40447)
    assert 0 < result.projection_probability < 1


def test_phase_jacobian_matches_finite_difference() -> None:
    x = torch.tensor([0.16, 0.47, 0.92], dtype=torch.float64)
    half = torch.tensor([0.13, -0.17, 0.24], dtype=torch.float64)
    values, derivative = phases.values_and_derivative(phases.signal(x), half)
    torch.testing.assert_close(
        values,
        phases.response(x, torch.cat((half, half.flip((0,))))),
        atol=1e-13,
        rtol=0,
    )
    for i in range(half.numel()):
        delta = torch.zeros_like(half)
        delta[i] = 1e-6
        a, b = half + delta, half - delta
        plus = phases.response(x, torch.cat((a, a.flip((0,)))))
        minus = phases.response(x, torch.cat((b, b.flip((0,)))))
        numerical = (plus - minus) / 2e-6
        torch.testing.assert_close(derivative[:, i], numerical, atol=1e-9, rtol=0)


def test_all_source_stages() -> None:
    folder = Path(workflow.__file__).parent
    notebook = json.loads((folder / "walkthrough.ipynb").read_text())
    proof = json.loads((folder / "provenance.json").read_text())
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(code) == len(proof["reviewed_code_cells"]) == 14
    assert all(c["execution_count"] is None for c in code)


def test_native_six_band_block_full_unitary_and_clean_workspace() -> None:
    import flagquantum as fq
    from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_chebyshev_lcu.overview.encoding import (
        banded_encoding,
        unitary,
    )
    from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview import (
        native,
    )

    matrix = workflow.source_matrix()
    assert matrix.shape == (16, 16) and int(matrix.count_nonzero()) == 45
    offsets = tuple(i for i in range(-15, 16) if bool(matrix.diagonal(i).any()))
    assert offsets == (-8, -7, -1, 0, 7, 8)
    zero = torch.eye(16, dtype=torch.float64)[0]
    program, scale = native.banded_program(matrix, zero)
    encoding = banded_encoding(matrix.to(torch.complex128))
    assert scale == encoding.alpha
    assert program.circuit.n_wires == 11
    assert program.report["workspace_qubits"] == 3
    assert not any(gate.name == "any" for gate in program.circuit.instructions)

    # QSVT reuses the block after its selector becomes nonzero: test all 256
    # logical columns, not just the success-input data subspace.
    initial = torch.zeros((256, 2048), dtype=torch.complex128)
    initial[torch.arange(256), 8 * torch.arange(256)] = 1
    circuit = fq.Circuit.from_ir(
        program.circuit, dtype=torch.complex128, bsz=256, inputs=initial
    )
    state = circuit.state().reshape(256, 256, 8)
    torch.testing.assert_close(state[:, :, 0].T, unitary(encoding), atol=1e-12, rtol=0)
    assert torch.count_nonzero(state[:, :, 1:]) == 0


@pytest.mark.parametrize("seed", [None, 37, 91])
def test_native_six_band_signed_source_inputs_and_named_success(
    seed: int | None,
) -> None:
    import flagquantum as fq
    from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_chebyshev_lcu.overview.encoding import (
        banded_encoding,
        unitary,
    )
    from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview import (
        native,
    )

    matrix = workflow.source_matrix()
    rhs = (
        workflow.input_vector("nozzle_008_b", 4)
        if seed is None
        else torch.randn(
            16, generator=torch.Generator().manual_seed(seed), dtype=torch.float64
        )
    )
    rhs /= rhs.norm()
    program, scale = native.banded_program(matrix, rhs)
    result = fq.run(program)
    state = result.statevector().reshape(256, 8)
    initial = torch.zeros(256, dtype=torch.complex128)
    initial[:16] = rhs
    expected = unitary(banded_encoding(matrix.to(torch.complex128))) @ initial
    torch.testing.assert_close(state[:, 0], expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(
        state[:16, 0],
        matrix.to(torch.complex128) @ rhs.to(torch.complex128) / scale,
        atol=1e-12,
        rtol=0,
    )
    assert torch.count_nonzero(state[:, 1:]) == 0
    assert program.report["resources"].n_operations == 343
    assert program.report["resources"].depth == 307
    probabilities = program.probabilities(result, registers=("selection", "flag"))
    assert probabilities[(0, 0)] == pytest.approx(
        float(state[:16, 0].abs().square().sum())
    )


def test_native_six_band_budget_and_input_refusals() -> None:
    from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview import (
        native,
    )
    from flagquantum import synth

    matrix = workflow.source_matrix()
    rhs = workflow.input_vector("nozzle_008_b", 4)
    rhs /= rhs.norm()
    for limits in (
        {"max_qubits": 8, "max_workspace": 0},
        {"max_operations": 342},
    ):
        with pytest.raises(synth.SynthesisError):
            native.banded_program(matrix, rhs, **limits)
    for invalid in (
        torch.zeros((8, 8)),
        torch.full((16, 16), float("nan")),
        matrix.to(torch.complex128) + 1j * torch.eye(16),
    ):
        with pytest.raises(ValueError, match="finite real 16x16"):
            native.banded_program(invalid, rhs)
    for invalid in (torch.zeros((16, 16)), torch.ones((16, 16))):
        with pytest.raises(ValueError, match="one through eight"):
            native.banded_program(invalid, rhs)


@pytest.mark.parametrize("offsets", [(-5, -2, 0, 4, 9), (-9, -6, -3, 0, 2, 5, 8)])
def test_native_bands_follow_bound_offsets_instead_of_source_positions(
    offsets: tuple[int, ...],
) -> None:
    import flagquantum as fq
    from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_chebyshev_lcu.overview.encoding import (
        banded_encoding,
        unitary,
    )
    from examples.synthesis.applications.cfd.qls_for_hybrid_solvers.qls_qsvt.overview import (
        native,
    )

    generator = torch.Generator().manual_seed(83 + len(offsets))
    matrix = torch.zeros((16, 16), dtype=torch.complex128)
    for offset in offsets:
        matrix.diagonal(offset).copy_(
            torch.randn(16 - abs(offset), generator=generator, dtype=torch.float64)
        )
    rhs = torch.randn(16, generator=generator, dtype=torch.float64)
    rhs /= rhs.norm()
    program, scale = native.banded_program(matrix, rhs, max_qubits=16, max_workspace=8)
    encoding = banded_encoding(matrix)
    assert scale == encoding.alpha
    initial = torch.zeros(256, dtype=torch.complex128)
    initial[:16] = rhs
    work = program.report["workspace_qubits"]
    state = fq.run(program).statevector().reshape(256, 1 << work)
    torch.testing.assert_close(
        state[:, 0], unitary(encoding) @ initial, atol=1e-12, rtol=0
    )
    assert torch.count_nonzero(state[:, 1:]) == 0

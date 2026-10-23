"""Enforce the representation-agreement invariant over a grid, not one circuit.

``test_vqe_backend_switch_golden_path.py`` asserts the three representations
agree at one configuration. That is necessary but it cannot say whether the
agreement survives a wider or deeper circuit, which is where a bond truncation
or a contraction order would be expected to matter. These tests drive the sweep
over a small grid so CI enforces the invariant across a regime rather than at a
point.

The subgrid is deliberately small: the checked-in artifact under
``artifacts/development/representation_agreement_precision/`` carries the full grid, and this
file's job is to fail when the invariant breaks, not to reproduce the payload.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from benchmarks import representation_agreement_sweep as sweep
from benchmarks.representation_agreement_sweep import (
    MODES,
    exact_max_bond,
    measure_cell,
    summarize,
)

pytestmark = pytest.mark.integration

#: Agreement bound for cells the dense path represents exactly. Both approximate
#: paths are doing exact arithmetic there, so only complex64 accumulation
#: separates them.
UNTRUNCATED_RELATIVE_BOUND = 1e-4

#: Agreement bound where the MPS cap actually discards weight. It is looser by
#: construction: a truncated state is a different state, and the point of
#: measuring is to state how much that costs rather than to assert it is free.
TRUNCATED_RELATIVE_BOUND = 0.5


def _cell(**overrides: object) -> dict:
    arguments = {
        "n_qubits": 6,
        "depth": 2,
        "structure": "nearest_neighbour",
        "max_bond": 16,
        "seed": 260720,
        "device": "cpu",
    }
    arguments.update(overrides)
    return measure_cell(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("n_qubits", [4, 6, 8])
@pytest.mark.parametrize("structure", ["nearest_neighbour", "all_to_all"])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_representations_agree_where_the_state_is_held_exactly(
    n_qubits: int, structure: str, dtype: torch.dtype
) -> None:
    """With a cap no narrower than the chain needs, all three paths must agree.

    This is the invariant in its strongest form: nothing has been discarded, so
    a disagreement beyond accumulation error would mean the representations do
    not compute the same program.
    """
    cell = _cell(
        n_qubits=n_qubits,
        structure=structure,
        max_bond=exact_max_bond(n_qubits),
        dtype=dtype,
    )

    assert "failures" not in cell, cell.get("failures")
    assert set(cell["energies"]) == set(MODES)
    assert cell["mps_truncated"] is False
    assert cell["spread_relative"] is not None
    bound = 1e-12 if dtype == torch.complex128 else UNTRUNCATED_RELATIVE_BOUND
    assert cell["spread_relative"] < bound, cell


@pytest.mark.parametrize("depth", [2, 6])
def test_every_representation_completes_across_depth(depth: int) -> None:
    """A representation that cannot run the program is a portability failure.

    ``mode="auto"`` is only a safe default if the planner's alternatives all
    execute the program, so an exception from any mode is reported as such
    rather than skipped.
    """
    cell = _cell(depth=depth, n_qubits=8)

    assert "failures" not in cell, cell.get("failures")
    assert set(cell["energies"]) == set(MODES)


def test_truncated_agreement_stays_within_the_measured_bound() -> None:
    """A cap that bites must degrade the energy gradually, not arbitrarily.

    Depth 6 at width 8 grows the state past a cap of 4, so this cell is in the
    regime the single-point result never reaches.
    """
    cell = _cell(n_qubits=8, depth=6, max_bond=4)

    assert cell["mps_truncated"] is True
    assert cell["observed_max_bond"] == 4
    assert cell["spread_relative"] is not None
    assert cell["spread_relative"] < TRUNCATED_RELATIVE_BOUND, cell


def test_reaching_the_cap_is_not_truncation_on_its_own() -> None:
    """A shallow circuit stops growing by itself, so equality with the cap lies.

    At width 8 and depth 2 the bond is 4 whether the cap is 4 or 256: nothing
    was discarded. Classifying that cell as truncated would put complex64 noise
    into the truncated bound and make the approximate regime look better than it
    is. The honest test compares against the bond the unconstrained state
    reaches.
    """
    assert exact_max_bond(4) == 4
    assert exact_max_bond(8) == 16

    coincidence = _cell(n_qubits=8, depth=2, max_bond=4)
    assert coincidence["observed_max_bond"] == 4
    assert coincidence["headroom_max_bond"] == 4
    assert coincidence["mps_truncated"] is False

    truncated = _cell(n_qubits=8, depth=8, max_bond=4)
    assert truncated["headroom_max_bond"] > 4
    assert truncated["mps_truncated"] is True


def test_summary_separates_the_two_regimes() -> None:
    """The bound over truncated cells must not be diluted by exact ones.

    Exact cells sit at epsilon and would dominate a pooled maximum, reporting a
    bound that describes arithmetic noise rather than behavior under
    approximation.
    """
    cells = [
        _cell(n_qubits=8, depth=2, max_bond=4),
        _cell(n_qubits=8, depth=8, max_bond=4),
    ]
    summary = summarize(cells)

    assert summary["cells_total"] == 2
    assert summary["truncated"]["cells"] == 1
    assert summary["untruncated"]["cells"] == 1
    assert summary["incomplete_cells"] == []
    assert (
        summary["truncated"]["max_spread_relative"]
        > summary["untruncated"]["max_spread_relative"]
    )


def test_spread_is_reported_relative_as_well_as_absolute() -> None:
    """A dimensionful spread grows with width by normalization alone.

    |E| grows with the number of sites, so the absolute spread of a wider cell
    can exceed a narrower one while the representations agree equally well. The
    relative figure is what a width sweep can be read from.
    """
    narrow = _cell(n_qubits=4, max_bond=exact_max_bond(4))
    wide = _cell(n_qubits=12, max_bond=exact_max_bond(12))

    for cell in (narrow, wide):
        assert cell["spread_absolute"] is not None
        assert cell["spread_relative"] is not None
        assert cell["spread_relative"] < UNTRUNCATED_RELATIVE_BOUND, cell

    assert abs(wide["energies"]["statevector"]) > abs(narrow["energies"]["statevector"])


def test_precision_comparison_keeps_the_angles_and_widens_the_arithmetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A precision comparison must change arithmetic, not the input circuit."""
    parameters_seen: list[torch.Tensor] = []
    build = sweep.build_ansatz

    def capture(parameters: torch.Tensor, structure: str) -> sweep.fq.Circuit:
        parameters_seen.append(parameters.clone())
        circuit = build(parameters, structure)
        expected = (
            torch.complex128 if parameters.dtype == torch.float64 else torch.complex64
        )
        assert circuit.state().dtype == expected
        return circuit

    monkeypatch.setattr(sweep, "build_ansatz", capture)
    single = _cell(n_qubits=4, dtype=torch.complex64)
    first_double = len(parameters_seen)
    double = _cell(n_qubits=4, dtype=torch.complex128)

    assert "failures" not in single, single.get("failures")
    assert "failures" not in double, double.get("failures")
    assert parameters_seen[0].dtype == torch.float32
    assert parameters_seen[first_double].dtype == torch.float64
    assert torch.equal(
        parameters_seen[0].to(torch.float64), parameters_seen[first_double]
    )
    assert double["mps_truncated"] is False
    assert double["spread_relative"] < 1e-12


def test_a_failed_headroom_probe_is_incomplete_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful energies cannot hide an unmeasured truncation classification."""
    build = sweep.build_ansatz
    calls = 0

    def fail_probe(parameters: torch.Tensor, structure: str) -> sweep.fq.Circuit:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("headroom probe refused")
        return build(parameters, structure)

    monkeypatch.setattr(sweep, "build_ansatz", fail_probe)
    cell = _cell(n_qubits=6, max_bond=4)

    assert all(value is not None for value in cell["energies"].values())
    assert cell["mps_truncated"] is None
    assert "headroom_probe" in cell["failures"]
    assert len(summarize([cell])["incomplete_cells"]) == 1


def test_recorded_precision_grids_compare_the_same_complete_workload() -> None:
    """The published comparison separates arithmetic from bond truncation."""
    root = (
        Path(__file__).resolve().parents[2]
        / "artifacts/development/representation_agreement_precision"
    )
    records = [
        json.loads((root / f"{dtype}.json").read_text())
        for dtype in ("complex64", "complex128")
    ]
    single, double = records
    for field in (
        "grid",
        "seed",
        "observable",
        "parameter_initialization",
        "source_commit",
    ):
        assert single[field] == double[field]
    for record, bound in zip(records, (1e-4, 1e-12), strict=True):
        assert record["measuring_source_committed"] is True
        assert record["summary"]["cells_total"] == 28
        assert record["summary"]["incomplete_cells"] == []
        assert record["summary"]["untruncated"]["cells"] == 22
        assert record["summary"]["truncated"]["cells"] == 6
        assert record["summary"]["untruncated"]["max_spread_relative"] < bound
        assert record["summary"]["truncated"]["max_spread_relative"] > 0.1
        assert record["release_gate_allowed"] is False
        assert record["scalability_claim_allowed"] is False
        assert all(set(cell["energies"]) == set(MODES) for cell in record["cells"])
    fields = ("n_qubits", "depth", "structure", "max_bond_cutoff", "mps_truncated")
    assert [tuple(c[f] for f in fields) for c in single["cells"]] == [
        tuple(c[f] for f in fields) for c in double["cells"]
    ]


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("structure", ["nearest_neighbour", "all_to_all"])
def test_batched_ising_energy_preserves_term_values_and_gradients(
    dtype: torch.dtype, structure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from flagquantum.runtime import run_native

    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    parameters = (
        (0.05 * torch.randn(2, 4, 2, generator=torch.Generator().manual_seed(19)))
        .to(real_dtype)
        .requires_grad_()
    )
    target = run_native(
        sweep.build_ansatz(parameters, structure), mode="tensor_network"
    )

    def refuse_dense_state(*args: object, **kwargs: object) -> None:
        raise AssertionError("energy must remain a tensor-network contraction")

    monkeypatch.setattr(target, "state", refuse_dense_state)
    batched = sweep.energy(target, 4)
    termwise = sum(-target.expectation_ps(z=(q, q + 1)).sum() for q in range(3))
    termwise += sum(-0.7 * target.expectation_ps(x=(q,)).sum() for q in range(4))
    tolerance = 1e-11 if dtype == torch.complex128 else 1e-5
    torch.testing.assert_close(batched, termwise, atol=tolerance, rtol=tolerance)
    actual = torch.autograd.grad(batched, parameters, retain_graph=True)[0]
    expected = torch.autograd.grad(termwise, parameters)[0]
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_deep_batched_ising_energy_agrees_with_independent_dense_reference(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    from flagquantum.runtime import run_native

    parameters = (
        0.05 * torch.randn(8, 8, 2, generator=torch.Generator().manual_seed(260720))
    ).to(torch.float64 if dtype == torch.complex128 else torch.float32)
    circuit = sweep.build_ansatz(parameters, "nearest_neighbour")
    network = run_native(circuit, mode="tensor_network")
    dense_reference = run_native(circuit, mode="statevector")

    def refuse_dense_state(*args: object, **kwargs: object) -> None:
        raise AssertionError("the tensor-network result must not materialize a state")

    monkeypatch.setattr(network, "state", refuse_dense_state)
    tolerance = 1e-11 if dtype == torch.complex128 else 1e-5
    torch.testing.assert_close(
        sweep.energy(network, 8),
        sweep.energy(dense_reference, 8),
        atol=tolerance,
        rtol=tolerance,
    )

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


# --- gradient and state conformance (#899) -----------------------------------
#
# The energy spread above answers "do the representations report the same
# number". A user who swaps representation mid-training needs three further
# answers that the spread cannot give: whether the state, not only one
# observable, agrees; whether the reported gradient is the gradient of the
# computation that actually ran; and whether that gradient still points the way
# the exact one does. These tests pin each separately because they fail
# independently.


def test_fidelity_is_one_where_no_weight_was_discarded() -> None:
    """Exact cells must agree as states, not only as one expectation value.

    A single Ising energy is a projection: two different states can report the
    same number. Fidelity is the quantity that says the states agree.
    """
    cell = _cell(n_qubits=4, max_bond=exact_max_bond(4), gradients=True)

    assert cell["mps_truncated"] is False
    fidelity = cell["states"]["mps"]["fidelity_vs_statevector"]
    network = cell["states"]["tensor_network"]["fidelity_vs_statevector"]
    assert fidelity is not None and network is not None
    assert fidelity > 1 - 1e-10, cell["states"]
    assert network > 1 - 1e-10, cell["states"]


def test_fidelity_separates_from_norm_drift() -> None:
    """Fidelity is a normalized overlap, so it must not absorb a norm error.

    An unnormalized state can be a pure rescaling: the physical state is
    identical while the norm is wrong. Reporting one number for both would let a
    norm bug read as an agreement failure, or vice versa.
    """
    reference = torch.tensor([[1.0, 0.0]], dtype=torch.complex128)
    rescaled = reference * 2.0

    conformed = sweep.state_conformance(rescaled, reference)

    assert abs(conformed["fidelity_vs_statevector"] - 1.0) < 1e-12
    assert abs(conformed["norm_drift"] - 1.0) < 1e-12


def test_fidelity_falls_when_the_cap_actually_discards_weight() -> None:
    """A truncated state is a different state and must be reported as one."""
    cell = _cell(n_qubits=8, depth=8, max_bond=4, gradients=True)

    assert cell["mps_truncated"] is True
    fidelity = cell["states"]["mps"]["fidelity_vs_statevector"]
    assert fidelity is not None
    assert fidelity < 1 - 1e-6, cell["states"]


def test_gradient_direction_is_reported_apart_from_its_magnitude() -> None:
    """An optimizer follows direction; a curve fit does not settle whether it does.

    A gradient can be wrong in scale, in direction, or in both. Cosine
    similarity isolates the direction, so a purely rescaled gradient is visible
    as a scale error rather than as a direction failure.
    """
    reference = torch.tensor([1.0, -2.0, 0.5])
    rescaled = 3.0 * reference

    conformed = sweep.gradient_conformance(rescaled, reference)

    assert abs(conformed["cosine_similarity"] - 1.0) < 1e-12
    assert abs(conformed["relative_l2_error"] - 2.0) < 1e-12

    reversed_direction = sweep.gradient_conformance(-reference, reference)
    assert abs(reversed_direction["cosine_similarity"] + 1.0) < 1e-12


def test_gradients_agree_where_no_weight_was_discarded() -> None:
    """The exact regime must agree in gradient, not only in energy.

    Double precision is used because the claim is about the representation, and
    at complex64 the float32 accumulation of the same arithmetic already
    separates the paths by about 4e-7.
    """
    cell = _cell(
        n_qubits=4,
        max_bond=exact_max_bond(4),
        dtype=torch.complex128,
        gradients=True,
    )

    assert cell["mps_truncated"] is False
    for mode in ("mps", "tensor_network"):
        entry = cell["gradients"]["per_mode"][mode]
        assert entry["relative_l2_error"] is not None
        assert entry["relative_l2_error"] < 1e-10, (mode, entry)
        assert entry["cosine_similarity"] > 1 - 1e-12, (mode, entry)


def test_the_truncated_derivative_check_separates_exact_from_truncated_cells() -> None:
    """The check must be able to fail, or its agreement would mean nothing.

    This is the distinction the issue turns on, measured against the map that
    actually ran rather than against the ideal circuit. Through a cap that does
    not bite, autograd reproduces the derivative of its own forward pass to
    within 1e-8 relative. Through a cap that does, the same check reports a
    discrepancy of about 5e-2 relative.

    That second number is a measured defect in the local MPS backward pass, not
    an expected property of truncation: a bond cap changes the forward map by
    about 1e-6 in energy here, so its true derivative is close to the exact one,
    and a derivative of the truncated map would have to land near it too. It is
    reported here and tracked separately; this test exists so the harness cannot
    quietly record agreement it did not measure.
    """
    parameters = (
        0.05 * torch.randn(6, 8, 2, generator=torch.Generator().manual_seed(11))
    ).to(torch.float64)

    exact = sweep.truncated_derivative_error(
        parameters, structure="nearest_neighbour", n_qubits=8, max_bond=16
    )
    truncated = sweep.truncated_derivative_error(
        parameters, structure="nearest_neighbour", n_qubits=8, max_bond=4
    )

    assert exact["checked"] >= 4
    assert exact["max_relative_error"] < 1e-8, exact
    assert truncated["max_relative_error"] > 1e-2, truncated


def test_truncation_error_and_differentiation_error_are_reported_apart() -> None:
    """Two different numbers: distance from the ideal, and from the map's own derivative.

    A truncated simulator can report a gradient that is far from the ideal one
    because the state is approximate, or because the backward pass differentiates
    something other than the forward pass. Reporting one number for both would
    let either hide behind the other.
    """
    parameters = (
        0.05 * torch.randn(6, 8, 2, generator=torch.Generator().manual_seed(11))
    ).to(torch.float64)

    ideal = sweep.ideal_gradient(parameters, structure="nearest_neighbour", n_qubits=8)
    approximate = sweep.truncated_gradient(
        parameters, structure="nearest_neighbour", n_qubits=8, max_bond=4
    )
    distance_from_ideal = sweep.gradient_conformance(approximate, ideal)
    distance_from_own_map = sweep.truncated_derivative_error(
        parameters, structure="nearest_neighbour", n_qubits=8, max_bond=4
    )

    assert distance_from_ideal["relative_l2_error"] is not None
    assert distance_from_ideal["relative_l2_error"] > 1e-3, distance_from_ideal
    assert distance_from_own_map["max_relative_error"] > 1e-2, distance_from_own_map


def test_dense_state_accessor_reads_every_representation() -> None:
    """One accessor, so the fidelity path cannot silently skip a mode."""
    from flagquantum.runtime import run_native

    parameters = (
        0.05 * torch.randn(2, 4, 2, generator=torch.Generator().manual_seed(260720))
    ).to(torch.float64)
    circuit = sweep.build_ansatz(parameters, "nearest_neighbour")

    states = {
        mode: sweep.dense_state(
            run_native(
                circuit, mode=mode, **({"max_bond": 16} if mode == "mps" else {})
            )
        )
        for mode in MODES
    }

    for mode, state in states.items():
        assert state.dtype == torch.complex128, mode
        assert state.numel() == 2**4, mode
    torch.testing.assert_close(states["mps"], states["statevector"])
    torch.testing.assert_close(states["tensor_network"], states["statevector"])


def test_a_mode_that_cannot_differentiate_is_reported_not_substituted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused derivative is evidence, not a reason to quietly use another mode."""
    import flagquantum.runtime as runtime

    original = runtime.run_native

    def refuse_tensor_network(*args: object, **kwargs: object) -> object:
        if kwargs.get("mode") == "tensor_network":
            raise RuntimeError("tensor-network derivative refused")
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, "run_native", refuse_tensor_network)
    monkeypatch.setattr(sweep.fqr, "run_native", refuse_tensor_network)
    cell = _cell(n_qubits=4, max_bond=exact_max_bond(4), gradients=True)

    assert "tensor_network" in cell["failures"]
    assert "tensor_network" in cell["gradients"]["unsupported"]
    assert cell["gradients"]["per_mode"]["mps"]["relative_l2_error"] is not None


def test_gradient_section_is_absent_unless_requested() -> None:
    """The published precision artifact must not change shape under this work."""
    cell = _cell(n_qubits=4, max_bond=exact_max_bond(4))

    assert "gradients" not in cell
    assert "states" not in cell

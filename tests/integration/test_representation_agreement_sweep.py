"""Enforce the representation-agreement invariant over a grid, not one circuit.

``test_vqe_backend_switch_golden_path.py`` asserts the three representations
agree at one configuration. That is necessary but it cannot say whether the
agreement survives a wider or deeper circuit, which is where a bond truncation
or a contraction order would be expected to matter. These tests drive the sweep
over a small grid so CI enforces the invariant across a regime rather than at a
point.

The subgrid is deliberately small: the checked-in artifact under
``benchmarks/results/representation_agreement/`` carries the full grid, and this
file's job is to fail when the invariant breaks, not to reproduce the payload.
"""

from __future__ import annotations

import pytest

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
def test_representations_agree_where_the_state_is_held_exactly(
    n_qubits: int, structure: str
) -> None:
    """With a cap no narrower than the chain needs, all three paths must agree.

    This is the invariant in its strongest form: nothing has been discarded, so
    a disagreement beyond accumulation error would mean the representations do
    not compute the same program.
    """
    cell = _cell(
        n_qubits=n_qubits, structure=structure, max_bond=exact_max_bond(n_qubits)
    )

    assert "failures" not in cell, cell.get("failures")
    assert set(cell["energies"]) == set(MODES)
    assert cell["mps_truncated"] is False
    assert cell["spread_relative"] is not None
    assert cell["spread_relative"] < UNTRUNCATED_RELATIVE_BOUND, cell


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

# Grid Trotter tutorials file audit

This batch covers Q-0433 through Q-0435 in Qualtran v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389`. Both unchanged original notebooks, with 50 and 43 cells, passed in independent kernels. The native tutorial tests execute both checked-in notebooks in isolated kernels that reject reference-library imports.

| Inventory | Native owner | Evidence |
| --- | --- | --- |
| Q-0433 | `examples/notebooks/register_grid_trotter.ipynb` | Twelve rendered graphs across all six source cases, one-level leaf totals, and a runnable native append example. The six immediate projections are compared by component class and multiplicity against the frozen source. |
| Q-0434 | `examples/notebooks/grid_trotter_costs.ipynb` | Seven graph views, three numerical figures, distance scaling, two cubic fits with corrected Newton updates, QROM ranges, and source and paper budgets. |
| Q-0435 | `tests/integration/test_register_graph_notebook.py` and `tests/unit/test_grid_hamiltonian_resources.py` | The two original notebook smoke checks now execute the two corresponding native assets, validate render formats and exercise locked-source model comparisons. |

The operation tutorial shows the two inverse-square-root declarations, QVR, kinetic energy, pair potential and total potential. Its projected call graph hides casts, matching the source's `ignore_split_join` projection on this set. The native graph combines nodes with identical declared costs. The resource comparison tutorial uses the native declared descendants and leaves model estimates separate from emitted instructions. It hides allocation and release nodes in that visual view.

The costs tutorial uses 61 positions with spacing 0.25 and all 60 distinct positive distances. The maximum cubic and corrected Newton errors reproduce the original runs at approximately `5.86751088012e-5` and `2.58210142157e-9`. The 14-bit range partitions into 28 regions with a modeled lookup cost of 26 Toffoli equivalents. The original standalone cubic error exceeds its displayed \(2^{-15}\) target. The independent corrected result is below that target.

| Example | Legacy model T before rotation synthesis | Paper T |
| --- | ---: | ---: |
| Polynomial | 2,676 | 2,880 |
| Newton | 5,768 | 5,664 |
| Pair potential | 9,176 | 9,052 |

The original pair display subtracts 1,248 from the already separated T count and prints 7,928. The native tutorial records that literal calculation and separately reports 10,424 when the 24 rotations are synthesized at tolerance `1e-11`. The native one-electron kinetic append example emits eight T-family instructions and two measurements. Its independent source budget gives one Toffoli equivalent. Complex-state and cleanup evidence remains in the grid operator test suites.

The tutorial projections do not add physical decompositions to the original opaque leaves or replace the frozen drawing generalizers. The source arithmetic, angle, labels and physical cleanup gaps remain in the individual operator audits. The native fixed-point appendix uses the conventional unsigned exponent `word_width - bit_index - 1`; the original appendix omits its final minus one.

Run the resource report independently with `python -m examples.algorithms.grid_hamiltonian_resources`.

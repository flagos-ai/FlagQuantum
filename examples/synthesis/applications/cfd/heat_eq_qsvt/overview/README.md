# QSVT for the finite-difference heat equation

The [walkthrough](walkthrough.ipynb) recreates the eight-point grid and all
1,024 time samples. Eight independent basis inputs reconstruct the full signed
finite-difference operator through a native block encoding. The classical
model also produces a degree-three inverse polynomial and all four Wx angles
from the source parameters. The phase response is verified on a dense grid.

The phase convention follows the original Laurent fit. The circuit applies
only the first three angles, as dictated by the source loop. Each of the four
iterations removes the phase of the first occupied basis amplitude in native
basis order, takes the real projection and prepares the normalized result for
the next iteration. The plot shows the four fields before their corresponding
updates and uses the cumulative measured projection norms. Those norms depend
on the phase convention and amplitude ordering used for the real projection.
The saved hosted execution reports projection norms near machine precision.
The local signed fields and norms are determined by the reproducible native
basis ordering.

Run the same workflow from the repository root with FlagQuantum and Matplotlib.

```sh
python -m examples.synthesis.applications.cfd.heat_eq_qsvt.overview.main
```

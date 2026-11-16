# QREF resource interchange source audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The notebook and export tests
use QREF 0.11.0 and Bartiq 0.12.1 from an isolated reference environment.

| Source file | Frozen Git blob | Native disposition |
| --- | --- | --- |
| `qref_interop/__init__.py` | `741991b9f3dce077089729e3cd75efdf904c56ee` | Compiler formats owns the optional QREF entry points |
| `qref_interop/_bloq_to_qref.py` | `aab3422ba170c0c909ee152d46537f78335829b8` | Native leaf, indexed component and logical call-graph exporters |
| `qref_interop/_bloq_to_qref_test.py` | `e7092a50ce00daf63488a419b0bd8609e937fc38` | All 28 frozen cases pass. Paired tests cover the admitted native paths |
| `qref_interop/bartiq_demo.ipynb` | `60e8055c7669d4dd69c8d42c4b882c9a7bfc7fc0` | Native `resource_interchange.ipynb` covers all 38 cells by topic and runs all 15 code cells in an independent kernel |

The optional format adapter receives native register declarations, typed
connections and modeled call graphs. It splits through declarations into
input and output ports, expands one-dimensional arrays, and preserves the
symbolic-size local variable convention required by Bartiq. The original
refusal of multidimensional ports is retained. Type labels and resource
projections come from the existing model owner. Named selection can unfold
available decompositions, using stable occurrence IDs for child names and
relative endpoints. Source-compatible logical call graphs preserve symbolic
repetition counts and zero expressions.

Two native Program fixtures reproduce the source two-CNOT topology,
emitted operands, output bindings and full complex amplitudes. Both emit
two CX gates, and both have a source Clifford count of two. The crossed
fixture returns the two physical wires in reversed order. These fixtures
have no private work or measurement feedback. Export emits zero instructions
and leaves the existing append path authoritative.

The source decomposed call-graph path clears parent ports before connections,
which causes QREF assignment validation to raise for the connected CNOT
fixtures. Native construction clears both fields together. It retains
the two declared child occurrences and adds the counted repetition.

The native notebook uses weights 0 through 23 with precision (10^{-5}).
The full alias preparation topology, port widths, five routines and
resources match the source QREF model. Recompiling child costs gives the
same historical total. The pinned and alternative uniform preparation
formulas remain visible. Substituting the qualified child parameter at
24 gives 313 T, and the supplied custom function at 101 evaluates to
16 T. The original notebook's unqualified `N` leaves the child parameter
symbolic. Both notebooks execute in separate kernels and render two SVGs.

| Alias preparation measurement | Result |
| --- | --- |
| Pinned historical model | T307, Clifford1770, rotations2 |
| Native default model | T307, Clifford1582, rotations2 |
| Native appended primitive stream | H244, X113, T156, Tdg151, CX1145, S68, measure68, conditional CZ68, reset68, Rz2 |

The existing alias-preparation regressions cover small-state complex
amplitudes and work cleanup. The notebook's 24-input case compares the
complete model and counts emitted instructions without allocating a dense
state. Generic foreign component reflection and native replay of the
larger chemistry and factoring fixtures remain open under Q-0794 and
Q-0795.

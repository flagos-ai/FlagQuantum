# Cirq notebook and legacy resource audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The five original source, test
and notebook blobs are pinned in
`tests/unit/test_cirq_notebook_resources_file.py`.

| Source responsibility | FlagQuantum owner |
| --- | --- |
| Legacy arithmetic and synthesis estimate | The typed legacy result adds and scales the three cost fields, and delegates rotation synthesis to the shared compiler formula. |
| Native legacy estimate | Query the compiler's owned resource graph with legacy shims and fold the categories with the existing legacy gate budget. Explicit native annotations remain compiler inputs. |
| Cirq compatibility estimate | The optional adapter reads an SDK annotation, tests the source's direct Clifford/T/rotation rules, then traverses Cirq decomposition. It handles circuits, moments and lists and retains tag, classical-control and repeated circuit-operation conventions for estimates. |
| Notebook cost tags and SVG | The optional notebook tools annotate operations with the legacy result and plot the source-order register wires. The empty-circuit checks include a circuit with an empty moment. |
| Paired layout and notebook execution | The original operation and its decomposition are displayed side by side or vertically. Notebook execution accepts a caller-relative name or an explicit path with a kernel and timeout. Optional display dependencies load only when used. |

The [legacy T complexity notebook](../../examples/notebooks/cirq_t_complexity.ipynb)
retains the original forward and adjoint conjunction, the five-control
example, and the complete three-through-eleven-control curve. The plot uses
the native compiler model and its source legacy conversion. The notebook
runs in an independent kernel with the reference packages and Cirq imports
blocked.

The unchanged original suite passed 13 cases with Cirq 1.6.1. With Cirq
1.7.0 it passed 12 cases and failed the old controlled-H assertion. That
SDK version specializes a controlled-Y half power as a CYPowGate, changing
the compatibility estimate produced by the original source.

| Controlled H query | T | Clifford | Rotations |
| --- | ---: | ---: | ---: |
| Native compiler legacy declaration | 0 | 4 | 2 |
| Original Cirq compatibility, SDK 1.6.1 | 2 | 4 | 2 |
| Original Cirq compatibility, SDK 1.7.0 | 1 | 3 | 1 |

The native adapter matches the observed compatibility outcome on each SDK.
The tests also match the original tagged and conditional cost estimates,
empty-diagram rejection, cost annotation and layout. All fields in the
table and plot are resource estimates. The helper does not lower or run a
native quantum circuit, and a compatibility cost does not confer static
import or export support.

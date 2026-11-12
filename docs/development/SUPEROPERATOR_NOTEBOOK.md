# Superoperator notebook acceptance

The unchanged reference is Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`,
`qualtran/simulation/supertensor.ipynb`, frozen blob
`6feb17250e1ed1cdd271e65dcb8f5036808cdae3` (Q-0884).
Implementation `37922753` contains the native notebook
`examples/notebooks/superoperator_simulation.ipynb`, its composition helper
`examples/algorithms/superoperator_workflow.py`, and the acceptance tests in
`tests/unit/test_superoperator_notebook.py`. Source attribution is retained.

Run the notebook from the repository root, or:

```bash
python -m examples.algorithms.superoperator_workflow
```

## Whole asset disposition

All 28 original cell positions are accounted for: 13 code and 15 Markdown.
The reference code strings execute unchanged in an independent namespace;
only the three score presentations and three Quimb drawing calls are captured.
The native notebook executes separately with imports of Qualtran, Cirq and
Quimb forbidden. Numerical statements in the reference are not rewritten.

| Original cell positions | Native disposition and acceptance |
| --- | --- |
| 0,1,4 | Closed/open systems, probability versus coherence, plus-state introduction; native terminology and PyTorch data |
| 2,3 | Plus statevector and density matrix; complete complex arrays compared |
| 5 | Zero-H-Z measurement coin; entire density compared, same explicit copy/environment trace |
| 6,7,10,11 | Matrix-vector interpretation, four-axis convention, conjugation, state/effect orientation and row-major reshaping |
| 8,9,12,13 | H channel and density shapes; coherent and incoherent evolution compared without phase alignment |
| 14,15 | Forward/backward graph inspection; native read-only incidence drawing of the same numerical leaves |
| 16,17,18,19 | Zero-H-CZ-H-Cast dilation, retained q/c, pure isometry and physical append; classical Cast is an identity, not measurement |
| 20,21,22,23 | Discard q and retain c; ordinary pure contraction refuses discard in both implementations |
| 24,25,26 | Probability trace, forward/backward index identification, rank-four measurement channel and Stinespring explanation |
| 27 | Existing numerical owner/dependency map; explicit trace order and no transpose of backward tensor data |

The native notebook has unique stable cell IDs, empty cell metadata, null
execution counts, no saved outputs or attachments, and a normalized Python 3
kernel. Old runtime/version metadata is not carried over. All six native
figures render; the coin and discarded-dilation incidence figures were visually
inspected. The notebook contains no embedded generated image asset.

## Ownership and representation

This batch adds example composition and tests only. Existing Simulation owners
provide measurement-copy tensors, conjugate pairing, trace-index merging,
register-axis grouping and contraction. Existing Drawer code displays the very
same nodes. The example composes Compiler measurement/reset primitives with
Circuit IR only for the physical consumer; Simulation imports no Compiler code.
No new production graph, registry, executor, protected API or dependency is added.

Each port is a tuple of row-major, MSB-first words. Connections are undoubled
integer labels; physical wire numbers are separate. Channels expose
`(output-forward, output-backward, input-forward, input-backward)`. Dilation's
output group orders q before c. The caller explicitly traces the copy environment
or discarded q after its producer; retained classical c is not zero scratch.

For coin, dilation and discarded dilation, every source tensor's data and axis
incidence is matched in order against the native pure/conjugate leaves. A
bijective connection renaming preserves all internal sharing and grouped outer
ports, including the environmental trace. The plus-state helper uses zero/H
leaves to produce the source's atomic PlusState vector; their complete numerical
results agree, without claiming the same node count for that atomic leaf.

Native integer indices replace foreign tuple objects. Native bipartite incidence
views replace three musical scores and three Quimb diagrams; connectivity and
axis visibility are preserved, not score layout, Quimb colors or spectral
coordinates. Pure-discard diagnostic wording is native. These adaptations are
explicit; no claim of foreign object or pixel equivalence is made.

## Physical path, phases and ownership

Users still write `circuit.append(ZMeasurementDilation())`. Its actual emitted
stream is H on record, CZ(record,q), H on record. Zero preparation is represented
by a caller-owned fresh zero carrier; Cast emits no physical instruction. The
full clean-record isometry, including every spectator basis column, is compared
to the unchanged reference dilation, without global-phase correction.

`ZMeasurementDilation(measured=True)` materializes c in classical slot 0 using
the existing Z-measure/conditional-X clear primitive, then resets consumed q.
For an entangled input with nontrivial complex phases, both exact unnormalized
measurement branches retain the correct spectator amplitudes. Consumed q and the
record carrier are zero in both branches; the spectator is not reset. Ordinary
dynamic append is also checked across eight trajectories. This physical
materialization is additional to the original numerical Cast/Discard example;
it is not presented as an upstream physical lowering.

The complete measurement channel matches source, including off-diagonal input
coherences. A phase-bearing differentiable input independently checks density
values and the weighted outcome derivative. There is no sampling or renormalizing
inside the tensor contraction itself.

## Resources and validation

| Reading | Result |
| --- | --- |
| Coherent emitted primitives | H2, CZ1; zero measurement/reset/T; caller-owned fresh record retained |
| Measured/cleared emitted primitives | H2, CZ1, measurement1, conditional X1, reset1; zero T |
| Physical storage | q wire0, fresh-zero record wire1, untouched spectator wire2; one classical slot for measured form |
| Unchanged source dilation QEC model | Clifford3, measurement0; state preparation and Cast add no counted primitive here |
| Unchanged source MeasureZ leaf QEC model | measurement1, Clifford0; not the physical materialization/clear sequence |
| Numerical leaves, traces and drawing | Zero emitted primitives; tensor storage and logical environment edges, not physical workspace |

Acceptance: six new cases; 277 related tests pass in 18.48s (four existing
numerical/complex-cast warnings), and 84 CPU/TN checks pass in 7.52s (two optional
dependency/deprecation warnings). All 13 original and 13 native code cells
execute; all 28 cell/metadata/asset dispositions are recorded above. Both Python
files pass strict mypy, Black and Ruff. Scope, architecture, multi-level IR,
protected API checks and the example entrypoint pass. The notebook, `_quimb.py`,
`_dense.py`, `z_basis.py`, `discard.py`, and `cast.py` match all six frozen blobs.

Helper coverage is 86% in the regression alone and 98% after the instrumented
standalone example (73 statements, 14 branches). The remaining uncovered path
is rejection of an unknown example kind outside its typed Literal interface.
The previous default-suite timeout remains recorded; it was not rerun and no
full-suite or GPU-green claim is made. These are local numerical/CPU results,
not distributed scalability evidence.

Q-0884 closes as a complete concrete notebook asset. General protocol boundaries
and source defects in Q-0882/Q-0883/Q-0887/Q-0889/Q-0891/Q-0895 stay open; this
asset neither changes those dispositions nor claims arbitrary Hermitian-channel
synthesis or general foreign compatibility. G3 has not started.

# Clean-bank QROAM source audit

This batch follows Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The locked package entry,
tutorial, implementation and tests match their inventory blobs.

| Inventory | Native owner and evidence |
| --- | --- |
| Q-0475 package entry | The four source imports map to `source_qrom.source_qrom_ir`, `clean_table_lookup.CleanTableLookup`, `qroam_clean_erasure.qroam_clean_measurement_erase_ir` and `select_swap_qrom.select_swap_qrom_ir`. Compiler owns lookup lowering and Algorithms owns the named append binding. The separate leaf declaration and model live in `qroam_clean_metadata` and `qroam_resource_model`. |
| Q-0476 tutorial | `examples/notebooks/register_qroam_clean.ipynb` covers both concrete datasets, both symbolic instances, signatures and the immediate call graph. It also executes the paired named append operations and displays the separate executable and modeled counts. |
| Q-0477 implementation | The forward emitter and resource models are reused. The wrapper's concrete and symbolic ports, packed leaf input positions and unsigned padded literals are verified. Measured erase uses the existing native signed reversal and replay feedback. Accepted under the explicitly approved faithful-migration rule; see retained source limits below. |
| Q-0478 original tests | The complete locked test file ran, including its slow classical case. It reported 25 passes, one skip and one expected failure. Native coverage uses the signature diagrams, resource and block selection checks, bank layout, direct source gate comparison and branch tests. |

## Binding and branch evidence

Use `circuit.append(lookup.make_on(...))` with caller-owned controls,
selections, zero target banks and temporary work. The controls and
selection words are big-endian. Each named `target{i}_` receives the
selected word. `junk_target{i}_` retains the other words of the block
in the routed order. The targets and retained bank cells are logical
outputs until consumed by erase. The work bits are temporary.

The compiler shares the original low-bit swap schedule between the
emitter and unsigned padded leaf data. The walk compares the untouched
source wrapper's input bindings down to each retained cell index. The
two-axis padded table keeps the original row-major axis order.

For the two-dataset, three-row append test, the native forward stream
matches the ordered source operations, operands and feedback. With
complex input across all four physical addresses, every forward-plus-
erase history returns the input vector up to its branch normalization.
All bank and work wires are zero, including for the padded address.
The named classical destinations are deliberately nonconsecutive.
The existing source gate and signed-channel regressions also cover
controlled and multidimensional layouts.

## Resource figures

The tutorial's uncontrolled five-row example loads two three-bit
datasets with block length two, using 16 wires including one scratch
bit. Its forward stream emits 28 T-family instructions, 53 CX and one
measurement. Native erase emits 28 T-family instructions, 41 CX and 13
measurements, of which 12 are bank readouts. The default source wrapper
and padded leaf each declare one Toffoli and zero explicit measurements
in this example. With a one-control five-row table and explicit unit
erase blocks, the wrapper and padded leaf instead declare three and
four Toffolis respectively. See
[forward and erase validation](REGISTER_SOURCE_QROAM_CLEAN.md).

## Retained source limits for accepted Q-0477

- The locked adjoint leaf provides a call and resource declaration. Its
  executable decomposition raises `DecomposeNotImplementedError`, so
  the native measured erase has no literal source gate stream to compare.
- The locked forward decomposition loses omitted selectors for some
  mixed-axis and fully blocked layouts. The native emitter rejects those
  layouts. The append binding takes canonical selector widths and
  positive scalar target widths.
- With data `[1, 2, 3, 4, 5]`, block length two, selection zero and
  control zero, the locked source's direct classical call omits the
  required RIGHT `target0_` and raises `ValueError`. The physical
  native path leaves the clean banks zero with an inactive control.

The original 15-cell tutorial and native tutorial both completed in
independent kernels. The focused native regression passed 275 tests
with no skips. Strict typing, format, team scope, architecture, IR
contracts and public API checks passed.

## Faithful-migration acceptance on 2026-11-27

The user explicitly approved the complete-file rule in
[G2 closure](G2_FAITHFUL_MIGRATION_CLOSURE_PROPOSAL.md). Q-0477 is accepted
for its complete source responsibility, reusing the published constructor,
forward stream, branch, resource and mandatory-check evidence above and in
[constructor closure](RESOLVER_QROAM_CLEAN_AUDIT.md). No implementation or
reference was changed to obtain this disposition.

Compiler's data-loading owners retain the follow-ups: an executable upstream
adjoint body would need a new literal stream comparison; omitted-selector
layouts need an upstream graph fix before parity can be claimed; inactive
control classical RIGHT-output failure and wrapper/padded-leaf cost differences
remain reproduced diagnostics. Native paper-based erasure and separately
supported whole-bank lowering keep their independent evidence and cannot be
called literal source decompositions. No unsupported source shape, performance
claim or new runtime certificate is inferred from migration acceptance.

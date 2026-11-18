# Complete integer conversion notebook migration

The untouched Qualtran v0.7.0 notebook at commit
`bcb17bb25360ed68ce3bc084645db3e929799389` is restored from exact Git blob
`9b9f5f1a659905b2fc9f7e6dfd7e7bc1a0cb4783`. All 34 cell types and original
teaching topics are preserved in `examples/notebooks/register_conversions.ipynb`.
Two additional cells demonstrate native coherent append execution and explicit
prefix cleanup. Original copyright and Apache-2.0 attribution are retained.

## Complete topic and direct dependency closure

| Original responsibility | Native placement and evidence |
| --- | --- |
| Ten-bit one's-complement to two's-complement declaration | Native sided signature has one's-complement x LEFT and signed y RIGHT. The existing first-quantization model supplies eight declared Toffoli calls. |
| Four-bit triangular index with eight-bit result | Native mu/nu/s through registers retain their widths. The existing contiguous-index resource owner supplies nineteen declared Toffoli calls. |
| Integer sign extension from eight to sixteen bits | Sided signed metadata retains x/y widths and directions. The direct graph contains one eight-target fanout. |
| Signed QFxp sign extension from eight to sixteen bits | The original additional declaration is retained with four fractional bits and signed=True on both sides. It executes in the original notebook despite the source annotation naming integer dtypes. Its source decomposition, scaled integer classical value and direct model are checked separately. |
| Sign truncation from sixteen to eight bits | The original correctly redundant prefix precondition is retained. Native sided signature and direct graph match the declared source. |
| Four signature displays and four one-level graphs | Existing register-table, generic graph and Graphviz owners render all original displayed examples. Original model leaf multiplicities are compared independently. |
| Native append and clean work | An explicit eight-qubit prefix and coherent spectator exercise all eight-bit inputs. Extension gives the correct sign prefix, truncation restores the complete complex state and zero prefix. |

The small example record and display helpers stay in the notebook. Register
metadata, fanout emission, Simulation and resource consumers retain their
existing owners. A test executes the original notebook unmodified and then runs
the native notebook with Qualtran/Cirq imports prohibited. All five declarations
are checked for register order, side, width and exact encoding, including both
fixed-point parameters. All model edges retain their original child identity
and multiplicity. Complete source stream, branch, classical and cost-consumer
evidence is reused from SIGNED_CONVERSION_FILE_AUDIT.md,
SIGN_EXTENSION_FILE_AUDIT.md and CONTIGUOUS_INDEX_FILE_AUDIT.md.

## Resource separation and remaining source boundaries

The eight-target fanout emits fifteen CX, zero T, measurements and hidden work.
Its extension/truncation round trip emits thirty CX. Eight explicit prefix
qubits are live during extension, and the spectator makes seventeen physical
wires in the notebook demonstration. The source one-level graphs each contain
one fanout child. Their coarse Clifford models and direct call counts remain
separate from the emitted stream.

The first two source roots have declared Toffoli models without executable root
decompositions. Their native execution and source admission gaps stay under
Q-0190/Q-0191 and Q-0193/Q-0194. The source index glyph's minus polynomial,
through-s classical callback mismatch, and SignTruncate's implicit QAny join
remain recorded in their complete file audits. The notebook uses their declared
signatures and models without claiming those defects have been repaired.
Generic symbolic/static conversion boundaries also remain open.

Notebook outputs are cleared. Rendered figures and batch logs are stored under
`ftqc-reference-env/g2-conversions-notebook-*` for review.

The focused original/native execution and concentrated regression passed
28 cases without skips. The existing IPython unwritable-directory fallback is
the only warning. Previously accepted broader primitive/branch evidence is
reused. Strict typing covers the test and all extracted notebook code.
Formatting, lint, Compiler scope, architecture, multilevel IR and protected API
checks passed. All eight signature/graph figures were visually inspected.

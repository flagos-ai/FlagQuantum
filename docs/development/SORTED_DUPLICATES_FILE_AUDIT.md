# Sorted duplicates complete source and test audit

Q-0201 and Q-0202 track the frozen implementation and original test file.
Their exact blobs are asserted by `test_sorted_duplicates_file.py`.

| Full source responsibility | Native owner and reused evidence |
| --- | --- |
| length/dtype/control fields and signature | `sorted_duplicates_signature` preserves signed/unsigned shaped xs, flag and final optional ctrl, including symbolic metadata |
| half-comparison dependency and graph | existing `sorted_duplicates_ir` follows adjacent half-LT, MCX, X/CNOT shell and reverse erasure, with exact gate/operand/feedback and alias-map comparisons |
| self adjoint | source returns itself, native append twice restores the valid sorted logical subspace on the preserving-order route |
| classical callback | Simulation `sorted_duplicates_classical_values`, with original shallow dictionary, ndarray aliases, sorting assertion, augmented XOR and ignored ctrl |
| specialized control | existing emitter positive-control input reuses shared MCX and controlled shell, both actual control inputs covered |
| three factories and DocSpec | concrete 4 by 3, symbolic word width and symbolic list/word widths all retain signatures, DocSpec contains only symbolic-width and concrete examples |
| original tests | original factory/count/classical tests execute unchanged, native suites independently verify actual gates, quantum amplitudes, scratch and raw measurement histories |

The untouched original test file reports 22 passed, 3 skipped and 1 xfailed,
with two existing unknown-slow-marker warnings. The fixture's skips/xfail
remain visible. Its classical composite tests pass because child callbacks
retain their logical contracts. They do not certify the original primitive
alias/erasure channel, whose counterexamples remain in the native gate suite.

The accepted 4 by 3 signed controlled circuit emits 48 T/TDG, 206 CX and 12
measurements on 32 physical wires. Its source model independently predicts
48 T equivalents, 12 measurements and 32 peak logical wires. Larger symbolic
root metadata is representable, while generic symbolic executable admission
and the original symbolic signed SignExtend width-query failure remain open.

Source aliases reorder [0,1] to [1,0]. Three-word [0,0,1] erasure can violate
its temporary-And target precondition. The uncontrolled singleton source
composite fails when X receives an empty controls port. Its native preserving
route is identity. The controlled raw callback ignores ctrl, unlike the quantum
control. Both interfaces preserve their documented distinct behavior. These
source/admission defects stay open under Q-0201 and Q-0202. No reference code
was repaired or replaced to claim equivalence.

Batch verification passed 718 concentrated native/reference cases with no skips or warnings.
Changed Python passed strict mypy, Black, Ruff, Compiler scope, architecture,
multilevel IR and protected API checks. Supplemental compare/exchange and
shared-control results are recorded in `g2-lists-file-control-dependencies.log`.

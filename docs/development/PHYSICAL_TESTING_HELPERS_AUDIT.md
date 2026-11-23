# Complete physical test-helper source mapping

## Interface and ownership before implementation

The next shared testing group is K-0207, `py/circuit/ops/op_testing_util.py`,
and K-0225, `py/circuit/subs/fuzz_test_util.py`, at locked Kickmix
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`. Both complete files were read and
their Git blob identities verified before implementation. The existing complete
main-binding receipt supplies the 63 frozen native-binding dependencies. The
extension version 0.1, Simulator and binary SHA256 are verified again by the
fixture generator. Reference sources are unchanged.

Verification owns reusable cases and assertions in `flagquantum.testing`.
Compiler retains operand identity, immutable operand arrays and physical circuit
records. Ecosystem owns Python numerical observations through the accepted
`PhysicalBasisSession`; Simulation owns the numerical state. Production code
does not import Testing. The helper owns no quantum work or compiler transform.

The complete operation-test helper is delivered as private
`_physical_operand_cases`. Its input families are native typed operands and
ordinary Boolean literals. Array cases use existing `bind_operand_array`.
Sequence position and hash-selected case order are retained. Native typed
Boolean identities represent the corresponding literal source values. The
64-lane mask bit k is shot k. X-basis classical IDs access the corresponding
ordinary classical storage, and X-basis literal values remain constants.

## Complete K-0207 definition map

| Source definition | Native implementation and behavior |
| --- | --- |
| U64_FALSE, U64_TRUE, U64_BIT0 through U64_BIT5 | Same eight exact 64-bit masks. Hexadecimal constants preserve the original binary words. |
| store64, read64 | Reuse the existing numerical session. Ordinary and X-basis literals return fixed masks. X-classical operands address ordinary bit storage. Quantum/classical writes and reads retain lane order. |
| x_cases_q, x_cases_bc, z_cases | Same scalar family order and literal polarities using native typed values. |
| x_list_cases | Same plain list, immutable arrays, optional X-basis families, tuple-hash mixed choices and scalar suffix order. |
| z_list_cases | Same quantum/classical/Boolean arrays, tuple-hash mixed choices and scalar suffix order. |
| broadcast_logic | First sequence sets width. Scalar values repeat to that width. Existing sequences retain identity. All sequence lengths must agree. Empty argument lists and zero-width sequences retain their original behavior. |

Thirty complete family configurations cover indices 0, 1 and 5, sizes -1, 0,
1, 3 and 8, with both include-X choices. The fixture also retains all eight
constants, 56 source mask read/write observations and eight broadcast expansion
observations. Every source scalar and sequence family maps to native values.
Fixtures regenerate byte-exactly. Invalid mismatched sequences raise AssertionError
and invalid nonscalar objects retain TypeError from their missing length.
Native masks then drive the existing CX numerical consumer over all 64 lanes.

The helper adds zero primitive gates, measurements and quantum workspace. The
consumer's actual CX records remain owned by the existing Compiler and numerical
execution paths. This test-data layer has analytical model None. Full complex,
all small measurement-history and work-zero consumers are reused in the
concentrated regression. They test the existing execution path rather than
claiming that a case generator implements quantum evolution.

## K-0225 source defects and native implementation

The complete original fuzz helper accumulates `all_inputs`, `all_expected` and
`all_actuals` across batches, but verifies each later chunk using the local
`shot` index starting at zero. It therefore repeatedly checks the first batch.

`g2-testing-helpers-intake-20261123.json` retains a direct locked-extension
witness. The circuit flips a named one-bit target. Input target is zero for
shots 1 through 256 and one thereafter. The reference Python specification
always predicts target one. At 257, 512 and 513 requested shots, respectively
1, 256 and 257 results intentionally disagree, yet the original helper returns
success. The matching 1-shot and 256-shot cases also pass. Source and extension
blobs are unchanged.

K-0225 remains open for recorded source defects. The complete native assertion
helper is implemented in `_physical_register_fuzz.py`. It validates the current
chunk, so the later mismatch is caught. This deliberate correction is separate
from source-equivalence claims. Original sources and the oracle are unchanged.

The source prefix also defines `neg()` in the separate exec locals dictionary,
but its body resolves `Z` in exec globals, where `Z` is absent. The original
`NEG` circuit with specification `neg()` raises `NameError` with message
`name 'Z' is not defined`. The native port preserves this source exception.
Direct top-level `Z(target)` specifications work in both implementations.
Existing comparator and lookup source defects, SIMD/performance gaps and
Proposals 116 through 119 retain their independent ledger entries.

## Complete register-fuzz definition and execution map

| Source definition or path | Native ownership and evidence |
| --- | --- |
| indentation_fix | Same CR/LF and space/tab state machine, conditional wrapping and exact phase prefix. Six focused inputs cover all state transitions. |
| Embedded Z and neg functions | Same exec namespace behavior, exact phase sign bookkeeping and original neg NameError. |
| Register discovery and unused qubits | Existing immutable register members, named-port refusal and typed unused quantum array. Original native IDs replace foreign repr in unnamed-port diagnostics. |
| Random/default input sampling | Same Python random draw order. Deterministic source fixtures reset the scalar RNG and native tests restore its caller-owned state afterward. |
| Custom sampling and context | Same key-set check, declaration-ordered writes even when sample keys are reversed, overlapping ports, context precedence, specification execution and missing-key/callback exceptions. |
| Chunk iteration | Existing persistent session is cleared and reused in chunks of at most 256. Native expected/input arrays belong to the current chunk, correcting the documented source indexing defect. |
| Actual outputs, phase and unused workspace | Existing session observations, exact sign checks, optional phase omission and independent unnamed-work-zero assertion. |
| Failure diagnostics | Source register/input/expected lines, phase lines, work lines and context ordering match byte-for-byte in the defined source cases. |
| Pytest traceback presentation | Native test helper retains ordinary diagnostic frames. Source-only traceback hiding metadata is retired. |

Twenty-three unchanged-source observations cover 0/1/3/256/257/513 shots,
quantum/classical mixed words, classical guards, alias write order, temporary
uncompute, HMR, exact phase, random/custom sampling, malformed keys, deleted
outputs and complete failure messages. All native results and diagnostics match
these observations, including the source neg NameError. The oracle regenerates
byte-exactly. Three additional native tests reject deliberately wrong later
chunks at 257, 512 and 513 shots that the unchanged source misses. A successful
513-shot path also verifies the final tail chunk.

The helper adds zero physical primitives, measurements or quantum workspace.
Circuits and actual counts remain owned by the existing compiler and simulator.
The layer has analytical model None. It asserts basis-register values, exact
phase and zero work, while the concentrated suite separately reruns accepted
full-complex all-measurement-branch consumers and normal append execution.

The complete K-0207 batch passes 1553 concentrated tests, 121 integration tests
and strict typing for three Python files. The new helper has 100 percent
statement and branch coverage. Black, Ruff, Verification scope, architecture,
multilevel IR and public API checks pass. Dependency findings retain the
identical 720-entry baseline.

The complete native register-fuzz batch passes 1587 concentrated tests,
121 integration tests and strict typing for three Python files. Its 34 focused
checks include all 23 frozen observations and the later-chunk fault injections.
The new module has 100 percent statement and branch coverage. Black, Ruff,
Verification scope, architecture, IR and protected API checks pass. The identical
720 dependency findings remain. K-0225 is still an open source parent with both
original defects explicitly recorded.

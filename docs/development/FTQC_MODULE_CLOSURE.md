# Fixed integration module closure inventory

The G2 scope remains the original **1,280 upstream files**: 945 Qualtran and
335 Kickmix files. It does not grow with test scenarios. See
[the frozen inventory](ftqc_source_inventory.csv), SHA256
`62970fe28e322a8d8f0f1921cf152d36a89eb96431cb474aa6116dce63f2c98e`. Original IDs, pinned revisions and Git blob identities are
preserved. Qualtran is v0.7.0; Kickmix is the separately pinned subtree revision.
The source-audit and definition-navigation columns are the initial audit snapshot,
not live acceptance decisions or an assertion that unlisted behavior is absent.
Unavailable reference assets remain open; differing blobs cannot certify an
unchanged upstream comparison. G3 integration acceptance remains separate.

## Module closure criteria

1. Audit the whole source file: functions, classes, methods, properties, constants,
   exports, dependencies, examples, tests and repository assets. AST names are
   navigation only. Every capability needs a native target or a justified asset
   disposition; a child checkbox never automatically closes a parent.
2. Establish native ownership, interfaces, bit order and work ownership before
   implementation. Reuse existing components and preserve source copyright.
   Do not mirror the upstream package or create another IR.
3. Compare unchanged upstream behavior, including failure paths. For emitted
   circuits, compare ordered operations, operands, feedback, exact complex
   amplitudes, small exhaustive measurement branches and clean workspace.
   Numerical-only modules instead verify values, precision, warnings and state.
4. Report measured primitives, source cost models and runtime statistics
   separately. Unimplemented variants, source defects or protected-contract gaps
   keep the file open. Do not silently repair the reference.
5. Commit complete modules independently; run concentrated regression once at
   batch end. Publish locally verified work to #1084 without waiting for CI.
   Do not merge or start G3. Retain baseline failures explicitly.

## Frozen execution order

| Order | Original parent groups | Completion unit |
| --- | --- | --- |
| 1 | Q-0757 / Q-0758 | Both matrix functions and the original test file |
| 2 | Remaining Q07 linalg/simulation | Whole numerical modules, filling actual gaps |
| 3 | Q01 / Q04 | Shared register/graph contracts and resource modules |
| 4 | Q02 / K01-K03 | Arithmetic, field, table, swap and synthesis modules |
| 5 | Q03 / Q05 | Algorithms, correction and cost models using closed dependencies |
| 6 | Q06 / Q07 drawing / K04-K06 | Formats, simulation, interoperability and tools |
| 7 | Q08 / K07 | Packaging, build, docs, licensing and repository asset dispositions |

Pause further GQSP scenario expansion while closing this backlog. Newly found
behavior belongs to an existing parent ID; any scope change needs an explicit
record. The Chinese workspace ledger remains the live status source and its
phase totals are generated from parent decisions, not test count.

## Closed source modules

[Matrix numerics](MATRIX_NUMERICS.md) closes Q-0757 and Q-0758. The parent ledger
now also closes [polynomial evaluation and matrix assertions](POLYNOMIAL_ASSERTION_CLOSURE.md)
(Q-0762, Q-0763, Q-0767), for **5 closed files and 1,275 open files**. These file
counts are not a functional completion percentage or an effort estimate. Some open modules
already have accepted partial capabilities; many require full audit.


## Audited source with an inherited open defect

The [QSP inspection audit](QSP_INSPECTION_CLOSURE.md) covers all five functions
in Q-0766, including default FFT, NumPy scalar precision and failure-domain
inputs. Its finite-grid assertion still misses a source counterexample with
between-grid magnitude 1.06. Per the frozen criteria, the parent stays open
rather than treating interface completion as repair of that upstream defect.
No additional scenario or parent ID was added to the completion denominator.

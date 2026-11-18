# List package export responsibility

Q-0200 covers the entire frozen `arithmetic/lists/__init__.py` blob
`7ed84457f5734a6e2d76fd6d883032fc6f815afa`. The initializer only re-exports
HasDuplicates, SortInPlace and SymmetricDifference in that order. The export
audit verifies the complete AST and definition identity without changing source.

The native owners are respectively `sorting/sorted_duplicates.py`,
`sorting/in_place_sort.py` and `sorting/symmetric_difference.py` under Compiler
registers. The raw duplicate callback belongs to Simulation. They share existing
comparison, control, bitwise and population-count components. Importing this
initializer declares no additional operation, primitive, work or measurement.

Existing append, coherent-state, branch, inverse and work-cleanup evidence stays
with each child module. Their source defects and generic admission gaps stay
open under Q-0201, Q-0204 and Q-0205. This package audit does not close them.

Batch verification passed 718 concentrated native/reference cases with no skips or warnings.
Changed Python passed strict mypy, Black, Ruff, Compiler scope, architecture,
multilevel IR and protected API checks. Supplemental compare/exchange and
shared-control results are recorded in `g2-lists-file-control-dependencies.log`.

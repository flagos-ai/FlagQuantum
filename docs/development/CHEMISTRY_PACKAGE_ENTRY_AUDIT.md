# Chemistry package entry disposition

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Source files retain their original
bytes. The inventory blob hashes were recomputed before disposition.

| Inventory | Original file | Blob | Complete disposition |
| --- | --- | --- | --- |
| Q-0334 | `chemistry/__init__.py` | `ce76a8d10a5909daeb7b1113c8cc7f26cc31b410` | Accepted. License comments only, zero AST statements and zero runtime dependencies. Chemistry algorithms belong to the existing Algorithms domain. |
| Q-0339 | `chemistry/df/__init__.py` | `ce76a8d10a5909daeb7b1113c8cc7f26cc31b410` | Accepted. License comments only, zero AST statements and zero runtime dependencies. Double-factorization composition belongs to the existing primitive modules. |
| Q-0347 | `chemistry/hubbard_model/__init__.py` | `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` | Accepted. Empty source with zero runtime dependencies. Hubbard composition belongs to existing primitive modules. |
| Q-0348 | `chemistry/hubbard_model/qubitization/__init__.py` | `89f9efc8bf19621827a31281d49e6b8d830bb5fd` | Accepted. Physical explanation and three imports expose five declarations, each with an owned consumer and actual package constructor-library/append/model evidence. Source-body defects remain with Q-0350/Q-0353/Q-0354. |

The first three files introduce no callable, allocation, serialization or
execution behavior. Their complete migration consists of documenting the native
owners and retaining source provenance. They require zero primitive operations,
zero measurements and zero work wires. There is no source resource model in these
files. Their acceptance makes no claim about their child algorithm files.

## Native ownership and remaining direct exports

`flagquantum/algorithms/primitives` owns domain composition. Register traversal,
conjunction, feedback and workspace recipes remain Compiler responsibilities.
Ecosystem owns foreign-object translation. Simulation executes native IR.
The native contributor guide provides the existing append workflow and focused
checks. Source license records remain in the repository's existing provenance
and the headers of adapted implementations.

| Q-0348 declaration | Existing native owner | Remaining scope |
| --- | --- | --- |
| `PrepareHubbard` | `hubbard_prepare.PrepareHubbard` | Actual package-export consumer and retained Q-0350 source boundaries. |
| `SelectHubbard` | `hubbard_select.SelectHubbard` | Actual package-export constructor-library/append/model consumer accepted. Q-0353 source-body boundaries remain. |
| `HubbardMajorannaOperator` | `hubbard_majorana.HubbardMajorana` with shared bounded traversal | Complete source builder, exact child generator, named constructor-library/append/model, complex branches and clean-work handoff accepted. |
| `HubbardSpinUpZ` | `hubbard_spin_up.HubbardSpinUpZ` with shared predicate traversal | Complete source builder, exact child generator, named constructor-library/append/model, complex branches and clean-work handoff accepted. |
| `get_walk_operator_for_hubbard_model` | `hubbard_walk.HubbardWalk.from_lattice` | Q-0355/Q-0356 already accepted. Reuse the original factory, reflection, full complex amplitudes and inverse evidence in HUBBARD_WALK_FILE_AUDIT.md. |

The original package explains the Jordan-Wigner Hubbard Hamiltonian and the
four PREPARE/SELECT sectors. Existing native notebooks and
`HUBBARD_QUBITIZATION_FILE_AUDIT.md` cover those physics conventions. The
five exported declarations now have complete native ownership and actual named
package consumers. Standalone components have their own append paths and
source-builder, primitive, full-complex and work evidence. Q-0348 closes this
initializer and its exports. Source-body defects and dependency-specific
classification differences remain recorded in their original source parents.

## Evidence

`g2-chemistry-package-entries-intake.json` records each original byte length,
recomputed blob, AST statement count and exact imported declaration list.
Q-0334 and Q-0339 each contain 585 bytes. Q-0347 is empty. Q-0348 contains
2402 bytes, one module docstring and three import statements.

Existing chemistry and Hubbard regression evidence is reused for the unchanged
native owners. The package dispositions change documentation only. The same focused regression
found six side-one Hubbard SELECT failures in an existing unary traversal caller.
A separate module commit supplies the explicit zero-selection leaf using its
existing predicate and spin-up CZ recipe. Existing tests exercise both control
polarities, full complex amplitudes, feedback branches and work cleanup. See
HUBBARD_QUBITIZATION_FILE_AUDIT.md for the independent repair. Production
signatures and source implementations retain their existing contracts. The pending protected proposals 116, 117 and 118 remain separate.


## Completed Q-0348 consumer closure

`test_hubbard_package_consumers.py` consumes every original package export,
round-trips its constructor library and original resolver, checks the captured
signature/model and binds native append with SDK imports blocked. PREPARE,
SELECT and walk evidence is reused, while the two previously internal children
now have independent domain owners. The original module docstring, three imports
and five export targets retain their exact blob and AST disposition. No
source-branded native package or duplicate IR was introduced.

`HUBBARD_QUBITIZATION_FILE_AUDIT.md` maps all 39 SELECT source definitions and
three documentation specifications. Its new receipts include 30 exactly unchanged
native SELECT streams, physical measurements/work and independent analytical
resource rows. The actual original unit suite passes with Cirq 1.6.1 and retains
the eight CY classifier failures on 1.7.0. Q-0353/Q-0354 remain open for the
original accepted scope. Closing this initializer does not claim their closure.

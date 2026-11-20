# Chemistry package entry disposition

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Source files retain their original
bytes. The inventory blob hashes were recomputed before disposition.

| Inventory | Original file | Blob | Complete disposition |
| --- | --- | --- | --- |
| Q-0334 | `chemistry/__init__.py` | `ce76a8d10a5909daeb7b1113c8cc7f26cc31b410` | Accepted. License comments only, zero AST statements and zero runtime dependencies. Chemistry algorithms belong to the existing Algorithms domain. |
| Q-0339 | `chemistry/df/__init__.py` | `ce76a8d10a5909daeb7b1113c8cc7f26cc31b410` | Accepted. License comments only, zero AST statements and zero runtime dependencies. Double-factorization composition belongs to the existing primitive modules. |
| Q-0347 | `chemistry/hubbard_model/__init__.py` | `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` | Accepted. Empty source with zero runtime dependencies. Hubbard composition belongs to existing primitive modules. |
| Q-0348 | `chemistry/hubbard_model/qubitization/__init__.py` | `89f9efc8bf19621827a31281d49e6b8d830bb5fd` | Open. Physical explanation and three imports expose five declarations. The two standalone child interfaces still require a complete owner and actual foreign-consumer audit. |

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
| `SelectHubbard` | `hubbard_select.SelectHubbard` | Actual package-export consumer and retained Q-0353 source boundaries. |
| `HubbardMajorannaOperator` | `bounded_majorana_select` recipe and Hubbard analytical formulas | Standalone typed component ownership and actual source-object handoff. |
| `HubbardSpinUpZ` | `hubbard_select` internal spin stage and Hubbard analytical formulas | Standalone typed component ownership and actual source-object handoff. |
| `get_walk_operator_for_hubbard_model` | `hubbard_walk.HubbardWalk.from_lattice` | Q-0355/Q-0356 already accepted. Reuse the original factory, reflection, full complex amplitudes and inverse evidence in HUBBARD_WALK_FILE_AUDIT.md. |

The original package explains the Jordan-Wigner Hubbard Hamiltonian and the
four PREPARE/SELECT sectors. Existing native notebooks and
`HUBBARD_QUBITIZATION_FILE_AUDIT.md` cover those physics conventions. Exact
whole-component coverage of every exported declaration remains necessary before
Q-0348 acceptance. In particular, the existing internal recipes do not establish
standalone component acceptance by themselves.

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

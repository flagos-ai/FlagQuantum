# Complete Kikuchi and k-XOR original test acceptance

The four complete original test files are frozen at Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`. Their implementations, tests and
autotester are unchanged. Existing acceptance of the four notebooks is reused;
the current non-notebook run deliberately excludes their four notebook tests.

| Parent | Complete test file | Every top-level responsibility |
|---|---|---|
| Q-0603 | kikuchi_adjacency_list_test.py | Four example protocols; exact symbolic column-query cost and big-O; notebook Q-0601 |
| Q-0606 | kikuchi_adjacency_matrix_test.py | Two example protocols; controlled QROM-only graph difference; concrete and symbolic costs and big-O; notebook Q-0604 |
| Q-0609 | kikuchi_block_encoding_test.py | Two example protocols; two controlled swaps only; concrete and symbolic cost queries; notebook Q-0607 |
| Q-0618 | load_kxor_instance_test.py | Two example protocols; every concrete scope lookup; concrete and symbolic costs; notebook Q-0616 |

The existing source-file tests hash both implementation and test blobs, compare
all constructor fields, register widths, shapes, sides and named child bindings,
and match all seven concrete/symbolic QEC categories. These exact equalities also
preserve the original total-T big-O assertions. Original ANY fields are further
checked against actual source values rather than weakened to native wildcards.

## Literal controlled sparse graph

KikuchiHamiltonian retains the Hermitian sparse call inventory before folding
its resources: diffusion, column, entry, system swap, flag swap, inverse entry,
inverse column and inverse diffusion. The existing resource method consumes this
same inventory and existing dependency models. Entry and column records include
the actual instance; diffusion includes width and sparsity. Distinct Hamiltonian
queries therefore do not coalesce merely because their role names agree.

Concrete and symbolic comparisons map every call to the untouched source
build_call_graph, including inverse direction and Always control bypass. Only
the system and flag swaps gain a control. Width-one swaps combine to multiplicity
two, as in the source Counter. Non-boolean control values remain rejected.
The original concrete widths are 32 and one; they are no longer represented
only by their aggregate width 33. The matrix-entry test separately retains the
original depth-two graph difference: only amplitude-QROM num_controls changes
from zero to one, with all other QROM fields and call multiplicity preserved.

## Actual scope loading and measurement branches

The original concrete instance contains five batched scopes. Native
circuit.append(op) emits the existing literal measured-QROM path. Every valid
address is checked with zero and all-one target words and both spectator values.
The expected packed word comes from the unchanged source call_classically,
independently checked against source scope_as_int. The exact And channel
certificate checks each literal compute/erase block's controls and clean target,
preserves the whole output word, and proves identical normalized action in all
eight measurement histories. Linearity preserves arbitrary complex amplitudes
on these inputs; the two target patterns are not claimed to exhaust all words.
Applying the same lookup again restores each whole input, including clean work.

Existing small-instance evidence independently compares every primitive,
operand and feedback slot with the untouched decomposition. All two forward
histories and four joined lookup/erase histories compare complete unnormalized
complex statevectors, including coherent spectators and zero work. It emits
27 primitives, four T-family gates, one measurement and one work wire. The
original five-scope example emits 86 primitives, three measurements and two work
wires. Its model independently reports three And and three measurements;
symbolic cost remains m-2 And, k*m*ceil(log2(n))+m-2 Clifford and m-2 measurements.

## Raw original protocol outcomes

All fifty example/mode pairs are invoked independently before pytest's skip or
xfail policy. This prevents a skipped early mode from hiding later observations.

| Example | Make | Decompose | Counts | Serialize | Strict types |
|---|---|---|---|---|---|
| col_kth_nz | PASS | MISSING | UNVERIFIED | FAIL | NA |
| col_kth_nz_symb | PASS | NA | UNVERIFIED | FAIL | NA |
| kikuchi_nonzero_index | PASS | PASS | FAIL | FAIL | PASS |
| kikuchi_nonzero_index_symb | PASS | NA | UNVERIFIED | FAIL | NA |
| kikuchi_matrix_entry | PASS | MISSING | UNVERIFIED | FAIL | NA |
| kikuchi_matrix_entry_symb | PASS | MISSING | UNVERIFIED | FAIL | NA |
| kikuchi_matrix | PASS | PASS | UNVERIFIED | FAIL | PASS |
| kikuchi_matrix_symb | PASS | PASS | UNVERIFIED | FAIL | PASS |
| load_scopes | PASS | PASS | FAIL | FAIL | PASS |
| load_scopes_symb | PASS | NA | UNVERIFIED | FAIL | NA |

This is 18 PASS, 12 FAIL, nine NA, eight UNVERIFIED and three MISSING.
All ten serializations fail on unsupported KXorInstance. Nonzero-index counts
disagree on explicit allocation/free and zero-state/effect bookkeeping; loader
counts disagree on Partition bookkeeping. Manual-only or decomposition-only
counts remain UNVERIFIED. Symbolic shapes and missing query bodies retain their
original decomposition/type boundaries. No failed protocol becomes a positive
native round-trip, and no original count annotation is silently repaired.

## Complete-file boundary and evidence

The four test parents close their actual assertions and recorded source protocol
outcomes. Implementation parents Q-0602/Q-0605/Q-0608/Q-0617 remain open for
physical query/RHS bodies, root return/free obligations and packed scalar/array
binding. In the column wrapper both native and source now raise the same
zero-dimensional IndexError; an obsolete earlier storage rejection was removed
from its audit. Typed identity-child substitution proves wiring only.
The source controlled-adjoint matrix cost still fails on atomic C[And].
There is no full Hamiltonian complex-amplitude or work-zero claim.

Receipts use `g2-kikuchi-four-*-20261125`. The unchanged four-file non-notebook
suite records 27 passes, 20 original skips, 12 original xfails and four notebook
deselections. The raw protocol JSON preserves diagnostics separately from test
policy. Final regression and strict-check results are synchronized into the
Markdown ledger after publication. Existing notebook and numerical evidence is
reused rather than relabeled as a new physical implementation.

The final batch passes 274 concentrated cases, 26 Cirq 1.6 cases and 145 local
integration cases. Strict mypy passes all three touched Python files. Black,
Ruff, Algorithms/Integration scope, local links, repository language,
architecture, multi-level IR and protected API checks pass. The dependency scan
retains the same 720 baseline findings after ignoring moved line numbers; it is
not a globally clean dependency result. Kickmix 0.1's native extension and
Simulator are present. The first regression exposed a missing explicit precision
in the new one-bit test fixture; specifying entry_bitsize=3 fixed that fixture
without changing constructor admission. The successful complete rerun is the
acceptance result.

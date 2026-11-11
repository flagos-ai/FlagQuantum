# Permutation cycle source audit

Compiler owns cycle discovery as a classical program transformation, separate
from register validation, physical-wire allocation and numerical execution.
The reference is unchanged Qualtran v0.7.0 `linalg/permutation.py`, revision
`bcb17bb25360ed68ce3bc084645db3e929799389`; copyright remains in the native owner.

## Complete source surface and ownership

| Source surface | Native disposition |
| --- | --- |
| `CycleT` type alias | Existing `tuple[int, ...]` annotations; no extra public type |
| `decompose_permutation_into_cycles` | `compiler/registers/sorting/cycle_decomposition.py::permutation_cycles` |
| `decompose_permutation_map_into_cycles` | Same owner, `partial_permutation_cycles` |
| Both original test functions | Run unchanged, then run with only helper bindings replaced |

There are no other source definitions or assets in these two files. The existing
validated `source_partial_permutation_cycles` consumer now delegates its traversal
to the shared owner. Its range/injectivity validation remains unchanged. The raw
helper deliberately performs no such validation. No Core contract, public export,
manifest, schema or IR is changed. This replaces duplicated traversal rather than
introducing an additional authority.

Dense discovery retains `DiGraph(enumerate(values))`, `simple_cycles`, the
singleton filter, node insertion order and generator laziness. No cycle sorting,
rotation normalization or input repair is added. Partial traversal retains the
source seen-set/path order, stopping on a missing edge or previously visited node.
Both accept classical indices; there is no physical bit order at this boundary.
The downstream emitter retains big-endian targets, caller-owned clean marker and
carry wires, and reusable classical feedback slots. Users continue to append an
operation: see `examples/algorithms/register_permutation.py`.

## Optional graph dependency review

NetworkX 3.6.1 is the reference graph provider, released under BSD-3-Clause. Its
installed source and metadata hashes accompany the three unchanged source Git
blobs in `g2-permutation-source-audit-source.json` workspace evidence. No package
code is vendored, installed automatically or added to the protected manifest.
Compiler owns the optional graph dependency; external graph objects remain inside
the discovery function. Import happens at first generator iteration, before input
consumption. Missing NetworkX raises the original `ModuleNotFoundError`; partial
traversal requires no NetworkX. No fallback is used.

The existing provider call is the replacement boundary: a recorded
`simple_cycles` provider is substituted without changing either source or native
consumer, verifying graph edges, output order and singleton filtering. A future
replacement must independently preserve these semantics and failures; it must
not silently introduce a different cycle order. This audit claims order parity
for the recorded provider version, not across every NetworkX version. The
optional provider can be removed only after equivalent discovery is verified;
there is no second graph implementation or registry in this batch.

## Verification and separate resource accounts

The new audit has 23 passing tests. Concentrated regression has 223 passes and
six reference Autoray numerical warnings, with no skips. Both original tests
pass unchanged and both also pass with native bindings. Exhaustive dense
permutations through size six and all four-node partial functions in forward and
reverse key order match source output. Invalid dense inputs, lazy import failures,
provider substitution and mutation after first yield retain source behavior.

Three append consumers compare every emitted instruction, operands and feedback
against the original arithmetic `Permutation` decomposition: a three-cycle, two
two-cycles, and a partial chain. They check all complex amplitudes with an
entangled spectator, and all 192 forward/inverse measurement-branch evaluations.
Each branch retains its unnormalized probability amplitude; both work bits are
zero within absolute tolerance 1e-12. Exact floating-point zero is not claimed.

| Consumer | Measured primitive instructions | Original/native model |
| --- | --- | --- |
| Three-cycle / partial chain | X24, H12, T8, Tdg8, CX32, S4, measurement4, conditional CZ4, reset4 | AND4, Clifford12, measurement4; equivalent T16; peak4 |
| Two two-cycles | X28, H18, T12, Tdg12, CX46, S6, measurement6, conditional CZ6, reset6 | AND6, Clifford16, measurement6; equivalent T24; peak4 |

Each consumer owns two clean work qubits and one reusable classical slot. The
spectator makes the executed state five qubits; it is excluded from the model's
four-qubit peak. Model Clifford/AND categories are not primitive gate counts.
Discovery itself emits zero gates and allocates no quantum/classical work.

Strict typing checks all three changed Python files; Black, Ruff, Compiler scope,
architecture, internal IR and API baseline checks pass. Coverage of the discovery
owner is 100% (21 statements/14 branches); the existing emitter is 95%
(27 statements/12 branches), combined 97%. The uncovered positive-size refusal
is unchanged. This scope is separate from repository coverage.
Default suite: 7 failed, 21,507 passed, 300 skipped, 1,037 warnings in
174.39 seconds, stopping at the existing maxfail bound. All seven failure IDs
match the prior baseline; no new failure is observed. This is not a green
default suite. The complete comparison is recorded in workspace evidence.

## Inherited open source defects

For `{0: 1, 2: 0}`, the original partial traversal yields only `(0, 1)` and loses
the required `2 -> 0` edge. Reversing insertion order yields `(2, 0, 1)`. The raw
native helper and existing validated wrapper preserve this source defect. The
source docstring also passes unsupported keyword `N=10`, which raises TypeError;
it is not a working example and is not copied as a native usage instruction.
Neither issue is repaired in the reference. Under the frozen closure criteria,
Q-0759 remains open despite full helper call-chain parity. Q-0760, the original
two-test file, closes. No new scenario IDs or parent denominator are introduced.

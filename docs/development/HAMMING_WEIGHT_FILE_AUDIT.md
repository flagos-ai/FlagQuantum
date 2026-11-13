# Hamming whole-file audit

Batch G2-FILE-HAMMING audits Q-0198 and Q-0199 from Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. It adds the missing complete original
test mapping and reuses accepted implementation evidence. Q-0199 is accepted as
a complete test file. Q-0198 remains open for its recorded empty-junk boundary.

## Complete source disposition

`qualtran/bloqs/arithmetic/hamming_weight.py` contains one class, one constructor
field, four cached properties and four methods. It declares no separate exports,
module constants, examples or additional functions. Its source header and paper
reference are retained through the native attribution and existing documentation.

| Source definition | Native owner and interface | Evidence or remaining boundary |
|---|---|---|
| `HammingWeightCompute` | bits-domain Hamming emitter, requirements, resource identity and model | `8477975b`, `0121b3ac`, `b780407a`, `5acb209b`; callers bind explicit words through `circuit.append(op)` |
| `bitsize` | `hamming_weight_requirements(width)` and existing `ArithmeticCall` width | Concrete gates and unchanged Python/Sympy width objects have separate admission |
| `signature` | Explicit source/junk/output words, requirements and connected width owner | THRU input and RIGHT logical outputs, inverse consumption, big-endian binding and all port connections are covered; source empty junk remains open |
| `junk_bitsize` | `hamming_weight_resource_widths` and concrete requirements | Python population count and symbolic lower-bound behavior covered by existing model tests |
| `out_bitsize` | Same width declarations with shared `bit_length` | Concrete and symbolic output widths covered by existing model tests |
| `bit_count_of_bitsize` | Shared resource-width calculation | Python and Sympy integer distinction retained; symbolic literals preserve source diagnostic counts |
| `_three_to_two_adder` | `bit_compression_structure.compression_components` and `compress_three_bits_ir` | Shared six logical components, ordered primitive expansion and measured AND erase |
| `_decompose_using_three_to_two_adders` | `hamming_weight_structure.hamming_weight_stages` | Source little-endian traversal and reversed-junk pop, shared by emission and width graph |
| `decompose_from_registers` | `hamming_weight_ir(source, junk, output, ...)` | Bound big-endian words, ordered gates/feedback, full complex amplitudes and clean inverse work; existing native validation remains explicit |
| `build_call_graph` | `arithmetic_callees`, QEC query and Hamming resource model | Ordered AND/CNOT declarations, modern categories, historical budget and symbolic diagnostics |

The existing immutable arithmetic identity replaces source attrs construction.
Native instructions and explicit words own compilation. Cirq and NumPy objects
stay in reference tests. `And` and CNOT use the existing conjunction and XOR
owners. Generic resource adjoint and connection scheduling remain with their
existing owners. No additional dependency implementation was required for this
batch's test-file closure.

## Complete original test mapping

`qualtran/bloqs/arithmetic/hamming_weight_test.py` has exactly two functions and
five parameter cases. The new test fixture verifies both source Git blobs and
the complete set of top-level test definitions against the frozen inventory.

| Original function | Full original parameter set | Native assertion mapping |
|---|---|---|
| `test_hamming_weight_t_complexity` | 3, 4, 5 bits | Shared QEC query with historical budget equals every original T/Clifford/rotation field and its formula; expanded primitive counts independently match |
| `test_hamming_weight_compute` | 3, 4 bits | Every basis input through normal `Circuit.append` yields the original formatted count assertion; native inverse restores every input for every forced measurement history, including unnormalized complex amplitude and clean logical outputs |

All five unchanged original test cases also run directly from the frozen test
module. Their `assert_valid_bloq_decomposition` calls remain source graph checks.
Native acceptance reuses complete weighted graph comparisons and successful
normal-entry emission/execution, with explicit register ownership. The original
Cirq simulator and helper remain reference infrastructure. Native execution uses
FlagQuantum Circuit and the existing forced-branch helper.

The original compute test checks count bits and a complete compute/inverse
round trip. Existing `test_hamming_weight.py` adds full coherent complex inputs
for widths one through five, every measured erasure branch and exact gate/
feedback comparisons over nine widths. Those accepted assertions are reused.

## Reused evidence and resource accounts

| Existing evidence | Reused acceptance |
|---|---|
| `8477975b` | Ordered compute/erase source streams, complete complex amplitudes, small exhaustive measurement branches, logical cleanup and runnable append example |
| `0121b3ac` | Complete direct calls, symbolic widths, both QEC profiles, phasing model replacement and diagnostic source boundaries |
| `b780407a` / `5acb209b` | Recursive width cache/order, seeded child peaks, fourteen weighted connection graphs, DOT output and forward/inverse ownership |

| Input width | Expanded forward primitives | Source modern category model | Source historical model |
|---|---|---|---|
| 3 | CX13, H2, T2, TDG2, S1 | AND1, Clifford7 | T4, Clifford16, rotations0 |
| 4 | CX34, H6, T6, TDG6, S3 | AND3, Clifford16 | T12, Clifford43, rotations0 |
| 5 | CX35, H6, T6, TDG6, S3 | AND3, Clifford17 | T12, Clifford44, rotations0 |

The new batch adds test coverage and audit disposition. Production gate emission,
quantum scratch and measurement requirements receive zero additions.

## Whole-file outcomes and remaining work

Q-0199 closes because every original function, parameter and assertion has an
explicit native mapping and all five original cases pass unchanged. Imports and
fixtures are accounted for as reference infrastructure. This acceptance covers
the original test file's three-to-five-bit scope.

Q-0198 stays open. Its existing Q-0198.02 records that the frozen one-bit generic
Bloq adjoint cannot bind the empty `QAny(0)` junk. Native one-bit compute/erase is
already supported, while successful upstream empty-junk Bloq conversion parity
remains unavailable. Existing failed decomposition and width-query witnesses are
retained. Compiler bits and generic register/adjoint owners retain the boundary.
Closure requires accepted coverage of that representation boundary under the
existing integration criteria. This batch preserves its previous failure evidence.

General dtype-specific width-cache seeds and dirty/array allocation belong to
the existing graph/resource parent files. They remain in the centralized ledger
with their own closure requirements. The next complete-file candidate is
Q-0174 addition and its original tests, using existing native arithmetic modules.

## Batch validation

All ten original/native mapping cases pass. Concentrated batch regression passes
422 tests in 8.74 seconds, with zero skips. It includes the existing Hamming
primitive, complex-branch, resource, width, phasing and gradient evidence plus
shared historical budget and QEC-category consumers. The previous batch's
8,646-test regression and 87 real-kernel/CPU integration acceptance are reused
for unchanged production paths. Strict typing, Black and Ruff pass for the new
test module. Team scope, architecture, internal IR and protected API checks pass.
The normal append example verifies all input/count complex amplitudes and six
clean temporary wires. Its full copy-count path emits CX53, H9, T6, TDG6, S3,
measurement3, feedback CZ3 and reset3, while its source model reports T12 and
measurement3 separately.

The initial standalone contract-check subprocess imported an installed package
instead of this checkout. The shared repository wrapper corrected import
precedence and the contract/API checks then passed. Reference files and
production interfaces retain their accepted versions.

# Sparse Walk bank-unitary dependency closure

Reference: unchanged Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, Google LLC, Apache-2.0.
The inherited source fixtures verify the clean-lookup, sparse PREPARE and Walk
source/test blobs, Qualtran 0.7.0 and the real Kickmix 0.1 Simulator.

## Responsibilities and interfaces

`lookup/qroam_bank_unitary_component.py` constructs a physical bank extension
using the existing selection partition, batch XOR and relative-phase routing
owners. Selected cells and retained dataset-major cells are caller-owned
MSB-first THRU words. Forward loads by XOR then routes; inverse reverses the
dataset/routing order before XORing again. Only QROM predicate scratch is
private and measurement-erased. Bank contents are never measured or freed.
This is a constructed unitary extension, not a side relabeling of the clean
lookup allocation graph or a replacement for its model-only source inverse.

The existing sparse PREPARE composition accepts that explicit dependency pair
and threads its bank inputs through the same eighteen source stages. Its old
RIGHT isometry and measured matching-image inverse remain available through
their existing owner. The bank-unitary pair supplies a different valid
execution domain: arbitrary bank inputs. No new Core operation or public API
is introduced, and numerical execution stays outside Compiler.

Ecosystem `_sparse_walk_component_library.py` captures the actual sparse Walk
and immediately converts its table, signatures and dependency pairs to native
records. It reuses the exact source SELECT/block, scalar carrier, prepared
reflection and Walk composition owners. Actual set-derived source port order
is retained. Reflection banks have independent `signal_` names; public SELECT
orbital words and reflection orbital banks are distinct storage.

The existing `WalkContext` explicitly records both the original narrow source
signature and the extended native signature. The source factory still refuses
to pretend the latter has the former ABI. The context pair can be bound and
appended after Qualtran/Cirq imports are unavailable. Existing Algorithms
`SparseChemistryWalk`, whose reflection covers the complete prepared signal,
retains its separate mathematical behavior. This adapter preserves the
original narrower selection projector instead of silently changing that API.

## Numerical and source evidence

Seven lookup fixtures include blocked/unblocked, one-/two-axis, multiple
datasets, multiple target bits and external controls. They compare every
dirty-bank computational column with the original extracted gate actions
where that source root decomposes, prove the exact joint inverse on every
column, enumerate every small raw history on a complex coherent witness,
and separately certify predicate erasure over arbitrary bank inputs. Testing
physical padded address columns does not expand the source SDK's BQUInt
admission contract.

For full-table single-block examples, the unchanged source root drops its
selection output. Both the one-axis/two-dataset and two-axis examples retain
explicit BloqError assertions. Their native extension is compared against the
already accepted physical batch/routing owner, not falsely reported as a
successful comparison to that source root.

Four-row one-bit sparse PREPARE fixtures enumerate all 64 joint histories for
unblocked banks and all four for blocked banks, using arbitrary complex bank
and selection witnesses. Entire joint states and private work are restored.
The reflection is independently tested by conjugating it into its known
selection-zero diagonal action, including both control polarities and both
global phases +1 and -1. Unblocked compositions also receive the all-history
And-support certificate; blocked relative-phase routing has separate full
small-operator evidence and representative reflection histories.

Small complete sparse Walk contexts, ordinary and positive-controlled, retain
source block calls/operands and restore the full complex joint input through
their exact inverse. Forward and inverse remain native after SDK capture.
The actual original 65-row, twelve-keep-bit, eight-spin-orbital example has
ordinary/positive forward/inverse binding and resource evidence. Its large
numerical output operator is not claimed here. The outer sparse SELECT
control remains the original bystander defect, rather than a corrected
mathematical control. Full Walk histories are not exhaustively enumerated.

Use the locked reference environment for the focused replacement scenario:

```bash
python -m pytest tests/unit/test_qroam_bank_unitary_component.py \
  tests/unit/test_sparse_preparation_bank_unitary.py \
  tests/unit/test_sparse_walk_component_library.py -q
```

Batch receipts live under `g2-bank-unitary-*`. The unchanged reference suite
uses the existing marker-registering wrapper and existing OpenFermion location;
the initial missing marker/dependency diagnostics are retained. Neither source
tests nor assertions are modified to repair the environment.

Acceptance has 261 concentrated passes, 145 local integration passes and
strict typing for six files. Format, both team scopes, language, architecture,
IR/API and the identical 720-entry dependency baseline pass independently.
The unchanged original Walk suite has fourteen passes, three existing xfails
and four slow/notebook deselections. Original PREPARE/SELECT and G1 primitive
receipts are reused; this is not a whole-repository or GPU release claim.

## Actual primitives and model scope

Twelve directional/control rows are recorded in
`g2-bank-unitary-resources-20261125.json`. Actual gates, retained logical bits,
private lease peaks and source model estimates are distinct accounts.

| Native ordinary case | T/T-dagger | CX | Measurements | Logical / private peak / physical | Original source model |
| --- | ---: | ---: | ---: | --- | --- |
| Four rows, unblocked | 222 | 398 | 24 | 32 / 9 / 41 | 16 CSwap, 22 And, 202 Clifford, 22 measurements, peak 34 |
| Four rows, blocked | 310 | 500 | 20 | 45 / 9 / 54 | 52 T, 16 CSwap, 20 And, 471 Clifford, 20 measurements, peak 47 |
| Original 65 rows, twelve keep bits | 1286 | 4644 | 198 | 107 / 35 / 142 | 128 T, 13 Toffoli, 34 CSwap, 167 And, 2918 Clifford, 4 rotations, 167 measurements, peak 142 |

Positive native contexts have respectively 33, 46 and 108 logical bits, with
physical peaks 41, 54 and 143. Every private lease is returned. Generic
controlled-adjoint QEC model evaluation fails in the unchanged source at
`C[And-dagger]`; the separately supplied native inverse remains executable and
tested. This is a source model boundary, not an invented successful source
primitive decomposition. Equal peak numbers in one profile do not establish
equal signatures or measurement costs.

This closes the explicit native bank/reflection dependency and actual sparse
Walk binding scope. Original narrow-ABI physical execution, large-case numerical
coverage, whole THC dependencies and complete Walk constructor acceptance
remain open. Constructor inventory therefore stays 119/303. G2 continues;
there is no PR merge or G3 transition.

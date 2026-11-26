# Dense measurement numerical ownership acceptance

Parent PR1334 is qualified at `10ba61639d984eda5250db6c130ae13008fc3175`.
The parent has 243 successful checks and 3 workflow-conditional
manual-pool skips. Main attempt 2 and Local GPU Gate attempt 1 succeeded.
Main attempt 1 timed out in original 20-qubit glued-trees execution; its failure
record is retained. A controlled same-source partition/dependent retry qualified
the final result. The unchanged full-size test also passed locally under
coverage in 83.49 seconds. No source, input, assertion, threshold or timeout was
changed for that retry.

## Ownership and preservation

See [the ownership map](density-instrument-numerical-ownership.md).
Seven private definitions and the unchanged normalization tolerance move from
Runtime into Simulation density-matrix and shared probability numerics.
Runtime retains backend/catalog selection, resource admission, generator/seed
ownership, readout metadata and evidence. All four production consumers and
seven test consumers use the canonical owners; eight old private attributes
are absent without forwarding.

The cumulative ledger contains 538 rows and the existing itemized public-entry
inventory remains 102. No public compatibility entry, contract type, package,
API, tolerance, native ABI or scientific evidence is added or changed.
The seven moved definition ASTs and 15 existing destination definitions are
unchanged. The remaining 46 Runtime definitions and 198 test assertions are
unchanged after normalizing canonical owner bindings.

## Validation

Focused validation passes 406 tests with no skips. Formal regressions ran in
the actual Git checkout with four workers:

- Unit/API/Team: **19414 passed / 957 skipped**, 437.11 seconds.
- Root/local integration: **1592 passed / 223 skipped**, 98.84 seconds.

Eleven source quality checks pass, including strict mypy over 1262 sources.
Black 24.1 and 26.10 both pass the full 3663-file Python surface.
Existing optional/hardware skip policies and the credential-dependent Kaiwu
live-smoke exclusion remain unchanged.

## Installed artifact

Wheel: `flagquantum-0.4.0-cp310-abi3-macosx_11_0_arm64.whl`.
SHA-256:

```text
556800078ee9ca8e713a08c8187a825054895fc289521ed60a802de1f07aa163
```

The 1262 product Python files in source, wheel and clean installation have
identical sets and bytes. Twenty-four public readout workflows preserve all
measurement values and metadata byte-for-byte against the parent across three
seeds, dense complex dtypes, four readout backends, Ising and Rydberg paths.
The original coarse-grid physical-state refusal preserves its error and cause.
All four probability kernels preserve raw value/gradient byte hashes, including
the normalization helper's unchanged alias behavior.

The installed benchmark CLI succeeds, kernel metadata leaves Torch and JAX
unloaded, and both native CPU and Sparse Blossom extensions actually load.
FlagQuantum is loaded solely from the installed wheel while the isolated
environment borrows existing third-party dependencies and their ordinary
sibling Torch library layout. This establishes local correctness and packaging;
it creates no accelerator, capacity or performance claim.

## Incoming and completion gates

The incoming audit reaches PR1084
`e92bf44bb5470535996a824e42d25bac6f871327`, still Draft and conflicting.
Its latest increment does not intersect the 19 candidate paths. The scoped
retirement audit has no errors over 3246 changed incoming Python trees; the
existing cumulative-ledger audit separately retains 15 gated old-import
obligations. Scoped success does not close those integration obligations.
Construction/IR, native build/development distribution and Rust keep their
existing gates. The new PR requires its own exact pushed-head CI qualification.

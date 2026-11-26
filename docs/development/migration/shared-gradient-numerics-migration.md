# Shared gradient numerics ownership

The finite-difference quotient and its sample-precision displacement move from
Core to `simulation/numerics/finite_differences.py`. The whole source file is
byte-identical after relocation. Public gradient requests and CPU conformance
oracles call this one implementation; there is no alternate quotient or new
method-selection policy. The old internal file is removed and all consumers
use the numerical owner.

The existing API record `docs/api-changes/FQ-GRADIENT-API-20261002.md` explicitly
says these helpers are not public and are absent from the gradient signature
contract. The move adds no compatibility facade. Stable root exports, gradient
signatures, result fields, reported exactness, displacement and failure behavior
remain unchanged. Core precision types remain Core-owned.

Do not regenerate the frozen P5 optimizer contract or rewrite its historical
`difference_implementation` path. The existing `source_path` resolver and explicit
whole-file ledger row locate the same authority at its current path. Its
contract checker already uses that resolver; no checker relaxation is needed.

PR1084 through `177ed24aba6c27d72b6fcb4b79b60747d8feec9e` does not modify
this source, has no changed-text references to the old module and has no target
collision. Verify relative and parent imports with the actual retired-import
guard. Complete formal regression and clean installed-artifact checks pass;
see the [acceptance](shared-implementation-ownership-acceptance.md). Exact-head
CI remains required for the pushed implementation.

Activation audit: the next nine-file mean-estimation/complex-phase increment
through `0742c440a100ce56eaec9934f9cf1487a84e0bc5` is SHA-verified.
It has no source/target intersection or new candidate reference; the actual
retired-import guard passes on 3,204 incoming Python trees.

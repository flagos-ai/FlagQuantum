# Runtime statevector kernel acceptance

Parent PR1333 head `ac2eee3c4ae85d3014bd98f76a14632f0df1b3ae` has
245 successful checks and three workflow-conditional manual-pool skips.
Main, Local GPU Gate and Optional OpenFermion workflows all succeeded;
192 combined coverage files passed the existing policy at 88.8%.

## Ownership and compatibility

See [the definition map](runtime-state-kernel-ownership.md). Runtime delegates
trajectory collapse and instruction application to Simulation, and two sweep
consumers share one Simulation tensor-product fold. Runtime retains execution
coordination, feedback, generator ownership and evidence. Three old private
Runtime attributes are absent, with no forwarding functions.

The cumulative migration ledger has 535 rows. The itemized public-entry
inventory remains at 102 with no additions. Stable exports, signatures,
exceptions, result schemas, native ABI and raw scientific evidence are unchanged.
The moved function bodies and all existing target definitions retain their ASTs;
all 17 remaining Runtime definitions differ only in direct-owner call bindings.

## Formal source validation

- Complete Unit/API/Team: **19411 passed / 957 skipped**, 402.64 seconds.
- Complete root/local integration: **1592 passed / 223 skipped**,
  95.49 seconds.
- Focused trajectory/RNG/feedback, matrix operations and reverse/fusion tests:
  359 passed / 13 existing environment skips. The complete carried-operand
  dispatch file passes 165 tests with its assertions unchanged.

The complete regressions above ran on product migration commit `f6aa8edef`,
with four workers in the actual Git checkout and normal process access. The existing credential-dependent Kaiwu live-smoke
exclusion and optional/hardware skip policies are unchanged. Eleven source
quality checks pass, including strict mypy over 1261 sources and Black over
3662 files. Operator manifest, runtime schema, legacy-root usage and correctness
certification also pass. Final documentation and applicable pinned source
hygiene checks pass; the pushed commit must pass its own full CI.

## Installed artifact

Wheel: `flagquantum-0.4.0-cp310-abi3-macosx_11_0_arm64.whl`.
SHA-256:

```text
fa71e2ad476fae5a59c944d5c007c9767fd5114a9c3049baa9a4d943d244d9d2
```

All 1261 product Python files match formal source, preview, wheel and clean
installed file sets and bytes. Six installed public dynamic workflows preserve
samples, classical bits and complex final states bit-for-bit against the parent
wheel across three seeds. Tensor-product values and gradients are bit-identical
to the original ordered fold. The three old Runtime private attributes are
absent, and all loaded FlagQuantum modules originate in the installation.

The installed benchmark CLI exits successfully; cold kernel metadata leaves
Torch/JAX unloaded. The actual native CPU and Sparse Blossom extensions load.
The isolated environment borrows existing third-party dependencies, including
the ordinary sibling `torch/lib` layout, while loading FlagQuantum solely from
the wheel. These are local correctness and installation checks, with no new
accelerator or distributed scalability claim.

## Incoming integration gate

The latest activation audit is PR1084
`e7259749b2686482ca33ba2d0d6049e913cb024e`, still Draft/CONFLICTING.
Its latest eight-file increment is blob-SHA verified, with no candidate path
intersection. The three new private retirements have no incoming consumers or
target collisions over 3221 changed Python trees. A separate full audit of the
parent's 533 migration rows retains 15 old-import obligations for the gated
joint migration; this batch does not resolve or qualify those obligations.
Construction/IR, joint native/development distribution families and Rust remain
behind their existing closeout gates. Exact pushed-head CI remains required.


## CI dependency-isolation correction

The native-fermion lane failed its pre-test optional-package absence assertion
when the runner hook supplied its installed scientific dependency overlay.
This lane now disables installed-dependency inheritance and creates a fresh
virtual environment. Other lanes keep their existing overlay behavior and
compatible wheel-download reuse remains available. The absence assertion keeps
all six original module names and reports any leaked module origins.

An actual-environment regression reproduces the ignored opt-out before the
fix. The new cached/fresh cases and lane wiring checks pass; the complete five
affected CI environment/routing/dependency suites pass **162 tests**. YAML,
Bash and Python 3.10 probe syntax checks pass. This CI-only correction leaves
all 1261 product Python files and the qualified wheel bytes unchanged; it adds
three CI tests to the prior full-regression surface. The corrected pushed
commit still requires its own complete CI qualification.

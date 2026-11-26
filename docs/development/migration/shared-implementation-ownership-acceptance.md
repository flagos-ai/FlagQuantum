# Shared implementation ownership acceptance

This batch is based on the CI-qualified #1332 parent
`85e8e32d42c8ace381348a28e7cac3c852385a94`. It gives shared kernels,
gradient numerics and existing experimental workflows one implementation owner.
It does not integrate the draft #1084 implementation or complete its deferred
Construction/IR, native packaging and development-distribution families.

## Source ownership and removal

- Three Simulation-root helpers move into existing Simulation Kernels:
  catalog admission, layout-aware complex BMM, and compiled/eager complex
  contraction with its cache. Their old files are removed; the package
  initializer remains metadata-only. Three whole-module ASTs are unchanged
  after import normalization. All 26 pre-existing top-level definitions
  across the three extraction targets, including their docstrings, also remain
  AST-identical; seven existing Pauli docstring examples pass.
- Experimental controlled-Pauli name decoding belongs to Core Operators;
  matrix construction belongs to Simulation; fixed-shot matching belongs to
  QEC Execution. All three function ASTs are identical to the original
  definitions. Drawer uses Core directly. The existing experimental modules
  expose the same shipped entry points, with no private implementation globals
  or additional root-discoverable namespaces.
- The single central-difference quotient and sample-precision displacement
  move to Simulation Numerics. The complete file is byte-identical. The old
  internal Core file is removed without forwarding, consistent with the
  existing gradient API record's explicit nonpublic classification.

The ledger has 533 rows and the public-entry inventory has 102 entries. The
last two entries are existing, explicitly unstable experimental facades, with
shipped example/README/docstring evidence; they are not new Stable Core
compatibility guarantees. No stable exports, signatures, result fields,
exception contracts or serialized schemas change. The controlled-Pauli
functions retain their original eager annotation values as well as resolved
hints. Original signatures were compared with fixed parent Git blobs rather
than with the replacement facades.

Frozen contracts and raw scientific evidence are unchanged. The P5 contract
keeps its historical finite-difference source identifier, resolved through the
existing whole-file migration ledger. Its checker and thresholds are unchanged.
Source ownership and retired-import guards pass; there is no second quotient.

## Formal source validation

- Complete Unit/API/Team regression: **19,411 passed / 957 skipped**,
  398.95 seconds, four workers. The existing credential-dependent Kaiwu live
  smoke exclusion is unchanged.
- Complete root tests and local training-parameter lifecycle integration:
  **1,592 passed / 223 skipped**, 85.77 seconds, four workers.
- Focused kernel consumers/catalog/cache/metadata: 288 passed / 7 existing
  conditional skips. Experimental/API/drawer workflows: 358 passed / 16
  existing conditional skips. Finite differences, public gradient modes and
  frozen P5 authority: 85 passed / 2 existing conditional skips.

The original Unit process completed normally. A local telemetry observer lost
permission to call `ps`; a Darwin kernel exit-status event recovered the same
original process's exit code **0**. Tests were not restarted or skipped to work
around the observer. Root integration also returned exit code 0.

The eleven source quality checks pass, including strict mypy for 1,260 Python
sources, Ruff/Black over the CI scope, frozen public API/vocabulary, ownership,
architecture, generated documentation, links, language and repository hygiene.
Applicable pinned source-hygiene hooks pass; this batch changes no YAML.
The post-acceptance documentation checks and exact pushed-head CI remain
required before the branch is considered qualified.

## Installed artifact validation

The macOS arm64 wheel is
`flagquantum-0.4.0-cp310-abi3-macosx_11_0_arm64.whl`, SHA-256:

```text
e9b2a858c264b9cfa76b790371787c6233b9410d4b4b8d746fe6b96d7744534f
```

All 1,260 packaged Python files match the formal product tree byte for byte.
A fresh isolated installation has exactly that Python-file set, with no extra
old files. Ten retired imports are rejected, including the old Core quotient.
The earlier installation environment was not used to certify removal because
an in-place install can retain files from an older wheel.

Repository-external execution verifies:

- Canonical module origins and all loaded FlagQuantum modules inside the fresh
  installation; historical public-plan pickle restoration and execution;
  public type identity, class hints, raw callable signatures and resolved hints.
- CPU expectation/autograd, complex contraction/gradients, cache clearing,
  controlled-Pauli names/matrices, and public `fq.gradient` reporting
  `finite_difference`, `exact=False`, a displacement and the analytic gradient.
- Actual native CPU and Sparse Blossom extension loading; original-circuit
  fixed-shot QEC with the bundled decoder while PyMatching is unavailable.
- Installed benchmark CLI help and a cold catalog import that leaves Torch
  and JAX unloaded. Installed QEC tests: 15 passed / 6 existing optional skips.

The test environment uses Python 3.12.14, Torch 2.14.1 and the repository-declared
optional Stim 1.16.0. These are local correctness/installation results, not
physical accelerator or distributed scalability qualification. No dependency
manifest, native ABI or hardware claim changes.

## Incoming audit and final gate

The latest activation input is #1084
`0742c440a100ce56eaec9934f9cf1487a84e0bc5`, still draft and unmerged.
The nine-file mean-estimation/complex-phase increment was blob-SHA verified,
with no source/target intersection or new candidate reference. The actual
retired-import guard passes on 3,204 changed incoming Python trees for this
batch's retired modules and removed implementation globals. Other deferred
families remain separate closeout work; this scoped audit does not qualify them.

See the [kernel batch](shared-kernel-structure-migration.md),
[experimental ownership batch](experimental-implementation-ownership.md),
[gradient numerics batch](shared-gradient-numerics-migration.md),
[itemized public entries](internal-forwarder-retirement.json), and
[path map](independent-structure-path-map.md). The pushed commit must pass its
own complete CI before another structural batch advances. Rust remains deferred.

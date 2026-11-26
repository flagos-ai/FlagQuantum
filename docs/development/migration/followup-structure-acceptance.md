# Example and numerical-owner migration acceptance

This follow-up is based on #1331 head
`471a2b938be1b6dd46078e433b307cbfd5bb7239`. Its main CI run
[45813169064](https://github.com/flagos-ai/FlagQuantum/actions/runs/45813169064)
passed on attempt 2. All 250 applicable check records succeeded. The four
manual-only pool/publication jobs were conditionally skipped; this record does
not claim their execution or hardware qualification.

## Source boundary

- Move 63 flat repository examples to capability directories. The VQE
  backend-switch training loop belongs to Training and retains its dedicated
  subprocess parity tests and JSON output. Algorithms keeps its existing
  curated example/premise inventory unchanged.
- Move four matrix-free modules to `simulation/numerics/matrix_free/` and three
  physical numerical modules to `simulation/dynamics/`. Runtime calls these
  numerical owners directly. Six previous internal numerical modules disappear.
- The old Hamiltonian path retains exactly two concrete type locators,
  `PauliSum` and `PauliSumTerm`. A real pre-move installed-wheel public-plan
  pickle proves this additional serialized boundary. It forwards no numerical
  operations or private helpers. The itemized public inventory has 100 entries.
- The cumulative migration ledger has 526 rows. All 63 example algorithms and
  seven numerical modules retain their normalized algorithm ASTs; only imports,
  documentation/physical paths and the two required serialized type modules
  change. Frozen API/contracts and raw scientific evidence are unchanged.

## Local verification

Environment: macOS arm64, Python 3.12.14, Torch 2.14.1. CPU tests establish
behavior and conformance rather than physical GPU/QPU or scalability capacity.

| Check | Result |
| --- | --- |
| Full Unit / API / Team, four workers; exclude the existing live Kaiwu service smoke | 19,410 passed; 958 skipped; 389.18 seconds |
| Full root tests and local training parameter lifecycle integration, four workers | 1,592 passed; 223 skipped; 91.02 seconds |
| Reclassified example, dedicated VQE subprocess parity, documentation and training consumers | 85 passed; 2 skipped |
| Frozen historical path/document generation regression suites | 76 passed |
| Strict package mypy | 1,260 source files passed |
| CI Ruff and Black scope (`flagquantum tests tools`) | passed |
| Public API, architecture, frozen vocabulary and ownership | passed |
| Documentation links/source, repository language and hygiene | passed |
| Historical source and retired-import guards | no errors |

The first full pass found four historical-path/rendering failures. The checks
now resolve immutable evidence identifiers through the existing relocation
ledger, and the generated parity document preserves the original contract's
identifiers with an explicit current-location map. A subsequent root pass
identified the training example's category mismatch; its classification and
consumers were corrected without changing the Algorithms inventory invariant.
The canonical entry-document test also follows the training example through
its split path, and the complete Unit / API / Team run was repeated on that
final layout. Original numerical assertions, contract values and recorded
evidence remain intact. The successful results above supersede those failed local attempts.

The extended examples/benchmarks lint surface has pre-existing debt and is not
represented as passing. No ignore rule, test selection exception or threshold
was added to obtain these results.

## Installed wheel

A clean source copy avoids stale build-library files. The built wheel is
`flagquantum-0.4.0-cp310-abi3-macosx_11_0_arm64.whl`, SHA-256:

`2dc40a229f644f23b026adc860e3c22983151c1075b468812a9a88d428d40418`

All 1,260 packaged Python files match the actual source bytes. Seven canonical
numerical owners are present and the six retired internal files are absent.
The packaged description matches the current README.

The wheel was installed into a disposable environment, outside the checkout.
Its existing dependency directory was reused with the normal adjacent
`torch/lib` layout; the shared environment and product loaders were unchanged.
Every loaded FlagQuantum module came from the new installed package. Checks
passed for CPU expectation/gradient, historical public-plan restore and
execution, exact locator identity and annotations, rejected retired imports,
native CPU and sparse-blossom extension loading, and the installed benchmark
CLI entry point.

## Incoming integration and remaining gates

The complete #1084 changed-text audit through `11fd9fa2` and its eight-file
Cirq / thirteen-file format-factory increments through
`7bbffeed70da51252e6b9f36fd05b1d85509cd6f` found no additional consumers or
source/target collisions for this batch. Blob SHAs and the actual retired-import
guard were checked. Recheck again if that incoming head changes.

This source acceptance does not qualify an unexecuted CI run. The follow-up
commit must pass the complete stacked-PR CI independently. #1084-dependent
construction, IR, shared packaging and other remaining migration families stay
open; Rust remains deferred. Neither parent PR has been merged or force-pushed.

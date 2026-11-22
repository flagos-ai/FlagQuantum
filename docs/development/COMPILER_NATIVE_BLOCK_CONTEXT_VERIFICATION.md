# Native block context verification

The private native-CZ verifier compares noninstruction semantics with instructions
removed from both temporary IR values. The previous comparison inserted the same
source instructions into the output and then serialized that instruction list in
both values. The remaining comparison uses the existing canonical serialization
and hash, including wire count, version, dtype, shape, observables, measurements
and metadata. No field projection, new contract, dependency or cross-call cache
is introduced.

This is a `single_device_fast_path` implementation change. Complete source/output
identity, independent routing/native replay, operator residuals, rewrite lineage,
resource checks and fresh reads of mutable tensors remain in place. Instruction
identity verification occurs before the context comparison and operator/lineage
verification after it. The public API, evidence formats and numerical tolerances
are unchanged. The [upstream reuse record](COMPILER_UPSTREAM_REUSE.md) describes
the Qiskit/Cirq algorithms already integrated into native compilation.

## Correctness and observations

[The paired record](evidence/native-block-context-verification.json) compares the
exact production verifier on main
`a40a3b7a328b879d4a5d94594ceab567d2e32c43` with the changed function in the same
source tree. The unchanged SDKBench checkout is pinned to
`9aa9260920984e2ca05bcb293abb885fccd9c144`. Sixty pairs span widths 6/8/12,
five applications and four topologies using the previously inspected nonce.
All 120 independent acceptances pass. Complete normalized outputs, resource
metrics, identity endpoint layouts, routing strategy and rewrite statistics
match within every pair. Six-wire acceptance checks complete operators;
eight/twelve-wire acceptance checks three probes only.

Seven added tests reject resealed changes to wire count, dtype, shape, observable,
measurement and nested metadata semantics and exercise successful rewrites with
observable/calibration tensors and symbolic metadata. Re-sealing unchanged
instructions cannot authorize altered context tensors. Existing operator, phase,
cost, membership, mutation and routing-corruption checks remain operative. The
fixed IR version is still enforced by its constructor.

The sparse checkout's default tier passes 102 unit tests; its runtime tier passes
42 integration tests. These are the complete seeded selections present in this
checkout, not whole-repository certification. Full-source CI remains required.

Black and Ruff pass. Strict Linux-target mypy now passes both the production
module and the complete unit test module with `--strict --python-version=3.12
--platform=linux --explicit-package-bases --ignore-missing-imports
--no-incremental --follow-imports=silent`. The seven errors previously recorded
against the baseline tests are resolved: evidence and synthesis types come from
their owning modules, the hash-property observer inspects and validates its
descriptor and return value, and negative tests explicitly cross runtime call
boundaries to supply invalid budgets and attempt forbidden assignment. Those
negative tests still require the original exceptions; no casts, ignores,
production API changes or checker exceptions were added. This checks these two
files with imported modules followed silently and unavailable dependencies
ignored; it is not whole-repository type certification. The earlier paired
performance archive predates these test-only repairs and remains unchanged.

Two warmups and seven alternating retained samples per variant, separate Python
allocation observations and separate instrumented profiles give:

| Inspected case | Production / changed median | Python traced peak bytes, production / changed | Instruction encodings, production / changed |
| --- | ---: | ---: | ---: |
| 6-wire QFT, grid | 107.672 / 117.931 ms | 2,040,124 / 1,964,718 | 21,958 / 20,794 |
| 6-wire repetition, ring | 138.521 / 135.474 ms | 2,238,379 / 2,378,617 | 27,570 / 26,286 |
| 12-wire QFT, grid | 668.776 / 657.327 ms | 10,116,044 / 10,107,789 | 110,872 / 105,148 |

Profiles confirm 4.7%–5.3% fewer instruction encodings; hash call counts remain 70.
Timing does not show a consistent improvement: six-wire QFT is approximately 9.5%
slower, while repetition's traced peak increases by 140,238 bytes. The final
twelve-wire case overlapped a 1.36-second local unit run; individual sample
overlap was not recorded. Retain those observations without interpreting them
as isolated performance or memory evidence. Shared-host timings, profiler totals
and Python allocations do not establish qualified latency, process RSS,
significance, SDK rankings, prospective behavior or scalability. The change
removes duplicated work; no general speedup is claimed.

## Existing workflow and reproduction

Users retain the native target-artifact compilation workflow. Existing local
physical-evidence acceptance exercises compilation, serialization, round-trip
verification and independent complete-operator reconstruction without hardware
or credentials:

```sh
python -m pytest tests/unit/test_native_block_optimization.py -q
python -m pytest tests/integration/test_native_block_physical_evidence.py -q
```

Expected: 102 unit and 42 integration tests pass, including rejection of altered
context and corrupted operators. The archive retains the exact paired driver,
source hashes, acceptances, timing samples, allocation observations and profiles.
Its temporary paths describe the measured setup and must be adjusted elsewhere.
Use a separate bounded supervisor; do not install an outer signal alarm around
SDKBench's own stage timer.

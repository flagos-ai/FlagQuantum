# Native block context verification

The native-CZ verifier's existing context checks now have explicit positive and
resealed-tamper tests for wire count, precision, shape, observables, measurements
and nested metadata, including mutable tensor values. The production verifier
remains unchanged: it compares the canonical source context while separately
checking complete identity, replay, operators, lineage and budgets.

An earlier private optimization removed instructions from both temporary context
values before comparing canonical hashes. That production change has been
withdrawn because the measurements below do not show a consistent workflow
benefit and include a latency regression and higher memory use. The paired record
is retained as historical evidence of that rejected proposal; it does not describe
a production optimization in the final change. The seven added scenario tests
also pass against the original verifier and are retained independently.

This is Compiler-owned `single_device_fast_path` test coverage. No public API,
evidence schema, numerical tolerance or production algorithm changes. The
[upstream reuse record](COMPILER_UPSTREAM_REUSE.md) describes the Qiskit/Cirq
algorithms already integrated into native compilation.

## Correctness and observations

[The paired record](evidence/native-block-context-verification.json) compares the
exact production verifier on main
`a40a3b7a328b879d4a5d94594ceab567d2e32c43` with the changed function in the same
source tree. The unchanged SDKBench checkout is pinned to
`9aa9260920984e2ca05bcb293abb885fccd9c144`. Sixty pairs span widths 6/8/12,
five applications and four topologies using the previously inspected nonce.
All 120 independent acceptances of the historical proposal pass. Complete normalized outputs, resource
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
significance, SDK rankings, prospective behavior or scalability. The proposal
removed duplicated work, but that alone did not justify adopting it. The
production optimization is withdrawn; these measurements are not evidence of
a speedup from the retained test-only change.

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


## Integration boundary

The earlier proposal head failed Linux coverage shard 191 at the unchanged
wide-disjoint compiler timing assertion. That failure is retained as a failure;
it does not justify relaxing the threshold or blindly retrying the same code.
The independent fixed-singleton matrix optimization in PR #1288 addresses
measured work in that path. Its passing shard does not certify this branch.
Final test-only verification and full-source CI must run on the revised head.

After restoring the original production verifier, all 144 focused tests pass:
102 unit and 42 integration tests. The sparse default/runtime tiers pass the
same seeded selections. Strict Linux-target mypy passes the complete test module;
Black, Ruff, English, links and whitespace checks pass. The retained scenarios
therefore validate existing behavior independently of the withdrawn optimization.

After synchronization to integrated main
`5bc56a03cc584030e68e00b55ea75e45f9da36bd`, the sparse default tier passes
134 unit tests and the runtime tier passes 42 integration tests. Strict
Linux-target mypy still passes the complete test module. The final diff against
that main changes no production Python source. The historical 120 acceptances
above belong to the rejected proposal; no new performance benefit is claimed.

# Complete packed basis simulator source disposition

## Interface, bit order and ownership

The locked source group is Kickmix `sim/sim.h`, `sim/sim.test.cc` and
`sim/sim.perf.cc` at `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`.
The complete `sim.cc` initialization grammar is already accepted separately.
This is a local `single_device_fast_path`.

Simulation owns the existing `PackedBasisState`, numerical executor, counters,
ChaCha and fixed-width sampling. `_basis_simulation.py` composes these existing
owners for persistent numerical lifecycle, capacity, register buffers and
source debug callbacks. It contains no second gate-execution loop. Core
`BinaryMatrix` owns bit transposition and `FixedWidthInteger` owns scalar cells.
Compiler circuit values supply borrowed physical metadata through a structural
protocol. Testing composes lifecycle sampling. Benchmarking owns workloads and
timing. Normal user circuit construction retains `circuit.append(op)`.

Wire masks hold one bit per trajectory, with the low bit representing shot zero.
Scalar register offset zero is the low value bit. Exact phase is represented in
the existing 128-bit full-turn ring. Condition masks persist between scans until
popped or explicitly cleared. Register buffers are caller-visible mutable rows
of immutable scalar values. Separate buffers preserve before/expected/after
ownership. Quantum workspace belongs to circuit declarations and is validated
by the existing append and erasure conformance paths.

Copying owns independent numerical state, counters, scalar buffers and standard
MT/ChaCha state. Callback closures retain their ordinary function ownership.
Raw mutable C++ spans become native indexed dictionaries or explicit accessors.
Invalid pointer access and allocation-address identity are outside the native
representation. Defined numerical behavior remains subject to source evidence.

## Intake and current evidence

All 92 frozen source/dependency blobs match. All 15 original simulator tests and
two direct helper tests pass unchanged. The four-width original lifecycle probe
confirms condition retention, independent dimension growth, preserved masks,
exact phases, inactive debug callbacks and end-of-scan sign commit. Native
header and original test scope is accepted after the batch checks below.

## Complete definition disposition

| Original definition | Native owner and evidence |
| --- | --- |
| Transpose pass and complete 64 by 64 transpose | Existing Core `BinaryMatrix.transposed`, existing binary matrix corpus and original helper tests. Benchmark composes that owner. The six C++ implementation passes remain source implementation detail. |
| `SimInitInstruction` value, equality, parser and stream output | Accepted Compiler `_register_initialization.py`, existing Ecosystem binding and 62 original cases. |
| Empty and enabled operation/rotation counters | Existing Simulation `PhysicalOperationCounters`, original statistics corpus and new per-lane workload traces. |
| State spans, phase reference and register-addressed values | Existing `PackedBasisState` dictionaries and shared register buffers. Native indexed access replaces mutable pointer/span storage. |
| Constructor, copy/move/assignment and owned random streams | `PackedBasisSimulation`, independently copyable standard MT state and immutable ChaCha snapshots. Four-width original copy/isolation and RNG advancement traces. Python assignment retains Python ownership, explicit `copy`/`deepcopy` creates an owned snapshot. |
| Safe scalar Boolean and wire reads | `safe_read`, with bounds, zero dimensions and Boolean broadcast checks. |
| Full and single register transposition in both directions | Existing `_register_buffer.py`, composed by the lifecycle owner. Mixed members, overlap ordering, multiword and hanging-bit evidence is reused and rerun. |
| Growth, configuration and shot clearing | Independent capacity doubling, exact phase/sign retention and separate register buffer ownership. Configuration preserves streams and counters. Shot clearing preserves buffers and counters. |
| Complete instruction scan and persistent conditions | Existing `apply_packed_basis_records`, now retaining the state-owned condition stack between scans. Its cached base condition follows source ordering even when a debug callback changes the stack. |
| Full debug callback | Lifecycle observer preserves record target/predicate, packed value, active mask and inactive calls. Source sign is committed after the full scan, including exception behavior. |
| Default stderr printer | Existing Ecosystem `kmx_sample.py` formats lane-zero bit/phase output over the existing executor debug path. Source text tests remain in the concentrated regression. Simulation has no formatting dependency on Compiler. |
| Statistics clearing | Existing counters are cleared independently of numerical state, streams and buffers. |
| Complete `sim.test.cc` | All 15 original tests run unchanged plus two helper tests. Native original source traces and existing gate, rotation, initialization, register-buffer, counter, reset, debug and swap corpora cover the corresponding owners. |
| Complete `sim.perf.cc` | Shared Benchmarking `_physical_basis_workloads.py` composes all three original sampling bodies and the Core transpose workload. Existing adaptive timing is reused. All 13 ARM-host and 16 Rosetta/SSE original registered benchmarks execute unchanged. |

The fixture generator verifies every referenced frozen Git blob before compiling
an external probe against unchanged source. Its checked-in numerical corpus
contains all four portable widths, copied RNG/counters/state, callback failure,
cached condition behavior and two complete repetitions of every benchmark body.
All wire masks, all lane phases, every scalar before/output cell and every lane
counter in the counting workload are compared as complete ordered text.
The input circuits are extracted from the hash-verified original benchmark file.

The fuzzer now consumes this shared lifecycle owner. Its 32 complete numerical
source scenarios, diagnostic scenarios, scalar-lane execution replacement and
forced measurement masks retain accepted output. Normal append, complete complex
amplitudes, zero workspace and all applicable erasure branches reuse the existing
integration and QROAM evidence, rerun at this batch boundary.

## Resource accounting and timing

Lifecycle, copy, transposition and benchmark orchestration add zero quantum
primitives, measurements or quantum workspace. The original sample circuit has
23 physical records, including 5 HMR operations. A single workload repetition
executes all configured lanes. Conditional CZ counts follow each measured mask.
The counting variant retains the complete per-lane actual primitive counts over
two repetitions, compared with the original counters. Analytical model is `None`.
Exact 128-bit phase units and separate sign masks are retained before formatting.

The original goal constants remain original host-specific targets. ARM source
benchmark goals span 43 ns to 4.8 us per repetition. A bounded local Python run
observed approximately 37–162 us for sample, 131–1145 us for counting and
251–2114 us for populated buffers across widths 64–512. Core transpose observed
about 2.9 us. Original C++ and Rosetta/SSE observations, target constants and
native timings are recorded separately in the external batch receipts.
These observations describe the single-device implementation and current host.
Native timings exceed the original goals. K-0271 retains an open performance
acceptance gap, consistent with the existing scalar and population-counter
benchmark parents. Its complete functional workloads are delivered.

## Explicit adaptations and remaining boundaries

- Standard Python integers and dictionaries own numerical masks and capacity.
  C++ allocation addresses, raw spans, dangling references and out-of-bounds
  pointer access have no native correspondence.
- Native copying clones the standard MT engine, numerical state, counters and
  scalar buffers. Python function closures retain their own captured references,
  including callbacks that intentionally observe the original owner. Arbitrary
  third-party draw iterators retain their own Python deepcopy constraints.
- Native canonical random draws follow portable low-lane-first order. The
  accepted SIMD audit records the original Clang SSE high-lane-first draw order.
  SSE benchmark bodies execute in Rosetta, with timings retained as separate
  observations. AVX/AVX512 runtime gaps remain under the six existing SIMD parents.
- Source-invalid raw records and invalid C++ pointer operations are outside the
  lifecycle conformance claim. Supported Compiler entry points retain their
  existing validation and error contracts.
- Pending Proposals 116–119 and remaining G2 source files stay separately tracked.
  This batch does not close those scopes.

## Reproduction

Run `tests/unit/fixtures/generate_physical_basis_lifecycle.py` with the locked
reference root and immutable inventory CSV. Compare its output byte for byte with
`tests/unit/data/physical_basis_lifecycle.json`. Run
`tests/unit/test_physical_basis_lifecycle.py`,
`tests/unit/test_physical_basis_workloads.py` and the existing fuzzer tests.
For timing, pass a returned workload body to the existing Benchmarking `measure`
function. Keep actual counters, model estimates and timing samples separate.

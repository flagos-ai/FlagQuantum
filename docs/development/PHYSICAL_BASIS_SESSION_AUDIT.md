# Complete interactive physical simulation source intake

## Scope and ownership before implementation

The locked group is Kickmix `py/interactive_simulator.pybind.cc`,
`interactive_simulator.pybind.h` and `interactive_simulator_test.py` at
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`. All 53 source/include dependency
blobs match the immutable inventory. All 102 original tests pass unchanged
against the hash-verified locked native extension.

Simulation `_basis_trajectories.py` owns persistent 256-lane numerical chunks,
master/chunk standard seed-sequence streams, growth, wire masks, exact phases,
shot clearing and valid-prefix actual counters. It composes accepted
`PackedBasisSimulation` objects and the existing packed executor. It imports no
Compiler or external reference object.

Ecosystem `_physical_basis_session.py` owns conversion of Python indices,
register aliases, arrays, phase inputs, shape/stride validation and external
debug callbacks. It immediately resolves values to existing native
`OperandIdentity`/`OperandArray`, borrowed Compiler circuit metadata and numerical
mask/phase values. Existing physical statistics and phase conversion owners are
reused. Both modules are private. User construction retains `circuit.append(op)`.

Shot zero is the low mask bit. Scalar register offset zero is the low value bit.
Chunks contain 256 lanes and the final chunk exposes only its requested prefix.
Numerical execution retains all 256 source lanes. Counters and external callback
arrays expose the requested prefix. Exact phase units use the existing 128-bit
full-turn ring. Quantum workspace belongs to the supplied circuit. Session
orchestration adds no quantum wires, gates or measurements.

## Source boundary witnesses

The original boundary receipt confirms the following defined behavior.

- Conditions persist across separate circuit calls. A 257-shot condition mask
  with only shots zero and 256 active counts two X operations per call.
- A failed second integer write preserves the first write and leaves the second
  wire zero. All target dimensions grow before the per-value loop.
- Empty index reads/writes skip lane bounds checks, but a negative shot still
  fails the binding's unsigned conversion before iteration.
- Phase input conversion happens before shot bounds checking.
- Boolean seeds are rejected. Every one of 256 seed bits participates through
  the original master and chunk seed sequences.

Reference C++ binding diagnostics are separated from native-owned validation
wording. The unexposed uint64 array helper is not registered as a Python method.
Its definition requires an explicit disposition before complete parent closure.
The complete native mapping and acceptance evidence follow below.

## Complete definition mapping

| Locked definition group | Native owner and disposition |
| --- | --- |
| External/master RNG and eight seed words | `_seed_words` and `BasisTrajectories.__init__`, eight little-endian words, four master uint64 draws per chunk, existing exact standard seed sequence and ChaCha initialization |
| `with_sims`, chunk allocation, capacity and clearing | One numerical chunk list with the existing optional counter owner, independent wire growth, retained RNG/counters and cleared wires/phases/condition stack |
| `register_data`, circuit execution | Borrowed immutable `KmxCircuit.registers`, copied aliases independent of execution, existing numerical circuit scan |
| `peek_single_bit`, `set_single_bit`, integer byte packing | Numerical wire masks and single-lane access, Python owned integers replace temporary C++ byte buffers, least significant bit first |
| Scalar/sequence across/within bool and int routes | Existing native operand binding and array views, exact scalar/list/array shapes, bool dtype, noncontiguous/negative/overlapping output strides, repeated indices and sequential failure mutation |
| Integer and NumPy writes | Validated original values, sequential wire updates, source growth order and partial writes, structural Python integer sequences, integer values wider than the shot count |
| Exact phase reads/writes and phase-flip outputs | Existing external Fraction conversion and PhaseAngle128, exact 128-bit units, delayed sign, first invalid phase preserves earlier output cells |
| Debug callbacks, property and stderr target | Existing executor observer, original order including inactive callbacks, independent arrays and valid tail prefix, chunk-lane-zero stderr |
| Operation statistics and clearing | Existing numerical raw counters and Compiler physical snapshot owner, valid-prefix aggregation, per-lane snapshots, all phase hierarchy levels remain in the accepted owner |
| Python class/method registration | Private native class and methods in the owning modules, native IDs and existing circuit formats |
| `peek_across_shots_1d_np_u64` | Unregistered source helper, unreachable from every source Python API. Its intended packed observation is represented by existing integer mask reads. No new exposed native method or executable parity claim for this unused pointer implementation |
| Header templates, storage and byte buffer capacity | Python-owned lists, integers and immutable metadata replace binding dispatch and temporary raw storage. Numerical state never borrows caller NumPy memory |

The original uint64 helper increments a uint64 pointer using a NumPy byte
stride. It has no registered caller and no runtime witness in this batch.
That observation is retained as a source inspection boundary without labelling
it a confirmed numerical defect.

## Reproducible source comparison

`tests/unit/fixtures/generate_physical_basis_session.py` requires the original
extension version 0.1, `Simulator`, and SHA256
`5e849075b49d895cca761fae864c8561b54362a6158fbfcabe7e9eb6e96d789e`.
The shared scenario runner records complete wire masks, every requested exact
lane phase and actual counters after three successive circuit scans. Shot counts
are 0, 1, 3, 255, 256, 257 and 513. Counted and uncounted execution agree.
Additional scenarios cover aliases, scalar/sequence shapes, mixed quantum and
classical wires, Boolean constants, two control scopes across separate calls,
strides, overlap, duplicates, structural sequences, phases, partial output
failure, partial writes, empty integer reads and 9/257/513/1000/2048-bit writes.
The JSON oracle regenerates byte-exact from the locked native extension.

All 102 original tests pass on the original extension. The opt-in
`tests.unit.fixtures.replace_physical_basis_session` pytest plugin converts
external values immediately and routes the same 102 unchanged source tests to
native execution. Every original assertion passes. The adapter translates only
IDs, circuits and callback IDs. It retains original builders and tests as
external consumers. Production modules import no reference extension.

The concentrated suite also reruns the accepted normal `circuit.append(op)`
full-complex phase cases, all four feedback measurement histories with zeroed
quantum workspace, all-branch lookup erasure, scalar execution replacement and
existing phase/counter projections. Those accepted evidence paths are reused.
The new session adds zero quantum primitive instructions, measurements or
workspace. Its host storage is 256 lanes per chunk with a valid-prefix external
view. Actual activated physical instruction counters remain separate from
analytical models. This orchestration layer has no analytical quantum model.

## Native boundary decisions and remaining scope

- Existing `OperandIdentity`, `OperandArray` and native scalar integers represent
  physical IDs. Their Python representations and diagnostics retain native
  wording. Existing Compiler input semantics are reused.
- Binding-specific unsigned-cast failures map to native TypeError for noninteger
  inputs and ValueError for unsigned range refusal. In-range lane bounds retain
  IndexError. Phase input conversion precedes lane bounds checks.
- The source's first-cell NumPy pointer refusal for empty arrays is retained as
  IndexError. Empty integer index lists retain their defined zero/empty result.
- Python immutable metadata and owned numerical state replace raw pointers,
  byte buffers, C++ variant dispatch and object-lifetime mechanics. Invalid
  pointers, unsafe extension mutation and unregistered helper execution have
  no native parity claim.
- These modules remain private. Existing normal construction and lowering
  routes supply the accepted physical records.
- Original SIMD ISA runtime gaps, original benchmark timing targets and pending
  Proposals 116–119 retain their independent open ledger entries.

## Local acceptance

The completed batch passes 1219 concentrated tests, 121 integration tests and
strict checking of six source/test/fixture files. New modules have 99% combined
statement/branch coverage in the concentrated suite. Black, Ruff, both team
scopes, architecture, multilevel IR and API checks pass. Dependency policy has
exactly the same 720 pre-existing findings as the retained baseline.

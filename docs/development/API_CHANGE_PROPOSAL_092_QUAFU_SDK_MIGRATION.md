# API Change Proposal 092: Quafu SDK Migration

**Status:** Approved for staged implementation on 2026-10-25
**Scope:** Quafu Task API transport, provider integration, and SQC retirement
**Public entry points preserved:** `fq.run`, `fq.submit`, `fq.restore_job`

## Decision summary

FlagQuantum will use the official `quafu-sdk` package for communication with
the maintained Quafu Task API. FlagQuantum will retain ownership of its public
API, compiler selection, routing, deployment packages, capability gates,
submission identity, and normalized results.

The unmaintained SQC protocol will be removed after the maintained Task API
path passes the static logical, QSteed physical, detached-job, collection, and
result-conformance gates in this proposal. There will be no automatic fallback
between Quafu protocols.

This migration replaces only the provider's HTTP protocol implementation. It
does not replace `QuafuProvider`, QSteed, FlagQuantum IR, `DeploymentPackage`,
`ProviderTaskHandle`, `RemoteJob`, or `ExecutionResult`.

## User contract

The shortest platform-compilation path remains:

```python
import flagquantum as fq

circuit = fq.Circuit(2).h(0).cx(0, 1)
result = fq.run(circuit, target="quafu:Baihua", shots=1024)
print(result.counts[0])
```

Expected result: a normalized FlagQuantum counts mapping. Internally, the
provider may ask the maintained Quafu service to compile the logical circuit,
but no Quafu SDK object is returned to the user.

The explicit local-compilation path remains:

```python
import flagquantum as fq

circuit = fq.Circuit(2).h(0).cx(0, 1)
result = fq.run(
    circuit,
    compiler="qsteed",
    target="quafu:Baihua",
    shots=1024,
)
print(result.provenance["compiler"])
```

Expected result: `qsteed`. QSteed owns compilation and routing; the provider
submits the resulting physical program with Quafu service compilation disabled.

## Responsibility boundary

FlagQuantum continues to own:

- stable user-facing execution and detached-job APIs;
- compiler selection and QSteed invocation;
- logical-to-physical routing and mapping validation;
- target capability and calibration identity;
- deployment artifact sealing and SHA-256 identities;
- fail-closed admission for unsupported capabilities;
- provider-neutral task handles, job status, and execution results; and
- credential redaction and public exception behavior.

`quafu-sdk` owns:

- maintained Task API endpoint and request shapes;
- authorization header construction;
- HTTP connection, TLS, timeout, and retry mechanics;
- idempotent Task API submission;
- Quafu-native job status, cancellation, and result retrieval; and
- decoding of Quafu-native error envelopes and result records.

SDK types stop at `flagquantum.remote`. Core, Compiler, Deployment, Runtime,
and stable public results must not import or serialize them.

## Compiler mapping

The two compiler parameters belong to different layers:

| FlagQuantum request | FlagQuantum responsibility | SDK submission |
|---|---|---|
| no local compiler | preserve a logical program | `compiler="qiskit"` |
| `compiler="qsteed"` | compile, route, and emit a physical program | `compiler="none"` |

The second row requires a physical-program contract. Physical qubit numbers
must appear in the submitted program itself; the retired SQC
`logical QASM + target_qubits` request shape is not carried forward.

The migration must not silently relabel the maintained service compiler as
`quarkcircuit`. Result provenance records the compiler the service reports.

## Dependency and credential policy

`quafu-sdk` is optional, imported lazily, and constrained to the reviewed
pre-1.0 compatibility range. A default FlagQuantum installation and every
non-Quafu path remain independent of the package.

`quafu-sdk` and `pyquafu` both own the Python `quafu` namespace and are not a
supported co-installation. The Quafu extra, diagnostics, and installation guide
must state this boundary and surface an actionable conflict error.

FlagQuantum remote execution accepts an explicitly supplied API key or
`QUAFU_API_KEY`. A library call must not open a browser, wait for interactive
OAuth, or silently consume a saved interactive login. Public read-only
discovery remains credential-free when the maintained service permits it.

## Staged implementation

### Batch A: dependency and adapter boundary

- add the reviewed optional SDK dependency and lazy loader;
- add a private, injectable SDK adapter;
- prove that normal FlagQuantum imports do not import `quafu`;
- detect an unavailable or conflicting SDK with an actionable error; and
- perform no provider request and no public behavior change.

### Batch B: read-only provider data

- migrate device discovery, backend metadata, limits, and calibration reads;
- normalize data into `CloudBackendProfile` and FlagQuantum-owned mappings;
- preserve backend and calibration identities without guessed defaults; and
- prove with fake SDK clients that read-only calls send no credential.

The implementation keeps SDK clients and native response models behind a
private adapter. `QuafuProvider().list_devices()` and
`fetch_calibration(...)` use that adapter by default, return only
FlagQuantum-owned values, reject missing authoritative widths or malformed
identities, and remain covered by offline fake-client tests. An explicitly
injected HTTP transport retains the compatibility path solely for staged
contract tests until Batch F removes it.

### Batch C: static logical task lifecycle

- migrate submission, status, wait, restore, cancellation, and result reads;
- map SDK failures into existing FlagQuantum exception categories;
- preserve bit order, measurement mapping, executed-program identity, and
  returned-shot counts; and
- record the actual maintained service compiler in provenance.

The implementation routes default single-circuit logical submissions through
`Client.submit(..., compiler="qiskit")`, preserves the SDK job identity in the
existing FlagQuantum receipt, and reconnects status, result, cancellation, and
restoration through that identity. SDK-native jobs and results remain private;
provider fields are copied into FlagQuantum-owned metadata, and SDK failures
are translated to stable FlagQuantum validation, execution, or timeout
categories. Physical QSteed and collection paths remain unchanged for Batches
D and E respectively.

### Batch D: QSteed physical submission

- define and implement physical-program emission after QSteed routing;
- encode physical qubit numbers in the submitted program;
- validate capacity, native operations, directed connectivity, measurements,
  and logical-result restoration before submission;
- bind the program digest, mapping, backend snapshot, and calibration identity;
  and
- submit through the SDK with `compiler="none"`.

The implementation keeps the public `fq.run(..., compiler="qsteed")` call
unchanged. Immediately before submission, the provider resolves one public SDK
backend snapshot, checks the routed program against its authoritative capacity,
native gates, and directed coupling map, then rewrites logical `q[i]` references
to the selected physical qubit numbers. Classical bits retain logical order, so
the result remains a normal FlagQuantum counts result. The receipt binds the
physical-program digest, ordered mapping, backend-snapshot digest, calibration
identity, and expected measurement map. Result decoding fails closed if the SDK
does not confirm that map, calibration, bit width, or `compiler="none"`.

All Batch D verification uses an injected in-memory SDK client. It neither uses
a credential nor creates a platform task.

### Batch E: collections and cutover

- migrate collections, detached jobs, pagination, partial failure, and
  expectation-measurement plans;
- switch maintained Quafu paths to the SDK only after parity tests pass; and
- prohibit fallback to another protocol, backend, or compiler after failure.

The implementation routes default logical collections through one SDK
`Client.submit([...], compiler="qiskit")` call and retains the parent SDK job
identity in the existing FlagQuantum receipt. Status, cancellation,
restoration, and ordered child results remain available through `RemoteJob`;
SDK pagination and native objects stay private. FlagQuantum still rejects
missing, duplicate, failed, malformed, or out-of-order child identities instead
of returning partial success. Historical `task_api_v1` receipts and explicitly
injected HTTP transports remain readable until Batch F removes SQC code.

All Batch E verification uses an injected in-memory SDK client. It neither uses
a credential nor creates a platform task.

### Batch F: SQC retirement

Batch F starts with a compiler-topology bridge: the default QSteed target
resolution now derives its existing private `chip_info` input from the
credential-free SDK calibration snapshot. The bridge maps SDK `qubits` and
`couplers[].cz_fidelity` into the shape consumed by the current QSteed plugin,
validates every physical index and calibrated edge, and does not invent a
missing one-qubit fidelity. This removes QSteed compilation's dependency on
the SQC `/task/backendtest/*` path without changing the public compiler call.

```python
import flagquantum as fq

compiled = fq.compile(
    fq.Circuit(2).h(0).cx(0, 1),
    compiler="qsteed",
    target="quafu:Baihua",
)
print(compiled.metadata["execution_target"]["backend"])
```

Expected result: `Baihua`. The read is credential-free, and compilation does
not submit a task.

The second Batch F slice moves `TwinExperiment.submit(provider)` onto the same
SDK physical-program lifecycle. FlagQuantum lowers a logical CX to the native
H-CZ-H sequence when required, embeds the ordered physical mapping in the
submitted QASM, disables service compilation, and binds both the frozen source
QASM identity and the materialized physical QASM identity. Saved Twin receipts
retain the SDK protocol, calibration, backend snapshot, and measurement map
required for fail-closed result restoration.

```python
import os

from flagquantum.remote import QuafuProvider

provider = QuafuProvider(api_key=os.environ["QUAFU_API_KEY"])
receipt = experiment.submit(provider)
print(receipt.payload["quafu_protocol"])
```

Expected result: `quafu_sdk_v1`; one task is submitted with
`compiler="none"`. The existing public Twin API is unchanged.

Full SDK replacement of `fetch_chip_info()` remains blocked until the
maintained calibration schema authoritatively supplies every quantity required
by the existing Twin noise model, including T2, one-qubit gate fidelity, and
one- and two-qubit gate duration with units. Batch F must not infer or fabricate
those values from topology-only fields.

- remove `QUAFU_API_TOKEN`, the SQC hostname, token-header authentication, and
  `/task/*` request paths;
- remove legacy discovery, submission, status, result, calibration, and
  fallback code;
- remove obsolete tests, examples, documentation, and environment guidance;
  and
- remove `quarkcircuit` from FlagQuantum extras if the post-cutover dependency
  audit finds no maintained use.

## Dynamic-circuit boundary

API Change 090 remains the authority for the Quafu DynamicCircuit vertical
slice. This migration can resolve physical submission representation, but it
does not establish that Quafu accepts FlagQuantum's OpenQASM 3 dynamic program
or supports its mid-circuit measurement, reset, and conditional semantics.

Until authoritative format and capability evidence exists, the dynamic path
continues to fail closed with a format-specific blocker such as
`quafu_dynamic_program_format_unconfirmed`. SDK adoption alone is not a remote
dynamic-QPU support claim.

## Verification and performance

Every implementation batch uses injected fake clients or transports. Network,
credential, simulator, and QPU checks require separate explicit authorization.

The migration must prove:

- importing FlagQuantum does not import the SDK;
- non-Quafu execution and planning are unchanged;
- credential-free validation finishes before SDK client construction;
- submission identity survives status, restore, and result operations;
- ambiguous submission cannot create an untracked second FlagQuantum task;
- credentials and authorization headers never appear in errors or receipts;
- SDK-native objects do not cross the Remote boundary; and
- the adapter adds no material regression relative to the measured existing
  provider overhead.

A performance budget is set from measured before-and-after distributions, not
from an arbitrary fixed line-count or latency threshold.

## Exit criteria

The migration is complete only when:

1. both documented user workflows pass provider-contract tests;
2. maintained discovery, calibration, static logical submission, QSteed
   physical submission, detached jobs, collections, and normalized results all
   use the SDK adapter;
3. the repository contains no SQC hostname, endpoint, credential variable, or
   automatic fallback in product code or user documentation;
4. the Quafu extra installs in a clean environment and reports a clear
   `pyquafu` namespace conflict;
5. the default installation and non-Quafu CI lanes remain unaffected; and
6. every pull request includes a runnable public `fq.*` example and its
   expected result.

## Non-goals

- exposing `quafu.Client`, `quafu.Job`, or `quafu.Result` publicly;
- making the SDK a core dependency;
- retaining compatibility with the unmaintained SQC protocol;
- adding browser-login behavior to FlagQuantum library calls;
- changing QSteed into a transport client; or
- claiming live or dynamic hardware support without separately authorized
  evidence.

# API Change 093: Quafu Dynamic Format Source Audit

**Audit date:** 2026-10-26

**Outcome:** Unsupported by the current maintained SDK contract

**Stable execution state:** Fail closed

## Scope and provenance

This is the approved Stage A, static source audit from
[API Change Proposal 093](API_CHANGE_PROPOSAL_093_QUAFU_DYNAMIC_FORMAT_ACTIVATION.md).
It inspected the public `QUAFU-Quantum/quafu-sdk` repository only. It did not
construct a client, read credentials, contact a Quafu service, discover a
backend, or submit a task.

The reviewed source is:

- repository: `https://github.com/QUAFU-Quantum/quafu-sdk`;
- commit: `b54c6e5dadde81e0bba8529a4e03523794fd97c7`;
- package version: `0.2.7` from `pyproject.toml`;
- repository state: `main` at capture time; and
- release tag: none was advertised by the repository at capture time, so the
  full commit identity, not a mutable branch name, is the audit authority.

The result is scoped to this source revision. It is not a permanent statement
about Quafu hardware or a substitute for a future device-scoped capability
record.

## Maintained-source findings

Source links below are pinned to the reviewed commit.

| Required fact | Maintained evidence | Audit result |
|---|---|---|
| Accepted program dialect | [`src/quafu/_qasm.py` lines 2-6](https://github.com/QUAFU-Quantum/quafu-sdk/blob/b54c6e5dadde81e0bba8529a4e03523794fd97c7/src/quafu/_qasm.py#L2-L6) states that inputs are OpenQASM 2 text or Qiskit circuits exported through `qiskit.qasm2`. [`Client.submit`](https://github.com/QUAFU-Quantum/quafu-sdk/blob/b54c6e5dadde81e0bba8529a4e03523794fd97c7/src/quafu/client.py#L274-L301) documents the same boundary. | OpenQASM 2 only; no maintained OpenQASM 3 or other dynamic dialect contract. |
| Physical-program submission | [`Client.submit`](https://github.com/QUAFU-Quantum/quafu-sdk/blob/b54c6e5dadde81e0bba8529a4e03523794fd97c7/src/quafu/client.py#L283-L301) maps `compiler="none"` to a circuit whose qubit indices are already physical. [`_job_body`](https://github.com/QUAFU-Quantum/quafu-sdk/blob/b54c6e5dadde81e0bba8529a4e03523794fd97c7/src/quafu/client.py#L337-L350) sends `circuit_level="physical"`. | Confirmed for static physical OpenQASM 2; it does not confirm dynamic semantics. |
| Mid-circuit measurement and reset | [`qiskit_to_qasm`](https://github.com/QUAFU-Quantum/quafu-sdk/blob/b54c6e5dadde81e0bba8529a4e03523794fd97c7/src/quafu/_qasm.py#L38-L59) describes the real-device terminal-measurement requirement. The maintained error-contract test names `mid_circuit_measurement` and `reset` as `precheck.badCircuit` reasons in [`tests/unit/test_errors.py` lines 90-94](https://github.com/QUAFU-Quantum/quafu-sdk/blob/b54c6e5dadde81e0bba8529a4e03523794fd97c7/tests/unit/test_errors.py#L90-L94). | The reviewed contract does not admit the required operations on a real device. |
| Feed-forward and conditional operations | No public submission field, device capability, syntax profile, condition-width rule, supported-value rule, or conditional-operation set is defined in the reviewed source. | Unconfirmed and therefore refused. |
| Device-scoped dynamic capability | [`Client.metadata`](https://github.com/QUAFU-Quantum/quafu-sdk/blob/b54c6e5dadde81e0bba8529a4e03523794fd97c7/src/quafu/client.py#L369-L373) documents qubit count, basis gates, coupling map, and resource limits, but no dynamic-circuit fields. | No authoritative dynamic capability for Baihua or another target. |
| Idempotent task creation | [`_create_job`](https://github.com/QUAFU-Quantum/quafu-sdk/blob/b54c6e5dadde81e0bba8529a4e03523794fd97c7/src/quafu/client.py#L345-L354) creates one `client_ref`; the client retry policy reuses the same request body. | Confirmed at the SDK job boundary, but insufficient to activate a program format. |
| Final result interpretation | [`Result`](https://github.com/QUAFU-Quantum/quafu-sdk/blob/b54c6e5dadde81e0bba8529a4e03523794fd97c7/src/quafu/jobs.py#L44-L98) exposes final counts, bit order, classical width, requested/returned shots, terminal measurement mapping, executed QASM, compiler, calibration, and task identity. | Final-count contract confirmed. No ordered mid-circuit record contract exists. |

## Decision

The Stage A outcome is **unsupported by the current maintained SDK contract**.
This is stronger than merely observing that an example is absent:

1. the public input contract is OpenQASM 2;
2. the maintained real-device path requires terminal measurements;
3. the maintained precheck vocabulary explicitly includes mid-circuit
   measurement and reset as bad-circuit reasons; and
4. neither metadata nor results define device-scoped dynamic capabilities or
   ordered mid-circuit records.

Consequently, FlagQuantum must not serialize a `DynamicCircuit` into this SDK
path, silently lower feedback into a static circuit, substitute a simulator,
or infer support from `compiler="none"`. The existing stable entry remains
side-effect-free and fail closed:

```python
import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit
from flagquantum.errors import CapabilityError

circuit = DynamicCircuit(2).measure(0, classical_bit=0)
circuit.conditional("x", 1, classical_bit=0)

try:
    fq.run(circuit, target="quafu:Baihua", shots=1024)
except CapabilityError as error:
    assert "dynamic_remote_preflight_failed" in str(error)
```

Expected result: FlagQuantum raises before importing or constructing the Quafu
provider, reading credentials, contacting the service, or creating a task.

## What would reopen activation

A newer maintained source may reopen Stage A only if it provides all of the
following for one named device and versioned service contract:

- an accepted dynamic program dialect and version;
- authoritative mid-circuit measurement, reset, repeated-measurement,
  feed-forward, and condition semantics;
- the physical-program and compiler-preservation rules;
- device-scoped capability and expiry fields;
- deterministic unsupported-feature errors; and
- final-count and ordered mid-circuit-record result semantics.

If those facts appear, pin the new SDK/service revision and repeat Stage A.
Maintainer confirmation, offline fixtures, stable-facade activation, and any
bounded live validation remain separately gated by the original proposal.

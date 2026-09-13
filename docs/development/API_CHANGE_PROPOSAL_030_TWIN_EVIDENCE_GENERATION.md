# API Change Proposal 030: Identity-bound Twin evidence generation

## Status

**Implementation authorized; pending API freeze review.** After merging the
offline persistence lifecycle, the user authorized the next Twin slice: turn a
bound hardware validation result into evidence without asking users to assemble
an envelope manually.

## Problem

`TwinExperiment` binds a prediction, submitted program, receipt, and hardware
result, but `TwinEvidenceEnvelope` is still constructed manually. More
importantly, accepting an arbitrary caller-supplied OpenQASM program does not
prove that it represents the FlagQuantum circuit whose prediction is being
validated. Automatically granting verified evidence without that missing link
would be unsound.

## Decision

- Make `submitted_qasm` optional in `TwinExperiment.prepare()`.
- When omitted, emit deterministic OpenQASM 2.0 directly from the supplied
  FlagQuantum circuit and freeze it before dispatch.
- Add `TwinExperiment.evidence_from_report(...)`.
- Require matching experiment, provider/backend, circuit identity, canonical
  IR-to-QASM text, submitted program identity, executed program identity,
  physical mapping, counts, validation report, and shot count. The executed
  program must come from the provider's explicit transpiled-program field; an
  echoed input circuit is insufficient.
- Derive a conservative exact-circuit TV error radius by adding the observed
  Twin-to-hardware TV distance to a 95% multinomial finite-shot radius, capped
  at one.
- Grant no structural estimate for unseen circuits.

## Public surface

```python
experiment = fq.twin.TwinExperiment.prepare(
    twin,
    circuit,
    name="frozen-bell",
    shots=1024,
)

# Submit and retrieve through the explicit provider workflow.
hardware_report = experiment.validate_result(result, receipt=handle)
evidence = experiment.evidence_from_report(
    hardware_report,
    circuit=circuit,
    confidence_level=0.95,
)
fq.twin.dump_evidence(evidence, "twin-evidence.json")
```

Existing callers may continue to pass `submitted_qasm`. Such experiments remain
valid retrospective diagnostics, but evidence generation rejects them unless
the text exactly equals FlagQuantum's canonical emission for the frozen circuit.

## Statistical meaning

For `K` measured outcomes, `N` shots, confidence `1 - delta`, and observed TV
distance `d`, the verified radius is conservatively bounded by:

```text
min(1, d + sqrt((log(2^K - 2) + log(1 / delta)) / (2N)))
```

This is a finite-sample bound on the measured output distribution. It is not
state fidelity, amplitude accuracy, or a guarantee for a different circuit.

## Boundary

Evidence generation does not submit or poll tasks, fetch calibration, train or
promote a model, route workloads, or make application decisions. Provider I/O
remains an explicit `TwinExperiment`/Remote operation. Agent, natural-language,
and MCP behavior remain outside FlagQuantum.

## Compatibility

The change is additive for existing calls: explicit `submitted_qasm` remains
accepted. Existing experiment, hardware-report, envelope, and report schemas do
not change.

## Acceptance

- The shortest path emits deterministic QASM without user-supplied text.
- A fully matching later result produces exact-circuit evidence.
- The error radius includes the recorded observation and finite-shot bound.
- Custom, rewritten, missing, foreign, or tampered identities fail closed.
- Evidence for the exact circuit does not imply support for unseen circuits.
- Documentation and PR description include complete usage examples.

# IR-006: Phase 1 Reversible Static Scope

Status: Approved
Date: 2026-09-01
Applicable phase: Phase 1 importer, round-trip, and differential bridge.

Approval record: the API owner explicitly approved IR-004 through IR-006 on
2026-09-01. Approval covers only the internal `circuit_ir_v1_static` profile. It
neither promotes dynamic, timing, pulse, or provider-native capabilities to
supported status nor authorizes replacing the default runtime.

## Background

CircuitIR-to-QuantumIR-to-CircuitIR round-trip needs a precise support boundary.
Describing every constructible CircuitIR as reversible would incorrectly include
dynamic metadata, provider evidence, unknown extensions, and semantics lacking
typed models.

## Decision

The internal Phase 1 profile is named `circuit_ir_v1_static` and supports:

- the 35 canonical opcodes currently registered in `OPERATOR_SCHEMAS`;
- unitary and channel schemas, with channels satisfying current typed semantic
  marker rules;
- concrete custom unitaries within IR-004's approved scope;
- Python/static scalars, Parameter, ParameterExpression, and IR-005 binding types;
- n_wires, instruction order, wire order, dtype, and shape constraints;
- observables and request-style measurements, split and reassembled through
  IR-001's composite object;
- classified metadata with typed destinations.

Phase 1 explicitly rejects:

- `is_dynamic`, condition/conditions, and classical-bit feedback;
- functions, calls, loops, branches, regions, and controller programs;
- timing, pulses, calibration, and provider-native instructions;
- unclassified metadata that could affect semantics/identity;
- matrices outside IR-004's scope;
- parameters without deterministic encoding or gradient preservation;
- any structure requiring lossy conversion back to CircuitIR schema 1.0.

Successful round-trip requires exact equality of canonical `CircuitIR.to_dict()`
payloads. Semantic equivalence after compilation optimization is outside Phase 1's
round-trip definition.

## Support States

The importer returns exactly one status per input:

```text
supported_exact
unsupported_with_diagnostics
invalid_input
```

No `best_effort`, implicit metadata omission, or default lossy mode is allowed.

## Rejected Alternatives

- **Support every CircuitIR**: contradicts current metadata/dynamic behavior.
- **Support only unitary gates**: omits current channel and execution-request boundaries.
- **Allow silent loss**: undermines scientific semantics and round-trip trust.

## Compatibility and Rollback

The profile is an internal importer capability label, not public API. Removing the
importer/profile leaves legacy paths unaffected. Expanding scope requires
fixtures, diagnostics, differential evidence, and an ADR revision.

## Acceptance

- The 35-opcode manifest has passing coverage.
- Supported fixtures round-trip canonical payloads exactly.
- Dynamic/unknown/lossy fixtures fail closed.
- Request splitting/reassembly preserves ordering, shots, and metadata.
- Public schemas/APIs remain unchanged.
- [x] API owner approved the profile scope.
- [ ] Compiler/runtime owners confirm importer and differential bridge details during review.

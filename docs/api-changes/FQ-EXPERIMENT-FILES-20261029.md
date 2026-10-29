# API change: portable experiment files (`fq.save`, `fq.load`)

## Authorization and scope

Implementation is explicitly authorized by the user's 2026-10-29 request to
prepare a PR implementing `save` and `load` and finalize the extension, following
the agreed path-first API and `.fqx` naming discussion. This records that user
instruction, not a fabricated approval token or completed CODEOWNER review.
Normal API-owner review and merge approval remain required on the PR.

Integration owns the additive contract and root exports. Runtime owns archive
I/O and result data serialization. No existing IR, ExecutionPlan, ExecutionResult,
ProgramArtifact, training checkpoint or deployment schema is changed. The
historical public API baseline is not regenerated.

## User journey and names

A script must save its actual compiled program and returned measurements without
importing a Web-specific module. The same file can be inspected by CLI, Web and
notebooks. `save` writes this experiment; `load` reads it without computing,
submitting, selecting a backend, executing code or drawing new samples. These two
infrastructure-neutral operations belong at the `import flagquantum as fq` face.
The return type is `flagquantum.runtime.experiment.Experiment`, a frozen record of
an existing `CircuitIR`, an optional existing `ExecutionResult`, and historical
execution data, not a second IR.

```python
fq.save(path, program, *, result=None, overwrite=False) -> None
fq.load(path) -> Experiment
```

`path` is a local `str` or `os.PathLike[str]`. `program` accepts Circuit, CircuitIR
or a static CircuitIR ExecutionPlan, or an Experiment previously returned by
`load`. Saving an Experiment preserves its recorded execution data; the separate
`result` argument must be omitted. A saved result requires its retained static
ExecutionPlan. Planless/provider/native-opaque results fail explicitly in 1.0.
An execution plan input must match the result's plan identity. A circuit/IR input
must match the retained executed program, ignoring only runtime configuration
metadata and measurements added by Runtime when the supplied program had none.
No recompilation is used to manufacture a match. If optimization or scheduling
changed the supplied representation, pass `result.plan` to archive the exact
executed program. Mutated programs and mismatched result plans are rejected.

## Format decision

- Extension: **`.fqx`**, named **FlagQuantum Experiment**.
- Schema: `flagquantum.experiment`; version: `1.0`.
- Proposed media type: `application/vnd.flagquantum.experiment+zip` (not a claim
  of IANA registration).
- Container: ordinary ZIP, no encryption, ZIP64, comments or extracted paths.
- Required members: `manifest.json`, `circuit.json`; optional `result.json`.
- Arrays: typed little-endian raw members `arrays/0000.bin`, etc., referenced by
  dtype and shape. This requires neither NumPy nor pickle. Only recorded arrays
  are saved; never request or materialize an absent state from a backend.
- Manifest: schema/version, SDK producer, exact CircuitIR content hash and a
  complete member table containing byte lengths and SHA-256 digests. Checksums
  detect corruption; they are not authentication or signatures.
- `circuit.json` is the existing CircuitIR 1.0 serialization, unchanged.
- `result.json` retains its actual static ExecutionPlan, measurement order,
  qubit order, shots, dtype/shape, numeric values, accuracy and diagnostics.
- Loaded `result.plan` is `None`; `experiment.execution_plan` exposes a fresh
  dictionary of historical plan data. Loading does not turn an old plan into
  execution authority or weaken live plan identity checks.
- Loaded tensors are detached owned CPU data, preserving dtype and shape. An
  experiment is not an autograd, optimizer or distributed-process checkpoint.
- `save` requires `.fqx` (case-insensitive), refuses an existing destination by
  default and publishes only a completed archive atomically. `overwrite=True`
  explicitly authorizes replacement. `load` identifies content, even if renamed.
- Readers fail closed on unsupported versions, missing/unknown/duplicate members,
  inconsistent program/plan identities, duplicate JSON keys, invalid tensor
  descriptors, checksums, truncated data and declared/actual size overruns.
- Limits: 256 members, 16 MiB per JSON member, 64 MiB per tensor and 256 MiB total
  uncompressed data. ZIP directory and archive size are bounded before parsing.

`.py`, `.ipynb`, `.qasm`, existing checkpoints and deployment files keep their own
formats. No `.fqc` or `.fqr` extension is introduced. Web auto-collection and
removing `flagquantum_web` are follow-up client work, not implicit SDK behavior.

## Compatibility and acceptance

Only `save` and `load` are added at root. Existing signatures, defaults and wire
schemas remain unchanged. The candidate contract, source-of-truth manifest,
behavior contracts, format documentation and release notes accompany the PR.
Tests must demonstrate Bell execution/save/load/replay without resampling during
I/O; program-only files; typed/batched measurements; stale program/result refusal;
atomic replacement; malicious/corrupt input rejection; lazy imports; exact API
signatures; and unchanged protected APIs. No provider or paid-resource tests.

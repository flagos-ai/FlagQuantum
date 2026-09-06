# Interoperability Adapter Development

FlagQuantum framework adapters implement the candidate-stable v1 protocol as
optional control-plane translators. They
convert external objects at the versioned `CircuitIR` boundary and must not
enter simulation, training, accelerator, or distributed runtime layers.

## Required structure

Each adapter lives under `flagquantum.ecosystem.<name>` and implements
`InteropAdapter`. Its descriptor belongs in the immutable lazy registry; the
descriptor may be imported without importing the external framework.

Protocol stability does not stabilize an adapter implementation. Each adapter
retains its own dependency window, operation mapping, issue codes and maturity.

The adapter dependency must have its own optional extra and be classified as
`interop` in `dependency-policy.toml`. Importing `flagquantum`,
`flagquantum.ecosystem`, or resolving the adapter object must continue to work in
a core-only installation.

## Common conformance suite

Use the framework-neutral suite before adding framework-specific numerical or
provider tests:

```python
from flagquantum.ecosystem import (
    InteropRejectionCase,
    InteropRoundTripCase,
    get_adapter,
    run_adapter_conformance,
)

result = run_adapter_conformance(
    get_adapter("example"),
    (InteropRoundTripCase("bell", bell_ir),),
    rejection_cases=(
        InteropRejectionCase(
            "unsupported-control-flow",
            "import",
            external_control_flow,
            "unsupported_control_flow",
        ),
    ),
)
assert result.passed, result.to_dict()
```

The suite checks:

- adapter protocol identity and `INTEROP_API_VERSION` compatibility;
- typed import/export results and adapter-owned conversion reports;
- lossless FlagQuantum IR export/import round trips;
- strict fail-closed behavior for unsupported values;
- explicit `allow_lossy=True` diagnostics when partial conversion is supported;
- a stable, machine-readable `flagquantum_interop_conformance_v1` result.

`semantic_fingerprint()` excludes transport provenance while retaining
instructions, observables, measurements, wire count, IR version, and optional
adapter-declared semantic metadata. An adapter must supply metadata such as a
global phase when that metadata changes executable semantics.

## Adapter-specific evidence

The common suite is necessary but not sufficient. Every adapter also needs a
versioned compatibility contract covering its supported versions, operation
mapping, parameter order, wire and bit order, symbolic parameters, unsupported
constructs, and issue codes. Where the external framework has an independent
simulator, compare deterministic complex128 reference states or observables in
addition to IR fingerprints.

Provider login, job submission, hardware behavior, and performance require
separate deployment and hardware evidence. Passing adapter conformance does not
install the dependency, certify a provider, or authorize runtime fallback.

## Review boundary

Reject an adapter change if it:

- adds the framework to core dependencies or imports it outside its adapter;
- stores external objects in FlagQuantum IR or runtime records;
- silently drops unsupported semantics;
- modifies device selection, Torch-FL, kernels, collectives, or CUDA behavior;
- treats import success or a smoke run as hardware certification.

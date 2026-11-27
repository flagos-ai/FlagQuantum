# Remaining Stim integration capabilities

Baseline: actual main `e399353303bd4b752c88686fe572cf0943e8801e`, after
[#1235](https://github.com/flagos-ai/FlagQuantum/pull/1235), inspected on
2026-11-20. This is an operation inventory, not a full upstream API-parity claim
or approval to change a public contract. Historical candidate sections retain
their original evidence and approval state.

## Current boundaries

| Workflow | Current implementation | Remaining boundary |
| --- | --- | --- |
| Native memory export | `adapters.stim.memory_to_stim_text` uses `_circuit_text` and the embedded parser in opt-in builds; marker-free builds use Stim | No general upstream circuit-generation API is introduced |
| Original noisy text experiment | `run_stim_experiment` uses embedded sampling and strict DEM analysis, then native decoding | Existing instruction/resource limits and strict-analysis refusals remain |
| External measurement table conversion | `convert_measurement_records` calls `_record_conversion._convert_stim_measurements`, which parses through the SDK | The embedded profile still needs the Python SDK for this operation, including supplied per-shot sweep bits |
| Compiled native record conversion | `_prepare_explicit_record_conversion` chooses the embedded converter for supported native IR | The native binding supplies a zero-width sweep table; this does not implement external per-shot sweeps |
| Measurement file conversion | `CompiledMeasurementsToDetectionEventsConverter.convert_file` validates bounded files | Explicit sweep files and selected PTB64 outputs are refused |
| Shot-file reading/writing | Existing readers, iterator and writer cover six encodings with declared bounds | Iterator requires an embedded build; zero-column and other existing refusals remain |
| Independent DEM sampling | `CompiledDemSampler` supports packed output, error recording/replay and six-format files | Exclusive `error_id` groups are refused; use the separate existing `dem_sampling` route |
| DEM text | The canonical parser handles bounded nested REPEAT and detector shifts | It is not an unrestricted upstream object model; unsupported text is rejected |
| Interactive stabilizer algebra | Vendor internals include tableau and Pauli machinery | No equivalent public Stim PauliString/Tableau/FlipSimulator surface is claimed |
| Statistical and hardware claims | Fixed-shot reports and bounded CPU development evidence exist | No new adaptive-stopping, threshold, general GPU-decoding or hardware claim follows |

Authoritative code: [record conversion](../../flagquantum/simulation/stabilizer/_record_conversion.py),
[public conversion](../../flagquantum/qec/execution/measurement_conversion.py),
[embedded binding](../../flagquantum/simulation/stabilizer/csrc/embedded_sampler.cpp),
and [DEM ownership](../../flagquantum/qec/dem/README.md).

## Recommended next implementation slice

First replace the SDK implementation behind the existing external
`convert_measurement_records` boundary in embedded builds. Preserve its exact
signature, CPU bool/integer and packed-uint8 inputs, ownership, padding,
zero-shot behavior, deterministic-reference validation, exception categories
and resource limits. Accept both omitted and explicitly supplied per-shot sweep
controls under that existing contract. Keep marker-free SDK behavior intact.
Do not route external text through native IR or silently discard sweep bits.

Simulation owns the numerical binding and backend selection; QEC retains input
validation and its existing `DemSample` result. Reuse the pinned 98-file Stim
1.16.0 closure if the required primitives are already present. Before code,
check the closure and record any missing upstream file as a separate scope
change. Do not add another parser, result class or dependency.

Acceptance must include an unchanged consumer with a replacement backend,
independent fixed-record expected parities, all four measurement/sweep packing
combinations, nonzero reference samples, zero rows, alias/padding errors and
nondeterministic-parity rejection. Compare against the installed Stim 1.16.0
SDK, then run the same public workflow from a fresh no-SDK installed wheel.
Run focused tests before the required smoke/unit and integration tiers.

Sweep-file conversion and PTB64 output should follow as separately bounded
changes. Their current explicit refusals are protected behavior; changing them
requires a concrete API-owner proposal and approval. Do not bundle a new
interactive tableau API, categorical DEM sampler or threshold study into this
slice.

## Reproduce the supported user workflow

Install an opt-in embedded wheel, then run:

```python
from flagquantum.qec import PhenomenologicalNoise, RepetitionCode, build_memory_circuit
from flagquantum.qec.adapters.stim import memory_to_stim_text
from flagquantum.qec.execution.experiments import run_stim_experiment

memory = build_memory_circuit(RepetitionCode(3), rounds=3)
text = memory_to_stim_text(memory, noise=PhenomenologicalNoise(
    data_flip=0.005, phase_flip=0.007, measurement_flip=0.009,
))
result = run_stim_experiment(
    text, decoder="sparse_blossom", shots=64, batch_size=16, seed=5303,
)
print(result.shots, result.logical_failures, result.sampling_source)
```

The accepted fixed-seed output is `64 0 stim_circuit`. This validates the
existing workflow supported by this documentation; it does not present the
proposed conversion replacement as implemented.

## Actual-main acceptance

The [installed receipt](../../artifacts/development/qec_sdk_free_main_acceptance_20261120.json)
records a fresh wheel from the baseline above, four d=3 family workflows,
and the complete 28-configuration Toric matrix. Every sampled syndrome was
reconstructed. The external sweep conversion probe passed with Stim 1.16.0
and reported the existing dependency error without the SDK. This scope does
not requalify full distance matrices for the other three families or certify
another platform. Existing CI and full regression evidence was not rerun.

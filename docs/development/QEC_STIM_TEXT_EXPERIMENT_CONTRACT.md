# Fixed-shot experiments from supported Stim circuit text

## Approval and canonical surface

The API owner approved implementation and normal delivery on 2026-11-05 after
reviewing the signature, separate result record, strict source law, supported
and refused domains, lifecycle, user example and ten-path proposal. The owner
separately approved the eleventh path to update only the existing experiment
module export assertion from two names to the four approved names. Final
reviewed-head and actual-main acceptance remain pending.

Only `flagquantum.qec.execution.experiments` defines and exports the two new
names. Existing root/parent exports, `MemoryExperimentResult`, registry entries,
noise law, stable fixtures and serialization behavior are unchanged.

```python
@dataclass(frozen=True)
class StimExperimentResult:
    shots: int
    logical_failures: int
    logical_error_rate: float
    observable_failures: tuple[int, ...]
    confidence_interval: tuple[float, float]
    seed: int | None
    sampling_source: Literal["stim_circuit"]


def run_stim_experiment(
    circuit_text: str,
    *,
    decoder: str,
    shots: int,
    batch_size: int,
    seed: int | None = None,
) -> StimExperimentResult:
    ...
```

The seven fields have the stated order and no constructor defaults. Returned
counts and seed use normalized Python integers, tuples contain immutable scalars,
and the confidence interval uses the existing two-sided 95% Wilson formula.
No save/load, portable pickle/JSON schema or cross-version persistence promise
is introduced. A distinct record preserves the existing memory result's exact
`Literal["memory_circuit"]` instead of broadening its public field.

The machine-readable surface is
[`qec-stim-text-experiment-v1.json`](../../contracts/qec-stim-text-experiment-v1.json).

## Source and supported domain

The input is an immutable string accepted by the existing annotated Stim bridge.
Positive non-boolean Integral shots and batch size and the existing optional
unsigned 64-bit seed normalize exactly as the memory experiment does. Strings
and floats are not coerced. At least one detector and observable are required;
observable index holes retain the declared width. Decoder selection uses the
registered name and constructor defaults, with the same exact certified concrete
types as complete-result independent-shot batching.

Analyze the original text through `stim_circuit_to_dem_text` with
`decompose_errors=False`, then the existing strict DEM reader. Keep correlated
and exclusive mechanism semantics and all owning decoder refusals. Do not merge,
decompose, approximate, twirl or silently change the chosen decoder. Admit the
selected decoder through the existing empty-batch operation before compilation.

Compile one stochastic sampler from the original text. DEM analysis and sampling
need not share a parse. Sample exactly `min(batch_size, remaining)` until the
requested total completes; 35 shots and size 16 give 16, 16, 3. The DEM is never
substituted for the sampling source. Actual logical flips are evaluation labels
and never enter decoder input. Predictions are compared over every logical
column to count any-logical and per-observable disagreements. Failed shots cannot
vanish from the denominator.

## Failure and cleanup

| Condition | Behavior |
| --- | --- |
| Invalid source/scalar/name | Existing TypeError/ValueError/seed ValidationError before sampling |
| Missing optional SDK | Existing lazy Stim/PyMatching dependency error; no install or fallback |
| Unsupported circuit/model/decoder | Existing CapabilityError or owning preparation error |
| Zero detectors or observables | ValueError before sampler compilation |
| Invalid sample/prediction buffer | ValidationError and owned sampler cleanup |
| Sampling/decoding failure | Original error, no partial successful record |
| Primary and cleanup failure | Preserve the primary error with cleanup as cause |
| Cleanup alone fails | Propagate it; do not return success |

The private aggregation implementation is shared with the existing memory entry.
Its preparation order, result shape, exception precedence and interval semantics
remain regression-tested. No cross-call cache, vendor result object, new backend
selector, noise parameter, adaptive stopping, decoder option map or threshold
analysis API is introduced.

## User workflow and acceptance

With the existing `stim,pymatching` extras, run:

```sh
python examples/qec/stim_memory_experiment.py
```

Expected output:

```text
35 0 (0, 0)
stim_circuit
```

The example uses two independent faults with probabilities 1/8 and 1/4. D0 maps
to L1 and D1 to L0; all four syndromes have the corresponding logical prediction.
Their joint probabilities are 21/32, 3/32, 7/32, 1/32 in fault-bit order. This is
an analytic conformance example, not a threshold or performance claim.

Pure tests independently check four-case probabilities, asymmetric labels,
nonzero report counts, Wilson endpoints, tail draws, validation order and failure
ownership. Existing native-memory and bridge/decoder tests remain applicable.
Required installed profiles exercise absence in core/PyMatching-only, native
choice in Stim-only, and PyMatching in combined wheel/sdist profiles. New report
acceptance is recorded separately from the original four sample cells. All
SDK cases must actually run in their required installed profiles.

See the [migration guide](../guides/STIM_USER_MIGRATION.md) and
[central delivery ledger](../roadmap/QEC_STIM_PYMATCHING_DELIVERY_PLAN.md).

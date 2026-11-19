# Bundled Sparse Blossom fixed-shot experiments

Status: implementation candidate for API-owner review.

## Contract

Accept `decoder="sparse_blossom"` in `run_memory_experiment` and
`run_stim_experiment` under `flagquantum.qec.execution.experiments`. The
selector is scoped to these entry points. Existing named decoders retain their
matching choice and full-record decode path. The new path reads dense CPU
detection-event batches and returns the existing immutable experiment result
records.

The original annotated circuit drives all sampling. The backend uses Stim's
graphlike decomposition for the matching graph, merges identical detector and
observable signatures, and admits one edge per detector pair. For exact DEM
analysis, equivalent uniform Pauli channels use their depolarizing
representation. Unsupported circuit or graph conversions propagate a
capability error. Weight discretization can tie paths with close costs.

## Ownership and evidence

QEC owns the decoder selection, graph, predictions, and experiment statistics.
The pinned Apache-2.0 Sparse Blossom source and lock remain behind the private
native decoder binding. Installation does not require the external PyMatching
Python distribution. The Stim sampler retains its existing installation
requirements.

Regression coverage includes batched logical prediction, the original source
sampling law, uniform gate-bound noise, partial batches, and real isolated
wheel installation. The platform wheel smoke runs both public experiment entry
points in an environment without PyMatching.

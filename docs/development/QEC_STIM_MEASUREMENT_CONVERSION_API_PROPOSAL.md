# External measurement-record conversion capability bundle

Status: approved contract with a locally validated implementation candidate.
Proposed PR title: feat(qec): convert external Stim measurement records.
Primary workflow: CPU local QEC; single_device_fast_path. Simulation owns the
numerical SDK boundary. Protected API approval is required before implementation.

## Verified gap and reuse

The pinned Stim 1.16.0 local stub exposes Circuit.compile_m2d_converter and
CompiledMeasurementsToDetectionEventsConverter.convert, including raw bool or
LSB-packed uint8 measurements, optional sweep controls, reference normalization
and separate detector/logical outputs. The verified source/distribution identity
is retained in this directory's upstream-identity records; this proposal does
not substitute a different installed package or a moving documentation version.

Current qec/adapters/stim.py exposes memory_to_stim_text, compile_stim_sampler
and stim_circuit_to_dem_text. compile_stim_sampler creates shots; it cannot accept
pre-existing raw records. tests/qec/test_stim_experiment_bridge.py uses upstream
m2d conversion as an independent reference, not as a FlagQuantum public workflow.
MeasurementMap describes native memory parity rows and does not own arbitrary
external-circuit reference samples or sweep-controlled readout normalization.
Reuse that evidence; do not relabel it acceptance for the proposed new entry.

This implementation starts from main revision
`03017ddf96dd2d9604f2dbc4d2cdfe2f3837f878`, which contains the existing Stim
experiment bridge. The conversion depends on that bridge's parser and
reference check and is independent of matrix-defined matching.

## Exact proposed public entry

Canonical module: flagquantum.qec.adapters.stim. Add only:

    def convert_stim_measurements(
        circuit_text: str,
        measurements: torch.Tensor,
        *,
        sweep_bits: torch.Tensor | None = None,
        bit_packed_measurements: bool = False,
        bit_packed_sweep_bits: bool = False,
    ) -> DemSample:

Add this name only to that module's explicit __all__. No root re-export, new
public result type, serialization schema, registry or dependency is proposed.
Reuse DemSample as the existing detector/logical-flip batch representation.
Its fields and constructor are unchanged; this path validates both tensors
before constructing it and produces owned CPU bool storage.

The two flags explicitly select encoding independently. Dense data is 2D CPU
strided bool or binary integer data; packed data is 2D CPU strided uint8, LSB first.
Measurements have exactly num_measurements columns, or ceil(count/8) packed bytes.
Supplied sweep controls have the same shot count and exactly num_sweep_bits
columns or corresponding packed width. None means upstream's zero sweep controls.
Noncanonical padding is rejected. A packing flag without supplied sweep data is
rejected rather than silently ignored. Reject non-bool flags, GPU, sparse,
floating/complex, nonbinary, wrong rank/width/row count and invalid source early.
Zero-shot input preserves detector/logical widths after validating circuit and
input. Zero-observable results have shape (shots,0), not an omitted field.

Reference normalization is mandatory. Do not expose skip_reference_sample=True,
substitute an all-zero reference or replace the input circuit by a DEM. Preserve
the existing deterministic-noiseless detector/logical-parity requirement and its
CapabilityError behavior. Sweep-control conformance must prove normalization
against the actual per-shot controls. If that cannot be established within this
contract, stop with evidence; do not silently drop sweep controls or weaken the
refusal. Stim objects remain inside simulation/stabilizer/engine.py. QEC handles
public tensor validation and result composition; Simulation returns only owned
vendor-neutral tensors, with retained exception causes and no process lifecycle.
Existing parser limits (1 MiB text, 4096 qubits, one million expanded instructions
and one million recorded columns, at least one detector) remain unchanged.

## Scope: fourteen paths

- `flagquantum/qec/adapters/stim.py`
- `flagquantum/simulation/stabilizer/engine.py`
- `flagquantum/qec/adapters/README.md`
- `flagquantum/simulation/stabilizer/README.md`
- `tests/qec/test_stim_measurement_conversion.py`
- `tests/team/simulation/test_stim_record_conversion_conformance.py`
- `tests/unit/test_qec_installed_profiles.py`
- `tools/qec_installed_smoke.py`
- `examples/qec/stim_measurement_conversion.py`
- `examples/README.md`
- `docs/development/QEC_STIM_MEASUREMENT_CONVERSION_API_PROPOSAL.md`
- `docs/roadmap/QEC_STIM_PYMATCHING_DELIVERY_PLAN.md`

- `contracts/qec-stim-measurement-conversion-v1.json`
- `tests/qec/test_stim_experiment_bridge.py`

The approved contract amendment adds the new signature separately and extends
the existing export and missing-SDK assertions. The original contract is retained.

## Contract-first and replacement acceptance

First commit the approved exact API/semantics document, a fake Simulation seam
and conformance cases. Implement the same contract afterward in the same PR.
Replace the private conversion engine in a test while keeping the QEC consumer
and example inputs unchanged. Do not add a public converter handle, manager,
cache or lifecycle solely to forward one stateless operation. Existing sampler
lifecycle, cleanup and seed behavior are untouched; no random sampling is needed
for fixed-record conversion tests.

## User example (planned, not executed)

```python
import torch
from flagquantum.qec.adapters.stim import convert_stim_measurements

text = "R 0\nX 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
raw = torch.tensor([[1], [0]], dtype=torch.bool)
result = convert_stim_measurements(text, raw)
print(result.detectors.tolist())
print(result.observables.tolist())
```

Expected by independent parity reasoning (not yet execution evidence):

```text
[[False], [True]]
[[False], [True]]
```

The noiseless reference is one, so raw zero is a flip. Also demonstrate packed
input for this same fixed record and sweep-controlled deterministic references.
The PR description must include the final reviewed head's actual runnable code,
command, prerequisites and observed output from outside the checkout.

## Acceptance matrix and commands

- Pure contract/fake cases: all four measurement/sweep input encodings, omitted
  sweeps, zero rows/observables, noncontiguous ownership, malformed SDK dtype/
  shape/nonfinite outputs, retained causes and unsupported-domain refusals.
- Independent fixed parity references: ideal-one versus ideal-zero, repeated
  detector parity, observable gaps, multiple measurements crossing byte boundary,
  reset/readout basis and explicit per-shot sweep controls. Test raw records,
  not statistical similarity from two independent random streams.
- Pinned upstream comparison: exact bool values and shape from one independent
  Stim converter call on identical records for every supported input layout.
- Replacement test: unchanged QEC consumer against a fake conversion engine.
- Missing-SDK path and SDK-free imports stay usable in core-only installations.
- Extend existing four-wheel-profile and sdist acceptance hook. No local install
  is implied; use already available SDK/core environments for source checks.

```bash
python -m pytest -q tests/qec/test_stim_measurement_conversion.py tests/team/simulation/test_stim_record_conversion_conformance.py tests/qec/test_stim_experiment_bridge.py tests/unit/test_qec_installed_profiles.py
python -m examples.qec.stim_measurement_conversion
python tools/check_qec_installed_profiles.py --dist-dir dist --report-dir /tmp/qec-installed-reports
```

Run Black24.1, minimum Ruff0.15, strict Mypy/Python3.12 on touched Python files,
ownership/architecture/dependency/English/hygiene/docs/source-of-truth checks,
and the existing example inventory test. Inspect existing marker reachability;
never add skips to hide a new optional integration path.

No sampler, subprocess, research probe, numerical worker or campaign is needed
for the fixed-record development checks. Use explicit command timeouts and
retain the first failure, then stop without automatic retry. No installation,
runner/workflow/dependency/threshold modification, historical evidence rewrite,
CI waiting or CI recovery is included. Publishing an approved implementation PR
may start its normal automatic checks; do not manually dispatch or recover them.

## Honest completion and remaining full-parity gaps

This bundle closes supported external-record conversion, not all Stim APIs.
Compiled reusable conversion lifecycle, packed output containers, convert_file,
CLI/format coverage, gauges and skip-reference behavior remain explicit gaps.
Existing DemSample remains dense bool; do not pretend packed output is supported
by the input packing flags. Full Stim/PyMatching parity, native replacement,
performance, FTS-02 baseline, R08 scale, and M0-M5 completion require their own
acceptance evidence. This extends measurement/sampling conformance relevant to
FTS-04 and the QEC workflow foundation, without certifying their complete gates.

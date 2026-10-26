# Ising CNN layout verification and prospective metadata correction

Date: 2026-10-25. This is a zero-sample input-verification slice for the
[frozen comparison protocol](QEC_AI_ISING_CNN_COMPARISON.md). Its budgets, model
recipes, streams, loss, inference choices, statistical method and quality gates
remain unchanged. No formal training, scoring, calibration, validation, physical
probe or primary samples have been collected for that new study.

## Correct the metadata from actual measurement identities

The original protocol states that subsequent stabilizer rounds are ordered Z/X.
Inspection of the pinned flattened circuit shows **X/Z**, not Z/X. Preserve the
original protocol document and this explicit correction together; do not silently
rewrite the preregistration or force the implementation to reproduce its mistaken
metadata. The corrected original-detector-index interpretation is:

- indices 0..39: initial X checks, time 0;
- each of eight subsequent 80-detector blocks: 40 X then 40 Z checks, times 1..8;
- indices 680..719: terminal X checks including final data support, time 9.

The upstream source's detector-emitting branch names are insufficient evidence
of check identity. At detector index 40, the actual absolute record pair is
(0,80); both records measure the first X ancilla (qubit 81) using MRX, at adjacent
rounds. The first Z pair, at detector index 80, is (40,120), measuring qubit 121
using MR. The first subsequent detector is therefore X even though the emitting
source branch is named `get_Z_detectors`. This observation is about the pinned
circuit record schedule. It does not establish a bug in final upstream decoding,
explain earlier native inactivity, or alter any earlier measured result.

The source bundle digest is the previously pinned
`ed9e6c56f263e0cd9fcc04a494e9cf6544c9f78cbb54c788fcce71b2e961b70e`.
The reconstructed d9/R9/X/XV circuit digest is
`0fa86bd7b534da215f58cb4611236db52d60ddc588da819b0d547a91026195e2`,
matching the archived input. These are source/circuit metadata checks, not fresh
noise realizations. The future formal manifest must bind both the original
protocol and this explicit correction before any data. All original
formal-execution gates still apply.

## Implemented boundary and independent checks

The optional `examples/qec/ising_layout.py` adapter parses the flattened circuit
and records each measured qubit, measurement/reset operation and absolute record
index. It verifies the complete MRX/MR round sequence and terminal MX data order.
For every original detector it checks exact record identity, adjacent-round time,
check type and terminal data support. It refuses unsupported instructions,
missing/duplicate references, altered order or basis, and changed terminal parity.

Vendor stabilizer rows remain the input authority. Their binary supports are
independently checked against adjacent data-qubit coordinates, with full data
coverage, nonoverlapping identities, unique check sites and integer/even check
coordinates. The grid uses row=vendor y/2 and column=vendor x/2; data-qubit numeric
index is not an axis convention. The volume is (10,10,10), with per-time counts
40,80,80,80,80,80,80,80,80,40; X count is 400 and Z count is 320. All 720 original
input bits are retained, including terminal complement bits.

The pinned-source entry reuses `compare_ising.reference(..., load_model=False)`
and `training_dem`, verifies vendor module origins and archived circuit/DEM
hashes, then calls the existing `SpacetimePredecoder.from_dem`. It neither loads
Fast weights nor changes its settings. A dependency-induced DEM mismatch fails
closed before constructing the native model; a locally different Stim/Torch
environment is not a substitute for matching-environment verification.

The encoding check visits zero, one and every one-hot row in batches <=32. It
checks original-order round trips and invalid-site zeros without invoking a
model forward, sampler or optimizer. Tiny analytic measurement-record fixtures
independently state expected parities and sites, and test malformed inputs.
Those fixtures do not authenticate a real vendor checkout or constitute actual
physical/device-parity evidence.

## Executable verification and remaining gates

Inside the pinned producing image/dependency environment, with the authorized
vendor source mounted read-only and repository source on `PYTHONPATH`, run:

```bash
python -m examples.qec.ising_layout --ising-source /vendor
```

Executed in the fixed Linux image with Python 3.12.13/Torch 2.13.0+cu130,
CPU only, one thread, 8 CPU/16 GiB limits, no network and read-only source/vendor
mounts, the focused suite passed 77 tests with 5 optional tests skipped. The
command completed successfully and checked zero/one/all 720 one-hot inputs.
An initial software package omitted `examples/README.md` and failed its inventory
test (74 passed, 5 skipped); that packaging attempt is not a passing check.

The actual JSON identifies `purpose=verify_layout_only`, `detectors=720`,
`grid_shape=[10,10,10]`, `x_detectors=400`, `z_detectors=320`,
`bulk_order=X_then_Z_from_measurement_records`, and both `new_samples` and
`new_optimizer_steps` equal zero. It binds source inputs and the complete layout
through SHA-256 digests: DEM
`cd96d502913b7f5000679d89d9bfe6d3ebfdcf12aaec3cf6acdb4d89b01a24d8`,
layout binding
`689366db11bce6c0481f383e6906fe50e5b7dabeddd0980a9132bec62dbdfdf4`.
The command checks construction and encoding; it does
not execute the matched-training study, establish CPU/CUDA prediction parity,
reproduce old primary arrays or measure a logical benefit.

The matched-study runner, external checkpoint/layout binding, complete failure
paths, final producing-head CI, matching-image preflight and actual two-host
parity receipts remain prerequisites to any new formal experiment. All earlier
frozen sources, checkpoints, raw data and inconclusive reports remain unchanged.

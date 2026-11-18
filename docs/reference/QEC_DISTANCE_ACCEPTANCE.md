# Surface and Repetition distance acceptance

The requested 14 Surface distances and 14 Repetition distances passed their
default Z-memory workflow on main `ce8f07b9de04e0c6de3064629c6e04479c0af314`. This is a fixed-budget CPU
workflow check. Each configuration uses three syndrome rounds, 64 original
noisy-circuit shots in four batches of 16, and the bundled Sparse Blossom
decoder. The noise rates are independent X faults 0.005, Z faults 0.007 and
measurement flips 0.009 per declared location, with no Y fault family.

The verifier exports the actual existing memory constructor, derives a strict
DEM without decomposition first, checks the detector/observable shape, then
requests graphlike decomposition for decoding. Disjoint-error approximation is
disabled. Original noisy-circuit detector records are sampled and compared with
decoded logical predictions. Every passing configuration observed nonzero
detector events. A nonzero logical failure count is reported as an observation,
rather than rejected as a software error.

## Per-distance observations

The Surface column lists X/Z logical failures out of 64 shots each. Repetition
uses its existing bit-flip code and Z readout. Timing includes construction,
analysis, sampler preparation, sampling and decoding within one process; it is
an observation, with no speedup claim.

| Family | Distance | Basis | Physical qubits | Detectors | Exact mechanisms | Logical failures / 64 | Seconds | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| surface | 3 | X | 17 | 24 | 51 | 1 | 0.206 | Passed |
| surface | 3 | Z | 17 | 24 | 51 | 0 | 0.228 | Passed |
| surface | 5 | X | 49 | 72 | 165 | 0 | 0.317 | Passed |
| surface | 5 | Z | 49 | 72 | 165 | 0 | 0.285 | Passed |
| surface | 7 | X | 97 | 144 | 343 | 0 | 0.631 | Passed |
| surface | 7 | Z | 97 | 144 | 343 | 0 | 0.494 | Passed |
| surface | 9 | X | 161 | 240 | 585 | 0 | 1.008 | Passed |
| surface | 9 | Z | 161 | 240 | 585 | 0 | 0.899 | Passed |
| surface | 11 | X | 241 | 360 | 891 | 0 | 1.681 | Passed |
| surface | 11 | Z | 241 | 360 | 891 | 0 | 1.396 | Passed |
| surface | 13 | X | 337 | 504 | 1261 | 0 | 2.034 | Passed |
| surface | 13 | Z | 337 | 504 | 1261 | 0 | 1.887 | Passed |
| surface | 15 | X | 449 | 672 | 1695 | 0 | 2.873 | Passed |
| surface | 15 | Z | 449 | 672 | 1695 | 0 | 2.510 | Passed |
| surface | 17 | X | 577 | 864 | 2193 | 0 | 4.096 | Passed |
| surface | 17 | Z | 577 | 864 | 2193 | 0 | 3.458 | Passed |
| surface | 19 | X | 721 | 1080 | 2755 | 0 | 4.822 | Passed |
| surface | 19 | Z | 721 | 1080 | 2755 | 0 | 4.388 | Passed |
| surface | 21 | X | 881 | 1320 | 3381 | 0 | 5.659 | Passed |
| surface | 21 | Z | 881 | 1320 | 3381 | 0 | 4.933 | Passed |
| surface | 23 | X | 1057 | 1584 | 4071 | 0 | 7.418 | Passed |
| surface | 23 | Z | 1057 | 1584 | 4071 | 0 | 6.621 | Passed |
| surface | 25 | X | 1249 | 1872 | 4825 | 0 | 8.707 | Passed |
| surface | 25 | Z | 1249 | 1872 | 4825 | 0 | 7.506 | Passed |
| surface | 27 | X | 1457 | 2184 | 5643 | 0 | 10.877 | Passed |
| surface | 27 | Z | 1457 | 2184 | 5643 | 0 | 9.405 | Passed |
| surface | 30 | X | — | — | — | — | — | Refused at memory construction |
| surface | 30 | Z | 1799 | 2696 | 6987 | 0 | 11.798 | Passed |
| repetition | 3 | Z | 5 | 8 | 15 | 1 | 0.037 | Passed |
| repetition | 5 | Z | 9 | 16 | 27 | 0 | 0.124 | Passed |
| repetition | 7 | Z | 13 | 24 | 39 | 0 | 0.137 | Passed |
| repetition | 9 | Z | 17 | 32 | 51 | 0 | 0.164 | Passed |
| repetition | 11 | Z | 21 | 40 | 63 | 0 | 0.165 | Passed |
| repetition | 13 | Z | 25 | 48 | 75 | 0 | 0.193 | Passed |
| repetition | 15 | Z | 29 | 56 | 87 | 0 | 0.196 | Passed |
| repetition | 17 | Z | 33 | 64 | 99 | 0 | 0.221 | Passed |
| repetition | 19 | Z | 37 | 72 | 111 | 0 | 0.211 | Passed |
| repetition | 21 | Z | 41 | 80 | 123 | 0 | 0.227 | Passed |
| repetition | 23 | Z | 45 | 88 | 135 | 0 | 0.244 | Passed |
| repetition | 25 | Z | 49 | 96 | 147 | 0 | 0.254 | Passed |
| repetition | 27 | Z | 53 | 104 | 159 | 0 | 0.269 | Passed |
| repetition | 29 | Z | 57 | 112 | 171 | 0 | 0.286 | Passed |

## Limits and reproducibility

- The primary distance list passed all 28 default Z-memory configurations.
  Supplementary X-memory checks passed 13 odd Surface distances. Surface d=30
  X-memory was refused: the all-data X product is a stabilizer for this even
  patch. An additional preparation using the transverse logical X column
  passed construction but the sampler rejected its existing X-basis layout.
  Both diagnostics are retained; d=30 X-memory is outside this acceptance.
- Three rounds are fixed for every distance. The campaign does not establish
  performance or capacity for rounds equal to distance, a threshold, logical
  suppression, hardware operation, or all noise models. Sixty-four shots cannot
  certify a small logical error rate, including rows with zero failures.
- The existing rotated-surface layout is the tested code. A differently scheduled
  Surface dataset with the same distance requires its own circuit comparison.
  Even d=30 is accepted here for the existing constructor's Z-memory workflow.
- The memory-to-text exporter requires Stim 1.16.0. Sampling and strict DEM use
  the installed embedded engine, while decoding uses bundled Sparse Blossom.
- Environment: Linux x86_64, Python 3.12.16, Torch 2.13.0+cpu,
  Stim 1.16.0. The isolated runtime installed the actual-main portable wheel
  `99b8bf2d8d1fed33f087cc804abb4391d227c973ae8b1b8690ed90ee1750ef62`.
  The locked image ID is
  `b12e3004927655651e3589bc9c263a7277574d66779d3c7713071959a7fd328e`.
- Each row's seed is 120500 plus its position in the ordered 42-configuration
  manifest. Seeded draws depend on build, SIMD and batch partition.

Run one accepted configuration from the repository:

```bash
python -m examples.qec.code_distance_acceptance surface 30 z
python -m examples.qec.code_distance_acceptance repetition 29 z
```

The JSON includes the full noise declaration, seed, observed detector events,
strict DEM size, logical failures, and dependency versions. The complete records
are retained in [the raw result file](../../artifacts/development/qec_distance_acceptance_20261118.json).

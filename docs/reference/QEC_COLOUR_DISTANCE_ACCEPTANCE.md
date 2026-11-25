# Triangular colour-code distance acceptance

The candidate `dc90ce6622b2e6b014a9ae4bc3bf966df13bf1fa` preserves the existing triangular 6.6.6
constructor and its exact distance semantics for d=3, 5, 7, 9, 11, 13 and 15.
Both X and Z memory readouts passed original noisy-circuit sampling, an
undecomposed strict DEM, and bundled belief propagation with order-zero OSD.

Each configuration uses three rounds, 64 shots in batches of 16, independent
data X faults 0.005, data Z faults 0.007, and measurement flips 0.009.
The seed is `120800 + 2 * distance + (basis == "x")`.

| Distance | Data / ancillas | Detectors | Mechanisms / hyperedges | Z / X logical failures per 64 | Z / X OSD shots | Constructor seconds Z / X |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 7 / 6 | 18 | 47 / 5 | 0 / 1 | 0 / 0 | 0.0007 / 0.0006 |
| 5 | 19 / 18 | 54 | 143 / 35 | 1 / 0 | 1 / 0 | 0.0024 / 0.0024 |
| 7 | 37 / 36 | 108 | 287 / 95 | 0 / 0 | 0 / 2 | 0.0085 / 0.0084 |
| 9 | 61 / 60 | 180 | 479 / 185 | 1 / 2 | 7 / 3 | 0.0291 / 0.0130 |
| 11 | 91 / 90 | 270 | 719 / 305 | 0 / 5 | 4 / 7 | 0.0507 / 0.0507 |
| 13 | 127 / 126 | 378 | 1007 / 455 | 3 / 1 | 8 / 4 | 0.2234 / 0.2259 |
| 15 | 169 / 168 | 504 | 1343 / 635 | 1 / 2 | 11 / 9 | 0.9173 / 0.9154 |

## Why the distance is still exact

The generic matrix constructor still validates dimensions, commutation, rank,
logical independence and logical pairing before searching. For an odd-width
self-orthogonal code with one logical qubit, the stabilizer span is exactly the
even-weight part of the check kernel: every check is even, and that subspace
has the same dimension as the stabilizer span. Odd kernel words are therefore
precisely the logical coset. The trellis finds their minimum weight, not the
weight of a declared representative.

At each wire it retains only the lightest prefix for each unfinished-check
syndrome and weight parity. Prefixes sharing that state admit identical
suffixes; a check must be zero when its last wire is consumed. This establishes
the minimum by state dominance. A bound still excludes heavier words.
The optimization is restricted to at most 16 live checks; other shapes and
wider frontiers retain the existing exact subset search. Small complete logical
cosets, reversed wire order, redundant checks and insufficient bounds are tested.

## Reference and limits

- The embedded strict DEM is compared against flattened Stim 1.16.0 SDK error
  instructions on every signature and probability, with the full SDK shape.
  Disjoint-error approximation and graphlike decomposition are disabled.
- All 896 sampled syndromes are reconstructed from the decoder-selected
  mechanisms. Logical residuals above are retained, not hidden or treated as
  implementation failures. Order-zero OSD is a syndrome-consistency fallback,
  not a maximum-likelihood guarantee.
- The models contain hyperedges. Matching remains outside its graphlike domain;
  this workflow uses the existing hyperedge-aware BP/OSD decoder.
- Memory text export and the independent reference require the Stim SDK.
  Sampling and strict DEM conversion use the existing embedded build.
- This is a fixed-budget CPU workflow check, not a threshold, logical
  suppression, rounds-equal-distance or hardware claim. The image supplied by
  the user does not identify its Color-code lattice; these results cover the
  existing triangular 6.6.6 family only.

## Run one configuration

```bash
python -m examples.qec.colour_distance_acceptance 15 z
```

In an embedded build with Stim 1.16.0, the JSON reports `"status": "passed"`,
`"distance": 15` and `"syndrome_reproduced": 64`.

Raw per-configuration evidence: [JSON](../../artifacts/development/qec_colour_distance_acceptance_20261118.json).

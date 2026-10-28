# State, observable and gradient conformance across representations, on CPU

This extends the representation-agreement grid of #709 and the state/gradient
grid of the #899 first leg to a corpus that varies the parameter draw, covers
both amplitude precisions and both sides of a truncating cap, and adds two
checks that do not run the backward pass. It is a conformance measurement on
CPU, not a capability promotion.

Payloads `complex64.json` and `complex128.json` were produced from source commit
`91c7fdd9b6bee2f04e99d956e98bd766ba4913ab` (base `origin/main`
`3594b02dfec813de2444dec2ced6186381893ef5`), by the producer committed in that
revision. Environment: Python 3.12.14, PyTorch 2.13.0, CPU, one thread per
process, 93.953 s (complex128) and 94.062 s (complex64) for 48 cells each.

Reproduce, from the checkout — the `PYTHONPATH` is not decoration:

```bash
for dtype in complex128 complex64; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH="$PWD" python benchmarks/representation_agreement_sweep.py \
    --gradients --widths 4,8 --depths 2,4 --max-bond 4,16 \
    --structures nearest_neighbour,all_to_all --all-to-all-max-width 8 \
    --seeds 260720,260721,260722 --dtype "$dtype" \
    --output "artifacts/development/representation_conformance_cpu_20261028/$dtype.json"
done
python -m pytest tests/integration/test_representation_agreement_sweep.py -q
```

Running the script puts its own directory first on `sys.path`, so an editable or
installed `flagquantum` wins over the checkout unless the checkout precedes it.
The producer now records `imported_package` and refuses to run when that path is
outside this repository: without it, a run that measured an installed 0.3.1
recorded this repository's revision while every cell reported a refused MPS
derivative, which reads as a property of the representation rather than as the
wrong program.

`statevector` is the exact reference throughout. `relative_l2_error` is
`‖g - g_ref‖ / ‖g_ref‖`, `cosine_similarity` is direction agreement,
`fidelity_vs_statevector` is the normalized overlap `|<ψ_ref|ψ>|²`, and
`norm_drift` is `|‖ψ‖/‖ψ_ref‖ - 1|`. Each payload states the bounds its cells are
read against in `tolerances`; the same test that reads this artifact asserts
them, so the numbers below fail loudly if they change.

## The corpus

48 cells per precision: widths 4 and 8, depths 2 and 4, bond caps 4 and 16,
nearest-neighbour and all-to-all at width 8, three seeded draws
(260720-260722), 39 untruncated and 9 truncated cells, no skipped configuration
and no incomplete cell. Complex128 widens the same float32 angles exactly, so
the two payloads are the same programs at two precisions rather than two draws.

Energy spread, for continuity with the earlier artifacts: `1.10583e-14`
(untruncated, complex128), `4.10343e-06` (untruncated, complex64), `0.28199` and
`0.233785` (truncated). Those are the same grid shape as
`representation_agreement_precision/`, which is the point: nothing here changes
the observable-agreement result.

## Where nothing is discarded, state and gradient agree to machine precision

Across the 39 untruncated cells per precision:

| Quantity | complex128 | complex64 |
| --- | --- | --- |
| max MPS gradient relative L2 error | 3.99256e-15 | 3.60876e-06 |
| max tensor-network gradient relative L2 error | 2.27939e-15 | 1.23751e-06 |
| min gradient cosine similarity | 0.9999999999999998 | 0.9999999999969 |
| max state fidelity deficit | 2.22045e-16 | 2.08515e-06 |
| max own-map finite-difference error | 3.87278e-09 | 2.24891 (**arithmetic**, see below) |

The tensor-network path never truncates and is exact in every cell, including
the truncated MPS cells, where its gradient still agrees to `2.28e-15` at
complex128.

## An oracle that does not run the backward pass

The parameter-shift rule is exact for a parameter entering through one rotation
whose generator has eigenvalues `±1/2`, which is every gate this workload is
built from. It therefore checks the exact path's autograd gradient against
arithmetic rather than against a second backward pass. It is run only up to
width 8, where its `2 × parameters` exact executions stay within budget, and a
wider cell records `skipped` with the width rather than an agreement it did not
measure. In this corpus all 48 cells per precision are inside the bound:
maximum relative L2 disagreement `2.92733e-15` (complex128) and `1.67403e-06`
(complex64).

The oracle differentiates the exact map. It says nothing about a truncated one,
which is why the cells record `applies_to: statevector` and `compares:
["statevector"]` beside the number.

## Where the cap bites, the gradient is measured rather than bounded

9 cells per precision truncate. Two distinct phenomena, kept apart:

| Structure | Seeds | Fidelity | Gradient rel. error | Cosine | Own-map error | Smooth | Largest jump | Schmidt gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nearest_neighbour, w8 d4 cap4 | 3 | 0.990352-0.994230 | 0.6753-0.7389 | +0.6744 to +0.7390 | 0.0748-0.0931 | true | 4.77e-04-1.78e-03 | 2.37e-05-1.42e-04 |
| all_to_all, w8 d2 cap4 | 3 | 0.915980-0.987678 | 1.0073-1.1392 | -0.4776 to -0.0409 | 1.37e+05-6.45e+05 | false | 0.5073-0.7692 | 1.84e-04-9.60e-04 |
| all_to_all, w8 d4 cap4 | 3 | 0.875059-0.969278 | 0.9954-1.0832 | -0.2114 to +0.0954 | 2.22e+04-8.09e+05 | false | 1.155-2.623 | 6.28e-05-1.06e-04 |

**On the chain the map stays smooth and the backward pass is wrong.** The
truncated energy moves by at most `1.8e-03` between scan points `0.01` apart,
the same order as an untruncated cell, so the map is differentiable there, and
autograd reproduces the derivative of its own forward pass only to 7.5-9.3%.
This reproduces across all three seeds and is the defect #941 closed by refusing
the truncated rule by default; this sweep authorizes the rule explicitly,
because measuring the rule's own derivative is the point.

**On all-to-all there is no derivative to reproduce.** The energy jumps by
`0.507` to `2.623` between scan points `0.01` apart: the SVD keeps a different
subspace either side of a point. The finite-difference comparison is meaningless
there, and a negative cosine similarity is a property of the landscape rather
than proof of a backward-pass bug. Operationally it still matters: at fidelity
`0.876` a gradient exists in the result object, reads as finite, and can point
against descent, with nothing in that object saying the surface has no
derivative.

Both are unchanged from the first #899 leg and are reported here across seeds
for the first time. They are not new findings of this corpus.

## The two domain boundaries this corpus records

**Near-degenerate spectra.** A truncated SVD picks a subspace; when the last
kept and first discarded exact Schmidt weights tie, that choice is arbitrary and
no derivative of the truncating map exists. Each cell records the smallest
exact Schmidt-weight gap over the cuts its cap truncates, read from the exact
dense state rather than from the MPS that discarded the weight. The smallest
recorded gap is `1.23553e-08` against a `1e-09` degeneracy threshold, so no
recorded cell is near-degenerate, and none of the jumping cells above is
explained by a tie: the chain's gap (`2.4e-05`-`1.4e-04`) overlaps the
all-to-all cells' range while its map is smooth. Degeneracy is a boundary this
harness can now report; it is not the cause of the measured jumps.

**A finite difference below double precision.** The own-map check differences
two energies with a `1e-5` step. At complex64 that subtracts two float32 values
that agree to about six digits, so the number is the probe's arithmetic: in
cells where nothing was discarded it reaches `2.25` relative error, while the
double-precision run records `3.87278e-09` for the same check. The cell states
which domain it is in (`finite_difference_domain`), and the checked-in-corpus
test asserts the single-precision cells carry the "not a check on autograd"
note. The parameter-shift oracle above is valid at both precisions and is what
the single-precision claims rest on.

No cell in either precision recorded an unsupported derivative
(`gradients.unsupported` is empty everywhere): the representations this corpus
asks for can all produce a state and a first-order derivative, and the harness
records a refusal per mode rather than substituting another representation.

## Discarded weight

Each cell records the observed bond per cut (`mps_bond_dims`), the state's
cumulative per-step discarded weight (`mps_discarded_weight`), its largest
single-step record (`mps_max_truncation_error`) and the number of truncating
steps. The cumulative figure is a sum over the splits a run performed, not the
norm lost: a narrower cap keeps narrower intermediate states and can discard
less in total, so it is evidence that the cap acted and by how much at its worst
step. Fidelity is read beside it as the loss. In the truncated cells the
cumulative record spans `0.1224`-`9.7712` (complex128) and
`0.1224`-`11.531` (complex64).

## What this does not establish

- **Not scalability or accelerator evidence.** Single process, CPU, one host;
  every payload reports `distribution_semantics="single_device_fast_path"`,
  `scalability_claim_allowed=false` and its blocker. No GPU or accelerator leg
  was run, and nothing here promotes one.
- **No sharded/unsharded comparison.** Placement is measured by the separate
  conformance leg (`benchmarks/mps_placement_conformance.py`,
  `artifacts/development/mps_placement_conformance_cpu_20261028/`), which holds
  circuit, angles and truncation policy fixed across worlds 2, 3 and 4. This
  artifact isolates truncation from precision, not from placement.
- Widths 4 and 8 only. All-to-all beyond width 8 remains excluded because the
  tensor-network contraction repeats per Pauli term, and the wider chain cells
  are outside this corpus's budget rather than beyond its representation.
- The oracle bound is a width bound. Above width 8 the cells record a skipped
  oracle, not an inherited agreement.
- First-order PyTorch autograd only: no adjoint, second-order, checkpoint,
  optimizer-step or multi-step training semantics. Those are #902.
- The smoothness probe scans one parameter index over `[-0.05, 0.05]` in 7
  points. It is a local sample, not a proof of differentiability, and a cell it
  calls smooth could still jump outside the scanned window.
- These are conformance measurements, not a performance or capability
  promotion; the capability entries and their measured bounds are updated from
  this payload, and nothing else.

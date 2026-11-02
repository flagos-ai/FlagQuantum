# Corrected state, observable and gradient conformance for #899

This corpus supersedes the scientific interpretation of the October 27/28
gradient artifacts. Their raw JSON remains unchanged. All measurements here
use the same grad-enabled forward for state, observables, autograd, central
differences and local scans. Actual intermediate SVD spectra are observed;
the final ideal-state spectrum is retained as a different diagnostic.

Producer: `71003d31521f6106feaacfe83fd46ef97fac9d44`, based on main
`ef95c04f08ddeb05c1c8469b51f4cfccb7c4fc62`. The producer was committed when
both runs executed; each JSON records the commit, command and imported
package identity. Environment: Python 3.12.14, PyTorch 2.14.1, macOS arm64, CPU,
one thread per process. Elapsed times are 162.485 s for complex128 and 128.987 s
for complex64, measured concurrently and not intended as performance evidence.

## Reproduce

Run from a checkout with project dependencies installed, using
`torch==2.14.1` for the recorded arithmetic. To replay the historical numerical
code exactly, first check out the full producer commit above. Run the evidence
validation tests from the revision containing this README and its artifacts.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD"
for dtype in complex128 complex64; do
  python benchmarks/representation_agreement_sweep.py \
    --gradients --widths 4,8 --depths 2,4 --max-bond 4,16 \
    --structures nearest_neighbour,all_to_all --all-to-all-max-width 8 \
    --seeds 260720,260721,260722 --dtype "$dtype" \
    --output "artifacts/development/representation_conformance_cpu_20261102/$dtype.json"
done
python -m pytest tests/integration/test_representation_agreement_sweep.py \
  tests/integration/test_representation_conformance_evidence.py \
  tests/unit/test_mps_truncated_gradient_policy.py \
  tests/unit/test_conformance_launcher.py -q
```

Each precision has 48 complete cells: widths 4/8, depths 2/4, both structures,
caps 4/16 and three seeds. There are 39 untruncated and nine truncated cells,
with no skipped configuration. Angles are drawn as `0.05 * float32 randn` and
widened exactly for complex128. Each case carries its own resolved tolerances,
seed, observed per-cut bonds and cumulative/maximum discarded-weight records.
The cumulative record is a sum of split records, not a final-state norm loss.

## Where the state is represented without truncation

| Maximum measured quantity | complex128 | complex64 |
| --- | --- | --- |
| MPS gradient relative L2 error | 3.99256e-15 | 3.60876e-06 |
| TN gradient relative L2 error | 2.27939e-15 | 1.23751e-06 |
| Normalized state fidelity deficit, MPS/TN | 3.33067e-16 | 2.25264e-13 |
| Norm drift, MPS/TN | 4.44089e-16 | 1.72681e-06 |
| Absolute error over Z0, X1 and Z0 Z1 | 6.66134e-16 | 3.33786e-06 |
| Exact-map parameter-shift relative error | 2.92733e-15 | 1.67403e-06 |

Minimum gradient cosine over these cells is 0.9999999999999997 at complex128
and 0.999999999996949 at complex64. The extra-observable absolute test bounds are 1e-10 (complex128) and 1e-5
(complex64). The parameter-shift oracle checks only the
exact statevector map, for independent one-gate parameters with rotation
generators having eigenvalues +/-1/2, inside width eight. It makes no claim
about the derivative of a truncated computation.

## The truncated forward and the derivative probe are now the same program

The earlier producer ran finite differences and scans with `no_grad`, while
autograd ran with trainable inputs. The factorization choice changes between
those cases. For seed 260720, nearest-neighbour width 8, depth 4, cap 4:

- The trainable energy is -5.575720620713117; the older no-grad energy was
  -5.569001087362569. They cannot serve as one another's own-map oracle.
- At parameter index 2, the corrected central difference is approximately
  -0.16308292; the old no-grad probe gave approximately -0.07786446.
- The explicit projected-subspace gradient remains approximate. The three
  chain cells have step-consistent own-map error 0.0887126-0.136455 relative to
  the full reported autograd gradient norm, rather than the earlier 0.075-0.093.

The complex128 probe evaluates eight selected components at h=1e-5 and h/2.
Only if the two differences agree within relative 1e-3 or absolute 1e-6, and
the base/perturbed actual SVDs carry no near-degenerate boundary, does it report
`status=step_consistent_probe` and a derivative-error aggregate. Even then it is
a local numerical probe, not proof of differentiability. All 39 untruncated
cells pass this probe, with maximum own-map relative error 2.52088e-10.

For the six truncated all-to-all cells, the probe records `domain_boundary`,
retains its per-component numerical samples, and sets the derivative-error
aggregate to null. For all 48 complex64 cells it records
`unsupported_precision`, performs no central-difference comparison and reports
no derivative-error aggregate. Exact-map parameter shift remains valid at both
precisions. Boundaries are not failed or silently substituted representations.

## Actual split boundaries, not an inference from the final ideal state

`actual_svd.records` observes the actual returned spectra of both individual
and bucketed factorization calls without modifying factors. Each record names
the call/spectrum index, retained/original rank, normalized discarded weight and
normalized kept/discarded boundary gap:

```text
gap = (s[rank-1]^2 - s[rank]^2) / sum(s^2)
near_degenerate = gap < 1e-9
```

Six cells per precision, all truncated all-to-all cases, have a near-degenerate
actual split. Some ties occur in negligible null-space weight; the record keeps
the corresponding discarded weight, so a tie alone is not promoted into proof
of physical instability. The final ideal-state Schmidt gaps cannot exclude
these intermediate boundaries. A synthetic tied matrix with 50% discarded
weight exercises both scalar and bucketed split observation in the tests.

The seven-point scan reports its actual spacing, 0.0166667 over [-0.05, 0.05],
and uses the same training forward. Its `smooth` flag is only a sampled jump
threshold of 1e-2. Chain scan jumps range 0.000900607-0.0135025 at complex128,
so even a chain can cross that threshold. All-to-all scan jumps range
0.534607-1.93250. Neither a large nor a small finite sampled change proves
discontinuity or differentiability. The earlier unconditional interpretation
is withdrawn; these data do not identify a unique cause of instability.

The truncated MPS fidelity spans 0.650270-0.997514 at complex128 and
0.868726-0.997514 at complex64. Its gradient error and direction are measured
relative to the ideal circuit, separately from the own-map diagnostic. No
ideal-gradient accuracy bound is claimed for truncated cells. The TN path is
untruncated; a truncated MPS label does not describe the TN computation.

## Acceptance scope

| #899 criterion | Evidence and boundary |
| --- | --- |
| Multiple seeds, structures, precisions and truncation regimes | Both 48-cell payloads, initialization, per-case bonds/weights/tolerances |
| States, multiple observables and gradient error/direction | Normalized fidelity, independent norm drift, three Pauli products, relative L2 and cosine; exact-map parameter shift |
| Matched sharded/unsharded MPS | Separate `../mps_placement_conformance_cpu_20261102/README.md`, worlds 2/3/4, fixed parameters and caps |
| Near-degenerate and unsupported derivative domains | Actual split records, step-consistency/precision status, refusal-without-substitution regressions |
| Optional accelerator leg and claim discipline | No accelerator leg; every payload refuses scalability promotion |
| Reproducible evidence, regression checks and capability limitations | Committed producer, full commands and tolerances; focused tests and updated capability source of truth |

These measurements cover first-order PyTorch simulation on small CPU circuits.
They do not repair the projected-subspace gradient, establish arbitrary-circuit
accuracy, certify GPU/multi-node execution, or cover optimizer/checkpoint and
multi-step training semantics. The default refusal and explicit approximation
policy remain in place. This is conformance reporting, not a capability promotion.

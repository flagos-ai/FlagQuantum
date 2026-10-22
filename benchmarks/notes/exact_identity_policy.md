# Exact-zero identity policy: measurement update for #704

The optimizer now preserves submitted nonzero small angles and matrix components.
A near-zero cutoff is not an exact identity test. This deliberately trades some
historical gate-count reductions for numerical meaning; target-basis synthesis's
existing residual tolerances remain a separate policy.

The historical W9/W18 counts remain in Git history and stored artifacts. They are
not silently regenerated or relabelled. The live benchmark contracts have been
re-measured with the original seeded populations under the new policy. A population
named `zero_angle_rotations` includes nonzero `1e-13` angles, so its old count is not
an admissible performance requirement for an exact-zero optimizer.

## Current measurements

Python 3.12.14, PyTorch 2.13.0, Qiskit 2.5.2, CPU, complex128, original seeds:

| Population | Historical predicate in current pipeline | Exact predicate in current pipeline | Signed reduction |
|---|---:|---:|---:|
| Parameter-free gates | 167 | 167 | 0 |
| Zero-angle-labelled rotations | 162 | 282 | -120 |
| Full-turn rotations | 139 | 150 | -11 |
| Zero-polar U3 | 176 | 176 | 0 |
| Two-qubit rotations | 200 | 200 | 0 |
| Mixed | 405 | 421 | -16 |
| Mixed with intermediate measurements | 505 | 532 | -27 |

The net signed reduction is **-174**, not a claimed speedup. The counterfactual
replaces only the identity predicate, with its former `1e-12` criterion fixed
explicitly; the rest of the pipeline stays current. Its counts are not a replay
of a complete historical compiler. The same inputs are also measured with the
fold and rotation merge independently paused, so attribution remains checked.

The current one-qubit no-decline control has 386 intermediate gates and 909 legal
gates on the measured basis population; its existing post-lowering platform band
and direction checks remain. The strict and shipped decline controls remain
771 and 763 respectively. No archived performance evidence is updated.

## Measurement correctness

The old instrument erased intermediate measurements before comparing coherent
statevectors. That is not the same experiment: H, computational-basis measurement,
H has final probabilities `(1/2, 1/2)`, while erasing the measurement gives `(1, 0)`.
A diagonal gate removable before measurement can therefore look incorrect under
that erased-measurement experiment.

The bounded three-qubit benchmark now retains all unnormalized measurement
branches and sums their squared magnitudes. There is no feedback in these
populations and terminal measurements overwrite all classical output bits. Each
optimized program is checked against its own unoptimized source; neither a vector
average nor agreement between two potentially wrong optimized programs is used.
The original `1e-12` probability-error bound remains. The largest measured source
comparison is below `5e-14`. Separate #704 regressions use zero absolute tolerance
on tiny amplitudes, which this coarse benchmark bound cannot certify.

## Re-run

Use a repository-certified Qiskit lane from `dependency-policy.toml`, then run:

```bash
python -m pytest tests/benchmark_contract/test_compiler_*.py -m benchmark_contract -q
```

The contracts check the exact-zero boundary, signed cost, independent source
probabilities, instrument distinction, unchanged control populations, and pinned
current attribution tables. They do not claim hardware timing, GPU performance,
or a general interval error certificate.

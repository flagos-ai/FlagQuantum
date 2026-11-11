# Sampled QSP polynomial checks

Simulation owns `numerics/qsp_sampling.py` and `numerics/qsp_grid.py`, adapted
from unchanged Qualtran v0.7.0 `linalg/polynomial/qsp_testing.py` at
`bcb17bb25360ed68ce3bc084645db3e929799389`. Copyright is retained. These modules
extend the existing numerical inspection path, not Compiler synthesis or a new
execution engine. No protected API, public export, or dependency manifest changes.

## Short local path

Numerical grid checks require an explicit FFT provider. The example uses the
reference environment's SciPy, which production modules do not import or install.

```python
import numpy as np
from scipy.fft import fft
from flagquantum.simulation.numerics.qsp_sampling import (
    random_qsp_polynomial,
    check_gqsp_polynomial_pair_on_random_points_on_unit_circle,
)
from flagquantum.simulation.numerics.qsp_grid import (
    polynomial_max_abs_value_on_unit_circle,
    scale_down_to_qsp_polynomial,
    assert_is_qsp_polynomial,
)

rng = np.random.RandomState(17)
coefficients = random_qsp_polynomial(4, random_state=rng)
assert len(coefficients) == 4  # Source counts coefficients, not degree+1.
assert sum(abs(c) for c in coefficients) <= 1
check_gqsp_polynomial_pair_on_random_points_on_unit_circle(
    (0.5, 0.5), (0.5, -0.5), random_state=rng, n_points=32,
)
scaled = scale_down_to_qsp_polynomial((1.0, -2.0), fft=fft, n_points=8)
np.testing.assert_array_equal(scaled, [1 / 3, -2 / 3])
assert polynomial_max_abs_value_on_unit_circle(scaled, fft=fft, n_points=8) <= 1
assert_is_qsp_polynomial(scaled, fft=fft, n_points=8)
```

Normal insertion remains `circuit.append(op)`. The replacement regression
retains an existing H/Rz operation, evaluates the scaled polynomial on a dense
unitary through the existing evaluator, and checks complete complex amplitudes
and the idle clean wire. This postprocessing is not a synthesized quantum
polynomial operation and need not preserve state normalization.

## Source semantics retained

- Random generation draws `degree` uniform magnitudes divided by `degree`, then
  optionally `degree` random phases. It returns exactly `degree` coefficients,
  so the maximum polynomial degree is `degree-1`. Real-only and complex modes,
  zero-length output, negative-size refusal and subsequent RNG state match.
- Pair checks construct both NumPy polynomials before one random angle draw.
  They retain the source `np.testing.assert_allclose` defaults and caller rtol.
  Invalid empty coefficient construction fails before consuming random numbers;
  a failed sampled norm assertion leaves the angles consumed. Zero sample
  count vacuously accepts even the invalid constant pair (2, 0).
- The source annotation admits a NumPy `Polynomial` object, but reconstructing
  it with `Polynomial(P)` produces a nested object and the source fails during
  absolute-value evaluation after drawing angles. This direct source
  counterexample is recorded separately; native typed pair inputs are finite
  coefficient sequences, not Polynomial objects or symbolic `Shaped` values.
- Grid maximum keeps source `asarray`, dtype-preserving zero padding, one FFT
  call and max-absolute reduction. The native float return retains the numeric
  value of the source NumPy scalar. Oversized coefficients and invalid grids
  retain assignment/FFT/reduction failures; they are not silently truncated.
- Scaling keeps source `asarray` then `list` conversion before the shared grid
  helper. Only a sampled maximum strictly greater than one causes division.
  `np.divide` is the same ufunc as source `/`. An unscaled input ndarray is
  returned unchanged by identity; scaling allocates and does not mutate it.
  Actual float/complex/integer dtypes are checked, not forced to complex128.
- NaN sampled maxima retain source comparison behavior: scaling leaves them
  unchanged, while the <=1 assertion fails. Provider errors propagate.

The native grid boundary is typed for float64, complex128 and int64 coefficient
arrays or finite complex coefficient sequences. Its FFT provider accepts the
padded array and returns complex128 data. Source comparisons use `scipy.fft.fft`;
a finite cached transform replaces it without changing consumers. There is no
implicit SciPy dependency, FFT implementation, or claim of upstream signature
compatibility. Other dtypes and symbolic/foreign objects remain unaccepted.

## Finite-grid counterexample

With four grid points, `P(z) = 0.53 + 0.53*exp(i*pi/4)*z` has sampled maximum
below one. Both original and native assertions pass, and scaling leaves P
unchanged. But at `z = exp(-i*pi/4)`, its magnitude is **1.06**. The original
TODO about always-correct circle bounds remains unresolved. The native functions
retain the algorithm rather than modifying the reference or claiming a global
bound. Random pair checks likewise prove consistency only at sampled points.

## Evidence and costs

Tests compare source coefficients, subsequent RNG draws, padding, FFT calls,
actual output dtypes, scaling/identity, assertion messages, invalid construction,
source object failure and the missed-peak counterexample. Full complex matrix
consumers and clean wires are checked without a phase-insensitive metric.
Selected unchanged upstream FFT-QSP tests exercise their original sampling and
pair-check consumers; this does not claim their circuit or complementary
polynomial algorithms are newly ported in this batch.

| Quantity | Numerical inspection helpers |
| --- | --- |
| Added physical gates / work wires / measurements | 0 / 0 / 0 |
| Source QEC physical gate/resource model | None |
| Random generation | degree coefficients, one or two uniform-array calls |
| Pair check | n_points angles and pointwise polynomial evaluation |
| Grid helper | One caller-owned FFT over n_points padded entries |
| Numeric storage | O(degree + n_points), multiple arrays; peak not measured |

The append consumer's two existing H/Rz gates are unchanged. Coefficient count,
RNG calls, FFT size and sampled maxima are separate from quantum primitive gate
counts. No rigorous whole-circle guarantee, synthesis, hardware performance,
PyTorch gradient, scalability, or full-parent/G2 acceptance is claimed.

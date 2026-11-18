# Complete FFT complementary polynomial source

Frozen Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389` supplies Q-0651
`qsp/fft_qsp.py`, blob `855d7a61b37516315fa4873c682754a4dcc11ef5`, and
Q-0652 `qsp/fft_qsp_test.py`, blob
`c2dd5eb4d8e774ed1280633b70d5934bbb12d02c`. Google LLC 2023 Apache-2.0
attribution is preserved. The owner is
`simulation/numerics/qsp_fft_complement.py`. Its consumer reuses Algorithms'
existing GQSP binding and Compiler signal sequence.

## Complete source and dependency mapping

| Original local function | Native responsibility |
|---|---|
| `_scale` | Multiply coefficients by exactly `1-tolerance/4`. |
| `_pad_poly` | Pad by exactly `num_modes-1`, retaining dtype conversion and source failures. |
| `_p_eval` | IFFT with `norm="forward"`. |
| `_get_log` | Compute log of one minus the squared absolute evaluation. |
| `_fourier_multiplier` | Forward-normalized FFT, halve mode0, zero from `num_modes//2+1`. |
| `_get_modes` | Forward-normalized inverse of the multiplier output. |
| `calculate_coeff` | Exponentiate, forward-normalized FFT, truncate to original coefficient length. |
| `fft_complementary_polynomial` | Preserve defaults, local-function staging and all numerical operations above. |

Coefficients are in ascending degree order. Inputs remain caller-owned. There are
zero quantum output registers, scratch wires or emitted operations in this
numerical function. Temporary arrays scale linearly with the actual transform
length and FFT arithmetic scales as O(n log n). The actual transform length is
`len(coefficients)+num_modes-1`. The source multiplier uses `num_modes` for its
cutoff. Both details are preserved and intermediate transform arrays are checked
against the unchanged implementation, including odd/even and very small mode
counts. NumPy owns transforms. Optional SDK objects stop at the reference test
boundary. This numerical work is `single_device_fast_path`.

The tolerance argument scales input coefficients. It supplies an approximation
margin and carries no rigorous whole-circle accuracy certificate. Invalid
unbounded input preserves the source invalid-log warning and NaN output. Empty
and nonpositive-mode failures retain their original exceptions. These are
explicit source boundaries.

The runnable composition is

```python
q = fft_complementary_polynomial(p, tolerance=1e-9, num_modes=500)
op = GeneralizedQSP((("q", 1),), unitary_emitter, p, tuple(q))
circuit.append(op.make_on(signal=0, target=(1,)))
```

## Verification and resource separation

All three original test groups cover real/complex complements up to10,000
coefficients and random one/two/three-qubit unitaries. The unchanged source suite
passes31 cases. A second run replaces only the coefficient solver with the native
implementation and retains the original test assertions.

Native small fixtures compare every intermediate FFT input and output coefficient
exactly. GQSP complete operators, a coherent complex state and absolute phase
match the unchanged source at negative powers0/1/5. One idle qubit remains exactly
zero. These fixtures borrow zero scratch wires and have one unitary branch with
zero measurements. Applicable measured-unitary branches reuse the accepted GQSP
provider evidence.

The actual primitive inventories and the independent declared child model equal
the three-row table in
[the optimization complement audit](QSP_OPTIMIZATION_FILE_AUDIT.md).
Both numerical solvers produce coefficients for the same circuit owner. The
fixture emits phases and control-polarity X gates explicitly. The child model
independently declares11/11/14 continuous rotations and peak2 logical qubits for
powers0/1/5. Numerical solver arithmetic has zero quantum gate cost. Concentrated
regression and strict-check results are appended after batch execution.

The source31-case run retains five imported upstream slow-marker warnings. The
native replacement run passes the same31 cases with zero warnings. Concentrated
regression passes390 cases without skips or warnings, followed by40 focused
passes after annotation-only fixes. Strict mypy checks both numerical owners and
both test modules. Black/Ruff, Simulation scope, architecture, multi-level IR and
protected API checks pass. Q-0651/Q-0652 close their complete source/test scope.

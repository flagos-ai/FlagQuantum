# Complete chemistry tutorial helpers

## Source and ownership

Q-0337 covers the complete four-function `chem_tutorials.py` from Qualtran
v0.7.0, revision `bcb17bb25360ed68ce3bc084645db3e929799389`, original blob
`e2ae163c0e2d33e072423bf4aa3e388fc1bfe89c`. Copyright Google LLC,
Apache-2.0. The native scaling example retains the source notice.

| Complete source function | Native owner and interface |
| --- | --- |
| linear | `examples/algorithms/chemistry_scaling.py` linear_values with float-array input, slope and intercept |
| fit_linear | Same owner, fit_line returning the two fitted Python floats |
| gen_random_chem_ham | Existing `chemistry_authoring.py` authoring_integrals with spatial-orbital count and explicit seed |
| plot_linear_log_log | Scaling example plot_power_law accepting existing Matplotlib axes and returning None |

These are classical example helpers. Root framework exports and algorithm
execution retain their existing owners. Reusing authoring_integrals avoids a
second random-Hamiltonian implementation. Convert source spin orbitals to
spatial orbitals with the same integer floor division before calling it.
The existing example uses explicit RandomState seeds, while the source consumes
global NumPy random state. Tests supply identical random streams to compare
every resulting tensor entry, including zero and odd orbital counts.

## Actual numerical and plotting consumers

The line evaluator preserves the original arithmetic expression. The fitter
uses the same SciPy curve_fit with the same default arguments. Source returns
a NumPy vector despite its tuple annotation. The native example returns a
pair of Python floats as its explicit interface adaptation.

The plotting helper fits log(x), log(y), generates 200 evenly spaced x values
between the first and last samples and evaluates exp(intercept) times x raised
to the fitted slope. It preserves both line styles, supplied color, sample
markers, generated legend text and log axes. The original prose describes an
additive constant, but the actual source implements a pure power law. Native
behavior follows that unchanged executed expression.

Tests compare actual source and native line coordinates, fitted parameters,
all plotted sample and curve coordinates, labels, styles and axes scales.
Both real figures are saved with the Agg renderer and compared pixel for pixel.
Empty, mismatched and non-finite fit inputs preserve the actual SciPy error
messages. All 18 seeded orbital fixtures match entry for entry.
The two original numeric tests execute unchanged after their source hash check.

## Acceptance boundary

The complete helper source can close independently of the original tutorial
notebook defects. Q-0338 includes two notebook executions and remains open for
that separate original-notebook execution scope. Existing authoring and resource tutorial audits retain
the actual source-notebook counterexamples and native independent execution.

This helper file emits no quantum primitives, has no measurement branches or
quantum work registers, and introduces no resource formulas. Existing tutorial
model estimates remain separate from emitted-circuit counts in their owners.
The concentrated batch reuses those numerical, stored graph and append
consumers. No full scientific environment or standalone chemistry calculation
is implied by these helper-level results.

The batch passes 45 concentrated and 29 local integration cases with zero
skips. Strict typing for both new Python files, Black, Ruff, Algorithms scope,
architecture, internal IR, protected API baseline and whitespace checks pass.

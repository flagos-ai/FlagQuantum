# Workflow acceptance

Run the module in `README.md` or execute every code cell in `walkthrough.ipynb`.
The source grid is 8 by 8, the estimator has six bits, and the state has 13 bits.
Recompute all 64 amplitudes and both eight-term Walsh expansions. Spatial
transforms, powered evolution, inversion, original postselection and classical
solve remain explicit. Compare all complex amplitudes with the independent
Circuit, then compare the named numeric marginal and compiler workspace.

The original integer-word inverse table and source normalization are preserved.
The measured source-scale classical relative error is about 78%, and the relative
residual is about 7.6. This is a reproduced limitation, not accurate PDE-solver
acceptance. The source axis assignment and finite precision remain unchanged;
no fitted scale or transpose is applied to the quantum result. The retained
estimator leakage is reported separately from clean compiler workspace.

Native QPE uses fractional numeric output and lives at the program root. The
source integer-word table is explicit; zero postselection is unchanged. Static
spatial unitary views use the existing supported interface. No generic callable,
Output-helper, numeric-array or integer-QPE API is introduced.

Both final contour figures, all source stage pins and full local acceptance are
required before the complete-workflow inventory is credited. The original
quantum-program counterpart stays frozen.

# Two-asset Rainbow option with native amplitude estimation

Recreate the original Gaussian bins, correlated log returns, rounded affine
maximum, and square-root payoff loading. Preserve two bits per asset, the
three-bit selector, both source width budgets, and the original precision and
confidence inputs. The five-qubit implementation reuses the reversible native
payoff loader and reflection.

Run from the repository root with
`python -m examples.synthesis.applications.finance.rainbow_options.rainbow_options_bruteforce_method.overview.main`.
The [Notebook](walkthrough.ipynb) displays the model, observed counts, and price
interval. The local count-only estimator uses bounded Hoeffding intervals and
a reproducible seed. Each needed Grover power is specialized locally, and its
complete complex state is checked against an independent Circuit. This is a
local estimator with the source precision and confidence inputs. It does not
reproduce the hosted IQAE sampling trajectory.

[Source provenance](provenance.json).

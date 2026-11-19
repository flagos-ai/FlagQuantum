# Rainbow option with exponential integration

Regenerate the original two-asset input, rounded affine selector and all 32
exponential reference weights. The native program retains the two asset
registers, five-bit reference and indicator, and evaluates the inclusive
comparison coherently. Only computed workspace returns to zero.

Run from the repository root with
`python -m examples.synthesis.applications.finance.rainbow_options.rainbow_options_integration_method.overview.main`.
The [Notebook](walkthrough.ipynb) uses the original width limit of 25 and the
original estimation precision and confidence inputs. A bounded local count-only
estimator uses actual sampled indicators. Each native power checks the full
logical state and clean workspace against an independent Circuit. The original
affine rescaling is applied to both interval endpoints.

[Source provenance](provenance.json).

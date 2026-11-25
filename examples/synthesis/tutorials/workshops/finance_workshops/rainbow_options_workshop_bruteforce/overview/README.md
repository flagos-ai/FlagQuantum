# Rainbow option estimation workshop

Follow the original two-asset workshop using a freshly regenerated model. The
native payoff preparation preserves the rounded affine maximum, the
three-bit selector, the reversible indicator and the original width bounds.

Run from the repository root with
`python -m examples.synthesis.tutorials.workshops.finance_workshops.rainbow_options_workshop_bruteforce.overview.main`.
The [Notebook](walkthrough.ipynb) walks through all stages, including the
original 20,000-shot experiment. The local estimate stops at the initial
Grover power when its fixed-batch Hoeffding interval meets the requested
precision. It reports the interval and the original workshop's historical
reference check alongside the independently enumerated finite payoff.

[Source provenance](provenance.json).

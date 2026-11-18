# iQuHack Gaussian challenge starter

Open [walkthrough.ipynb](walkthrough.ipynb) or execute both local stages.

```bash
python -m examples.synthesis.community.hackathons.iquhack_2025.challenge_template.overview.main
```

The source starter applies Hadamards across the signed output. This
walkthrough runs that starter at the original eight-bit stage-one width and
compiles the original four-bit stage two. It checks all 256 complex
amplitudes of stage one and reports the actual Gaussian mean squared error
on the signed grid. One seeded draw provides the source’s single-shot
example.

The optional native reference prepares the exact Gaussian on those same
two finite grids. Its basis-converted CX and depth counts are reported
alongside the starter. The reference is intended for checking the exercise
on these bounded widths. The team export cell stays disabled as in the
source. The export helper writes native IR when a team name is supplied.

# Native workflow acceptance

Retain both oracle-mapping stages and both Simon algorithms: five bits with secret 6, six bits with partition index 4 and secret 60. Retain width/depth objectives and 250/300 algorithm shots. Mapping stages bind 1000 local shots because the source leaves the cloud default unspecified; every local sampler uses seed 17. Exact GF(2) elimination replaces the galois dependency and consumes named decoded integers, with no physical wires or bitstring parsing. Both mapping scatter plots are retained. Resource summaries replace the cloud circuit viewer; no hardware or Classiq synthesis-resource equivalence is claimed.

The source Notebook and every code cell are hash-bound in provenance. User
examples use native Python and public FlagQuantum APIs. No source-language
loader, cloud credentials or production dependency changes are required.

Independent Circuit references compare full complex amplitudes without aligning
away global phase. Simon checks both mappings and both interference stages,
register order, zero workspace, recovered secrets 6 and 60, all 120 nonzero
secrets for widths 1 through 6, and insufficient-rank or invalid-sample rejection.
The independent oracle uses small per-output-bit truth-table matrices rather
than a dense matrix over both complete registers.

Graph coloring checks all 1024 assignments against the independent predicate,
all 16 marked assignments, one full Grover iteration, native and reference
workspace cleanup, the width budget, sampled counts and unfiltered ranking.
The original graph has 13 distinct undirected edges. Plot checks retain the
original graph plus 16 sampled panels, including validity annotations. Real
Jupyter execution and visual inspection complement numeric acceptance. Plot
tests skip if optional Matplotlib is unavailable; numeric tests still run.

Notebook acceptance is separate from the frozen 216-program quantum corpus.
There is no hardware, scalable-simulation or resource-optimality claim.

```sh
python -m pytest tests/integration/test_synthesis_simon_coloring_notebooks.py -q
```

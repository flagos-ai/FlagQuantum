# Quantum sine and cosine transforms

Run the complete four-stage workflow from the repository root:

```sh
python -m examples.synthesis.functions.qct_qst.overview.main
python -m examples.synthesis.functions.qct_qst.overview.comparisons.circuit
```

[walkthrough.ipynb](walkthrough.ipynb) shows native qfunc definitions, amplitude
inputs, synthesis, execution and measured/expected arrays for all four transforms.
It retains the source seed 123 and sequential draws of 9, 7 and 8 real amplitudes.
Classical real-array parameters are bound when synthesizing; this is not runtime
parameter binding or a differentiable training example.

Existing native preparation and transform primitives do the quantum work.
Type I places DCT I(9) and i times DST I(7) in separate blocks. Combined type II
uses DCT II(8) for selector 0 and negative DST II(8) for selector 1. Standalone
sine uses positive DST II. These signs are observable and are checked without
aligning away global or block phases. Combined register order is (x, q).

The source's executable classical references resolve its explanatory prose's
Type I sine-index and type II dimension typos. The source compares phase-aligned
real arrays at 0.01 tolerance; native acceptance additionally compares complete
complex amplitudes at 1e-12. There are no plots or hardware jobs in this workflow.

[Workflow acceptance](WORKFLOW.md). [Source provenance](provenance.json).
[Third-party license](../../../CLASSIQ_LICENSE.txt).

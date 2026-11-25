# Cooling-system optimization

Run the native thermal solve and mock-QAE experiment from the repository root.

```sh
python -m examples.synthesis.applications.automotive.cooling_systems_optimization.overview.main
```

[The walkthrough](walkthrough.ipynb) keeps the four-node heat model, the
six candidate conductance links, the original block-encoding normalization
and the degree-11 QSVT inverse. It regenerates all 64 thermal matrices and
classical solutions and checks all coherent design branches. The projected
temperature offsets are checked against an independent diagonalization
before comparison with the classical solution.

The source defines a five-phase-bit thermal optimization circuit and skips
its large execution. The native Circuit builder is available as
`full_model.build_full_quso`. This builder retains the 20-bit design,
phase-estimation and solver layout. The local run follows the source and
executes the separate one-phase-bit example using the original
[mock-QAE cost input](data/README.md). Its QAE amplitudes are rebuilt from
all 64 costs and checked against an independent controlled Grover step.
Two nonzero angle bindings reuse one symbolic eight-bit circuit. The
complete complex states are verified before taking the design marginal.

Resource rows count simulator instructions, including the packed local
conditional and projector operators.

[Source mapping](provenance.json) · [Third-party license](../../../../CLASSIQ_LICENSE.txt)

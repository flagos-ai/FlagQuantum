# Native high-level algorithm-design tutorial part I

Open [walkthrough.ipynb](walkthrough.ipynb) from the repository root, or run:

```bash
python -m examples.synthesis.tutorials.workshops.algo_design_qce_tutorial.algo_design_qce_tutorial_part_i.overview.main
```

The complete native tutorial retains all eighteen original code stages. Exercise
placeholders become labeled reference solutions; the final solution section
rebuilds and executes each solution again. Bell/GHZ preparation, inferred integer
and fractional arithmetic, signed conditions, Grover marking, product phase and
Fourier square arithmetic share the same qfunc/synthesize/run flow.

Original widths remain unchanged. Native named a/b/c ports replace the source's
quantum struct fields. Numeric output metadata is read after synthesis using public
program reports, rather than introspection or printing while tracing. Source
implicit sampling defaults are made explicit as1000 local draws per batch. The
complete Notebook executes fifteen actual observation batches and two complete
sixteen-row complex-amplitude tables. Bitstrings need no user decoding.

All seventeen executed programs are compared against independent absolute Circuit
states, including global phase and zero additional compiler-workspace probability.
The Fourier reference is a modular permutation, independent of QFT/phase lowering.
The Grover reference preserves the source reflection's sign and two rounds. The
frozen sibling quantum example and all216 counterparts remain unchanged.

The native tutorial replaces the hosted program viewer with actual synthesis
resource reports. Resource counts precede Runtime compilation; they are not counts
in a hardware basis. No remote execution or source importer is used. There is no
hardware, optimality, scalability, speedup or backend-equivalence claim. See
[WORKFLOW.md](WORKFLOW.md) and [provenance.json](provenance.json).

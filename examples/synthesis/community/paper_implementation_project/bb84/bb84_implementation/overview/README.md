# Native BB84 teaching Notebook

Run from the repository root:

```bash
python -m examples.synthesis.community.paper_implementation_project.bb84.bb84_implementation.overview.main
python -m examples.synthesis.community.paper_implementation_project.bb84.bb84_implementation.comparisons.circuit
```

Open `walkthrough.ipynb` for the original eight-signal preparation, native named
readout, 2048 actual observations and classical postprocessing. Reuse the frozen
native qfunc and independent Circuit. The source seed-3 message and bases are
unchanged. Every observed row presents logical signal order without user-side
wire numbering or bitstring decoding.

Pass the actual preparation arrays to postprocessing. The source's final call
omitted them and regenerated unrelated random data; this native version corrects
that input mismatch explicitly. Retain selection of the most observed packet,
basis reconciliation, the strict 0.15 error threshold, matching-bit filtering and
adjacent-pair XOR. These last two steps are teaching illustrations, not
cryptographic error correction or privacy amplification. No secure-key claim is
made. Missing matching bases or threshold rejection stays explicit; no fake
bits are inserted. Resolve count ties by ascending numeric word.

This local workflow needs no credentials, remote execution or foreign source
loader. See [source mapping and acceptance](WORKFLOW.md).

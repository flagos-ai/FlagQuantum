# Quantum cosine and sine transforms

Start with the [complete native Notebook](overview/walkthrough.ipynb), which covers
all four transforms, source-seeded input generation and classical verification.
The [workflow guide](overview/README.md) explains phase and selector conventions.

The smaller frozen quantum-program examples remain available separately:

- [Combined type I](qct_qst_type1/README.md)
- [Combined type II](qct_qst_type2/README.md)
- [Cosine type II](qct_type2/README.md)
- [Sine type II](qst_type2/README.md)

The complete Notebook uses existing FlagQuantum qfunc preparation and transform
primitives. Independent Circuit checks reuse the shared Fourier-based reference;
no original source-language files or source importers are used by these examples.

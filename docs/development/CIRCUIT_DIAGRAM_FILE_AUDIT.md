# Logical circuit diagram source audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The frozen source and test blobs
are `a11a9eecc1bde23f43918ae177b728a01b667dc1` and
`c5aeb0ed27cc84ef72e64a99a6da399d3cf7807b`.

| Source responsibilities | Native owner |
| --- | --- |
| Wire labels, TeX escaping and gate widths | The logical exporter uses formal register names and the source text and width rules. Unsafe wire-token punctuation receives a safe wire identifier while the labels keep their formal spelling. |
| Wire allocation and soquet mapping | The exporter binds live register-score identities to declared wire names. Through ports retain their wire and released lanes permit suffix reuse. |
| Signature and component traversal | The register score and shared adjacency order supply input, output and per-call port metadata. One logical word remains one diagram wire regardless of its bit width. |
| Glyph and lifetime emission | The exporter uses the shared native symbol callback. It emits open and filled controls, XOR targets, text boxes, directional empty boxes, type labels and on/off transitions. Zero-port operations use a single hidden wire. |
| qpic source | The exporter returns the complete qpic text. Callers can save it as UTF-8. |
| TeX, PDF and PNG | An optional, bounded qpic process produces TikZ. The existing native LatexFigure owns document construction, rendering, temporary directories and explicit output paths. |

The original fixture compares exactly, including the one selection lane,
`BQUInt(2, 4)`, and the width-17 `R_L` box. The logical box represents the
reflection with global phase minus i. The compiler regression owns that
phase and the physical reflection decomposition. The exporter consumes
an explicit logical score and emits zero physical gates. The reflection
regression independently verifies gate order, complex amplitudes and
measurement-assisted work cleanup.

The full frozen source test passes. Real qpic 1.1.1 runs the original TeX
conversion and the native TikZ conversion. Both produce the same TikZ
body for the original case. Tectonic and Poppler render the native figure
to PDF and PNG. The source PDF and PNG conversion additionally requires
pdfLaTeX, which is absent from this test environment.

Source adapters accept foreign Bloq and Soquet objects and infer output
names from Python classes. The native export takes owned register metadata
and explicit output paths. Its tests also verify escaped adjoints, negative
controls, allocation, release, sequential name reuse and zero-port boxes.
